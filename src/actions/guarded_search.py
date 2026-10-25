"""带守卫的 Search 动作:intervention 实验的核心组件。

来自 REPORT §3.8 的两个观测:
1. react 30% 检索是重复 query(step5+ 100% 复读)
2. 第 3 步起边际金标恢复率全面归零——该停不停才是浪费

守卫两级(浪费 = 重复 query,或检索结果全部是已见文档):
- 第 1 次浪费:拦截(重复 query 不检索)或在 observation 追加纠错警告,
  给模型一次自救机会(换 query 或直接交卷)
- 连续第 waste_limit 次浪费:触发早停 STOP 指令;hard 模式下模型再检索
  则 agent 循环直接强制终止(finish_reason="guard_early_stop"),
  nudge 模式只注入指令不强制,测"3B 是否服从纠错反馈"

设计约束:守卫只改写 Search 的 observation,agent 循环主体不感知
(控制组与实验组的唯一差异 = search 是否带守卫),保证受控比较。
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from .base import ActionResult, BaseAction
from ..retrievers.base import BaseRetriever

# 与 scripts/10_stepwise_coverage.py 的重复判定(strip 精确匹配)同口径,
# 再加大小写/标点归一(严格超集:分析口径认定的重复守卫全拦,额外多拦的
# 只是大小写/标点变体,不改变分析可比性)
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_query(query: str) -> str:
    """归一化 query:小写、去标点、压空白。"""
    q = _PUNCT_RE.sub(" ", (query or "").lower())
    return re.sub(r"\s+", " ", q).strip()


class GuardedSearchAction(BaseAction):
    """Search 动作 + 重复 query 检测器 + 零收益早停守卫。

    状态按"单题"生命周期管理:runner 在每题开始前调用 reset()。
    events 记录守卫干预事件,由 runner 附进结果记录供离线分析。
    """

    name = "search"

    def __init__(
        self,
        retriever: BaseRetriever,
        waste_limit: int = 2,
        hard_stop: bool = True,
        finish_hint: str = "Action: Finish[shortest answer]",
    ) -> None:
        self.retriever = retriever
        self.waste_limit = max(1, waste_limit)
        self.hard_stop = hard_stop
        self.finish_hint = finish_hint
        self.reset()

    # ---------------- 单题状态 ----------------

    def reset(self) -> None:
        """每题开始前清空守卫状态。"""
        self.queries: set[str] = set()          # 归一化后的已执行 query
        self.doc_ids: set[str] = set()          # 已见过的文档 doc_id
        self.waste_streak: int = 0              # 连续浪费次数(重复/零新增)
        self.stop_requested: bool = False       # 已触发早停(STOP 指令生效)
        self.stop_signaled: bool = False        # STOP 指令后模型仍在检索
        self.events: List[dict] = []

    @property
    def should_stop(self) -> bool:
        """agent 循环读这个属性决定是否强制终止(hard 模式专属)。"""
        return self.hard_stop and self.stop_signaled

    # ---------------- 守卫消息 ----------------

    def _duplicate_msg(self) -> str:
        return (
            "[Guard] Duplicate query: you have already searched for this and the "
            "results are in the history above. Repeating a search wastes a step. "
            f"Either Search with a clearly different query, or give your final "
            f"answer now ({self.finish_hint})."
        )

    def _zero_new_msg(self) -> str:
        return (
            "\n[Guard] Warning: this search returned no new documents - every "
            "result was already retrieved in an earlier step. Searching similar "
            f"queries again will not help. If you have enough information, give "
            f"your final answer now ({self.finish_hint}); otherwise Search with "
            "a clearly different query."
        )

    def _stop_msg(self) -> str:
        return (
            "[Guard] STOP. Multiple consecutive searches returned no new "
            "information, and further retrieval will not change the evidence "
            f"you already have. Do NOT search again. You MUST now output your "
            f"final answer in the format: {self.finish_hint}"
        )

    # ---------------- 主逻辑 ----------------

    def _log(self, event: str, query: str) -> None:
        self.events.append({"event": event, "query": query})

    def _register_waste(self, event: str, query: str) -> None:
        """登记一次浪费;连续达到 waste_limit 则触发早停。"""
        self.waste_streak += 1
        self._log(event, query)
        if self.waste_streak >= self.waste_limit:
            self.stop_requested = True
            self._log("stop_triggered", query)

    def __call__(self, query: str, **kwargs: Any) -> ActionResult:
        q = (query or "").strip()

        # 已触发早停后仍检索:hard 模式交由 agent 强制终止,nudge 模式持续注入 STOP
        if self.stop_requested:
            self.stop_signaled = True
            self._log("stop_ignored", q)
            return ActionResult(
                name=self.name,
                success=False,
                observation=self._stop_msg(),
                error="early_stop_requested",
            )

        # 空 query:与 SearchAction 同语义
        if not q:
            return ActionResult(
                name=self.name,
                success=False,
                observation="[Search] Empty query.",
                error="empty_query",
            )

        # 重复 query 检测器:拦截,不检索
        norm = normalize_query(q)
        if norm in self.queries:
            self._register_waste("duplicate_query", q)
            return ActionResult(
                name=self.name,
                success=False,
                observation=self._duplicate_msg(),
                error="duplicate_query",
            )

        # 执行检索
        try:
            docs = self.retriever.retrieve(q, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(
                name=self.name,
                success=False,
                observation=f"[Search] Retrieval error: {exc}",
                error=str(exc),
            )

        self.queries.add(norm)
        all_ids = {d.doc_id for d in docs}
        new_ids = all_ids - self.doc_ids
        self.doc_ids |= all_ids

        if not docs:
            observation = f"[Search] No documents found for query: {q!r}"
        else:
            joined = "\n\n".join(
                f"[Doc {i+1}] {doc.content}" for i, doc in enumerate(docs)
            )
            observation = f"[Search] Results for {q!r}:\n{joined}"
            # 零收益检测:top-k 全部是已见文档 → 浪费
            if not new_ids:
                self._register_waste("zero_new_docs", q)
                if self.stop_requested:
                    observation = observation + "\n" + self._stop_msg()
                else:
                    observation = observation + self._zero_new_msg()
            else:
                self.waste_streak = 0  # 有效检索打断浪费连击

        return ActionResult(
            name=self.name,
            success=True,
            observation=observation,
            raw_input={"query": q, "top_k": kwargs.get("top_k")},
        )

    @property
    def description(self) -> str:
        return "Search[query]: 检索相关文档(带重复 query 检测器与零收益早停守卫)。"


__all__ = ["GuardedSearchAction", "normalize_query"]
