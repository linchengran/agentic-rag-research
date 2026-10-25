"""Lookup 动作:在上一步检索结果中查找具体 term。

在多跳问答中,Lookup 用于"在已知相关文档中精确定位关键句子"。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional

from .base import ActionResult, BaseAction, register_action


class LookupAction(BaseAction):
    """Lookup 动作:在最近检索结果中查找 term。"""

    name = "lookup"

    def __init__(self, last_observations: Optional[Deque[str]] = None) -> None:
        # 与 Agent 共享一个 deque,记录最近检索结果
        self.last_observations = last_observations or deque(maxlen=3)

    def __call__(self, term: str, **kwargs: Any) -> ActionResult:
        """在最近检索结果中查找包含 term 的句子。

        Args:
            term: 要查找的关键词或短语。
        """
        if not term or not term.strip():
            return ActionResult(
                name=self.name,
                success=False,
                observation="[Lookup] Empty term.",
                error="empty_term",
            )
        if not self.last_observations:
            return ActionResult(
                name=self.name,
                success=False,
                observation="[Lookup] No previous retrieval context.",
                error="no_context",
            )

        target = term.strip().lower()
        matched_sentences: list[str] = []
        for obs in self.last_observations:
            for sentence in obs.split(". "):
                if target in sentence.lower():
                    matched_sentences.append(sentence.strip())

        if not matched_sentences:
            observation = f"[Lookup] No matching sentence for term: {term!r}"
        else:
            observation = "[Lookup] Found:\n" + "\n".join(
                f"- {s}" for s in matched_sentences[:5]
            )

        return ActionResult(
            name=self.name,
            success=True,
            observation=observation,
            raw_input={"term": term, "matched": len(matched_sentences)},
        )

    @property
    def description(self) -> str:
        return "Lookup[term]: 在最近检索结果中查找 term 出现的句子。"


__all__ = ["LookupAction"]