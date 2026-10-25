"""GuardedSearchAction(重复 query 检测器 + 零收益早停守卫)单元测试。

覆盖:
- 守卫本体:重复拦截(不检索)、归一化口径、零新增文档警告 → 早停触发 →
  STOP-only、有效检索重置浪费连击、nudge 模式不强制、reset 隔离
- 4 范式集成:hard 模式 STOP 后再检索 → guard_early_stop 强制终止;
  服从 STOP 指令正常交卷不受影响
"""
from __future__ import annotations

from typing import List

from src.actions.finish import FinishAction
from src.actions.guarded_search import GuardedSearchAction, normalize_query
from src.actions.lookup import LookupAction
from src.agents.ircot import IRCoTAgent
from src.agents.plan_execute import PlanExecuteAgent
from src.agents.react import ReActAgent
from src.agents.self_ask import SelfAskAgent
from src.retrievers.base import BaseRetriever, Document


class MockLLM:
    """按调用顺序返回预定输出的 Mock LLM。"""

    def __init__(self, outputs: List[str]):
        self.outputs = list(outputs)
        self.call_count = 0

    def chat(self, messages, config=None) -> str:
        if self.call_count >= len(self.outputs):
            return "Action: Finish[unknown]"
        out = self.outputs[self.call_count]
        self.call_count += 1
        return out


class CountingRetriever(BaseRetriever):
    """可切换返回文档、统计调用次数的 Mock 检索器。"""

    name = "counting"

    def __init__(self, docs=None):
        self.docs = docs or [
            Document(doc_id="d1", content="Paris is the capital of France."),
            Document(doc_id="d2", content="The Eiffel Tower is in Paris."),
        ]
        self.calls = 0

    def set_docs(self, docs):
        self.docs = docs

    def index_documents(self, documents):
        pass

    def retrieve(self, query, top_k=None):
        self.calls += 1
        return self.docs

    def save(self, path):
        pass

    def load(self, path):
        pass


def make_guard_actions(retriever, hard_stop=True, waste_limit=2, finish_hint="Action: Finish[shortest answer]"):
    return {
        "search": GuardedSearchAction(
            retriever, waste_limit=waste_limit, hard_stop=hard_stop, finish_hint=finish_hint
        ),
        "lookup": LookupAction(),
        "finish": FinishAction(),
    }


# ============================================================
# 守卫本体
# ============================================================

def test_normalize_query():
    assert normalize_query("Ed Wood nationality") == normalize_query("ed wood  NATIONALITY!")
    assert normalize_query("  q1  ") == "q1"
    assert normalize_query("") == ""


def test_duplicate_blocked_without_retrieval():
    r = CountingRetriever()
    g = GuardedSearchAction(r)
    r1 = g("Ed Wood nationality")
    assert r1.success and r.calls == 1

    r2 = g("Ed Wood nationality")
    assert not r2.success
    assert r2.error == "duplicate_query"
    assert r.calls == 1  # 重复被拦截,不进检索器
    assert g.waste_streak == 1
    assert not g.stop_requested
    assert g.events[0]["event"] == "duplicate_query"


def test_duplicate_normalized_variant():
    """大小写/标点变体也算重复(分析口径 strip 精确匹配的严格超集)。"""
    r = CountingRetriever()
    g = GuardedSearchAction(r)
    g("Ed Wood nationality")
    r2 = g("ed wood nationality!")
    assert r2.error == "duplicate_query"


def test_zero_new_docs_warn_then_stop():
    """q1 有效 → q2 全是已见文档(警告)→ q3 触发早停 → q4 STOP-only。"""
    r = CountingRetriever()
    g = GuardedSearchAction(r)

    r1 = g("q1")
    assert r1.success and g.waste_streak == 0

    r2 = g("q2")  # 不同 query,但 top-k 全部是已见文档
    assert r2.success
    assert "[Guard] Warning" in r2.observation
    assert g.waste_streak == 1 and not g.stop_requested

    r3 = g("q3")
    assert r3.success
    assert "[Guard] STOP" in r3.observation
    assert g.stop_requested and not g.stop_signaled
    assert not g.should_stop  # 触发步不立即硬停,给模型一次服从机会

    r4 = g("q4")
    assert not r4.success
    assert r4.error == "early_stop_requested"
    assert "[Guard] STOP" in r4.observation
    assert g.stop_signaled and g.should_stop
    assert r.calls == 3  # q2/q3 真实检索,q4 被拦


def test_effective_search_resets_waste_streak():
    """有效检索(带来新文档)应打断浪费连击,不累计误伤。"""
    r = CountingRetriever()
    g = GuardedSearchAction(r)
    g("q1")
    g("q1")            # waste 1
    r.set_docs([Document(doc_id="d3", content="Brand new doc.")])
    g("q2")            # 有效 → streak 归零
    assert g.waste_streak == 0
    g("q1")            # 重新从 waste 1 开始
    assert g.waste_streak == 1
    assert not g.stop_requested


def test_nudge_mode_never_hard_stops():
    """nudge 模式只注入指令,should_stop 恒为 False。"""
    r = CountingRetriever()
    g = GuardedSearchAction(r, hard_stop=False)
    g("q1")
    g("q1")   # waste 1
    g("q1")   # waste 2 → stop_requested
    g("q1")   # STOP-only
    assert g.stop_requested and g.stop_signaled
    assert not g.should_stop


def test_reset_clears_state():
    r = CountingRetriever()
    g = GuardedSearchAction(r)
    g("q1")
    g("q1")
    g.reset()
    assert not g.queries and not g.doc_ids
    assert g.waste_streak == 0 and not g.stop_requested
    assert g.events == []
    r2 = g("q1")  # 新题里同一 query 不再算重复
    assert r2.success


# ============================================================
# ReAct 集成
# ============================================================

REACT_SEARCH = "Thought: need info\nAction: Search[q1]"


def test_react_hard_guard_early_stop():
    """hard 模式:STOP 指令后模型仍复读检索 → 强制终止。"""
    llm = MockLLM([REACT_SEARCH] * 4)
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=8)
    output = agent.run("?")

    assert not output.finished
    assert output.finish_reason == "guard_early_stop"
    assert output.answer == ""  # 从未 Finish 过,兜底为空
    assert len(output.steps) == 4
    assert retriever.calls == 1  # 只有第一次是真检索
    assert [s.error for s in output.steps] == [
        "", "duplicate_query", "duplicate_query", "early_stop_requested",
    ]


def test_react_nudge_mode_runs_to_max_steps():
    """nudge 模式:不强制终止,模型无视 STOP 则走到 max_steps。"""
    llm = MockLLM([REACT_SEARCH] * 8)
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever, hard_stop=False)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=8)
    output = agent.run("?")

    assert output.finish_reason == "max_steps"
    assert retriever.calls == 1
    # STOP 之后的步全部被拦截
    assert sum(1 for s in output.steps if s.error == "early_stop_requested") >= 4


def test_react_complies_with_stop_finishes_normally():
    """hard 模式:模型服从 STOP 指令直接 Finish → 正常终止,守卫不劫持。"""
    llm = MockLLM([
        REACT_SEARCH,
        REACT_SEARCH,  # waste 1
        REACT_SEARCH,  # waste 2 → STOP 指令
        "Thought: I have enough\nAction: Finish[Paris]",  # 服从
    ])
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=8)
    output = agent.run("?")

    assert output.finished
    assert output.finish_reason == "finish_action"
    assert output.answer == "Paris"


def test_react_zero_new_docs_hard_stop():
    """零新增文档路径:不同 query 但检索结果全重复,同样触发早停。"""
    llm = MockLLM([
        "Thought: a\nAction: Search[q1]",
        "Thought: b\nAction: Search[q2]",  # 零新增,警告
        "Thought: c\nAction: Search[q3]",  # 零新增,触发 STOP
        "Thought: d\nAction: Search[q4]",  # 无视 → 强制终止
    ])
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=8)
    output = agent.run("?")

    assert output.finish_reason == "guard_early_stop"
    assert retriever.calls == 3


# ============================================================
# Self-Ask / Plan-Execute / IRCoT 集成(各验一条 hard 早停路径)
# ============================================================

def test_self_ask_hard_guard_early_stop():
    llm = MockLLM(["Intermediate Question: q1"] * 4)
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever, finish_hint="Final Answer: <shortest answer>")
    agent = SelfAskAgent(llm=llm, actions=actions, max_steps=8)
    output = agent.run("?")

    assert output.finish_reason == "guard_early_stop"
    assert retriever.calls == 1
    assert len(output.steps) == 4


def test_plan_execute_hard_guard_early_stop():
    llm = MockLLM([
        "Plan:\n1. Search q1\n2. Search q1\n3. Search q1\n4. Search q1",
        "Thought: a\nAction: Search[q1]",
        "Thought: b\nAction: Search[q1]",  # waste 1
        "Thought: c\nAction: Search[q1]",  # waste 2 → STOP
        "Thought: d\nAction: Search[q1]",  # 无视 → 终止
    ])
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever)
    agent = PlanExecuteAgent(llm=llm, actions=actions, max_steps=4)
    output = agent.run("?")

    assert output.finish_reason == "guard_early_stop"
    assert retriever.calls == 1
    assert len(output.steps) == 4


def test_ircot_hard_guard_early_stop():
    llm = MockLLM([
        "Thought 1: q1",             # iter1 检索
        "evidence so far is unclear",  # follow-up 无 Final Answer
        "Thought 2: q1",              # 重复 → waste 1
        "still unclear",
        "Thought 3: q1",              # 重复 → waste 2,STOP
        "still unclear",
        "Thought 4: q1",              # 无视 STOP → 强制终止
        "still unclear",
    ])
    retriever = CountingRetriever()
    actions = make_guard_actions(retriever, finish_hint="Final Answer: <shortest answer>")
    agent = IRCoTAgent(llm=llm, actions=actions, max_steps=8)
    output = agent.run("?")

    assert output.finish_reason == "guard_early_stop"
    assert retriever.calls == 1
    assert len(output.steps) == 4
