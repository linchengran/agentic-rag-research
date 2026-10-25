"""Plan-Execute / Self-Ask / IRCoT 三个 Agent 范式的单元测试。

使用 Mock LLM 验证核心循环和终止逻辑。
"""
from __future__ import annotations

from typing import Any, List

from src.actions.finish import FinishAction
from src.actions.lookup import LookupAction
from src.actions.search import SearchAction
from src.agents.ircot import IRCoTAgent
from src.agents.plan_execute import PlanExecuteAgent
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


class MockRetriever(BaseRetriever):
    name = "mock"

    def __init__(self):
        self.docs = [
            Document(doc_id="d1", content="Paris is the capital of France."),
            Document(doc_id="d2", content="The Eiffel Tower is in Paris."),
        ]

    def index_documents(self, documents):
        pass

    def retrieve(self, query, top_k=None):
        return self.docs

    def save(self, path):
        pass

    def load(self, path):
        pass


def make_actions(retriever):
    return {
        "search": SearchAction(retriever),
        "lookup": LookupAction(),
        "finish": FinishAction(),
    }


# ============================================================
# Plan-and-Execute
# ============================================================

def test_plan_execute_generates_plan_then_executes():
    llm = MockLLM([
        # 阶段一:生成 plan
        "Plan:\n1. Search for capital of France.\n2. Finish with the answer.",
        # 阶段二第 1 步:执行检索
        "Thought: Now searching\nAction: Search[capital of France]",
        # 阶段二第 2 步:提交答案
        "Thought: I have enough info\nAction: Finish[Paris]",
    ])
    retriever = MockRetriever()
    agent = PlanExecuteAgent(llm=llm, actions=make_actions(retriever), max_steps=4)
    output = agent.run("What is the capital of France?")

    assert output.finished
    assert output.answer == "Paris"
    assert "plan" in output.metadata
    assert len(output.metadata["plan"]) == 2
    assert len(output.steps) == 2


def test_plan_execute_max_steps():
    llm = MockLLM([
        "Plan:\n1. Search a\n2. Search b\n3. Search c",
        "Thought: search\nAction: Search[a]",
        "Thought: search more\nAction: Search[b]",
        "still no valid action here",  # 收尾 turn 也解析失败
    ])
    retriever = MockRetriever()
    agent = PlanExecuteAgent(llm=llm, actions=make_actions(retriever), max_steps=2)
    output = agent.run("?")

    assert not output.finished
    assert output.finish_reason == "max_steps"


def test_plan_execute_wrap_up_finish():
    """plan 步耗尽仍没 Finish 时,收尾 turn 应能救回一个 Finish。"""
    llm = MockLLM([
        "Plan:\n1. Search a\n2. Search b",
        "Thought: search\nAction: Search[a]",
        "Thought: more\nAction: Search[b]",
        "Thought: enough evidence now\nAction: Finish[42]",
    ])
    retriever = MockRetriever()
    agent = PlanExecuteAgent(llm=llm, actions=make_actions(retriever), max_steps=2)
    output = agent.run("?")

    assert output.finished
    assert output.finish_reason == "finish_action"
    assert output.answer == "42"


def test_plan_execute_handles_plan_failure():
    """LLM 没生成 plan 时,应当平稳降级到 no_plan 终止。"""
    llm = MockLLM([
        "I don't know how to answer.",  # 没有 Plan 编号
    ])
    retriever = MockRetriever()
    agent = PlanExecuteAgent(llm=llm, actions=make_actions(retriever), max_steps=4)
    output = agent.run("?")

    # plan 解析失败时,finish_reason 应该是 plan_error
    assert output.finish_reason in ("plan_error", "max_steps")


# ============================================================
# Self-Ask
# ============================================================

def test_self_ask_intermediate_question_then_search():
    llm = MockLLM([
        # 第 1 步:问中间问题
        "Intermediate Question: What is the capital of France?\nIntermediate Answer:",
        # 第 2 步:直接给出最终答案
        "Final Answer: Paris",
    ])
    retriever = MockRetriever()
    agent = SelfAskAgent(llm=llm, actions=make_actions(retriever), max_steps=4)
    output = agent.run("What is the capital of France?")

    assert output.finished
    assert output.answer == "Paris"
    assert output.finish_reason == "final_answer"
    # 第一步的 action_name 应该是 search(因为中间问题被路由到 Search)
    assert output.steps[0].action_name == "search"


def test_self_ask_max_steps_with_fallback():
    llm = MockLLM([
        "Intermediate Question: What is the capital?",
        "Intermediate Question: What is in Paris?",
    ])
    retriever = MockRetriever()
    agent = SelfAskAgent(llm=llm, actions=make_actions(retriever), max_steps=2)
    output = agent.run("?")

    assert not output.finished
    assert output.finish_reason == "max_steps"


# ============================================================
# IRCoT
# ============================================================

def test_ircot_thought_then_final():
    """IRCoT 第 1 步给出 Thought,经检索后再决策。"""
    llm = MockLLM([
        # 第 1 次 chat: 给 Thought 1
        "Thought 1: I need to find the capital of France.",
        # 第 2 次 chat (after Observation): 给 Final Answer
        "Final Answer: Paris",
    ])
    retriever = MockRetriever()
    agent = IRCoTAgent(llm=llm, actions=make_actions(retriever), max_steps=4)
    output = agent.run("What is the capital of France?")

    # 应当 finish
    assert output.finished
    assert output.answer == "Paris"


def test_ircot_max_steps():
    """持续 Thought 但从未给 Final Answer,达到 max_steps。"""
    llm = MockLLM([
        "Thought 1: Looking up...",
        "Thought 1: Still looking up...",
        "Thought 1: Searching more...",
        "Thought 1: Almost there...",
    ])
    retriever = MockRetriever()
    agent = IRCoTAgent(llm=llm, actions=make_actions(retriever), max_steps=2)
    output = agent.run("?")

    assert not output.finished
    assert output.finish_reason == "max_steps"


# ============================================================
# 基类 parse_action 测试
# ============================================================

def test_base_agent_parse_action_valid():
    from src.agents.base import BaseAgent

    name, arg = BaseAgent.parse_action("Action: Search[hello world]")
    assert name == "search"
    assert arg == "hello world"


def test_base_agent_parse_action_finish():
    from src.agents.base import BaseAgent

    name, arg = BaseAgent.parse_action("Action: Finish[Paris]")
    assert name == "finish"
    assert arg == "Paris"


def test_base_agent_parse_action_error():
    from src.agents.base import BaseAgent

    name, arg = BaseAgent.parse_action("nonsense output without action")
    assert name == "parse_error"
    assert "nonsense" in arg