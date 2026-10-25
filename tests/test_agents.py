"""Agent 单元测试。

使用 Mock LLM 和 Mock Retriever 测试 ReAct 的核心循环。
"""
from __future__ import annotations

from typing import Any, List

import pytest

from src.actions.finish import FinishAction
from src.actions.search import SearchAction
from src.agents.base import AgentOutput
from src.agents.react import ReActAgent
from src.evaluation.answer_metrics import exact_match, f1_score, normalize_answer
from src.retrievers.base import BaseRetriever, Document


class MockLLM:
    """模拟 LLM,返回预定的输出序列。"""

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
        self.docs = [Document(doc_id="d1", content="Paris is the capital of France.")]

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
        "finish": FinishAction(),
    }


def test_react_single_search_then_finish():
    llm = MockLLM([
        "Thought: I need to find the answer.\nAction: Search[capital of France]",
        "Thought: Got it.\nAction: Finish[Paris]",
    ])
    retriever = MockRetriever()
    actions = make_actions(retriever)

    agent = ReActAgent(llm=llm, actions=actions, max_steps=4)
    output = agent.run("What is the capital of France?")

    assert output.finished
    assert output.finish_reason == "finish_action"
    assert output.answer == "Paris"
    assert len(output.steps) == 2
    assert llm.call_count == 2


def test_react_max_steps_without_finish():
    llm = MockLLM([
        "Thought: searching\nAction: Search[query1]",
        "Thought: searching more\nAction: Search[query2]",
    ])
    retriever = MockRetriever()
    actions = make_actions(retriever)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=2)
    output = agent.run("?")

    assert not output.finished
    assert output.finish_reason == "max_steps"


def test_react_recovers_from_parse_error():
    llm = MockLLM([
        "Some random nonsense output",
        "Thought: let me search\nAction: Search[query1]",
        "Thought: ok now I know\nAction: Finish[42]",
    ])
    retriever = MockRetriever()
    actions = make_actions(retriever)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=4)
    output = agent.run("?")

    assert output.finished
    assert output.answer == "42"


def test_react_rejects_finish_without_search():
    """零检索 Finish 应被拒收,不允许模型跳过检索直接交卷。"""
    llm = MockLLM([
        "Thought: I already know this\nAction: Finish[Paris]",
        "Thought: fine, searching\nAction: Search[capital of France]",
        "Thought: got it\nAction: Finish[Paris]",
    ])
    retriever = MockRetriever()
    actions = make_actions(retriever)
    agent = ReActAgent(llm=llm, actions=actions, max_steps=4)
    output = agent.run("?")

    assert output.finished
    assert output.answer == "Paris"
    assert output.steps[0].error == "finish_without_search"


def test_answer_metrics():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("Paris", "London") == 0.0
    assert 0.0 < f1_score("Paris France", "France") < 1.0
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_relaxed_em():
    from src.evaluation.answer_metrics import relaxed_em
    # 严格命中仍然命中
    assert relaxed_em("Paris", "paris") == 1.0
    # 粒度近失:子集 containment
    assert relaxed_em("New York City", "Greenwich Village, New York City") == 1.0
    assert relaxed_em("Greenwich Village, New York City", "New York City") == 1.0
    # 范围近失:1969-1974 vs from 1969 until 1974(连字符拆开)
    assert relaxed_em("1969-1974", "from 1969 until 1974") == 1.0
    # 括号消歧后缀被吸收
    assert relaxed_em("Compliance (film)", "Compliance") == 1.0
    # 单词答案不做 containment:'no' 不因是 'north' 子串而误判
    assert relaxed_em("no", "north") == 0.0
    # 真错仍判错
    assert relaxed_em("London", "Paris") == 0.0
    # yes/no 反向错答不被吸收
    assert relaxed_em("yes", "no") == 0.0