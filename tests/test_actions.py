"""Action 单元测试。

测试 Search / Lookup / Finish 三个动作在 Mock Retriever 下的行为。
"""
from __future__ import annotations

from collections import deque

import pytest

from src.actions.finish import FinishAction
from src.actions.lookup import LookupAction
from src.actions.search import SearchAction
from src.retrievers.base import BaseRetriever, Document


class MockRetriever(BaseRetriever):
    """测试用 Retriever,返回固定文档。"""

    name = "mock"

    def __init__(self, docs=None, empty=False):
        self.docs = docs if docs is not None else [
            Document(doc_id="d1", content="The Eiffel Tower is in Paris."),
            Document(doc_id="d2", content="Paris is the capital of France."),
        ]
        self.empty = empty
        self.indexed = False
        self.last_query = None

    def index_documents(self, documents):
        self.indexed = True

    def retrieve(self, query, top_k=None):
        self.last_query = query
        return [] if self.empty else self.docs[: (top_k or len(self.docs))]

    def save(self, path):
        pass

    def load(self, path):
        pass


def test_search_action_returns_docs():
    retriever = MockRetriever()
    action = SearchAction(retriever)
    result = action("Paris")
    assert result.success
    assert "Eiffel Tower" in result.observation
    assert retriever.last_query == "Paris"


def test_search_action_handles_empty_query():
    retriever = MockRetriever()
    action = SearchAction(retriever)
    result = action("   ")
    assert not result.success
    assert "Empty query" in result.observation


def test_search_action_handles_no_results():
    retriever = MockRetriever(empty=True)
    action = SearchAction(retriever)
    result = action("nothing")
    assert result.success
    assert "No documents" in result.observation


def test_lookup_action_finds_term():
    obs_deque = deque(["The Eiffel Tower is in Paris."])
    action = LookupAction(last_observations=obs_deque)
    result = action("Paris")
    assert result.success
    assert "Paris" in result.observation


def test_lookup_action_handles_no_context():
    action = LookupAction()
    result = action("Paris")
    assert not result.success
    assert "No previous" in result.observation


def test_finish_action_returns_answer():
    action = FinishAction()
    result = action("Paris")
    assert result.success
    assert action.last_answer == "Paris"
    assert "Paris" in result.observation