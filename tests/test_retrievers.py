"""Retriever 单元测试。

注:完整测试需要真实模型,这里用 Mock 测试 Retriever 抽象基类实现。
"""
from __future__ import annotations

import pytest

from src.retrievers.base import BaseRetriever, Document


class FakeRetriever(BaseRetriever):
    name = "fake"

    def __init__(self):
        self.indexed = False
        self.saved = False
        self.loaded = False

    def index_documents(self, documents):
        self.indexed = True

    def retrieve(self, query, top_k=None):
        return [Document(doc_id="d1", content=query, score=1.0)]

    def save(self, path):
        self.saved = True

    def load(self, path):
        self.loaded = True


def test_base_retriever_interface():
    r = FakeRetriever()
    docs = [Document(doc_id="d1", content="hello world")]
    r.index_documents(docs)
    assert r.indexed

    retrieved = r.retrieve("hello")
    assert len(retrieved) == 1
    assert retrieved[0].content == "hello"

    r.save("/tmp/index")
    assert r.saved

    r.load("/tmp/index")
    assert r.loaded


def test_document_truncation():
    d = Document(doc_id="d1", content="a" * 100)
    truncated = d.truncated(20)
    assert len(truncated.content) == 23  # 20 + "..."
    assert truncated.metadata["truncated"] is True

    # 不需要截断时返回原对象
    same = d.truncated(200)
    assert same is d