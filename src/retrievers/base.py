"""Retriever 基类与文档结构。

Retriever 是 Agent 可以调用的"检索工具",对外只暴露 retrieve 接口。
具体实现可以是 FAISS、BGE + FAISS、BM25、混合检索等。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """检索命中的单个文档。"""

    doc_id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def truncated(self, max_chars: int) -> "Document":
        """返回内容被截断到 max_chars 的副本(便于嵌入 LLM 上下文)。"""
        if len(self.content) <= max_chars:
            return self
        return Document(
            doc_id=self.doc_id,
            content=self.content[:max_chars] + "...",
            score=self.score,
            metadata={**self.metadata, "truncated": True},
        )


@dataclass
class RetrievedDocs:
    """检索结果包装,便于日志和测试。"""

    query: str
    documents: List[Document]

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __getitem__(self, idx: int) -> Document:
        return self.documents[idx]


class BaseRetriever(ABC):
    """检索器抽象基类。"""

    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """根据 query 检索 top_k 个文档。

        Args:
            query: 检索关键词。
            top_k: 返回的文档数,为 None 时使用构造时的默认值。

        Returns:
            按相关性降序排列的文档列表。
        """

    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """建立索引,通常在数据准备阶段调用一次。"""

    @abstractmethod
    def save(self, path: str) -> None:
        """持久化索引到磁盘。"""

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载索引。"""