"""Search 动作:基于 query 在检索器中查找相关文档。

这是 Agentic RAG 的核心动作,所有检索决策最终都体现为 Search 调用。
"""
from __future__ import annotations

from typing import Any

from .base import ActionResult, BaseAction, register_action
from ..retrievers.base import BaseRetriever


class SearchAction(BaseAction):
    """Search 动作:返回检索到的 Top-K 文档内容。"""

    name = "search"

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def __call__(self, query: str, **kwargs: Any) -> ActionResult:
        """根据 query 检索文档。

        Args:
            query: 检索关键词。
            kwargs: 其他参数(top_k 等),传递给检索器。

        Returns:
            包含检索结果的 ActionResult。
        """
        if not query or not query.strip():
            return ActionResult(
                name=self.name,
                success=False,
                observation="[Search] Empty query.",
                error="empty_query",
            )
        try:
            docs = self.retriever.retrieve(query, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(
                name=self.name,
                success=False,
                observation=f"[Search] Retrieval error: {exc}",
                raw_input={"query": query},
                error=str(exc),
            )

        if not docs:
            observation = f"[Search] No documents found for query: {query!r}"
        else:
            joined = "\n\n".join(
                f"[Doc {i+1}] {doc.content}" for i, doc in enumerate(docs)
            )
            observation = f"[Search] Results for {query!r}:\n{joined}"

        return ActionResult(
            name=self.name,
            success=True,
            observation=observation,
            raw_input={"query": query, "top_k": kwargs.get("top_k")},
        )

    @property
    def description(self) -> str:
        return "Search[query]: 根据 query 检索相关文档。"


# 模块导入时默认注册一个占位实例。
# 实际使用时通过 SearchAction(retriever) 创建并 register_action。
__all__ = ["SearchAction"]