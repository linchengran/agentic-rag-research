"""FAISS + BGE Reranker 的检索器。

在 FAISS 检索基础上,对 Top-K 候选用 Cross-Encoder 重排序,
参考 rag-research-project/src/reranker.py 的思路。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import BaseRetriever, Document
from .faiss_retriever import FAISSRetriever


class RerankRetriever(BaseRetriever):
    """FAISS 粗排 + BGE Reranker 精排。"""

    name = "faiss_rerank"

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        first_stage_k: int = 20,
        final_k: int = 5,
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        self.first_stage_k = first_stage_k
        self.final_k = final_k

        self.faiss_retriever = FAISSRetriever(
            embedding_model=embedding_model,
            normalize=normalize,
            default_top_k=first_stage_k,
            device=device,
        )
        self.reranker_model_name = reranker_model
        self.reranker = None  # 延迟加载
        self.device = device

    def _ensure_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        return self.reranker

    def index_documents(self, documents: List[Document]) -> None:
        self.faiss_retriever.index_documents(documents)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """先 FAISS 召回 first_stage_k,再用 reranker 重排取 final_k。"""
        k = top_k or self.final_k
        candidates = self.faiss_retriever.retrieve(query, top_k=self.first_stage_k)
        if not candidates:
            return []

        reranker = self._ensure_reranker()
        pairs = [(query, doc.content) for doc in candidates]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[:k]

        results: List[Document] = []
        for score, doc in ranked:
            results.append(
                Document(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    metadata={**doc.metadata, "reranked": True},
                )
            )
        return results

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_retriever.save(path)

    def load(self, path: str) -> None:
        self.faiss_retriever.load(path)