"""基于 FAISS 的稠密向量检索器。

参考 rag-research-project/src/vector_store.py 的接口风格,
默认使用 IndexFlatIP(精确内积)保证实验可复现。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("FAISS is required. Install faiss-cpu.") from exc

from sentence_transformers import SentenceTransformer

from .base import BaseRetriever, Document


class FAISSRetriever(BaseRetriever):
    """BGE Embedding + FAISS 检索器。"""

    name = "faiss"

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        normalize: bool = True,
        default_top_k: int = 5,
        device: str = "cpu",
    ) -> None:
        self.embedding_model_name = embedding_model
        self.normalize = normalize
        self.default_top_k = default_top_k
        self.device = device

        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []

    def _ensure_encoder(self) -> SentenceTransformer:
        if self.encoder is None:
            self.encoder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
        return self.encoder

    def index_documents(self, documents: List[Document]) -> None:
        """对文档列表建立 FAISS 索引。"""
        self.documents = list(documents)
        encoder = self._ensure_encoder()
        texts = [doc.content for doc in self.documents]
        embeddings = encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=self.normalize
        ).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """检索 top_k 个最相关的文档。"""
        if self.index is None or not self.documents:
            return []

        k = top_k or self.default_top_k
        encoder = self._ensure_encoder()
        q_vec = encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=self.normalize
        ).astype("float32")

        scores, idxs = self.index.search(q_vec, k)
        results: List[Document] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                Document(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    metadata=dict(doc.metadata),
                )
            )
        return results

    def save(self, path: str) -> None:
        """保存 FAISS 索引和文档列表到磁盘。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        np.save(path.with_suffix(".docs.npy"), np.array(self.documents, dtype=object), allow_pickle=True)

    def load(self, path: str) -> None:
        """从磁盘加载索引。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self.index = faiss.read_index(str(path))
        self.documents = list(np.load(path.with_suffix(".docs.npy"), allow_pickle=True))