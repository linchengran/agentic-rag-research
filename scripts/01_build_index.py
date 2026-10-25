"""构建多跳问答语料的向量索引。

用法:
    python scripts/01_build_index.py --data data/raw/hotpotqa/hotpot_dev_distractor_v1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把项目根加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrievers.base import Document  # noqa: E402
from src.retrievers.faiss_retriever import FAISSRetriever  # noqa: E402
from src.retrievers.rerank_retriever import RerankRetriever  # noqa: E402
from src.tasks.multihop_qa import CorpusDocument, extract_corpus, load_hotpotqa  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.env import setup_offline_mode  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def build_documents(corpus: list[CorpusDocument]) -> list[Document]:
    return [
        Document(
            doc_id=d.doc_id,
            content=f"{d.title}: {d.content}",
            metadata=d.metadata,
        )
        for d in corpus
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建多跳问答语料索引")
    parser.add_argument("--data", required=True, help="HotpotQA JSON 路径")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--out", default="data/processed/index/hotpot_faiss")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    logger = get_logger("build_index", level="INFO")
    setup_offline_mode(offline=False)  # 首次构建索引需要联网下载模型
    config = load_config(args.config)

    logger.info("加载数据: %s", args.data)
    questions = load_hotpotqa(args.data, limit=config.get("task", {}).get("eval_limit"))
    logger.info("加载了 %d 个问题", len(questions))

    corpus = extract_corpus(questions)
    logger.info("提取语料段落: %d 个", len(corpus))

    documents = build_documents(corpus)
    embedding_cfg = config.get("embedding", {})
    if args.no_reranker:
        retriever = FAISSRetriever(
            embedding_model=embedding_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
            normalize=embedding_cfg.get("normalize", True),
            default_top_k=config.get("retriever", {}).get("top_k", 5),
        )
    else:
        retriever = RerankRetriever(
            embedding_model=embedding_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
            reranker_model=config.get("reranker", {}).get("model_name", "BAAI/bge-reranker-base"),
            final_k=config.get("retriever", {}).get("top_k", 5),
        )

    logger.info("开始编码 + 建立索引...")
    retriever.index_documents(documents)
    logger.info("索引建立完成,持久化到 %s", args.out)
    retriever.save(args.out)


if __name__ == "__main__":
    main()