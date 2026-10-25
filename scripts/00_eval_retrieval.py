"""检索层独立评估(不调 LLM)。

用法:
    python scripts/00_eval_retrieval.py --limit 500 --topk 5 10 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks.multihop_qa import load_hotpotqa  # noqa: E402
from src.retrievers.rerank_retriever import RerankRetriever  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.env import setup_offline_mode  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层独立评估")
    parser.add_argument(
        "--data",
        default=str(ROOT / "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--index",
        default=str(ROOT / "data/processed/index/hotpot_faiss"),
    )
    parser.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    logger = get_logger("eval_retrieval", level="INFO")
    setup_offline_mode(offline=True)
    config = load_config(args.config)

    logger.info("加载问题: %s (limit=%s)", args.data, args.limit)
    questions = load_hotpotqa(args.data, limit=args.limit)
    logger.info("加载了 %d 个问题", len(questions))

    retriever = RerankRetriever(
        embedding_model=config.get("embedding", {}).get(
            "model_name", "BAAI/bge-small-zh-v1.5"
        ),
        reranker_model=config.get("reranker", {}).get(
            "model_name", "BAAI/bge-reranker-base"
        ),
        final_k=max(args.topk),
        device="cuda",
    )
    retriever.load(args.index)
    logger.info("索引加载完成")

    golds = [q.gold_titles() for q in questions]
    hits = {k: 0 for k in args.topk}

    for i, (q, gold) in enumerate(zip(questions, golds)):
        if not gold:
            continue
        docs = retriever.retrieve(q.question, top_k=max(args.topk))
        retrieved_titles = [d.metadata.get("title", "") for d in docs]
        for k in args.topk:
            if gold & set(retrieved_titles[:k]):
                hits[k] += 1
        if (i + 1) % 50 == 0:
            logger.info("已跑 %d / %d 题", i + 1, len(questions))

    logger.info("=== 检索层结果 ===")
    n = len(questions)
    for k in args.topk:
        recall = hits[k] / n if n else 0.0
        logger.info("Recall@%d: %.4f (%d / %d)", k, recall, hits[k], n)
    print()
    print(f"样本数: {n}")
    for k in args.topk:
        print(f"  Recall@{k}: {hits[k]/n:.4f}" if n else f"  Recall@{k}: N/A")


if __name__ == "__main__":
    main()