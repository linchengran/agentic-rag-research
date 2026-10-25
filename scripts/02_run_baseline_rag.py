"""运行基础 RAG 基线(单次检索,无 Agent)。

作为 Agentic RAG 的对照组,复用已有项目的单次检索流程。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_metrics import evaluate_answer  # noqa: E402
from src.evaluation.trace import save_traces  # noqa: E402
from src.llm.local_model import LocalModel  # noqa: E402
from src.llm.prompts import SINGLE_RAG_SYSTEM_PROMPT, SINGLE_RAG_USER_TEMPLATE  # noqa: E402
from src.retrievers.rerank_retriever import RerankRetriever  # noqa: E402
from src.tasks.multihop_qa import load_hotpotqa  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.env import setup_offline_mode  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行基础 RAG 基线")
    parser.add_argument("--data", required=True, help="HotpotQA JSON 路径")
    parser.add_argument("--index", default="data/processed/index/hotpot_faiss")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--out", default="experiments/results/baseline_rag.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logger = get_logger("baseline_rag", level="INFO")
    setup_offline_mode(offline=True)
    config = load_config(args.config)

    logger.info("加载问题...")
    questions = load_hotpotqa(args.data, limit=args.limit)
    logger.info("共 %d 个问题", len(questions))

    logger.info("加载索引 %s", args.index)
    retriever = RerankRetriever(
        embedding_model=config.get("embedding", {}).get("model_name", "BAAI/bge-small-zh-v1.5"),
        reranker_model=config.get("reranker", {}).get("model_name", "BAAI/bge-reranker-base"),
        final_k=config.get("retriever", {}).get("top_k", 5),
    )
    retriever.load(args.index)

    logger.info("加载模型 %s", config["llm"]["model_name"])
    llm = LocalModel(
        model_name=config["llm"]["model_name"],
        quantization=config["llm"].get("quantization"),
    )

    records = []
    for i, q in enumerate(questions):
        docs = retriever.retrieve(q.question)
        context = "\n\n".join(f"[Doc {j+1}] {d.content}" for j, d in enumerate(docs))
        user_prompt = SINGLE_RAG_USER_TEMPLATE.format(question=q.question, context=context)
        messages = [
            {"role": "system", "content": SINGLE_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        answer = llm.chat(messages)
        metrics = evaluate_answer(answer, q.answer)
        records.append({
            "id": q.id,
            "question": q.question,
            "gold_answer": q.answer,
            "pred_answer": answer,
            **metrics,
            "n_retrieved": len(docs),
        })
        if (i + 1) % 20 == 0:
            logger.info("处理 %d / %d", i + 1, len(questions))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    avg = {k: sum(r.get(k, 0.0) for r in records) / max(1, len(records)) for k in ("em", "answer_f1")}
    logger.info("完成。平均指标: %s", avg)


if __name__ == "__main__":
    main()