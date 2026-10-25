"""离线重放 Agent 检索 query,计算真实的金标文档命中率。

背景:轨迹保存时 observation 被截断到 300 字符(agents/base.py:98),
每步只能看到前 ~2 篇文档,直接解析 trace 会低估命中率。
本脚本不跑 LLM,只把 trace 里每个检索步骤的 action_input 喂给同一个
检索器(确定性),拿完整 top-5 标题集重算。

额外对照:同一批问题用"原问题单次检索"(即 baseline 的检索方式)重放,
得到同题集的 apples-to-apples baseline 命中率。

用法:
    python scripts/08_replay_gold_hit.py \
        --results experiments/results/matrix/self_ask_n30_v3.json \
        --trace-dir experiments/traces/self_ask_v3 \
        --out experiments/results/matrix/gold_hit_replay_self_ask.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrievers.rerank_retriever import RerankRetriever  # noqa: E402
from src.tasks.multihop_qa import load_hotpotqa  # noqa: E402
from src.utils.env import setup_offline_mode  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

# 这些 action 会触发索引检索(见各 agent 源码)
SEARCH_ACTIONS = {"search", "ircot_search"}


def collect_queries(trace: dict) -> list[str]:
    """按序提取一条轨迹中所有实际触发检索的 query(去重)。"""
    seen: set[str] = set()
    queries: list[str] = []
    for step in trace.get("steps", []):
        if step.get("action_name") not in SEARCH_ACTIONS:
            continue
        q = (step.get("action_input") or "").strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="重放检索 query 计算金标命中率")
    parser.add_argument("--results", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--data", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--index", default="data/processed/index/hotpot_faiss")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logger = get_logger("replay_gold_hit", level="INFO")
    setup_offline_mode(offline=True)

    import yaml

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    retriever = RerankRetriever(
        embedding_model=config.get("embedding", {}).get("model_name", "BAAI/bge-small-zh-v1.5"),
        reranker_model=config.get("reranker", {}).get("model_name", "BAAI/bge-reranker-base"),
        final_k=config.get("retriever", {}).get("top_k", 5),
    )
    retriever.load(args.index)
    logger.info("索引加载完成:%s", args.index)

    with open(args.results, encoding="utf-8") as f:
        records = json.load(f)

    # question -> trace
    q2trace: dict[str, dict] = {}
    for tp in sorted(Path(args.trace_dir).glob("trace_*.json")):
        with open(tp, encoding="utf-8") as f:
            t = json.load(f)
        q2trace.setdefault(t.get("question", ""), t)

    # id -> 金标支持标题(data_loader 已封装)
    qmap: dict[str, object] = {}
    for q in load_hotpotqa(args.data, limit=None):
        qmap[q.id] = q

    def titles_of(query: str) -> set[str]:
        return {d.metadata.get("title", "") for d in retriever.retrieve(query)}

    def coverage(retrieved: set[str], gold: set[str]) -> float:
        return len(gold & retrieved) / len(gold) if gold else 0.0

    per_q = []
    n_full = n_partial = 0
    cov_sum = cov_first_sum = base_full = base_partial = 0.0
    base_cov_sum = 0.0
    for rec in records:
        gold = set(qmap[rec["id"]].gold_titles()) if rec["id"] in qmap else set()
        if not gold:
            continue
        queries = collect_queries(q2trace.get(rec["question"], {}))
        agent_titles: set[str] = set()
        first_titles: set[str] = set()
        for i, q in enumerate(queries):
            titles = titles_of(q)
            agent_titles |= titles
            if i == 0:
                first_titles = titles
        # baseline:原问题单次检索
        base_titles = titles_of(rec["question"])

        cov = coverage(agent_titles, gold)
        cov_f = coverage(first_titles, gold)
        bcov = coverage(base_titles, gold)
        cov_sum += cov
        cov_first_sum += cov_f
        base_cov_sum += bcov
        n_full += cov == 1.0
        n_partial += cov > 0
        base_full += bcov == 1.0
        base_partial += bcov > 0

        per_q.append({
            "id": rec["id"],
            "n_queries": len(queries),
            "queries": queries,
            "gold_titles": sorted(gold),
            "hit_titles": sorted(gold & agent_titles),
            "coverage": cov,
            "first_query_coverage": cov_f,
            "baseline_coverage": bcov,
            "em": rec.get("em", 0),
        })

    n = len(per_q)
    analysis = {
        "n": n,
        "agent": {
            "gold_hit_full": n_full / n if n else 0.0,
            "gold_hit_partial": n_partial / n if n else 0.0,
            "gold_coverage": cov_sum / n if n else 0.0,
            "first_query_coverage": cov_first_sum / n if n else 0.0,
            "avg_queries": sum(p["n_queries"] for p in per_q) / n if n else 0.0,
        },
        "baseline_single_query": {
            "gold_hit_full": base_full / n if n else 0.0,
            "gold_hit_partial": base_partial / n if n else 0.0,
            "gold_coverage": base_cov_sum / n if n else 0.0,
        },
        "per_question": per_q,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    logger.info("n=%d", n)
    logger.info("agent   full=%.3f partial=%.3f coverage=%.3f first_q_cov=%.3f avg_q=%.2f",
                analysis["agent"]["gold_hit_full"], analysis["agent"]["gold_hit_partial"],
                analysis["agent"]["gold_coverage"], analysis["agent"]["first_query_coverage"],
                analysis["agent"]["avg_queries"])
    logger.info("baseline full=%.3f partial=%.3f coverage=%.3f",
                analysis["baseline_single_query"]["gold_hit_full"],
                analysis["baseline_single_query"]["gold_hit_partial"],
                analysis["baseline_single_query"]["gold_coverage"])
    logger.info("写入 %s", out_path)


if __name__ == "__main__":
    main()
