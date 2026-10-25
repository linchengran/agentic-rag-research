"""金标文档命中分析:Agent 各范式实际检索到的文档 vs HotpotQA 金标支持文档。

与答案对错完全解耦的"检索决策质量"指标:
- gold_hit_full:  两篇金标文档都被检索到(至少一次)的比例
- gold_hit_partial: 至少一篇被检索到的比例
- gold_coverage:  平均每篇金标文档被检索到的比例(0-1)

对比参照:W3 单次检索 baseline Recall@5 = 0.918(500 题,top-5 含金标句)。

用法:
    python scripts/07_gold_doc_hit.py \
        --results experiments/results/matrix/self_ask_n30_v3.json \
        --trace-dir experiments/traces/self_ask_v3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging import get_logger  # noqa: E402

# 观察文本里的文档格式:"[Doc 1] Title: content..."
DOC_TITLE_RE = re.compile(r"\[\s*Doc\s*\d+\s*\]\s*(.+?):\s")


def parse_retrieved_titles(observation: str) -> list[str]:
    """从一条 observation 里解析出所有检索到的文档标题。"""
    if not observation:
        return []
    return [m.strip() for m in DOC_TITLE_RE.findall(observation)]


def main() -> None:
    parser = argparse.ArgumentParser(description="金标文档命中分析")
    parser.add_argument("--results", required=True, help="结果 JSON(含 id)")
    parser.add_argument("--trace-dir", required=True, help="轨迹目录")
    parser.add_argument(
        "--data", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        help="HotpotQA dev JSON(取 supporting_facts)",
    )
    parser.add_argument("--out", default=None, help="输出 JSON 路径(默认不写盘)")
    args = parser.parse_args()

    logger = get_logger("gold_doc_hit", level="INFO")

    # 1. id -> 金标支持文档标题集合
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    id2gold: dict[str, set[str]] = {
        item["_id"]: {sf["title"] for sf in item["supporting_facts"]}
        for item in data
    }

    with open(args.results, encoding="utf-8") as f:
        records = json.load(f)

    # 2. question -> 轨迹(一次性建索引,避免按序号对齐的脆弱假设)
    trace_dir = Path(args.trace_dir)
    q2trace: dict[str, list[dict]] = defaultdict(list)
    for tp in sorted(trace_dir.glob("trace_*.json")):
        with open(tp, encoding="utf-8") as f:
            t = json.load(f)
        q2trace[t.get("question", "")].append(t)

    id2retrieved: dict[str, set[str]] = {}
    for rec in records:
        retrieved: set[str] = set()
        for t in q2trace.get(rec["question"], []):
            for step in t.get("steps", []):
                retrieved.update(parse_retrieved_titles(step.get("observation", "")))
        id2retrieved[rec["id"]] = retrieved

    # 3. 命中统计
    n = 0
    full = partial = 0
    coverage_sum = 0.0
    per_q: list[dict] = []
    for rec in records:
        gold = id2gold.get(rec["id"])
        if not gold:
            continue
        n += 1
        retrieved = id2retrieved[rec["id"]]
        hit = gold & retrieved
        cov = len(hit) / len(gold)
        coverage_sum += cov
        if hit:
            partial += 1
        if hit == gold:
            full += 1
        per_q.append({
            "id": rec["id"],
            "gold_titles": sorted(gold),
            "hit_titles": sorted(hit),
            "coverage": cov,
            "em": rec.get("em", 0),
        })

    analysis = {
        "n": n,
        "gold_hit_full": full / n if n else 0.0,
        "gold_hit_partial": partial / n if n else 0.0,
        "gold_coverage": coverage_sum / n if n else 0.0,
        "baseline_reference_recall_at_5": 0.918,
        "per_question": per_q,
    }

    logger.info("n=%d full=%.3f partial=%.3f coverage=%.3f",
                n, analysis["gold_hit_full"], analysis["gold_hit_partial"],
                analysis["gold_coverage"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info("写入 %s", out_path)


if __name__ == "__main__":
    main()
