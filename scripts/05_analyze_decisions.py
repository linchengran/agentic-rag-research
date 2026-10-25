"""分析 Agent 决策行为,输出统计报告。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.decision_metrics import aggregate_decision_metrics  # noqa: E402
from src.evaluation.trace import load_traces  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="分析决策行为")
    parser.add_argument("--trace-dir", required=True, help="轨迹目录")
    parser.add_argument("--results", required=True, help="结果 JSON 路径")
    parser.add_argument("--out", default="experiments/results/decision_analysis.json")
    args = parser.parse_args()

    logger = get_logger("decision_analysis", level="INFO")

    traces = load_traces(args.trace_dir)
    with open(args.results, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("加载 %d 条轨迹 / %d 条结果", len(traces), len(records))

    # 失败模式分布
    fail_reasons: dict[str, int] = {}
    for r in records:
        fail_reasons[r.get("finish_reason", "unknown")] = (
            fail_reasons.get(r.get("finish_reason", "unknown"), 0) + 1
        )

    # 决策指标聚合
    decision_metrics = aggregate_decision_metrics([
        type("E", (), {
            "decision_correct": r["decision_correct"],
            "over_retrieved": r["over_retrieved"],
            "under_retrieved": r["under_retrieved"],
            "n_retrievals": r["n_retrievals"],
            "n_needed": r.get("n_needed", 1),
            "final_answer_correct": r["final_answer_correct"],
        })
        for r in records
    ])

    analysis = {
        "n_samples": len(records),
        "finish_reason_distribution": fail_reasons,
        "decision_metrics": decision_metrics,
        "answer_metrics": {
            "em": sum(r["em"] for r in records) / max(1, len(records)),
            "answer_f1": sum(r["answer_f1"] for r in records) / max(1, len(records)),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logger.info("决策分析完成,写入 %s", out_path)
    logger.info("关键指标: %s", json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()