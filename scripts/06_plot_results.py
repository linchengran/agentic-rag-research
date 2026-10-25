"""从实验结果生成可视化图表。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.visualization import (  # noqa: E402
    plot_accuracy_vs_steps,
    plot_retrieval_steps_distribution,
)
from src.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可视化图表")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out-dir", default="experiments/figures")
    args = parser.parse_args()

    logger = get_logger("plot_results", level="INFO")
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有结果文件
    points = []
    for f in results_dir.glob("agentic_rag_*.json"):
        with f.open("r", encoding="utf-8") as fp:
            records = json.load(fp)
        n = max(1, len(records))
        em = sum(r["em"] for r in records) / n
        avg_steps = sum(r["n_steps"] for r in records) / n
        points.append({
            "label": f.stem.replace("agentic_rag_", ""),
            "steps": avg_steps,
            "accuracy": em,
        })

    if points:
        plot_accuracy_vs_steps(points, out_dir / "accuracy_vs_steps.png")
        logger.info("已生成 accuracy_vs_steps.png")

    # 直方图
    for f in results_dir.glob("agentic_rag_*.json"):
        with f.open("r", encoding="utf-8") as fp:
            records = json.load(fp)
        from src.evaluation.trace import load_traces
        try:
            traces = load_traces(f.parent.parent / "traces" / f.stem.replace("agentic_rag_", ""))
        except Exception:
            continue
        from src.agents.base import AgentOutput
        outputs = [AgentOutput(**{k: t.get(k) for k in ("question", "answer", "steps", "finished", "finish_reason", "total_input_tokens", "total_output_tokens", "latency_seconds", "metadata")}) for t in traces]
        plot_retrieval_steps_distribution(outputs, out_dir / f"steps_dist_{f.stem.replace('agentic_rag_', '')}.png")
        logger.info("已生成 steps_dist_*.png")


if __name__ == "__main__":
    main()