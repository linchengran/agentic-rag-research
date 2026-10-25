"""运行 Agent 范式 × 检索器 对比矩阵。

遍历 4 种 Agent 范式 × 1 种模型,生成统一格式的结果 CSV。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging import get_logger  # noqa: E402


PARADIGMS = ["react", "plan_execute", "self_ask", "ircot"]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Agent 范式对比矩阵")
    parser.add_argument("--data", required=True, help="HotpotQA JSON 路径")
    parser.add_argument("--index", default="data/processed/index/hotpot_faiss")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--paradigms", nargs="+", default=PARADIGMS, choices=PARADIGMS)
    parser.add_argument("--out-dir", default="experiments/results/matrix")
    args = parser.parse_args()

    logger = get_logger("run_matrix", level="INFO")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for paradigm in args.paradigms:
        logger.info("=== 运行 %s ===", paradigm)
        out_path = out_dir / f"{paradigm}.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "03_run_agentic_rag.py"),
            "--data", args.data,
            "--index", args.index,
            "--agent", paradigm,
            "--out", str(out_path),
            "--limit", str(args.limit),
        ]
        subprocess.run(cmd, check=True)
        with out_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        n = max(1, len(records))
        avg = {k: sum(r.get(k, 0.0) for r in records) / n for k in ("em", "answer_f1", "n_steps")}
        summary.append({"paradigm": paradigm, **avg})

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("矩阵完成。汇总写入 %s", summary_path)
    for row in summary:
        logger.info("%s: %s", row["paradigm"], row)


if __name__ == "__main__":
    main()