"""抽查 relaxed_em 吸收的近失例,人工核对是否有假阳。

用法:python scripts/09_check_relaxed_em.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_metrics import relaxed_em  # noqa: E402


def main() -> None:
    files = [
        ("react_ms8", "experiments/results/matrix/react_n30_ms8.json"),
        ("plan_execute_ms8", "experiments/results/matrix/plan_execute_n30_ms8.json"),
        ("self_ask_ms8", "experiments/results/matrix/self_ask_n30_ms8.json"),
        ("ircot_ms8", "experiments/results/matrix/ircot_n30_ms8.json"),
    ]
    for name, fp in files:
        with open(fp, encoding="utf-8") as f:
            recs = json.load(f)
        for r in recs:
            if r["em"] == 0 and relaxed_em(r["pred_answer"], r["gold_answer"]) == 1.0:
                print(f"[{name}] gold={r['gold_answer']!r:45} pred={r['pred_answer']!r}")

    with open(ROOT / "experiments/results/baseline_rag_v2.json", encoding="utf-8") as f:
        recs = json.load(f)
    print("--- baseline absorbed (前 12) ---")
    k = 0
    for r in recs:
        if r["em"] == 0 and relaxed_em(r["pred_answer"], r["gold_answer"]) == 1.0:
            print(f"gold={r['gold_answer']!r:45} pred={r['pred_answer']!r}")
            k += 1
            if k >= 12:
                break


if __name__ == "__main__":
    main()
