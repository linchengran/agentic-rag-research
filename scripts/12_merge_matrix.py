"""正式矩阵分批结果合并:ms100_batches → {paradigm}_n100_ms8.json(+ guard 臂)。

输入约定:
- 控制组:n=30 旧批(ms8_batches/{p}_off*.json,ircot 10/批、其余 6/批)
        + n=100 新批(ms100_batches/{p}_off{30..90}.json,10/批)
- guard-nudge 臂:react 旧批(ms8_batches/guard/react_nudge_off*.json)
        + ms100_batches/guard/{p}_nudge_off*.json(react 补 30-99,其余 0-99)

合并规则:按 offset 升序拼接、按 id 去重;输出 n 必须等于 100 否则报错退出。
同时输出 summary(EM / relaxed EM / F1 / AvgSteps / finish 分布)。

用法:
    python scripts/12_merge_matrix.py            # 全部范式 + guard 臂
    python scripts/12_merge_matrix.py --paradigm react
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_metrics import relaxed_em  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

PARADIGMS = ["react", "plan_execute", "self_ask", "ircot"]
MATRIX = ROOT / "experiments" / "results" / "matrix"
N_TARGET = 100


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def batch_files(pattern_dir: Path, stem_glob: str) -> list[tuple[int, Path]]:
    """收集 (offset, path),offset 从文件名 _off<N> 解析。"""
    out = []
    for p in pattern_dir.glob(stem_glob):
        name = p.stem
        off = name.rsplit("_off", 1)[1]
        out.append((int(off), p))
    return sorted(out)


def merge_stem(paradigm: str, groups: list[tuple[Path, str]], logger) -> list[dict]:
    """groups: [(目录, glob), ...] 按 offset 升序合并、按 id 去重。"""
    merged: dict[str, dict] = {}
    for d, g in groups:
        for off, path in batch_files(d, g):
            for rec in load_json(path):
                merged.setdefault(rec["id"], rec)
    records = list(merged.values())
    if len(records) != N_TARGET:
        logger.error("[%s] 合并后 n=%d != %d,拒绝写出", paradigm, len(records), N_TARGET)
        raise SystemExit(1)
    return records


def summarize(paradigm: str, tag: str, records: list[dict]) -> dict:
    n = len(records)
    return {
        "paradigm": paradigm,
        "arm": tag,
        "n": n,
        "em": sum(r.get("em", 0.0) for r in records) / n,
        "relaxed_em": sum(
            r.get("relaxed_em", relaxed_em(r.get("pred_answer", ""), r.get("gold_answer", "")))
            for r in records
        ) / n,
        "f1": sum(r.get("answer_f1", 0.0) for r in records) / n,
        "avg_steps": sum(r.get("n_steps", 0) for r in records) / n,
        "finish_reason_dist": dict(Counter(r.get("finish_reason", "") for r in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="正式矩阵分批结果合并")
    parser.add_argument("--paradigm", choices=PARADIGMS)
    args = parser.parse_args()

    logger = get_logger("merge_matrix", level="INFO")
    paradigms = [args.paradigm] if args.paradigm else PARADIGMS

    summary = []
    for p in paradigms:
        # 控制组:n=30 旧批(在 ms8_batches)+ n=100 新批(offsets 30-99)
        ctrl = merge_stem(
            p,
            [(MATRIX / "ms8_batches", f"{p}_off*.json"), (MATRIX / "ms100_batches", f"{p}_off*.json")],
            logger,
        )
        ctrl_path = MATRIX / f"{p}_n100_ms8.json"
        ctrl_path.write_text(json.dumps(ctrl, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(summarize(p, "control", ctrl))
        logger.info("[%s] 控制组 %d 题 → %s", p, len(ctrl), ctrl_path)

        # guard-nudge 臂:react 旧批在 ms8_batches/guard,其余全在 ms100_batches/guard
        guard_groups = [(MATRIX / "ms100_batches" / "guard", f"{p}_nudge_off*.json")]
        if p == "react":
            guard_groups.insert(0, (MATRIX / "ms8_batches" / "guard", f"{p}_nudge_off*.json"))
        guard = merge_stem(p, guard_groups, logger)
        guard_path = MATRIX / f"{p}_n100_ms8_guard-nudge.json"
        guard_path.write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(summarize(p, "guard-nudge", guard))
        logger.info("[%s] guard-nudge %d 题 → %s", p, len(guard), guard_path)

    summary_path = MATRIX / "summary_n100.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("汇总写入 %s", summary_path)
    for row in summary:
        logger.info(
            "%s/%s: EM %.3f relaxed %.3f F1 %.3f steps %.2f finish %s",
            row["paradigm"], row["arm"], row["em"], row["relaxed_em"], row["f1"],
            row["avg_steps"], row["finish_reason_dist"],
        )


if __name__ == "__main__":
    main()
