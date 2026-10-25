"""Per-step 边际金标命中曲线:第 1/2/3... 次检索各贡献多少新金标覆盖。

回答的问题:"agent 多走一步,检索决策质量真的变好了吗?"
- 边际覆盖 = 第 k 次检索新增的金标标题数 / 该步之前未命中的金标标题数
- 与"重复 query 检测器"一起输出:第 k 步是否在重复之前的 query

数据源:ms8 轨迹(观察截断已修,trace 含完整 5 篇文档),离线运行,不跑 LLM。
金标标题用 hotpot_dev_distractor_v1.json 的 supporting_facts。

用法:
    python scripts/10_stepwise_coverage.py --results <results.json> --trace-dir <dir> --label <范式>
    或 --all 一次性处理 4 个范式
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

DOC_TITLE_RE = re.compile(r"\[\s*Doc\s*\d+\s*\]\s*(.+?):\s")
SEARCH_ACTIONS = {"search", "ircot_search"}

PARADIGMS = {
    "react": ("experiments/results/matrix/react_n30_ms8.json", "experiments/traces/react_ms8"),
    "plan_execute": ("experiments/results/matrix/plan_execute_n30_ms8.json", "experiments/traces/plan_execute_ms8"),
    "self_ask": ("experiments/results/matrix/self_ask_n30_ms8.json", "experiments/traces/self_ask_ms8"),
    "ircot": ("experiments/results/matrix/ircot_n30_ms8.json", "experiments/traces/ircot_ms8"),
}


def parse_retrieved_titles(observation: str) -> set[str]:
    if not observation:
        return set()
    return {m.strip() for m in DOC_TITLE_RE.findall(observation)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-step 边际金标命中曲线")
    parser.add_argument("--results", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--all", action="store_true", help="处理 4 个 ms8 范式")
    parser.add_argument("--data", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--out", default="experiments/results/matrix/stepwise_coverage.json")
    args = parser.parse_args()

    logger = get_logger("stepwise_coverage", level="INFO")

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    id2gold: dict[str, set[str]] = {
        item["_id"]: {sf["title"] for sf in item["supporting_facts"]}
        for item in data
    }

    jobs = []
    if args.all:
        for label, (res, trace) in PARADIGMS.items():
            jobs.append((label, res, trace))
    else:
        assert args.results and args.trace_dir and args.label
        jobs.append((args.label, args.results, args.trace_dir))

    report: dict[str, dict] = {}
    for label, res_path, trace_dir in jobs:
        with open(res_path, encoding="utf-8") as f:
            records = json.load(f)

        # question -> steps(每步:query, 是否重复, 该步命中的金标)
        q2steps: dict[str, list[dict]] = {}
        for tp in sorted(Path(trace_dir).glob("trace_*.json")):
            with open(tp, encoding="utf-8") as f:
                t = json.load(f)
            steps_out = []
            seen_queries: set[str] = set()
            for s in t.get("steps", []):
                if s.get("action_name") not in SEARCH_ACTIONS:
                    continue
                q = (s.get("action_input") or "").strip()
                if not q:
                    continue
                titles = parse_retrieved_titles(s.get("observation", ""))
                steps_out.append({
                    "query": q,
                    "is_repeat": q in seen_queries,
                    "titles": titles,
                })
                seen_queries.add(q)
            q2steps[t.get("question", "")] = steps_out

        # 边际覆盖统计
        # 桶 k = 第 k 次检索(1-based);统计:有多少题走到第 k 步、该步新增命中多少金标
        step_reached: dict[int, int] = defaultdict(int)     # 走到第 k 步的题数
        step_new_hit: dict[int, int] = defaultdict(int)      # 第 k 步新增命中的金标标题总数
        step_pending: dict[int, int] = defaultdict(int)      # 第 k 步开始时仍未命中的金标标题总数
        step_repeat: dict[int, int] = defaultdict(int)       # 第 k 步是重复 query 的次数
        step_no_new: dict[int, int] = defaultdict(int)       # 第 k 步零新增的次数

        n_q = 0
        for rec in records:
            gold = id2gold.get(rec["id"])
            if not gold:
                continue
            n_q += 1
            covered: set[str] = set()
            steps = q2steps.get(rec["question"], [])
            for k, st in enumerate(steps, start=1):
                step_reached[k] += 1
                pending = gold - covered
                step_pending[k] += len(pending)
                if st["is_repeat"]:
                    step_repeat[k] += 1
                new = gold & st["titles"] - covered
                step_new_hit[k] += len(new)
                if not new:
                    step_no_new[k] += 1
                covered |= gold & st["titles"]

        curve = {}
        for k in sorted(step_reached):
            reached = step_reached[k]
            curve[k] = {
                "n_questions_reached": reached,
                "repeat_queries": step_repeat[k],
                "repeat_rate": step_repeat[k] / reached,
                "zero_new_gain_rate": step_no_new[k] / reached,
                "marginal_hits_per_question": step_new_hit[k] / reached,
                "marginal_recovery_rate": (
                    step_new_hit[k] / step_pending[k] if step_pending[k] else 0.0
                ),
            }

        report[label] = {
            "n_questions": n_q,
            "curve": curve,
            "total_queries": sum(step_reached.values()),
            "repeat_queries_total": sum(step_repeat.values()),
        }
        logger.info("[%s] n=%d 总检索=%d 重复=%d", label, n_q,
                    report[label]["total_queries"],
                    report[label]["repeat_queries_total"])
        for k, c in curve.items():
            logger.info("  step %d: reached=%d repeat=%.2f zero_new=%.2f marginal_recovery=%.3f",
                        k, c["n_questions_reached"], c["repeat_rate"],
                        c["zero_new_gain_rate"], c["marginal_recovery_rate"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("写入 %s", out_path)


if __name__ == "__main__":
    main()
