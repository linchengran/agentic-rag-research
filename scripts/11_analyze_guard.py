"""Intervention 实验分析:守卫组(重复 query 检测器 + 零收益早停)vs 控制组。

回答的问题(REPORT §5 下一步 #2):
1. 守卫省了多少步/多少次无效检索?(工程收益)
2. EM/F1 有没有掉?(干预无代价检验)
3. 被守卫截停的题,控制组本来能答对吗?(伤害定位)
4. hard(强制终止)与 nudge(只注入指令)差多少?(3B 服从性)

数据源:控制组 = ms8 矩阵结果,实验组 = --guard hard/nudge 重跑(同 30 题)。
离线运行,不跑 LLM。

用法:
    python scripts/11_analyze_guard.py --all                 # 4 范式默认路径
    python scripts/11_analyze_guard.py --label react --control <json> --treated <json> \
        --control-trace-dir <dir> --treated-trace-dir <dir>
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

SEARCH_ACTIONS = {"search", "ircot_search"}
BLOCKED_ERRORS = {"duplicate_query", "early_stop_requested"}
GUARD_EVENT_TYPES = ("duplicate_query", "zero_new_docs", "stop_triggered", "stop_ignored")

# (控制组结果, 控制组轨迹, 实验组结果, 实验组轨迹)
PARADIGMS = {
    "react": (
        "experiments/results/matrix/react_n30_ms8.json",
        "experiments/traces/react_ms8",
        "experiments/results/matrix/react_n30_ms8_guard-hard.json",
        "experiments/traces/react_ms8_guard-hard",
    ),
    "plan_execute": (
        "experiments/results/matrix/plan_execute_n30_ms8.json",
        "experiments/traces/plan_execute_ms8",
        "experiments/results/matrix/plan_execute_n30_ms8_guard-hard.json",
        "experiments/traces/plan_execute_ms8_guard-hard",
    ),
    "self_ask": (
        "experiments/results/matrix/self_ask_n30_ms8.json",
        "experiments/traces/self_ask_ms8",
        "experiments/results/matrix/self_ask_n30_ms8_guard-hard.json",
        "experiments/traces/self_ask_ms8_guard-hard",
    ),
    "ircot": (
        "experiments/results/matrix/ircot_n30_ms8.json",
        "experiments/traces/ircot_ms8",
        "experiments/results/matrix/ircot_n30_ms8_guard-hard.json",
        "experiments/traces/ircot_ms8_guard-hard",
    ),
}


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_trace_steps(trace_dir: str) -> dict[str, list[dict]]:
    """question -> steps(轨迹里的 step 列表,含 action_name / error)。"""
    q2steps = {}
    for tp in sorted(Path(trace_dir).glob("trace_*.json")):
        t = load_json(tp)
        q2steps[t.get("question", "")] = t.get("steps", [])
    return q2steps


def count_real_searches(steps: list[dict]) -> int:
    """真实进检索器的次数:search 步里守卫拦截(error 属于 BLOCKED_ERRORS)除外。"""
    return sum(
        1
        for s in steps
        if s.get("action_name") in SEARCH_ACTIONS
        and s.get("error") not in BLOCKED_ERRORS
    )


def arm_stats(records: list[dict], trace_steps: dict[str, list[dict]]) -> dict:
    """单臂统计:EM/F1 双口径、步数、真实检索、终止原因分布。"""
    n = len(records)
    em = sum(r.get("em", 0.0) for r in records) / n
    f1 = sum(r.get("answer_f1", 0.0) for r in records) / n
    # records 里已存 relaxed_em(runner 的 evaluate_answer 计算);缺字段时离线重算兜底
    rem = sum(
        r.get("relaxed_em", relaxed_em(r.get("pred_answer", ""), r.get("gold_answer", "")))
        for r in records
    ) / n
    avg_steps = sum(r.get("n_steps", 0) for r in records) / n
    finish_dist = Counter(r.get("finish_reason", "") for r in records)

    real_searches_total = sum(
        count_real_searches(trace_steps.get(r.get("question", ""), [])) for r in records
    )

    # 守卫干预统计(控制组没有 guard_events,全为 0)
    event_counts = Counter()
    touched_ids, stopped_ids = [], []
    for r in records:
        events = r.get("guard_events") or []
        if events:
            touched_ids.append(r.get("id"))
        if r.get("finish_reason") == "guard_early_stop":
            stopped_ids.append(r.get("id"))
        for e in events:
            event_counts[e.get("event", "?")] += 1

    return {
        "n": n,
        "em": em,
        "relaxed_em": rem,
        "f1": f1,
        "avg_steps": avg_steps,
        "total_real_searches": real_searches_total,
        "avg_real_searches": real_searches_total / n,
        "finish_reason_dist": dict(finish_dist),
        "guard_events": {k: event_counts.get(k, 0) for k in GUARD_EVENT_TYPES},
        "n_guard_touched": len(touched_ids),
        "n_guard_stopped": len(stopped_ids),
        "_touched_ids": touched_ids,
        "_stopped_ids": stopped_ids,
    }


def compare(label: str, control_recs: list[dict], treated_recs: list[dict],
            control_trace: dict, treated_trace: dict, logger) -> dict:
    ctrl = arm_stats(control_recs, control_trace)
    trtd = arm_stats(treated_recs, treated_trace)

    # 同题 join:被守卫截停/触碰的题,控制组本来表现如何(伤害定位)
    ctrl_by_id = {r["id"]: r for r in control_recs}
    trtd_by_id = {r["id"]: r for r in treated_recs}
    shared = set(ctrl_by_id) & set(trtd_by_id)

    def em_on(ids, table):
        ids = [i for i in ids if i in shared]
        return (sum(table[i].get("em", 0.0) for i in ids) / len(ids)) if ids else None

    stopped_ctrl_em = em_on(trtd["_stopped_ids"], ctrl_by_id)
    stopped_trtd_em = em_on(trtd["_stopped_ids"], trtd_by_id)
    touched_ctrl_em = em_on(trtd["_touched_ids"], ctrl_by_id)
    touched_trtd_em = em_on(trtd["_touched_ids"], trtd_by_id)

    report = {
        "paradigm": label,
        "n": len(shared),
        "control": {k: v for k, v in ctrl.items() if not k.startswith("_")},
        "treated": {k: v for k, v in trtd.items() if not k.startswith("_")},
        "delta": {
            "em": trtd["em"] - ctrl["em"],
            "relaxed_em": trtd["relaxed_em"] - ctrl["relaxed_em"],
            "f1": trtd["f1"] - ctrl["f1"],
            "avg_steps": trtd["avg_steps"] - ctrl["avg_steps"],
            "avg_real_searches": trtd["avg_real_searches"] - ctrl["avg_real_searches"],
        },
        "harm_check": {
            "note": "守卫截停/触碰的题在控制组的 EM——若控制组也接近 0,则干预无伤害",
            "guard_stopped_questions": {
                "n": len(trtd["_stopped_ids"]),
                "control_em": stopped_ctrl_em,
                "treated_em": stopped_trtd_em,
            },
            "guard_touched_questions": {
                "n": len(trtd["_touched_ids"]),
                "control_em": touched_ctrl_em,
                "treated_em": touched_trtd_em,
            },
        },
    }

    logger.info(
        "[%s] EM %.3f -> %.3f (Δ%+.3f) | relaxed %.3f -> %.3f | F1 %.3f -> %.3f | "
        "steps %.2f -> %.2f | real_search %.2f -> %.2f | 截停 %d 题(控制组EM=%s)",
        label, ctrl["em"], trtd["em"], report["delta"]["em"],
        ctrl["relaxed_em"], trtd["relaxed_em"], ctrl["f1"], trtd["f1"],
        ctrl["avg_steps"], trtd["avg_steps"],
        ctrl["avg_real_searches"], trtd["avg_real_searches"],
        report["treated"]["n_guard_stopped"],
        f"{stopped_ctrl_em:.3f}" if stopped_ctrl_em is not None else "n/a",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="守卫 intervention 实验分析")
    parser.add_argument("--all", action="store_true", help="4 范式默认路径(ms8 控制 vs guard-hard)")
    parser.add_argument("--label")
    parser.add_argument("--control")
    parser.add_argument("--treated")
    parser.add_argument("--control-trace-dir")
    parser.add_argument("--treated-trace-dir")
    parser.add_argument("--out", default="experiments/results/matrix/guard_comparison.json")
    args = parser.parse_args()

    logger = get_logger("analyze_guard", level="INFO")

    jobs = []
    if args.all:
        for label, paths in PARADIGMS.items():
            jobs.append((label,) + paths)
    else:
        assert args.label and args.control and args.treated
        jobs.append((args.label, args.control, args.control_trace_dir or "",
                     args.treated, args.treated_trace_dir or ""))

    report = {}
    for label, ctrl_res, ctrl_tr, trtd_res, trtd_tr in jobs:
        if not Path(ctrl_res).exists() or not Path(trtd_res).exists():
            logger.warning("[%s] 结果文件缺失,跳过(%s / %s)", label, ctrl_res, trtd_res)
            continue
        control_recs = load_json(ctrl_res)
        treated_recs = load_json(trtd_res)
        ctrl_trace = load_trace_steps(ctrl_tr) if ctrl_tr and Path(ctrl_tr).exists() else {}
        trtd_trace = load_trace_steps(trtd_tr) if trtd_tr and Path(trtd_tr).exists() else {}
        report[label] = compare(label, control_recs, treated_recs, ctrl_trace, trtd_trace, logger)

    if not report:
        logger.error("没有可分析的范式;先跑实验组:python scripts/03_run_agentic_rag.py --guard hard ...")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("写入 %s", out_path)


if __name__ == "__main__":
    main()
