"""导出论文表格(Markdown + LaTeX/booktabs)→ experiments/tables/。

数据源全部可选:存在才出表,可在任意阶段重复运行:
  T1 主对比(baseline + 4 范式,summary_n100)
  T2 决策指标(decision_analysis_n100_*)
  T3 金标命中重放(gold_hit_replay_n100_*)
  T4 per-step 边际恢复率(stepwise_coverage_n100)
  T5 guard 对比(guard_comparison_n100_*)
  T6 self_ask n=500(n500 合并文件,存在时)
  T7 泛化(2wiki / musique,存在时)

用法:python scripts/15_export_paper_tables.py
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_metrics import relaxed_em  # noqa: E402

MATRIX = ROOT / "experiments" / "results" / "matrix"
TABLES = ROOT / "experiments" / "tables"
PARADIGMS = ["react", "plan_execute", "self_ask", "ircot"]
DISPLAY = {"react": "ReAct", "plan_execute": "Plan-and-Execute",
           "self_ask": "Self-Ask", "ircot": "IRCoT"}


def load(p: Path):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def mcnemar_p(agent_recs: list[dict], base_map: dict[str, dict]) -> float:
    only_a = only_b = 0
    for r in agent_recs:
        if r["id"] not in base_map:
            continue
        a, b = bool(r["em"]), bool(base_map[r["id"]]["em"])
        only_a += a and not b
        only_b += b and not a
    m = only_a + only_b
    if m == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(m, i) for i in range(0, k + 1)) / 2 ** m)


def write(stem: str, header: list[str], rows: list[list[str]], note: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (TABLES / f"{stem}.md").write_text("\n".join(md) + f"\n\n注:{note}\n", encoding="utf-8")

    tex = [
        "\\begin{table}[htbp]", "\\centering", "\\caption{" + note + "}",
        "\\begin{tabular}{" + "l" * (len(header) - 1) + "c}", "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule",
        *[ " & ".join(r) + " \\\\" for r in rows ],
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    (TABLES / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"[ok] {stem}.md / .tex")


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def t1_main() -> None:
    summary = load(MATRIX / "summary_n100.json")
    base500 = load(ROOT / "experiments/results/baseline_rag_v2.json")
    if not (summary and base500):
        print("[skip] T1(缺 summary_n100 或 baseline)")
        return
    ctrl = {r["paradigm"]: r for r in summary if r["arm"] == "control"}
    ids100 = [r["id"] for r in load(MATRIX / "react_n100_ms8.json")]
    bmap = {r["id"]: r for r in base500}
    sub = [bmap[i] for i in ids100 if i in bmap]
    n = len(sub)
    base_same = {
        "em": sum(r["em"] for r in sub) / n,
        "relaxed_em": sum(r.get("relaxed_em", relaxed_em(r["pred_answer"], r["gold_answer"])) for r in sub) / n,
        "f1": sum(r["answer_f1"] for r in sub) / n,
    }
    base500m = {
        "em": sum(r["em"] for r in base500) / len(base500),
        "relaxed_em": sum(r.get("relaxed_em", relaxed_em(r["pred_answer"], r["gold_answer"])) for r in base500) / len(base500),
        "f1": sum(r["answer_f1"] for r in base500) / len(base500),
    }
    header = ["范式", "EM", "EM(relaxed)", "F1", "平均步数", "McNemar p (vs 同题集 baseline)"]
    rows = [[f"Baseline (n=500)", pct(base500m["em"]), pct(base500m["relaxed_em"]),
             pct(base500m["f1"]), "1.0", "—"],
            [f"Baseline (同题集 n={n})", pct(base_same["em"]), pct(base_same["relaxed_em"]),
             pct(base_same["f1"]), "1.0", "—"]]
    for p in PARADIGMS:
        r = ctrl[p]
        pv = mcnemar_p(load(MATRIX / f"{p}_n100_ms8.json"), bmap)
        rows.append([DISPLAY[p], pct(r["em"]), pct(r["relaxed_em"]), pct(r["f1"]),
                     f"{r['avg_steps']:.2f}", f"{pv:.3f}"])
    write("t1_main_comparison", header, rows,
          "HotpotQA dev 前 100 题,max_steps=8,贪心解码;baseline 同题集为同一百题的单次检索结果")


def t2_decision() -> None:
    rows, header = [], ["范式", "平均检索次数", "过度检索率", "检索不足率", "决策正确率"]
    for p in PARADIGMS:
        d = load(MATRIX / f"decision_analysis_n100_{p}.json")
        if not d:
            print("[skip] T2(缺 decision_analysis)")
            return
        m = d.get("decision_metrics", d)
        rows.append([DISPLAY[p], f"{m['avg_n_retrievals']:.2f}", pct(m["over_retrieval_rate"]),
                     pct(m["under_retrieval_rate"]), pct(m["decision_accuracy"])])
    write("t2_decision_metrics", header, rows,
          "n_needed=2(每题两篇金标支撑文档);under = 检索不足且答案错误;n=100")


def t3_gold_hit() -> None:
    rows, header = [], ["范式", "金标覆盖率", "full 命中(两篇全中)", "首 query 覆盖", "唯一 query/题"]
    base_row = None
    for p in PARADIGMS:
        d = load(MATRIX / f"gold_hit_replay_n100_{p}.json")
        if not d:
            print("[skip] T3(缺 gold_hit_replay)")
            return
        a, b = d["agent"], d["baseline_single_query"]
        rows.append([DISPLAY[p], pct(a["gold_coverage"]), pct(a["gold_hit_full"]),
                     pct(a["first_query_coverage"]), f"{a['avg_queries']:.2f}"])
        base_row = [b["gold_coverage"], b["gold_hit_full"]]
    rows.append(["Baseline(原问题单查)", pct(base_row[0]), pct(base_row[1]), "—", "1"])
    write("t3_gold_hit_replay", header, rows,
          "重放式命中率:把轨迹中每个检索 query 喂回同一检索器取完整 top-5 再算,确定性、不跑 LLM")


def t4_stepwise() -> None:
    d = load(MATRIX / "stepwise_coverage_n100.json")
    if not d:
        print("[skip] T4(缺 stepwise)")
        return
    max_step = 5
    header = ["范式", "总检索", "重复 query"] + [f"step {k}" for k in range(1, max_step + 1)]
    rows = []
    for p in PARADIGMS:
        s = d.get(f"{p}_n100")
        if not s:
            continue
        curve = s["curve"]
        row = [DISPLAY[p], str(s["total_queries"]),
               f"{s['repeat_queries_total']}({s['repeat_queries_total'] / max(1, s['total_queries']) * 100:.0f}%)"]
        for k in range(1, max_step + 1):
            c = curve.get(str(k))
            row.append(pct(c["marginal_recovery_rate"]) if c else "—")
        rows.append(row)
    write("t4_stepwise_recovery", header, rows, "边际恢复率 = 第 k 步新命中金标标题 / 该步开始时未命中数;step k 到达题数见 stepwise_coverage_n100.json")


def t5_guard() -> None:
    rows, header = [], ["范式", "EM (控制→守卫)", "F1 (控制→守卫)", "步数", "真实检索/题", "Δ检索", "触碰题数"]
    for p in PARADIGMS:
        d = load(MATRIX / f"guard_comparison_n100_{p}.json")
        if not d:
            print("[skip] T5(缺 guard_comparison)")
            return
        c, t = d[p]["control"], d[p]["treated"]
        delta = (t["avg_real_searches"] - c["avg_real_searches"]) / c["avg_real_searches"] * 100
        rows.append([DISPLAY[p], f"{c['em']:.3f}→{t['em']:.3f}", f"{c['f1']:.3f}→{t['f1']:.3f}",
                     f"{c['avg_steps']:.2f}→{t['avg_steps']:.2f}",
                     f"{c['avg_real_searches']:.2f}→{t['avg_real_searches']:.2f}",
                     f"{delta:+.0f}%", f"{t['n_guard_touched']}"])
    write("t5_guard_comparison", header, rows,
          "guard-nudge 臂(重复 query 检测器 + 零收益早停),n=100 同题;触碰题答案变化处 EM 变化 0")


def t6_self_ask_n500() -> None:
    path = MATRIX / "self_ask_n500_ms8.json"
    if not path.exists():
        print("[skip] T6(self_ask n500 未完成)")
        return
    recs = load(path)
    base500 = load(ROOT / "experiments/results/baseline_rag_v2.json")
    bmap = {r["id"]: r for r in base500}
    pv = mcnemar_p(recs, bmap)
    n = len(recs)
    em = sum(r["em"] for r in recs) / n
    rem = sum(r.get("relaxed_em", relaxed_em(r["pred_answer"], r["gold_answer"])) for r in recs) / n
    f1 = sum(r["answer_f1"] for r in recs) / n
    b = {k: sum(r[k] for r in base500) / len(base500) for k in ("em", "answer_f1")}
    header = ["范式", "n", "EM", "EM(relaxed)", "F1", "baseline EM", "McNemar p"]
    rows = [["Self-Ask", str(n), pct(em), pct(rem), pct(f1), pct(b["em"]), f"{pv:.3f}"],
            ["Baseline", str(len(base500)), pct(b["em"]), "30.2", pct(b["answer_f1"]), "—", "—"]]
    write("t6_self_ask_n500", header, rows, "self_ask 扩样到与 baseline 相同的 500 题(同题集配对)")


def t7_generalization() -> None:
    header = ["数据集", "范式", "EM", "EM(relaxed)", "F1", "平均步数"]
    rows = []
    for ds in ("2wiki", "musique"):
        base_p = ROOT / "experiments" / "results" / f"baseline_{ds}.json"
        bdir = MATRIX / f"{ds}_batches"
        if base_p.exists():
            recs = load(base_p)
            n = len(recs)
            rows.append([ds.upper(), "Baseline", pct(sum(r["em"] for r in recs) / n), "—",
                         pct(sum(r["answer_f1"] for r in recs) / n), "1.0"])
        # 按范式聚合(从 batch 文件)
        bdir = MATRIX / f"{ds}_batches"
        if bdir.exists():
            for p in PARADIGMS:
                recs = []
                for f in sorted(bdir.glob(f"{p}_off*.json"), key=lambda x: int(x.stem.rsplit("_off", 1)[1])):
                    recs += load(f)
                if not recs:
                    continue
                n = len(recs)
                rows.append([ds.upper(), DISPLAY[p], pct(sum(r["em"] for r in recs) / n),
                             pct(sum(r.get("relaxed_em", relaxed_em(r["pred_answer"], r["gold_answer"])) for r in recs) / n),
                             pct(sum(r["answer_f1"] for r in recs) / n),
                             f"{sum(r['n_steps'] for r in recs) / n:.2f}"])
    if rows:
        write("t7_generalization", header, rows,
              "泛化实验:2WikiMultihopQA / MuSiQue dev 前 100 题,索引取各自前 500 题语料,max_steps=8")
    else:
        print("[skip] T7(泛化实验未完成)")


def main() -> None:
    t1_main()
    t2_decision()
    t3_gold_hit()
    t4_stepwise()
    t5_guard()
    t6_self_ask_n500()
    t7_generalization()
    print(f"表格目录:{TABLES}")


if __name__ == "__main__":
    main()
