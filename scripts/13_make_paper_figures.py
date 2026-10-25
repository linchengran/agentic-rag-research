"""论文级图表生成(正式矩阵 n=100,REPORT §3.10)。

四张图,统一身份系统:颜色=范式(Okabe-Ito 色盲安全色板),跨图一致:
  Fig1 主对比    分组柱状:x=3 指标组(EM/relaxed EM/F1),每组 4 范式;
                每组黑色虚线=baseline 同题集;strict-EM 组标注 McNemar p
  Fig2 Pareto   散点:x=平均真实检索次数(trace 重放),y=EM;
                baseline 星标 + 参考十字线
  Fig3 per-step 边际恢复率折线:x=检索步,y=边际恢复率(%);
                reached<10 的点空心(小样本警示)
  Fig4 guard    1×2:左=真实检索次数 control vs guard-nudge(标 Δ%),
                右=EM control vs guard-nudge

导出:PDF/SVG(矢量)+ PNG(300dpi)→ experiments/figures/
图注(中文)由论文正文承载:误差交代 = 各柱为原始比例(n=100),
显著性 = 配对 McNemar 精确检验(strict EM,vs baseline 同题集)。
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_metrics import relaxed_em  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

MATRIX = ROOT / "experiments" / "results" / "matrix"
FIGDIR = ROOT / "experiments" / "figures"

PARADIGMS = ["react", "plan_execute", "self_ask", "ircot"]
DISPLAY = {"react": "ReAct", "plan_execute": "Plan-and-Execute",
           "self_ask": "Self-Ask", "ircot": "IRCoT"}
# Okabe-Ito 色盲安全色板
COLOR = {"react": "#0072B2", "plan_execute": "#E69F00",
         "self_ask": "#009E73", "ircot": "#CC79A7"}
MARKER = {"react": "o", "plan_execute": "s", "self_ask": "^", "ircot": "D"}
LS = {"react": "-", "plan_execute": "--", "self_ask": "-.", "ircot": ":"}

BASELINE_FILE = ROOT / "experiments" / "results" / "baseline_rag_v2.json"


def load_json(p: Path):
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def baseline_same_subset(ids: list[str]) -> dict:
    """baseline 同题集指标(strict/relaxed EM、F1)+ McNemar 精确检验 p 值。"""
    bmap = {r["id"]: r for r in load_json(BASELINE_FILE)}
    sub = [bmap[i] for i in ids]
    n = len(sub)
    metrics = {
        "em": sum(r["em"] for r in sub) / n,
        "relaxed_em": sum(
            r.get("relaxed_em", relaxed_em(r["pred_answer"], r["gold_answer"])) for r in sub
        ) / n,
        "answer_f1": sum(r["answer_f1"] for r in sub) / n,
    }
    return {"n": n, **metrics}


def mcnemar_p(agent_recs: list[dict], base_map: dict[str, dict]) -> tuple[int, int, float]:
    """配对 McNemar(strict EM):返回 (agent独对, baseline独对, 双侧精确 p)。"""
    only_a = only_b = 0
    for r in agent_recs:
        a, b = bool(r["em"]), bool(base_map[r["id"]]["em"])
        if a and not b:
            only_a += 1
        elif b and not a:
            only_b += 1
    m = only_a + only_b
    if m == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(m, i) for i in range(0, k + 1)) / 2 ** m)
    return only_a, only_b, p


def export(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> {name}.pdf/.svg/.png")


# ---------------------------------------------------------------- Fig 1
def fig1_main(summary: list[dict], base: dict, pvals: dict) -> None:
    ctrl = {r["paradigm"]: r for r in summary if r["arm"] == "control"}
    # (summary 键, baseline 键, 显示名)
    metrics = [("em", "em", "EM (strict)"),
               ("relaxed_em", "relaxed_em", "EM (relaxed)"),
               ("f1", "answer_f1", "Answer F1")]
    x = np.arange(len(metrics))
    w = 0.19
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for j, p in enumerate(PARADIGMS):
        vals = [ctrl[p][mk] for mk, _, _ in metrics]
        ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=COLOR[p],
               label=DISPLAY[p], edgecolor="white", linewidth=0.4, zorder=3)
    # baseline 参考线(每组一条,黑虚线)
    for i, (_, bk, _) in enumerate(metrics):
        ax.hlines(base[bk], x[i] - 0.44, x[i] + 0.44, colors="black",
                  linestyles="--", linewidth=1.0, zorder=4)
    ax.hlines([], [], [], colors="black", linestyles="--", linewidth=1.0,
              label="Baseline (same 100 q.)")
    # strict-EM 组上方标注 McNemar p(对 baseline);白衬底防参考线穿字
    pbox = dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85)
    for j, p in enumerate(PARADIGMS):
        pv = pvals[p]
        tag = f"p={pv:.2f}" if pv >= 0.1 else (f"p={pv:.2f}†" if pv >= 0.05 else f"p={pv:.3f}*")
        ax.text(x[0] + (j - 1.5) * w, ctrl[p]["em"] + 0.012, tag,
                ha="center", va="bottom", fontsize=6.2, rotation=90, color=COLOR[p],
                bbox=pbox, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, _, lab in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.47)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    leg = ax.legend(loc="upper right", ncol=1, handlelength=1.2, borderaxespad=0.1)
    leg.set_frame_on(True)
    leg.get_frame().set_edgecolor("0.8")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    export(fig, "fig1_main_comparison")


# ---------------------------------------------------------------- Fig 2
def fig2_pareto(guard_cmp: dict[str, dict], summary: list[dict], base_em: float) -> None:
    ctrl = {r["paradigm"]: r for r in summary if r["arm"] == "control"}
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    ax.axvline(1.0, color="0.75", linestyle=":", linewidth=0.9, zorder=1)
    ax.axhline(base_em, color="0.75", linestyle=":", linewidth=0.9, zorder=1)
    for p in PARADIGMS:
        xv = guard_cmp[p]["control"]["avg_real_searches"]
        yv = ctrl[p]["em"]
        ax.scatter(xv, yv, s=52, color=COLOR[p], marker=MARKER[p],
                   edgecolor="white", linewidth=0.7, zorder=5)
        dx, dy = (0.05, 0.008)
        if p == "self_ask":
            dx, dy = (0.06, 0.0)
        if p == "ircot":
            dy = -0.004
        ax.annotate(DISPLAY[p], (xv, yv), xytext=(xv + dx, yv + dy),
                    fontsize=7, color=COLOR[p], zorder=6)
    ax.scatter([1.0], [base_em], s=90, color="black", marker="*",
               edgecolor="white", linewidth=0.5, zorder=6)
    ax.annotate("Baseline\n(single query)", (1.0, base_em), xytext=(1.05, base_em - 0.028),
                fontsize=7, color="black", va="top")
    ax.text(2.42, base_em + 0.004, "baseline EM", fontsize=6.2, color="0.45")
    ax.set_xlabel("Avg. real retrievals per question")
    ax.set_ylabel("EM (strict)")
    ax.set_xlim(0.85, 2.95)
    ax.set_ylim(0.09, 0.35)
    export(fig, "fig2_pareto_accuracy_vs_cost")


# ---------------------------------------------------------------- Fig 3
def fig3_stepwise(stepwise: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for p in PARADIGMS:
        curve = stepwise[f"{p}_n100"]["curve"]
        steps = sorted(int(k) for k in curve)
        y = [curve[str(s)]["marginal_recovery_rate"] * 100 for s in steps]
        ax.plot(steps, y, LS[p], marker=MARKER[p], color=COLOR[p],
                label=DISPLAY[p], markerfacecolor="white", markeredgewidth=1.1,
                markersize=4.5)
        # reached<10 的点改为实心小点被空心覆盖的警示:再叠一层实心
        solid = [(s, v) for s, v in zip(steps, y)
                 if curve[str(s)]["n_questions_reached"] >= 10]
        if solid:
            ax.scatter([s for s, _ in solid], [v for _, v in solid], s=13,
                       color=COLOR[p], zorder=5)
    ax.set_xlabel("Retrieval step $k$")
    ax.set_ylabel("Marginal recovery rate (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-3, 70)
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    ax.axhline(0, color="0.6", linewidth=0.7)
    export(fig, "fig3_stepwise_marginal_recovery")


# ---------------------------------------------------------------- Fig 4
def fig4_guard(guard_cmp: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    x = np.arange(len(PARADIGMS))
    w = 0.36

    ax = axes[0]
    ctrl = [guard_cmp[p]["control"]["avg_real_searches"] for p in PARADIGMS]
    trtd = [guard_cmp[p]["treated"]["avg_real_searches"] for p in PARADIGMS]
    ax.bar(x - w / 2, ctrl, w, color="0.62", label="Control", edgecolor="white",
           linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, trtd, w, color=COLOR["self_ask"], label="Guard (nudge)",
           edgecolor="white", linewidth=0.4, zorder=3)
    for i, p in enumerate(PARADIGMS):
        d = (trtd[i] - ctrl[i]) / ctrl[i] * 100
        ax.text(x[i], max(ctrl[i], trtd[i]) + 0.06, f"{d:+.0f}%",
                ha="center", va="bottom", fontsize=7,
                color="#B2182B" if d < -5 else "0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[p] for p in PARADIGMS], rotation=14, ha="right")
    ax.set_ylabel("Avg. real retrievals per question")
    ax.set_ylim(0, 2.0)
    ax.set_title("(a) Retrieval cost", loc="left")
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)

    ax = axes[1]
    ctrl_em = [guard_cmp[p]["control"]["em"] for p in PARADIGMS]
    trtd_em = [guard_cmp[p]["treated"]["em"] for p in PARADIGMS]
    ax.bar(x - w / 2, ctrl_em, w, color="0.62", edgecolor="white", linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, trtd_em, w, color=COLOR["self_ask"], edgecolor="white",
           linewidth=0.4, zorder=3)
    for i in range(len(PARADIGMS)):
        top = max(ctrl_em[i], trtd_em[i])
        ax.text(x[i], top + 0.008, f"Δ={trtd_em[i] - ctrl_em[i]:+.2f}",
                ha="center", va="bottom", fontsize=6.2, color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[p] for p in PARADIGMS], rotation=14, ha="right")
    ax.set_ylabel("EM (strict)")
    ax.set_ylim(0, 0.33)
    ax.set_title("(b) Answer accuracy", loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="0.62"),
               plt.Rectangle((0, 0), 1, 1, fc=COLOR["self_ask"])]
    ax.legend(handles, ["Control", "Guard (nudge)"], loc="upper right",
              handlelength=1.2, borderaxespad=0.1)
    fig.tight_layout(w_pad=1.6)
    export(fig, "fig4_guard_intervention")


def main() -> None:
    logger = get_logger("make_figures", level="INFO")
    setup_style()

    summary = load_json(MATRIX / "summary_n100.json")
    ids = [r["id"] for r in load_json(MATRIX / "react_n100_ms8.json")]
    base = baseline_same_subset(ids)
    base_map = {r["id"]: r for r in load_json(BASELINE_FILE)}

    pvals = {}
    for p in PARADIGMS:
        recs = load_json(MATRIX / f"{p}_n100_ms8.json")
        _, _, pv = mcnemar_p(recs, base_map)
        pvals[p] = pv
        logger.info("[fig1] %s McNemar p=%.3f", p, pv)

    guard_cmp = {p: load_json(MATRIX / f"guard_comparison_n100_{p}.json")[p] for p in PARADIGMS}
    stepwise = load_json(MATRIX / "stepwise_coverage_n100.json")

    logger.info("baseline 同题集: EM %.3f / relaxed %.3f / F1 %.3f (n=%d)",
                base["em"], base["relaxed_em"], base["answer_f1"], base["n"])
    fig1_main(summary, base, pvals)
    fig2_pareto(guard_cmp, summary, base["em"])
    fig3_stepwise(stepwise)
    fig4_guard(guard_cmp)
    logger.info("全部图表写入 %s", FIGDIR)


if __name__ == "__main__":
    main()
