"""决策行为可视化工具(占位实现)。

后续会用 matplotlib / seaborn 生成具体图表,这里定义接口规范。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")  # 无显示器环境也能画图

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..agents.base import AgentOutput


def plot_retrieval_steps_distribution(
    outputs: List[AgentOutput],
    save_path: str | Path,
) -> None:
    """画检索步数分布直方图。"""
    counts = [len([s for s in o.steps if s.action_name in ("search", "self_ask_query", "ircot_search")]) for o in outputs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=range(0, max(counts or [1]) + 2), edgecolor="black")
    ax.set_xlabel("Retrieval steps per question")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Retrieval Steps")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_steps(
    points: List[Dict[str, Any]],
    save_path: str | Path,
) -> None:
    """画检索步数-准确率 Pareto 曲线。

    points: 每个元素形如 {"steps": 3, "accuracy": 0.45, "label": "react"}
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in sorted({p["label"] for p in points}):
        xs = [p["steps"] for p in points if p["label"] == label]
        ys = [p["accuracy"] for p in points if p["label"] == label]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Max retrieval steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Retrieval Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_decision_heatmap(
    confusion: np.ndarray,
    save_path: str | Path,
    labels: List[str] | None = None,
) -> None:
    """画决策行为混淆矩阵热力图。"""
    labels = labels or ["correct", "over", "under"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Decision Behavior Confusion")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)