"""Agent 决策行为评估指标。

与"答案对错"解耦,专门评估 Agent 的检索决策质量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set

from ..agents.base import AgentOutput


@dataclass
class DecisionEval:
    """单条样本的决策评估结果。"""

    decision_correct: bool
    over_retrieved: bool
    under_retrieved: bool
    n_retrievals: int
    n_needed: int
    final_answer_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_correct": self.decision_correct,
            "over_retrieved": self.over_retrieved,
            "under_retrieved": self.under_retrieved,
            "n_retrievals": self.n_retrievals,
            "n_needed": self.n_needed,
            "final_answer_correct": self.final_answer_correct,
        }


def compute_decision_eval(
    output: AgentOutput,
    gold_supporting_titles: Set[str],
    answer_correct: bool,
) -> DecisionEval:
    """评估 Agent 在单条样本上的决策行为。

    Args:
        output: Agent 推理输出。
        gold_supporting_titles: 该样本需要的支撑文档标题集合(从数据集获取)。
        answer_correct: Agent 最终答案是否正确(由 answer_metrics 判断)。

    评估逻辑:
        - over_retrieved: 检索了不必要的事实 / 检索次数 > 必要次数
        - under_retrieved: 检索次数不足,导致未能获得全部支撑事实
        - decision_correct: 决策时机和次数基本合理(实际答案正确即算合理)
    """
    # 统计 Agent 实际检索次数
    n_retrievals = sum(
        1
        for step in output.steps
        if step.action_name in ("search", "self_ask_query", "ircot_search")
    )
    n_needed = max(1, len(gold_supporting_titles))  # 简化:每个支撑标题至少一次

    over = n_retrievals > n_needed + 1
    under = n_retrievals < n_needed and not answer_correct
    correct = answer_correct and not over

    return DecisionEval(
        decision_correct=correct,
        over_retrieved=over,
        under_retrieved=under,
        n_retrievals=n_retrievals,
        n_needed=n_needed,
        final_answer_correct=answer_correct,
    )


def aggregate_decision_metrics(evals: List[DecisionEval]) -> Dict[str, float]:
    """对一组决策评估求平均。"""
    if not evals:
        return {}
    n = len(evals)
    return {
        "decision_accuracy": sum(e.decision_correct for e in evals) / n,
        "over_retrieval_rate": sum(e.over_retrieved for e in evals) / n,
        "under_retrieval_rate": sum(e.under_retrieved for e in evals) / n,
        "avg_n_retrievals": sum(e.n_retrievals for e in evals) / n,
        "answer_accuracy": sum(e.final_answer_correct for e in evals) / n,
    }