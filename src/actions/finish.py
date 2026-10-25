"""Finish 动作:标记 Agent 完成任务并提交答案。

Agent 通过 Finish 提交最终答案,通常触发整个推理循环的终止。
"""
from __future__ import annotations

from typing import Any

from .base import ActionResult, BaseAction


class FinishAction(BaseAction):
    """Finish 动作:提交最终答案。"""

    name = "finish"

    def __init__(self) -> None:
        self.last_answer: str = ""

    def __call__(self, answer: str, **kwargs: Any) -> ActionResult:
        """提交最终答案。

        Args:
            answer: Agent 给出的最终答案。
        """
        self.last_answer = answer.strip() if answer else ""
        return ActionResult(
            name=self.name,
            success=True,
            observation=f"[Finish] Answer: {self.last_answer}",
            raw_input={"answer": self.last_answer},
        )

    @property
    def description(self) -> str:
        return "Finish[answer]: 提交最终答案。"


__all__ = ["FinishAction"]