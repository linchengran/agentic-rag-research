"""ReAct 范式实现。

ReAct 是 Yao et al. 2023 提出的范式:Reason + Act 交替。
每个循环:LLM 输出 Thought + Action,执行 Action 得到 Observation,再进入下一轮。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict

from ..actions.base import ActionResult
from ..llm.local_model import LocalModel
from ..llm.prompts import REACT_SYSTEM_PROMPT, REACT_USER_TEMPLATE, get_prompt
from .base import AgentOutput, BaseAgent, StepRecord, fallback_answer


class ReActAgent(BaseAgent):
    """ReAct 范式 Agent。"""

    name = "react"

    def __init__(
        self,
        llm: LocalModel,
        actions: Dict[str, Any],
        max_steps: int = 8,
    ) -> None:
        super().__init__(llm=llm, actions=actions, max_steps=max_steps)

    def _format_history(self, steps: list[StepRecord]) -> str:
        if not steps:
            return "(none)"
        lines = []
        for s in steps:
            lines.append(
                f"Thought: {s.thought}\n"
                f"Action: {s.action_name}[{s.action_input}]\n"
                f"Observation: {s.observation[:1500]}"
            )
        return "\n\n".join(lines)

    def run(self, question: str, **kwargs: Any) -> AgentOutput:
        """ReAct 推理循环。"""
        output = AgentOutput(question=question)
        t0 = time.time()
        steps: list[StepRecord] = []

        for step_idx in range(self.max_steps):
            history = self._format_history(steps)
            user_prompt = REACT_USER_TEMPLATE.format(
                question=question, history=history
            )
            messages = [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            llm_out = self.llm.chat(messages)
            thought, action_text = self._parse_thought_and_action(llm_out)
            action_name, action_input = self.parse_action(action_text)
            record = StepRecord(
                step=step_idx + 1,
                thought=thought,
                action_name=action_name,
                action_input=action_input,
            )

            # 零检索 Finish 拒收:多跳问题必须先检索(与 self_ask v3 同一守卫)
            if action_name == "finish" and not any(
                s.action_name in ("search", "lookup") for s in steps
            ):
                record.action_name = "parse_error"
                record.error = "finish_without_search"
                record.observation = "[Error] You must Search at least once before Finish."
                steps.append(record)
                continue

            # 执行动作
            if action_name in self.actions:
                action = self.actions[action_name]
                result: ActionResult = action(action_input)
                record.observation = result.observation
                record.error = result.error
            else:
                record.observation = f"[Error] Unknown action: {action_name}"
                record.error = "unknown_action"
                result = None

            steps.append(record)

            # 守卫早停(hard 模式):STOP 指令后模型仍在检索 → 直接终止
            guard = self.actions.get("search")
            if getattr(guard, "should_stop", False):
                output.finish_reason = "guard_early_stop"
                output.answer = fallback_answer(steps)
                break

            # 终止判断
            if action_name == "finish" and result is not None and result.success:
                output.answer = action_input
                output.finished = True
                output.finish_reason = "finish_action"
                break

            if action_name == "parse_error":
                # 解析失败:继续,给模型一次纠错机会
                continue

        # 达到 max_steps 仍未 finish(守卫早停已设 finish_reason,不覆盖)
        if not output.finished and not output.finish_reason:
            output.finish_reason = "max_steps"
            output.answer = fallback_answer(steps)

        output.steps = steps
        output.latency_seconds = time.time() - t0
        output.metadata["paradigm"] = self.name
        return output

    @staticmethod
    def _parse_thought_and_action(text: str) -> tuple[str, str]:
        """分离 Thought 和 Action 文本。"""
        thought = ""
        action = ""
        # Thought 段
        m = re.search(r"Thought:\s*(.*?)(?:\n|Action:)", text, re.DOTALL)
        if m:
            thought = m.group(1).strip()
        # Action 段
        m = re.search(r"Action:\s*.*", text, re.DOTALL)
        if m:
            action = m.group(0).strip()
        if not action:
            action = text.strip()  # 兜底
        return thought, action


__all__ = ["ReActAgent"]