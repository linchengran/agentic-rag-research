"""IRCoT 范式实现。

Trivedi et al. 2023 提出:在 CoT 推理的每一步自动触发一次检索,
检索 query 从当前推理步骤中自动抽取。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from ..actions.base import ActionResult
from ..llm.local_model import LocalModel
from ..llm.prompts import IRCOT_SYSTEM_PROMPT
from .base import AgentOutput, BaseAgent, StepRecord, extract_final_answer


class IRCoTAgent(BaseAgent):
    """IRCoT 范式 Agent。"""

    name = "ircot"

    def run(self, question: str, **kwargs: Any) -> AgentOutput:
        output = AgentOutput(question=question)
        t0 = time.time()
        steps: List[StepRecord] = []

        for step_idx in range(self.max_steps):
            prompt = self._build_prompt(question, steps)
            messages = [
                {"role": "system", "content": IRCOT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            llm_out = self.llm.chat(messages)

            # 检查是否已经给出 Final Answer
            final_answer = self._parse_final_answer(llm_out)
            thought = self._parse_thought(llm_out, step_idx + 1)
            record = StepRecord(
                step=step_idx + 1,
                thought=thought,
                action_name="ircot_search",
                action_input=thought,  # 把当前 Thought 作为检索 query
            )

            # 零检索 Final Answer 不收(与 self_ask v3 同一守卫);至少先检索一轮
            if final_answer is not None and steps:
                # LLM 直接给出 Final Answer,终止
                output.answer = final_answer
                output.finished = True
                output.finish_reason = "final_answer"
                record.observation = "直接给出最终答案"
                steps.append(record)
                break

            # 执行检索
            if "search" in self.actions:
                result: ActionResult = self.actions["search"](thought)
                record.observation = result.observation
                record.error = result.error
            else:
                record.observation = "[Error] Search action not available."
                record.error = "no_search_action"

            steps.append(record)

            # 再次检查是否在后续 Observation 后给出 Final Answer
            follow_up = self.llm.chat(
                [
                    {"role": "system", "content": IRCOT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt + "\n" + record.observation[:1500]},
                ]
            )
            final = self._parse_final_answer(follow_up)
            if final is not None:
                output.answer = final
                output.finished = True
                output.finish_reason = "final_answer"
                break

            # 守卫早停(hard 模式):STOP 指令后模型仍在检索 → 直接终止
            guard = self.actions.get("search")
            if getattr(guard, "should_stop", False):
                output.finish_reason = "guard_early_stop"
                output.answer = self._parse_final_answer(steps[-1].thought) or ""
                break

        if not output.finished and not output.finish_reason:
            output.finish_reason = "max_steps"
            output.answer = ""

        output.steps = steps
        output.latency_seconds = time.time() - t0
        output.metadata["paradigm"] = self.name
        return output

    @staticmethod
    def _build_prompt(question: str, steps: List[StepRecord]) -> str:
        parts = [f"Question: {question}"]
        for s in steps:
            parts.append(f"Thought {s.step}: {s.thought}")
            parts.append(f"Observation {s.step}: {s.observation[:1500]}")
        parts.append(f"Thought {len(steps)+1}:")
        return "\n".join(parts)

    @staticmethod
    def _parse_thought(text: str, step_idx: int) -> str:
        m = re.search(rf"Thought\s*{step_idx}:\s*(.+)", text)
        if m:
            return m.group(1).strip()
        # 兜底:取最后一段不含 Final Answer 的内容
        for line in text.splitlines()[::-1]:
            line = line.strip()
            if line and not line.lower().startswith("final answer"):
                return line
        return ""

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        """取最后一个 Final Answer 匹配并清洗(见 base.extract_final_answer)。"""
        return extract_final_answer(text)


__all__ = ["IRCoTAgent"]