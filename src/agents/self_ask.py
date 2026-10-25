"""Self-Ask 范式实现。

Press et al. 2023 提出:让模型"在回答之前先问自己中间问题",
每个中间问题自动触发一次检索。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from ..actions.base import ActionResult
from ..llm.local_model import LocalModel
from ..llm.prompts import SELF_ASK_SYSTEM_PROMPT
from .base import AgentOutput, BaseAgent, StepRecord, extract_final_answer


class SelfAskAgent(BaseAgent):
    """Self-Ask 范式 Agent。"""

    name = "self_ask"

    def run(self, question: str, **kwargs: Any) -> AgentOutput:
        output = AgentOutput(question=question)
        t0 = time.time()
        steps: List[StepRecord] = []

        for step_idx in range(self.max_steps):
            prompt = self._build_prompt(question, steps)
            messages = [
                {"role": "system", "content": SELF_ASK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            llm_out = self.llm.chat(messages)

            # Self-Ask 不使用 Search/Lookup/Finish 格式,而是问 Intermediate Question
            intermediate_q = self._parse_intermediate_question(llm_out)
            record = StepRecord(
                step=step_idx + 1,
                thought=llm_out,
                action_name="self_ask_query",
                action_input=intermediate_q or "",
            )

            # 如果 LLM 直接给出 Final Answer,终止(零检索时不收——多跳问题必须先检索)
            final_answer = self._parse_final_answer(llm_out)
            if final_answer is not None and steps:
                output.answer = final_answer
                output.finished = True
                output.finish_reason = "final_answer"
                record.observation = "直接给出最终答案"
                steps.append(record)
                break

            # 否则,把中间问题作为 Search query
            if intermediate_q and "search" in self.actions:
                result: ActionResult = self.actions["search"](intermediate_q)
                record.action_name = "search"
                record.observation = result.observation
                record.error = result.error
            else:
                record.observation = "[Error] Cannot generate intermediate question."
                record.error = "no_intermediate_question"

            steps.append(record)

            # 守卫早停(hard 模式):STOP 指令后模型仍在检索 → 直接终止
            guard = self.actions.get("search")
            if getattr(guard, "should_stop", False):
                output.finish_reason = "guard_early_stop"
                output.answer = self._parse_final_answer(steps[-1].thought) or ""
                break

        if not output.finished and not output.finish_reason:
            output.finish_reason = "max_steps"
            # 兜底,如果最后一步有 final answer 尝试提取
            final_answer = self._parse_final_answer(steps[-1].thought) if steps else None
            output.answer = final_answer or ""

        output.steps = steps
        output.latency_seconds = time.time() - t0
        output.metadata["paradigm"] = self.name
        return output

    @staticmethod
    def _build_prompt(question: str, steps: List[StepRecord]) -> str:
        parts = [f"Question: {question}"]
        for s in steps:
            parts.append(f"Intermediate Question: {s.action_input}")
            parts.append(f"Observation: {s.observation[:1500]}")
        if not steps:
            # 首轮:只要中间问题,防止模型零检索直接交卷
            parts.append("Ask the first intermediate question (do NOT give the Final Answer yet):")
        else:
            parts.append("Output the next step (another Intermediate Question, or Final Answer):")
        return "\n".join(parts)

    @staticmethod
    def _parse_intermediate_question(text: str) -> str:
        m = re.search(r"Intermediate Question:\s*(.+)", text)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        """取最后一个 Final Answer 匹配并清洗(见 base.extract_final_answer)。"""
        return extract_final_answer(text)


__all__ = ["SelfAskAgent"]