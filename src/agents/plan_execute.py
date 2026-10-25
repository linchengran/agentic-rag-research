"""Plan-and-Execute 范式实现。

两阶段:
- 阶段一:LLM 生成整体检索计划(步骤列表)
- 阶段二:逐步执行计划,每步决定是否检索或提交答案
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from ..actions.base import ActionResult
from ..llm.local_model import LocalModel
from ..llm.prompts import PLAN_GEN_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT, REACT_USER_TEMPLATE
from .base import AgentOutput, BaseAgent, StepRecord, fallback_answer


class PlanExecuteAgent(BaseAgent):
    """Plan-and-Execute 范式 Agent。"""

    name = "plan_execute"

    def run(self, question: str, **kwargs: Any) -> AgentOutput:
        output = AgentOutput(question=question)
        t0 = time.time()
        steps: List[StepRecord] = []

        # 阶段一:生成计划
        plan = self._generate_plan(question)
        if not plan:
            output.finish_reason = "plan_error"
            output.steps = steps
            output.latency_seconds = time.time() - t0
            return output

        # 阶段二:逐步执行
        for step_idx, planned_step in enumerate(plan[: self.max_steps]):
            step_prompt = (
                f"Question: {question}\n\n"
                f"Full plan:\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan)) +
                f"\n\nCurrent step {step_idx+1}: {planned_step}\n"
                f"Execution history:\n" + self._format_history(steps) +
                "\n\nOutput the next Thought and Action."
            )
            messages = [
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": step_prompt},
            ]
            llm_out = self.llm.chat(messages)
            thought, action_text = self._parse(llm_out)
            action_name, action_input = self.parse_action(action_text)
            record = StepRecord(
                step=step_idx + 1,
                thought=thought,
                action_name=action_name,
                action_input=action_input,
                metadata={"planned_step": planned_step},
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

            if action_name in self.actions:
                result: ActionResult = self.actions[action_name](action_input)
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

            if action_name == "finish" and result is not None and result.success:
                output.answer = action_input
                output.finished = True
                output.finish_reason = "finish_action"
                break

        if not output.finished and not output.finish_reason:
            # plan 耗尽仍未 Finish:给一轮收尾 turn,明确要求交卷
            wrap_prompt = (
                f"Question: {question}\n\n"
                f"Full plan:\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(plan)) +
                f"\nExecution history:\n" + self._format_history(steps) +
                "\n\nAll plan steps are done. Output your final answer now:\n"
                "Thought: [summarize the evidence]\n"
                "Action: Finish[shortest answer]"
            )
            llm_out = self.llm.chat([
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": wrap_prompt},
            ])
            _, action_text = self._parse(llm_out)
            action_name, action_input = self.parse_action(action_text)
            if action_name == "finish":
                output.answer = action_input
                output.finished = True
                output.finish_reason = "finish_action"
            if not output.finished:
                output.finish_reason = "max_steps"
                output.answer = fallback_answer(steps)

        output.steps = steps
        output.latency_seconds = time.time() - t0
        output.metadata["paradigm"] = self.name
        output.metadata["plan"] = plan
        return output

    def _generate_plan(self, question: str) -> List[str]:
        """阶段一:用 LLM 生成检索计划(专用 plan prompt,避免混入执行格式)。"""
        prompt = (
            f"Question: {question}\n\n"
            "Output the Plan only (numbered retrieval steps, no Thought/Action yet):"
        )
        messages = [
            {"role": "system", "content": PLAN_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        out = self.llm.chat(messages)
        return self._parse_plan(out)

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        """从 LLM 输出中提取编号计划。

        支持两种格式:
        - 多行: "1. xxx\\n2. xxx\\n3. xxx"
        - 单行: "Plan: 1. xxx 2. xxx 3. xxx"(Plan 标签后所有编号集中在一行)
        """
        # 提取 "Plan:" 之后的 plan 段(到 Thought/Action 之前)
        m = re.search(r"Plan:\s*(.+?)(?=\n\s*Thought|\n\s*Action|$)", text, re.DOTALL)
        plan_text = m.group(1) if m else text
        # 把单行多 plan 转多行:在每个 "数字.+空格" 前插入换行
        plan_text = re.sub(r"\s+(\d+)\.\s+", r"\n\1. ", plan_text)
        plan: List[str] = []
        for line in plan_text.splitlines():
            mm = re.match(r"\s*\d+\.\s*(.+)", line)
            if mm:
                plan.append(mm.group(1).strip())
        return plan

    @staticmethod
    def _parse(text: str) -> tuple[str, str]:
        thought = ""
        action = ""
        m = re.search(r"Thought:\s*(.*?)(?:\n|Action:)", text, re.DOTALL)
        if m:
            thought = m.group(1).strip()
        m = re.search(r"Action:\s*.*", text, re.DOTALL)
        if m:
            action = m.group(0).strip()
        if not action:
            action = text.strip()
        return thought, action

    @staticmethod
    def _format_history(steps: List[StepRecord]) -> str:
        if not steps:
            return "(none)"
        lines = []
        for s in steps:
            lines.append(
                f"Step {s.step}: Thought: {s.thought} | "
                f"Action: {s.action_name}[{s.action_input}] | "
                f"Observation: {s.observation[:1500]}"
            )
        return "\n".join(lines)


__all__ = ["PlanExecuteAgent"]