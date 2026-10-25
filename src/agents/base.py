"""Agent 基类与推理轨迹数据结构。

所有具体范式(ReAct / Plan-Execute / Self-Ask / IRCoT)继承 BaseAgent,
实现 run(question) -> AgentOutput 接口。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.local_model import LocalModel


# Final Answer 之后可能跟着的控制标签(模型偶发把多轮格式塞进同一行;冒号可缺)
_ANSWER_TAG_RE = re.compile(
    r"\s*(?:Intermediate Question|Thought\s*\d*|Observation\s*\d*|Final Answer)\s*[:：]?",
    re.IGNORECASE,
)


def clean_final_answer(text: str) -> str:
    """从 Final Answer 之后的原始文本中清洗出短答案。

    处理模型把 "Final Answer: xxx Intermediate Question: ..." 等杂物
    塞进同一行的情况:截断到第一个控制标签、只取首行。
    """
    if not text:
        return ""
    answer = _ANSWER_TAG_RE.split(text.strip(), maxsplit=1)[0]
    if not answer:
        return ""
    return answer.splitlines()[0].strip()


def extract_final_answer(text: str) -> Optional[str]:
    """取最后一个 "Final Answer:" 匹配并清洗(模型可能多轮输出/带前缀垃圾)。"""
    matches = list(re.finditer(r"Final Answer\s*[:：]\s*(.+)", text))
    if not matches:
        return None
    return clean_final_answer(matches[-1].group(1))


def fallback_answer(steps: List["StepRecord"]) -> str:
    """max_steps 截断时的兜底答案:取最后一个 Finish 尝试的答案。

    不能直接用 steps[-1].action_input——最后一步若是 Search,
    那是检索 query 而非答案,会让 F1 虚高且 EM 永远错。
    没有任何 Finish 尝试时返回空串。
    """
    for s in reversed(steps):
        if s.action_name == "finish" or s.error == "finish_without_search":
            return s.action_input
    return ""


@dataclass
class StepRecord:
    """单步推理记录。"""

    step: int
    thought: str = ""
    action_name: str = ""
    action_input: str = ""
    observation: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Agent 单次执行的完整输出。"""

    question: str
    answer: str = ""
    steps: List[StepRecord] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""  # "finish_action" / "max_steps" / "parse_error" / "no_answer"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "n_steps": len(self.steps),
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "action_name": s.action_name,
                    "action_input": s.action_input,
                    "observation": s.observation[:1500] + ("..." if len(s.observation) > 1500 else ""),
                    "error": s.error,
                }
                for s in self.steps
            ],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Agent 抽象基类。"""

    name: str = "base"

    def __init__(
        self,
        llm: LocalModel,
        actions: Dict[str, Any],
        max_steps: int = 8,
        save_trace: bool = True,
    ) -> None:
        self.llm = llm
        self.actions = actions
        self.max_steps = max_steps
        self.save_trace = save_trace

    @abstractmethod
    def run(self, question: str, **kwargs: Any) -> AgentOutput:
        """对单个问题运行 Agent,返回 AgentOutput。

        实现要点:
        1. 构造 system / user 提示词
        2. 在最多 max_steps 步内迭代:
           - 调用 LLM 生成 Thought + Action
           - 解析 Action,执行对应的 BaseAction
           - 记录 StepRecord
           - 若 Action 是 finish 或达到 max_steps,终止
        3. 组装 AgentOutput 返回
        """

    @staticmethod
    def parse_action(text: str) -> tuple[str, str]:
        """从 LLM 输出中解析动作名和参数。

        约定格式:Action: Name[arg]
        Finish 额外容忍冒号写法(Action: Finish: answer)——3B 模型偶发。

        Args:
            text: LLM 的完整输出文本。

        Returns:
            (action_name, action_input) 元组。
            解析失败时返回 ("parse_error", text)。
        """
        import re

        pattern = re.compile(
            r"Action:\s*(Search|Lookup|Finish)\s*\[([^\]]*)\]",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            # Finish 冒号兜底:括号格式缺失但语义明确时不要浪费一轮
            colon = re.search(r"Action:\s*Finish\s*[:：]\s*(.+)", text, re.IGNORECASE)
            if colon:
                return "finish", clean_final_answer(colon.group(1))
            return "parse_error", text.strip()
        action_input = match.group(2).strip()
        if match.group(1).lower() == "finish":
            action_input = clean_final_answer(action_input)
        return match.group(1).lower(), action_input