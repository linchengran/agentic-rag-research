"""Action 基类与注册表。

Action 是 Agent 可以执行的基本动作,例如 Search / Lookup / Finish。
所有具体动作继承 BaseAction 并实现 __call__。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ActionResult:
    """动作执行结果。"""

    name: str
    success: bool
    observation: str
    raw_input: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseAction(ABC):
    """Action 抽象基类。"""

    name: str = "base"

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ActionResult:
        """执行动作,返回 ActionResult。"""

    @property
    def description(self) -> str:
        """动作说明,用于提示词构造。"""
        return self.name


# ============================================================
# 全局动作注册表
# ============================================================

_REGISTRY: Dict[str, BaseAction] = {}


def register_action(action: BaseAction) -> None:
    """注册一个 Action 实例。"""
    _REGISTRY[action.name] = action


def get_action(name: str) -> BaseAction:
    """根据名称获取已注册的 Action。"""
    if name not in _REGISTRY:
        raise KeyError(f"Action not registered: {name}")
    return _REGISTRY[name]


def list_actions() -> List[str]:
    """列出所有已注册的 Action 名称。"""
    return list(_REGISTRY.keys())


def reset_registry() -> None:
    """清空注册表(主要用于测试)。"""
    _REGISTRY.clear()