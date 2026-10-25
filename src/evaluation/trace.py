"""Agent 推理轨迹的持久化与回放。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..agents.base import AgentOutput


def save_trace(output: AgentOutput, path: str | Path) -> None:
    """把单次 Agent 输出保存为 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output.to_dict(), f, ensure_ascii=False, indent=2)


def save_traces(outputs: Iterable[AgentOutput], dir_path: str | Path, start_idx: int = 0) -> None:
    """批量保存一组 Agent 输出(文件序号从 start_idx 起,避免分批互相覆盖)。"""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for idx, output in enumerate(outputs):
        save_trace(output, dir_path / f"trace_{start_idx + idx:04d}.json")


def load_trace(path: str | Path) -> Dict[str, Any]:
    """从 JSON 文件加载轨迹。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_traces(dir_path: str | Path) -> List[Dict[str, Any]]:
    """从目录加载所有轨迹文件,按文件名排序。"""
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("trace_*.json"))
    return [load_trace(f) for f in files]


def filter_finished(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只保留成功 finish 的轨迹(便于分析正常完成路径)。"""
    return [t for t in traces if t.get("finished")]