"""配置加载工具。

与 rag-research-project 的 config.py 风格保持一致。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """加载 YAML 配置。

    Args:
        config_path: 配置文件路径,为 None 时使用默认路径。

    Returns:
        解析后的配置字典。
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """获取配置中的指定小节,缺省返回空字典。"""
    return config.get(section, {}) or {}


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个配置字典,override 优先。"""
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result