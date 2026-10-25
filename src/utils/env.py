"""Hugging Face 离线模式集中配置。

与 rag-research-project 的 env.py 风格保持一致。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# 项目根目录的 .env 在 import 时自动加载(需要 python-dotenv)
# 任何脚本只要 from src.utils.env import ... 就会触发
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    # python-dotenv 未安装时静默跳过,不影响其他功能
    pass

# 强制覆盖 huggingface_hub 的 endpoint 常量
# 原因:HF Hub 库在某些代码路径用 HUGGINGFACE_CO_URL_TEMPLATE 拼 URL,
# 不读 HF_ENDPOINT。所有相关常量都改。
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "").rstrip("/")
if _HF_ENDPOINT:
    try:
        import huggingface_hub.constants as _hub_constants

        _hub_constants.HF_ENDPOINT = _HF_ENDPOINT
        _hub_constants.ENDPOINT = _HF_ENDPOINT
        _hub_constants._HF_DEFAULT_ENDPOINT = _HF_ENDPOINT
        # URL 模板里直接写了 huggingface.co,也得换
        _hub_constants.HUGGINGFACE_CO_URL_HOME = _HF_ENDPOINT + "/"
        _hub_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            _HF_ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"
        )
    except ImportError:
        pass


def setup_offline_mode(offline: bool = True) -> None:
    """启用或关闭 HuggingFace 离线模式。

    离线模式下,所有模型加载必须使用本地缓存。
    """
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"


def is_offline() -> bool:
    """当前是否处于离线模式。"""
    return os.environ.get("HF_HUB_OFFLINE") == "1"


def get_api_key(env_var: str = "LLM_API_KEY") -> Optional[str]:
    """从环境变量读取 API Key,未设置返回 None。"""
    return os.environ.get(env_var)


def get_endpoint() -> Optional[str]:
    """当前 HF endpoint(国内镜像用)。"""
    return os.environ.get("HF_ENDPOINT")