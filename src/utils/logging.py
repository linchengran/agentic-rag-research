"""统一日志工具。

与 rag-research-project 的 logging.py 风格保持一致。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """获取一个配置好的 logger。

    Args:
        name: logger 名称(通常用 __name__)。
        level: 日志级别。
        log_file: 可选日志文件路径。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level.upper())
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger