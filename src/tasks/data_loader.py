"""通用数据加载与切分工具。

沿用 rag-research-project 中 crc32 + seed 的可复现切分方案。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def stable_hash(text: str, seed: int = 42) -> int:
    """基于文本内容 + seed 的稳定哈希值。"""
    payload = f"{seed}::{text}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def split_by_hash(
    items: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    id_field: str = "id",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """基于稳定哈希的可复现切分。

    Args:
        items: 待切分的样本列表。
        ratios: (train, dev, test) 比例。
        seed: 随机种子。
        id_field: 样本 ID 字段名。

    Returns:
        (train, dev, test) 三个列表。
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    train, dev, test = [], [], []
    cum = 0.0
    thresholds = []
    for r in ratios:
        cum += r
        thresholds.append(cum)

    for item in items:
        sample_id = str(item.get(id_field, ""))
        h = stable_hash(sample_id, seed=seed)
        normalized = (h % 10_000) / 10_000  # [0, 1)
        if normalized < thresholds[0]:
            train.append(item)
        elif normalized < thresholds[1]:
            dev.append(item)
        else:
            test.append(item)

    return train, dev, test


def save_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """把样本列表写成 JSONL。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载样本列表。"""
    path = Path(path)
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def shuffle_with_seed(items: List[Any], seed: int = 42) -> List[Any]:
    """带种子的乱序。"""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items