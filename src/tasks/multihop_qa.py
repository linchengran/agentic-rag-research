"""多跳问答任务定义与数据 schema。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MultiHopQuestion:
    """单条多跳问答样本。"""

    id: str
    question: str
    answer: str
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    context: List[Dict[str, Any]] = field(default_factory=list)
    type: str = ""
    difficulty: str = ""  # easy / medium / hard
    level: str = ""       # 2-hop / 3-hop ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiHopQuestion":
        return cls(
            id=data.get("id") or data.get("_id", ""),
            question=data["question"],
            answer=data.get("answer", ""),
            supporting_facts=data.get("supporting_facts", []),
            context=data.get("context", []),
            type=data.get("type", ""),
            difficulty=data.get("difficulty", ""),
            level=data.get("level", ""),
        )

    def gold_titles(self) -> set[str]:
        """支撑事实的标题集合(用于评估支撑事实匹配)。"""
        return {sf.get("title", "") for sf in self.supporting_facts}

    def gold_paragraphs(self) -> List[str]:
        """支撑事实对应的段落文本。"""
        titles = self.gold_titles()
        return [c[1] for c in self.context if c[0] in titles]


@dataclass
class CorpusDocument:
    """语料库中的单篇文档(用于检索)。"""

    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_hotpotqa(
    json_path: str | Path,
    limit: Optional[int] = None,
) -> List[MultiHopQuestion]:
    """从 JSON 文件加载 HotpotQA 数据。

    Args:
        json_path: HotpotQA 原始 JSON 文件路径。
        limit: 限制加载样本数,None 表示全量。

    Returns:
        MultiHopQuestion 列表。
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else [raw]
    questions = [MultiHopQuestion.from_dict(item) for item in items]
    if limit is not None:
        questions = questions[:limit]
    return questions


def extract_corpus(
    questions: List[MultiHopQuestion],
) -> List[CorpusDocument]:
    """从多跳问答样本中提取完整语料(去重)。

    Returns:
        CorpusDocument 列表,每个段落作为一个独立文档。
    """
    seen: set[tuple[str, int]] = set()
    docs: List[CorpusDocument] = []
    for q in questions:
        for title, sentences in q.context:
            for idx, sent in enumerate(sentences):
                key = (title, idx)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(
                    CorpusDocument(
                        doc_id=f"{title}#{idx}",
                        title=title,
                        content=sent,
                        metadata={"title": title, "sent_idx": idx},
                    )
                )
    return docs