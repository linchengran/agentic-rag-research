"""答案评估指标:EM、Answer F1、Support F1。

EM/F1 沿用 SQuAD 评估惯例。
Support F1 沿用 HotpotQA 官方评估协议。
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List

import jieba

# 强制初始化 jieba(避免首次分词慢)
jieba.initialize()


def normalize_answer(s: str) -> str:
    """SQuAD 风格答案归一化:小写、去标点、去多余空白、去冠词。

    中文用 jieba 分词(英文按词切),保证中英混合输出能跟英文 gold 比较。
    """
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # jieba 切分(中英都支持)
    tokens = [tok.strip() for tok in jieba.cut(s) if tok.strip()]
    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def relax_answer(s: str) -> str:
    """放宽归一化,吸收 3B 输出的粒度/格式近失(v2 矩阵 failure analysis 校准)。

    在 normalize_answer 基础上追加:
    - 去尾部括号消歧后缀:"Compliance (film)" -> "Compliance","USS Essex (CV-9)" 同
    - 范围连字符展开:"1969-1974" -> "1969 1974"("from 1969 until 1974" 归一后同)
    - 日期统一:"september 8, 2017" -> "8 9 2017"(月份名转数字)
    - 句尾句号残留:"New York City." 与 "New York City" 同(normalize 已处理)
    """
    # 去尾部括号消歧后缀(近失样本里全是尾括号;中置括号可能承载实义,不动)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s.strip())
    s = normalize_answer(s)
    # 月份名转数字,统一日期表示
    months = {
        "january": "1", "february": "2", "march": "3", "april": "4",
        "may": "5", "june": "6", "july": "7", "august": "8",
        "september": "9", "october": "10", "november": "11", "december": "12",
    }
    s = " ".join(months.get(tok, tok) for tok in s.split())
    # 范围连字符(年份区间):"1969-1974" 归一化后是 "19691974",拆回两个年份
    s = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", s)
    return s


def relaxed_em(pred: str, gold: str) -> float:
    """放宽 EM:严格 EM 或(双向 containment)之一成立。

    containment 规则针对近失样本(v2/v_ms8 failure analysis):
    - 短的一方(>=2 词)完全包含在长的一方内,视为命中:
      'New York City' ⊆ 'Greenwich Village, New York City'(粒度近失)
      '1969 1974' ⊆ 'from 1969 until 1974'(范围近失)
    - 单词答案不做 containment(避免 'no' ⊆ 'north' 类假阳),仍要求严格相等
    yes/no 类近失(反向错答)不会被该规则吸收,保持判错——比较推理失败是独立失败模式。
    """
    p, g = relax_answer(pred), relax_answer(gold)
    if p == g:
        return 1.0
    p_toks, g_toks = p.split(), g.split()
    # 双向 containment,短方至少 2 词
    if len(p_toks) >= 2 and len(g_toks) >= 2:
        if set(p_toks) <= set(g_toks) or set(g_toks) <= set(p_toks):
            return 1.0
    return 0.0


def exact_match(pred: str, gold: str) -> float:
    """Exact Match:归一化后完全匹配。"""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """SQuAD F1:基于词级别的 precision / recall。"""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def support_em(pred_titles: Iterable[str], gold_titles: Iterable[str]) -> float:
    """支撑事实 EM:预测的支撑标题集合是否与 gold 完全一致。"""
    return float(set(pred_titles) == set(gold_titles))


def support_f1(pred_titles: Iterable[str], gold_titles: Iterable[str]) -> float:
    """支撑事实 F1。"""
    pred_set = set(pred_titles)
    gold_set = set(gold_titles)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(pred: str, gold: str) -> Dict[str, float]:
    """对单个答案返回所有答案指标。"""
    return {
        "em": exact_match(pred, gold),
        "relaxed_em": relaxed_em(pred, gold),
        "answer_f1": f1_score(pred, gold),
    }


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    """对一组样本的指标求平均。"""
    if not records:
        return {}
    keys = records[0].keys()
    return {key: sum(r.get(key, 0.0) for r in records) / len(records) for key in keys}