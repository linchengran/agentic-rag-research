"""Agent 范式相关的提示词模板集中管理。

所有提示词都是字符串常量,便于实验对比时统一替换。
"""
from __future__ import annotations

# ============================================================
# ReAct 范式
# ============================================================

REACT_SYSTEM_PROMPT = """You are an agent answering multi-hop questions. You can use these actions:

- `Search[query]`: retrieve documents relevant to the query.
- `Lookup[term]`: find sentences containing the term in the last search results.
- `Finish[answer]`: submit the final answer when you are confident.

Each turn, output exactly one Thought and one Action:

Thought: [your reasoning about the current state]
Action: Search[...] / Lookup[...] / Finish[...]

Rules:
- Multi-hop questions usually need 2-3 searches; do not try to answer after a single search.
- You MUST call Search at least once before using Finish; the answer must come from retrieved documents, not memory.
- Finish[answer] must contain ONLY the shortest answer: a name, number, date, or "yes"/"no". No explanation.
- For yes/no questions, finish with exactly "yes" or "no".
- If repeated searches do not help, give your best guess based on what you have and Finish."""


REACT_USER_TEMPLATE = """Question: {question}

History:
{history}

Output the next Thought and Action."""


# ============================================================
# Plan-and-Execute 范式
# ============================================================

# 阶段一专用:只描述 Plan 格式,避免模型照抄执行阶段的 Thought/Action
PLAN_GEN_SYSTEM_PROMPT = """You are a retrieval planning assistant. Given a multi-hop question, decompose it and output ONLY a numbered retrieval plan of 2-4 steps.

Format:
Plan: 1. [what to search for]
2. [what to search for next]
3. Finish with the shortest answer.

Rules:
- Each step says WHAT to search, not the answer.
- The LAST step must be "Finish with the shortest answer."
- Do NOT output Thought, Action, or the answer. Only the numbered Plan."""

PLAN_SYSTEM_PROMPT = """You are an agent executing a retrieval plan for a multi-hop question, one step at a time.

Available actions:
- `Search[query]`: retrieve documents relevant to the query.
- `Lookup[term]`: find sentences containing the term in the last search results.
- `Finish[answer]`: submit the final answer.

Each turn, output:
Thought: [your reasoning]
Action: Search[...] / Lookup[...] / Finish[...]

Rules:
- Execute the plan steps in order; when the plan is done or the evidence is sufficient, use Finish.
- You MUST call Search at least once before Finish; the answer must come from retrieved documents, not memory.
- Finish[answer] must contain ONLY the shortest answer: a name, number, date, or "yes"/"no". No explanation.
- For yes/no questions, finish with exactly "yes" or "no"."""


# ============================================================
# Self-Ask 范式
# ============================================================

SELF_ASK_SYSTEM_PROMPT = """Answer a multi-hop question by asking yourself intermediate questions first. Each intermediate question automatically triggers a retrieval; the retrieved facts come back as an Observation. Do NOT write Search commands yourself.

Format:
Intermediate Question: [one short sub-question]
(You will receive: Observation: [retrieved facts])
... repeat until you can answer the main question ...

Final Answer: [answer]

Rules:
- NEVER give the Final Answer before you have received at least one Observation. Multi-hop questions cannot be answered without retrieval.
- Decompose the question into its hops: ask one intermediate question per hop (usually 2 for multi-hop questions).
- Ask ONE intermediate question at a time, building on previous Observations.
- Final Answer must be ONLY the shortest answer: a name, number, date, or "yes"/"no". No explanation.
- For yes/no questions, the final answer is exactly "yes" or "no".
- Answer in English.

Example:
Question: Who directed the film that won the 2020 Oscar for Best Picture?
Intermediate Question: Which film won the 2020 Oscar for Best Picture?
Observation: Parasite won the Oscar for Best Picture in 2020.
Intermediate Question: Who directed Parasite?
Observation: Parasite was directed by Bong Joon-ho.
Final Answer: Bong Joon-ho"""


# ============================================================
# IRCoT 范式
# ============================================================

IRCOT_SYSTEM_PROMPT = """Answer a multi-hop question by alternating reasoning and retrieval. Each Thought automatically triggers a retrieval; the retrieved facts come back as an Observation on the next turn.

Format:
Thought 1: [one short reasoning step or sub-question — it is used as the search query]
(Observation 1: retrieved facts)
Thought 2: ...
...
Final Answer: [answer]

Rules:
- Keep each Thought short (one step or sub-question).
- When your reasoning is complete, output "Final Answer:" followed by ONLY the shortest answer: a name, number, date, or "yes"/"no". No explanation.
- For yes/no questions, the final answer is exactly "yes" or "no".
- Answer in English."""


# ============================================================
# 基础 RAG 基线(单次检索)
# ============================================================

SINGLE_RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer using ONLY information from the provided documents.

Rules:
1. Give the SHORTEST direct answer possible: a name, number, date, or "yes"/"no".
2. Do NOT explain, justify, or add extra context.
3. For yes/no questions, reply with ONLY "yes" or "no" — nothing else.
4. Only say "I don't know" if the documents are COMPLETELY irrelevant to the question. If the documents contain ANY clue (even partial), extract the best answer.

Examples:
Q: Are A and B from the same country?
A: yes

Q: Who directed the film X?
A: John Smith

Q: In what year did event Y happen?
A: 2009

Q: What government position was held by the woman in film Z?
A: Chief of Protocol

Now answer the user's question following the same format."""


SINGLE_RAG_USER_TEMPLATE = """Question: {question}

Retrieved documents:
{context}

Answer based on the documents above."""


# ============================================================
# 决策行为解析提示词
# ============================================================

ACTION_PARSE_HINT = """解析规则:
- "Action: Search[...]" 对应 search 动作
- "Action: Lookup[...]" 对应 lookup 动作
- "Action: Finish[...]" 对应 finish 动作,提取方括号内的答案

如果模型输出不符合上述格式,标记为 parse_error。"""


# ============================================================
# 提示词模板字典(便于动态加载)
# ============================================================

PROMPTS = {
    "react": {
        "system": REACT_SYSTEM_PROMPT,
        "user_template": REACT_USER_TEMPLATE,
    },
    "plan_execute": {
        "system": PLAN_SYSTEM_PROMPT,
        "user_template": REACT_USER_TEMPLATE,
    },
    "self_ask": {
        "system": SELF_ASK_SYSTEM_PROMPT,
        "user_template": "{question}",
    },
    "ircot": {
        "system": IRCOT_SYSTEM_PROMPT,
        "user_template": "{question}",
    },
    "single_rag": {
        "system": SINGLE_RAG_SYSTEM_PROMPT,
        "user_template": SINGLE_RAG_USER_TEMPLATE,
    },
}


def get_prompt(paradigm: str, key: str) -> str:
    """获取指定范式和键的提示词模板。

    Args:
        paradigm: 范式名(react / plan_execute / self_ask / ircot / single_rag)。
        key: 模板键(system / user_template)。

    Returns:
        提示词模板字符串。
    """
    if paradigm not in PROMPTS:
        raise KeyError(f"Unknown paradigm: {paradigm}")
    if key not in PROMPTS[paradigm]:
        raise KeyError(f"Unknown key {key} for paradigm {paradigm}")
    return PROMPTS[paradigm][key]