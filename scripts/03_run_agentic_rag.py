"""运行 Agentic RAG 主入口。

支持切换 Agent 范式:--agent react / plan_execute / self_ask / ircot
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.actions.finish import FinishAction  # noqa: E402
from src.actions.guarded_search import GuardedSearchAction  # noqa: E402
from src.actions.lookup import LookupAction  # noqa: E402
from src.actions.search import SearchAction  # noqa: E402
from src.agents.ircot import IRCoTAgent  # noqa: E402
from src.agents.plan_execute import PlanExecuteAgent  # noqa: E402
from src.agents.react import ReActAgent  # noqa: E402
from src.agents.self_ask import SelfAskAgent  # noqa: E402
from src.evaluation.answer_metrics import evaluate_answer  # noqa: E402
from src.evaluation.decision_metrics import compute_decision_eval  # noqa: E402
from src.evaluation.trace import save_traces  # noqa: E402
from src.llm.local_model import LocalModel  # noqa: E402
from src.retrievers.rerank_retriever import RerankRetriever  # noqa: E402
from src.tasks.multihop_qa import load_hotpotqa  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.env import setup_offline_mode  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


AGENT_REGISTRY = {
    "react": ReActAgent,
    "plan_execute": PlanExecuteAgent,
    "self_ask": SelfAskAgent,
    "ircot": IRCoTAgent,
}


def build_actions(
    retriever: RerankRetriever,
    guard_mode: str = "off",
    waste_limit: int = 2,
    finish_hint: str = "Action: Finish[shortest answer]",
) -> dict:
    """构造 Search / Lookup / Finish 动作字典。

    guard_mode != off 时,Search 换成带守卫版本(重复 query 检测器 +
    零收益早停),hard 模式 agent 循环读到 should_stop 后强制终止。
    """
    last_obs = deque(maxlen=3)

    if guard_mode == "off":
        search = SearchAction(retriever)
    else:
        search = GuardedSearchAction(
            retriever,
            waste_limit=waste_limit,
            hard_stop=(guard_mode == "hard"),
            finish_hint=finish_hint,
        )
    lookup = LookupAction(last_observations=last_obs)

    # Search 后把 observation 推入 deque
    orig_call = search.__call__

    def wrapped_search(query: str, **kwargs):
        result = orig_call(query, **kwargs)
        if result.success:
            last_obs.append(result.observation)
        return result

    search.__call__ = wrapped_search  # 简单 monkey patch

    return {
        "search": search,
        "lookup": lookup,
        "finish": FinishAction(),
    }


# 各范式交卷的输出格式不同,守卫 STOP 指令必须用对应格式才有效
FINISH_HINTS = {
    "react": "Action: Finish[shortest answer]",
    "plan_execute": "Action: Finish[shortest answer]",
    "self_ask": "Final Answer: <shortest answer>",
    "ircot": "Final Answer: <shortest answer>",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Agentic RAG")
    parser.add_argument("--data", required=True, help="HotpotQA JSON 路径")
    parser.add_argument("--index", default="data/processed/index/hotpot_faiss")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--agent", default="react", choices=list(AGENT_REGISTRY.keys()))
    parser.add_argument("--out", default=None, help="结果 JSON 路径")
    parser.add_argument("--trace-dir", default=None, help="推理轨迹保存目录")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 题(分批跑不重叠)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--guard", default="off", choices=["off", "nudge", "hard"],
        help="检索守卫:off=无(控制组);nudge=只注入纠错/STOP 指令;"
             "hard=STOP 后再检索则强制终止(intervention 实验组)",
    )
    parser.add_argument(
        "--guard-waste-limit", type=int, default=2,
        help="连续浪费性检索(重复 query / 零新增文档)多少次触发早停",
    )
    args = parser.parse_args()

    logger = get_logger("agentic_rag", level="INFO")
    setup_offline_mode(offline=True)
    config = load_config(args.config)

    guard_suffix = "" if args.guard == "off" else f"_guard-{args.guard}"
    out_path = Path(args.out or f"experiments/results/agentic_rag_{args.agent}{guard_suffix}.json")
    trace_dir = Path(args.trace_dir or f"experiments/traces/{args.agent}{guard_suffix}")

    total_limit = None if args.limit is None else args.offset + args.limit
    questions = load_hotpotqa(args.data, limit=total_limit)[args.offset:]
    logger.info("加载 %d 个问题(offset=%d)", len(questions), args.offset)

    retriever = RerankRetriever(
        embedding_model=config.get("embedding", {}).get("model_name", "BAAI/bge-small-zh-v1.5"),
        reranker_model=config.get("reranker", {}).get("model_name", "BAAI/bge-reranker-base"),
        final_k=config.get("retriever", {}).get("top_k", 5),
    )
    retriever.load(args.index)

    llm = LocalModel(
        model_name=config["llm"]["model_name"],
        quantization=config["llm"].get("quantization"),
    )

    actions = build_actions(
        retriever,
        guard_mode=args.guard,
        waste_limit=args.guard_waste_limit,
        finish_hint=FINISH_HINTS[args.agent],
    )
    search_action = actions["search"]
    max_steps = args.max_steps or config.get("agent", {}).get("max_steps", 8)
    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls(llm=llm, actions=actions, max_steps=max_steps)

    outputs = []
    records = []
    for i, q in enumerate(questions):
        if hasattr(search_action, "reset"):
            search_action.reset()  # 守卫状态按题隔离
        out = agent.run(q.question)
        outputs.append(out)

        answer_metrics = evaluate_answer(out.answer, q.answer)
        decision_eval = compute_decision_eval(
            out,
            gold_supporting_titles=set(q.gold_titles()),
            answer_correct=bool(answer_metrics["em"]),
        )
        rec = {
            "id": q.id,
            "question": q.question,
            "gold_answer": q.answer,
            "pred_answer": out.answer,
            "n_steps": len(out.steps),
            "finish_reason": out.finish_reason,
            **answer_metrics,
            **decision_eval.to_dict(),
        }
        if hasattr(search_action, "events"):
            rec["guard_events"] = list(search_action.events)
        records.append(rec)
        if (i + 1) % 10 == 0:
            logger.info("处理 %d / %d", i + 1, len(questions))

    save_traces(outputs, trace_dir, start_idx=args.offset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("完成。结果写入 %s,轨迹写入 %s", out_path, trace_dir)


if __name__ == "__main__":
    main()