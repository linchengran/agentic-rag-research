# Agentic RAG 研究项目

面向多跳问答的迭代式检索决策机制研究。本项目在已有 `rag-research-project`(基础 RAG)之上,新增 **Agent 决策层**,研究 LLM Agent 何时检索、检索什么、检索几次,以最少的检索步数获得最高的答案准确率。

## 当前实现概览

- 复用已有项目的 FAISS 检索 + BGE Reranker 作为基础检索器。
- 实现 4 种 Agent 范式:ReAct、Plan-and-Execute、Self-Ask、IRCoT。
- 完整的 Agent 推理轨迹记录(Thought / Action / Observation)。
- 决策行为分析:检索决策正确率、检索步数-准确率 Pareto、失败模式分类。
- 基础 RAG 单次检索作为对照基线。
- 多跳问答评估数据集(HotpotQA / 2WikiMultihopQA / MuSiQue)。

## 默认配置

- Agent 模型:Qwen2.5-3B-Instruct(本地,4bit 量化)
- Embedding:BAAI/bge-small-zh-v1.5(沿用已有项目)
- 检索后端:FAISS IndexFlatIP
- 重排序:BAAI/bge-reranker-base
- 最大检索步数:8
- 评估:HotpotQA dev 集,500 样本

## 目录结构

```text
agent-research-project/
├── config/
│   └── config.yaml                 # 全局配置
├── data/
│   ├── raw/                        # HotpotQA / 2WikiMultihopQA / MuSiQue
│   └── processed/                  # 预处理后的样本 + 索引
├── src/
│   ├── agents/                     # Agent 决策层
│   │   ├── base.py                 # Agent 基类
│   │   ├── react.py                # ReAct 范式
│   │   ├── plan_execute.py         # Plan-and-Execute
│   │   ├── self_ask.py             # Self-Ask
│   │   └── ircot.py                # IRCoT
│   ├── retrievers/                 # 基础检索器(可复用已有项目)
│   │   ├── base.py
│   │   ├── faiss_retriever.py
│   │   └── rerank_retriever.py
│   ├── actions/                    # Agent 可执行的动作
│   │   ├── base.py
│   │   ├── search.py
│   │   ├── lookup.py
│   │   └── finish.py
│   ├── llm/                        # 模型加载与提示词
│   │   ├── local_model.py
│   │   └── prompts.py
│   ├── tasks/                      # 任务定义
│   │   ├── multihop_qa.py
│   │   └── data_loader.py
│   ├── evaluation/                 # 评估模块
│   │   ├── answer_metrics.py
│   │   ├── decision_metrics.py
│   │   ├── trace.py
│   │   └── visualization.py
│   └── utils/
│       ├── config.py
│       ├── env.py
│       └── logging.py
├── scripts/
│   ├── 00_eval_retrieval.py        # 检索质量评估(无 LLM)
│   ├── 01_build_index.py           # 构建多跳问答语料索引
│   ├── 02_run_baseline_rag.py      # 基础 RAG 基线(单次检索)
│   ├── 03_run_agentic_rag.py       # 运行 Agentic RAG(--guard off/nudge/hard)
│   ├── 04_run_matrix.py            # Agent 范式 × 模型对比
│   ├── 05_analyze_decisions.py     # 决策行为分析(计数型指标)
│   ├── 06_plot_results.py          # 结果可视化(基础版)
│   ├── 07_gold_doc_hit.py          # 金标文档命中直算
│   ├── 08_replay_gold_hit.py       # 金标命中重放(确定性,不跑 LLM)
│   ├── 09_check_relaxed_em.py      # EM 归一化放宽检查
│   ├── 10_stepwise_coverage.py     # per-step 边际命中曲线
│   ├── 11_analyze_guard.py         # 守卫干预对比分析
│   ├── 12_merge_matrix.py          # 正式矩阵分批结果合并(n=100)
│   ├── 13_make_paper_figures.py    # 论文图表(fig1-fig4,出版级)
│   ├── 14_prepare_dataset.py       # 泛化数据集准备(2Wiki / MuSiQue)
│   ├── 15_export_paper_tables.py   # 论文表格导出(MD + LaTeX)
│   ├── run_formal_matrix.sh        # 正式矩阵 + guard 臂驱动(幂等)
│   └── run_remaining_experiments.sh# 收尾实验驱动(消融/扩样/泛化,幂等)
├── tests/
├── experiments/
├── requirements.txt
└── README.md
```

## 与已有 RAG 项目的边界

本项目与 `rag-research-project` 是**独立但互补**的关系:

| 模块 | 已有 RAG 项目 | 新 Agentic RAG 项目 |
|------|--------------|---------------------|
| Embedding | ✅ 实现 | **复用**(import) |
| FAISS 检索 | ✅ 实现 | **复用**(import) |
| BGE Reranker | ✅ 实现 | **复用**(import) |
| 配置 / 日志 / 离线 | ✅ 实现 | **复用**(import) |
| Agent 决策层 | ❌ | ✅ 新增 |
| 决策行为评估 | ❌ | ✅ 新增 |
| 多跳问答任务 | ❌ | ✅ 新增 |

工程集成方案(2026-10-02 评估后更新):经实测,原 `pip install -e` 方案不可行(对方项目无打包入口),且两项目顶层包名均为 `src` 会命名空间冲突,检索语义亦有分化(查询前缀 / MMR)。**决策:两项目方法同源、实现独立**,本项目检索后端冻结以保证已完成实验的可复现性。评估详情见 `docs/工程决策-代码复用评估.md`。

## 环境安装

```bash
python -m venv venv
# Windows CMD
venv\Scripts\activate.bat
# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

如需复用已有项目代码:
```bash
pip install -e ../rag-research-project
```

## 运行方式

构建多跳问答语料的向量索引:
```bash
python scripts/01_build_index.py
```

运行基础 RAG 基线(单次检索):
```bash
python scripts/02_run_baseline_rag.py
```

运行 Agentic RAG(默认 ReAct):
```bash
python scripts/03_run_agentic_rag.py
python scripts/03_run_agentic_rag.py --agent ircot    # 切换为 IRCoT
```

运行 Agent 范式 × 模型对比矩阵:
```bash
python scripts/04_run_matrix.py
```

分析决策行为:
```bash
python scripts/05_analyze_decisions.py
```

可视化结果:
```bash
python scripts/06_plot_results.py
```

生成论文图表(fig1-fig4,PDF/SVG/PNG → `experiments/figures/`):
```bash
python scripts/13_make_paper_figures.py
```

导出论文表格(MD + LaTeX → `experiments/tables/`):
```bash
python scripts/15_export_paper_tables.py
```

准备泛化数据集(2WikiMultihopQA / MuSiQue,首次需联网走 hf-mirror):
```bash
python scripts/14_prepare_dataset.py --dataset all
```

泛化 / 扩样 / 消融实验可幂等重跑(分批落盘、自动跳过已完成批):
```bash
bash scripts/run_remaining_experiments.sh
```

正式矩阵 + guard 臂(已全部完成,可重跑校验):
```bash
bash scripts/run_formal_matrix.sh
```

## 评估指标

- **答案指标**:EM、Answer F1、Support F1
- **决策指标**:检索决策正确率、过度检索率、检索不足率
- **效率指标**:平均检索步数、token 消耗、延迟

详见 `docs/选题方案与开题报告思路.md` 第五节。

## 已知限制

- 默认本地生成依赖 CUDA 和 4bit 量化,CPU 环境不能直接照搬。
- 多跳问答数据集下载和处理需要网络。
- Agent 决策质量评估协议仍处于研究阶段,可能需要根据实验反馈调整。
- 与已有 RAG 项目的代码复用:经评估**决策不合并**(原方案不可行 + 实验冻结需要),见 `docs/工程决策-代码复用评估.md`。

## 研究用途

本项目用于研究:

- 不同 Agent 范式在多跳问答上的决策行为差异。
- 检索步数与答案准确率之间的 Pareto 关系。
- Agent 检索决策的失败模式与归因。
- Agentic RAG 相比基础 RAG 在多跳问答任务上的提升与成本。

详细研究方案见 `docs/选题方案与开题报告思路.md`。
## License

[MIT](LICENSE)
