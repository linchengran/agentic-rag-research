# W4-W5 前置 · Baseline v2 + 范式矩阵 Smoke 报告(2026-09-24)

## 一、TL;DR

- Prompt 修复让 baseline **EM 0.020 → 0.212(10.6×)/ F1 0.113 → 0.300(2.6×)**
- **4 范式 v2 矩阵终版(2026-09-25,各 n=30,英文 prompt + 全部修复):全部低于 baseline**——self_ask 16.7% > plan_execute 13.3% > ircot 10.0% > react 6.7%,baseline 21.2%
- self_ask smoke 的 50% 确认 lucky;n=4 时代的一切旧数字作废
- 三轮迭代修完 5 类工程缺陷(中文 prompt、Final Answer 解析、零检索交卷、query 当答案、plan 两阶段格式混杂)——**plan_execute 0%→13.3% 证明旧数字大部分是工程缺陷,不是范式能力**
- 共性失败模式:**yes/no 反向错答**(三范式合计 ~15 处)+ 答案粒度/格式近失——3B 的比较推理是跨范式瓶颈
- **正式矩阵终版(2026-10-02,n=100 × ms=8):self_ask EM 26.0% / F1 31.3% 追平 baseline 同题集(23.0% / 28.4%,McNemar p=0.664)**;ircot 显著低于 baseline(p=0.031),plan_execute 边缘(p=0.064),react 方向性低但不显著(p=0.18)——n=30 时代"全部低于 baseline"的结论被样本量部分修正,详见 §3.10
- **⚠️ 上行结论已被 §3.12 扩样修正:self_ask 在 n=500 上显著低于 baseline(EM 15.2% vs 21.2%,McNemar p=0.003)**——n=100 的"追平"是子集偏差;HotpotQA 的最终结论回到"4 范式全部不优于单次检索 baseline",范式间真实排位 self_ask 最好
- **guard-nudge 臂并入正式矩阵:react 真实检索 -19%、self_ask -22%,EM 全范式零损失**(答案变化 19 处、EM 变化 0 处)——§3.9 的 -33% 节省在 n=100 上方向复现,详见 §3.10;guard 的价值独立于 EM 结论方向
- **收尾实验(2026-10-02/03,§3.11-3.13)**:react 归因消融(截断修复 / max_steps 各贡献约 +3pp relaxed)· self_ask n=500 扩样 · 2WikiMultihopQA / MuSiQue 泛化(**范式排名数据集依赖**:2wiki 上 ircot 反超居首,musique 全体贴地板)——实验阶段全部完成
- 下一步见 §5(论文初稿)

## 二、Baseline v1 → v2(500 题,HotpotQA dev distractor 前 500)

| 指标 | v1 旧 prompt | v2 新 prompt | Δ |
|---|---|---|---|
| EM | 0.0200 (10/500) | **0.2120 (106/500)** | +19.2pp |
| F1 | 0.1134 | **0.2996** | +18.6pp |
| Dont-know 率 | 26.2% | 13.0% | -13.2pp |
| Yes/No 答对率 | 0/33 (0%) | 18/33 (54.5%) | +54.5pp |
| F1=0 零重叠率 | 43.6% | 61.6% | ⚠️ +18pp |

### v1 错例诊断(修复依据)

1. **Yes/No 全军覆没(0/33)**:模型不给 yes/no,只展开解释
2. **Dont-know 题 87.8% 的 gold ≤3 词**(如 "Sonic"/"1999"/"9,984"):prompt 鼓励"找不到就说不知道",但其实是简单短答
3. **F1=0 达 43.6%**:长解释跟 gold 无 token 重叠

### 修复内容(2026-09-24)

| 文件 | 变更 |
|---|---|
| `src/llm/prompts.py` | `SINGLE_RAG_SYSTEM_PROMPT` 重写:强制最短答案 + yes/no 显式规则 + 4 个 few-shot 示例 + 抑制 "I don't know" |
| `config/config.yaml` | `max_new_tokens: 512 → 64`(512 鼓励模型展开解释,答案对不上 gold) |
| `src/agents/plan_execute.py` | `_parse_plan()` 支持单行 `Plan: 1. xxx 2. xxx` 格式(旧版只认多行,曾致 29/30 plan_error) |

产物:
- `experiments/results/baseline_rag.json` — v1(旧 prompt,留档)
- `experiments/results/baseline_rag_v1_prompt.json` — v1 备份
- `experiments/results/baseline_rag_v2.json` — **v2(新 prompt,当前基准)**

## 三、范式矩阵 Smoke(max_steps=4)

| 范式 | n | EM | F1 | AvgSteps | Finish 分布 |
|---|---|---|---|---|---|
| baseline_v2(参考) | 500 | 0.2120 | 0.2996 | 1.0 | — |
| react | 30 | 0.1000 | 0.1714 | 3.07 | 25 finish / 5 max_steps |
| plan_execute | 8 | 0.0000 | 0.0333 | 3.62 | 8/8 max_steps 截断 |
| self_ask | 4 | **0.5000** | **0.5000** | 3.25 | 4/4 final_answer |
| ircot | 4 | 0.0000 | 0.0000 | 2.00 | 4/4 final_answer |

## 三·二、self_ask n=30 验证(2026-09-24,max_steps=4,题 0-29)

| 指标 | smoke n=4 | **n=30(验证)** | baseline_v2 参考 |
|---|---|---|---|
| EM | 0.500 | **0.167 (5/30)** | 0.212 |
| F1 | 0.500 | **0.221** | 0.300 |
| AvgSteps | 3.25 | 3.13 | — |
| Finish | 4/4 final_answer | 26 final_answer / 4 max_steps | — |

产物:`experiments/results/matrix/self_ask_n30.json`(含 smoke 4 题 + 4 个 batch);轨迹 `experiments/traces/self_ask/`(trace_0000-0029)

### 结论

1. **EM=50% 不成立**(n=4 → n=30 掉到 16.7%);self_ask 当前**低于 baseline**,不是 smoke 暗示的第一名
2. **但数字被两个 bug 压低,不可直接下"self_ask 无效"的结论**:
   - 🔴 **`SELF_ASK_SYSTEM_PROMPT` / `IRCOT_SYSTEM_PROMPT` 是中文**(`src/llm/prompts.py:62/78`)——W4 v1→v2 已证明中文 prompt 让 Qwen 答中文,EM 掉 10×。v2 修复只改了 `SINGLE_RAG_SYSTEM_PROMPT`,**三个 agent prompt 全部漏网**。7/30 的 pred 含中文,直接 0 分
   - 🔴 `_parse_final_answer`(`src/agents/self_ask.py:97`)取第一个 `Final Answer:` 匹配且不清洗,pred 里残留 `Final Answer: ... Intermediate Question: ...` 垃圾
3. 近失例(修 EM 归一化可救):`'1986-2013'` vs gold `'from 1986 to 2013'`;`'4,000 People (3,677 Seated)'` 包含 gold `'3,677 seated'`
4. 26/30 正常 final_answer 终止,4/30 max_steps 截断——self_ask 的流程本身跑得通

## 三·三、self_ask prompt 修复迭代 v2 / v3(2026-09-24,同 30 题,max_steps=4)

修复内容(对全部 4 范式生效):
- `src/llm/prompts.py`:REACT / PLAN / SELF_ASK / IRCOT 四个 system prompt + REACT_USER_TEMPLATE 全部改英文,统一加"Finish/Final Answer 只给最短答案(yes/no、名字、数字、日期)"规则,self_ask / ircot 加 few-shot 示例
- `src/agents/base.py`:新增 `extract_final_answer()`(取**最后一个** Final Answer 匹配)+ `clean_final_answer()`(截断同行残留的 Intermediate Question / Thought / Final Answer 标签)
- `src/agents/self_ask.py` / `ircot.py`:解析改用上述助手;用户侧 scaffold 改英文;ircot 删除"`Thought` 出现在 Final Answer 前则不终止"的反逻辑条件
- `src/agents/plan_execute.py` / `react.py`:所有中文 scaffold 改英文
- 单测 22/22 通过;`--offset` 参数加入 `03_run_agentic_rag.py`(分批跑不重叠)

| 版本 | 改动 | EM | F1 | AvgSteps | 中文 pred | 截断/格式问题 |
|---|---|---|---|---|---|---|
| v1(旧中文 prompt) | — | 0.167 | 0.221 | 3.13 | 7/30 | 4 截断 + 垃圾前缀 |
| v2(英文+解析修复) | prompt+parse | 0.133 | 0.192 | **1.23** | 0 | 无,但 24/30 零检索硬答 |
| **v3(v2+强制先检索)** | +防零检索 | **0.167** | 0.187 | 2.43 | 0 | 无 |

产物:`self_ask_n30_v2.json` / `self_ask_n30_v3.json`;轨迹 `experiments/traces/self_ask_v2` / `_v3`

### 关键发现

1. **格式问题彻底解决**:中文输出 0、30/30 干净 final_answer、无截断——prompt 修复达成工程目标
2. **但 EM 没有超过 baseline(21.2%)**:三轮迭代后 self_ask 稳定在 13-17%,**smoke 的 50% 确认纯属 lucky**(n=4 恰好抽中简单题)
3. **v2 的意外教训**:格式修干净后模型学会"直接交卷"——24/30 第一步就 Final Answer、零检索,step=1 硬答 EM 仅 8.3%;加了"必须先问中间问题"约束(v3)后 steps 回到 2.43。**教训:agent prompt 的 few-shot 示例结尾会被 3B 模型当成"抄答案模板",约束"何时允许 Finish"和"怎么 Finish"同样重要**
4. **v3 失败模式**(论文 failure analysis 素材):
   - yes/no 题大量反向错答(gold yes 答 no):5 处,3B 的比较推理不足
   - 数字/日期近失:`'3100'` vs `'3,677 seated'`、`'1991'` vs `'1999'`
   - 不确定时倾向答 "no" 或日期,而非放弃
5. **结论:在 Qwen2.5-3B + max_steps=4 条件下,self_ask(含修复后)不优于单次检索 baseline**。要么 max_steps=8 / 检索质量提升后有转机,要么这本身就是"3B 规模 agentic 开销不回本"的证据——正式矩阵(不同 max_steps/更大 n)会给答案

### 解读(谨慎,样本量不齐)

- **self_ask** 4/4 正常终止,EM=50%:显式子问题分解 + 自动检索的结构对 3B 模型最友好,**值得扩样本验证**
- **react** EM=10% < baseline 21%:5/30 被 max_steps 截断;可能 max_steps=4 太紧(config 默认 8),也可能 3B 的 ReAct 格式遵循不稳定
- **plan_execute** 8/8 max_steps 截断:plan 解析修了,但执行阶段 LLM 每步给不出有效 Finish
- **ircot** 4 题全错但全部正常终止:疑 CoT 检索融合 prompt 把答案带偏,需看 `experiments/traces/ircot`

## 四、已知问题 / 风险

1. **样本量不齐**(30/8/4/4):self_ask 50% 可能是 lucky,必须扩到 n=30
2. **max_steps=4 是临时选择**(迁就 Bash 10min timeout),与 config 默认 8 不一致——正式矩阵必须统一
3. **BG 任务被 Claude Code 内存压力 reaper 杀过 3 次**(matrix 4×50、self_ask 30、ircot 30 均中招)→ 长跑任务改 foreground + 小 limit(4-8 题)模式;正式矩阵跑前设 `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` 或分批
4. **F1=0 率上升**(43.6%→61.6%):新 prompt 让模型更敢猜,答非所问比例变大——论文 failure analysis 素材

## 三·四、4 范式 v2 矩阵终版(2026-09-25,同 30 题,max_steps=4,英文 prompt + 全部解析修复)

| 范式 | n | EM | F1 | AvgSteps | Finish 分布 | vs 旧 smoke |
|---|---|---|---|---|---|---|
| baseline_v2(参考,n=500) | 500 | **0.212** | **0.300** | 1.0 | — | — |
| self_ask v3 | 30 | 0.167 (5) | 0.187 | 2.43 | 30 final_answer | 50%(n=4)→lucky |
| plan_execute v2 | 30 | 0.133 (4) | 0.177 | 2.77 | 29 finish / 1 max_steps | 0% → 13.3% |
| ircot v2 | 30 | 0.100 (3) | 0.138 | **1.10** | 30 final_answer | 0% → 10% |
| react v2 | 30 | 0.067 (2) | 0.080 | 3.20 | 25 finish / 5 max_steps | 10% → 6.7% |

产物:`react_n30_v2.json` / `plan_execute_n30_v2.json` / `self_ask_n30_v3.json` / `ircot_n30_v2.json` + `summary.json`(终版);轨迹 `experiments/traces/*_v2`

### 本轮追加修复(2026-09-24/25 夜,单测 24/24 过)

| 修复 | 文件 | 动机 |
|---|---|---|
| `parse_action` 容忍 `Finish: answer` 冒号写法 | `base.py` | react 3B 偶发丢括号,浪费整轮 |
| 零检索 Finish/Final Answer 拒收(4 范式统一守卫) | `react/plan_execute/ircot` | self_ask v2 发现:格式修干净后 3B 学会零检索直接交卷(step=1 硬答 EM 8.3%) |
| max_steps 兜底答案改为"最后一个 Finish 尝试",否则留空 | `base.py fallback_answer` | 旧逻辑 `steps[-1].action_input` 把**检索 query 当答案**交(`pred='Ed Wood nationality'`),F1 虚高 |
| plan 阶段拆专用 `PLAN_GEN_SYSTEM_PROMPT` | `prompts.py` / `plan_execute.py` | 两阶段格式混在一个 system prompt 里,3B 生成计划时照抄 Thought/Action → plan_error |
| plan 生成要求最后一步 Finish + plan 耗尽补一轮"收尾 turn" | `prompts.py` / `plan_execute.py` | 旧 plan 全是检索步,执行完循环直接结束,模型从没被要求交卷(6/6 全空) |

### 解读

1. **全部 4 范式低于单次检索 baseline(21.2%)**——Qwen2.5-3B + max_steps=4 条件下,agentic 检索开销未转化为答案质量。排名:self_ask > plan_execute > ircot > react
2. **每范式的"学坏"方式不同**(论文 failure analysis 素材):
   - react:不确定时用 yes/no 兜底(gold 是实体时答 no/yes,约 10 处),5/30 截断
   - ircot:1 跳就交卷(27/30 steps=1,avg 1.10)——follow-up 收尾发生在检索后不违反守卫,但多跳问题只查 1 跳
   - plan_execute:修复后正常,但答案粒度粗(答 "New York" 而 gold 是 "Greenwich Village")
   - self_ask:最稳但 yes/no 反向错答多(gold yes 答 no,5 处)
3. **共性失败模式:yes/no 反向**——三范式合计 ~15 处 gold yes/no 答反,3B 的比较推理不足是跨范式瓶颈
4. **近失例(EM 归一化可救)**:`'1969-1974'` vs `'1969 until 1974'`;`'New York City'` vs `'Greenwich Village, New York'`(粒度)
5. **修复的工程结论**:plan_execute 0%→13.3% 证明旧数字里大部分是**工程缺陷**而非范式能力;跨范式公平比较必须在"格式合规 + 强制检索 + 正常终止"三件事修好后才有意义



## 三·五、决策质量分析(② 2026-09-25,项目核心)

### 1. 计数型决策指标(`05_analyze_decisions.py`,基于 v2 结果)

| 范式 | avg_retrievals | over率 | under率 | decision_acc |
|---|---|---|---|---|
| self_ask v3 | 2.43 | 13.3% | **0%** | 0.167 |
| react v2 | 1.63 | 3.3% | 60.0% | 0.067 |
| plan_execute v2 | 1.47 | 0% | 56.7% | 0.133 |
| ircot v2 | **1.10** | 0% | **80.0%** | 0.100 |

(n_needed=2,HotpotQA 每题两篇金标支持文档;under = 检索不足且答案错)

### 2. 金标文档命中重放(`07_gold_doc_hit.py` + `08_replay_gold_hit.py`)

轨迹 observation 被截到 300 字符,直接解析会漏掉每步 top-5 中后 ~3 篇 → `08` 把 trace 里每个检索步骤的 query 喂回同一检索器(确定性、不跑 LLM),拿完整 top-5 重算。**同一批 30 题同时用"原问题单次检索"重放作为同题集 baseline**:

| 范式 | 唯一query数/题 | agent金标覆盖率 | full命中(两篇全中) | baseline-query覆盖率(同30题) | EM |
|---|---|---|---|---|---|
| **react v2** | 1.43 | **70.0%** | **43.3%** | 68.3% | **6.7%(最低)** |
| plan_execute v2 | 1.40 | 61.7% | 36.7% | 68.3% | 13.3% |
| self_ask v3 | 0.93 | 36.7% | 20.0% | 68.3% | 16.7%(agent最高) |
| ircot v2 | 1.10 | 35.0% | 16.7% | 68.3% | 10.0% |
| baseline(原问题单查) | 1 | 68.3% | 36.7% | — | 21.2% |

产物:`decision_analysis_<paradigm>.json`(计数型)+ `gold_hit_replay_<paradigm>.json`(重放型,含 per_question 明细)

### 3. 核心发现(论文"决策机制 vs 决策评估"的直接素材)

1. **检索决策质量与答案质量解耦(甚至反相关)**:react 的 query 检索金标文档的能力**超过原问题单查**(覆盖率 70.0% vs 68.3%,full 命中 43.3% vs 36.7%),EM 却是四范式最低(6.7%)——**react 找到了对的文档,3B 读不懂/组合不出来**。反之 self_ask 的 query 质量差(36.7%)但 EM 最高。跨范式看,覆盖率与 EM 呈负相关:范式间 EM 差异主要由"答案组合能力"驱动,不是"检索决策能力"
2. **分解式 query 在这个索引上是检索负资产**:self_ask/ircot 的子问题只瞄准一跳,第二跳金标文档查不到(full 命中 16-20% vs baseline 36.7%)。原问题(两跳实体都出现)反而是最好的 query——**多跳问题"先分解再检索"的常识假设在 BGE+句子级索引上不成立**
3. **self_ask 的"over-retrieval"一半是浪费**:avg 2.43 次检索但唯一 query 只有 0.93 个/题——模型反复问同一个问题,计数型 over 指标掩盖了"重复"与"探索"的区别(决策指标设计素材)
4. **第 4 类工程缺陷:观察截断(2026-09-25 已修)**:`top_k=5` 但 agent 历史里 observation 截到 `[:500]` 字符(≈只见前 2-3 篇)、trace 存储截到 `[:300]`——**react 的检索优势根本没被 LLM 看见**,也没进 trace。修复:六处截断统一放宽到 1500/trace 1500(`base.py:98` / `react.py:39` / `self_ask.py:83` / `ircot.py:71,95` / `plan_execute.py:174`),单测 24/24 过
5. 计数型 under 与真实命中一致(ircot under 80% ↔ 覆盖率最低),计数型 over 与真实命中**不一致**(self_ask)——计数指标可作粗筛,金标命中重放才是决策质量的地面真相



1. **max_steps=8 对照**:4 范式各 n=30(当前 4 步可能限制 react/self_ask 的多跳分解;ircot 需另想办法阻止 1 跳交卷)
2. **决策指标(项目核心)**:跑 `05_analyze_decisions.py` 出 decision_accuracy / over/under_retrieval——EM 全输给 baseline 的负结果下,研究问题"决策机制 vs 决策评估"反而有了理想素材:**范式 EM 低不代表其检索决策无价值**
3. **正式矩阵**:4 范式 × n≥100 × max_steps=8
4. **可救的 EM 近失**:归一化层放宽数字/连字符/介词(`1969-1974` vs `1969 until 1974`)——预计 +2-4pp
5. **W5 核心矩阵**:4 范式 × 500 题

## 三·十、正式矩阵终版(① 2026-10-02,n=100 × max_steps=8,guard-nudge 臂并入)

**运行方式**:`scripts/run_formal_matrix.sh` 分批驱动(65 批 × 10 题,幂等可续跑),01:15-07:13 共约 6 小时;控制组复用 §3.6 的 n=30(offsets 0-29)只补跑 30-99,guard 臂 react 同样复用 0-29。分批原始:`ms100_batches/`(控制)+ `ms100_batches/guard/`(nudge);合并:`scripts/12_merge_matrix.py`(按 id 去重,n≠100 拒绝写出)。

### 1. 控制组主表(n=100,baseline 取同题集对齐)

| 范式 | EM | relaxed EM | F1 | AvgSteps | Finish 分布 | McNemar vs baseline |
|---|---|---|---|---|---|---|
| **self_ask** | **0.260** | **0.280** | **0.313** | 2.79 | 99 final_answer / 1 max_steps | p=0.664(12 vs 9 双对) |
| react | 0.170 | 0.200 | 0.220 | 3.28 | 93 finish / 7 max_steps | p=0.180(4 vs 10) |
| plan_execute | 0.140 | 0.160 | 0.180 | 2.68 | 89 finish / 11 max_steps | p=0.064(5 vs 14) |
| ircot | 0.130 | 0.140 | 0.161 | **1.06** | 100 final_answer | **p=0.031**(4 vs 14) |
| baseline 同题集(n=100) | 0.230 | 0.280 | 0.284 | 1.0 | — | — |
| baseline n=500(参考) | 0.212 | 0.302 | 0.300 | 1.0 | — | — |

### 2. 决策指标(计数型,`05_analyze_decisions.py`)

| 范式 | avg_retrievals | over率 | under率 | decision_acc |
|---|---|---|---|---|
| self_ask | 2.79 | 15% | **0%** | **0.22** |
| react | 1.56 | 7% | 63% | 0.17 |
| plan_execute | 1.30 | 0% | 65% | 0.14 |
| ircot | 1.06 | 0% | **81%** | 0.13 |

### 3. 金标命中重放(`08_replay_gold_hit.py`)

| 范式 | 金标覆盖率 | full 命中(两篇全中) | first-query 覆盖 | 唯一query数/题 | EM |
|---|---|---|---|---|---|
| react | **65.5%** | **38.0%** | 61.5% | 1.29 | 0.170 |
| plan_execute | 57.0% | 23.0% | 49.0% | 1.27 | 0.140 |
| self_ask | 37.0% | 18.0% | 32.0% | 1.09 | **0.260** |
| ircot | 29.5% | 14.0% | 28.5% | 1.04 | 0.130 |
| baseline(原问题单查,同 100 题) | 65.0% | 37.0% | — | 1 | 0.230 |

### 4. per-step 边际恢复率(`10_stepwise_coverage.py`)

| 范式 | 总检索 | 重复 query | step1 | step2 | step3 | step4+ |
|---|---|---|---|---|---|---|
| react | 156 | 27(17%) | 62.1% | 31.8% | 11.1% | **0%**(repeat 率 57-67%) |
| self_ask | 135 | 26(19%) | 51.6% | 28.1% | 7.7% | **0%** |
| plan_execute | 130 | 3(2%) | 49.0% | 44.1% | 25.0% | (只走到 step3) |
| ircot | 104 | 0 | 28.0% | 0%(仅 4/100 到达) | — | — |

### 5. guard-nudge 臂(`11_analyze_guard.py`,控制 = 上表控制组)

| 范式 | EM | F1 | AvgSteps | 真实检索/题 | Δ检索 | 守卫触碰 |
|---|---|---|---|---|---|---|
| react | 0.170→0.170 | 0.220→0.215 | 3.28→3.18 | 1.56→1.26 | **-19%** | 12 题(dup 15 / stop 6) |
| self_ask | 0.260→0.260 | 0.313→0.313 | 2.79→2.65 | 1.35→1.05 | **-22%** | 15 题(dup 15 / stop 2) |
| plan_execute | 0.140→0.140 | 0.180→0.180 | 2.68→2.68 | 1.30→1.26 | -3% | 5 题 |
| ircot | 不变 | 不变 | 1.06→1.06 | 1.06→1.06 | 0% | **0 题(守卫零触发)** |

伤害定位:三范式合计 19 题答案变化,**EM 变化 0 处**(触碰题控制组 EM:react 0/12、self_ask 0.2、plan_execute 0.2);react F1 微降 0.005 来自单题部分重叠(5a85b2d9),与 §3.9 同一模式。

### 6. 解读(对 §3.4-§3.9 结论的规模修正)

1. **"agentic 不回本"修正为"范式依赖"**:self_ask 在 n=100 追平 baseline(McNemar p=0.664,EM +3pp / F1 +2.9pp 不显著),不再是 n=30 时的显著落后;ircot 成为唯一显著低于 baseline 的范式(p=0.031)。n=30 的小样本噪声方向性误导——**跨范式结论必须以正式矩阵为准**
2. **解耦结论在 n=100 更极端**:self_ask 金标覆盖率最低(37% vs baseline 65%)却 EM 最高;react 覆盖率与 baseline 持平(65.5% vs 65.0%)但 EM -6pp。**EM 排名与检索覆盖率排名反相关(自_ask > react > plan_execute > ircot vs react > plan_execute > self_ask > ircot)——答案组合能力而非检索决策质量决定 3B 的成败**这一核心发现再次确认
3. **per-step 病理在 n=100 复现**:react/self_ask 第 4 步起边际恢复率 0 且 repeat 率过半;plan_execute 低压重复(2%)但 step2 恢复率仍 44%(计划式分解的检索效率最好);ircot 81% under + 单跳交卷 = "退化为单次检索的 agent"
4. **guard 节省稳定复现**:react -19% / self_ask -22%,EM 零损失;plan_execute -3%(干预面本来就小);ircot 零触发——守卫"无为而治",没有病理就没有干预,干预边界干净
5. **成本记录**:正式矩阵 + guard 臂合计 650 个新问题-run 约 6 小时(含 65 次模型加载);复用 n=30 省下 120 run

产物:`{paradigm}_n100_ms8.json` + `{paradigm}_n100_ms8_guard-nudge.json` + `summary_n100.json` + `decision_analysis_n100_{p}.json` + `gold_hit_replay_n100_{p}.json` + `stepwise_n100_{p}.json`(合并版 `stepwise_coverage_n100.json`)+ `guard_comparison_n100_{p}.json`;轨迹 `experiments/traces/{p}_ms8`、`{p}_ms8_guard-nudge`(文件号 0030-0099 为本轮新增);McNemar 为离线配对精确检验(strict EM,双对计数)



## 三·十一、react 回升归因消融(④ 2026-10-02)

§3.6 发现 react 从 v2(ms=4)到 ms8 EM +50%,但 ms8 同时带着观察截断修复,贡献无法拆分——本轮补"ms=4 + 截断修复在位"臂(同 30 题,`react_n30_ms4_fixed.json`,分批 `ms4fixed_batches/`,轨迹 `react_ms4_fixed`):

| 臂 | 观察截断 | max_steps | EM | relaxed EM | F1 | AvgSteps |
|---|---|---|---|---|---|---|
| v2(§3.4) | 500 字符 | 4 | 0.067 | 0.067 | 0.080 | 3.20 |
| 本轮 | 1500 字符 | 4 | 0.067 | **0.100** | 0.102 | 2.97 |
| ms8(§3.6) | 1500 字符 | 8 | **0.100** | **0.133** | 0.136 | 3.47 |

### 解读

1. **截断修复单独贡献:strict EM 不动、relaxed +3.3pp、F1 +2.2pp**——"让 LLM 看见完整检索结果"改善的是答案粒度/完整性(近失被 containment 吸收),不是把错误答案变对
2. **max_steps 4→8 在修复之上再 +3.3pp**(EM / relaxed 同幅)——预算给了 react 多一跳检索的空间
3. §3.6 的 react 回升 = 两因素叠加、各约 +3pp relaxed,不存在单一主因

## 三·十二、self_ask 扩样 n=500(⑤ 2026-10-03,§3.10 结论的规模检验)

self_ask 补跑 offsets 100-499(40 批),与 §3.10 的 n=100 合并成与 baseline 完全同题集的 n=500:

| 范式 | n | EM | relaxed EM | F1 | AvgSteps | McNemar vs baseline |
|---|---|---|---|---|---|---|
| self_ask | 500 | 0.152 | 0.190 | 0.215 | 2.76 | **p=0.003(显著低于)** |
| baseline | 500 | 0.212 | 0.302 | 0.300 | 1.0 | — |

产物:`self_ask_n500_ms8.json` + `n500_batches/`;轨迹 `self_ask_ms8`(trace_0100-0499)

### 核心修正(对 §3.10)

1. **n=100 的"追平"不是总体结论**:同题集配对下 self_ask 在 500 题上显著低于 baseline(双对 32 vs 62)。前 100 题恰是对 self_ask 有利的子集——其 EM 从子集 0.260 掉到全集 0.152,而 baseline 只从 0.230 到 0.212
2. **方法学教训(论文评估协议一节的直接素材)**:同一数据,n=100 给 p=0.664("无显著差异"),n=500 给 p=0.003("显著更差")——**样本量不足时"不显著"只是证据不足,不等于相等**
3. HotpotQA 最终结论:4 范式全部不优于单次检索 baseline,3B 规模 agentic 开销不回本;范式间真实排位 self_ask > react > plan_execute > ircot 仍然成立
4. **guard 的工程价值独立于此**:检索 -20% 零 EM 损失的结论与"agentic 是否追平 baseline"无关,任何方向下都成立

## 三·十三、跨数据集泛化(⑥ 2026-10-03,2WikiMultihopQA + MuSiQue)

**设置**:各 dev 前 100 题;索引 = 各自前 500 题语料(2wiki 12,209 句段 / musique 6,287 段落,musique 为段落级文档);同 HotpotQA 协议(ms=8、top-5、同 3B 模型)。数据由 `14_prepare_dataset.py` 从 hf-mirror 下载并转成 HotpotQA schema——**下游索引 / 基线 / Agent / 重放脚本零改动复用**(转换文件 `data/processed/{2wiki,musique}_dev_hotpot_schema.json`)。

### 1. 主表(baseline 为同 100 题单次检索)

| 数据集 | 范式 | EM | relaxed EM | F1 | AvgSteps | McNemar p |
|---|---|---|---|---|---|---|
| 2wiki | baseline | 0.060 | — | 0.156 | 1.0 | — |
| 2wiki | react | 0.060 | 0.060 | 0.076 | 3.31 | 1.000 |
| 2wiki | plan_execute | 0.070 | 0.120 | 0.119 | 2.73 | 1.000 |
| 2wiki | self_ask | 0.090 | 0.090 | 0.124 | 2.69 | 0.508 |
| 2wiki | ircot | **0.130** | 0.130 | **0.171** | 1.13 | **0.065†** |
| musique | baseline | 0.010 | — | 0.070 | 1.0 | — |
| musique | react | 0.010 | 0.020 | 0.071 | 3.44 | 1.000 |
| musique | plan_execute | 0.030 | 0.030 | 0.063 | 2.82 | 0.625 |
| musique | self_ask | 0.020 | 0.040 | 0.099 | 2.70 | 1.000 |
| musique | ircot | 0.010 | 0.020 | 0.100 | 1.05 | 1.000 |

### 2. 决策指标(`05_analyze_decisions.py`)

| 数据集 | 范式 | avg_retrievals | over率 | under率 | decision_acc |
|---|---|---|---|---|---|
| 2wiki | react | 1.55 | 7% | 79% | 0.060 |
| 2wiki | plan_execute | 1.43 | 0% | 71% | 0.070 |
| 2wiki | self_ask | 2.69 | 13% | **18%** | 0.070 |
| 2wiki | ircot | 1.13 | 0% | 78% | 0.130 |
| musique | react | 1.83 | 10% | 67% | 0.010 |
| musique | plan_execute | 1.48 | 1% | 63% | 0.030 |
| musique | self_ask | 2.70 | 16% | **0%** | 0.010 |
| musique | ircot | 1.05 | 0% | 90% | 0.010 |

### 3. 解读

1. **范式排名是数据集依赖的——HotpotQA 的排位在 2Wiki 上完全倒转**:ircot(13%)成为唯一名义超过 baseline 的范式(p=0.065 边缘),self_ask(9%)第二;HotpotQA 上垫底的 ircot 在 2wiki 上居首。其 CoT 式"检索-阅读-追问"的一跳节奏契合 2wiki 的组合型问题(先找 A 的属性再比对 B)。**没有任何范式跨数据集占优**
2. **难度阶梯**:baseline EM = HotpotQA 23.0% >> 2wiki 6.0% > musique 1.0%(同为前 500 题闭式语料协议)。musique 上全部范式贴地板(EM 1-3%,所有 p≥0.625)——3B + 小型闭式索引对 MuSiQue 基本不可解,只能比 F1(ircot / self_ask 0.100 / 0.099 vs baseline 0.070)与决策指标
3. **under-retrieval 结构随数据集改变**:2wiki 上 self_ask under 仅 18%(四范式最低)但 EM 只有 9%——检索不是 2wiki 的瓶颈,长尾实体答案的组合才是;musique 全体 under 63-90%(baseline 都几乎检索不到金标),问题在语料/索引规模而非决策
4. **金标命中重放(`08_replay_gold_hit.py`,按各自索引重跑;首次误用 hotpot 索引得到全 0 已作废)**

| 数据集 | 范式 | 金标覆盖率 | full 命中 | first-query 覆盖 | 唯一query/题 | EM |
|---|---|---|---|---|---|---|
| 2wiki | react | 62.5% | 29.0% | 59.0% | 1.28 | 0.060 |
| 2wiki | plan_execute | **63.2%** | **32.0%** | 53.5% | 1.37 | 0.070 |
| 2wiki | self_ask | 36.5% | 18.0% | 32.3% | 1.09 | 0.090 |
| 2wiki | ircot | **27.5%** | 12.0% | 26.0% | 1.08 | **0.130** |
| 2wiki | baseline(原问题单查) | 59.0% | 25.0% | — | 1 | 0.060 |
| musique | react | 53.0% | 16.0% | 51.0% | 1.41 | 0.010 |
| musique | plan_execute | **58.0%** | **23.0%** | 52.0% | 1.35 | 0.030 |
| musique | self_ask | 39.5% | 15.0% | 37.0% | 1.15 | 0.020 |
| musique | ircot | **27.5%** | 9.0% | 26.5% | 1.04 | 0.010 |
| musique | baseline(原问题单查) | 48.0% | 17.0% | — | 1 | 0.010 |

5. **解耦结论跨三个数据集成立且方向一致**:三个数据集上,范式级 EM 排名与金标覆盖率排名均反相关——hotpot:self_ask 覆盖 37%(最低)EM 最高;2wiki:ircot 覆盖 27.5%(最低)EM 最高,而 plan_execute 覆盖最高(63.2%)EM 仅 7%;musique 贴地板但同构。**"检索决策质量 × 答案组合能力 = 最终表现,且 3B 的瓶颈在后者"是跨数据集的稳健结论**
6. **分解式 query 的检索负资产跨数据集复现**:self_ask / ircot 的子问题式 query 在 2wiki(36.5% / 27.5% vs baseline 59.0%)和 musique(39.5% / 27.5% vs 48.0%)上都大幅低于原问题单查——与 §3.5 发现 2 一致,且排除了"HotpotQA 特有"的解释

产物:`{2wiki,musique}_batches/`(各 40 批)+ `{ds}_{p}_n100.json`(按范式合并)+ `baseline_{2wiki,musique}.json` + `decision_analysis_{ds}_{p}.json` + `stepwise_{ds}_{p}.json` + `gold_hit_replay_{ds}_{p}.json`;轨迹 `experiments/traces/{p}_{2wiki,musique}`;泛化实验 + 消融 + 扩样合计 650 个新问题-run 约 7.5 小时(react ms4 30 + self_ask n500 400 + 2wiki 500 + musique 500 中扣除复用)

## 五、下一步(按优先级,2026-10-03 更新)

1. **实验阶段全部完成** ✅:baseline(3 数据集)· 正式矩阵 n=100 + guard 臂(§3.10)· react 归因消融(§3.11)· self_ask n=500(§3.12)· 2Wiki / MuSiQue 泛化(§3.13)· 决策指标 / 金标重放 / per-step 曲线(§3.5-3.8)· intervention(§3.9)· 论文图表(fig1-4,§六)· 论文表格(t1-t7,`15_export_paper_tables.py`)
2. **论文初稿(W13-W14,唯一剩余大项)**:故事线按最终证据更新——工程缺陷系统性排除(§3.2-3.4)→ 正式矩阵 n=100(§3.10)→ **扩样修正:n=500 全范式显著低于 baseline(§3.12,含"不显著≠相等"的方法学讨论)** → 决策-答案解耦(§3.5/§3.10)→ per-step 病理(§3.8)→ guard 干预闭环(§3.9/§3.10)→ 跨数据集泛化:排名数据集依赖 + 难度阶梯(§3.13)
3. (论文素材备用)self_ask n=500 的金标命中重放与 guard n=500 臂——若初稿需要更强证据再补
4. (远期)更大规模索引(全语料而非前 500 题闭式语料)· 更大模型(7B)复验"不回本"边界

## 三·六、max_steps=8 对照矩阵(① 2026-09-25,n=30,同 30 题,带观察截断修复)

| 范式 | EM(v2/ms=4) | **EM(ms=8)** | F1 v2→ms8 | AvgSteps v2→ms8 | 金标覆盖率(ms8) | vs baseline 21.2% |
|---|---|---|---|---|---|---|
| self_ask | 0.167 | **0.167 (5/30)** | 0.187→**0.231** | 2.43→2.90 | 36.7% | 仍低 |
| plan_execute | 0.133 | **0.133 (4/30)** | 0.177→0.147 | 2.77→2.63 | 58.3% | 仍低 |
| react | 0.067 | **0.100 (3/30)** | 0.080→**0.136** | 3.20→3.47 | **70.0%** | 仍低 |
| ircot | 0.100 | **0.100 (3/30)** | 0.138→0.113 | 1.10→**1.07** | 31.7% | 仍低 |

产物:`react_n30_ms8.json` / `plan_execute_n30_ms8.json` / `self_ask_n30_ms8.json` / `ircot_n30_ms8.json` / `summary_ms8.json` / `gold_hit_*_ms8.json`;轨迹 `experiments/traces/*_ms8`(此时 trace 已含完整 5 篇文档,`07` 直算与 `08` 重放互相验证:react 70.0% 完全一致)

### 解读

1. **"max_steps=4 太紧"假设基本被否**:放宽到 8 步后各范式步数几乎没涨(react 3.20→3.47,self_ask 2.43→2.90,ircot 1.07→1.10)。**瓶颈不是步数预算,是每一步的质量**——ircot 照旧一跳交卷,agent 自己不想多走
2. **唯一显著回升的是 react(EM +50%,F1 +70%)**,且截断题 5→3、步数几乎没变——回升更可能来自**观察截断修复**(react 步数最多、历史最长,是 500 字符截断的最大受害者)而非 ms=8。这与 §3.5 发现 1 闭环:react 的 query 金标覆盖率(70%)本就是四范式最高,现在"看见了检索结果",EM 立刻回升——**检索决策质量 × 结果可见性 = 答案质量**的直接证据
3. **其余三范式 EM 原地踏步**:self_ask F1 升(0.187→0.231)但 EM 不动;plan_execute/ircot F1 反而略降(粒度近失 + 截断修复后输入变长、3B 注意力被稀释的候选解释)。**"3B 规模 agentic 开销不回本"的结论在 ms=8 下依然成立**
4. **金标覆盖率格局不变**:react 70.0% > plan_execute 58.3% > self_ask 36.7% > ircot 31.7%,EM 与覆盖率仍然错位——"答案组合能力是 3B 的真瓶颈"再次确认
5. ms8 矩阵内部自洽且跨范式公平(同 30 题、同修复、同预算),可直接作为正式矩阵(n≥100)的前置 pilot

## 三·七、EM 归一化放宽(② 2026-09-25,relaxed_em)

**动机**:v2/ms8 failure analysis 显示存在一批"答案对但格式/粒度错"的近失(`1969-1974` vs `1969 until 1974`、`New York City` vs `Greenwich Village, New York City`、`Compliance (film)` vs `Compliance`)。strict EM 按字面全错计,低估真实答案能力。

**规则**(`src/evaluation/answer_metrics.py` 新增 `relax_answer` + `relaxed_em`,单测 25/25):
- 尾部括号消歧后缀剥除(`Compliance (film)` → `Compliance`;中置括号可能载实义,不动)
- 年份区间连字符拆开(`1969-1974` → `1969 1974`)
- 月份名转数字(`september 8, 2017` → `8 9 2017`)
- 双向 token containment:短方 ≥2 词且完全落 in 长方 → 命中(粒度近失);**单词答案不适用**(防 `no` ⊆ `north` 假阳);**yes/no 反向错答不被吸收**(比较推理失败是独立失败模式,保持判错)

**结果(离线重评,不重跑 GPU)**:

| 结果 | strict EM | relaxed EM | 被吸收 |
|---|---|---|---|
| baseline_v2(n=500) | 0.212 | **0.302** | 45 题(+9.0pp) |
| self_ask ms8 | 0.167 | **0.200** | +1 题 |
| plan_execute ms8 | 0.133 | **0.167** | +1 题 |
| react ms8 | 0.100 | **0.133** | +1 题 |
| ircot ms8 | 0.100 | 0.100 | 0 |

**解读**:
1. **近失吸收对 baseline 更有利(+9pp vs 各 agent +3pp)**——单次检索的短答输出格式更贴 gold,agentic 范式答案更长、更容易跑偏。放宽后差距不但没缩小,反而从 4.5pp 扩到 10.2pp。**"agentic 不回本"的结论在放宽口径下更加稳固**
2. agent 侧被吸收的三例全是教科书近失(react/plan_execute 的 `1969-1974`、self_ask 的 `New York City.`),抽查无假阳;ircot 零吸收(它的问题是粒度反方向:答 `Fight for Kansas` 丢实词,containment 救不了)
3. **方法学意义**:论文里 strict/relaxed 双口径都报,relaxed 的 +9pp baseline 提升本身说明 HotpotQA 的 EM 对"格式正确答案"存在系统性低估——评估协议讨论的一节素材

## 三·八、per-step 边际命中曲线(③ 2026-09-25,`10_stepwise_coverage.py`,ms8 轨迹)

**问的问题:"agent 多走一步,检索决策质量真的变好了吗?"** 边际恢复率 = 第 k 步新命中的金标标题数 / 该步开始时未命中的金标标题数。

| 范式 | 总检索 | 重复 query | step1 恢复率 | step2 恢复率 | step3+ |
|---|---|---|---|---|---|
| react | 57 | **17(30%)** | 65.0% | 42.9% | **全部归零**(且 repeat 率 60-100%) |
| self_ask | 45 | 8(18%) | 50.0% | 22.2% | 33.3% 后归零 |
| plan_execute | 39 | 2(5%) | 50.0% | 45.5% | 归零 |
| ircot | 32 | 0 | 31.7% | —(只有 2 题走到 step2) | — |

### 解读(论文"何时该停"决策分析的核心素材)

1. **检索收益集中在第 1-2 步,第 3 步起全面归零**:react step3+ / plan_execute step3 / self_ask step4+ 的边际恢复率全部 = 0,且 zero_new_gain 率 75-100%。**max_steps=8 的预算大量被浪费在零收益步上**——agent 该停的时候不停。这是"under-retrieval 不是瓶颈,该停不停才是浪费"的直接证据,呼应 §3.6 "步数没涨"的发现:不是不能走,是走了没用
2. **react 是"重复查询大户"**:17/57 次检索(30%)是重复 query,step5+ 100% 重复——3B 在长历史下失去生成新 query 的能力,开始复读。**重复 query 检测器 + 早停守卫是明确的改进方向**(工程上 trivially implementable,论文的 intervention 一节)
3. **self_ask 前两步最像"有规划的检索"**(step1 50% → step2 22%,无重复),但第 2 步恢复率掉到 22% 说明其分解式子问题第二跳质量差(与 §3.5 发现 2 一致)
4. **ircot 几乎没有"多步"可言**:28/30 题一跳就交卷(step2 只有 2 题到达),其覆盖率天花板(31.7%)纯粹是 step1 单发决定的——它是"退化成单次检索的 agent",开销却比单次检索高(每题 2 次 LLM 调用)
5. **跨范式对照的决策质量排名(step1 恢复率)**:react 65% > self_ask/plan_execute 50% > ircot 32%——与 §3.5 金标覆盖率排名一致,互为印证




## 三·九、Intervention 实验:重复 query 检测器 + 零收益早停守卫(2026-09-26)

**§3.8 直接推论的闭环实验:"决策评估(定位复读与零收益步)→ 指导决策机制设计(守卫)→ 验证收益"。**

### 设计

守卫只换 Search 动作(`src/actions/guarded_search.py`),agent 循环主体不变;控制组 = §3.6 ms8 结果,同 30 题。

- **重复 query 检测器**:query 归一化(小写/去标点/压空白)后与已检索集合比对(§3.8 分析口径的严格超集),重复 → **拦截不检索**,注入纠错指令("换 query 或立即 Finish")
- **零收益早停**:检索 top-k 全部是已见文档 → 警告;连续 2 次浪费(重复/零新增)→ **STOP 指令**("不得再检索,立即交卷")
- 两臂:**nudge**(只注入指令)vs **hard**(STOP 后再检索则 agent 强制终止,`finish_reason="guard_early_stop"`)
- react × n=30 × ms=8——重复复读大户(§3.8:17/57 检索是重复,step5+ 100% 复读)

### 结果(react,控制 = §3.6 ms8)

| 指标 | 控制组 | guard-hard | guard-nudge | Δ |
|---|---|---|---|---|
| EM | 0.100 | **0.100** | 0.100 | **0** |
| relaxed EM | 0.133 | 0.133 | 0.133 | 0 |
| F1 | 0.136 | 0.119 | 0.119 | -0.017 |
| AvgSteps | 3.47 | 3.23 | 3.23 | -0.24 |
| 真实检索/题 | 1.90 | **1.27** | 1.27 | **-33%** |
| 终止分布 | 27 finish / 3 max_steps | **30 finish** | 30 finish | 截断 3→0 |

守卫干预明细(hard 与 nudge 完全一致):9 次重复 query 被拦截、2 次 STOP 触发、触碰 7/30 题、0 次强制终止。

### 核心发现

1. **检索算力 -33%,答案质量零损失**:EM / relaxed EM 一点不动;F1 -0.017 全部来自单题部分重叠丢失(5a85b2d9:`'Enesul Fateh'`→`'Aladin'`)。被守卫触碰的 7 题在**控制组 EM = 0/7**——守卫没有改错任何一题,截掉的全是本来就不会做对的检索步。"该停不停才是浪费"(§3.8)的干预版证实
2. **3B 100% 服从 STOP 指令**:2 次触发、0 次无视;hard 的强制终止从未启用,nudge 与 hard **逐题完全一致**(30 题答案 0 差异)。纠错反馈 + STOP 指令对 3B 已经足够,不需要硬终止——nudge-only 是更便宜的设计
3. **复读病理被治好**:控制组 3 题 max_steps 截断(复读烧完预算、fallback 空答案)→ 实验组 30/30 正常 finish_action。守卫把"复读烧步数"转成"被拦 → 转向交卷"
4. **干预边界干净**:7 个答案变化全部发生在触碰题内,干净题 0 变化——受控比较成立,变化可完全归因于守卫

### 解读与方法学意义

1. "决策评估指导决策机制设计"的完整闭环拿到第一段证据:§3.8 per-step 曲线定位病理(复读/零收益步)→ 两个 trivially implementable 守卫 → **检索 -33%、正常终止 100%、EM 零损失**。守卫不提升 EM(3B 的瓶颈在答案组合,§3.5),但把浪费的算力全部省下——"agentic 不回本"的负面结论之下,这是把亏损砍掉的正路
2. F1 微降的机制:被拦的复读步原本会"把同样 5 篇文档再喂一遍",对部分重叠答案偶有零星 F1 收益——这是复读步唯一的"功能",不值得 33% 的检索开销
3. n=30 是 pilot;若守卫臂并入正式矩阵(n≥100)可验证节省的稳定性。其余三范式重复率低(self_ask 18%/plan_execute 5%/ircot 0%),预期收益主要来自零收益步拦截而非重复检测

产物:`react_n30_ms8_guard-hard.json` / `react_n30_ms8_guard-nudge.json` / `guard_comparison_react_{hard,nudge}.json`(分析);轨迹 `experiments/traces/react_ms8_guard-{hard,nudge}`;分批原始 `ms8_batches/guard/`;代码:`src/actions/guarded_search.py` + `03_run_agentic_rag.py --guard {off,nudge,hard}` + `scripts/11_analyze_guard.py`;单测 39/39

## 六、论文图表(2026-10-02,`scripts/13_make_paper_figures.py`)

产物:`experiments/figures/fig{1,2,3,4}_*.{pdf,svg,png}`(矢量 PDF/SVG + 300dpi PNG,单栏宽 4.7in / 双面板 6.3in,Okabe-Ito 色盲安全色板,颜色=范式跨图统一)。

| 图 | 文件 | 论点 | 图注要点(中文,论文直接可用) |
|---|---|---|---|
| 图 1 | `fig1_main_comparison` | self_ask 唯一追平 baseline,ircot 显著最差 | 柱高为原始比例(n=100,ms=8);每组黑色虚线 = 单次检索 baseline 同题集指标(0.230/0.280/0.284);strict-EM 组柱上为配对 McNemar 精确检验 p 值(对 baseline;†:0.05≤p<0.10;*:p<0.05) |
| 图 2 | `fig2_pareto_accuracy_vs_cost` | EM 与检索成本反相关 | x = 每题平均真实检索次数(控制臂轨迹重放),y = strict EM;星标 = baseline(单次检索,同题集 EM 0.230),灰色点线为其参考线 |
| 图 3 | `fig3_stepwise_marginal_recovery` | 检索收益集中第 1-2 步,react/self_ask 第 4 步归零 | 边际恢复率 = 第 k 步新命中金标标题 / 该步起未命中数;实心点 = 到达该步题数 ≥10,空心 = <10(小样本仅示意);线终止于该范式实际到达的最大步 |
| 图 4 | `fig4_guard_intervention` | 守卫 -19%/-22% 检索、EM 零损失 | (a) 每题平均真实检索次数,柱顶 = 相对变化;(b) strict EM,Δ = guard−control;灰 = 控制组,绿 = guard-nudge(n=100 同题) |

复现:`python scripts/13_make_paper_figures.py`(读 summary_n100 / decision_analysis_n100 / gold_hit_replay_n100 / stepwise_coverage_n100 / guard_comparison_n100 + baseline_rag_v2,离线,无 GPU)。
