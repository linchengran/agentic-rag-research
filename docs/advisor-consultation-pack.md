# 导师咨询稿 —— Agentic RAG 选题方案沟通

> 适用场景:**邮件 / 微信**发送,或者**面对面开会**当 agenda 用。
> 配套材料:`docs/开题报告.md`、`docs/选题方案与开题报告思路.md`(13,000 字)、`docs/references/reading-list.md`(24 篇 / 9 篇必读)。

---

## 一、一句话项目描述(30 秒电梯版)

> "本项目研究 **Agentic RAG 在多跳问答上的迭代式检索决策机制**——Agent
> 应该何时检索、检索什么、检索几次。
> 在我已有 RAG 项目基础上,我新增 Agent 决策层(ReAct / Plan-Execute /
> Self-Ask / IRCoT 四范式),以最少的检索步数获得最高的答案准确率,
> 形成"基础 RAG → Agentic RAG"的演进证据链。"

---

## 二、已完成工作(展示节奏感,不是求表扬)

| 阶段 | 状态 | 产出 |
|------|------|------|
| 选题定案 | ✅ | 选题方案 + 开题报告 |
| 文献调研 | ✅ | 24 篇文献,9 篇已标必读(IRCoT / Self-RAG / FLARE / ReAct 主轴) |
| 代码骨架 | ✅ | 1960 行 src/,5 Agent + 3 Retriever + 4 Action + 2 评估模块 |
| 单元测试 | ✅ | 22 / 22 通过(ReAct 完整,其余用 Mock) |
| 数据 / 索引 | ⏳ | 待 W3-W4 启动,**有意暂缓,先确认以下 10 个问题** |

---

## 三、想请教您的 10 个问题(按优先级排序)

### 🔴 P0 · 不解决会影响后面所有工作

**Q1. 选题方向是否符合您组的研究主线?**
> 我组目前的 RAG 工作偏检索质量优化,这个选题偏 Agent 推理与决策,
> **重心偏离**还是**自然延伸**,请您给个判断。

**Q2. 研究问题应聚焦到"决策机制"还是"决策质量评估"?**
> 方案文档里我把**决策质量评估**列为重点,但**决策机制**(Agent 范式本身)也很有论文空间。哪边更好发?

**Q6. "检索决策正确率"的金标准如何定?**
> 这个指标需要"理想检索 oracle"或"人工标注"。我的初步方案:
> 取 HotpotQA 每个问题的所有 `support_facts` 拼接成 oracle 答案 → 计算 Agent
> 实际检索集合与 oracle 的精确率 / 召回率。
> **这个定义您接受吗,还是有更标准的做法?**

**Q8. 实验周期与硬件:32 组对比(4 范式 × 2 模型 × 4 数据集)可行性?**
> 默认配置:Qwen2.5-3B + Qwen2.5-7B(本地 4bit),HotpotQA / 2Wiki / MuSiQue ×
> dev / distractor。**组里 GPU 是否够跑 32 组?不够的话优先级?**
> 如果您定优先级,我就只跑**最关键的 P0 矩阵**(比如 4 范式 × 1 模型 × 1 数据集)。

### 🟡 P1 · 影响工作量和复用边界

**Q3. Agent 范式选择:ReAct / Plan-Execute / Self-Ask / IRCoT 是否合理?**
> 我选这四个是因为各自代表一类决策风格(交错推理、单步规划、显式子问题、CoT 检索融合)。
> **是否要补 ReAct 的反思变体(Reflextion)或 Self-RAG 这种自批判范式?**

**Q4. 检索器:直接复用已有项目 vs 自包含独立实现?**
> 我的两个方案:A. `pip install -e ../rag-research-project` 复用(代码量小,
> 但工作量可能被质疑);B. 核心文件复制到 `src/retrievers/legacy/`(自包含证据链)。
> **毕设标准您倾向于哪个?如果是投稿(顶会/核心),A 是否可接受?**

**Q9. 工程组织:独立项目(A)还是 Monorepo(B)?**
> 当前是 A(`agent-research-project` 与 `rag-research-project` 平级)。
> 您希望**毕设答辩时**两个项目都展示,还是只展示新项目 + 引用已有项目作 evidence?

**Q10. 引用关系:已有 RAG 项目如何在我论文里出现?**
> 候选写法:作为"前置工作"在第一章背景介绍 / 作为 baseline 在第四章实验。
> **您建议放哪?感谢致谢部分谁?**

### 🟢 P2 · 影响细节但容易达成一致

**Q5. 数据集语言:是否需要补充中文场景?**
> 当前默认 HotpotQA / 2WikiMultihopQA / MuSiQue(英文)。
> 中文方案:CMRC / C³ / DuReader_multihop。**毕设需要中文吗?**

**Q7. 论文目标:毕设即可,还是投稿会议?**
> 如果目标是投稿,我会把"决策行为可解释性"和"失败模式分类"再加权重;
> 如果只是毕设,可以减少消融实验换更多 analysis 图表。
> **您的预期?这决定 W11-W16 的实验规模。**

---

## 三·附 · 一个需要您拍板的工程层硬约束(影响 W3-W4 第一步)

这不属于研究问题,但**会卡 W3-W4 的起跑姿势**,所以单独列出来。

### 议题:Windows + Python 3.13 下 `bitsandbytes` 的可用性

**现状**
- 我的本地环境是 **Windows 11 / Python 3.13.5**。
- `config/config.yaml` 里 `llm.load_in_4bit: true` 已经写死 4bit 量化路径。
- 但 **`bitsandbytes` 在 Windows + Python 3.13 上很可能装不上**(没有官方 wheel,
  新版 Python 缺位的现象历来如此)。
- 这意味着 Qwen2.5-3B / 7B 的 4bit 路径在我的本机可能直接跑不通。

**三个候选方案,请您选一个**

| 方案 | 怎么做 | 优点 | 缺点 |
|------|--------|------|------|
| **A. 改配置,放弃 4bit** | 把 `load_in_4bit` 改成 `false`,Agent 用 fp16(3B) 或 fp16 截断(7B) | 本机能跑,W3 不卡 | 显存占用大,Qwen2.5-7B 在单卡 12GB 跑不动 |
| **B. 改用 AWQ / GPTQ 预量化模型** | 选现成的 AWQ / GPTQ 版(如 `Qwen2.5-3B-Instruct-AWQ`),用 `auto-gptq` 加载 | 本机能跑且显存省,效果与 4bit 接近 | 又引入一个新依赖(`auto-gptq`),模型精度由社区版本决定 |
| **C. 申请组里 Linux + CUDA 服务器** | W3 起实验全在服务器跑,本机只写代码 | 一切照原计划,4bit 路径不动 | 需要您批服务器时间,排队 / 共享问题 |

**顺带问一句(与上面耦合)**:**本项目是否需要放到 GitHub 公开仓库?**
- 如果公开:必须在 README 里写清楚 Windows / Mac 用户的替代方案(影响毕设验收材料)
- 如果纯本地(私有 / 不公开):AWQ / fp16 路径内部消化即可

**两问给出结论后,我会立即把 `config/config.yaml` 的 `llm.*` 段按选定方案改掉,
然后才进 W3-W4 实跑。**

---

## 四、我会从您那里带走的"决策"

按重要性,我希望本次能拿到以下**结论性答复**(yes/no 或 选项):

1. ☐ **方向 OK / 需要调整**(Q1)
2. ☐ **研究问题聚焦到 决策机制 / 决策评估 / 两者并重**(Q2)
3. ☐ **检索决策正确率的金标准方案 OK**(Q6)
4. ☐ **Agent 范式列表 OK / 需补充 / 需删减**(Q3)
5. ☐ **检索器复用策略 A / B**(Q4)
6. ☐ **论文目标 = 毕设 / 投稿**(Q7)
7. ☐ **P0 矩阵固定为 4 范式 × 1 模型 × 1 数据集 / 其他组合**(Q8)
8. ☐ **工程组织 A / B**(Q9)
9. ☐ **Q5 中文需求 / Q10 引用位置**(P2,可在 W3 内邮件补答)
10. ☐ **bitsandbytes 替代路径 A / B / C**(三·附)
11. ☐ **仓库公开 / 不公开**(三·附 · 顺带问)

---

## 五、如果您今天时间紧

**至少给我 Q1 / Q2 / Q4 / Q6 / Q7 这五个**,其他我邮件补问。这样 W3
启动不会被卡。

---

## 六、我现在的请求

- **面对面**:本周末 / 下周一您哪个时段方便?约 30-45 分钟。
- **邮件**:如果您更愿意先看材料再回复,请回这封邮件,我**周四 / 周五**再发一次进展。
- **异步反馈**:您只要在每个 P0 / P1 问题旁写一行意见,我据此调整方案文档 v2。

---

## 七、v2 · 答卷回填(2026-09-23,导师下放决策权后)

> 导师原话:"**自己判断,怎样才能做到最好**"。
> 本节为我方的答卷,作为 W3-W16 的执行依据。配套同步:`docs/选题方案与开题报告思路.md` v2 + `config/config.yaml`(三·附 已落)。

### 11 项决策(填入第四节勾选清单)

| # | 议题 | 决策 | 一句话理由 |
|---|------|------|-----------|
| 1 | Q1 方向 OK? | ✅ OK | Agentic RAG 决策机制是 2024-2026 主流研究线 |
| 2 | Q2 机制 vs 评估 | **两者并重** | 范式对比(贡献 1)+ 决策评估新指标(贡献 2),体量充实 |
| 3 | Q3 范式列表 | 保持 4 个 | 已覆盖四类决策风格,加 Self-RAG 跑不完 |
| 4 | Q4 复用 vs 自包含 | **B 自包含** | 毕设评审看"从头构建",A 像抄 |
| 5 | Q5 中文 | 不加 | 主流多跳 QA 全英文,工程量 ×2 收益不稳 |
| 6 | Q6 决策金标准 | **双 oracle** | support_facts(段落级)+ IRCoT sentence-level(句子级) |
| 7 | Q7 毕设 vs 投稿 | **毕设为主 + 投稿弹性** | 核心矩阵跑通 = 毕设;评估/失败分类做厚 = 留投稿 |
| 8 | Q8 32 组可行性 | **核心 4 组,扩展看时间** | 32 组 ≈ 7.4 天连续跑;4 组 1 天完,留时间做分析 |
| 9 | Q9 工程组织 | A 独立项目 | 当前事实,改 Monorepo 收益小风险大 |
| 10 | Q10 引用位置 | 背景简短 + baseline 详细 | 背景交代起点,实验对比新方法 |
| 11 | 三·附 量化路径 | **GPTQ + 3B + auto-gptq** | 8GB VRAM 硬约束,9-23 已落 |

### W3-W16 执行节奏

- **W3-W4 · 数据 + 检索层** — 装包 + 下载 HotpotQA / 2WikiMultihopQA / MuSiQue + BGE-small-zh 索引 + retrieval-only 评估(3 数据集 Recall@5/10)
- **W5-W8 · 核心 Agent 实跑** — **4 范式 × 3B-GPTQ × HotpotQA = 4 组核心矩阵**(1-2 天跑完),评估 EM / F1 / decision_accuracy / over_retrieval_rate / under_retrieval_rate / avg_steps / latency
- **W9-W16 · 扩展 + 分析** — 加 2Wiki / MuSiQue / 7B(看 8GB 跑不跑得动);失败模式分类;决策可解释性;bootstrap 显著性检验

### W3 进度(2026-09-23 晚,全部完成)

- ✅ 装包 + 环境验证:torch 2.9.1+cu128 / RTX 5060 8GB / BGE-small-zh / FAISS / HF mirror 全通
- ✅ **重大转折**:transformers 5.x 原生支持 `GPTQConfig`/`AwqConfig`,`local_model.py` 改用 native,**不再依赖 auto-gptq / autoawq**
- ✅ 数据下载:HotpotQA dev distractor **7405 题**(JSON 44.6 MB)
- ✅ 索引构建:BGE-small-zh + BGE-reranker-base,**20631 段 / 512 dim / IndexFlatIP**
- ✅ **检索层 baseline(不动 LLM)**:**Recall@5 = 0.918 / @10 = 0.922 / @20 = 0.922**(500 题)
- ✅ 新增 `scripts/00_eval_retrieval.py`(评估脚本)

### W4 进度(2026-09-23 夜,LLM 装通)

- ✅ **量化路径 pivot**:GPTQ 走不通(optimum 2.3.0 bug)+ autoawq 装不上(triton)→ 退回 **bitsandbytes 4bit 动态量化**(咨询稿方案 A 备选,Win+Py3.13 wheel 全)
- ✅ 模型:Qwen/Qwen2.5-3B-Instruct 完整 fp16 6GB,bnb 在线 4bit → **VRAM 2.05GB**,8GB 宽裕
- ✅ **Smoke test**:装 4:22 / 稳态推理 ~1.1s/题(短答)/ 500 题估 9 分钟
- ✅ **Pipeline 修复**:
  - `src/utils/env.py` 强制覆盖 `huggingface_hub.constants` 全部 endpoint(`HF_ENDPOINT`/`ENDPOINT`/`HUGGINGFACE_CO_URL_TEMPLATE`/`HUGGINGFACE_CO_URL_HOME`/`_HF_DEFAULT_ENDPOINT`)——HF Hub URL 模板硬编码了 huggingface.co,必须改常量才行
  - 缓存中 stub 了 `adapter_config.json`/`processor_config.json`/`preprocessor_config.json` 三个 sentence-transformers 6.x 必查的可选文件
- ✅ **Prompt 修复**:`SINGLE_RAG_SYSTEM_PROMPT` 改成英文 + 短答指令
- ✅ **F1 修复**:`normalize_answer` 用 jieba 分词
- 🟡 **正在跑**:`02_run_baseline_rag.py --limit 500`(预计 ~30 分钟)
- 5 题样本:q3/q4 答对,q1 答错,q2/q5 模型放弃 → F1 平均 0.024
- ⏳ 下一步:W5 跑 4 范式 Agentic 核心矩阵

---

> 文档版本:**v2** · 2026-09-23(答卷回填)· 配套项目 `agent-research-project`
> 版本沿革:v1 → v1.1(三·附 增加)→ **v2**(导师授权下放后答卷回填)
> W3 已启动,11 项决策锁定,W3-W16 节奏已规划。
