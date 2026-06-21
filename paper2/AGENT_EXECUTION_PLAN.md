# THEMIS 论文 Agent 执行方案

> 目标: 使用 `academic-researcher` skill 完成 ARR/EACL 投稿论文  
> 论文主线: KG 分解需求 + Developer 生成 + Judge 验证  
> 验证支撑: 300 benchmark + 消融实验 + 语义契约实验  
> 最后更新: 2026-05-16

---

## 一、论文整体设计

### 标题 (建议)

> **Requirement-Aware Code Repair via Knowledge Graph Decomposition and Semantic Contract Guidance**

备选:
- *From Issues to Graphs: Structured Requirement Decomposition for LLM-Based Program Repair*
- *Decomposing Issues into Verifiable Constraints: A Knowledge Graph Approach to Program Repair*

### 作者与投稿

- 投稿系统: ACL Rolling Review (ARR)
- 目标会议: EACL (或其他 *ACL 会议)
- 论文类型: Long Paper (8 pages + references + limitations)
- 模板: ACL LaTeX (`\usepackage[review]{acl}`)

### 贡献声明 (3 Contributions)

1. **A requirement-aware code repair framework** that decomposes natural language issues into structured Knowledge Graph constraints, enabling verifiable repair against decomposed requirements
2. **A large-scale evaluation** on SWE-bench Lite (300 instances across 12 repositories) with comprehensive ablation studies isolating component contributions
3. **A controlled empirical finding** that explicit semantic contract guidance improves repair outcomes over context-only baselines (0/4 → 1/4), while more complex reranking does not help further

---

## 二、论文结构 (章节分配)

| Section | 内容 | 预估页数 | 数据来源 |
|---------|------|:---:|------|
| Abstract | 200 词摘要 | 0.3 | — |
| 1. Introduction | 问题动机、贡献列表 | 1.0 | paper2/*.md, 现有 paper |
| 2. Related Work | APR, LLM-based repair, KG from code | 1.0 | 需 agent 搜索 |
| 3. System Architecture | KG 分解 + 4-Agent 流水线 | 1.5 | paper2/swebench_benchmark_architecture.md |
| 4. Experimental Setup | Dataset, workflow, presets, metrics | 1.0 | paper2/gpt51_codexmini_experiments.md, swebench_benchmark_architecture.md |
| 5. Results | 300-case resolved/patch rate + Ablation + Semantic | 2.0 | eval JSON summaries, gpt51_codexmini_experiments.md |
| 6. Discussion & Analysis | 19.3% resolved vs 94% patch, semantic contracts, negative rerank | 0.6 | outline_and_claims.md |
| 7. Limitations | 系统限制 + ARR 必需 Limitations | 0.5 | docs/ARR_EACL_SUBMISSION_GUIDE.md |
| 8. Conclusion | 总结与未来工作 | 0.2 | — |

---

## 三、分阶段执行计划

### ═══════════════════════════════════════
### Stage 0: 环境准备
### ═══════════════════════════════════════

**目标**: 确保 agent 已加载 skill 并能访问所有材料

**重启 OpenCode 后发送的第一条消息**:

```
load skill academic-researcher

我要写一篇投稿 EACL (via ARR) 的 Long Paper。

论文主题: 基于知识图谱的需求分解方法用于 LLM 驱动的代码修复

论文标题候选:
1. Requirement-Aware Code Repair via Knowledge Graph Decomposition and Semantic Contract Guidance
2. From Issues to Graphs: Structured Requirement Decomposition for LLM-Based Program Repair

请先不要开始写作，等我给你完整的论文设计和材料。
```

**让 agent 读取的材料** (逐条发送):

```
请先阅读以下文件了解项目:
1. README.md — 项目概览
2. PROJECT_FLOW_REFERENCE.rst — 完整架构
3. paper2/swebench_benchmark_architecture.md — swebench_benchmark系统架构详解
4.THEMIS_METHODOLOGY.md - 完整方法论
5. paper2/gpt51_codexmini_experiments.md — 实验数据
6. docs/ARR_EACL_SUBMISSION_GUIDE.md — 投稿格式要求
7. paper/SUBMISSION_STATUS.md — 现有论文参考 (了解之前的定位)
```

---

### ═══════════════════════════════════════
### Stage 1: 论文大纲 + Claims-Evidence 矩阵
### ═══════════════════════════════════════

**发送**:

```
## 论文结构要求

### 贡献声明 (请逐条验证是否可行)

C1: 我们提出了一个 requirement-aware 代码修复框架，将自然语言的 issue 分解为结构化的知识图谱约束，使修复可针对分解后的需求进行验证。

C2: 在 SWE-bench Lite 的 300 个实例（12 个仓库）上进行了大规模评估，包含 6 组消融实验以隔离各组件贡献。

C3: 受控实证发现：显式语义契约指导比纯上下文基线改善了修复结果（0/4 → 1/4），而更复杂的重排序未带来额外收益。

### 论文结构

Section 1: Introduction — 问题动机 + 贡献
Section 2: Related Work — APR, LLM-based repair, KG from code  
Section 3: System Architecture — THEMIS 的 4-Agent 流水线设计
Section 4: Experimental Setup — Dataset (SWE-bench Lite 300), Presets (8种), Metrics
Section 5: Results
  5.1 Overall Benchmark Performance (300 instances, 94% patch rate)
  5.2 Ablation Studies (6组消融)
  5.3 Semantic Contract Analysis (contract vs control vs rerank)
Section 6: Discussion — 发现解读、负结果价值
Section 7: Limitations — ARR 必需
Section 8: Conclusion

### 请执行

Phase 1: 评审上述结构和贡献，指出任何不可行之处
Phase 2: 生成完整的 claims-evidence 矩阵 (每个 claim 对应什么实验证据)
Phase 3: 搜索 Related Work (APR, LLM code repair, KG from source code, semantic-guided repair)。每次搜索后用 search 工具验证论文真实存在。不要编造引用。
```

**期望产出**: 
- `paper2/outline_and_claims.md` — 大纲 + claims-evidence 矩阵
- `paper2/related_work_notes.md` — 真实文献列表

---

### ═══════════════════════════════════════
### Stage 2: System Architecture 章节
### ═══════════════════════════════════════

**这是论文的核心创新章节，需要最仔细地写。**

**发送**:

```
请撰写 Section 3: System Architecture。

核心叙事:
传统的 APR 方法将 issue 当纯文本喂给 LLM。THEMIS 的不同之处在于:
1. 将 issue 分解为知识图谱中的结构化约束
2. 修复在图约束指导下进行
3. 验证基于图约束的一致性检查

请描述以下 4 个 Agent:

### Agent 1: Enhanced Graph Manager (KG 构建)
- Structural Extractor: AST 解析 → 代码结构图
- Semantic Injector: LLM 丰富图节点语义
- Dependency Tracer: 模块间/函数间依赖
- Violation Flagger: 需求-代码不一致检测

### Agent 2: Advanced Code Analyzer (语义分析)
- Bug Classifier: 问题分类 → 选择最优分析策略
- Concept Mapper: 代码结构 → 高层概念映射
- Context Enhancer: 丰富 LLM 上下文
- Multi-Round Reasoner: 迭代验证提高准确性

### Agent 3: Developer (代码修复)
- 在图约束指导下生成修复
- 最多重试 5 次修复语法错误
- 保留文件原始格式 (换行符风格等)

### Agent 4: Judge (验证)
- 硬检查: 图结构冲突数量 (violates_count, blocking_conflicts_count)
- 软检查: LLM 咨询 (advisory_conflicts_count)

### 请使用 ACL LaTeX 格式撰写

- \documentclass[11pt]{article}
- \usepackage[review]{acl}
- 双栏，11pt 正文
- natbib 引用 (\citep/\citet)

关键源文件参考:
- src/state.py (AgentState 定义)
- src/agents/developer.py
- src/agents/judge.py
- src/enhanced_graph_manager/ (各子模块)
- src/advanced_code_analysis/ (语义分析管道)
- PROJECT_FLOW_REFERENCE.rst (架构概览)

请生成可编译的 LaTeX 文件: paper2/sections/03_system_architecture.tex
```

**期望产出**: `paper2/sections/03_system_architecture.tex`

---

### ═══════════════════════════════════════
### Stage 3: Introduction + Related Work
### ═══════════════════════════════════════

**发送**:

```
请撰写 Section 1 (Introduction) 和 Section 2 (Related Work)。

### Introduction 结构:
Para 1: 代码修复的挑战 — NLP/SE 的背景
Para 2: 现有方法的局限 — 纯文本 vs 结构化需求
Para 3: 我们的方法概述 — KG 分解 + 4-Agent 流水线
Para 4: 贡献列表 (3条，编号)
Para 5: 论文结构预览

### Related Work 需覆盖:
- Automated Program Repair (APR): 传统方法和 LLM-based APR 的区别
- LLM-based Code Repair: 如何利用 LLM 进行代码修复
- Knowledge Graphs from Code: 从源代码构建 KG 的方法
- Semantic-Guided Repair: 语义指导在修复中的作用

每个子领域引用 3-5 篇真实论文。

请生成:
- paper2/sections/01_introduction.tex
- paper2/sections/02_related_work.tex
- paper2/references.bib (使用 ACL Anthology 中的真实 BibTeX)
```

---

### ═══════════════════════════════════════
### Stage 4: Experimental Setup + Results
### ═══════════════════════════════════════

**当前状态更新 (2026-05-16)**:
- 前 3 章已完成: `paper2/sections/01_introduction.tex`, `02_related_work.tex`, `03_system_architecture.tex`
- 参考文献已生成: `paper2/references.bib`
- Claims/Evidence 已更新: `paper2/outline_and_claims.md`
- 300-case 官方 harness resolved rate 已可报告: `58/300 = 19.3%`；completed/non-empty-patch denominator 为 `58/279 = 20.8%`
- `~94%` 只能作为 patch generation rate，不得写成 resolved/pass rate
- `chat_usage.total_tokens=0` 是统计限制，不得解释为 Developer/Judge 未调用 LLM

**发送**:

```
请基于 paper2/ 下的最新材料撰写 Section 4 和 Section 5。

必须读取并对齐以下文件:
- paper2/outline_and_claims.md
- paper2/gpt51_codexmini_experiments.md
- paper2/swebench_benchmark_architecture.md
- paper2/sections/01_introduction.tex
- paper2/sections/02_related_work.tex
- paper2/sections/03_system_architecture.tex
- paper2/references.bib

重要写作约束:
1. 明确区分 resolved rate 与 patch generation rate。
2. 当前 300-case 官方 harness 结果为 58/300 resolved = 19.3%；若使用 completed/non-empty-patch denominator，可补充 58/279 = 20.8%。
3. 94% 是非空 patch 生成率，不是 pass rate。
4. chat_usage.total_tokens=0 是统计/instrumentation limitation，不说明 Developer/Judge 没有调用。
5. 6 组消融目前只能按“ablation-labeled variants on a fixed 5-instance slice”谨慎表述；除非能从文件中明确变体含义，不要过度声称完全隔离组件贡献。
6. 语义契约结果应作为 bounded controlled finding：4-case slice 中 control 0/4, contract 1/4, rerank 1/4。

### Section 4: Experimental Setup
- Dataset: SWE-bench Lite test split, 300 instances, Python repositories
- Model: gpt-5.1-codex-mini for repair workflow and advanced analysis
- Workflow: integrated mode (Advanced Analysis → KG construction → Developer → Judge)
- File selection: conservative target-file selection, optional context/neighborhood/retrieval presets
- Experiment presets:
  - default / integrated + auto_select
  - graph_only ablation variants on fixed first-5 slice
  - fault-space and neighborhood/context/retrieval variants where documented
  - semantic contract/control/rerank variants
- Metrics:
  - official SWE-bench resolved rate: resolved/submitted
  - completed-only resolved rate: resolved/completed non-empty-patch instances
  - patch generation rate: non-empty git diff / submitted instances
  - runtime per instance
  - Advanced Analysis token usage
  - chat token accounting caveat: do not use chat_usage as reliable cost evidence
- Implementation: Python + LangGraph + NetworkX + SWE-bench harness + OpenAI-compatible model interface

### Section 5: Results
#### 5.1 Overall Benchmark
- 300 submitted instances
- Official resolved: 58/300 = 19.3%
- Completed/non-empty-patch denominator: 58/279 = 20.8%
- Patch generation: ~283/300 ≈ 94%
- Average runtime: ~523s / instance (8.7 min)
- Advanced Analysis tokens: ~28.3M total, ~94k / instance
- 覆盖仓库数/仓库分布: 使用 gpt51_codexmini_experiments.md 中的 repository table
- 明确说明: patch generation rate 与 resolved rate 是不同指标

#### 5.2 Ablation Studies
- 固定 first-5 slice 上的 6 组 ablation-labeled variants
- 报告各 variant 的 patch/output 差异与可验证指标
- 仅在变体定义清楚时讨论组件贡献；否则使用谨慎措辞: “probes selected design choices” 而不是 “fully isolates every component”
- 如数据不足，给出表格并把解释放在 Discussion，避免过度归因

#### 5.3 Semantic Contract Analysis
- 重点主张使用已执行且最稳妥的 4-case slice:
  - control: 0/4
  - semantic contract: 1/4
  - rerank: 1/4
- 可补充说明 paper2/gpt51_codexmini_experiments.md 中还记录了 semantic4/semantic5/semantic_work5 批次与输出文件
- 结论必须谨慎: semantic contracts show bounded positive evidence; rerank adds no observed benefit in the controlled slice

请生成:
- paper2/sections/04_experimental_setup.tex
- paper2/sections/05_results.tex

LaTeX 要求:
- 与前 3 章一致，生成 standalone 可编译 section draft（可含 \documentclass / \usepackage[review]{acl}）
- 使用 natbib 引用；只使用 paper2/references.bib 中已有 key，除非先验证并补充 BibTeX
```

---

### ═══════════════════════════════════════
### Stage 5: Discussion + Limitations + Conclusion
### ═══════════════════════════════════════

**发送**:

```
请撰写 Section 6 (Discussion), Section 7 (Limitations), Section 8 (Conclusion)。

必须基于最新结果:
- Official resolved rate: 58/300 = 19.3%
- Completed-only resolved rate: 58/279 = 20.8%
- Patch generation rate: ~94%
- Advanced Analysis tokens: ~28.3M total
- Semantic contract controlled finding: 0/4 → 1/4; rerank remains 1/4
- chat_usage.total_tokens=0 是统计限制，不是调用缺失

### Section 6: Discussion
讨论以下发现:
1. Requirement-aware decomposition 的意义 — 把 issue 从纯文本转换为可审计的 requirement-code constraints
2. 为什么 19.3% resolved 与 94% patch generation 同时存在 — patch 生成能力强，但正确修复仍受定位、语义理解、测试适配影响
3. 为什么语义契约指导可能有效 — 显式约束减少模型推理空间；但证据目前是小样本 bounded finding
4. 为什么 rerank 未能改善 — 更复杂 pipeline 不一定优于清晰的 semantic constraint；作为负结果报告
5. effective_rounds / revision behavior — 如果使用相关指标，必须从日志或 gpt51_codexmini_experiments.md 明确取证
6. 成本与可复现性 — Advanced Analysis token 可报告；Developer/Judge chat token 需要重新 instrumentation 才能用于成本结论

### Section 7: Limitations
ARR 必需。包括:
1. Single model limitation — 仅测试了 gpt-5.1-codex-mini
2. Python/SWE-bench Lite scope — 当前主评估集中在 SWE-bench Lite 的 Python 仓库
3. Bounded fault space / file selection — 默认保守目标文件选择，部分预设使用 neighborhood/context/retrieval 扩展；仍可能漏掉跨文件修复
4. Resolved rate 与 patch rate 的差距 — 94% patch generation 不代表 94% correctness；当前 official resolved 为 19.3%
5. Manual semantic contracts — 语义契约目前对困难实例需要人工策划，扩展性有限
6. Ablation scope — 6 组消融主要是 fixed first-5 slice，组件贡献结论需要谨慎
7. Token accounting limitation — chat_usage 统计路径未可靠捕获 Developer/Judge chat-model usage；总成本分析不完整
8. Requirement decomposition noise — issue 元数据/低信号词可能导致 requirement node 或 MAPS_TO edge 噪声

### Section 8: Conclusion
- 总结主要发现
- 未来工作方向
- 不要夸大为通用 APR SOTA；强调 requirement-aware KG decomposition + bounded empirical evidence

请生成:
- paper2/sections/06_discussion.tex
- paper2/sections/07_limitations.tex
- paper2/sections/08_conclusion.tex

LaTeX 要求:
- 与前文术语一致: THEMIS, requirement-aware, knowledge graph decomposition, semantic contracts
- Limitations 章节标题必须严格为 “Limitations” 以满足 ARR 要求
- 不新增未验证引用；需要新增引用时先验证并更新 paper2/references.bib
```

---

### ═══════════════════════════════════════
### Stage 6: Abstract + 组装 + 编译
### ═══════════════════════════════════════

**发送**:

```
请撰写 Abstract（200 词以内），然后组装完整论文。

### Abstract 要求:
- Line 1: 问题背景 (1句)
- Line 2-3: 现有方法的局限
- Line 4-5: 我们的方法 (KG 分解 + 4-Agent)
- Line 6-7: 主要结果 (300 instances, 58/300 = 19.3% resolved, ~94% patch generation, semantic contract 0/4→1/4, rerank no gain)
- Line 8: 结论

### 组装要求:
创建 paper2/main.tex，将以下文件合并。注意: 当前各 section draft 是 standalone LaTeX 文件，组装 main.tex 时应只抽取正文中的 \section... 内容，不能重复包含每个文件的 \documentclass、\usepackage、\begin{document}、\maketitle、\bibliography、\end{document}。

合并文件:
- sections/01_introduction.tex
- sections/02_related_work.tex
- sections/03_system_architecture.tex
- sections/04_experimental_setup.tex
- sections/05_results.tex
- sections/06_discussion.tex
- sections/07_limitations.tex
- sections/08_conclusion.tex

使用 ACL 格式:
\documentclass[11pt]{article}
\usepackage[review]{acl}
\title{...}
\author{...}

Bibliography:
- 使用 paper2/references.bib
- 使用 \bibliographystyle{acl_natbib}
- 使用 \bibliography{references}

### 验证:
编译 main.tex，确保:
- 无 LaTeX 错误
- 正文 ≤ 8 页
- Limitations 章节存在
- 引用格式正确 (natbib \citep/\citet)
- 所有 citation key 都在 paper2/references.bib 中存在
- Abstract ≤ 200 词
- 正文中不出现“94% resolved/pass rate”或“Developer/Judge LLM 未调用”这类错误表述
```

---

## 四、Agent 使用技巧

### 关键原则

1. **逐节撰写，不要一次全写** — 每节单独反馈迭代
2. **每节写完后要求编译** — 避免最后才发现 LaTeX 错误
3. **引用必须验证** — 每次看到新引用，要求 agent 提供 search 验证结果
4. **保留所有中间产物** — 方便回溯和复用

### 常用指令

| 场景 | 指令 |
|------|------|
| 某节太长 | "这节超过了 1.5 页，请精简到 1 页" |
| 引用可疑 | "请验证引用 [xxx] 是否真实存在，给出 DOI/URL" |
| 需要更多细节 | "请展开描述 X 的工作方式，加入技术细节" |
| AI 痕迹太重 | "这段读起来像 AI 写的，请改得更学术化、更 human" |
| 编译检查 | "请编译 main.tex 并报告是否有错误" |
| 格式合规 | "请对照 docs/ARR_EACL_SUBMISSION_GUIDE.md 检查格式是否合规" |

---

## 五、所需材料清单

| 文件 | 用途 | 位置 |
|------|------|------|
| 系统架构详解 | Stage 1-3 核心输入 | `paper2/swebench_benchmark_architecture.md` |
| 实验数据 | Stage 4-5 核心输入 | `paper2/gpt51_codexmini_experiments.md` |
| Claims/Evidence 矩阵 | Stage 4-8 约束 claims 边界 | `paper2/outline_and_claims.md` |
| Related Work 与 BibTeX | Stage 6 组装与引用检查 | `paper2/related_work_notes.md`, `paper2/references.bib` |
| 已完成章节 | Stage 4-8 术语与叙事保持一致 | `paper2/sections/01_introduction.tex`, `02_related_work.tex`, `03_system_architecture.tex` |
| 300-case eval summaries | Section 5 resolved rate 证据 | `gpt-5.1-codex-mini.cases*_eval.json` |
| ARR 格式指南 | Stage 6 验证 | `docs/ARR_EACL_SUBMISSION_GUIDE.md` |
| 项目架构 | AgentState、工作流 | `PROJECT_FLOW_REFERENCE.rst` |
| 项目概览 | 系统定位 | `README.md` |
| 现有论文参考 | 了解之前定位 | `paper/SUBMISSION_STATUS.md`, `paper/paper.md` |
| AGENTS.md | 项目规范 | `AGENTS.md` |
| 核心源码 | 技术细节 | `src/state.py`, `src/agents/`, `src/enhanced_graph_manager/` |

---

## 六、预计时间

| Stage | 内容 | 预计时间 |
|:---:|---|:---:|
| 0 | 环境准备 + 材料阅读 | 5 min |
| 1 | 大纲 + Claims-Evidence 矩阵 | 20 min |
| 2 | System Architecture (核心) | 30-45 min |
| 3 | Introduction + Related Work | 30 min |
| 4 | Experimental Setup + Results | 30-40 min |
| 5 | Discussion + Limitations + Conclusion | 25-30 min |
| 6 | Abstract + 组装 + 编译 | 20-30 min |
| **剩余** | Stage 4-6 | **~1.5 小时** |
| **总计** | Stage 0-6 | **~3 小时** |

---

## 七、快速启动 (复制即用)

重启 OpenCode 后，发送以下消息启动 Stage 0-1:

```
load skill academic-researcher

我要写一篇投稿 EACL (via ARR) 的 Long Paper，8 页正文 + unlimited references + limitations。

论文主线: 基于知识图谱的需求分解方法用于 LLM 驱动的代码修复。
验证支撑: SWE-bench Lite 300 实例 + 6 组消融实验 + 语义契约实验。

请先阅读以下文件:
1. README.md
2. PROJECT_FLOW_REFERENCE.rst  
3. paper2/swebench_benchmark_architecture.md
4. paper2/gpt51_codexmini_experiments.md
5. docs/ARR_EACL_SUBMISSION_GUIDE.md
6. paper/SUBMISSION_STATUS.md

读完告诉我你理解了什么，我来纠正和补充。
```
