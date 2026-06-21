# ACL/ARR 模拟审稿报告：THEMIS 论文初稿

审阅对象：`paper2/main.tex`  
审稿模式：`/ars-reviewer` full mode，多视角 ACL/ARR 模拟审稿  
语言：中文  
生成日期：2026-05-25

---

## 1. 总体元审稿结论

**综合建议：Weak Reject / Reject，建议 major revision 后重投。**

论文提出 THEMIS：一个面向 repository-level issue repair 的 requirement-aware code repair framework。核心思想是将自然语言 issue text 显式分解为 requirement nodes，并将这些需求与 code elements、violation/advisory edges 一起放入知识图谱，用于指导 LLM Developer 和 Judge。这个方向有潜在价值，尤其适合作为“可审计的中间表示 / inspectable interface”来讨论。

但以 ACL/ARR 标准看，当前稿件更像一篇 **软件工程/APR 系统经验报告或 preliminary study**，尚不足以支撑强方法论文接收。最核心问题不是写作质量，而是：

1. 缺少同设置强 baseline，无法判断 THEMIS 的真实增益；
2. ablation 定义不完整，不能支撑因果解释；
3. semantic-contract 证据只有 4 个 case，从 0/4 到 1/4，过小；
4. ACL/NLP framing 不够强，issue requirement decomposition 本身没有被作为语言理解问题充分评估；
5. SWE-bench 输入边界需要澄清，尤其是 failing test identifiers 是否构成 oracle leakage；
6. 关键模块可复现细节不足，包括 requirement extraction、MAPS_TO、violation flagging、prompt templates、token/cost accounting。

如果作为 ACL/ARR main research paper，目前倾向拒稿；如果改投 SE venue、workshop、Findings-style empirical/system paper，经过重定位和补实验后会更有机会。

---

## 2. 审稿人评分汇总

| 角色 | 主要关注点 | 总体倾向 | 核心理由 |
|---|---|---|---|
| Reviewer 1 | ACL fit、novelty、significance | Weak Reject / Reject | 论文更像 SE/APR 系统，NLP 贡献边界弱；缺少强对比证明 novelty 和 significance。 |
| Reviewer 2 | technical soundness、实验设计、baseline/ablation | Reject / Weak Reject | SWE-bench 输入边界、baseline 缺失、ablation 不可解释、semantic contract 不公平风险。 |
| Reviewer 3 | clarity、organization、related work、presentation | Weak Reject，约 3/5 | 写作清楚但证据链弱，图表和消融部分信息密度不足。 |
| Devil's Advocate | 最严厉拒稿风险 | Reject | 证据强度不足以支撑方法贡献；多个关键实验设计问题无法靠 rebuttal 解决。 |
| Area Chair / Editor | meta-review、ACL 适配、最终建议 | Weak Reject / Reject | 想法有潜力，但当前是 preliminary system report，需 major revision。 |

建议分项评分：

| 维度 | 建议分数 | 说明 |
|---|---:|---|
| Soundness | 2.0-2.5 / 5 | 主系统可理解，但实验设计和输入边界存在严重疑问。 |
| Excitement / Significance | 2.5-3.0 / 5 | requirement-aware graph idea 有潜力，但结果无法证明影响力。 |
| Reproducibility | 2.0-2.5 / 5 | 关键算法、prompt、配置、成本统计不足。 |
| Clarity | 3.0-3.5 / 5 | 文字清楚，限制诚实，但部分章节像工程日志整理。 |
| Overall | 2.0-3.0 / 5 | 当前不建议接收。 |

---

## 3. 论文摘要性总结

本文研究 LLM-based repository-level code repair。作者认为，现有 repair pipelines 往往把 issue report 作为 prompt text 直接输入模型，使得 requirement interpretation 隐含在模型内部、不可审计。THEMIS 试图将 issue text 分解为 explicit requirement nodes，并把需求与 code graph 中的 functions/classes/variables 建立 mapping 和 violation/advisory relations，再由 Advanced Code Analyzer、Developer agent、Judge agent 组成修复闭环。

论文在 SWE-bench Lite 300 个实例上报告：

- 58/300 resolved，即 19.3%；
- 279 个 completed non-empty-patch cases 上为 20.8%；
- 非空 patch generation rate 约 94%；
- Advanced Analysis tokens 约 28.3M；
- 平均运行时间约 523 秒/instance；
- 一个 4-case semantic-contract slice 中，context-only 为 0/4，semantic contract 为 1/4，rerank 仍为 1/4。

论文最大优点是诚实区分 patch generation 与 official resolved rate，并承认多个限制。但当前 empirical evidence 不足以证明 requirement-aware graph decomposition 本身有效。

---

## 4. 主要优点

### 4.1 问题动机合理

Repository-level issue repair 确实需要同时处理自然语言需求理解、代码定位、修复生成和测试验证。作者提出将 issue understanding 外显为 graph objects，这一动机合理。

### 4.2 系统设计有可审计性价值

THEMIS 将 requirement nodes、requirement-code edges、violation/advisory edges 作为中间表示，使失败案例可以从 missing mappings、wrong target symbol、low-confidence violation evidence 等角度分析。这比纯 prompt repair 更容易调试。

### 4.3 主结果覆盖完整 SWE-bench Lite

相比只报告 toy examples，300-case run 是一个有工程工作量的结果。即使 19.3% resolved rate 不一定强，它仍可作为系统 characterization 的基础。

### 4.4 结果呈现相对诚实

论文没有把 94% patch generation rate 误称为 repair success，而是明确区分 non-empty diff 与 official harness resolved rate。这一点对 APR/SWE-bench 论文很重要。

### 4.5 Limitations 写得充分

论文承认 single-model evaluation、Python/SWE-bench Lite scope、bounded fault space、manual semantic contracts、ablation scope、token accounting、requirement decomposition noise 等问题。语气比常见过度宣称稿件更谨慎。

---

## 5. 主要问题

### 5.1 ACL/ARR 适配性偏弱

论文虽然使用 ACL 模板，并涉及自然语言 issue understanding 和 LLM code repair，但核心任务、benchmark、指标和 related work 都更接近 software engineering / automated program repair。

当前论文没有充分证明其对 ACL 社区的核心贡献：

- 没有提出新的 NLP 模型或语言理解方法；
- 没有系统评估 requirement decomposition 本身的语言质量；
- 没有分析 issue text 中哪些语言现象影响 repair；
- 没有评估 requirement-code grounding 的 precision/recall；
- 没有对 LLM semantic guidance 的行为机制做深入分析。

如果投 ACL/ARR，需要将 framing 从“APR 系统”调整为：

> natural-language software issue understanding and requirement grounding for LLM-based code repair。

否则更适合 ICSE/FSE/ASE/ISSTA/SANER 或 SWE-agent 相关系统/工具方向。

### 5.2 缺少同设置强 baseline

这是最大拒稿风险。当前主结果只报告 THEMIS 自身在 300 个 SWE-bench Lite cases 上的 resolved rate。没有以下同模型、同预算、同文件选择、同 revision count 的比较：

- direct prompting；
- context-only prompting；
- graph-free variant；
- Advanced Analyzer only；
- graph-only；
- no semantic contract；
- Agentless/SWE-agent-style internal baseline。

因此 reviewer 无法判断 19.3% resolved rate 来自：

- requirement graph；
- file selection；
- failing test identifiers；
- prompt engineering；
- base model 能力；
- manual semantic contracts；
- 还是其他工程细节。

没有 baseline，论文只能说明“THEMIS 能跑出一些结果”，不能说明“THEMIS 的设计有效”。

### 5.3 SWE-bench 输入边界与 potential oracle leakage 风险

论文写道 requirement string 拼接了 problem statement、hints、failing test identifiers，以及 selected semantic-contract presets 中的 repair contracts。

这需要严肃澄清：

- failing test identifiers 是否来自 SWE-bench 官方公开输入？
- 是否来自 `FAIL_TO_PASS` / `PASS_TO_PASS` 等 evaluator metadata？
- baseline 是否也能使用同样信息？
- semantic contracts 是否由作者人工编写？
- contract 编写者是否看过 gold patch、测试、issue discussion 或失败日志？

如果 failing test identifiers 或 contracts 带有 evaluator oracle 信息，那么结果不能与常规 SWE-bench setting 公平比较。即使不违规，也必须在 experimental setup 中明确标注为 assisted setting / augmented input setting。

### 5.4 Ablation study 目前不能成立

论文中的 ablation-labeled variants 只有固定 5-instance slice，且多个名称为 `ablation13`、`ablation15`、`ablation456` 等内部标签。论文也承认这些 label 未充分定义。

这会让 reviewer 认为：

- 作者无法明确说明每个 variant 移除了什么；
- 实验无法隔离 component contribution；
- table 更像工程日志摘要，而不是 ablation study；
- 该部分反而削弱可信度。

建议：

1. 主文删除或降级这部分；
2. 不再称为 ablation；
3. 重新设计清晰 variant：NoGraph、NoAnalyzer、NoJudgeBrief、NoContract、Full；
4. 对每个 variant 给出输入、模型、文件选择、预算、revision count、prompt 差异；
5. 至少在 30-50 个 representative cases 上运行，否则只作为 appendix diagnostics。

### 5.5 Semantic-contract 证据过小且可能不公平

4-case slice 从 0/4 到 1/4 的结果太小，只能作为 anecdotal pilot evidence。它不能作为独立核心贡献。

主要问题：

- 样本只有 4 个；
- 不清楚 case selection 是否 cherry-picked；
- contracts 是 hand-crafted；
- 不清楚是否盲写；
- 不清楚是否看过 oracle 信息；
- reranking 结果没有提升，却未充分分析原因。

建议将 semantic-contract result 改写为 **pilot case study**，不要放在 contribution list 中作为主要贡献。若要保留为贡献，需要扩大样本并定义 contract creation protocol。

### 5.6 Requirement-aware graph 的作用没有被量化

论文核心 claim 是 graph 用于 localization、repair guidance 和 validation。但结果部分没有报告：

- requirement extraction precision/recall；
- requirement-code mapping accuracy；
- violation edge precision；
- graph conflict 是否与 resolved cases 相关；
- target hit rate 与 success 的关系；
- graph constraints 是否减少 irrelevant edits；
- Judge hard check 是否与 official tests 一致；
- false positive / false negative violation examples。

因此 reviewer 可能认为 graph 是叙事装置，而不是被证明有用的机制。

### 5.7 方法细节不足以复现

System Architecture 写得清楚，但偏高层说明。以下细节不足：

- sentence filtering rules；
- requirement indicators list；
- term matching / MAPS_TO edge algorithm；
- confidence score 计算；
- severity/blocking flag 判定；
- Developer prompt template；
- Judge hard/soft check prompt；
- edit modes 的 exact fallback order；
- file selection top-k 与 ranking details；
- revision budget 与 stopping policy；
- failure handling；
- exact command lines；
- artifact release plan。

建议新增 algorithm boxes 和 appendix，至少提供伪代码与关键 prompt skeleton。

### 5.8 成本与实用性不可解释

论文报告 Advanced Analysis 约 28.3M tokens 和 523 秒/instance，但也承认 Developer/Judge chat token accounting 不可靠，记录为 0 是 instrumentation limitation。

这带来两个问题：

1. 无法评价系统总成本；
2. 523 秒/instance 加 94k tracked tokens/instance 可能让系统看起来昂贵。

如果论文主张工程实用性或 scalable repair，需要更完整成本统计；如果无法统计，则应明确只报告 partial cost，并避免任何 cost-effectiveness claim。

---

## 6. 次要问题与呈现建议

1. **标题可能过强。** “Semantic Contract Guidance” 让人期待自动 contract induction 或大规模 contract evaluation，但当前只有 4-case hand-crafted slice。建议标题中弱化或在摘要中明确 pilot status。

2. **Architecture figure 过于文本化。** 当前图像更像 boxed text pipeline。建议改为正式 dataflow diagram，显示 inputs、state、graph、LLM calls、outputs、feedback loop。

3. **Results 中 batch-level table 信息价值有限。** cases 1-100、101-140 等 batch breakdown 不如 repo-level、failure-type、component-level breakdown 有意义。可移至 appendix。

4. **Related Work 需要更贴近当前系统。** 应补充和 SWE-agent、Agentless、AutoCodeRover、OpenHands、Aider、SWE-bench leaderboard 的关系。当前 related work 覆盖面可以，但对比不够锋利。

5. **重复内容略多。** Patch generation vs resolved rate 在 abstract、results、discussion、limitations、conclusion 多次出现。可以保留核心一次，其余压缩。

6. **分章文件是 standalone document。** `sections/*.tex` 各自包含 `\documentclass`、`\begin{document}` 等，这适合单独草稿，但若作为主文 source organization，后续需要改成可 `\input{}` 的纯 section 文件，避免维护重复文本。

---

## 7. Reviewer Questions for Authors

1. Failing test identifiers 是否是 SWE-bench 官方 agent-visible input？如果不是，为什么可以放入 requirement string？
2. Semantic contracts 是如何写成的？作者是否看过 gold patch、tests、issue discussions 或前一轮失败日志？
3. 58/300 的结果与 direct prompt baseline 相比如何？
4. 如果只使用 selected files + issue prompt，不使用 graph，resolved rate 是多少？
5. Graph 的 MAPS_TO 和 VIOLATES edges 准确率如何？是否有人类标注评估？
6. Judge 的 blocking conflicts 与 official SWE-bench tests 的一致性如何？
7. 为什么 revision count 设为 1？更深 loop 是否改善结果？
8. ablation13、ablation15、ablation456 分别代表什么具体组件差异？
9. Developer/Judge 的 token usage 为什么无法统计？总成本如何估计？
10. 论文的 ACL contribution 是 issue language understanding、LLM repair behavior analysis，还是 SE repair system？

---

## 8. Rebuttal 可解决与不可解决的问题

### Rebuttal 可以部分解决

- 澄清 ACL framing；
- 弱化 claims；
- 解释 failing test identifiers 的来源；
- 明确 semantic contracts 的生成流程；
- 补充少量实现细节；
- 承诺 artifact release；
- 说明为何不与 closed-source leaderboard 严格比较。

### Rebuttal 难以解决

- 缺少同设置强 baseline；
- ablation 不完整；
- semantic-contract 样本过小；
- requirement graph 作用未量化；
- ACL/NLP contribution 需要重写 framing 与新增分析。

因此，当前版本不太可能通过 rebuttal 从 reject 转为 accept。需要实质性修改与新增实验。

---

## 9. 建议的 Major Revision 路线图

### P0：必须完成

1. **补 baseline**
   - Direct prompt；
   - Context-only；
   - Graph-only；
   - Full THEMIS；
   - 最好同模型、同预算、同文件选择、同 revision count。

2. **澄清 SWE-bench input protocol**
   - 明确 failing test identifiers 是否允许；
   - 若属于 augmented setting，必须标注；
   - baseline 必须使用同样输入。

3. **重写 ablation**
   - 删除内部标签；
   - 清晰定义每个 variant；
   - 报告 resolved rate、empty patch rate、failure categories。

4. **收窄 claims**
   - 不宣称 broad repair improvement；
   - 主张改为 inspectable requirement-aware interface；
   - semantic-contract 改为 pilot case study。

### P1：强烈建议完成

1. 增加 requirement decomposition quality evaluation；
2. 增加 graph edge usefulness analysis；
3. 增加 failure taxonomy；
4. 增加 running example；
5. 增加 algorithm boxes；
6. 补充 prompt templates 与 exact configs；
7. 给出 artifact/reproducibility statement。

### P2：提升投稿质量

1. 改 architecture figure；
2. related work 增加 comparison table；
3. 压缩重复 discussion；
4. 将 batch-level table 移至 appendix；
5. 检查所有 citation 是否为 peer-reviewed 或明确标注 arXiv/preprint；
6. 根据 ACL Responsible NLP Checklist 准备 ethics/reproducibility 说明。

---

## 10. 建议的新论文定位

当前版本如果继续以 “Requirement-Aware Code Repair via Knowledge Graph Decomposition and Semantic Contract Guidance” 为方法性能论文，证据不足。

更安全的定位是：

> THEMIS studies requirement-aware graph decomposition as an inspectable interface for repository-level LLM repair, and provides an empirical characterization of where such interfaces help or fail on SWE-bench Lite.

围绕这个定位，论文可以强调：

1. 自然语言 issue 到 structured requirements 的外显化；
2. requirement-code grounding 作为 LLM repair 的可审计接口；
3. patch generation 与 correctness 的 gap；
4. graph-guided repair 的 failure modes；
5. semantic contracts 作为 future direction，而非已充分证明的核心方法。

---

## 11. 最终建议

**当前决定：Weak Reject / Reject。**

这篇论文有一个值得继续发展的核心想法，但当前版本不应以强方法论文形式提交 ACL/ARR。建议先完成同设置 baseline、重新设计 ablation、澄清 SWE-bench 输入边界、补充 requirement grounding 分析，并将贡献重定位为可审计中间表示与系统性经验分析。完成这些修改后，论文会更有机会进入 ACL Findings、workshop、或 SE/APR 方向会议。
