# THEMIS 论文初稿审稿式问题分析与修改建议

生成日期：2026-05-24  
审阅范围：`paper2/main.tex`、`paper2/sections/*.tex`、`paper2/references.bib`  
建议使用的 skill：`academic-researcher`

## 1. 最适合当前任务的 skill 筛选结论

当前任务最适合使用 **`academic-researcher`**。

理由：

- 该任务本质是学术论文优化，而不是代码实现或普通文档润色。
- 需要从同行评审视角检查论文结构、贡献表述、实验设计、引用质量、claim/evidence 对齐、可复现性与投稿风险。
- 当前论文使用 ACL/ARR 风格模板，并涉及 SWE-bench、APR、LLM-based repair、code graph 等研究主题，正好需要该 skill 的学术写作、文献与实验严谨性检查能力。
- 其他可用 skill 中，`context-optimization` 只适合上下文受限时辅助检索；`review-work` 更偏代码实现后验证；`ai-slop-remover` 面向单文件代码气味；`frontend-ui-ux`、`playwright`、`git-master` 与本论文审稿任务不匹配。

**结论**：后续论文优化应以 `academic-researcher` 为主，必要时辅以 `context-optimization` 做长稿检索，但不应替代学术审稿标准。

## 2. 总体审稿判断

当前稿件的核心想法是清楚的：THEMIS 将自然语言 issue 分解为 requirement nodes，并把需求、代码元素、violation/advisory edges 放入知识图谱，用于指导 LLM Developer 和 Judge。这个定位有潜在价值，尤其适合作为“可审计的中间表示 / inspectable interface”来讨论。

但是，以严格 reviewer 视角看，当前论文存在一个主要矛盾：**系统描述较完整，但实验证据不足以支撑强方法贡献**。主实验只报告 THEMIS 自身在 300 个 SWE-bench Lite case 上的结果，缺少同设置强基线；semantic-contract 结果只有 4 个 case；ablation labels 未充分定义。因此论文目前更像一篇诚实的系统经验报告，而不是可以强力宣称方法优越性的完整实证论文。

建议将论文主张从“THEMIS 显著提升 repair performance”收窄为：

> THEMIS provides an inspectable requirement-aware interface for repository-level LLM repair, and preliminary experiments show both its promise and its current bottlenecks.

## 3. 主要拒稿风险与优先级

### Risk 1：缺少强 baseline，导致贡献有效性难以判断

当前结果报告 THEMIS 在 SWE-bench Lite 上 58/300 resolved，即 19.3%。但读者无法判断这个数值相对 direct prompting、Agentless、SWE-agent、无图版本、无 Advanced Analyzer 版本是否更好。

**问题影响**：这是最大拒稿风险。没有 baseline，reviewer 很可能认为论文只是报告了一个系统跑分，而不是证明 requirement-aware graph decomposition 有用。

**修改建议**：

1. 增加同一模型、同一预算、同一 harness 下的内部 baseline：
   - Direct issue prompt + selected files
   - Context-only prompt
   - Graph-only without Advanced Analyzer
   - Advanced Analyzer only without graph constraints
   - Full THEMIS
2. 如果资源有限，至少在一个固定代表性 slice 上做完整内部比较，并明确 slice 选择标准。
3. 把公开系统如 SWE-agent、Agentless、SWE-bench leaderboard 作为“not directly comparable reference”，但不要混作严格 baseline。

### Risk 2：ablation 不完整且标签未定义

论文目前承认 ablation-labeled variants 有些“documented by name rather than fully specified component removal”。这种写法诚实，但会直接削弱实验可信度。

**问题影响**：reviewer 会质疑作者是否知道每个 ablation 具体移除了什么组件，也会怀疑是否存在 cherry-picking。

**修改建议**：

1. 重新命名 ablation，避免 `ablation13`、`ablation456` 这类内部工程标签。
2. 对每个 ablation 给出明确表格：移除/保留哪些组件、输入是否相同、模型是否相同、revision budget 是否相同、file selection 是否相同。
3. 不要把未定义 variant 放入主结果表。可以移到 appendix 或完全删除。
4. 主文只保留能支撑因果解释的 ablation。

### Risk 3：semantic-contract 实验证据太小

当前 semantic-contract slice 从 0/4 提升到 1/4。这个结果可以作为 qualitative evidence 或 pilot study，但不适合作为核心贡献之一。

**问题影响**：如果把 4-case improvement 写进贡献列表，reviewer 会认为证据过弱。

**修改建议**：

1. 将 semantic-contract 从“贡献”降级为“controlled pilot analysis”。
2. 扩大样本规模，至少覆盖 20-30 个明确标注的 case；如无法扩大，则只作为 case study。
3. 补充每个 case 的简短错误分析：contract 帮在哪里、为什么 rerank 没提升、失败 case 的原因是什么。
4. 明确 semantic contracts 是否人工编写、是否可自动生成、是否引入额外人工知识。

### Risk 4：claim 与 evidence 不完全对齐

引言和结论说结果支持 explicit requirement decomposition 是有用接口；这个结论可以成立，但必须非常谨慎。当前 evidence 更能支持“可检查、可调试、有工程价值”，而不是“带来显著修复性能提升”。

**修改建议**：

1. 在 abstract、introduction、conclusion 中统一使用 bounded language：
   - “provide preliminary evidence”
   - “support an inspectable interface”
   - “suggest potential utility”
   - 避免 “improves repair” 这类泛化表达。
2. 明确 19.3% 是当前系统配置结果，不证明 graph decomposition 本身优于其他方法。
3. 将“performance claim”和“diagnostic/inspectability claim”分开。

### Risk 5：可复现细节不足

Experimental Setup 描述了 dataset、model、file selection、presets、metrics，但关键可复现信息仍不足。

**需要补充**：

- target files 如何选择，具体 top-k、过滤规则、fallback 条件是什么。
- requirement sentence extraction 的规则、阈值、stopwords 或 indicator list。
- requirement-code MAPS_TO edge 的匹配算法。
- violation flagger 如何决定 SATISFIES / VIOLATES / ADVISORY。
- confidence、severity、blocking flag 的计算方式。
- Developer prompt 的核心结构。
- Judge hard check 与 soft check 的伪代码。
- 最大 revision count 为 1 的动机与影响。
- semantic contracts 的来源与人工参与程度。
- 失败、空 patch、非空 patch、completed 的准确定义。

建议新增一个 **Algorithm / Implementation Details** 小节，或者将细节放入 appendix。

## 4. 分章节修改建议

### Abstract

**当前优点**：摘要诚实报告 58/300、19.3%、约 94% patch generation、4-case semantic-contract result。  
**主要问题**：缺少 baseline，导致读者看到 19.3% 后难以判断价值；4-case result 放在摘要里显得证据过小。

**建议**：

- 将摘要中心从“性能”改为“inspectable requirement decomposition interface”。
- 保留 300-case result，但明确它是 bounded empirical characterization。
- 对 4-case semantic-contract 结果使用更弱表述，如 “pilot slice”。

### Introduction

**当前优点**：动机自然，问题定义合理，贡献列表清楚。  
**主要问题**：第三个 contribution 基于 4 个 case，分量不足；与 prompt engineering / agent feedback 的差异还可以更早讲清楚。

**建议**：

- 将贡献列表改为：
  1. Requirement-aware graph representation for issue-level repair。
  2. Integrated repair loop with graph-guided Developer/Judge。
  3. Empirical characterization on SWE-bench Lite, including bottleneck analysis。
- 将 semantic-contract result 放入第三点的子句，而不是独立贡献。

### Related Work

**当前优点**：覆盖 APR、LLM repair、code graphs、semantic-guided repair。  
**主要问题**：目前偏并列综述，缺少对比矩阵，没充分说明 THEMIS 与现有 agent repair / graph code representation 的具体差异。

**建议**：

- 增加一个 comparison paragraph 或 table，列出：
  - 是否显式建模 issue requirements
  - 是否将 requirement 与 code nodes 连接
  - 是否产生 violation/advisory edges
  - 是否用同一表示指导 repair 与 validation
- 强化与 SWE-agent、Agentless、ChatRepair、code property graph 类工作的边界。

### System Architecture

**当前优点**：系统流程完整，四个 agent 描述清楚。  
**主要问题**：更像工程说明，缺少形式化定义与算法细节。图是文本框图，说服力有限。

**建议**：

- 加入 formal graph definition：`G=(V_code, V_req, E_dep, E_map, E_violation)`。
- 用 Algorithm 1 描述 integrated workflow。
- 用 Algorithm 2 或伪代码描述 requirement extraction / violation flagging。
- 将当前文字框图改成更正式的 pipeline figure。
- 给出一个 running example：从 issue sentence 到 requirement node，再到 MAPS_TO edge、VIOLATES edge，最后影响 Developer prompt。

### Experimental Setup

**当前优点**：覆盖 dataset、model、file selection、presets、metrics。  
**主要问题**：复现细节不足，特别是 file selection、semantic contracts、allowed inputs。

**建议**：

- 明确 failing test identifiers 是否是 SWE-bench task 中允许使用的信息。
- 说明 hints、problem statement、test metadata 如何组合为 requirement string。
- 给出每个 preset 的确切配置。
- 对人工设置的 semantic contracts 单独标注为 human-provided oracle-like guidance 或 human-authored constraints。

### Results

**当前优点**：诚实区分 resolved rate 与 patch generation rate，这是重要优点。  
**主要问题**：没有 baseline；batch-level table 信息价值有限；ablation table 不能支撑因果结论。

**建议**：

- 优先增加 baseline table。
- 将 batch-level table 移到 appendix，主文只保留整体结果与关键 breakdown。
- 增加 failure category table，例如：localization failure、wrong semantic inference、syntax/edit failure、test mismatch、overfitting/underfitting。
- 增加 resolved/unresolved 的 qualitative examples。

### Discussion

**当前优点**：语气谨慎，能承认 patch generation 与 correctness gap。  
**主要问题**：部分内容重复 Results 和 Limitations；缺少深入 failure mode 分析。

**建议**：

- 用 2-3 个具体 failure modes 组织 discussion。
- 讨论 requirement decomposition 什么时候有帮助、什么时候有噪声。
- 解释 Judge graph checks 与 official SWE-bench tests 不一致的原因。

### Limitations

**当前优点**：限制写得全面且诚实。  
**主要问题**：限制太多且有些属于实验设计缺口，会让 reviewer 认为工作尚未完成。

**建议**：

- 保留关键 limitations：single model、Python/SWE-bench scope、manual contracts、incomplete ablation、token accounting。
- 对可以补救的内容尽量在实验部分解决，而不是只放在 limitations。

### Conclusion

**当前优点**：结论克制，没有过度宣称 broad superiority。  
**主要问题**：仍需与全文最终定位完全一致。

**建议**：

- 强调“inspectable interface”和“empirical characterization”。
- 避免读者误解为已证明方法相对 baseline 有显著提升。

## 5. 论文定位建议

### 如果目标是 ACL/ARR

需要强化 NLP/LLM 角度：

- requirement extraction 是否是语言理解问题？
- issue text decomposition 相比普通 prompting 有何语言层面创新？
- LLM reasoning、semantic contracts、natural-language constraints 的贡献在哪里？
- 需要更明确说明为什么这篇不是纯 SE system paper。

### 如果目标是 ICSE/FSE/ASE 等 SE venue

需要强化软件工程实验严谨性：

- 同设置 baseline。
- 完整 ablation。
- 可复现配置。
- failure analysis。
- 与 APR/agent repair systems 的公平比较。

当前稿件更自然适合 SE / APR / software maintenance 方向；若坚持 ACL/ARR，应重写 framing，让自然语言 requirement decomposition 成为中心方法贡献。

## 6. 推荐修改优先级

### P0：必须先改

1. 增加至少一个同模型 direct/context-only baseline。
2. 重新定义并清理 ablation variants。
3. 收窄 abstract、introduction、conclusion 中的方法有效性主张。
4. 明确 semantic contracts 的来源与实验边界。

### P1：强烈建议

1. 增加 architecture formalization 和算法伪代码。
2. 增加 one running example。
3. 增加 failure analysis table。
4. 补充 reproducibility details。

### P2：进一步提升质量

1. 增加 related work comparison table。
2. 改进 architecture figure。
3. 将 batch-level 和 under-specified runs 移入 appendix。
4. 检查 references.bib 中所有 citation 是否真实、完整、格式一致。

## 7. 建议的重写主线

建议全文采用如下 narrative：

1. Repository-level issue repair 的难点不只是生成代码，而是将自然语言需求、代码定位、修复约束和验证信号连接起来。
2. 现有 LLM repair pipeline 多数把 issue 当作 prompt text，缺少可审计的中间表示。
3. THEMIS 提出 requirement-aware graph，把 issue-derived requirements 显式连接到 code elements，并用 violation/advisory edges 指导 Developer 和 Judge。
4. 在 SWE-bench Lite 上，THEMIS 的结果显示该表示可以稳定产生候选 patch，但 correctness 仍明显受 localization、semantic mapping、bounded repair loop 限制。
5. 因此本文贡献不是证明 THEMIS 已优于所有系统，而是提出并实证分析一种可审计的 requirement-aware repair interface，为后续更强 repair loops 提供基础。

## 8. 可直接执行的修改清单

- [ ] 将 contribution 3 从 semantic-contract improvement 改为 empirical characterization。
- [ ] 新增 baseline experiment 或至少补充固定 slice baseline。
- [ ] 重命名 ablation variants，并给出精确定义。
- [ ] 为 graph schema 增加形式化定义。
- [ ] 增加 integrated workflow 伪代码。
- [ ] 增加 running example。
- [ ] 增加 failure taxonomy。
- [ ] 将 semantic-contract result 降级为 pilot/case study。
- [ ] 强化 related work 中与 SWE-agent、Agentless、ChatRepair、code property graph 的差异。
- [ ] 重写 abstract 和 conclusion，使主张与证据强度一致。
