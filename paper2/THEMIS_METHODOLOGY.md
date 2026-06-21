# THEMIS 核心方法论完整文档

> 目标: 为论文写作提供完整的方法论描述  
> 覆盖: 从 issue 输入到 patch 输出的完整流水线  
> 基于: 源码分析 (`src/` 目录下所有核心模块)  
> 最后更新: 2026-05-16

---

## 目录

1. [系统总览: 从 Issue 到 Patch](#1-系统总览-从-issue-到-patch)
2. [阶段一: 需求分解与知识图谱构建](#2-阶段一-需求分解与知识图谱构建)
3. [阶段二: 语义分析与问题分类](#3-阶段二-语义分析与问题分类)
4. [阶段三: 图指导下的代码修复](#4-阶段三-图指导下的代码修复)
5. [阶段四: 双层验证机制](#5-阶段四-双层验证机制)
6. [核心技术贡献总结](#6-核心技术贡献总结)

---

## 1. 系统总览: 从 Issue 到 Patch

### 1.1 核心思想

传统的 LLM-based program repair 将 issue 描述作为纯文本直接喂给 LLM，要求模型同时理解需求、定位代码、生成修复。THEMIS 的不同之处在于：

> **将 issue 文本显式分解为知识图谱中的结构化约束，然后在图约束空间中指导修复和验证。**

这带来了三个关键优势：
1. **需求可验证**: 每条需求被分解为图谱中的独立节点，可单独检查是否满足
2. **关系可追溯**: 需求与代码之间的关系以图边的形式显式表达
3. **修复可解释**: 修复过程被限定在图约束空间内，每一步都有明确的语义依据

### 1.2 四阶段流水线

```
Input: Issue 文本 + 目标代码文件
         │
    ┌────▼─────────────────────────────────────┐
    │  Phase 1: 需求分解与 KG 构建              │
    │  ┌──────────┐   ┌──────────┐   ┌───────┐ │
    │  │Semantic  │   │Structural│   │Dep.    │ │
    │  │Injector  │   │Extractor │   │Tracer  │ │
    │  └────┬─────┘   └────┬─────┘   └───┬───┘ │
    │       └──────────────┼──────────────┘     │
    │                      ▼                    │
    │              Knowledge Graph              │
    └──────────────────────┬───────────────────┘
                           ▼
    ┌──────────────────────┴───────────────────┐
    │  Phase 2: 语义分析与问题分类               │
    │  ┌──────────┐ ┌────────┐ ┌────────────┐  │
    │  │Bug       │ │Concept │ │Multi-Round │  │
    │  │Classifier│ │Mapper  │ │Reasoner    │  │
    │  └──────────┘ └────────┘ └────────────┘  │
    │  → 策略选择 → 修复假设 → 置信度评估        │
    └──────────────────────┬───────────────────┘
                           ▼
    ┌──────────────────────┴───────────────────┐
    │  Phase 3: 图指导下的代码修复               │
    │  ┌────────────────────────────────────┐  │
    │  │        Developer Agent             │  │
    │  │  KG约束 + 需求文本 + 冲突报告 → 修复   │  │
    │  │  支持: SymbolRewrite + SearchReplace │  │
    │  └────────────────────────────────────┘  │
    └──────────────────────┬───────────────────┘
                           ▼
    ┌──────────────────────┴───────────────────┐
    │  Phase 4: 双层验证                       │
    │  ┌──────────────┐  ┌──────────────┐     │
    │  │  Hard Check  │  │  Soft Check  │     │
    │  │  (图冲突检测)  │  │  (LLM咨询)   │     │
    │  └──────────────┘  └──────────────┘     │
    │  → 冲突报告 → 如有冲突则循环 Phase 3     │
    └──────────────────────┬───────────────────┘
                           ▼
Output: 修改后的代码 (Patch)
```

### 1.3 核心状态模型

```python
AgentState = {
    "files": {},               # 文件路径 → 内容 (可编辑的 source of truth)
    "requirements": "",        # issue 文本 (自然语言)
    "knowledge_graph": DiGraph, # 当前代码的 KG (动态更新)
    "baseline_graph": DiGraph,  # 原始代码的 KG (不变基准)
    "conflict_report": "",     # Judge 输出的冲突报告
    "revision_count": 0,       # 修复循环计数
    "analysis_report": {},     # 结构化分析报告 (含 violation/dependency 统计)
    "advanced_analysis": {},   # LLM 语义分析结果
    "developer_metrics_history": [],  # Developer 每轮指标
    "conflict_metrics_history": [],   # Judge 每轮指标
}
```

---

## 2. 阶段一: 需求分解与知识图谱构建

此阶段由 **Enhanced Graph Manager** 执行，包含三个子组件。

### 2.1 Semantic Injector: 需求分解

**输入**: 自然语言的 issue 文本  
**输出**: `RequirementNode` 列表 + Requirement-Code 映射边

**算法流程**:

1. **句子分割** (`_split_into_sentences`): 将 issue 文本按句号、换行等分割为独立句子
2. **需求句子识别** (`_is_requirement_sentence`): 识别包含需求指示词的句子（如 "should", "must", "need to", "bug", "error", "fix", "expected" 等）
3. **元数据过滤** (`_is_metadata_sentence`): 剔除环境信息、版本号等非需求句子
4. **原子需求创建** (`_create_requirement_node`):
   - 分配唯一 REQ-ID (REQ-001, REQ-002, ...)
   - 提取优先级 (基于关键词: "critical"=1, "must"=2, "should"=3, 默认=5)
   - 标记可测试性 (是否包含可验证的行为描述)
5. **需求-代码映射** (`_find_relevant_code_nodes`):
   - 将需求文本中的关键词与图节点名进行匹配
   - 按相关性评分排序 (直接匹配 > 部分匹配 > 语义相关)
   - 生成 `DependencyEdge` (type=MAPS_TO)

**数据模型**:
```python
RequirementNode(id="REQ-001", text="...", priority=3, testable=True)
DependencyEdge(source="REQ-001", target="function_name", 
               dependency_type="MAPS_TO", context="...")
```

### 2.2 Structural Extractor: 代码结构提取

**输入**: Python 源代码  
**输出**: 包含 `FunctionNode`, `ClassNode`, `VariableNode` 和 `CallEdge` 的有向图

**算法流程**:

1. **AST 解析**: 使用 `ast.parse()` 解析 Python 代码
2. **函数提取**:
   - 提取函数名、参数列表、返回类型注解、docstring、行号
   - 生成 `FunctionNode`
3. **类提取**:
   - 提取类名、基类列表、方法列表、docstring、行号
   - 生成 `ClassNode`
4. **成员变量提取**:
   - 检测 `self.xxx` 赋值语句
   - 推断变量类型（基于赋值右侧表达式）
   - 生成 `VariableNode`
5. **调用关系提取**:
   - 遍历 AST 中的 `Call` 节点
   - 建立 caller → callee 的 `CallEdge`

**图节点类型**:
| 节点类型 | 属性 | 用途 |
|---------|------|------|
| `FunctionNode` | name, args, return_type, docstring, line_number | 函数单元 |
| `ClassNode` | name, bases, methods, docstring, line_number | 类单元 |
| `VariableNode` | name, var_type, defined_in, line_number | 成员变量 |
| `RequirementNode` | id, text, priority, testable | 需求单元 |

**图边类型**:
| 边类型 | 属性 | 含义 |
|--------|------|------|
| `CallEdge` | caller, callee, line_number | 函数调用关系 |
| `DependencyEdge` | source, target, dependency_type, context | 通用依赖 (DEPENDS_ON, USES_VAR, DEFINED_IN, MAPS_TO) |
| `ViolationEdge` | requirement, code_node, status, reason, confidence, blocking | 需求满足/违反关系 |

### 2.3 Dependency Tracer: 依赖追踪

**输入**: 包含函数/类节点的图  
**输出**: 增强后的图（含依赖关系和传递闭包）

**核心功能**:
- **显式依赖**: 基于 `CallEdge` 建立 DEPENDS_ON 关系
- **成员变量使用**: 检测 `self.xxx` 访问，建立 USES_VAR 关系
- **定义位置**: 函数/方法定义的类归属，建立 DEFINED_IN 关系
- **传递闭包**: 通过 BFS/DFS 计算间接可达节点（`transitive_dependencies`）

### 2.4 Violation Flagger: 需求违反检测

**输入**: 含 RequirementNode 和 DependencyEdge 的图  
**输出**: ViolationEdge 列表 + ViolationReport（含严重程度、置信度、阻止标记）

**算法流程**:

1. **需求-代码关联分析** (`analyze_requirement_satisfaction`):
   - 遍历每个 RequirementNode
   - 检查其 MAPS_TO 边的目标节点是否存在于代码图中
   - 分配状态: SATISFIES / VIOLATES / ADVISORY

2. **违反检测规则**:
   - **缺失检测**: 需求要求的函数在代码中不存在 → VIOLATES
   - **证据评分**: 基于关键词匹配计算 `evidence_score` (0.0~1.0)
   - **标识信号过滤** (`low_signal_tokens`): 排除过于通用的标识符（如 "set", "get", "type" 等）以避免误报

3. **严重程度分配**:
   - severity=1: blocking=True 的 VIOLATES
   - severity=2-3: 高置信度 VIOLATES
   - severity=4-5: 低置信度 VIOLATES / ADVISORY

4. **通用原因模式识别** (`generic_reason_patterns`):
   - 识别如 "no clear relationship found" 等模糊报告，标记为低置信度

---

## 3. 阶段二: 语义分析与问题分类

此阶段由 **Advanced Code Analyzer** 执行。该子系统在集成工作流中的 `advanced_analysis` 节点运行，为后续的 KG 分析和代码修复提供语义理解。

### 3.1 Bug Classifier: 智能问题分类

**目的**: 将 issue 自动分类为 8 种预定义的 Bug 类别，选择最优分析策略。

**8 种 Bug 类别** (`BugCategory`):
```
LOGIC_ERROR        — 逻辑错误、条件错误、算法缺陷
API_ISSUE          — API 使用问题、参数错误
PERFORMANCE        — 性能问题、内存泄漏
BOUNDARY_CONDITION — 边界条件、空值检查、输入验证
TYPE_ERROR         — 类型不匹配、转换问题
CONCURRENCY        — 竞态条件、死锁
RESOURCE_MANAGEMENT — 文件句柄、连接管理
CONFIGURATION      — 配置、环境变量问题
```

**算法流程**:
1. **LLM 分类**: 使用专门设计的 Prompt Template，要求 LLM 根据 issue 描述选择最匹配的类别
2. **置信度评分**: 输出分类置信度 (0.0~1.0)
3. **备选类别**: 同时提供 2-3 个备选分类及其置信度
4. **策略匹配**: 根据 BugType 从 `PromptTemplateLibrary` 中选择对应的 `AnalysisStrategy`（含特定 prompt、上下文需求、验证步骤）

### 3.2 Concept Mapper: 概念映射

**目的**: 将代码中的具体实现映射到高层语义概念。

- 解析代码中的类名、函数名、变量名
- 映射到领域概念（如 `SetPasswordForm` → "认证表单验证"）
- 建立概念层级关系

### 3.3 Context Enhancer: 上下文增强

**目的**: 为 LLM 提供更有针对性的代码上下文。

- **DependencyContext**: 提取函数签名、类方法、import 语句、调用图
- **DomainKnowledge**: 注入领域术语、常见模式、最佳实践、反模式
- **ContextWindow**: 管理上下文大小（默认 8000 tokens）

### 3.4 Multi-Round Reasoner: 多轮推理

**目的**: 通过迭代验证提高分析准确性。

- 最多 N 轮推理（默认 3 轮）
- 每轮输出置信度，当超过阈值（默认 0.8）时提前终止
- 使用 `ReasoningChain` 跟踪每轮的推理过程

### 3.5 辅助组件

| 组件 | 功能 |
|------|------|
| **Conflict Detector** | 检测代码变更与需求之间的语义冲突 |
| **Semantic Extractor** | 从代码中提取语义信息（意图、约束、前置条件） |
| **Pattern Matcher** | 匹配历史成功案例中的修复模式 |
| **Result Sorter** | 按置信度、严重程度排序分析结果 |

### 3.6 分析策略枚举

```python
AnalysisStrategy:
  AUTO_SELECT   — 自动选择（默认）
  ADVANCED_ONLY — 仅 LLM 语义分析
  GRAPH_ONLY    — 仅图结构分析
  INTEGRATED    — 两者结合
```

---

## 4. 阶段三: 图指导下的代码修复

此阶段由 **Developer Agent** 执行。

### 4.1 核心方法: `revise()`

**输入**:
- `files`: 当前代码文件
- `requirements`: 需求文本
- `conflict_report`: Judge 的冲突报告（含 blocking/advisory 冲突列表）
- `repair_hypotheses`: 来自 Advanced Analysis 的修复假设
- `code_ingredients`: 代码分析成分（可选）

**输出**: 修改后的文件内容

### 4.2 修复提示工程

Developer 的 LLM prompt 包含以下关键信息:

1. **冲突焦点** (`conflict_focus`): 从冲突报告中提取的关键标识符（函数名、变量名等）
2. **修复假设** (`repair_hypotheses`): 来自 Advanced Analysis 的结构化推理:
   - `fault_mechanism`: 故障机制
   - `expected_fix_behavior`: 期望的修复行为
   - `minimal_edit_scope`: 最小修改范围
   - `change_operator`: 所需的变更操作类型
3. **图约束** (`code_ingredients`): KG 中与当前修复相关的节点和边

### 4.3 修复输出格式

LLM 可以生成两种类型的修复:

```python
DevOutput:
  rewrites: List[SymbolRewrite]           # 符号级重写: 替换整个函数/类
  search_replace_edits: List[SearchReplaceEdit]  # 搜索替换: 精确行级编辑
  hypothesis: RepairHypothesisChoice      # 选中的修复假设
```

**修复应用过程**:
1. 解析 LLM 的结构化输出 (`DevOutput`)
2. 应用 `SearchReplaceEdit`（精确字符串匹配替换）
3. 支持模糊匹配（`_apply_fuzzy_edit`）以处理 LLM 输出的微小偏差
4. 如果 search_replace 失败，回退到 `SymbolRewrite`
5. 验证语法正确性（`ast.parse`）- 最多重试 5 次

### 4.4 文件风格保持

- 检测并保持原始文件的换行符风格 (LF / CRLF)
- 保持文件末尾换行符的存在性
- 保持缩进风格

---

## 5. 阶段四: 双层验证机制

此阶段由 **Judge Agent** 执行。

### 5.1 硬检查: `_hard_check()`

**定义**: 基于图结构的自动冲突检测，**不需要 LLM**。

**算法**:
1. 收集图中所有 `VIOLATES` 和 `ADVISORY` 类型的边
2. `VIOLATES` → 归为 **blocking**（必须修复）
3. `ADVISORY` → 归为 **advisory**（建议修复，不阻止流程）
4. 按置信度降序排序
5. 如果有 blocking 冲突 → 输出冲突报告（含具体需求ID、代码节点、原因、置信度）
6. 如果无 blocking 冲突 → 硬检查通过

**关键数据结构**:
```python
blocking_conflicts = [{
    "requirement_id": "REQ-003",
    "code_node": "some_function",
    "reason": "missing error handling",
    "confidence": 0.85,
    "tags": ["error_handling", "validation"],
}]
```

### 5.2 软检查: `_soft_check()`

**定义**: 基于 LLM 的语义一致性检查，**需要 LLM 调用**。

**Prompts**:
- **系统提示**: "You are a code auditor. Review the edges and determine if any requirement is violated or logically inconsistent with the code."
- **用户提示**: 包含需求文本 + baseline 图边 + 当前图边
- **要求输出**: 具体的函数名、失败的测试用例、精确的修复位置、最小修改建议

**输出处理**: 如果 LLM 返回 "OK" → 无附加冲突；否则将 LLM 的 advisory 附加到报告中。

### 5.3 Repair Brief 生成

Judge 还生成结构化的 `RepairBrief`，为 Developer 提供精确的修复指导:

```python
RepairBrief:
  requirement_id: "REQ-003"
  target_symbol: "some_function"     # 需要修改的目标符号
  related_symbols: ["helper_func"]   # 相关符号
  issue_summary: "missing error..."  # 问题摘要
  expected_behavior: "should handle" # 期望行为
  minimal_change_hint: "add try..."  # 最小修改提示
  blocking: True                     # 是否阻止流程
  confidence: 0.85                   # 置信度
```

### 5.4 修复循环

```
Judge 输出 conflict_report
        │
   ┌────▼────┐
   │ blocking│──Yes──→ Developer.revise() → 重新构建 KG → Judge 再次评估
   │ > 0 ?   │         (revision_count += 1)
   └────┬────┘
        │ No
        ▼
       END (revision_count ≤ MAX_REVISIONS=1)
```

---

## 6. 核心技术贡献总结

### 6.1 方法论贡献

| # | 贡献 | 技术实现 |
|---|------|---------|
| 1 | **需求的结构化分解** | Semantic Injector 将自然语言 issue 分解为原子 RequirementNode，每条需求独立可验证 |
| 2 | **图约束空间** | 代码结构、需求、依赖关系统一表示为有向图中的节点和边 |
| 3 | **双层验证** | Hard Check (图冲突检测, 零 LLM 成本) + Soft Check (LLM 语义咨询) |
| 4 | **修复假设注入** | Advanced Analysis 生成结构化修复假设，减小 LLM 推理空间 |
| 5 | **自适应策略选择** | Bug Classifier 根据问题类型自动选择最优分析策略 |

### 6.2 工程贡献

| # | 贡献 | 技术实现 |
|---|------|---------|
| 6 | **多 Agent 流水线** | 4 个独立 Agent (Advanced Analyzer, KG Builder, Developer, Judge) 基于 LangGraph 编排 |
| 7 | **文件风格保持** | 修复后的代码保持原始换行符风格、缩进和末尾换行 |
| 8 | **Fault Space 控制** | 支持有界文件选择（单文件 / 邻域扩展 / 回退扩展）以控制修复范围 |
| 9 | **语义契约系统** | 允许为特定实例注入精确的修复约束（semantic_contract_lines） |
| 10 | **实验预设体系** | 8 种预设支持组件贡献的系统性消融分析 |

### 6.3 与现有方法的本质区别

| 维度 | 传统 LLM-based APR | THEMIS |
|------|---------------------|--------|
| **需求表示** | 纯文本 issue | 分解为 KG 中的原子约束节点 |
| **验证方式** | 依赖测试套件 | 图一致性硬检查 + LLM 软检查 |
| **修复指导** | 隐式（全部由 LLM 推理） | 显式（图约束 + 修复假设 + 语义契约） |
| **错误边界** | 无限制（可能修改任意代码） | 受控（fault space 限制在选定文件范围内） |
| **可解释性** | 低（黑盒 LLM 决策） | 中高（可追溯的 KG 约束和冲突检测） |

---

## 附录 A: 组件依赖关系

```
Issue Text
    │
    ├──→ SemanticInjector.decompose_requirements()
    │       └──→ RequirementNode[]
    │
    └──→ BugClassifier.classify()
            └──→ BugType → AnalysisStrategy

Code Files
    │
    ├──→ StructuralExtractor.extract()
    │       ├──→ FunctionNode[]
    │       ├──→ ClassNode[]
    │       ├──→ VariableNode[]
    │       └──→ CallEdge[]
    │
    └──→ DependencyTracer.trace()
            └──→ DependencyEdge[]

KG = Graph(RequirementNode[] + FunctionNode[] + ... + CallEdge[] + DependencyEdge[])

KG
    │
    ├──→ ViolationFlagger.analyze_requirement_satisfaction()
    │       └──→ ViolationEdge[] + ViolationReport[]
    │
    └──→ JudgeAgent.evaluate()
            ├──→ _hard_check()  → blocking + advisory conflicts
            └──→ _soft_check()  → LLM advisory

Conflict Report + Repair Brief
    │
    └──→ DeveloperAgent.revise()
            ├──→ LLM 生成修复候选
            ├──→ 应用 SearchReplace 或 SymbolRewrite
            └──→ 语法验证 + 格式保持
```

## 附录 B: 关键文件索引

| 文件 | 核心功能 |
|------|---------|
| `src/state.py` | AgentState 类型定义 (59 个字段) |
| `src/agents/developer.py` | Developer Agent (1035 行): revise(), 修复应用, 语法验证 |
| `src/agents/judge.py` | Judge Agent (417 行): evaluate(), hard_check, soft_check, repair_brief |
| `src/enhanced_graph_manager/structural_extractor.py` | AST 解析 → 代码结构图 |
| `src/enhanced_graph_manager/semantic_injector.py` | Issue 文本 → 需求节点 + 映射边 |
| `src/enhanced_graph_manager/dependency_tracer.py` | 依赖关系 + 传递闭包 |
| `src/enhanced_graph_manager/violation_flagger.py` | 需求违反检测 + 严重程度排序 |
| `src/enhanced_graph_manager/models.py` | KG 数据模型 (FunctionNode, RequirementNode, ViolationEdge 等) |
| `src/enhanced_graph_adapter.py` | 统一分析接口 + 策略选择 |
| `src/advanced_code_analysis/bug_classifier.py` | Bug 分类 + 策略匹配 (709 行) |
| `src/advanced_code_analysis/concept_mapper.py` | 代码→概念映射 |
| `src/advanced_code_analysis/context_enhancer.py` | 上下文增强 (依赖、领域知识) |
| `src/advanced_code_analysis/multi_round_reasoner.py` | 迭代推理 + 置信度收敛 |
| `src/advanced_code_analysis/models.py` | 分析数据模型 (BugType, AnalysisStrategy 等) |
| `src/main_enhanced.py` | Traditional Enhanced 工作流构建 (255 行) |
| `run_experiment_integrated.py` | Integrated 工作流 + 8 个变体 builder |
