# THEMIS SWE-bench Lite Benchmark 系统架构文档

> 生成日期: 2026-05-16  
> 基于: `run_swebench_lite_predictions.py` (1337 lines)  
> 用途: 为 Agent 论文写作提供系统理解所需的完整技术上下文

---

## 目录

1. [系统概览](#1-系统概览)
2. [数据流架构](#2-数据流架构)
3. [核心组件详解](#3-核心组件详解)
4. [实验预设体系](#4-实验预设体系)
5. [工作流模式](#5-工作流模式)
6. [文件选择策略](#6-文件选择策略)
7. [需求构建机制](#7-需求构建机制)
8. [关键数据结构](#8-关键数据结构)
9. [命令行接口](#9-命令行接口)
10. [Benchmark 测试覆盖](#10-benchmark-测试覆盖)

---

## 1. 系统概览

THEMIS 的 SWE-bench Lite Benchmark 是一个端到端的自动化代码修复评估流水线，核心流程如下：

```
SWE-bench Lite Dataset (HF)
        │
        ▼
  ┌─ Instance Selection ──────────────────────┐
  │  • 种子随机采样 (--seed)                      │
  │  • 指定 instance_id                          │
  │  • 索引范围 (--start/--end)                    │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─ Repository Setup ────────────────────────┐
  │  • Clone repo at base_commit               │
  │  • 跳过已完成的 instance (增量重跑)            │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─ File Selection ──────────────────────────┐
  │  • 直接路径提取 (problem_statement)           │
  │  • Token 搜索 (ripgrep → git grep → Python) │
  │  • 上下文扩展 (preset 驱动的额外文件加载)       │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─ Requirements Building ───────────────────┐
  │  • Problem statement                       │
  │  • Hints                                   │
  │  • Failing tests (FAIL_TO_PASS)             │
  │  • Semantic repair contracts (可选)          │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─ LangGraph Workflow Execution ────────────┐
  │  ┌──────────────────────────────────┐      │
  │  │ Advanced Analysis (LLM Semantic) │      │
  │  │         ↓                        │      │
  │  │ Enhanced Graph Manager           │      │
  │  │   (Structural KG + Violations)   │      │
  │  │         ↓                        │      │
  │  │ Developer Agent (Code Revision)  │      │
  │  │         ↓                        │      │
  │  │ Judge Agent (Verification)       │      │
  │  └──────────────────────────────────┘      │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─ Patch Generation & Output ───────────────┐
  │  • git diff 生成标准 patch                   │
  │  • 写入 predictions .jsonl                  │
  │  • 记录 token 消耗、耗时等元数据               │
  └───────────────────────────────────────────┘
```

### 关键设计原则

1. **保守的文件选择**：默认仅选择 1 个 Python 文件（GraphManager 的限制：单模块 AST 解析）
2. **增量重跑**：自动跳过已存在于 `.jsonl` 中的 instance_id
3. **文件风格保持**：保留原始文件的换行符风格（LF vs CRLF）和末尾换行
4. **Dry-run 模式**：`--dry-run` 跳过 LLM 调用，输出空 patch（用于验证基础设施）

---

## 2. 数据流架构

### 2.1 AgentState 核心状态

```python
initial_state: AgentState = {
    "messages": [],                # LangGraph 消息历史
    "files": {},                   # 文件路径 -> 内容映射（可编辑的 source of truth）
    "requirements": "",            # 需求文本（issue + constraints）
    "knowledge_graph": nx.DiGraph(),  # 当前代码知识图谱
    "baseline_graph": None,        # 原始代码基准图谱
    "conflict_report": None,       # Judge 的冲突报告
    "revision_count": 0,           # 修正循环计数
    "repo_root": str(repo_root),   # 仓库根目录（Swebench 特有字段）
    "advanced_analysis": None,     # 高级分析结果
    "analysis_report": None,       # 结构化分析报告
}
```

### 2.2 数据流转换

```
Instance JSON
  ├── instance_id      → 标识符
  ├── repo             → 仓库名
  ├── base_commit      → 基准提交
  ├── problem_statement → requirements (主要部分)
  ├── hints_text       → requirements (附加上下文)
  ├── FAIL_TO_PASS     → requirements (测试列表) + file selection input
  └── [其他字段]        → 文件选择辅助

        ↓ 处理后

AgentState {
    files: {target.py: content, [extra.py: content...]},
    requirements: "problem\nhints\ntests",
    ...
}

        ↓ 工作流执行后

updated_files: {target.py: revised_content}

        ↓ 应用并生成 patch

git diff → patch → predictions.jsonl
```

---

## 3. 核心组件详解

### 3.1 数据集加载 (`_load_dataset`)

支持三种数据源：
- **HuggingFace**: `SWE-bench/SWE-bench_Lite`（默认）
- **本地 JSON**: `dataset.json`
- **本地 JSONL**: `dataset.jsonl`

```python
SWE_BENCH_LITE_DATASET = "SWE-bench/SWE-bench_Lite"
```

### 3.2 仓库管理 (`_ensure_checkout`)

- 使用 `git clone --bare` + 增量 fetch 策略
- 支持 GitHub (`github.com`) 和 GitLab (`gitlab.com`) 仓库
- 在 `base_commit` 处 checkout，保留干净的工作目录
- 设置 `git config user.email/user.name` 以便后续提交

### 3.3 工作流构建器 (`_build_langgraph_app`)

通过反射动态加载工作流构建函数：

```python
# 两种运行时模式
"traditional" → "src.main_enhanced:build_workflow"
"integrated"  → "run_experiment_integrated:build_integrated_workflow"

# 实验预设可覆盖
"ablation1"   → "run_experiment_integrated:build_integrated_workflow_ablation1"
"fault_space_fallback" → "...build_integrated_workflow_fault_space_fallback"
# ... 等
```

**传入参数**：`llm_model`, `analysis_model`, `analysis_strategy`, `max_revisions`, `callbacks`

### 3.4 LangGraph 工作流执行 (`_run_langgraph_workflow`)

```python
app = _build_langgraph_app(...)      # 构建编译后的 LangGraph app
final_state = app.invoke(            # 同步调用执行
    initial_state,
    config={"recursion_limit": recursion_limit}  # 默认 50
)
```

返回：
- `updated_files`: 修改后的文件内容映射
- `meta`: 元数据（token 消耗、工作流步骤等）
- `last_effective_files`: 最后一次有效修改的文件（用于故障恢复）

### 3.5 Patch 生成 (`_generate_patch`)

```python
def _generate_patch(repo_root):
    return _git(repo_root, ["diff", "--no-color"])
```

使用标准 `git diff` 生成 SWE-bench 评估框架兼容的 patch。

### 3.6 Token 计数器 (`TokenCounterCallback`)

LangChain 回调，记录：
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `successful_requests`

通过 `on_llm_end` / `on_chat_model_end` 钩子捕获使用量。

---

## 4. 实验预设体系

系统支持 8 个实验预设，用于对比不同配置的效果：

### 4.1 预设详情

| 预设名 | 工作流 | 策略 | 修订次数 | 说明 |
|--------|--------|------|----------|------|
| **`default`** | `build_integrated_workflow` | `AUTO_SELECT` | 1 | 完整集成工作流（默认） |
| **`ablation1`** | `build_integrated_workflow_ablation1` | `GRAPH_ONLY` | 1 | 消融基线：仅图结构 + 仅冲突检测 |
| **`fault_space_fallback`** | `build_integrated_workflow_fault_space_fallback` | `GRAPH_ONLY` | 1 | ablation1 + 有界回退目标扩展 |
| **`fault_space_neighborhood`** | `build_integrated_workflow_fault_space_neighborhood` | `GRAPH_ONLY` | 1 | ablation1 + 文件本地结构邻域扩展 |
| **`fault_space_neighborhood_context`** | `build_integrated_workflow_fault_space_neighborhood` | `GRAPH_ONLY` | 1 | neighborhood + 上下文文件 |
| **`fault_space_neighborhood_retrieval`** | `build_integrated_workflow_fault_space_neighborhood` | `GRAPH_ONLY` | 1 | context + 从 instance 文本提取的文件 |
| **`semantics_contract_prompt`** | `build_integrated_workflow_fault_space_neighborhood` | `GRAPH_ONLY` | 1 | + 测试特定的语义修复合约 |
| **`semantics_contract_rerank`** | `build_integrated_workflow_semantics_contract_rerank` | `GRAPH_ONLY` | 1 | 合约 prompt + harness 对齐的候选重排 |

### 4.2 分析策略枚举

```python
class AnalysisStrategy(Enum):
    AUTO_SELECT = "auto_select"           # 自动选择最佳策略
    GRAPH_ONLY = "graph_only"             # 纯图结构分析
    LLM_ONLY = "llm_only"                 # 纯 LLM 语义分析
    INTEGRATED = "integrated"             # 完整集成分析
```

### 4.3 预设选择的含义

- **default** 使用 `AUTO_SELECT`：系统根据实例特征自动决定使用哪种分析策略
- **所有其他预设** 使用 `GRAPH_ONLY`：仅依赖结构图分析，跳过 LLM 语义分析，用于对比消融实验

---

## 5. 工作流模式

### 5.1 Traditional Enhanced (`src/main_enhanced:build_workflow`)

```
┌─────────────────────────────────┐
│  initial_graph_builder          │  ← parse_code_structure → enrich_with_requirements
│  (构建基线 KG)                    │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  developer                      │  ← DevelopmentAgent.revise(files, requirements, conflict_report)
│  (代码修正)                       │     最多重试 5 次语法错误
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  graph_builder                  │  ← 对修改后代码重新构建 KG
│  (更新 KG)                       │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  judge                          │  ← JudgeAgent.evaluate(kg, baseline, requirements)
│  (验证冲突)                       │     硬检查: 图冲突  +  软检查: LLM 咨询
└──────────────┬──────────────────┘
               ▼
         ┌─────┴─────┐
         │ 有冲突?     │──Yes──→ developer (loop up to MAX_REVISIONS=1)
         │ 且未达上限  │
         └─────┬─────┘
               │ No
               ▼
             END
```

**核心组件**：
- `EnhancedGraphAdapter()` — 统一的图操作接口
- `DeveloperAgent(llm)` — LLM 驱动的代码修正
- `JudgeAgent(llm)` — 硬检查 + LLM 软检查

### 5.2 Integrated Workflow (`run_experiment_integrated:build_integrated_workflow`)

```
┌─────────────────────────────────┐
│  advanced_analysis              │  ← AdvancedCodeAnalyzer (多阶段语义分析)
│  (LLM 语义理解)                   │     • Bug Classifier → 分类问题类型
│                                  │     • Concept Mapper → 概念映射
│                                  │     • Context Enhancer → 上下文增强
│                                  │     • Pattern Learner → 模式学习
│                                  │     • Conflict Detector → 冲突检测
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  initial_graph_builder          │  ← parse_code_structure + enrich_with_requirements
│  (构建 KG，注入 LLM 分析结果)      │     LLM 分析结果注入到 KG 节点属性中
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  developer                      │  ← 使用语义 + 结构双重视角进行修正
│  (增强代码修正)                    │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  graph_builder                  │  ← 重新构建 KG
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  judge                          │  ← 结构 + 语义双重验证
│  (综合验证)                       │
└──────────────┬──────────────────┘
               ▼
         ┌─────┴─────┐
         │ 需修正?     │──Yes──→ developer (up to max_revisions)
         └─────┬─────┘
               │ No
               ▼
             END
```

**附加特性**：
- 支持 `repo_root` 字段：运行时可加载额外文件
- 语义合约注入：针对特定实例的精确修复指导
- 文件回退机制：`fault_space_fallback` 在有界范围内扩展目标文件
- 邻域扩展：`fault_space_neighborhood` 加载结构相关的邻近文件

---

## 6. 文件选择策略

### 6.1 整体流程 (`_select_target_file`)

```
输入: problem_statement + hints_text + FAIL_TO_PASS
                    │
        ┌───────────┴───────────┐
        │ 1. 直接路径提取         │
        │   _extract_candidate_paths │
        │   正则匹配 .py 文件路径    │
        └───────────┬───────────┘
                    │ 未找到
                    ▼
        ┌───────────┴───────────┐
        │ 2. Token 提取          │
        │   _extract_tokens       │
        │   提取有意义标识符(≤12个)  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────┴───────────┐
        │ 3. 多引擎搜索          │
        │   ripgrep → git grep   │
        │   → Python scan        │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────┴───────────┐
        │ 4. Token 加权排序       │
        │   dunder → 3.0x        │
        │   含下划线 → 1.5x        │
        │   纯字母 → 1.0x         │
        │   过滤测试文件           │
        └───────────┬───────────┘
                    │
                    ▼
              得分最高的 .py 文件
```

### 6.2 搜索引擎优先级

| 优先级 | 引擎 | 说明 |
|--------|------|------|
| 1 | **ripgrep** (`rg`) | 最快，性能最优 |
| 2 | **git grep** | 次选，普遍可用 |
| 3 | **Python scan** | 兜底，Python os.walk 扫描 |

### 6.3 上下文扩展策略

**额外上下文文件** (`_extra_context_files_for_preset`)：
- 针对特定 instance_id 手动策划的辅助文件列表
- 仅在 `fault_space_neighborhood_context` / `fault_space_neighborhood_retrieval` / `semantics_contract_prompt` 预设中激活
- 覆盖 20 个实例（django, matplotlib, flask, seaborn, requests, xarray, sphinx, sympy, scikit-learn, astropy）

**检索式上下文文件** (`_retrieved_context_files_for_preset`)：
- 从 instance 文本中动态提取 `.py` 文件提及
- 在 `fault_space_neighborhood_retrieval` / `semantics_contract_prompt` / `semantics_contract_rerank` 预设中激活

---

## 7. 需求构建机制

### 7.1 `_build_requirements` 结构

```
requirements = 
    [Problem Statement]      ← 实例的主要问题描述
    + [Hints]                ← 可选提示信息
    + [Failing Tests]        ← FAIL_TO_PASS 列表 (最多25个)
    + [Semantic Contracts]   ← 可选语义修复合约 (特定预设)
```

### 7.2 语义修复合约 (`_semantic_contract_lines_for_preset`)

针对特定困难实例提供精确的修复指导。仅对以下实例生效：

| Instance | 合约要点 |
|----------|---------|
| `django__django-12308` | JSONField admin 渲染保持 JSON 序列化语义 |
| `django__django-13028` | `resolve_expression` 的过滤语义 |
| `matplotlib__matplotlib-18869` | 版本 API 的精确语义 |
| `pallets__flask-4045` | Blueprint 点号名称的异常语义 |
| `pallets__flask-4992` | `Config.from_file` 的二进制加载器支持 |
| `sympy__sympy-13773` | 矩阵运算的语义边界 |
| `psf__requests-3362` | 请求头的编码语义 |
| `sphinx-doc__sphinx-7975` | 索引条目的语义 |
| `scikit-learn__scikit-learn-10297` | Ridge 回归模型语义 |
| `astropy__astropy-14365` | QDP 文件解析语义 |
| `mwaskom__seaborn-3407` | AxisGrid 行为语义 |

---

## 8. 关键数据结构

### 8.1 SolveResult

```python
@dataclass
class SolveResult:
    instance_id: str          # SWE-bench 实例 ID
    repo: str                 # 仓库名
    base_commit: str          # 基准提交哈希
    model: str                # 使用的 LLM 模型
    selected_files: List[str] # 选中的目标文件列表
    patch: str                # 生成的 git diff patch
    success: bool             # 是否成功
    error: Optional[str]      # 错误信息
    duration_s: float         # 执行耗时（秒）
```

### 8.2 ExperimentPreset

```python
@dataclass(frozen=True)
class ExperimentPreset:
    name: str                              # 预设名称
    workflow_builder: Optional[str] = None  # 工作流构建器路径
    analysis_strategy: Optional[str] = None # 分析策略
    max_revisions: Optional[int] = None     # 最大修订次数
```

### 8.3 Predictions JSONL 格式

```json
{
    "instance_id": "django__django-12308",
    "model_name_or_path": "gpt-5.1-codex-mini",
    "model_patch": "diff --git a/django/forms/models.py b/..."
}
```

---

## 9. 命令行接口

```bash
python run_swebench_lite_predictions.py \
    --dataset SWE-bench/SWE-bench_Lite \   # HF 数据集名 或 本地文件
    --split test \                          # 数据集分割
    --num 30 \                              # 采样数量
    --seed 42 \                             # 随机种子
    --start 0 --end 100 \                   # 或使用索引范围
    --instance-id django__django-12308 \    # 或指定特定实例（可多次）
    --workdir swebench_runs \               # 工作目录
    --output predictions/swebench_lite_sample.jsonl \  # 输出文件
    --model gpt-5.1-codex-mini \            # LLM 模型（Developer/Judge）
    --analysis-model gpt-5.1-codex-mini \   # LLM 模型（Advanced Analysis）
    --mode integrated \                     # traditional | integrated
    --experiment-preset default \           # 实验预设
    --analysis-strategy auto_select \       # auto_select | graph_only | llm_only | integrated
    --max-revisions 1 \                     # 最大修订次数
    --recursion-limit 50 \                  # LangGraph 递归限制
    --dry-run                               # 跳过 LLM 调用
```

---

## 10. Benchmark 测试覆盖

### 10.1 数据集

- **SWE-bench Lite**: 300 个精选实例（来自 12 个 Python 仓库的 real-world GitHub issues）
- 覆盖仓库：django, scikit-learn, matplotlib, sympy, flask, requests, seaborn, xarray, sphinx, astropy, pydata, pallets

### 10.2 评估指标

| 指标 | 说明 |
|------|------|
| **Resolved Rate** | 通过 SWE-bench 评估框架中对应测试的实例比例 |
| **Token 消耗** | LLM 调用的总 token 数（prompt + completion） |
| **执行耗时** | 每个实例的处理时间 |
| **Patch 质量** | 是否生成非空 patch、是否语法正确 |
| **消融对比** | 各预设间的性能差异 |

### 10.3 实验对比维度

系统支持的实验预设体系允许系统性地对比：

1. **完整系统 vs 消融基线** (`default` vs `ablation1`)
2. **目标文件扩展策略** (`ablation1` vs `fault_space_fallback` vs `fault_space_neighborhood`)
3. **上下文增益效果** (`fault_space_neighborhood` vs `fault_space_neighborhood_context`)
4. **检索增强效果** (`fault_space_neighborhood_context` vs `fault_space_neighborhood_retrieval`)
5. **语义合约价值** (`fault_space_neighborhood_retrieval` vs `semantics_contract_prompt`)
6. **重排序策略** (`semantics_contract_prompt` vs `semantics_contract_rerank`)
7. **传统 vs 集成** (`--mode traditional` vs `--mode integrated`)

---

## 附录

### A. 相关源文件

| 文件 | 说明 |
|------|------|
| `run_swebench_lite_predictions.py` | Benchmark 入口脚本（本文件） |
| `src/state.py` | AgentState 核心状态定义 |
| `src/agents/developer.py` | DeveloperAgent 实现 |
| `src/agents/judge.py` | JudgeAgent 实现 |
| `src/main_enhanced.py` | Traditional Enhanced 工作流构建 |
| `run_experiment_integrated.py` | Integrated 工作流构建（含 7 个变体） |
| `src/enhanced_graph_adapter.py` | 统一图操作适配器 |
| `src/enhanced_graph_manager/` | 增强图管理器子系统 |
| `src/advanced_code_analysis/` | 高级语义分析管道 |

### B. 关键依赖

- **LangGraph**: 工作流编排
- **LangChain**: LLM 交互抽象
- **NetworkX**: 图数据结构
- **OpenAI GPT-5.x**: LLM 后端
- **Git**: 仓库操作和 patch 生成
- **HuggingFace Datasets**: 数据加载
