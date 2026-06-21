# THEMIS gpt-5.1-codex-mini 完整实验记录

> 生成日期: 2026-05-16  
> 模型: `gpt-5.1-codex-mini`  
> 数据来源: `swebench_runs/` + `predictions/`  
> 用途: 论文 Experimental Setup & Results 章节的原始数据

---

## 目录

1. [实验总览](#1-实验总览)
2. [完整 Benchmark（300 实例）](#2-完整-benchmark300-实例)
3. [消融实验（6 组 × 5 实例）](#3-消融实验6-组--5-实例)
4. [语义合约实验（3 批）](#4-语义合约实验3-批)
5. [探索性实验](#5-探索性实验)
6. [各实验使用的文件清单](#6-各实验使用的文件清单)
7. [关键指标汇总](#7-关键指标汇总)

---

## 1. 实验总览

### 1.1 实验矩阵

| 实验组 | 实例数 | 运行批次 | 预设 | 目标 |
|--------|--------|---------|------|------|
| **完整 Benchmark** | 300 | 8 批 | `default` (integrated + auto_select) | 全量评估 |
| **消融实验** | 5 × 6 = 30 | 同 1 批 | 各种 `ablation*` | 组件贡献分析 |
| **语义合约 (4)** | 4 × 3 = 12 | 1 批 | contract/control/rerank | 语义合约验证 |
| **语义合约 (5)** | 5 × 3 = 15 | 1 批 | contract/control/rerank | 语义合约验证 |
| **语义合约 (7)** | 7 × 2 = 14 | 1 批 | contract/control | 语义合约验证 |
| **探索性实验** | 5 × 5 = 25 | 同 1 批 | balanced/hypfirst/operator/p1 | 假设验证 |
| **总计** | ~396 | | | |

### 1.2 发现的架构特征

> ⚠️ **重要发现**: 所有代码修复完全由 Advanced Analysis（语义分析管道）完成。这是一个需要在论文中讨论的重要架构特征。

### 1.3 统一配置

| 参数 | 值 |
|------|-----|
| 模型 | `gpt-5.1-codex-mini` |
| 分析模型 | `gpt-5.1-codex-mini` |
| 最大修订次数 | 1 |
| 随机种子 | 42 |
| 数据集 | `SWE-bench/SWE-bench_Lite` (test split) |
| 工作流模式 | `integrated` (Advanced Analysis → KG → Developer → Judge) |
| 分析策略 | `auto_select` (default) 或 `graph_only` (语义/消融实验) |

---

## 2. 完整 Benchmark（300 实例）

### 2.1 批次划分

| 批次 | 索引范围 | 实例数 | 有 Patch | Patch 率 | 总 Adv Tokens | 平均耗时 |
|------|---------|--------|----------|----------|---------------|----------|
| work3 | cases 1-20 | 20 | 18 | 90% | 1,718,695 | 530s |
| work3 | cases 21-40 | 20 | 18 | 90% | 1,969,766 | 665s |
| work3 | cases 41-80 | 40 | 38 | 95% | 3,384,739 | 450s |
| work3 | cases 81-100 | 20 | 19 | 95% | 2,015,605 | 466s |
| work4 | cases 101-140 | 40 | 38 | 95% | 3,342,472 | 480s |
| work5 | cases 141-180 | 40 | 36 | 90% | 3,800,290 | 506s |
| work5 | cases 181-220 | 40 | 39 | 97.5% | 4,209,118 | 539s |
| work5 | cases 221-260 | 40 | 39 | 97.5% | 3,906,052 | 537s |
| work5 | cases 261-300 | 40 | 38 | 95% | 3,940,926 | 539s |
| **总计** | **1-300** | **300** | **~283** | **~94%** | **~28.3M** | **~523s** |

### 2.2 覆盖的仓库

| 仓库 | 约实例数 |
|------|----------|
| django/django | ~70 |
| sympy/sympy | ~45 |
| scikit-learn/scikit-learn | ~35 |
| matplotlib/matplotlib | ~30 |
| sphinx-doc/sphinx | ~20 |
| astropy/astropy | ~18 |
| pydata/xarray | ~16 |
| pallets/flask | ~12 |
| mwaskom/seaborn | ~12 |
| psf/requests | ~10 |
| pytest-dev/pytest | ~10 |
| 其他 | ~22 |

### 2.3 输出文件

- `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases1_100_work3.jsonl` (100 instances)
- `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases101_140_work4.jsonl` (40 instances)
- `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases141_180_work5.jsonl` (40 instances)
- `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases181_220_work5.jsonl` (40 instances)
- `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases221_260_work5.jsonl` (40 instances)
- `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases261_300_work5.jsonl` (40 instances)

### 2.4 日志目录

- `swebench_runs/themis_seed42_cases1_20_gpt51_codexmini_work3/logs/` (20 files)
- `swebench_runs/themis_seed42_cases21_40_gpt51_codexmini_work3/logs/` (20 files)
- `swebench_runs/themis_seed42_cases41_80_gpt51_codexmini_work3/logs/` (40 files)
- `swebench_runs/themis_seed42_cases81_100_gpt51_codexmini_work3/logs/` (20 files)
- `swebench_runs/themis_seed42_cases101_140_gpt51_codexmini_work4/logs/` (40 files)
- `swebench_runs/themis_seed42_cases141_180_gpt51_codexmini_work5/logs/` (40 files)
- `swebench_runs/themis_seed42_cases181_220_gpt51_codexmini_work5/logs/` (40 files)
- `swebench_runs/themis_seed42_cases221_260_gpt51_codexmini_work5/logs/` (40 files)
- `swebench_runs/themis_seed42_cases261_300_gpt51_codexmini_work5/logs/` (40 files)

---

## 3. 消融实验（6 组 × 5 实例）

### 3.1 固定测试集（所有消融实验共用）

| # | Instance ID | 仓库 | 目标文件 |
|---|------------|------|---------|
| 1 | `django__django-16139` | django/django | `django/contrib/auth/forms.py` |
| 2 | `sympy__sympy-13773` | sympy/sympy | `sympy/matrices/common.py` |
| 3 | `django__django-12497` | django/django | `django/db/models/fields/related.py` |
| 4 | `django__django-15320` | django/django | `django/template/defaulttags.py` |
| 5 | `django__django-12284` | django/django | `django/utils/html.py` |

### 3.2 消融变体详情

| 消融变体 | 预设 | 策略 | 说明 |
|----------|------|------|------|
| **ablation1** | `ablation1` | `graph_only` | 消融基线：仅图结构 + 仅冲突检测 |
| **ablation2** | `ablation2` | `graph_only` | 消融变体 2 |
| **ablation13** | — | — | 消融变体 13 |
| **ablation15** | — | — | 消融变体 15 |
| **ablation15_rankfix** | — | `graph_only` | ablation15 + ranking 修复 |
| **ablation456** | — | `graph_only` | 消融变体 456（组合） |

> 注: 消融变体编号对应论文中的实验配置编号，具体差异需要在论文中明确定义。

### 3.3 输出文件

- `predictions/gpt-5.1-codex-mini.themis_ablation1_seed42_first5.jsonl`
- `predictions/gpt-5.1-codex-mini.themis_ablation2_seed42_first5.jsonl`
- `predictions/gpt-5.1-codex-mini.themis_ablation13_seed42_first5.jsonl`
- `predictions/gpt-5.1-codex-mini.themis_ablation15_seed42_first5.jsonl`
- `predictions/gpt-5.1-codex-mini.themis_ablation15_rankfix_seed42_first5.jsonl`
- `predictions/gpt-5.1-codex-mini.themis_ablation456_seed42_first5.jsonl`

---

## 4. 语义合约实验（3 批）

语义合约实验使用专门预设，为特定困难实例提供精确的修复指导。

### 4.1 批次 1: semantic4 (4 实例)

**条件设置**:
- `s1_contract`: `semantics_contract_prompt` 预设（有语义合约）
- `s1_control`: `fault_space_neighborhood_context` 预设（无合约对照组）
- `s2_rerank`: `semantics_contract_rerank` 预设（合约 + harness 对齐重排）

| Instance | 目标文件 |
|----------|---------|
| `django__django-14915` | `django/utils/tree.py` + `django/forms/models.py` |
| `matplotlib__matplotlib-22711` | `lib/matplotlib/widgets.py` + `lib/matplotlib/tests/test_widgets.py` |
| `mwaskom__seaborn-3407` | `seaborn/axisgrid.py` + `tests/test_axisgrid.py` |
| `sphinx-doc__sphinx-7975` | `sphinx/environment/adapters/indexentries.py` + `tests/test_environment_indexentries.py` |

### 4.2 批次 2: semantic5 (5 实例)

| Instance | 目标文件 |
|----------|---------|
| `django__django-12308` | `django/forms/fields.py` + `django/contrib/admin/utils.py` |
| `django__django-13028` | `django/shortcuts.py` + `django/db/models/sql/query.py` |
| `pallets__flask-4992` | `src/flask/config.py` + `src/flask/app.py` |
| `pydata__xarray-4493` | `xarray/core/variable.py` + `xarray/core/dataarray.py` |
| `sympy__sympy-13773` | `sympy/matrices/common.py` + `sympy/matrices/matrices.py` |

### 4.3 批次 3: semantic_work5 (7 实例)

| Instance | 目标文件 |
|----------|---------|
| `astropy__astropy-14365` | `astropy/io/ascii/qdp.py` + `astropy/io/ascii/tests/test_qdp.py` |
| `django__django-11283` | `django/contrib/auth/migrations/0011_update_proxy_permissions.py` |
| `django__django-13768` | `tests/dispatch/tests.py` |
| `django__django-14017` | `django/db/models/query_utils.py` |
| `mwaskom__seaborn-2848` | `seaborn/relational.py` |
| `scikit-learn__scikit-learn-10297` | `sklearn/linear_model/_ridge.py` |
| `scikit-learn__scikit-learn-13496` | `sklearn/ensemble/_iforest.py` |

### 4.4 输出文件

| 文件 | 条件 | 实例数 |
|------|------|--------|
| `*semantic4_s1_contract_work4.jsonl` | 有合约 | 4 |
| `*semantic4_s1_control_work4.jsonl` | 无合约对照 | 4 |
| `*semantic4_s2_rerank_work4.jsonl` | 合约 + 重排 | 4 |
| `*semantic5_s1_contract_work3.jsonl` | 有合约 | 5 |
| `*semantic5_s1_control_work3.jsonl` | 无合约对照 | 5 |
| `*semantic5_s2_rerank_work3.jsonl` | 合约 + 重排 | 5 |
| `*semantic_work5_s1_contract.jsonl` | 有合约 | 7 |
| `*semantic_work5_s1_control.jsonl` | 无合约对照 | 7 |
| `*semantic_work5_s2_rerank.jsonl` | 合约 + 重排 | 1 |

---

## 5. 探索性实验

所有探索性实验使用相同 5 个实例（同消融实验）。

| 实验名 | 说明 |
|--------|------|
| **balanced** | 平衡的目标文件选择策略 |
| **hypfirst** | 假设驱动的修复优先级 |
| **hypfirst_v2** | 假设驱动 v2 |
| **operator_ingredient** | 操作符级别的修复粒度分析 |
| **operator_ingredient_failureclass** | 操作符 + 失败分类 |
| **p1** | 优先级 1 修复策略 |

输出文件均在 `predictions/` 目录下，命名模式为 `gpt-5.1-codex-mini.themis_{exp}_seed42_first5.jsonl`。

---

## 6. 各实验使用的文件清单

### 6.1 完整 Benchmark 涉及的仓库文件（部分代表性文件）

**django**:
- `django/contrib/auth/forms.py` — 认证表单
- `django/db/models/fields/related.py` — 关联字段
- `django/db/models/sql/query.py` — SQL 查询
- `django/template/defaulttags.py` — 模板标签
- `django/utils/html.py` — HTML 工具
- `django/forms/models.py` — 模型表单
- `django/db/migrations/serializer.py` — 迁移序列化
- 等 40+ 文件

**sympy**:
- `sympy/matrices/common.py` — 矩阵运算
- `sympy/interactive/printing.py` — 交互式打印
- `sympy/core/numbers.py` — 数值类型
- 等 30+ 文件

**scikit-learn**:
- `sklearn/metrics/tests/test_classification.py` — 分类指标测试
- `sklearn/linear_model/_ridge.py` — 岭回归
- `sklearn/ensemble/_iforest.py` — 孤立森林
- 等 25+ 文件

### 6.2 语义合约实验的额外上下文文件

语义合约预设 (`semantics_contract_prompt`) 为 20 个特定实例提供了预定义的额外上下文文件（详见 `_extra_context_files_for_preset` 函数），以及语义修复合约文本（详见 `_semantic_contract_lines_for_preset` 函数）。

---

## 7. 关键指标汇总

### 7.1 Benchmark 整体指标

| 指标 | 值 |
|------|-----|
| 总实例数 | 300 |
| 成功生成 Patch | ~283 (94%) |
| 总 Advanced Analysis Tokens | ~28.3M |
| 总 Chat Tokens (Developer/Judge) | 0 |
| 平均单实例耗时 | ~523s (8.7 min) |
| 平均单实例 Advanced Tokens | ~94,000 |
| 覆盖仓库数 | 12+ |

### 7.2 日志中记录的关键指标

每个实例日志包含：

```json
{
  "instance_id": "django__django-15320",
  "selected_files": ["django/template/defaulttags.py"],
  "mode": "integrated",
  "experiment_preset": "default",
  "model": "gpt-5.1-codex-mini",
  "analysis_strategy": "auto_select",
  "duration_s": 407.9,
  "patch_chars": 1986,
  "meta": {
    "revision_count": 0,
    "advanced_usage": {
      "prompt_tokens": 51575,
      "completion_tokens": 31736,
      "total_tokens": 83311
    },
    "chat_usage": {
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0
    },
    "advanced_summary": {
      "strategy": "integrated",
      "confidence": 0.2,
      "findings_count": 8,
      "recommendations_count": 1
    },
    "loop_summary": {
      "developer_rounds": 1,
      "effective_rounds": 1,
      "effective_modification_rate": 1.0,
      "target_hit_rate": 0.0
    },
    "repair_brief": {
      "target_symbol": "SetPasswordForm",
      "confidence": 0.2
    }
  }
}
```

### 7.3 可用于论文的关键发现

1. **300-case resolved rate**: 官方 SWE-bench harness 评估文件合计显示 `58/300` resolved（约 `19.3%`；在 279 个非空 patch completed instances 上为约 `20.8%`）
2. **94% patch 生成率**: 系统在大多数情况下都能生成非空 patch
3. **chat token 统计限制**: 当前 `chat_usage.total_tokens=0` 反映回调统计路径未可靠捕获 Developer/Judge 的 chat-model usage，不应解释为 Developer/Judge LLM 未调用
4. **平均 8.7 分钟/实例**: 单实例处理时间合理
5. **confidence 普遍偏低 (0.2-0.7)**: 语义分析置信度有待提高
6. **target_hit_rate 为 0**: 修复的精确目标命中率需要改进
7. **语义合约的有效性**: contract 预设 vs control 预设的对比可量化语义指导的价值

---

## 附录

### A. 实验目录结构

```
swebench_runs/
├── themis_seed42_cases1_20_gpt51_codexmini_work3/     # batch 1
├── themis_seed42_cases21_40_gpt51_codexmini_work3/    # batch 2
├── themis_seed42_cases41_80_gpt51_codexmini_work3/    # batch 3
├── themis_seed42_cases81_100_gpt51_codexmini_work3/   # batch 4
├── themis_seed42_cases101_140_gpt51_codexmini_work4/   # batch 5
├── themis_seed42_cases141_180_gpt51_codexmini_work5/   # batch 6
├── themis_seed42_cases181_220_gpt51_codexmini_work5/   # batch 7
├── themis_seed42_cases221_260_gpt51_codexmini_work5/   # batch 8
├── themis_seed42_cases261_300_gpt51_codexmini_work5/   # batch 9
├── themis_seed42_case21_gpt51_codexmini/              # single retry
├── themis_seed42_case21_gpt51_codexmini_retry1/       # retry
├── themis_seed42_cases1_3_gpt51_codexmini_retry1/     # retry
├── themis_semantic4_s1_contract_work4/                 # semantic batch 1
├── themis_semantic4_s1_control_work4/
├── themis_semantic4_s2_rerank_work4/
├── themis_semantic5_s1_contract_work3/                 # semantic batch 2
├── themis_semantic5_s1_control_work3/
├── themis_semantic5_s2_rerank_work3/
├── themis_semantic_work5_s1_contract/                  # semantic batch 3
├── themis_semantic_work5_s1_control/
├── themis_semantic_work5_s2_rerank/
├── themis_seed42_n20_gpt4o_mini/                       # gpt-4o-mini experiments
├── themis_seed42_n20_gpt4o_mini_graph_only/
├── themis_seed42_n20_gpt4o_mini_integrated/
├── themis_seed42_n20_gpt4o_mini_integrated_remaining/
├── themis_seed42_n20_gpt4o_mini_v2/
├── themis_seed42_n20_gpt4o_mini_v2_final/
├── themis_seed42_n20_gpt4o_mini_v2_remaining/
└── themis_seed42_n20_gpt4o_mini_v2_sympy24152/

predictions/
├── gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases*.jsonl   # 完整 benchmark
├── gpt-5.1-codex-mini.themis_ablation*_seed42_first5.jsonl        # 消融实验
├── gpt-5.1-codex-mini.themis_semantic*_work*.jsonl               # 语义合约实验
├── gpt-5.1-codex-mini.themis_{balanced,hypfirst,operator,p1}_*.jsonl  # 探索性实验
└── gpt-4o-mini.*.jsonl                                           # gpt-4o-mini 实验
```
