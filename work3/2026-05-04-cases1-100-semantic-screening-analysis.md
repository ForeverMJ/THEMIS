# `cases1_100_work3` 语义筛选分析

日期：2026-05-04

## 1. 目标

按 `work3/2026-04-30-semantic-case-selection-standard.md` 的标准，判断 `cases1_100_work3` 里能严格筛出多少个 semantic-contract candidate。

## 2. 本次采用的判定基线

按标准文件，严格筛选至少需要两类证据：

1. **generation-side**
   - prediction record
   - per-instance run log JSON
   - 尤其要有：`selected_files`、`patch_chars`、`empty_patch_reason`、`developer_metrics_history`、`selection_reason`、`hypothesis_root_cause`、`expected_invariant`、`patch_strategy`、`target_hit*` 等
2. **benchmark-side**
   - harness result / report
   - eval log
   - resolved / unresolved 结果

标准流程要求先去掉明显非候选，再在“已接近 repair region 但仍 unresolved”的集合里做语义性判断，并对模糊 case 用 activation / tie-break 规则补充判断。

## 3. 本地证据审计

### 3.1 已确认存在的证据

| 证据 | 状态 | 说明 |
|---|---|---|
| `cases1_100_work3` predictions | 有 | `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases1_100_work3.jsonl` |
| 官方 harness 汇总报告 | 有 | `gpt-5.1-codex-mini.cases1_100_work3_eval.json` |
| harness 控制台日志 | 有 | `work3/cases1_100_work3_eval.console.log` |
| 89 个非空 patch case 的 per-case harness 目录 | 有 | `logs/run_evaluation/cases1_100_work3_eval/gpt-5.1-codex-mini/*/` |

### 3.2 缺失的关键证据

严格筛选最关键的缺口是：**与 `work3` 这次 100-case predictions 精确对应的 generation-side per-instance JSON logs 本地并不齐全。**

已知记录里写明“应该有”的日志目录包括：

- `swebench_runs/themis_seed42_cases21_40_gpt51_codexmini_work3/logs/*.json`
- `swebench_runs/themis_seed42_cases41_80_gpt51_codexmini_work3/logs/*.json`
- `swebench_runs/themis_seed42_cases81_100_gpt51_codexmini_work3/logs/*.json`

但这些 `work3` generation logs 目录当前本地没有找到可用文件。

### 3.3 为什么不能拿仓库里已有的旧日志替代

仓库里存在一组旧的 `swebench_runs/logs/*.json`，其中 20 个 instance_id 与 `cases1_100_work3` 的前 20 个 case 重合；但是它们**不能**作为这次 `work3` 的严格筛选依据，因为它们与当前 predictions 不一致。

我对这 20 个重合 case 做了比对：

- `work3` predictions 中的 `model_patch` 长度
- 旧日志中的 `patch_chars`

结果：**20 / 20 全部不一致**。

这说明这些旧日志对应的是另一次生成运行，而不是当前 `cases1_100_work3` 的本地 authority。

## 4. 先按标准流程做能做的部分

### Step 1. Remove obvious non-candidates

按标准，`patch_chars = 0` / 空 patch 应直接剔除。

本次可直接剔除的 empty-patch case 共 **11** 个：

1. `django__django-15320`
2. `sympy__sympy-24152`
3. `sympy__sympy-14396`
4. `matplotlib__matplotlib-23299`
5. `sympy__sympy-15609`
6. `sphinx-doc__sphinx-10451`
7. `sympy__sympy-18621`
8. `matplotlib__matplotlib-25442`
9. `sphinx-doc__sphinx-10325`
10. `sphinx-doc__sphinx-7686`
11. `sympy__sympy-13915`

### Step 2. 排除已 resolved 的 case

已通过官方 benchmark 的 case 不应进入“semantic candidate”池。

本次 resolved 共 **19** 个：

1. `django__django-10914`
2. `django__django-11620`
3. `django__django-11848`
4. `django__django-12589`
5. `django__django-12856`
6. `django__django-13265`
7. `django__django-13447`
8. `django__django-13933`
9. `django__django-13964`
10. `django__django-15061`
11. `django__django-15814`
12. `django__django-16046`
13. `django__django-16595`
14. `psf__requests-2674`
15. `sympy__sympy-15346`
16. `sympy__sympy-18189`
17. `sympy__sympy-18698`
18. `sympy__sympy-20049`
19. `sympy__sympy-20154`

### Step 3. 剩余 unresolved 池

在 100 个 predictions 中：

- 11 个空 patch 已剔除
- 19 个已 resolved 已排除
- 剩余 **70** 个为“已进入 harness 且最终 unresolved”的 case

这 70 个 case 是后续 semantic screening 的**上游候选池**，但还**不是** semantic candidate。

## 5. 为什么严格筛选在这里停住

标准文件要求对候选 case 继续回答这些问题：

- 是否真正 **reached repair region**？
- remaining failure 是否更像 **semantic mismatch** 而不是纯 target miss / localization miss？
- 对模糊 case，是否有明确 **post-activation evidence**？
- 是否存在 later retrieval/context rescue，从而触发 **exclusion override**？

这些判断严重依赖 `work3` 本次生成时的 per-instance JSON logs；而这些本地 authority 目前缺失，所以无法对 70 个 unresolved case 做严格、可复核的逐案判定。

## 6. 严格结论

### 6.1 严格按标准可确认筛出的 semantic candidate 数量

**0 个。**

原因不是“70 个都不是”，而是：

- 本地缺少与 `cases1_100_work3` 精确对应的 generation-side per-instance logs
- 因而无法严格完成 base screen / tie-break / exclusion override

### 6.2 当前最稳妥的分类

| 类别 | 数量 | 说明 |
|---|---:|---|
| confirmed semantic candidates | 0 | 证据不足，无法严格确认 |
| unresolved but unscreenable | 70 | benchmark 已知 unresolved，但 generation-side authority 缺失 |
| obvious rejects | 11 | 空 patch，按 Step 1 直接剔除 |
| resolved / excluded | 19 | benchmark 已通过，不属于 semantic candidate |

## 7. 如果做“宽松代理筛选”，最多能看到什么

如果只做非常宽松的代理判断，而**不**坚持标准中的 generation-side authority 约束，那么可继续人工查看这 **70 个 unresolved case** 的 harness 失败表现，形成一个“待人工复核池”。

但这不应被记为“按 2026-04-30 标准完成的严格筛选结果”。

## 8. 下一步建议

如果你希望我继续把 70 个 unresolved case 真正筛成 semantic candidate / reject / ambiguous，建议先补齐或重建这批 generation logs：

1. 恢复或重新生成 `cases1_100_work3` 对应的 per-instance JSON logs
2. 至少保证每个 case 能看到：
   - `selected_files`
   - `patch_chars`
   - `empty_patch_reason`
   - `developer_metrics_history`
   - `selection_reason`
   - `hypothesis_root_cause`
   - `expected_invariant`
   - `patch_strategy`
   - `target_hit` / `target_hit_rate`
3. 然后再按标准文件里的 Step 1–6 对 70 个 unresolved case 做逐案 decision card

## 9. 本次结论一句话版

`cases1_100_work3` 的官方 benchmark 结果是 **19 / 100 通过**；但若严格按 `2026-04-30-semantic-case-selection-standard.md` 筛选，当前由于缺失精确对应的 generation-side per-instance logs，**可严格确认的 semantic candidate 数量为 0，另有 70 个 unresolved case 处于“可疑但暂不可严格筛”的状态。**
