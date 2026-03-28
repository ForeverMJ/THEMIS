# Fault-space neighborhood experiments

Date: 2026-03-28

## Goal

Continue the fault-space line under a controlled setup and test whether file-local structural neighborhood expansion can convert better target activation into higher SWE-bench Lite resolution.

## Baseline and presets

- Behavioral baseline: `ablation1`
- Deprecated comparison line: `fault_space_fallback`
- Current comparison line: `fault_space_neighborhood`

All presets are exposed through `run_swebench_lite_predictions.py --experiment-preset ...`.

## What changed in code

### Workflow

- Added explicit preset builders for:
  - `build_integrated_workflow_ablation1`
  - `build_integrated_workflow_fault_space_fallback`
  - `build_integrated_workflow_fault_space_neighborhood`
- Added fault-space instrumentation for:
  - `entered_fallback`
  - `primary_targets`
  - `expanded_targets`
  - `fallback_added_targets`
  - `fallback_target_hit`
  - `selected_target_source`
  - `fallback_would_have_triggered_but_not_used_reason`
  - `target_expansion_mode`
  - `expansion_anchor_file`
  - `expansion_anchor_symbol`

### Developer / state plumbing

- Added structured developer outputs for:
  - search/replace edits
  - chosen repair hypothesis
  - patch strategy metadata
- Extended `AgentState` to preserve the extra repair and failure-analysis fields used by the integrated workflow.

### Runner surface

- Added experiment preset resolution in `run_swebench_lite_predictions.py`
- Logged effective preset / workflow builder / analysis strategy / revision cap into per-instance run metadata

## Validation steps completed

Focused test command:

```bash
".venv/bin/python" -m pytest tests/test_experiment_presets.py tests/test_repair_brief_policy.py tests/test_judge_repair_brief.py -q
```

Result at implementation time: `13 passed`

## Experiment summary

### Stage A - single-case gate

Case:

- `django__django-15996`

Question:

- Does neighborhood expansion actually activate and improve target-hit behavior?

Result:

- `ablation1`: `entered_fallback = false`, `target_hit_rate = 0.0`
- `fault_space_neighborhood`: `entered_fallback = true`, `target_hit_rate = 0.25`

Interpretation:

- Neighborhood expansion moved this line from inactive fallback to active, file-local target expansion.

### Stage B - five-case discriminative matrix

Cases:

- `django__django-12497`
- `django__django-15320`
- `django__django-15996`
- `matplotlib__matplotlib-18869`
- `sympy__sympy-13773`

Harness results:

- `ablation1`: `1/5 resolved`
- `fault_space_neighborhood`: `1/5 resolved`

Important internal signals:

- All neighborhood runs entered fallback
- `django__django-15320`: `fallback_target_hit = true`, `selected_target_source = fallback`
- `matplotlib__matplotlib-18869`: `fallback_target_hit = true`, `selected_target_source = fallback`

Interpretation:

- Neighborhood expansion fixed the activation problem, but did not improve headline resolution.

### Stage C - conversion-gap mini experiment

Focal cases:

- `django__django-15320`
- `matplotlib__matplotlib-18869`
- control: `sympy__sympy-13773`

Commands:

```bash
".venv/bin/python" run_swebench_lite_predictions.py --mode integrated --experiment-preset ablation1 --instance-id django__django-15320 --instance-id matplotlib__matplotlib-18869 --instance-id sympy__sympy-13773 --workdir swebench_runs_stageC_conversion_ablation1 --output predictions/stageC_conversion_ablation1.jsonl

".venv/bin/python" run_swebench_lite_predictions.py --mode integrated --experiment-preset fault_space_neighborhood --instance-id django__django-15320 --instance-id matplotlib__matplotlib-18869 --instance-id sympy__sympy-13773 --workdir swebench_runs_stageC_conversion_neighborhood --output predictions/stageC_conversion_neighborhood.jsonl

".venv/bin/python" -m swebench.harness.run_evaluation --predictions_path predictions/stageC_conversion_ablation1.jsonl --run_id stageC_conversion_ablation1_eval --dataset_name SWE-bench/SWE-bench_Lite --split test --max_workers 1

".venv/bin/python" -m swebench.harness.run_evaluation --predictions_path predictions/stageC_conversion_neighborhood.jsonl --run_id stageC_conversion_neighborhood_eval --dataset_name SWE-bench/SWE-bench_Lite --split test --max_workers 1
```

Harness results:

- `ablation1`: `0 resolved / 2 unresolved / 1 empty patch`
- `fault_space_neighborhood`: `0 resolved / 3 unresolved / 0 empty patch`

Per-case comparison:

- `django__django-15320`: unresolved -> unresolved
- `matplotlib__matplotlib-18869`: empty patch -> unresolved
- `sympy__sympy-13773`: unresolved -> unresolved

Key harness reports:

- `django__django-15320`: patch applied, still failed `test_subquery_sql`
- `matplotlib__matplotlib-18869`: baseline produced no effective patch; neighborhood produced a real patch but still failed `test_parse_to_version_info[...]`
- `sympy__sympy-13773`: patch applied, still failed `test_matmul`

Interpretation:

- This is a null-to-weak-negative result for end-to-end resolution.
- The strongest positive signal is qualitative: neighborhood expansion turned one no-op / empty-patch case into a real patch attempt.
- The strongest negative signal is that fallback-target hits still did not convert into harness-resolved fixes.

## Current conclusion

The current fault-space neighborhood line is more promising than the older related-symbol fallback because it actually activates and can reach fallback targets. However, it is still bottlenecked downstream of localization.

The immediate research focus should therefore move from “expand target coverage” to “understand why fallback-hit patches still fail harness tests”.

## Recommended next step

Run a narrow forensic pass on the Stage C traces and isolate one single semantic failure hypothesis that explains why:

- `django__django-15320` still fails `test_subquery_sql`
- `matplotlib__matplotlib-18869` still fails `test_parse_to_version_info[...]`
- `sympy__sympy-13773` remains a semantics-space failure even after fallback selection

Do not widen the matrix or add new retrieval/ranking variables until that conversion gap is understood.
