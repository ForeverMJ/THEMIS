# Cases 81-100 Patch Generation Record

Date: 2026-05-02

## Scope

This note records the actual execution of SWE-bench Lite cases 81-100 using `gpt-5.1-codex-mini` for **patch generation only**.

No benchmark evaluation was run for this batch.

---

## Patch-generation command used

```bash
".venv/bin/python" run_swebench_lite_predictions.py \
  --mode integrated \
  --model gpt-5.1-codex-mini \
  --analysis-model gpt-5.1-codex-mini \
  --seed 42 \
  --start 80 \
  --end 100 \
  --workdir swebench_runs/themis_seed42_cases81_100_gpt51_codexmini_work3 \
  --output predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases81_100_work3.jsonl
```

---

## Artifacts produced

- predictions file:
  - `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases81_100_work3.jsonl`
- per-instance run logs:
  - `swebench_runs/themis_seed42_cases81_100_gpt51_codexmini_work3/logs/*.json`

---

## Verified patch-generation result

I verified the generated predictions file directly.

### Counts

- total cases completed: **20**
- non-empty patches: **18**
- empty patches: **2**

### Empty-patch cases

1. `sphinx-doc__sphinx-7686`
2. `sympy__sympy-13915`

---

## Spot-check of batch boundaries

### First 5 case IDs in this batch

1. `sympy__sympy-18189`
2. `psf__requests-2148`
3. `pydata__xarray-4094`
4. `sphinx-doc__sphinx-8801`
5. `pallets__flask-4992`

### Last 5 case IDs in this batch

1. `matplotlib__matplotlib-25433`
2. `django__django-15388`
3. `sympy__sympy-13915`
4. `django__django-15252`
5. `pylint-dev__pylint-5859`

---

## Operational note

This run only generated patch predictions.

No harness command was executed for this batch in this step, so there are currently:

- no benchmark pass/fail labels for cases 81-100 from this run
- no `logs/run_evaluation/...` artifacts tied to this batch yet

---

## Files to inspect later if benchmark is run

- predictions:
  - `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases81_100_work3.jsonl`
- generation logs:
  - `swebench_runs/themis_seed42_cases81_100_gpt51_codexmini_work3/logs/`
