# Cases 41-80 Patch Generation Record

Date: 2026-05-02

## Scope

This note records the actual execution of SWE-bench Lite cases 41-80 using `gpt-5.1-codex-mini` for **patch generation only**.

No benchmark evaluation was run for this batch.

---

## Patch-generation command used

```bash
".venv/bin/python" run_swebench_lite_predictions.py \
  --mode integrated \
  --model gpt-5.1-codex-mini \
  --analysis-model gpt-5.1-codex-mini \
  --seed 42 \
  --start 40 \
  --end 80 \
  --workdir swebench_runs/themis_seed42_cases41_80_gpt51_codexmini_work3 \
  --output predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases41_80_work3.jsonl
```

---

## Artifacts produced

- predictions file:
  - `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases41_80_work3.jsonl`
- per-instance run logs:
  - `swebench_runs/themis_seed42_cases41_80_gpt51_codexmini_work3/logs/*.json`

---

## Verified patch-generation result

I verified the generated predictions file directly.

### Counts

- total cases completed: **40**
- non-empty patches: **35**
- empty patches: **5**

### Empty-patch cases

1. `sympy__sympy-15609`
2. `sphinx-doc__sphinx-10451`
3. `sympy__sympy-18621`
4. `matplotlib__matplotlib-25442`
5. `sphinx-doc__sphinx-10325`

---

## Spot-check of batch boundaries

### First 5 case IDs in this batch

1. `sympy__sympy-15609`
2. `sympy__sympy-13971`
3. `sympy__sympy-20154`
4. `django__django-14534`
5. `sympy__sympy-18532`

### Last 5 case IDs in this batch

1. `django__django-11019`
2. `matplotlib__matplotlib-24970`
3. `django__django-15814`
4. `sympy__sympy-20049`
5. `pydata__xarray-3364`

---

## Operational note

This run only generated patch predictions.

No harness command was executed for this batch in this step, so there are currently:

- no benchmark pass/fail labels for cases 41-80 from this run
- no `logs/run_evaluation/...` artifacts tied to this batch yet

---

## Files to inspect later if benchmark is run

- predictions:
  - `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases41_80_work3.jsonl`
- generation logs:
  - `swebench_runs/themis_seed42_cases41_80_gpt51_codexmini_work3/logs/`
