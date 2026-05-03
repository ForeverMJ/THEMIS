# Cases 21-40 Patch Generation and Benchmark Record

Date: 2026-05-01

## Scope

This record captures the actual execution of SWE-bench Lite cases 21-40 using `gpt-5.1-codex-mini`, including:

- patch generation
- patch emptiness check
- harness evaluation attempt
- verified benchmark outcomes available before evaluation stopped

Execution stopped after this batch, per user instruction.

---

## Patch-generation command used

```bash
".venv/bin/python" run_swebench_lite_predictions.py \
  --mode integrated \
  --model gpt-5.1-codex-mini \
  --analysis-model gpt-5.1-codex-mini \
  --seed 42 \
  --start 20 \
  --end 40 \
  --workdir swebench_runs/themis_seed42_cases21_40_gpt51_codexmini_work3 \
  --output predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases21_40_work3.jsonl
```

The batch initially timed out and was resumed with the same command. Because the same output file was reused, already-completed instance IDs were skipped automatically and the batch was completed incrementally.

---

## Patch-generation artifacts

- Predictions file:
  - `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases21_40_work3.jsonl`
- Per-instance run logs:
  - `swebench_runs/themis_seed42_cases21_40_gpt51_codexmini_work3/logs/*.json`

---

## Patch-generation summary

Verified counts from the predictions file:

- total cases completed in this batch: **20**
- non-empty patches: **18**
- empty patches: **2**

Empty-patch cases:

1. `sympy__sympy-14396`
2. `matplotlib__matplotlib-23299`

All case IDs in this batch:

1. `sympy__sympy-14317`
2. `astropy__astropy-14182`
3. `sympy__sympy-11870`
4. `sympy__sympy-15345`
5. `mwaskom__seaborn-3010`
6. `matplotlib__matplotlib-24334`
7. `sympy__sympy-14396`
8. `scikit-learn__scikit-learn-25747`
9. `django__django-15202`
10. `astropy__astropy-7746`
11. `sympy__sympy-21171`
12. `matplotlib__matplotlib-23299`
13. `scikit-learn__scikit-learn-13142`
14. `django__django-12308`
15. `sympy__sympy-13031`
16. `pytest-dev__pytest-8365`
17. `sympy__sympy-15346`
18. `django__django-16229`
19. `django__django-16820`
20. `django__django-13158`

---

## Harness evaluation command used

Current installed harness required explicit `--log_dir`, `--swe_bench_tasks`, and `--testbed`.

```bash
mkdir -p "logs/run_evaluation/cases21_40_work3" "testbed/cases21_40_work3" && \
".venv/bin/python" -m swebench.harness.run_evaluation \
  --predictions_path predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases21_40_work3.jsonl \
  --log_dir logs/run_evaluation/cases21_40_work3 \
  --swe_bench_tasks SWE-bench/SWE-bench_Lite \
  --testbed testbed/cases21_40_work3 \
  --timeout 900 \
  --num_processes 1
```

Evaluation artifacts:

- eval logs:
  - `logs/run_evaluation/cases21_40_work3/gpt-5.1-codex-mini/*.eval.log`
- testbed logs:
  - `logs/run_evaluation/cases21_40_work3/gpt-5.1-codex-mini/testbed_*.log`

---

## Verified evaluation progress

The harness did **not** finish the full 20-case batch. It aborted during testbed setup for a later repository family after partially evaluating the batch.

At the time of stop, there were **10** per-case eval logs written.

### Verified per-case status from eval logs

#### Passed

1. `sympy__sympy-15346`
   - eval log indicates `>>>>> All Tests Passed`

#### Failed (tests ran and failed)

1. `mwaskom__seaborn-3010`
2. `sympy__sympy-13031`
3. `sympy__sympy-14317`
4. `sympy__sympy-15345`
5. `sympy__sympy-21171`

#### Install failed / environment failed before test execution

1. `astropy__astropy-14182`
   - install failure
2. `astropy__astropy-7746`
   - install failure

#### Timed out / empty patch / special failure mode

1. `sympy__sympy-11870`
   - test script timed out after 900 seconds
2. `sympy__sympy-14396`
   - empty patch caused `git apply` failure (`No valid patches in input`)

---

## Cases not yet evaluated before harness abort

These case IDs were present in the predictions file but no eval log had been created yet at stop time:

1. `matplotlib__matplotlib-24334`
2. `scikit-learn__scikit-learn-25747`
3. `django__django-15202`
4. `matplotlib__matplotlib-23299`
5. `scikit-learn__scikit-learn-13142`
6. `django__django-12308`
7. `pytest-dev__pytest-8365`
8. `django__django-16229`
9. `django__django-16820`
10. `django__django-13158`

---

## Verified harness blocker

The harness abort that stopped the batch occurred during environment creation for a Matplotlib testbed.

Observed blocker:

- `pikepdf` wheel build failed
- missing `qpdf` headers such as:
  - `qpdf/Constants.h`
  - `qpdf/QPDFJob.hh`

This failure occurred while creating the Matplotlib testbed environment and prevented further evaluation progress in the current sequential run.

---

## Important interpretation notes

1. Patch generation for 21-40 was completed successfully as a batch.
2. The batch contains **2 empty-patch cases**.
3. Benchmark evaluation was only **partially completed** before an external environment/dependency blocker stopped progress.
4. Therefore, this run currently provides:
   - full patch-generation evidence
   - partial benchmark evidence
   - one confirmed pass
   - several confirmed test failures
   - several infra/install failures
   - several still unevaluated cases

---

## Files to inspect next if work resumes later

- predictions:
  - `predictions/gpt-5.1-codex-mini.themis_swebench_lite_seed42_cases21_40_work3.jsonl`
- generation logs:
  - `swebench_runs/themis_seed42_cases21_40_gpt51_codexmini_work3/logs/`
- evaluation logs:
  - `logs/run_evaluation/cases21_40_work3/gpt-5.1-codex-mini/`
- testbeds:
  - `testbed/cases21_40_work3/`
