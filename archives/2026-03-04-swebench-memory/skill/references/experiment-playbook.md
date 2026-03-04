# Experiment Playbook

## Current evaluation principle

Judge method changes by SWE-bench Lite harness outcome.
Do **not** keep a method change only because blocking metrics or loop counts improved.

## Fixed 6-case cohort

Use this exact cohort unless the user explicitly changes it:
- `django__django-11099`
- `django__django-11620`
- `django__django-11848`
- `django__django-12284`
- `django__django-12497`
- `django__django-12589`

Rationale:
- `11099`, `12284`: patch/application edge cases (`APPLY_FAIL` risk)
- `11620`, `11848`: behavior-level failures under current method
- `12497`: historically had a `PASS` baseline and therefore protects against regression
- `12589`: non-empty patch but still harness-failing baseline

## Frozen run config

Use:
- mode: `integrated`
- workflow builder: `run_experiment_integrated:build_integrated_workflow_conflict_only`
- analysis strategy: `auto_select`
- model: `gpt-4o-mini`
- analysis model: `gpt-4o-mini`
- max revisions: `3`
- recursion limit: `120`

## Baseline command

```bash
HF_HOME=/Users/tao/.cache/huggingface \
HF_DATASETS_CACHE=/Users/tao/.cache/huggingface/datasets \
HF_HUB_CACHE=/Users/tao/.cache/huggingface/hub \
.venv/bin/python run_swebench_lite_predictions.py \
  --instance-id django__django-11099 \
  --instance-id django__django-11620 \
  --instance-id django__django-11848 \
  --instance-id django__django-12284 \
  --instance-id django__django-12497 \
  --instance-id django__django-12589 \
  --mode integrated \
  --workflow-builder run_experiment_integrated:build_integrated_workflow_conflict_only \
  --analysis-strategy auto_select \
  --model gpt-4o-mini \
  --analysis-model gpt-4o-mini \
  --max-revisions 3 \
  --recursion-limit 120 \
  --output predictions/cohort6_baseline.jsonl \
  --workdir swebench_runs_cohort6_baseline
```

## Harness command pattern

Use the local patched harness environment:

```bash
PATH=/tmp:$PATH LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONUTF8=1 \
HF_HOME=/Users/tao/.cache/huggingface \
HF_DATASETS_CACHE=/Users/tao/.cache/huggingface/datasets \
HF_HUB_CACHE=/Users/tao/.cache/huggingface/hub \
.venv/bin/python -m swebench.harness.run_evaluation \
  --predictions_path <predictions.jsonl> \
  --log_dir <eval_log_dir> \
  --swe_bench_tasks SWE-bench/SWE-bench_Lite \
  --testbed <testbed_dir> \
  --timeout 1800 \
  --num_processes 1 \
  --verbose
```

## Hard rollback rule

Keep a candidate method **only if** all conditions hold:
1. candidate `PASS` count is strictly greater than baseline `PASS` count;
2. no baseline `PASS` case regresses to `FAIL`, `APPLY_FAIL`, or `RESET_FAIL`;
3. no new patch-application failure appears on a previously passing case.

Otherwise roll back the repo change immediately.

## Known baseline result from this session

Baseline harness result on the fixed cohort:
- `django__django-11099`: `APPLY_FAIL`
- `django__django-11620`: `FAIL`
- `django__django-11848`: `FAIL`
- `django__django-12284`: `APPLY_FAIL`
- `django__django-12497`: `PASS`
- `django__django-12589`: `FAIL`

Aggregate:
- `PASS 1 / FAIL 3 / APPLY_FAIL 2`

## Candidate result that was rejected

Candidate: generic direct `VIOLATES` downgrade.

Harness result:
- `django__django-11099`: `APPLY_FAIL`
- `django__django-11620`: `FAIL`
- `django__django-11848`: `FAIL`
- `django__django-12284`: `APPLY_FAIL`
- `django__django-12497`: `APPLY_FAIL`
- `django__django-12589`: `FAIL`

Aggregate:
- `PASS 0 / FAIL 3 / APPLY_FAIL 3`

Reason rejected:
- No solve-rate gain.
- Regressed `django__django-12497` from `PASS` to `APPLY_FAIL`.

## What this implies for next methods

Do not prioritize gate softening or blocking-count optimization.
Prioritize methods that improve patch executability and patch correctness, especially:
- reducing `APPLY_FAIL`
- reducing malformed rewrites
- adding execution-grounded critique before final patch acceptance
