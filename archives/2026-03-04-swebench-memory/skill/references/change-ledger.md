# Change Ledger

## Repository documents added

- `PROJECT_FLOW_REFERENCE.rst`
  - Purpose: become the mandatory pre-change reference for this repo.
  - Status: retained.
- `AGENTS.md`
  - Purpose: contributor guide for repository layout, commands, style, testing, and PR expectations.
  - Status: retained.

## Retained repository code changes

### 1. Stage 1 - safer UNKNOWN escalation

Files:
- `src/enhanced_graph_manager/violation_flagger.py`
- `tests/enhanced_graph_manager/test_violation_flagger_behavior.py`

Change:
- Added `_should_escalate_unknown(...)`.
- UNKNOWN now escalates to blocking only when the reason is concrete enough and the evidence has a stronger anchor.

Purpose:
- Stop vague graph findings from becoming blocking conflicts that repeatedly drive low-value loops.

Current status:
- Retained.

### 2. Stage 2 - Judge repair brief and policy gate

Files:
- `src/agents/judge.py`
- `run_experiment_integrated.py`
- `run_swebench_lite_predictions.py`
- `src/state.py`
- `tests/test_judge_repair_brief.py`
- `tests/test_repair_brief_policy.py`

Change:
- Judge emits structured `repair_brief` data.
- Workflow records `repair_brief` / `repair_brief_history`.
- `_should_apply_repair_brief(...)` prevents advisory-only briefs from over-constraining Developer.
- `src/state.py` gained Python 3.9-compatible `NotRequired` import handling.

Purpose:
- Separate gate information from coaching information.
- Preserve richer Judge output without degrading advisory-only runs.

Current status:
- Retained.

## Rolled-back repository experiments

### Generic direct VIOLATES downgrade

Files touched during experiment:
- `src/enhanced_graph_manager/violation_flagger.py`
- `tests/enhanced_graph_manager/test_violation_flagger_behavior.py`

Change:
- Generic direct `VIOLATES` reasons such as `No validation functions found in codebase` were downgraded to `UNKNOWN` / `ADVISORY`.

Purpose:
- Remove false blocking loops, especially on `django__django-11848`.

Observed effect:
- Internal metrics improved for `django__django-11848` (`blocking [3,3,3] -> [0]`).
- 6-case SWE-bench Lite cohort did not improve solve rate.
- Worse: `django__django-12497` regressed from `PASS` to `APPLY_FAIL`.

Current status:
- Rolled back from repo code.

### Stage 3 family (all rolled back)

Themes attempted:
- repeated blocking fingerprint
- target-hit hard constraint
- runtime target resolver
- baseline symbol anchor
- symbol preservation guard
- body-only rewrite auto-heal
- behavior-gap prompt enrichment
- Judge-side canonical target normalization

Why rolled back:
- No harness improvement.
- Often reduced `effective_modification_rate` or caused `unknown symbol` / malformed patch behavior.

## Local-only harness changes retained for evaluation

These are **not** repo logic changes. They live in `.venv/lib/python3.9/site-packages/swebench/harness/*` and exist only to make local SWE-bench evaluation runnable on this machine.

Files:
- `.venv/lib/python3.9/site-packages/swebench/harness/context_manager.py`
- `.venv/lib/python3.9/site-packages/swebench/harness/utils.py`

Local fixes:
- Preserve parent environment and locale in harness subprocesses.
- Split Miniconda install from `conda init` so shell operators are not treated as argv.
- Reuse local `swebench_runs_*/repos/django__django` repos instead of flaky network clone.
- Copy full local repos instead of `git clone` from detached local source so old commits remain checkout-able.
- Require local repo candidate to contain needed commit(s).

Other local environment changes:
- Installed `swebench==1.1.5` into `.venv` because newer versions required unsupported syntax under local Python 3.9.
- Added `/tmp/wget` wrapper using `curl` because old harness shell-outs expected `wget`.

Use these notes when reproducing local evaluation. Do not confuse them with product changes.
