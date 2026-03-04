# Conversation Archive

This is a condensed turn-by-turn archive of the session that led to the current THEMIS SWE-bench state.
It preserves the user requests, decisions, and conclusions so future work does not repeat dead ends.

## 1. Repository grounding

- User asked for a full-project analysis and a reference document that must be checked before every future code change.
- Created `PROJECT_FLOW_REFERENCE.rst` to capture architecture, workflows, risk points, and validation rules.
- User later asked for contributor guidance; created `AGENTS.md` for repo structure, commands, style, and PR expectations.

## 2. SWE-bench runner questions

- User asked which options `python run_swebench_lite_predictions.py` supports.
- Listed runner flags and example invocations.
- User asked whether `graph-only` and `conflict-only` existed.
- Confirmed `graph_only` exists as analysis strategy and `conflict_only` exists through the integrated workflow builder.

## 3. Early loop experiment on `django__django-12497`

- User wanted to reproduce earlier 5-round Judge/Developer loop testing.
- Proposed a 5-round main experiment and 1-round control.
- User fixed scope to: use `gpt-4o-mini`, run only the 5-round main experiment first.
- First execution failed because `git fetch` to GitHub was unavailable.
- After GitHub recovered, reran the experiment.
- Result: 5 rounds did not reduce blocking; blocking stayed `1` for all rounds, advisory stayed `15`, later rounds degraded to no-op.

## 4. Judge strategy diagnosis

- User asked what the current Judge output strategy was.
- Identified current behavior: hard graph gate first, soft LLM advisory only when blocking is already clear.
- Concluded that Developer often receives graph summaries rather than executable repair instructions.
- User then asked for a broader expert plan around three issues:
  - Judge feedback not producing effective edits
  - whether the two-layer strategy should change
  - why blocking stayed unresolved
- Proposed staged remediation: tighten blocking generation, decouple gate/coaching, narrow Developer targets, add stronger loop discipline.

## 5. Stage-based method experiments

### Stage 1
- User approved stage 1.
- Implemented safer UNKNOWN escalation in `violation_flagger.py`.
- Result on `django__django-12497`: avoided false blocking and useless loops.
- Retained.

### Stage 2
- User approved stage 2.
- Implemented structured `repair_brief`, state logging, and advisory-only guard logic.
- Result: preserved stage-1 convergence and added richer observability.
- Retained.

### Stage 3 and follow-up variants
- User asked to continue with stricter target/fingerprint methods.
- Tried several variants:
  - repeated blocking fingerprint
  - target-hit hard constraints
  - target resolver
  - baseline anchor
  - symbol preservation guard
  - body-only rewrite auto-heal
  - behavior-gap prompt enrichment
  - canonical target normalization
- These consistently failed one of the rollback conditions:
  - no blocking improvement
  - lower effective modification rate
  - unknown symbol errors
  - malformed rewrites
- All of these were rolled back.

## 6. Root-cause shift: from gate logic to patch correctness

- After repeated failures, the focus shifted from target routing to patch-level failure analysis.
- For `django__django-11848`, the crucial finding was:
  - Developer could target `parse_http_date`, but the produced patch was still wrong.
  - Graph/Judge did not reliably catch this semantic failure.
- This led to the hypothesis that false blocking reduction alone would not improve solve rate.

## 7. Generic direct violation downgrade experiment

- Implemented a candidate method that downgraded generic direct `VIOLATES` to advisory.
- Internal metrics improved on `django__django-11848`:
  - blocking dropped from `[3,3,3]` to `[0]`
  - revision count dropped to `0`
- User then clarified the real acceptance bar: the patch must pass SWE-bench Lite harness.
- At this point the session moved from workflow metrics to benchmark-grounded validation.

## 8. Bringing up local SWE-bench harness

The old harness did not run cleanly on the machine. Multiple local environment fixes were required:

- `swebench` was not installed in `.venv`.
- Newer `swebench` versions were incompatible with local Python 3.9, so `swebench==1.1.5` was installed.
- Old harness expected `wget`; a `/tmp/wget` wrapper using `curl` was added.
- Locale and subprocess environment propagation had to be patched.
- Miniconda install and `conda init` had to be separated in the harness.
- Local repo reuse and full-copy fallback had to be added because remote clone and detached local clone approaches were unstable.

These changes were explicitly treated as local evaluation infrastructure, not product changes.

## 9. Single-case harness result for `django__django-11848`

- Harness finally ran for `django__django-11848`.
- Result: `FAIL`.
- Failure mode: the generated patch used current time logic in a way that broke tests when `datetime` was mocked.
- Conclusion: the generic-blocking candidate improved internal metrics but did not solve the benchmark case.

## 10. Cohort decision by the user

- User asked whether one failed case should be enough to reject a method.
- Recommendation: do not reject a method from a single case; evaluate on a fixed cohort.
- User agreed and asked for strict rollback rules, then asked to execute directly.

## 11. Fixed 6-case cohort experiment

Defined fixed cohort:
- `django__django-11099`
- `django__django-11620`
- `django__django-11848`
- `django__django-12284`
- `django__django-12497`
- `django__django-12589`

Keep rule:
- candidate `PASS` count must exceed baseline `PASS` count;
- no baseline `PASS` may regress.

Executed baseline predictions and harness.
Executed candidate predictions and harness.

### Baseline harness result
- `11099`: `APPLY_FAIL`
- `11620`: `FAIL`
- `11848`: `FAIL`
- `12284`: `APPLY_FAIL`
- `12497`: `PASS`
- `12589`: `FAIL`
- Aggregate: `PASS 1 / FAIL 3 / APPLY_FAIL 2`

### Candidate harness result
- `11099`: `APPLY_FAIL`
- `11620`: `FAIL`
- `11848`: `FAIL`
- `12284`: `APPLY_FAIL`
- `12497`: `APPLY_FAIL`
- `12589`: `FAIL`
- Aggregate: `PASS 0 / FAIL 3 / APPLY_FAIL 3`

## 12. Final conclusion of the session

- The generic direct violation downgrade is not a valid method improvement.
- It was rolled back from repo code.
- The correct next method direction is not more gate softening.
- The next method should be execution-grounded: reduce `APPLY_FAIL`, malformed rewrites, and semantically invalid patches before final patch submission.
