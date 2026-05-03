# Semantic Case Selection Standard for Future Contract Experiments

Date: 2026-04-30

## Purpose

This note records the **previous semantic-case screening standard** used in this repository and the **artifacts that must be prepared** before selecting new cases for semantic-contract experiments.

The goal is to let future selection work reuse the same logic instead of re-inventing the criteria.

---

## Canonical local evidence

Use these files as the local authority for the prior standard:

1. `session-ses_2487.md`
   - tie-break rule
   - frozen trio rationale
   - promoted fourth-case rationale
2. `SEMANTICS_GROUNDING_EXPERIMENT_PLAN.md`
   - semantics-limited trio framing
   - retrieval-saturated / semantics-limited interpretation
3. `FAULT_SPACE_NEIGHBORHOOD_EXPERIMENTS.md`
   - activation evidence
   - unresolved-after-activation evidence

Important: the exact four-case promotion rule was preserved most explicitly in `session-ses_2487.md`, not in a cleaner standalone paper-side document.

---

## Previously used semantic screening standard

## 1. Base screen

A case is eligible for semantic review only if all of the following are true:

1. It remains **unresolved** after a bounded-fault-space / graph-oriented run.
2. There is local evidence that the workflow **reached the repair region** or at least a nearby target region.
3. The remaining failure looks more like a **semantic mismatch** than a pure localization / target-miss failure.

Short form:

- unresolved
- reached repair region
- still failing for semantic reasons

---

## 2. Tie-break rule for ambiguous candidates

If a case is still ambiguous after the base screen, default to **exclude** unless both conditions below are satisfied:

1. There is **explicit post-activation evidence** in the same experimental line.
   - Typical examples:
     - `entered_fallback = true`
     - `target_hit_rate > 0`
     - `fallback_target_hit = true`
     - clear `selected_target_source = fallback` or equivalent local evidence
2. The case remains **persistently unresolved after that activation**.

This prevents promoting cases where the workflow never really got near the right area.

---

## 3. Exclusion override

Even if a case looks semantics-suspect, exclude it from the semantics-only promoted subset if later evidence shows that a **retrieval/context change already rescued it**.

Reason:

- once retrieval/context rescue works, the case is no longer a clean semantics-only follow-on example

---

## 4. Practical interpretation of “semantic candidate”

In this repository, a good semantic-contract candidate is usually a case with this shape:

- the runner chose a plausible file or nearby file
- the developer produced a real patch or at least a coherent repair hypothesis
- benchmark still said unresolved
- the failure seems tied to:
  - expected behavior interpretation
  - exception semantics
  - output ordering / formatting semantics
  - protocol behavior (`NotImplemented` vs `ValueError`, etc.)
  - preserving invariant behavior while fixing one path

Bad semantic candidates usually look like:

- no patch
- empty patch
- wrong file selected with weak evidence
- pure infra/build failure
- obvious localization miss

---

## 5. Previous four-case outcome, for reference

The previously frozen 4-case semantic slice was:

1. `matplotlib__matplotlib-18869`
2. `pallets__flask-4045`
3. `sympy__sympy-13773`
4. `django__django-15996`

Interpretation:

- the first three were the frozen semantics-limited trio
- `django__django-15996` was promoted with the tie-break rule because it had:
  - explicit activation improvement
  - persistent unresolved outcome
  - no later context-first rescue evidence

---

## Artifacts required before selecting new semantic cases

For each candidate case, prepare both **generation-side** and **benchmark-side** evidence.

## A. Patch-generation artifacts

Required per case:

1. prediction record
   - `instance_id`
   - `model_patch`
2. per-instance run log JSON
   - `selected_files`
   - `experiment_preset`
   - `workflow_builder`
   - `analysis_strategy`
   - `max_revisions`
   - `patch_chars`
   - `meta.*`

Important fields to preserve for screening:

- `selected_files`
- `patch_chars`
- `empty_patch_reason`
- `developer_metrics_history`
- `selection_reason`
- `hypothesis_root_cause`
- `expected_invariant`
- `patch_strategy`
- `entered_fallback`
- `fallback_target_hit`
- `target_hit`
- `target_hit_rate`
- `selected_target_source`
- `failure_class`
- `fault_space_signal`
- `semantics_space_signal`

## B. Benchmark artifacts

Required per case:

1. harness result / report
2. eval log
3. resolved vs unresolved outcome
4. FAIL_TO_PASS result
5. evidence of regressions if any

---

## Minimal preparation checklist for a future 100-case sweep

Before screening cases from a larger pool (for example, the first 100 SWE-bench Lite cases), make sure all of the following are uniform:

1. Same model family
   - Example: `gpt-5.1-codex-mini`
2. Same broad workflow line
   - do not mix unrelated strategies without labeling them
3. Same evaluation method
   - same harness and same success interpretation
4. Per-case artifacts saved consistently
   - prediction output
   - run log JSON
   - harness log/report

If these are mixed, the semantic slice becomes hard to defend.

---

## Recommended future screening workflow

When screening a larger pool, use this order:

### Step 1. Remove obvious non-candidates

Drop cases with:

- `patch_chars = 0`
- `empty_patch_reason = no_effective_changes_in_workflow`
- infra-only failures
- missing benchmark result
- obvious wrong-file selection with no stronger evidence

### Step 2. Build a semantics-suspect pool

Keep cases that look like:

- real patch or coherent repair hypothesis exists
- selected file seems plausible or near-plausible
- benchmark still unresolved
- failure appears behavior/invariant/protocol/output-semantic related

### Step 3. Apply base screen

Retain only cases that are:

- unresolved
- reached repair region
- still semantic-limited

### Step 4. Apply tie-break for ambiguous cases

For borderline cases, require:

- explicit activation evidence
- unresolved after activation

### Step 5. Apply exclusion override

Exclude cases later rescued by retrieval/context improvements.

### Step 6. Produce a decision card per case

For each retained or rejected case, record:

- `instance_id`
- candidate / reject / ambiguous
- reached repair region? how?
- unresolved after activation?
- semantics-suspect or fault-space-suspect?
- exclusion override triggered?
- final decision
- evidence file paths

---

## Output format recommended for future screening

Use a table like this:

| instance_id | patch? | reached target region? | unresolved? | semantics-suspect? | later retrieval rescue? | final decision |
|---|---:|---|---|---|---|---|

And add a short note per row when the judgment is not obvious.

---

## Warning about known failure mode

Do not automatically trust the selected file.

The current runner can sometimes select a file from weak lexical signals instead of the true patch site. For semantic screening, treat cases like this carefully:

- selected file and advanced-analysis target disagree
- developer proposes edits against a different file than the selected context file
- patch is empty because edits were not applicable to the loaded file set

Such cases may be:

- true semantic cases
- or selector-confounded false positives

They need manual review before entering a semantic-contract pool.

---

## Short operational rule

If a case is unresolved, appears to have reached the right repair neighborhood, and still fails for behavior/invariant reasons after activation, it is a semantic-contract candidate **unless** later retrieval/context rescue evidence disqualifies it.
