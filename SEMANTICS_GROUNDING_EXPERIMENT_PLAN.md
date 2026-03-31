# Retrieval-to-semantics experiment pivot

Date: 2026-03-31

## Why this document exists

This document records the retrieval/context-first experiment results after the fault-space neighborhood line, explains why the project should pivot to a semantics-grounding main line, and defines the next staged experiment program.

## Current project position

The repo goal remains improving SWE-bench Lite resolve rate while keeping experiments controlled and interpretable.

Recent work established three important facts:

1. Better fault-space activation alone was not enough.
2. Initial context/file recall can rescue real cases.
3. Some remaining cases are now retrieval-saturated and semantics-limited.

## Retrieval/context-first recap

### Stable comparison point

- `Ablation 1` remains the stable local behavioral baseline.

### Fault-space neighborhood line

- `fault_space_fallback` (related-symbol fallback) was largely inactive.
- `fault_space_neighborhood` fixed activation and could reach fallback targets, but did not improve headline resolution on the discriminative matrix.

### Context-first rescue line

- `fault_space_neighborhood_context` injected one extra implementation file before workflow execution.
- This rescued `django__django-15320` in Stage D and later reproduced that rescue in Phase 1.
- Phase 2 extended the fixed matrix to five cases and improved from `0/5` (`ablation1`) to `2/5` with context-first retrieval.
- The two rescued cases were:
  - `django__django-15320`
  - `psf__requests-3362`

### Retrieval-explicit comparison

- `fault_space_neighborhood_retrieval` kept the same hand-crafted context files and additionally loaded explicit `.py` mentions from the instance text.
- Phase 3 matched the Phase 2 hand-crafted context arm exactly at `2/5` and rescued the same two cases.
- Conclusion: retrieval/context-first is the right high-level direction, but this particular automatic explicit-file heuristic did not beat the simpler hand-crafted context variant.

## Forensic conclusion after Phase 3

The unresolved trio after Phase 3 was:

- `matplotlib__matplotlib-18869`
- `pallets__flask-4045`
- `sympy__sympy-13773`

For all three cases, Phase 2 and Phase 3 stabilized on the same implementation files and produced real patches that applied successfully, but harness still marked them unresolved.

This means the current line reached a boundary:

- **retrieval-first/context-first is still valuable overall**
- but these three cases are now better treated as **retrieval-saturated, semantics-limited**

## New main line: semantics-grounding

### Main question

Once the workflow has the right files, how can we convert failing-test semantics into more correct patches?

### What this line does NOT do

- no new retrieval expansion
- no new fallback mechanics
- no operator/ingredient overhaul
- no broad reranking bundles up front

### What this line does

- strengthen the semantic contract extracted from failing tests
- align candidate scoring with actual SWE-bench harness goals
- test whether semantic grounding can rescue cases already saturated on retrieval

## Improved phased plan

The plan below is the revised version after expert review. It intentionally reduces overlap and keeps attribution clean.

### S0 - Freeze the semantics trio

#### Goal

Freeze the semantics-limited trio and treat it as the semantic benchmark slice.

#### Cases

- `matplotlib__matplotlib-18869`
- `pallets__flask-4045`
- `sympy__sympy-13773`

#### Inputs to preserve for every run

- same model
- same workflow builder
- same retrieval/context preset (`fault_space_neighborhood_context` unless explicitly overridden)
- same revision budget
- same harness evaluation procedure

#### Acceptance criteria

- for each case, record:
  - selected files
  - current FAIL_TO_PASS failures
  - patch strategy / expected invariant / hypothesis root cause
  - exact harness-unresolved reason

#### Stop criteria

- if the semantic failure card for any case is still vague or cannot be tied to the failing tests, do not proceed to S1

### S1 - Contract-aware prompt only

#### Goal

Test the smallest semantic upgrade: explicitly encode the failing-test contract into the repair guidance without changing reranking or retrieval.

#### Code touchpoints

- `src/agents/judge.py`
  - strengthen `expected_behavior`
  - strengthen `minimal_change_hint`
- `run_experiment_integrated.py`
  - thread the failing-test contract into repair-brief/hypothesis prompt construction
- optionally `src/agents/developer.py`
  - ensure prompt sections preserve the contract verbatim

#### Command pattern

Generation:

```bash
".venv/bin/python" run_swebench_lite_predictions.py --mode integrated --experiment-preset <semantic-contract-preset> --instance-id matplotlib__matplotlib-18869 --instance-id pallets__flask-4045 --instance-id sympy__sympy-13773 --workdir swebench_runs_semantics_s1 --output predictions/semantics_s1.jsonl
```

Evaluation:

```bash
".venv/bin/python" -m swebench.harness.run_evaluation --predictions_path predictions/semantics_s1.jsonl --run_id semantics_s1_eval --dataset_name SWE-bench/SWE-bench_Lite --split test --max_workers 1
```

#### Acceptance criteria

- at least one case improves from unresolved to resolved, or
- FAIL_TO_PASS success count increases on at least one case without introducing pass-to-pass regression

#### Stop criteria

- harness outcome is unchanged across all three cases
- only internal explanatory fields improve without any harness effect

### S2 - Harness-aligned semantic reranking (only if S1 passes)

#### Goal

If S1 shows positive signal, add a single reranking layer aligned to SWE-bench grading: fix FAIL_TO_PASS and preserve PASS_TO_PASS.

#### Code touchpoints

- `run_experiment_integrated.py`
  - candidate scoring / selection logic
  - scoring surface should explicitly weight:
    - fail-to-pass improvement likelihood
    - pass-to-pass preservation risk
- no retrieval changes
- no broader verifier stack yet

#### Command pattern

Generation:

```bash
".venv/bin/python" run_swebench_lite_predictions.py --mode integrated --experiment-preset <semantic-rerank-preset> --instance-id matplotlib__matplotlib-18869 --instance-id pallets__flask-4045 --instance-id sympy__sympy-13773 --workdir swebench_runs_semantics_s2 --output predictions/semantics_s2.jsonl
```

Evaluation:

```bash
".venv/bin/python" -m swebench.harness.run_evaluation --predictions_path predictions/semantics_s2.jsonl --run_id semantics_s2_eval --dataset_name SWE-bench/SWE-bench_Lite --split test --max_workers 1
```

#### Acceptance criteria

- improves on S1 harness results, or
- makes an S1 rescue more stable on reruns without increasing regressions

#### Stop criteria

- no improvement over S1
- more complicated ranking but no harness gain

### S3 - Small confirmation only (only if S1 or S2 passes)

#### Goal

Confirm that the winning semantics-grounding variant is not just trio-specific.

#### Scope

- expand only to 5-8 semantics-suspect cases
- keep the winning semantic strategy fixed
- do not add new retrieval or fallback variables here

#### Acceptance criteria

- retains positive signal on the expanded semantics slice

#### Stop criteria

- gain disappears immediately when expanding beyond the trio

## Why this revised plan is preferable

- It isolates one semantic variable at a time.
- It avoids repeating the earlier failure mode of adding many weak signals together.
- It treats harness results as the governing truth signal.
- It uses retrieval-first as an already-proven front-end, not as the main bottleneck anymore.

## Current recommendation

Keep `fault_space_neighborhood_context` as the best practical retrieval/context variant. Do not keep widening retrieval heuristics for the unresolved trio. Treat the next main line as semantic grounding inside already-correct files.
