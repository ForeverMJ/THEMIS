# Q4/Q7: Compact Quantitative Comparison Table

## Status
Updated after the executed four-case extension.

## Reusable table

| Case ID | Slice role | S1 control | S1 contract | S2 rerank | Key interpretation note |
|---|---|---:|---:|---:|---|
| `matplotlib__matplotlib-18869` | original trio | unresolved | unresolved | unresolved | Known `test_parse_to_version_info[...]` targets exist, but no arm produced a resolved harness outcome. |
| `pallets__flask-4045` | original trio | unresolved | unresolved | unresolved | The two blueprint target tests are known from metadata, but all three arms remained unresolved. |
| `sympy__sympy-13773` | original trio | unresolved | resolved | resolved | `test_matmul` remains the only resolved case; rerank matches rather than exceeds the contract gain. |
| `django__django-15996` | promoted extension case | unresolved | unresolved | unresolved | The beyond-trio `test_serialize_enum_flags` case stayed unresolved in all three arms. |
| **Frozen trio total** | original trio aggregate | **0/3** | **1/3** | **1/3** | The original trio signal remains intact. |
| **Four-case total** | trio + extension | **0/4** | **1/4** | **1/4** | The extension supports persistence of the trio pattern, but does not strengthen it with a new beyond-trio rescue. |

## Short interpretation note
The executed extension is confirmatory but not amplifying: adding `django__django-15996` does not overturn the original trio pattern, yet it also does not create a stronger success story. The safest manuscript reading is therefore a **small controlled study with one executed beyond-trio confirmation attempt**, not a broader generalization claim.

## Table provenance
- Four-case harness outcomes: `gpt-5.1-codex-mini.semantics_4case_s1_control_eval.json`, `gpt-5.1-codex-mini.semantics_4case_s1_contract_eval.json`, `gpt-5.1-codex-mini.semantics_4case_s2_rerank_eval.json`
- Benchmark target-test identities: task metadata under `research_repos/auto-code-rover/results/acr-run-2/applicable_patch/*/meta.json`
- Aggregate run/provenance summary: `paper/work1/Q5_FOUR_CASE_EXECUTION_REPORT.md`

## Suggested manuscript placement
Insert in Results immediately after the paragraph introducing the four-case extension and before the scope-bounding interpretation paragraph.
