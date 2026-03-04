# THEMIS Project Flow Reference

This document is the pre-change reference for this repository.
Before any code modification, review this file first.

## 1) Goal and Scope

THEMIS is a code analysis and repair framework centered on:

- structural graph analysis of Python code,
- requirement-to-code consistency checks,
- LLM-driven code revision,
- and optional advanced semantic analysis.

The repo supports two primary usage tracks:

- Experiment track (single case from `experiment_data/`).
- SWE-bench Lite track (batch patch generation across many repos).

## 2) High-Level Architecture

Core state contract lives in `src/state.py` as `AgentState`.
The workflow graph passes this state across nodes.

Main state fields:

- `files`: filename -> content map (the editable source of truth).
- `requirements`: requirement text (issue + constraints).
- `knowledge_graph`: latest graph under analysis.
- `baseline_graph`: graph from original code before edits.
- `conflict_report`: judge result that drives next revision.
- `revision_count`: loop counter.
- optional metrics/analysis fields for integrated workflows.

Key engines/components:

- `src/agents/developer.py`: applies symbol-level rewrites from LLM output.
- `src/agents/judge.py`: hard-checks graph conflicts, soft-checks via LLM advisory.
- `src/graph_manager.py`: legacy AST + LLM requirement linking.
- `src/enhanced_graph_manager/*`: structural extractor + semantic injector + dependency tracer + violation flagger.
- `src/enhanced_graph_adapter.py`: unifies advanced analysis and enhanced graph workflows.
- `src/advanced_code_analysis/*`: multi-stage semantic analysis pipeline.

## 3) Entry Points and Runtime Modes

### A. Traditional (legacy graph path)

- script: `run_experiment.py`
- workflow builder: `src/main.py::build_workflow`
- node order:
  1. `initial_graph_builder` (baseline graph)
  2. `developer`
  3. `graph_builder`
  4. `judge`
  5. conditional loop (`MAX_REVISIONS`)

### B. Traditional Enhanced (recommended structural baseline)

- script: `run_experiment_enhanced.py`
- workflow builder: `src/main_enhanced.py::build_workflow`
- same main loop as A, but graph operations use `EnhancedGraphAdapter` compatibility wrappers:
  - `parse_code_structure`
  - `enrich_with_requirements`
  - `get_analysis_report`

### C. Advanced Analysis Only (semantic analysis without automatic code rewrite)

- script: `run_experiment_advanced.py`
- engine: `EnhancedGraphAdapter.analyze(...)`
- returns findings/recommendations/confidence and can compare against traditional run.

### D. Integrated Workflow (advanced + graph + developer + judge)

- script: `run_experiment_integrated.py`
- workflow builder: `build_integrated_workflow`
- node order:
  1. `advanced_analysis_step` (async semantic analysis)
  2. `initial_graph_builder`
  3. `developer`
  4. `graph_builder`
  5. `judge`
  6. conditional loop (`stop_policy`, `MAX_REVISIONS`)

Integrated mode also tracks:

- developer metrics (effective edits, symbol hit rate),
- conflict metrics (blocking/advisory deltas),
- loop summary and optional fallback to last effective files.

### E. SWE-bench Lite batch generation

- script: `run_swebench_lite_predictions.py`
- per instance:
  1. load dataset/sample instance
  2. checkout repo/base commit
  3. select target file(s)
  4. run selected workflow builder (integrated or traditional enhanced)
  5. apply updated files and generate patch via `git diff`
  6. write prediction + per-instance logs
  7. reset/clean repo checkout

## 4) Detailed Data Flow (Default Integrated Path)

1. Input assembly

- load requirements from `experiment_data/issue.txt`
- load source from `experiment_data/source_code.py`
- initialize `AgentState`

2. Advanced semantic pass

- classification/extraction/context/pattern/reasoning pipeline in `AdvancedCodeAnalyzer`
- produce findings, recommendations, confidence, usage stats

3. Graph construction and requirement mapping

- structure extraction (functions/classes/variables/calls/definitions)
- semantic requirement decomposition (`REQ-xxx`) and `MAPS_TO` edges
- dependency edges and violation/advisory edges

4. Developer revision

- LLM returns symbol rewrites (`filename`, `symbol`, `replacement`)
- system applies rewrites by symbol table, validates syntax, rejects weak edits
- tracks target-symbol hit to keep changes conflict-focused

5. Judge evaluation

- hard check: graph `VIOLATES`/`ADVISORY` edges drive blocking logic
- soft check: LLM advisory only (does not keep loop alive by itself)

6. Loop control

- continue or end based on stop policy + revision limit
- common stop conditions:
  - no conflict report,
  - no `VIOLATES` edges,
  - or max revisions reached.

## 5) File Responsibility Map

- workflow orchestration:
  - `src/main.py`
  - `src/main_enhanced.py`
  - `run_experiment_integrated.py`
- agents:
  - `src/agents/developer.py`
  - `src/agents/judge.py`
- graph layer:
  - `src/graph_manager.py`
  - `src/enhanced_graph_manager/*.py`
  - `src/enhanced_graph_adapter.py`
- advanced analysis:
  - `src/advanced_code_analysis/advanced_code_analyzer.py`
  - `src/advanced_code_analysis/llm_interface.py`
  - `src/advanced_code_analysis/config.py`
- benchmarks and batch runner:
  - `run_swebench_lite_predictions.py`
- model switching/config:
  - `switch_model.py`
  - `src/model_switcher.py`
  - `configs/models/*.json`

## 6) Pre-Change Checklist (Mandatory)

Before changing any code:

1. Identify the execution path you are touching.

- Traditional: `src/main.py` / `src/graph_manager.py`
- Enhanced: `src/main_enhanced.py` / `src/enhanced_graph_manager/*`
- Integrated: `run_experiment_integrated.py` + all above
- SWE-bench: `run_swebench_lite_predictions.py`

2. Confirm state contract impact.

- If adding/removing fields in `AgentState`, update all workflow initial states.
- Keep optional fields backward-compatible when possible.

3. Confirm loop and stopping behavior impact.

- Any change to judge output format or violation edges can alter loop termination.
- Validate against at least one integrated run.

4. Confirm Developer rewrite application impact.

- Symbol resolution, syntax checks, and no-op rejection are high-risk areas.
- Avoid broad rewrites that increase failed rewrite rate.

5. Confirm configuration/model impact.

- `LLM_MODEL` / `LLM_PROVIDER` override defaults in config.
- `gpt-5*` and codex models follow special API paths in `llm_interface.py`.

## 7) Validation Matrix After Changes

Run minimal checks based on modified area:

- general:
  - `pytest -q`
- workflow sanity:
  - `python run_experiment_enhanced.py`
  - `python run_experiment_integrated.py`
- advanced analysis path:
  - `python run_quick_test.py`
- model/config changes:
  - `python test_model_switch.py`
  - `python test_env_override.py`
- developer rewrite logic changes:
  - `python test_real_developer_fuzzy.py` (if environment path is valid)

For batch path changes:

- run a very small subset first:
  - `python run_swebench_lite_predictions.py --num 1 --mode integrated --dry-run`

## 8) Common Risk Points

- Judge/developer coupling:
  - if conflict text loses concrete symbols, edit quality drops.
- Overly strict rewrite filters:
  - can cause "no effective file changes" loops.
- Requirement decomposition noise:
  - metadata sentences can create low-signal requirements.
- Stop policy mismatch:
  - conflict-only vs violates-only can change termination behavior.
- Temp file handling in async analysis:
  - ensure cleanup paths remain safe.

## 9) Practical Change Strategy

When implementing non-trivial changes:

1. Start with one mode (usually enhanced or integrated), not all modes at once.
2. Keep interfaces stable; add optional parameters instead of breaking signatures.
3. Add logs/metrics only where they influence diagnosis of revision failures.
4. Re-run the smallest relevant script first, then broaden validation.

## 10) Quick Command Reference

- `python run_experiment.py`
- `python run_experiment_enhanced.py`
- `python run_experiment_advanced.py`
- `python run_experiment_integrated.py`
- `python compare_workflows.py`
- `python run_quick_test.py`
- `python run_demo_mode.py`
- `python switch_model.py --list`
- `pytest -q`

---

Operational rule for this repository:

- Treat this file as the required pre-change checklist and flow reference.
