# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/`. Main workflow entry points are `src/main.py`, `src/main_enhanced.py`, and `run_experiment_integrated.py`. Agents are in `src/agents/`, baseline workflows in `src/baselines/`, the enhanced graph stack in `src/enhanced_graph_manager/`, and the semantic analysis pipeline in `src/advanced_code_analysis/`.

Experiment fixtures live in `experiment_data/` and its `case1/`-`case3/` subdirectories. Regression tests live in `tests/`, especially `tests/enhanced_graph_manager/`. Utility scripts such as `run_experiment_enhanced.py`, `run_quick_test.py`, and `run_swebench_lite_predictions.py` are the main local entry points. Review `PROJECT_FLOW_REFERENCE.rst` before changing workflow logic.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install runtime and test dependencies.
- `pytest -q`: run the repository test suite under `tests/`.
- `python run_experiment_enhanced.py`: run the enhanced graph workflow on `experiment_data/`.
- `python run_experiment_integrated.py`: run the full Advanced Analysis -> Graph -> Developer -> Judge loop.
- `python run_quick_test.py`: smoke-test the advanced analysis adapter.
- `python run_swebench_lite_predictions.py --num 1 --dry-run`: validate the SWE-bench pipeline without LLM calls.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and follow PEP 8 naming: `snake_case` for functions/files, `PascalCase` for classes, and clear constant names like `MAX_REVISIONS`. Keep edits surgical: preserve docstrings, comments, and unrelated logic, especially in workflow and rewrite code. No formatter or linter is enforced in the repo today, so match surrounding style and keep imports/typing consistent with adjacent files.

## Testing Guidelines

Pytest is the standard framework; `pytest.ini` collects `tests/test_*.py` and `tests/enhanced_graph_manager/test_*.py`. Hypothesis is available for property tests via the `property` marker. When changing graph extraction, violation logic, or rewrite behavior, add or update focused tests near the affected module and run the narrowest relevant command first, then `pytest -q`.

## Commit & Pull Request Guidelines

Recent history mixes short save-point messages with conventional prefixes such as `feat(...)`. Prefer concise, imperative commit messages, optionally scoped, for example: `fix(developer): reject whitespace-only rewrites`. In pull requests, include the target workflow (`traditional`, `enhanced`, `integrated`, or `swebench`), the commands you ran, key metrics or conflict deltas if loop behavior changed, and any required environment variables or model settings.

## Security & Configuration Tips

Keep secrets in `.env`; never commit API keys. Model selection is controlled by `LLM_MODEL`, `LLM_PROVIDER`, and helper commands like `python switch_model.py --list`. When changing loop behavior, judge output, or state fields, verify all workflow entry points still initialize `AgentState` correctly.
