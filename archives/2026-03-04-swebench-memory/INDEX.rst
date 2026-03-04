2026-03-04 SWE-bench Memory Archive
===================================

Contents
--------

- ``skill/``: exported copy of the Codex skill ``themis-swebench-memory`` created for this session.
- ``predictions/``: 6-case baseline and candidate prediction files used for SWE-bench Lite evaluation.
- ``logs/cohort6_baseline_runner_logs/``: per-instance workflow logs for the baseline cohort run.
- ``logs/cohort6_candidate_runner_logs/``: per-instance workflow logs for the first candidate cohort run.
- ``logs/cohort6_candidate_tail_runner_logs/``: additional candidate runner logs for the tail cases.
- ``logs/swebench_eval_logs_baseline_v3/``: baseline harness evaluation logs.
- ``logs/swebench_eval_logs_candidate_v3/``: candidate harness evaluation logs.

Notes
-----

- ``.env`` is intentionally excluded.
- Generated repo checkouts under ``swebench_runs_*/repos`` are intentionally excluded.
- The candidate method from this session was rolled back from repo code because the 6-case cohort regressed from ``PASS 1`` to ``PASS 0``.
