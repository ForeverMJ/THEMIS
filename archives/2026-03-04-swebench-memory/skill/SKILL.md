---
name: themis-swebench-memory
description: Preserve and reuse the experiment history for the THEMIS repository's SWE-bench Lite work. Use when continuing this repo's multi-agent repair experiments, needing the prior dialogue decisions, retained vs rolled-back code changes, 6-case cohort rules, or local SWE-bench harness workarounds before making new changes.
---

# THEMIS SWE-bench Memory

## Overview

Use this skill when continuing SWE-bench work in `/Users/tao/Dev/THEMIS`.
Treat it as the project memory for this session: what was changed, why it was changed, what was rolled back, how evaluation was run, and what rules now govern keep vs rollback decisions.

## Required startup

1. Read `PROJECT_FLOW_REFERENCE.rst` first.
2. Read `references/change-ledger.md` to understand retained changes, rolled-back changes, and local-only harness patches.
3. Read `references/experiment-playbook.md` before proposing new SWE-bench method experiments.
4. Read `references/conversation-archive.md` only when you need the detailed chronology of decisions.

## Working rules

- Judge candidate method changes by SWE-bench Lite harness outcome, not by internal blocking metrics alone.
- Use the fixed 6-case cohort unless the user explicitly changes it.
- Preserve retained stage 1/2 repo changes unless the user asks to revisit them.
- Treat `.venv/lib/python3.9/site-packages/swebench/harness/*` patches as local evaluation infrastructure, not product code.
- Before keeping a new method change, compare baseline vs candidate on the same cohort and apply the rollback rule from `references/experiment-playbook.md`.

## Resources

### references/
- `references/change-ledger.md`: full ledger of repo changes, attempted changes, purpose, outcome, and current status.
- `references/experiment-playbook.md`: the current 6-case cohort, commands, rollback rule, and evaluation procedure.
- `references/conversation-archive.md`: turn-by-turn archive of the user requests, actions, and conclusions from this session.

### scripts/
- `scripts/summarize_eval_status.py`: summarize SWE-bench harness result logs into `PASS` / `FAIL` / `APPLY_FAIL` / `RESET_FAIL`.
