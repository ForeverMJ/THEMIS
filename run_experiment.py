from __future__ import annotations

import difflib
from pathlib import Path

import networkx as nx

from src.main import build_workflow
from src.state import AgentState


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"

    requirements = load_text(req_path)
    source_code = load_text(code_path)

    target_filename = "target_file.py"

    workflow = build_workflow()
    app = workflow.compile()

    initial_state: AgentState = {
        "messages": [],
        "files": {target_filename: source_code},
        "requirements": requirements,
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    conflict_report = final_state.get("conflict_report")
    final_files = final_state.get("files", {})
    final_code = final_files.get(target_filename, "")

    print("=== Conflict Report ===")
    print(conflict_report or "None")

    print("\n=== Diff (original -> revised) ===")
    diff = difflib.unified_diff(
        source_code.splitlines(keepends=True),
        final_code.splitlines(keepends=True),
        fromfile=target_filename,
        tofile=f"{target_filename} (revised)",
    )
    diff_text = "".join(diff)
    print(diff_text or "(no changes)")


if __name__ == "__main__":
    main()
