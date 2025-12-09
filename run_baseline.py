from __future__ import annotations

import difflib
from pathlib import Path

from src.baselines.vanilla import app


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

    initial_state = {
        "messages": [],
        "files": {target_filename: source_code},
        "requirements": requirements,
        "knowledge_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }

    final_state = app.invoke(initial_state)
    final_code = final_state["files"][target_filename]

    print("=== Diff (original -> baseline) ===")
    diff = difflib.unified_diff(
        source_code.splitlines(keepends=True),
        final_code.splitlines(keepends=True),
        fromfile=target_filename,
        tofile=f"{target_filename} (baseline)",
    )
    diff_text = "".join(diff)
    print(diff_text or "(no changes)")


if __name__ == "__main__":
    main()
