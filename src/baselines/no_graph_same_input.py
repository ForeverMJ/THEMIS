from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Sequence

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.state import AgentState


class FileEdit(BaseModel):
    """Full replacement content for one provided file."""

    filename: str = Field(description="Path of a file from the provided input set.")
    content: str = Field(description="Complete updated file content.")


class CodebaseUpdate(BaseModel):
    """Structured no-graph repair output."""

    files: List[FileEdit]


def _format_files(files: Dict[str, str]) -> str:
    return "\n\n".join(
        f"### File: {name}\n```python\n{content}\n```"
        for name, content in files.items()
    )


def _filter_to_provided_files(
    original_files: Dict[str, str],
    update: CodebaseUpdate,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    """Apply only supplied-file edits that pass a basic Python syntax gate."""

    updated_files = dict(original_files)
    allowed = set(original_files)
    applied_files: List[str] = []
    ignored_files: List[str] = []
    syntax_rejected_files: List[str] = []
    syntax_rejection_reasons: Dict[str, str] = {}

    for file_edit in update.files:
        if file_edit.filename not in allowed:
            ignored_files.append(file_edit.filename)
            continue

        try:
            ast.parse(file_edit.content, filename=file_edit.filename)
        except SyntaxError as exc:
            syntax_rejected_files.append(file_edit.filename)
            syntax_rejection_reasons[file_edit.filename] = (
                f"{exc.msg} at line {exc.lineno}, column {exc.offset}"
            )
            continue

        updated_files[file_edit.filename] = file_edit.content
        applied_files.append(file_edit.filename)

    meta = {
        "enabled": True,
        "applied_files": applied_files,
        "ignored_files": ignored_files,
        "syntax_rejected_files": syntax_rejected_files,
        "syntax_rejection_reasons": syntax_rejection_reasons,
    }
    return updated_files, meta


def build_app(
    llm_model: str = "gpt-5.1-codex-mini",
    callbacks: Optional[Sequence[Any]] = None,
) -> Any:
    """
    Build the NoGraph-SameInput baseline.

    This baseline receives the same requirements string and selected files as
    THEMIS, but it performs a single direct repair without requirement graphs,
    advanced analysis, or judge feedback.
    """

    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        callbacks=list(callbacks or []),
    ).with_structured_output(CodebaseUpdate)

    def direct_repair(state: AgentState) -> AgentState:
        original_files = dict(state["files"])
        prompt = f"""You are evaluating a no-graph code repair baseline.

You are given the exact same issue input and selected source files that the
THEMIS system receives. Repair the code directly. Do not assume access to any
requirement graph, MAPS_TO edges, violation edges, advanced analyzer output, or
judge feedback.

Rules:
- Only modify files that are provided below.
- Return complete updated content for any file you include.
- If a provided file does not need changes, you may omit it or return it unchanged.
- Do not add new files.
- Preserve unrelated behavior.

Requirements:
{state['requirements']}

Selected files:
{_format_files(original_files)}

Return JSON with schema:
{{"files": [{{"filename": string, "content": string}}]}}
"""
        update: CodebaseUpdate = llm.invoke(prompt)
        updated_files, syntax_gate_meta = _filter_to_provided_files(original_files, update)
        new_state = state.copy()
        new_state["files"] = updated_files
        new_state["revision_count"] = 1
        new_state["conflict_report"] = None
        new_state["no_graph_baseline_meta"] = {
            "syntax_gate": syntax_gate_meta,
            "llm_calls": 1,
            "second_repair_attempted": False,
        }
        return new_state

    graph = StateGraph(AgentState)
    graph.add_node("direct_repair", direct_repair)
    graph.set_entry_point("direct_repair")
    graph.add_edge("direct_repair", END)
    return graph.compile()
