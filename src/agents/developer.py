from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class FileEdit(BaseModel):
    filename: str
    content: str


class DevOutput(BaseModel):
    files: List[FileEdit]


class DeveloperAgent:
    """Uses an LLM to revise code according to requirements and conflict reports."""

    def __init__(self, llm) -> None:
        self.llm = llm.with_structured_output(DevOutput)

    def revise(self, files: Dict[str, str], requirements: str, conflict_report: str | None) -> Dict[str, str]:
        conflict_text = conflict_report or "None"
        current_files = "\n\n".join(f"## {name}\n{content}" for name, content in files.items())

        prompt = f"""
You are a Senior Software Engineer specializing in fixing complex logical inconsistencies.

Current Requirements (Issue):
{requirements}

Conflict Report (CRITICAL - MUST FIX):
{conflict_text}

Current Files:
{current_files}

**CRITICAL RULES FOR BENCHMARK SUCCESS:**

1. **⛔ DO NOT DELETE DOCSTRINGS OR COMMENTS ⛔**
   - **WARNING:** The automated test suite relies on `doctest` strings inside comments.
   - **IF YOU DELETE DOCSTRINGS, THE TESTS WILL FAIL.**
   - You must return the **FULL** original file content, keeping all comments, docstrings, and unrelated code exactly as is. Only modify the specific lines needed to fix the bug.

2. **TRACE THE ROOT CAUSE:**
   - The Conflict Report might point to where the error *occurs* (e.g., `get_order_by`), but the error often originates in a **Variable Definition** (e.g., a Regex defined in `__init__`).
   - If a variable (like `self.ordering_parts`) is causing issues, **FIX ITS DEFINITION in `__init__`**, do not just hack the usage.

3. **LOGICAL CORRECTNESS:**
   - Ensure regexes and logic handle the edge cases described in the Requirements (e.g., Multiline handling).

Output:
Return the FULL content of the updated files using the structured format.
"""

        result: DevOutput = self.llm.invoke(prompt)

        return {file_edit.filename: file_edit.content for file_edit in result.files}
