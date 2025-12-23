from __future__ import annotations

import difflib
import re
from typing import Dict, List

from pydantic import BaseModel


class TextEdit(BaseModel):
    filename: str
    before: str
    after: str


class DevOutput(BaseModel):
    edits: List[TextEdit]


class DeveloperAgent:
    """Uses an LLM to revise code according to requirements and conflict reports."""

    def __init__(self, llm) -> None:
        self.llm = llm.with_structured_output(DevOutput)

    @staticmethod
    def _extract_focus_tokens(text: str, *, max_tokens: int = 12) -> list[str]:
        stopwords = {
            "the",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "with",
            "a",
            "an",
            "is",
            "are",
            "be",
            "as",
            "on",
            "this",
            "that",
            "it",
            "when",
            "from",
            "by",
            "should",
            "must",
            "not",
            "does",
            "do",
            "will",
            "can",
        }

        tokens: list[str] = []
        tokens.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", text))
        tokens.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", text))

        filtered: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            tl = t.lower()
            if tl in stopwords:
                continue
            if tl.isdigit():
                continue
            if len(t) > 64:
                continue
            if t in seen:
                continue
            seen.add(t)
            filtered.append(t)

        def _priority(tok: str) -> int:
            if tok.startswith("__") and tok.endswith("__"):
                return 0
            if "_" in tok:
                return 1
            if any(c.isupper() for c in tok[1:]):
                return 2
            return 3

        filtered.sort(key=lambda s: (_priority(s), -len(s), s))
        return filtered[:max_tokens]

    def _format_files_for_prompt(self, files: Dict[str, str], *, requirements: str, conflict_text: str) -> str:
        # For large files, provide focused excerpts around likely-relevant tokens to make exact-copy edits easier.
        focus_tokens = self._extract_focus_tokens(requirements + "\n\n" + conflict_text)
        formatted_parts: list[str] = []

        for name, content in files.items():
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            if len(content) <= 20000 or not focus_tokens:
                formatted_parts.append(f"## {name}\n{content}")
                continue

            lines = content.split("\n")
            scores = [0] * len(lines)
            for tok in focus_tokens:
                tok_l = tok.lower()
                tok_is_lower = tok == tok_l
                for i, line in enumerate(lines):
                    hay = line.lower() if tok_is_lower else line
                    needle = tok_l if tok_is_lower else tok
                    if needle in hay:
                        scores[i] += 1

            anchors = sorted(range(len(lines)), key=lambda i: (-scores[i], i))
            picked: list[int] = []
            for idx in anchors:
                if scores[idx] <= 0:
                    break
                if any(abs(idx - p) < 40 for p in picked):
                    continue
                picked.append(idx)
                if len(picked) >= 3:
                    break

            if not picked:
                # Fallback: show the first part of the file.
                excerpt = "\n".join(lines[:200])
                formatted_parts.append(f"## {name} (excerpt)\n{excerpt}")
                continue

            for idx in picked:
                start = max(0, idx - 40)
                end = min(len(lines), idx + 41)
                excerpt = "\n".join(lines[start:end])
                formatted_parts.append(f"## {name} (excerpt lines {start + 1}-{end})\n{excerpt}")

        return "\n\n".join(formatted_parts)

    @staticmethod
    def _indent_width(prefix: str) -> int:
        width = 0
        for ch in prefix:
            if ch == " ":
                width += 1
            elif ch == "\t":
                width += 4
            else:
                break
        return width

    @staticmethod
    def _min_indent(text: str) -> int:
        lines = text.split("\n")
        indents = []
        for line in lines:
            if not line.strip():
                continue
            prefix = ""
            for ch in line:
                if ch in (" ", "\t"):
                    prefix += ch
                else:
                    break
            indents.append(DeveloperAgent._indent_width(prefix))
        return min(indents) if indents else 0

    @staticmethod
    def _shift_indentation(text: str, *, delta: int) -> str:
        if delta == 0:
            return text
        out_lines: list[str] = []
        for line in text.split("\n"):
            if not line.strip():
                out_lines.append(line)
                continue
            prefix = ""
            i = 0
            while i < len(line) and line[i] in (" ", "\t"):
                prefix += line[i]
                i += 1
            rest = line[i:]
            width = DeveloperAgent._indent_width(prefix)
            new_width = max(0, width + delta)
            out_lines.append((" " * new_width) + rest)
        return "\n".join(out_lines)

    @staticmethod
    def _norm_for_match(text: str) -> str:
        # Normalize for matching: ignore line-ending differences and indentation.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join(line.strip() for line in text.split("\n"))

    def _apply_fuzzy_edit(self, content: str, before: str, after: str) -> str | None:
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        before = before.replace("\r\n", "\n").replace("\r", "\n")
        after = after.replace("\r\n", "\n").replace("\r", "\n")

        if not before.strip():
            return None

        content_lines = content.split("\n")
        before_lines = before.split("\n")

        if not before_lines:
            return None

        # Use a few long, non-empty lines from `before` as anchors to localize candidates.
        anchors: list[tuple[int, str]] = [
            (i, line) for i, line in enumerate(before_lines) if line.strip() and len(line.strip()) >= 8
        ]
        anchors.sort(key=lambda x: -len(x[1]))

        candidates: list[tuple[float, int]] = []  # (ratio, start_idx)

        for before_idx, anchor_line in anchors[:6]:
            anchor_stripped = anchor_line.strip()
            positions = [i for i, line in enumerate(content_lines) if line.strip() == anchor_stripped]
            if not positions or len(positions) > 50:
                continue

            for pos in positions:
                start = pos - before_idx
                if start < 0:
                    continue
                end = start + len(before_lines)
                if end > len(content_lines):
                    continue
                window = "\n".join(content_lines[start:end])
                ratio = difflib.SequenceMatcher(None, self._norm_for_match(before), self._norm_for_match(window)).ratio()
                candidates.append((ratio, start))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_ratio, best_start = candidates[0]
        second_ratio = candidates[1][0] if len(candidates) > 1 else 0.0

        # Only accept high-confidence, unambiguous fuzzy matches.
        if best_ratio < 0.92 or (best_ratio - second_ratio) < 0.02:
            return None

        window_text = "\n".join(content_lines[best_start : best_start + len(before_lines)])
        delta = self._min_indent(window_text) - self._min_indent(before)
        after = self._shift_indentation(after, delta=delta)
        after_lines = after.split("\n")
        new_lines = (
            content_lines[:best_start] + after_lines + content_lines[best_start + len(before_lines) :]
        )
        return "\n".join(new_lines)

    def revise(self, files: Dict[str, str], requirements: str, conflict_report: str | None) -> Dict[str, str]:
        conflict_text = conflict_report or "None"
        current_files = self._format_files_for_prompt(files, requirements=requirements, conflict_text=conflict_text)

        prompt = f"""
You are a Senior Software Engineer specializing in fixing complex logical inconsistencies.

Current Requirements (Issue):
{requirements}

Conflict Report (CRITICAL - MUST FIX):
{conflict_text}

Current Files:
{current_files}

**CRITICAL RULES FOR BENCHMARK SUCCESS:**

1. DO NOT DELETE DOCSTRINGS OR COMMENTS
   - WARNING: The automated test suite may rely on doctest strings inside comments/docstrings.
   - Keep all comments, docstrings, and unrelated code exactly as is.

2. MAKE MINIMAL CHANGES (prefer 1-2 lines)
   - Change ONLY the lines needed to satisfy the requirements and fix the conflict report.
   - Do NOT refactor or reformat unrelated code.
   - Do NOT change function signatures unless explicitly required.

3. OUTPUT FORMAT (IMPORTANT)
Return a list of exact text edits to apply.

Each edit must include:
- filename: which file to edit
- before: an EXACT snippet copied from the current file (include enough surrounding lines to make it unique)
- after: the replacement snippet

Rules:
- `before` MUST appear exactly once in the file content.
- If the file is shown as an excerpt, `before` MUST be copied verbatim from that excerpt (including indentation).
- Include at least 2 unchanged context lines above and below the changed lines in `before`.
- Do NOT use "..." or placeholders inside `before`/`after`.
- Preserve indentation; the final file must be syntactically valid.
"""

        result: DevOutput = self.llm.invoke(prompt)

        updated = dict(files)
        for edit in result.edits:
            filename = edit.filename
            if filename not in updated:
                raise ValueError(f"Developer proposed edit for unknown file: {filename}")

            before = edit.before.replace("\r\n", "\n").replace("\r", "\n")
            after = edit.after.replace("\r\n", "\n").replace("\r", "\n")

            content = updated[filename].replace("\r\n", "\n").replace("\r", "\n")
            occurrences = content.count(before)
            if occurrences == 0:
                fuzzy = self._apply_fuzzy_edit(content, before, after)
                if fuzzy is None:
                    preview = before.strip().splitlines()[:3]
                    preview_text = "\\n".join(preview)
                    raise ValueError(
                        "Could not apply edit: 'before' snippet not found in "
                        f"{filename}. Preview:\\n{preview_text}"
                    )
                updated[filename] = fuzzy
                continue
            if occurrences > 1:
                raise ValueError(f"Could not apply edit: 'before' snippet is not unique in {filename}")

            updated[filename] = content.replace(before, after, 1)

        return updated
