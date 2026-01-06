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

    def _format_files_for_prompt(
        self,
        files: Dict[str, str],
        *,
        requirements: str,
        conflict_text: str,
        force_full_files: set[str] | None = None,
    ) -> str:
        # For large files, provide focused excerpts around likely-relevant tokens to make exact-copy edits easier.
        focus_tokens = self._extract_focus_tokens(requirements + "\n\n" + conflict_text)
        formatted_parts: list[str] = []

        for name, content in files.items():
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            if force_full_files and name in force_full_files:
                formatted_parts.append(f"## {name}\n{content}")
                continue
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

        def _exact_match_count(before_lines: list[str], window_lines: list[str]) -> int:
            return sum(
                1 for b, w in zip(before_lines, window_lines) if b.strip() and b.strip() == w.strip()
            )

        candidates: dict[int, dict[str, object]] = {}

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
                window_lines = content_lines[start:end]
                window = "\n".join(window_lines)
                ratio = difflib.SequenceMatcher(
                    None, self._norm_for_match(before), self._norm_for_match(window)
                ).ratio()
                exact_matches = _exact_match_count(before_lines, window_lines)
                prev = candidates.get(start)
                if (
                    prev is None
                    or ratio > prev["ratio"]  # type: ignore[operator]
                    or (ratio == prev["ratio"] and exact_matches > prev["exact_matches"])  # type: ignore[operator]
                ):
                    candidates[start] = {
                        "ratio": ratio,
                        "exact_matches": exact_matches,
                        "window": window,
                    }

        if not candidates:
            return None

        ordered = sorted(
            candidates.items(),
            key=lambda item: (item[1]["ratio"], item[1]["exact_matches"]),
            reverse=True,
        )
        best_start, best = ordered[0]
        best_ratio = float(best["ratio"])
        best_exact = int(best["exact_matches"])
        second_ratio = float(ordered[1][1]["ratio"]) if len(ordered) > 1 else 0.0
        second_exact = int(ordered[1][1]["exact_matches"]) if len(ordered) > 1 else 0

        # Only accept high-confidence, unambiguous fuzzy matches.
        if best_ratio < 0.92:
            return None
        if (best_ratio - second_ratio) < 0.01 and best_exact <= second_exact:
            # If all candidate windows are identical, apply to all occurrences.
            window_norms = {self._norm_for_match(str(c["window"])) for c in candidates.values()}
            if len(window_norms) == 1:
                new_lines = content_lines[:]
                for start in sorted(candidates.keys(), reverse=True):
                    window_text = str(candidates[start]["window"])
                    delta = self._min_indent(window_text) - self._min_indent(before)
                    adjusted_after = self._shift_indentation(after, delta=delta).split("\n")
                    new_lines = (
                        new_lines[:start]
                        + adjusted_after
                        + new_lines[start + len(before_lines) :]
                    )
                return "\n".join(new_lines)
            return None

        window_text = str(best["window"])
        delta = self._min_indent(window_text) - self._min_indent(before)
        after = self._shift_indentation(after, delta=delta)
        after_lines = after.split("\n")
        new_lines = (
            content_lines[:best_start] + after_lines + content_lines[best_start + len(before_lines) :]
        )
        return "\n".join(new_lines)

    @staticmethod
    def _apply_line_replacements(content: str, before: str, after: str) -> str | None:
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        before = before.replace("\r\n", "\n").replace("\r", "\n")
        after = after.replace("\r\n", "\n").replace("\r", "\n")

        before_lines = before.split("\n")
        after_lines = after.split("\n")
        replacements: list[tuple[str, str]] = []
        sm = difflib.SequenceMatcher(
            a=[line.strip() for line in before_lines],
            b=[line.strip() for line in after_lines],
        )
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace":
                continue
            span = min(i2 - i1, j2 - j1)
            for k in range(span):
                b = before_lines[i1 + k]
                a = after_lines[j1 + k]
                if b.strip() == a.strip():
                    continue
                if not b.strip() or not a.strip():
                    continue
                replacements.append((b, a))

        if not replacements:
            return None

        content_lines = content.split("\n")
        changed = False
        for idx, line in enumerate(content_lines):
            stripped = line.strip()
            for b, a in replacements:
                if stripped == b.strip():
                    indent = line[: len(line) - len(line.lstrip(" \t"))]
                    new_line = indent + a.strip()
                    if new_line != line:
                        content_lines[idx] = new_line
                        changed = True

        return "\n".join(content_lines) if changed else None

    def revise(
        self,
        files: Dict[str, str],
        requirements: str,
        conflict_report: str | None,
        *,
        force_full_files: set[str] | None = None,
    ) -> Dict[str, str]:
        conflict_text = conflict_report or "None"
        current_files = self._format_files_for_prompt(
            files,
            requirements=requirements,
            conflict_text=conflict_text,
            force_full_files=force_full_files,
        )

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
        if not result.edits:
            raise ValueError("Developer returned no edits; please provide at least one change.")

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
                    fallback = self._apply_line_replacements(content, before, after)
                    if fallback is not None:
                        updated[filename] = fallback
                        continue
                    preview = before.strip().splitlines()[:3]
                    preview_text = "\\n".join(preview)
                    raise ValueError(
                        "Could not apply edit: 'before' snippet not found in "
                        f"{filename}. Preview:\\n{preview_text}"
                    )
                updated[filename] = fuzzy
                continue
            if occurrences > 1:
                updated[filename] = content.replace(before, after)
                continue

            updated[filename] = content.replace(before, after, 1)

        return updated
