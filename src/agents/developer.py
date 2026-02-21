from __future__ import annotations

import ast
import difflib
import re
from typing import Any, Dict, List

from pydantic import BaseModel


class SymbolRewrite(BaseModel):
    filename: str
    symbol: str
    replacement: str


class DevOutput(BaseModel):
    rewrites: List[SymbolRewrite]


class DeveloperAgent:
    """Uses an LLM to revise code according to requirements and conflict reports."""

    def __init__(self, llm) -> None:
        self.llm = llm.with_structured_output(DevOutput)
        self.last_revision_meta: Dict[str, Any] = {
            "proposed_rewrites": 0,
            "applied_rewrites": 0,
            "applied_symbols": [],
            "failed_rewrites": [],
        }

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

    @staticmethod
    def _changed_line_stats(old_text: str, new_text: str) -> tuple[int, int]:
        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")
        total_lines = max(len(old_lines), 1)
        changed_lines = 0
        sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            changed_lines += max(i2 - i1, j2 - j1)
        return changed_lines, total_lines

    @staticmethod
    def _is_whitespace_only_change(old_text: str, new_text: str) -> bool:
        def _normalize(text: str) -> str:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return "\n".join(line.rstrip() for line in text.split("\n"))

        return _normalize(old_text) == _normalize(new_text) and old_text != new_text

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
            print("DEBUG: Fuzzy match failed - empty 'before' string")
            return None

        def _heal_suffix(target_lines: list[str], window_text: str, before_lines: list[str]) -> list[str]:
            """Helper to restore truncated suffixes from the original window."""
            if not target_lines or not window_text or not before_lines:
                 return target_lines
            
            window_lines = window_text.split("\n")
            w_last = window_lines[-1].rstrip("\r\n")
            b_last = before_lines[-1].rstrip("\r\n")
            w_stripped = w_last.strip()
            b_stripped = b_last.strip()

            if w_stripped.startswith(b_stripped) and len(w_stripped) > len(b_stripped):
                suffix = w_stripped[len(b_stripped):]
                # Only append if the suffix suggests meaningful syntax (not just whitespace/comments)
                # or if it completes a structural element. 
                if not target_lines[-1].strip().endswith(suffix.strip()):
                    print(f"DEBUG: Restoring truncated suffix '{suffix.strip()}' to line: {target_lines[-1].strip()}")
                    target_lines[-1] = target_lines[-1].rstrip("\r\n") + suffix
            return target_lines

        content_lines = content.split("\n")
        before_lines = before.split("\n")

        if not before_lines:
            print("DEBUG: Fuzzy match failed - empty 'before' lines")
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
            print("DEBUG: Fuzzy match failed - no candidates found")
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
        # Only accept high-confidence, unambiguous fuzzy matches.
        # (Removed hardcoded check to allow dynamic threshold logic below)
        min_ratio = 0.92
        if best_exact >= 4 and best_ratio >= 0.88:
            min_ratio = 0.88
        elif best_exact >= 2 and (best_ratio - second_ratio) >= 0.03:
            min_ratio = 0.80  # Relaxed from 0.88/0.90
        elif best_exact > second_exact and best_exact >= 2:
            min_ratio = 0.80  # Distinct exact match advantage
        
        # If the match is unique (second best is far away), allow lower ratio
        if (best_ratio - second_ratio) >= 0.10 and best_ratio >= 0.75:
            min_ratio = 0.75
        

        if best_ratio < min_ratio:
            print(f"DEBUG: Fuzzy match failed - best_ratio {best_ratio:.4f} < min_ratio {min_ratio}")
            return None
            # If all candidate windows are identical, apply to all occurrences.
            window_norms = {self._norm_for_match(str(c["window"])) for c in candidates.values()}
            if len(window_norms) == 1:
                new_lines = content_lines[:]
                for start in sorted(candidates.keys(), reverse=True):
                    window_text = str(candidates[start]["window"])
                    delta = self._min_indent(window_text) - self._min_indent(before)
                    adjusted_after = self._shift_indentation(after, delta=delta).split("\n")
                    # Apply suffix healing to identical match case
                    adjusted_after = _heal_suffix(adjusted_after, window_text, before_lines)
                    new_lines = (
                        new_lines[:start]
                        + adjusted_after
                        + new_lines[start + len(before_lines) :]
                    )
                print(f"DEBUG: Fuzzy match succeeded (applied to {len(candidates)} identical identical occurrences).")
                return "\n".join(new_lines)
            print("DEBUG: Fuzzy match failed - multiple candidates but windows not identical")
            return None

        window_text = str(best["window"])
        delta = self._min_indent(window_text) - self._min_indent(before)
        after = self._shift_indentation(after, delta=delta)
        after_lines = after.split("\n")

        # Heal truncated suffixes (e.g. missing ');' or ') % (') in the last line
        after_lines = _heal_suffix(after_lines, window_text, before_lines)

        # DEBUG: Print the first few lines of the replacement to check for corruption

        after_lines = after_lines
        # DEBUG: Print the first few lines of the replacement to check for corruption
        if after_lines:
            print(f"DEBUG: Fuzzy Edit applied. First line: {repr(after_lines[0])}")
            if len(after_lines) > 1:
                print(f"DEBUG: Second line: {repr(after_lines[1])}")
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

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        return "\n".join(lines).strip("\n")

    @staticmethod
    def _symbol_table(content: str) -> dict[str, tuple[int, int, int]]:
        """
        Build symbol table for Python source.

        Returns:
            Mapping symbol_name -> (start_idx, end_idx, indent_width)
            where indexes are 0-based line indexes and end_idx is exclusive.
        """
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        tree = ast.parse(normalized)
        lines = normalized.split("\n")
        table: dict[str, tuple[int, int, int]] = {}

        def _record(name: str, node: ast.AST) -> None:
            lineno = int(getattr(node, "lineno", 0) or 0)
            end_lineno = int(getattr(node, "end_lineno", 0) or 0)
            if lineno <= 0 or end_lineno <= 0:
                return
            start = lineno - 1
            end = end_lineno
            if start < 0 or start >= len(lines):
                return
            indent = len(lines[start]) - len(lines[start].lstrip(" \t"))
            table[name] = (start, end, indent)

        def _visit(node: ast.AST, class_stack: list[str]) -> None:
            if isinstance(node, ast.ClassDef):
                class_name = ".".join(class_stack + [node.name]) if class_stack else node.name
                _record(class_name, node)
                for child in node.body:
                    _visit(child, class_stack + [node.name])
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = ".".join(class_stack + [node.name]) if class_stack else node.name
                _record(fn_name, node)
                return
            for child in ast.iter_child_nodes(node):
                _visit(child, class_stack)

        _visit(tree, [])
        return table

    @staticmethod
    def _resolve_symbol_name(symbol_table: dict[str, tuple[int, int, int]], raw_symbol: str) -> str:
        symbol = raw_symbol.replace("::", ".").strip()
        if not symbol:
            raise ValueError("Developer returned empty symbol name.")
        if symbol in symbol_table:
            return symbol

        # Fallback by leaf symbol (e.g. method name) when unique.
        leaf = symbol.split(".")[-1]
        candidates = [
            name for name in symbol_table.keys() if name == leaf or name.endswith("." + leaf)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"Developer proposed unknown symbol: {raw_symbol}")
        raise ValueError(
            f"Developer proposed ambiguous symbol: {raw_symbol}. Candidates: {', '.join(sorted(candidates)[:8])}"
        )

    def _apply_symbol_rewrite(
        self,
        content: str,
        *,
        symbol: str,
        replacement: str,
        filename: str,
    ) -> tuple[str, str]:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        replacement_text = self._strip_code_fence(replacement)
        if not replacement_text.strip():
            raise ValueError(f"Developer returned empty replacement for {filename}:{symbol}")

        table = self._symbol_table(normalized)
        resolved_symbol = self._resolve_symbol_name(table, symbol)
        start, end, target_indent = table[resolved_symbol]

        # Align replacement indentation to target symbol indentation.
        replacement_min_indent = self._min_indent(replacement_text)
        delta = target_indent - replacement_min_indent
        aligned = self._shift_indentation(replacement_text, delta=delta)
        aligned_lines = aligned.split("\n")

        source_lines = normalized.split("\n")
        new_lines = source_lines[:start] + aligned_lines + source_lines[end:]
        updated = "\n".join(new_lines)

        return updated, resolved_symbol

    @staticmethod
    def _python_syntax_error(filename: str, content: str) -> str | None:
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            lineno = int(e.lineno or 0)
            offset = int(e.offset or 0)
            msg = e.msg or "SyntaxError"
            return f"{filename}:{lineno}:{offset}: {msg}"

    def revise(
        self,
        files: Dict[str, str],
        requirements: str,
        conflict_report: str | None,
        *,
        force_full_files: set[str] | None = None,
    ) -> Dict[str, str]:
        conflict_text = conflict_report or "None"
        forced_full_files = set(files.keys())
        if force_full_files:
            forced_full_files.update(force_full_files)
        conflict_focus = self._extract_focus_tokens(conflict_text, max_tokens=20)
        conflict_focus_text = ", ".join(conflict_focus) if conflict_focus else "(none)"
        current_files = self._format_files_for_prompt(
            files,
            requirements=requirements,
            conflict_text=conflict_text,
            force_full_files=forced_full_files,
        )

        allowed_files = ", ".join(sorted(files.keys())) if files else "(none)"
        prompt = f"""
You are a Senior Software Engineer specializing in fixing complex logical inconsistencies.

Current Requirements (Issue):
{requirements}

Conflict Report (CRITICAL - MUST FIX):
{conflict_text}

Conflict Focus Symbols (prioritize these):
{conflict_focus_text}

Allowed Files (edit ONLY these exact filenames):
{allowed_files}

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

3. CONFLICT-FIRST STRATEGY
   - Prioritize edits in code regions directly tied to conflict report entities.
   - If you cannot fully resolve all conflicts in one round, make one highest-impact conflict-reducing change.
   - Avoid broad cleanup; this round is judged only on conflict reduction.

4. OUTPUT FORMAT (IMPORTANT)
Return a list of SYMBOL rewrites. The list MUST be non-empty.

Each rewrite must include:
- filename: target file
- symbol: exact Python symbol to replace
  Allowed forms:
  - top-level function: `function_name`
  - top-level class: `ClassName`
  - method: `ClassName.method_name`
- replacement: complete replacement code for that symbol only

Rules:
- Do NOT propose rewrites for files not listed in Allowed Files.
- Do NOT use before/after snippets.
- Do NOT return full-file content.
- Do NOT use "..." or placeholders.
- Keep unrelated code/comments/docstrings unchanged.
- Preserve indentation; the final file must be syntactically valid.
- replacement MUST differ meaningfully from the current symbol body.
- Do NOT submit whitespace-only or formatting-only rewrites.

5. SELF-CHECK BEFORE RETURN
- Verify Python syntax is valid.
- Verify you did not remove unrelated classes/functions/imports.
- Verify each rewrite targets conflict-related logic, not just style.
"""
        result: DevOutput = self.llm.invoke(prompt)
        if not result.rewrites:
            print("DEBUG: Developer returned NO rewrites.")
            raise ValueError("Developer returned no rewrites; please provide at least one symbol rewrite.")

        updated = dict(files)
        changed_any = False
        applied_symbols: list[str] = []
        failed_rewrites: list[dict[str, str]] = []
        for rewrite in result.rewrites:
            filename = rewrite.filename
            if filename not in updated:
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": str(rewrite.symbol or ""),
                        "reason": "unknown file",
                    }
                )
                continue

            symbol = rewrite.symbol.strip()
            if not symbol:
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": "",
                        "reason": "empty symbol",
                    }
                )
                continue

            old_content = updated[filename].replace("\r\n", "\n").replace("\r", "\n")
            if not filename.endswith(".py"):
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": symbol,
                        "reason": "non-python file",
                    }
                )
                continue

            try:
                new_content, resolved_symbol = self._apply_symbol_rewrite(
                    old_content,
                    symbol=symbol,
                    replacement=rewrite.replacement,
                    filename=filename,
                )
            except Exception as e:
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": symbol,
                        "reason": str(e),
                    }
                )
                continue
            if self._is_whitespace_only_change(old_content, new_content):
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": symbol,
                        "reason": "whitespace-only rewrite",
                    }
                )
                continue

            changed_lines, total_lines = self._changed_line_stats(old_content, new_content)
            if total_lines >= 200 and (changed_lines / total_lines) > 0.25:
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": symbol,
                        "reason": (
                            "changed too much unrelated code "
                            f"({changed_lines}/{total_lines} lines)"
                        ),
                    }
                )
                continue

            syntax_error = self._python_syntax_error(filename, new_content)
            if syntax_error:
                failed_rewrites.append(
                    {
                        "filename": filename,
                        "symbol": symbol,
                        "reason": f"syntax error after rewrite: {syntax_error}",
                    }
                )
                continue

            print(
                f"DEBUG: Applying symbol rewrite {filename}:{symbol} "
                f"(old={len(old_content)} chars, new={len(new_content)} chars)"
            )
            updated[filename] = new_content
            if new_content != old_content:
                changed_any = True
                applied_symbols.append(resolved_symbol)

        self.last_revision_meta = {
            "proposed_rewrites": len(result.rewrites),
            "applied_rewrites": len(applied_symbols),
            "applied_symbols": sorted(set(applied_symbols)),
            "failed_rewrites": failed_rewrites,
        }
        if not changed_any:
            failure_preview = "; ".join(
                f"{f.get('filename')}:{f.get('symbol')} -> {f.get('reason')}"
                for f in failed_rewrites[:3]
            )
            if not failure_preview:
                failure_preview = "no rewrite produced a semantic code delta"
            raise ValueError(
                "Developer returned rewrites but no effective file changes. "
                f"Failures: {failure_preview}"
            )

        return updated
