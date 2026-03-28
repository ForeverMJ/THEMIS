from __future__ import annotations

import ast
import difflib
import re
from typing import Any, Dict, List

from pydantic import BaseModel


class SearchReplaceEdit(BaseModel):
    filename: str
    search: str  # Exact lines from original file
    replacement: str  # New lines to replace with


class SymbolRewrite(BaseModel):
    filename: str
    symbol: str
    replacement: str


class RepairHypothesisChoice(BaseModel):
    chosen_hypothesis_label: str = ""
    hypothesis_root_cause: str = ""
    expected_invariant: str = ""
    patch_strategy: str = ""


class DevOutput(BaseModel):
    hypothesis: RepairHypothesisChoice = RepairHypothesisChoice()
    rewrites: List[SymbolRewrite]
    search_replace_edits: List[SearchReplaceEdit] = []


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
        max_files: int | None = None,
    ) -> str:
        # For large files, provide focused excerpts around likely-relevant tokens to make exact-copy edits easier.
        focus_tokens = self._extract_focus_tokens(requirements + "\n\n" + conflict_text)
        formatted_parts: list[str] = []

        file_items = list(files.items())
        if max_files is not None and max_files > 0:
            file_items = file_items[:max_files]

        for name, content in file_items:
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

    def _apply_search_replace_edit(
        self,
        content: str,
        *,
        search: str,
        replacement: str,
        filename: str,
    ) -> tuple[str, bool]:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        
        if isinstance(search, (list, tuple)):
            search = "\n".join(str(s) for s in search)
        if isinstance(replacement, (list, tuple)):
            replacement = "\n".join(str(s) for s in replacement)
        
        search_text = self._strip_code_fence(str(search))
        replacement_text = self._strip_code_fence(str(replacement))

        if not search_text.strip():
            raise ValueError(f"Developer returned empty search block for {filename}")
        if not replacement_text.strip():
            raise ValueError(f"Developer returned empty replacement for {filename}")

        search_lines = search_text.split("\n")
        content_lines = normalized.split("\n")

        best_start = None
        best_ratio = 0.0

        anchors = [
            (i, line) for i, line in enumerate(search_lines) if line.strip() and len(line.strip()) >= 8
        ]
        anchors.sort(key=lambda x: -len(x[1]))

        for before_idx, anchor_line in anchors:
            anchor_stripped = anchor_line.strip()
            positions = [i for i, line in enumerate(content_lines) if line.strip() == anchor_stripped]

            for pos in positions:
                start = pos - before_idx
                if start < 0:
                    continue
                end = start + len(search_lines)
                if end > len(content_lines):
                    continue

                window_lines = content_lines[start:end]
                window = "\n".join(window_lines)
                ratio = difflib.SequenceMatcher(
                    None, self._norm_for_match(search_text), self._norm_for_match(window)
                ).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start

        if best_start is None or best_ratio < 0.85:
            raise ValueError(
                f"Search block not found in {filename} (best ratio: {best_ratio:.2f}). "
                "Copy exact lines from the file."
            )

        delta = self._min_indent(replacement_text) - self._min_indent(search_text)
        aligned_replacement = self._shift_indentation(replacement_text, delta=delta)
        aligned_lines = aligned_replacement.split("\n")

        new_lines = (
            content_lines[:best_start]
            + aligned_lines
            + content_lines[best_start + len(search_lines):]
        )
        updated = "\n".join(new_lines)

        return updated, True

    def revise(
        self,
        files: Dict[str, str],
        requirements: str,
        conflict_report: str | None,
        *,
        force_full_files: set[str] | None = None,
        preferred_edit_mode: str | None = None,
        repair_hypotheses: List[Dict[str, Any]] | None = None,
        preferred_hypothesis_label: str | None = None,
        code_ingredients: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, str]:
        conflict_text = conflict_report or "None"
        mode = str(preferred_edit_mode or "auto").strip().lower()
        if mode not in {"auto", "search_replace", "symbol_rewrite"}:
            mode = "auto"
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
        hypothesis_lines: list[str] = []
        for hypothesis in list(repair_hypotheses or [])[:3]:
            label = str(hypothesis.get("label") or "").strip()
            target_symbol = str(hypothesis.get("target_symbol") or "").strip()
            fault_mechanism = str(hypothesis.get("fault_mechanism") or "").strip()
            expected_fix_behavior = str(hypothesis.get("expected_fix_behavior") or "").strip()
            minimal_edit_scope = str(hypothesis.get("minimal_edit_scope") or "").strip()
            change_operator = str(hypothesis.get("change_operator") or "").strip()
            why_this_target = str(hypothesis.get("why_this_target") or "").strip()
            confidence = hypothesis.get("confidence")
            confidence_text = ""
            if confidence is not None:
                try:
                    confidence_text = f" ({float(confidence):.2f})"
                except Exception:
                    confidence_text = f" ({confidence})"
            if not label:
                continue
            hypothesis_lines.append(
                f"- {label}: target={target_symbol}{confidence_text}; "
                f"mechanism={fault_mechanism}; expected={expected_fix_behavior}; "
                f"scope={minimal_edit_scope}; operator={change_operator}; why={why_this_target}"
            )
        repair_hypotheses_text = "\n".join(hypothesis_lines) if hypothesis_lines else "(none)"
        preferred_hypothesis = str(preferred_hypothesis_label or "").strip()
        if not preferred_hypothesis and hypothesis_lines:
            first_line = hypothesis_lines[0]
            match = re.match(r"-\s+([A-Za-z0-9_-]+):", first_line)
            if match:
                preferred_hypothesis = match.group(1)
        if not preferred_hypothesis:
            preferred_hypothesis = "H1"
        ingredient_lines: list[str] = []
        for item in list(code_ingredients or [])[:3]:
            path = str(item.get("path") or "").strip()
            symbol = str(item.get("symbol") or "").strip()
            role = str(item.get("role") or "").strip()
            snippet = str(item.get("snippet") or "").rstrip()
            if not path or not snippet:
                continue
            ingredient_lines.append(
                f"- {role or 'ingredient'} from {path} for {symbol}:\n"
                f"```python\n{snippet}\n```"
            )
        code_ingredients_text = "\n".join(ingredient_lines) if ingredient_lines else "(none)"
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

Repair Operator Plans (use as candidate generation hints):
{repair_hypotheses_text}

Preferred Plan:
{preferred_hypothesis}

Relevant Code Ingredients (reuse local patterns when helpful):
{code_ingredients_text}

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

4. OUTPUT FORMAT - CHOOSE ONE (IMPORTANT)

**OPTION A: Search/Replace Block (PREFERRED)**
Use this for surgical edits. Return search_replace_edits with:
- filename: target file
- search: EXACT lines from the file you want to replace (min 3 lines, max 20 lines)
- replacement: new lines to replace with

Example:
```
<<<<<<< SEARCH
def process_data(items):
    for item in items:
        print(item)
=======
def process_data(items):
    for item in items:
        if item is not None:
            print(item)
>>>>>>> REPLACE
```

**OPTION B: Symbol Rewrite (FALLBACK)**
Use this for larger refactors. Return rewrites with:
- filename: target file
- symbol: exact Python symbol to replace
  Allowed forms:
  - top-level function: `function_name`
  - top-level class: `ClassName`
  - method: `ClassName.method_name`
- replacement: complete replacement code for that symbol only

Rules:
- Do NOT propose rewrites for files not listed in Allowed Files.
- Do NOT use "..." or placeholders in search blocks.
- Keep unrelated code/comments/docstrings unchanged.
- Preserve indentation; the final file must be syntactically valid.
- replacement MUST differ meaningfully from the current code.
- Do NOT submit whitespace-only or formatting-only rewrites.

5. SELF-CHECK BEFORE RETURN
- Verify Python syntax is valid.
- Verify you did not remove unrelated classes/functions/imports.
- Verify each rewrite targets conflict-related logic, not just style.

6. OPERATOR-PLAN ALIGNMENT (IMPORTANT)
- First choose ONE repair plan label.
- Default to the preferred plan unless the local code strongly contradicts it.
- Keep the plan lightweight and actionable.
- The final edits should directly follow the chosen plan and may reuse the provided code ingredients.

7. EDIT MODE (STRICT)
Current mode: {mode}
- If mode is `search_replace`: return ONLY `search_replace_edits` and set `rewrites` to empty.
- If mode is `symbol_rewrite`: return ONLY `rewrites` and set `search_replace_edits` to empty.
- If mode is `auto`: choose one mode and do not mix both in the same response.
"""
        result: DevOutput = self.llm.invoke(prompt)

        has_symbol_rewrites = result.rewrites and len(result.rewrites) > 0
        has_search_replace = result.search_replace_edits and len(result.search_replace_edits) > 0

        if mode == "search_replace":
            if has_symbol_rewrites:
                raise ValueError(
                    "Developer mixed edit modes: search_replace mode requires empty rewrites."
                )
            if not has_search_replace:
                raise ValueError(
                    "Developer returned no rewrites; search_replace mode requires at least one search_replace_edit."
                )
        elif mode == "symbol_rewrite":
            if has_search_replace:
                raise ValueError(
                    "Developer mixed edit modes: symbol_rewrite mode requires empty search_replace_edits."
                )
            if not has_symbol_rewrites:
                raise ValueError(
                    "Developer returned no rewrites; symbol_rewrite mode requires at least one symbol rewrite."
                )
        elif has_symbol_rewrites and has_search_replace:
            raise ValueError("Developer mixed edit modes in auto mode; return one mode only.")

        if not has_symbol_rewrites and not has_search_replace:
            print("DEBUG: Developer returned NO rewrites.")
            raise ValueError("Developer returned no rewrites; please provide at least one edit.")

        chosen_hypothesis_label = str(result.hypothesis.chosen_hypothesis_label or "").strip()
        if chosen_hypothesis_label.lower() in {"auto", "(auto)"}:
            chosen_hypothesis_label = preferred_hypothesis_label or preferred_hypothesis
        valid_labels = {
            str(item.get("label") or "").strip()
            for item in list(repair_hypotheses or [])
            if str(item.get("label") or "").strip()
        }
        if valid_labels and chosen_hypothesis_label and chosen_hypothesis_label not in valid_labels:
            raise ValueError(
                f"Developer selected unknown repair hypothesis `{chosen_hypothesis_label}`."
            )
        if valid_labels and not chosen_hypothesis_label:
            raise ValueError("Developer must choose one repair plan label.")

        updated = dict(files)
        changed_any = False
        applied_symbols: list[str] = []
        failed_rewrites: list[dict[str, str]] = []
        edits_processed = 0

        if has_search_replace:
            for edit in result.search_replace_edits:
                filename = edit.filename
                if filename not in updated:
                    failed_rewrites.append({
                        "filename": filename,
                        "symbol": "[search/replace]",
                        "reason": "unknown file",
                    })
                    continue

                old_content = updated[filename].replace("\r\n", "\n").replace("\r", "\n")

                search_val = edit.search
                replacement_val = edit.replacement
                if isinstance(search_val, (list, tuple)):
                    search_val = "\n".join(str(s) for s in search_val)
                if isinstance(replacement_val, (list, tuple)):
                    replacement_val = "\n".join(str(s) for s in replacement_val)

                try:
                    new_content, success = self._apply_search_replace_edit(
                        old_content,
                        search=str(search_val),
                        replacement=str(replacement_val),
                        filename=filename,
                    )
                except Exception as e:
                    failed_rewrites.append({
                        "filename": filename,
                        "symbol": "[search/replace]",
                        "reason": str(e),
                    })
                    continue

                if self._is_whitespace_only_change(old_content, new_content):
                    failed_rewrites.append({
                        "filename": filename,
                        "symbol": "[search/replace]",
                        "reason": "whitespace-only change",
                    })
                    continue

                syntax_error = self._python_syntax_error(filename, new_content)
                if syntax_error:
                    failed_rewrites.append({
                        "filename": filename,
                        "symbol": "[search/replace]",
                        "reason": f"syntax error: {syntax_error}",
                    })
                    continue

                print(f"DEBUG: Applying search/replace edit {filename}")
                updated[filename] = new_content
                if new_content != old_content:
                    changed_any = True
                    edits_processed += 1

        if has_symbol_rewrites:
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
            "chosen_hypothesis_label": chosen_hypothesis_label or None,
            "hypothesis_root_cause": str(result.hypothesis.hypothesis_root_cause or "").strip() or None,
            "expected_invariant": str(result.hypothesis.expected_invariant or "").strip() or None,
            "patch_strategy": str(result.hypothesis.patch_strategy or "").strip() or None,
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
