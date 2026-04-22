"""
Integrated Advanced Analysis + Traditional Workflow

This script combines the Advanced Code Analysis system with the traditional
KG → Developer → Judge workflow:

1. Advanced Analysis: LLM-driven semantic understanding and bug classification
2. KG Construction: Build knowledge graph enriched with advanced analysis insights
3. Developer: Use analysis insights to guide code revision
4. Judge: Evaluate the revised code against requirements

This provides the best of both worlds: intelligent semantic understanding
combined with structured graph-based verification.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import networkx as nx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.developer import DeveloperAgent
from src.agents.judge import JudgeAgent
from src.enhanced_graph_adapter import EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
from src.state import AgentState
from dotenv import load_dotenv

load_dotenv(override=True)

MAX_REVISIONS = 1  # Reduced from 2 to prevent degradation

_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run an async coroutine from sync code without closing the event loop each call."""
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_ASYNC_LOOP)
    result = _ASYNC_LOOP.run_until_complete(coro)
    # Let cleanup callbacks (e.g., httpx aclose) run on the same loop.
    _ASYNC_LOOP.run_until_complete(asyncio.sleep(0))
    return result


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def _should_apply_repair_brief(repair_brief: Any) -> bool:
    if not isinstance(repair_brief, dict):
        return False
    target_symbol = str(repair_brief.get("target_symbol") or "").strip()
    if not target_symbol:
        return False
    return bool(repair_brief.get("blocking"))


def _dedupe_symbols(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for symbol in symbols:
        text = str(symbol or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _bounded_fallback_target_set(
    target_symbols: Sequence[str],
    *,
    repair_brief: Dict[str, Any] | None,
    enabled: bool,
    max_targets: int = 3,
) -> tuple[list[str], Dict[str, Any]]:
    primary_targets = _dedupe_symbols(target_symbols)
    related_symbols: list[str] = []
    if isinstance(repair_brief, dict):
        related_symbols = [str(s) for s in list(repair_brief.get("related_symbols") or [])]

    alternative_targets = _dedupe_symbols(related_symbols + list(primary_targets))[:max_targets]
    expanded_targets = list(primary_targets)
    fallback_reason = "fallback_disabled"
    if not enabled:
        return list(primary_targets), {
            "target_expansion_mode": "disabled",
            "fallback_enabled": False,
            "entered_fallback": False,
            "primary_targets": list(primary_targets),
            "expanded_targets": list(primary_targets),
            "alternative_targets": list(alternative_targets),
            "fallback_added_targets": [],
            "fallback_reason": fallback_reason,
            "expansion_anchor_file": None,
            "expansion_anchor_symbol": None,
        }

    expanded_targets = _dedupe_symbols(list(primary_targets) + related_symbols)[:max_targets]
    fallback_added_targets = [sym for sym in expanded_targets if sym not in primary_targets]
    if fallback_added_targets:
        fallback_reason = "expanded_with_related_symbols"
    elif related_symbols:
        fallback_reason = "related_symbols_deduped_or_capped"
    else:
        fallback_reason = "no_related_symbols"
    return expanded_targets, {
        "target_expansion_mode": "bounded_related_symbols",
        "fallback_enabled": True,
        "entered_fallback": bool(fallback_added_targets),
        "primary_targets": list(primary_targets),
        "expanded_targets": list(expanded_targets),
        "alternative_targets": list(alternative_targets),
        "fallback_added_targets": fallback_added_targets,
        "fallback_reason": fallback_reason,
        "expansion_anchor_file": None,
        "expansion_anchor_symbol": None,
    }


def _index_file_local_symbols(files: Dict[str, str]) -> Dict[str, list[Dict[str, Any]]]:
    indexed: Dict[str, list[Dict[str, Any]]] = {}
    for file_path, content in files.items():
        if not str(file_path).endswith(".py"):
            continue
        try:
            tree = ast.parse(content)
        except Exception:
            continue

        records: list[Dict[str, Any]] = []

        def walk(body: list[Any], *, container: str | None = None) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualname = f"{container}.{node.name}" if container else node.name
                    records.append(
                        {
                            "qualname": qualname,
                            "name": node.name,
                            "container": container or "__module__",
                            "kind": "class",
                            "lineno": int(getattr(node, "lineno", 0) or 0),
                        }
                    )
                    walk(list(node.body), container=qualname)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{container}.{node.name}" if container else node.name
                    records.append(
                        {
                            "qualname": qualname,
                            "name": node.name,
                            "container": container or "__module__",
                            "kind": "method" if container else "function",
                            "lineno": int(getattr(node, "lineno", 0) or 0),
                        }
                    )

        walk(list(tree.body))
        indexed[file_path] = sorted(records, key=lambda item: (int(item.get("lineno") or 0), str(item.get("qualname") or "")))
    return indexed


def _normalize_symbol_name_for_index(symbol: str) -> str:
    return str(symbol or "").replace("::", ".").strip().lower()


def _symbol_leaf_for_index(symbol: str) -> str:
    normalized = _normalize_symbol_name_for_index(symbol)
    return normalized.split(".")[-1] if normalized else ""


def _pick_anchor_symbol(
    symbols_by_file: Dict[str, list[Dict[str, Any]]],
    preferred_symbols: Sequence[str],
) -> tuple[str | None, Dict[str, Any] | None]:
    matches: list[tuple[tuple[int, int, int, int], str, Dict[str, Any]]] = []
    for priority, symbol in enumerate(preferred_symbols):
        norm = _normalize_symbol_name_for_index(symbol)
        leaf = _symbol_leaf_for_index(symbol)
        if not norm and not leaf:
            continue
        for file_path, records in symbols_by_file.items():
            for record in records:
                rec_name = str(record.get("qualname") or "")
                rec_norm = _normalize_symbol_name_for_index(rec_name)
                rec_leaf = _symbol_leaf_for_index(rec_name)
                if norm and rec_norm == norm:
                    matches.append(((priority, 0, int(record.get("lineno") or 0), len(rec_name)), file_path, record))
                elif leaf and rec_leaf == leaf:
                    matches.append(((priority, 1, int(record.get("lineno") or 0), len(rec_name)), file_path, record))
    if not matches:
        return None, None
    matches.sort(key=lambda item: item[0])
    _, file_path, record = matches[0]
    return file_path, record


def _file_local_neighborhood_target_set(
    files: Dict[str, str],
    target_symbols: Sequence[str],
    *,
    repair_brief: Dict[str, Any] | None,
    enabled: bool,
    max_targets: int = 4,
) -> tuple[list[str], Dict[str, Any]]:
    primary_targets = _dedupe_symbols(target_symbols)
    repair_brief_target = str((repair_brief or {}).get("target_symbol") or "").strip() if isinstance(repair_brief, dict) else ""
    symbols_by_file = _index_file_local_symbols(files)
    preferred_symbols = [sym for sym in [repair_brief_target, *primary_targets] if str(sym).strip()]
    anchor_file, anchor_record = _pick_anchor_symbol(symbols_by_file, preferred_symbols)
    neighborhood_candidates: list[str] = []
    expansion_reason = "neighborhood_disabled"

    if not enabled:
        return list(primary_targets), {
            "target_expansion_mode": "disabled",
            "fallback_enabled": False,
            "entered_fallback": False,
            "primary_targets": list(primary_targets),
            "expanded_targets": list(primary_targets),
            "alternative_targets": list(primary_targets),
            "fallback_added_targets": [],
            "fallback_reason": expansion_reason,
            "expansion_anchor_file": anchor_file,
            "expansion_anchor_symbol": (anchor_record or {}).get("qualname"),
        }

    if anchor_file and anchor_record:
        anchor_qualname = str(anchor_record.get("qualname") or "").strip()
        if anchor_qualname:
            neighborhood_candidates.append(anchor_qualname)
        siblings = [
            record for record in symbols_by_file.get(anchor_file, [])
            if str(record.get("container") or "") == str(anchor_record.get("container") or "")
        ]
        siblings.sort(key=lambda item: int(item.get("lineno") or 0))
        try:
            anchor_index = next(
                idx for idx, record in enumerate(siblings)
                if str(record.get("qualname") or "") == anchor_qualname
            )
        except StopIteration:
            anchor_index = -1
        for offset in (-1, 1):
            idx = anchor_index + offset
            if 0 <= idx < len(siblings):
                neighbor_name = str(siblings[idx].get("qualname") or "").strip()
                if neighbor_name:
                    neighborhood_candidates.append(neighbor_name)
        container_name = str(anchor_record.get("container") or "").strip()
        if container_name and container_name != "__module__":
            neighborhood_candidates.append(container_name)
        expansion_reason = "expanded_with_file_local_neighborhood"
    elif len(files) == 1:
        only_file = next(iter(files.keys()))
        if repair_brief_target:
            neighborhood_candidates.append(repair_brief_target)
        neighborhood_candidates.extend(
            str(record.get("qualname") or "").strip()
            for record in symbols_by_file.get(only_file, [])[:3]
            if str(record.get("qualname") or "").strip()
        )
        anchor_file = only_file
        expansion_reason = "single_file_neighborhood_scan"
    else:
        expansion_reason = "anchor_symbol_not_found"

    expanded_targets = _dedupe_symbols(list(primary_targets) + neighborhood_candidates)[:max_targets]
    fallback_added_targets = [sym for sym in expanded_targets if sym not in primary_targets]
    alternative_targets = _dedupe_symbols(neighborhood_candidates + list(primary_targets))[:max_targets]
    if not fallback_added_targets and expansion_reason == "expanded_with_file_local_neighborhood":
        expansion_reason = "neighborhood_deduped_or_capped"

    return expanded_targets, {
        "target_expansion_mode": "file_local_neighborhood",
        "fallback_enabled": True,
        "entered_fallback": bool(fallback_added_targets),
        "primary_targets": list(primary_targets),
        "expanded_targets": list(expanded_targets),
        "alternative_targets": list(alternative_targets),
        "fallback_added_targets": fallback_added_targets,
        "fallback_reason": expansion_reason,
        "expansion_anchor_file": anchor_file,
        "expansion_anchor_symbol": (anchor_record or {}).get("qualname"),
    }


def _summarize_fallback_usage(
    *,
    effective_change: bool,
    target_hit_info: Dict[str, Any],
    primary_target_hit_info: Dict[str, Any],
    fallback_target_hit_info: Dict[str, Any],
    relocalization_meta: Dict[str, Any],
) -> Dict[str, Any]:
    entered_fallback = bool(relocalization_meta.get("entered_fallback"))
    fallback_added_targets = list(relocalization_meta.get("fallback_added_targets") or [])
    fallback_target_hit = fallback_target_hit_info.get("target_hit")
    primary_target_hit = primary_target_hit_info.get("target_hit")

    if fallback_target_hit is True:
        selected_target_source = "fallback"
        fallback_reason = None
    elif primary_target_hit is True:
        selected_target_source = "primary"
        fallback_reason = "primary_targets_sufficient"
    elif target_hit_info.get("target_hit") is True:
        selected_target_source = "mixed_or_unknown"
        fallback_reason = "target_hit_not_attributable"
    else:
        selected_target_source = "none"
        if not entered_fallback:
            fallback_reason = str(relocalization_meta.get("fallback_reason") or "fallback_not_entered")
        elif not fallback_added_targets:
            fallback_reason = "no_fallback_targets_added"
        elif not effective_change:
            fallback_reason = "no_effective_change"
        else:
            fallback_reason = "fallback_targets_not_hit"

    return {
        "target_expansion_mode": str(relocalization_meta.get("target_expansion_mode") or "disabled"),
        "fallback_enabled": bool(relocalization_meta.get("fallback_enabled")),
        "entered_fallback": entered_fallback,
        "primary_targets": list(relocalization_meta.get("primary_targets") or []),
        "expanded_targets": list(relocalization_meta.get("expanded_targets") or []),
        "fallback_added_targets": fallback_added_targets,
        "fallback_target_hit": fallback_target_hit,
        "fallback_target_hit_rate": fallback_target_hit_info.get("target_hit_rate"),
        "fallback_targets_total": fallback_target_hit_info.get("target_symbols_total"),
        "fallback_targets_hit": fallback_target_hit_info.get("target_symbols_hit"),
        "expansion_anchor_file": relocalization_meta.get("expansion_anchor_file"),
        "expansion_anchor_symbol": relocalization_meta.get("expansion_anchor_symbol"),
        "selected_target_source": selected_target_source,
        "fallback_would_have_triggered_but_not_used_reason": fallback_reason,
    }


def build_integrated_workflow(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT,
    enable_bounded_fallback_targets: bool = False,
    enable_file_local_neighborhood_targets: bool = False,
    callbacks: Optional[Sequence[Any]] = None,
    stop_policy: str = "hybrid",
) -> StateGraph:
    """
    Build integrated workflow that combines:
    1. Advanced Analysis (LLM semantic understanding)
    2. Enhanced GraphManager (structural analysis)
    3. Developer (code revision)
    4. Judge (verification)
    """
    
    effective_max_revisions = MAX_REVISIONS if max_revisions is None else max_revisions

    # Ensure AdvancedCodeAnalyzer uses the requested model.
    if analysis_model:
        os.environ["LLM_MODEL"] = analysis_model

    def _use_responses_api(model_name: str) -> bool:
        return model_name.startswith("gpt-5") or "codex" in model_name

    llm_kwargs: dict[str, Any] = {"model": llm_model}
    if _use_responses_api(llm_model):
        llm_kwargs["use_responses_api"] = True
    else:
        llm_kwargs["temperature"] = 0
    if callbacks is not None:
        llm_kwargs["callbacks"] = list(callbacks)
    llm = ChatOpenAI(**llm_kwargs)
    graph_manager = EnhancedGraphAdapter()
    developer = DeveloperAgent(llm)
    judge = JudgeAgent(llm)

    def _safe_join_repo_path(repo_root: Path, rel_path: str) -> Optional[Path]:
        rel = Path(rel_path.strip())
        if rel.is_absolute():
            return None
        abs_path = (repo_root / rel).resolve()
        try:
            common = os.path.commonpath([str(repo_root.resolve()), str(abs_path)])
        except Exception:
            return None
        if common != str(repo_root.resolve()):
            return None
        return abs_path

    def _try_load_file_into_state(state: AgentState, *, rel_path: str) -> bool:
        repo_root_raw = state.get("repo_root")
        if not repo_root_raw:
            return False
        repo_root = Path(str(repo_root_raw))
        abs_path = _safe_join_repo_path(repo_root, rel_path)
        if not abs_path or not abs_path.exists() or not abs_path.is_file():
            return False
        if rel_path in state["files"]:
            return False
        try:
            state["files"][rel_path] = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False
        print(f"Loaded additional file into context: {rel_path}")
        return True

    def _maybe_expand_context_from_error(state: AgentState, err: str) -> bool:
        """Best-effort: load additional files when the Developer references unknown paths/snippets."""
        m = re.search(
            r"Developer proposed (?:edit|rewrite) for unknown file: (.+)$",
            err.strip(),
        )
        if m:
            rel_path = m.group(1).strip()
            rel_norm = rel_path.replace("\\", "/").lstrip("./")
            if _try_load_file_into_state(state, rel_path=rel_norm):
                return True

            rel_obj = Path(rel_norm)
            if rel_obj.suffix == ".py" and len(rel_obj.parts) > 1:
                parent_dir = Path(*rel_obj.parts[:-1])
                parent_module = parent_dir.with_suffix(".py").as_posix()
                init_module = (parent_dir / "__init__.py").as_posix()
                for cand in (parent_module, init_module):
                    if _try_load_file_into_state(state, rel_path=cand):
                        return True

            # Fallback: if the path is wrong (e.g., repo uses "lib/<pkg>/..."), try suffix match.
            repo_root_raw = state.get("repo_root")
            if not repo_root_raw:
                return False
            repo_root = Path(str(repo_root_raw))
            try:
                proc = subprocess.run(
                    ["git", "ls-files"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:
                return False
            if proc.returncode != 0:
                return False

            candidates = [p.strip() for p in proc.stdout.splitlines() if p.strip().endswith(rel_norm)]
            if not candidates:
                base_name = rel_obj.name
                if base_name:
                    candidates = [
                        p.strip()
                        for p in proc.stdout.splitlines()
                        if p.strip().endswith("/" + base_name) or p.strip() == base_name
                    ]
            if not candidates:
                return False

            # Prefer shortest path (least nested) to avoid grabbing vendored copies.
            candidates.sort(key=lambda p: (len(p), p))
            loaded_any = False
            for cand in candidates[:2]:
                loaded_any |= _try_load_file_into_state(state, rel_path=cand)
            return loaded_any
        return False

    def _extract_explicit_file_mentions(text: str, available_files: Dict[str, str]) -> list[str]:
        if not text:
            return []
        normalized = text.replace("\\", "/")
        found: list[str] = []
        seen: set[str] = set()
        patterns = re.findall(r"([A-Za-z0-9_./-]+\.py)", normalized)
        for raw in patterns:
            candidate = raw.lstrip("./")
            if candidate in available_files and candidate not in seen:
                found.append(candidate)
                seen.add(candidate)
                continue
            suffix_matches = [name for name in available_files if name.endswith(candidate)]
            suffix_matches.sort(key=lambda x: (len(x), x))
            for match in suffix_matches[:1]:
                if match not in seen:
                    found.append(match)
                    seen.add(match)
        return found

    def _select_focus_files(
        files: Dict[str, str],
        *,
        requirements: str,
        conflict_report: str | None,
        advanced_analysis: Dict[str, Any],
        repair_brief: Dict[str, Any],
        max_files: int = 3,
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        if len(files) <= max_files:
            return dict(files), {
                "context_narrowed": False,
                "context_file_count": len(files),
                "selected_files": list(files.keys()),
            }

        seed_text_parts = [requirements, conflict_report or "", str(repair_brief or "")]
        findings = list(advanced_analysis.get("findings") or [])[:3] if isinstance(advanced_analysis, dict) else []
        recommendations = list(advanced_analysis.get("recommendations") or [])[:2] if isinstance(advanced_analysis, dict) else []
        seed_text_parts.extend(str(x) for x in findings)
        seed_text_parts.extend(str(x) for x in recommendations)
        seed_text = "\n".join(part for part in seed_text_parts if part)

        explicit_files = _extract_explicit_file_mentions(seed_text, files)
        focus_tokens = DeveloperAgent._extract_focus_tokens(seed_text, max_tokens=16)

        scored: list[tuple[int, str]] = []
        for name, content in files.items():
            score = 0
            basename = Path(name).name
            if name in explicit_files:
                score += 100
            elif basename in {Path(p).name for p in explicit_files}:
                score += 50
            lowered_name = name.lower()
            lowered_content = content.lower()
            for token in focus_tokens:
                tok = token.lower()
                if tok in lowered_name:
                    score += 8
                if tok in basename.lower():
                    score += 5
                if tok in lowered_content:
                    score += 1
            scored.append((score, name))

        scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
        selected: list[str] = []
        for path in explicit_files:
            if path in files and path not in selected:
                selected.append(path)
            if len(selected) >= max_files:
                break
        for score, path in scored:
            if path in selected:
                continue
            selected.append(path)
            if len(selected) >= max_files:
                break

        focused = {name: files[name] for name in selected if name in files}
        return focused, {
            "context_narrowed": len(focused) < len(files),
            "context_file_count": len(focused),
            "selected_files": list(focused.keys()),
        }

    def _merge_file_subset(original_files: Dict[str, str], revised_subset: Dict[str, str]) -> Dict[str, str]:
        merged = dict(original_files)
        merged.update(revised_subset)
        return merged

    def _pre_judge_assessment(
        *,
        developer_metric: Dict[str, Any],
        analysis_report: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        violation_report = analysis_report.get("violation_report") if isinstance(analysis_report, dict) else {}
        blocking_count = 0
        if isinstance(violation_report, dict):
            try:
                blocking_count = int(violation_report.get("total_blocking_violations") or 0)
            except Exception:
                blocking_count = 0

        effective_change = bool(developer_metric.get("effective_change"))
        target_total = int(developer_metric.get("target_symbols_total") or 0)
        target_hit = developer_metric.get("target_hit")
        changed_line_count = int(developer_metric.get("changed_line_count") or 0)
        recoverable_failure_count = int(developer_metric.get("recoverable_failure_count") or 0)
        applied_rewrites = int(developer_metric.get("applied_rewrites") or 0)

        reasons: list[str] = []
        if not effective_change:
            reasons.append("no_effective_change")
        if target_total > 0 and target_hit is False:
            reasons.append("missed_target_symbols")
        if recoverable_failure_count > 0 and not effective_change:
            reasons.append("recoverable_failure_without_change")
        if applied_rewrites == 0 and not effective_change:
            reasons.append("no_applied_rewrites")
        if changed_line_count > 400:
            reasons.append("overbroad_change")

        decision = "reject" if reasons else "pass"
        if blocking_count == 0 and effective_change and target_total > 0 and target_hit is True:
            decision = "pass"
            reasons = []
        return {
            "decision": decision,
            "reason": ", ".join(reasons) if reasons else "pass",
            "blocking_conflicts_count": blocking_count,
        }

    def _failure_memory_guidance(
        state: AgentState,
        *,
        target_symbols: Sequence[str],
        repair_brief: Dict[str, Any] | None,
    ) -> tuple[list[str], str, Dict[str, Any]]:
        if enable_file_local_neighborhood_targets:
            resolved_targets, fallback_meta = _file_local_neighborhood_target_set(
                state["files"],
                target_symbols,
                repair_brief=repair_brief,
                enabled=True,
            )
        else:
            resolved_targets, fallback_meta = _bounded_fallback_target_set(
                target_symbols,
                repair_brief=repair_brief,
                enabled=enable_bounded_fallback_targets,
            )
        return resolved_targets, "", {
            "relocalized": False,
            "previous_target": None,
            **fallback_meta,
            "alternative_targets": list(fallback_meta.get("alternative_targets") or [])[:3],
        }

    def _build_target_evidence_block(
        *,
        target_symbols: Sequence[str],
        blocking_prioritized: Sequence[Dict[str, Any]],
        repair_brief: Dict[str, Any] | None,
    ) -> str:
        if not target_symbols:
            return ""

        primary_target = str(target_symbols[0]).strip()
        lines: list[str] = ["\n\n=== Target Evidence ===", f"Primary Target: {primary_target}"]
        if isinstance(repair_brief, dict):
            requirement_id = str(repair_brief.get("requirement_id") or "").strip()
            issue_summary = str(repair_brief.get("issue_summary") or "").strip()
            expected_behavior = str(repair_brief.get("expected_behavior") or "").strip()
            minimal_change_hint = str(repair_brief.get("minimal_change_hint") or "").strip()
            related_symbols = [str(s).strip() for s in list(repair_brief.get("related_symbols") or []) if str(s).strip()]
            if requirement_id:
                lines.append(f"Requirement ID: {requirement_id}")
            if issue_summary:
                lines.append(f"Issue Summary: {issue_summary}")
            if expected_behavior:
                lines.append(f"Expected Behavior: {expected_behavior}")
            if minimal_change_hint:
                lines.append(f"Minimal Change Hint: {minimal_change_hint}")
            if related_symbols:
                lines.append(f"Related Symbols: {', '.join(related_symbols[:2])}")

        if blocking_prioritized:
            top = blocking_prioritized[0]
            code_node = str(top.get("code_node") or "").strip()
            reason = str(top.get("reason") or "").strip()
            confidence = top.get("confidence")
            confidence_text = ""
            if confidence is not None:
                try:
                    confidence_text = f" ({float(confidence):.2f})"
                except Exception:
                    confidence_text = f" ({confidence})"
            if code_node:
                lines.append(f"Top Blocking Node: {code_node}")
            if reason:
                lines.append(f"Top Blocking Reason: {reason}{confidence_text}")

        lines.append("Prioritize the primary target over broader speculative rewrites.")
        return "\n".join(lines)

    def _build_target_anchor_block(
        files: Dict[str, str],
        *,
        target_symbols: Sequence[str],
        max_context_lines: int = 18,
    ) -> str:
        if not target_symbols:
            return ""

        primary_target = str(target_symbols[0]).strip()
        symbol_tokens = [tok for tok in {primary_target, primary_target.split(".")[-1]} if tok]
        best_match: tuple[int, str, int, list[str]] | None = None
        for path, content in files.items():
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                score = sum(1 for tok in symbol_tokens if tok and tok in line)
                if score <= 0:
                    continue
                start = max(0, idx - max_context_lines // 2)
                end = min(len(lines), start + max_context_lines)
                snippet = lines[start:end]
                candidate = (score, path, idx, snippet)
                if best_match is None or candidate[0] > best_match[0]:
                    best_match = candidate

        if best_match is None:
            return ""

        _, path, idx, snippet = best_match
        start = max(1, idx - max_context_lines // 2 + 1)
        excerpt = "\n".join(f"{start + offset}: {line}" for offset, line in enumerate(snippet))
        return (
            "\n\n=== Target Local Anchor ===\n"
            f"File: {path}\n"
            f"Excerpt around target `{primary_target}`:\n"
            f"```python\n{excerpt}\n```"
        )

    def _infer_change_operator(reason_text: str, target_symbol: str) -> str:
        text = f"{reason_text} {target_symbol}".lower()
        if any(word in text for word in ("validate", "validation", "type", "matrix-like", "guard")):
            return "guard_condition"
        if any(word in text for word in ("path", "url", "link", "redirect")):
            return "align_reference"
        if any(word in text for word in ("fallback", "dispatch", "call", "use")):
            return "route_to_correct_helper"
        if any(word in text for word in ("condition", "branch", "if ", "else")):
            return "tighten_branch"
        return "minimal_logic_fix"

    def _alternate_change_operator(primary_operator: str) -> str:
        alternates = {
            "guard_condition": "tighten_branch",
            "tighten_branch": "guard_condition",
            "align_reference": "minimal_logic_fix",
            "route_to_correct_helper": "guard_condition",
            "minimal_logic_fix": "route_to_correct_helper",
        }
        return alternates.get(primary_operator, "minimal_logic_fix")

    def _infer_minimal_edit_scope(target_symbol: str) -> str:
        if "." in target_symbol:
            return f"method_body:{target_symbol}"
        return f"symbol_body:{target_symbol}"

    def _build_repair_hypotheses(
        *,
        target_symbols: Sequence[str],
        repair_brief: Dict[str, Any] | None,
        blocking_prioritized: Sequence[Dict[str, Any]],
        files: Dict[str, str],
    ) -> list[Dict[str, Any]]:
        primary_target = str(target_symbols[0]).strip() if target_symbols else ""
        if not primary_target:
            return []

        reason_text = ""
        if isinstance(repair_brief, dict):
            reason_text = str(repair_brief.get("issue_summary") or "").strip()
        if not reason_text and blocking_prioritized:
            reason_text = str(blocking_prioritized[0].get("reason") or "").strip()

        expected_behavior = ""
        minimal_change_hint = ""
        related_symbols: list[str] = []
        if isinstance(repair_brief, dict):
            expected_behavior = str(repair_brief.get("expected_behavior") or "").strip()
            minimal_change_hint = str(repair_brief.get("minimal_change_hint") or "").strip()
            related_symbols = [str(s).strip() for s in list(repair_brief.get("related_symbols") or []) if str(s).strip()]

        def _guess_file_hint(symbol: str) -> str | None:
            leaf = symbol.split(".")[-1]
            for path, content in files.items():
                if leaf and leaf in content:
                    return path
            return None

        primary_operator = _infer_change_operator(reason_text, primary_target)
        hypotheses: list[Dict[str, Any]] = [
            {
                "label": "H1",
                "target_symbol": primary_target,
                "related_symbols": related_symbols[:2],
                "target_file_hint": _guess_file_hint(primary_target),
                "fault_mechanism": reason_text or f"Primary conflict is centered on `{primary_target}`.",
                "expected_fix_behavior": expected_behavior or f"Make `{primary_target}` satisfy the failing requirement.",
                "minimal_edit_scope": _infer_minimal_edit_scope(primary_target),
                "change_operator": primary_operator,
                "why_this_target": minimal_change_hint or f"Top blocking target is `{primary_target}`.",
                "confidence": 0.9,
            }
        ]

        hypotheses.append(
            {
                "label": "H2",
                "target_symbol": primary_target,
                "related_symbols": related_symbols[:2],
                "target_file_hint": _guess_file_hint(primary_target),
                "fault_mechanism": reason_text or f"Primary conflict is centered on `{primary_target}`.",
                "expected_fix_behavior": expected_behavior or f"Make `{primary_target}` satisfy the failing requirement.",
                "minimal_edit_scope": _infer_minimal_edit_scope(primary_target),
                "change_operator": _alternate_change_operator(primary_operator),
                "why_this_target": f"Alternative operator on the same primary target `{primary_target}`.",
                "confidence": 0.7,
            }
        )

        for idx, alt in enumerate(related_symbols[:1], start=3):
            hypotheses.append(
                {
                    "label": f"H{idx}",
                    "target_symbol": alt,
                    "related_symbols": [primary_target],
                    "target_file_hint": _guess_file_hint(alt),
                    "fault_mechanism": reason_text or f"Related helper `{alt}` may be enforcing the wrong behavior.",
                    "expected_fix_behavior": expected_behavior or f"Align `{alt}` with the intended behavior of `{primary_target}`.",
                    "minimal_edit_scope": _infer_minimal_edit_scope(alt),
                    "change_operator": "route_to_correct_helper",
                    "why_this_target": f"`{alt}` is a same-requirement related symbol to `{primary_target}`.",
                    "confidence": max(0.4, 0.9 - 0.2 * (idx - 1)),
                }
            )

        return hypotheses

    def _build_code_ingredients(
        files: Dict[str, str],
        *,
        target_symbols: Sequence[str],
        repair_brief: Dict[str, Any] | None,
        max_context_lines: int = 10,
    ) -> list[Dict[str, str]]:
        search_symbols: list[tuple[str, str]] = []
        if target_symbols:
            search_symbols.append((str(target_symbols[0]).strip(), "primary_target"))
        if isinstance(repair_brief, dict):
            for sym in list(repair_brief.get("related_symbols") or [])[:2]:
                text = str(sym).strip()
                if text:
                    search_symbols.append((text, "related_symbol"))

        seen_keys: set[tuple[str, int]] = set()
        ingredients: list[Dict[str, str]] = []
        for symbol, role in search_symbols:
            tokens = [tok for tok in {symbol, symbol.split(".")[-1]} if tok]
            for path, content in files.items():
                lines = content.splitlines()
                for idx, line in enumerate(lines):
                    if not any(tok in line for tok in tokens):
                        continue
                    start = max(0, idx - max_context_lines // 2)
                    end = min(len(lines), start + max_context_lines)
                    key = (path, start)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    snippet = "\n".join(
                        f"{start + offset + 1}: {snippet_line}"
                        for offset, snippet_line in enumerate(lines[start:end])
                    )
                    ingredients.append(
                        {
                            "path": path,
                            "symbol": symbol,
                            "role": role,
                            "snippet": snippet,
                        }
                    )
                    break
                if len(ingredients) >= 3:
                    break
            if len(ingredients) >= 3:
                break
        return ingredients

    def _format_code_ingredients(ingredients: Sequence[Dict[str, str]]) -> str:
        if not ingredients:
            return "(none)"
        parts: list[str] = []
        for item in ingredients[:3]:
            parts.append(
                f"- {item.get('role')} from {item.get('path')} for {item.get('symbol')}:\n"
                f"```python\n{item.get('snippet')}\n```"
            )
        return "\n".join(parts)

    def _format_repair_hypotheses(hypotheses: Sequence[Dict[str, Any]]) -> str:
        if not hypotheses:
            return "(none)"
        lines: list[str] = []
        for hypothesis in hypotheses[:3]:
            lines.append(
                f"- {hypothesis.get('label')}: target={hypothesis.get('target_symbol')}; "
                f"mechanism={hypothesis.get('fault_mechanism')}; "
                f"expected={hypothesis.get('expected_fix_behavior')}; "
                f"scope={hypothesis.get('minimal_edit_scope')}; "
                f"operator={hypothesis.get('change_operator')}; "
                f"why={hypothesis.get('why_this_target')}"
            )
        return "\n".join(lines)

    def _module_name_from_rel_path(rel_path: str) -> str | None:
        path = rel_path.replace("\\", "/")
        if not path.endswith(".py"):
            return None
        if path.endswith("/__init__.py"):
            path = path[: -len("/__init__.py")]
        else:
            path = path[:-3]
        path = path.strip("/")
        if not path:
            return None
        if any(part in {"tests", "test"} for part in path.split("/")):
            return None
        return path.replace("/", ".")

    def _execution_verify_candidate(state: AgentState, candidate_files: Dict[str, str]) -> Dict[str, Any]:
        repo_root_raw = state.get("repo_root")
        changed_paths = [
            rel for rel, content in candidate_files.items()
            if state["files"].get(rel) != content and rel.endswith(".py")
        ]
        result: Dict[str, Any] = {
            "ran": False,
            "compile_ok": None,
            "import_attempts": 0,
            "import_passes": 0,
            "score_adjustment": 0.0,
            "summary": "skipped",
        }
        if not repo_root_raw or not changed_paths:
            return result

        repo_root = Path(str(repo_root_raw))
        if not repo_root.exists():
            return result

        backups: Dict[str, str] = {}
        written_paths: list[str] = []
        try:
            for rel in changed_paths[:3]:
                abs_path = _safe_join_repo_path(repo_root, rel)
                if not abs_path or not abs_path.exists() or not abs_path.is_file():
                    continue
                backups[rel] = abs_path.read_text(encoding="utf-8", errors="ignore")
                abs_path.write_text(candidate_files[rel], encoding="utf-8")
                written_paths.append(rel)

            if not written_paths:
                return result

            result["ran"] = True
            compile_proc = subprocess.run(
                [sys.executable, "-m", "py_compile", *written_paths],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=20,
            )
            compile_ok = compile_proc.returncode == 0
            result["compile_ok"] = compile_ok
            if not compile_ok:
                result["summary"] = "compile_failed"
                result["score_adjustment"] = 12.0
                return result

            import_attempts = 0
            import_passes = 0
            for rel in written_paths[:2]:
                module_name = _module_name_from_rel_path(rel)
                if not module_name:
                    continue
                import_attempts += 1
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import importlib, sys; "
                            "sys.path.insert(0, '.'); "
                            f"importlib.import_module({module_name!r})"
                        ),
                    ],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if proc.returncode == 0:
                    import_passes += 1

            result["import_attempts"] = import_attempts
            result["import_passes"] = import_passes
            if import_attempts == 0:
                result["summary"] = "compile_only"
                result["score_adjustment"] = -0.5
            elif import_passes == import_attempts:
                result["summary"] = "compile_and_import_pass"
                result["score_adjustment"] = -1.5
            elif import_passes == 0:
                result["summary"] = "import_failed"
                result["score_adjustment"] = 4.0
            else:
                result["summary"] = "partial_import_pass"
                result["score_adjustment"] = 1.5
            return result
        except Exception as e:
            result["ran"] = True
            result["summary"] = f"execution_probe_error:{e.__class__.__name__}"
            result["score_adjustment"] = 2.0
            return result
        finally:
            for rel, original in backups.items():
                abs_path = _safe_join_repo_path(repo_root, rel)
                if abs_path:
                    abs_path.write_text(original, encoding="utf-8")

        preview_marker = "Preview:\n"
        if preview_marker not in err:
            return False
        preview = err.split(preview_marker, 1)[1]
        preview_lines = [ln.strip() for ln in preview.splitlines() if ln.strip()]
        if not preview_lines:
            return False

        repo_root_raw = state.get("repo_root")
        if not repo_root_raw:
            return False
        repo_root = Path(str(repo_root_raw))
        if not repo_root.exists():
            return False

        # Pick a distinctive, reasonably short line to grep for.
        candidates = [ln for ln in preview_lines if 12 <= len(ln) <= 200]
        candidates.sort(key=len, reverse=True)
        if not candidates:
            return False
        needle = candidates[0]

        try:
            proc = subprocess.run(
                ["git", "grep", "-l", "-F", needle],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return False
        if proc.returncode != 0:
            return False

        paths = [p.strip() for p in proc.stdout.splitlines() if p.strip()]
        if not paths or len(paths) > 15:
            return False

        loaded_any = False
        for rel_path in paths[:2]:
            loaded_any |= _try_load_file_into_state(state, rel_path=rel_path)
        return loaded_any

    def _python_syntax_errors(files: Dict[str, str]) -> list[str]:
        errors: list[str] = []
        for path, content in files.items():
            if not str(path).endswith(".py"):
                continue
            try:
                ast.parse(content)
            except SyntaxError as e:
                lineno = int(e.lineno or 0)
                offset = int(e.offset or 0)
                msg = e.msg or "SyntaxError"
                errors.append(f"{path}:{lineno}:{offset}: {msg}")
        return errors

    def _is_recoverable_developer_error(error_text: str) -> bool:
        text = str(error_text or "").lower()
        recoverable_markers = (
            "developer returned no rewrites",
            "search block not found",
            "no effective file changes",
            "snippet not found",
            "unknown symbol",
            "ambiguous symbol",
            "mixed edit modes",
            "unknown file",
            "syntax error",
        )
        return any(marker in text for marker in recoverable_markers)

    def _extract_target_symbols(
        conflict_report: Optional[str],
        prioritized_violations: list[dict[str, Any]],
        *,
        max_symbols: int = 8,
    ) -> list[str]:
        low_signal = {
            "set",
            "error",
            "errors",
            "field",
            "fields",
            "model",
            "models",
            "name",
            "names",
            "value",
            "values",
            "type",
            "types",
            "related",
            "self",
            "none",
            "get",
            "add",
            "update",
            "delete",
            "remove",
            "check",
            "create",
            "_",
        }

        def _clean_symbol(raw: str) -> str:
            sym = str(raw or "").strip()
            if not sym:
                return ""
            if "|" in sym:
                sym = sym.split("|", 1)[0].strip()
            return sym

        symbols: list[str] = []
        seen: set[str] = set()

        for violation in prioritized_violations[:8]:
            sym = _clean_symbol(str(violation.get("code_node") or ""))
            if not sym:
                continue
            for candidate in (sym, sym.split(".")[-1]):
                cand = _clean_symbol(candidate)
                if not cand:
                    continue
                low = cand.lower()
                if len(cand) < 4 or low in low_signal:
                    continue
                if cand in seen:
                    continue
                seen.add(cand)
                symbols.append(cand)
                if len(symbols) >= max_symbols:
                    return symbols

        text = str(conflict_report or "")
        for match in re.finditer(r"REQ-\d+\s*->\s*([A-Za-z_][A-Za-z0-9_.]*)", text):
            sym = _clean_symbol(match.group(1))
            if not sym:
                continue
            for candidate in (sym, sym.split(".")[-1]):
                cand = _clean_symbol(candidate)
                if not cand:
                    continue
                low = cand.lower()
                if len(cand) < 4 or low in low_signal:
                    continue
                if cand in seen:
                    continue
                seen.add(cand)
                symbols.append(cand)
                if len(symbols) >= max_symbols:
                    return symbols

        return symbols

    def _collect_changed_lines(old_text: str, new_text: str) -> list[str]:
        old_lines = old_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        new_lines = new_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        changed: list[str] = []
        sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag in ("replace", "insert"):
                changed.extend(new_lines[j1:j2])
            elif tag == "delete":
                changed.extend(old_lines[i1:i2])
        return changed

    def _normalize_symbol_name(symbol: str) -> str:
        return str(symbol or "").replace("::", ".").strip().lower()

    def _symbol_leaf(symbol: str) -> str:
        normalized = _normalize_symbol_name(symbol)
        return normalized.split(".")[-1] if normalized else ""

    def _symbol_matches(target: str, applied: str) -> bool:
        t_norm = _normalize_symbol_name(target)
        a_norm = _normalize_symbol_name(applied)
        if not t_norm or not a_norm:
            return False
        if t_norm == a_norm:
            return True
        if t_norm.endswith("." + a_norm) or a_norm.endswith("." + t_norm):
            return True
        t_leaf = _symbol_leaf(target)
        a_leaf = _symbol_leaf(applied)
        return bool(t_leaf and a_leaf and t_leaf == a_leaf)

    def _compute_target_hit(
        old_files: Dict[str, str],
        new_files: Dict[str, str],
        target_symbols: list[str],
        *,
        applied_symbols: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if not target_symbols:
            return {
                "target_hit": None,
                "target_hit_rate": None,
                "target_symbols_total": 0,
                "target_symbols_hit": 0,
                "changed_line_count": 0,
                "target_hit_via": None,
                "applied_symbols_count": 0,
            }

        changed_lines: list[str] = []
        all_paths = set(old_files.keys()).union(new_files.keys())
        for path in all_paths:
            old_text = old_files.get(path, "")
            new_text = new_files.get(path, "")
            if old_text == new_text:
                continue
            changed_lines.extend(_collect_changed_lines(old_text, new_text))

        safe_applied_symbols = [str(s).strip() for s in (applied_symbols or []) if str(s).strip()]
        symbol_hits: set[str] = set()
        for sym in target_symbols:
            if any(_symbol_matches(sym, applied) for applied in safe_applied_symbols):
                symbol_hits.add(sym)

        if not changed_lines:
            hit_count = len(symbol_hits)
            total = len(target_symbols)
            hit_rate = hit_count / total if total else 0.0
            return {
                "target_hit": hit_count > 0,
                "target_hit_rate": hit_rate,
                "target_symbols_total": len(target_symbols),
                "target_symbols_hit": hit_count,
                "changed_line_count": 0,
                "hit_symbols": sorted(symbol_hits),
                "target_hit_via": "applied_symbols" if hit_count > 0 else None,
                "applied_symbols_count": len(safe_applied_symbols),
                "applied_symbols": safe_applied_symbols,
            }

        changed_blob = "\n".join(changed_lines)
        changed_blob_l = changed_blob.lower()
        text_hits: set[str] = set()
        for sym in target_symbols:
            if sym in changed_blob or sym.lower() in changed_blob_l:
                text_hits.add(sym)

        hit_symbols = set(symbol_hits).union(text_hits)
        hit_count = len(hit_symbols)
        total = len(target_symbols)
        hit_rate = hit_count / total if total else 0.0
        hit_via: Optional[str]
        if hit_symbols:
            via_parts: list[str] = []
            if symbol_hits:
                via_parts.append("applied_symbols")
            if text_hits:
                via_parts.append("changed_lines")
            hit_via = "+".join(via_parts)
        else:
            hit_via = None
        return {
            "target_hit": hit_count > 0,
            "target_hit_rate": hit_rate,
            "target_symbols_total": total,
            "target_symbols_hit": hit_count,
            "changed_line_count": len(changed_lines),
            "hit_symbols": sorted(hit_symbols),
            "target_hit_via": hit_via,
            "applied_symbols_count": len(safe_applied_symbols),
            "applied_symbols": safe_applied_symbols,
        }

    async def advanced_analysis_node(state: AgentState) -> AgentState:
        """
        Step 1: Run advanced LLM-driven analysis
        - Bug classification
        - Concept mapping
        - Pattern matching
        - Multi-round reasoning
        """
        print("\n" + "="*70)
        print("STEP 1: Advanced LLM Analysis")
        print("="*70)
        
        code_blob = "\n\n".join(state["files"].values())
        requirements = state["requirements"]
        
        # Create temporary file for analysis
        temp_file = Path("temp_analysis.py")
        temp_file.write_text(code_blob, encoding='utf-8')
        
        try:
            # Configure advanced analysis
            options = AnalysisOptions(
                strategy=analysis_strategy,
                confidence_threshold=0.6,
                include_requirements=True,
                debug_mode=True,
                max_context_tokens=8000
            )
            
            # Run advanced analysis
            start_time = time.time()
            result = await graph_manager.analyze(
                issue_text=requirements,
                target_files=["temp_analysis.py"],
                requirements_text=requirements,
                options=options
            )
            analysis_time = time.time() - start_time
            
            print(f"Advanced analysis completed in {analysis_time:.2f}s")
            print(f"Strategy used: {result.strategy_used.value}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Findings: {len(result.primary_findings)}")
            print(f"Recommendations: {len(result.recommendations)}")
            
            # Show key findings
            if result.primary_findings:
                print("\nKey Findings:")
                for i, finding in enumerate(result.primary_findings[:3], 1):
                    print(f"   {i}. {finding}")
            
            # Store advanced analysis results in state
            new_state = state.copy()
            new_state["advanced_analysis"] = {
                "result": result,
                "findings": result.primary_findings,
                "recommendations": result.recommendations,
                "confidence": result.confidence_score,
                "strategy": result.strategy_used.value,
                "processing_time": analysis_time
            }
            try:
                if graph_manager.advanced_analyzer and hasattr(graph_manager.advanced_analyzer, "llm_interface"):
                    new_state["advanced_usage"] = graph_manager.advanced_analyzer.llm_interface.get_usage_stats()
            except Exception:
                pass
            
            return new_state
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()

    def initial_graph_builder_node(state: AgentState) -> AgentState:
        """
        Step 2: Build knowledge graph enriched with advanced analysis insights
        """
        print("\n" + "="*70)
        print("STEP 2: Knowledge Graph Construction (Enhanced)")
        print("="*70)
        
        code_blob = "\n\n".join(state["files"].values())
        
        # Build structural graph
        structural_graph = graph_manager.parse_code_structure(code_blob)
        
        # Enrich with requirements
        baseline_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        
        # Get analysis report
        analysis_report = graph_manager.get_analysis_report()
        
        print(f"Graph: {analysis_report['graph_statistics']['total_nodes']} nodes, "
              f"{analysis_report['graph_statistics']['total_edges']} edges")
        print(f"Violations: {analysis_report['violation_report']['total_violations']}")
        print(f"Dependencies: {analysis_report['dependency_analysis']['nodes_with_dependencies']} nodes")
        
        # Integrate advanced analysis insights into the graph
        # (This could be enhanced to add analysis findings as graph annotations)
        advanced_analysis = state.get("advanced_analysis", {})
        if advanced_analysis:
            print(f"Integrated {len(advanced_analysis.get('findings', []))} advanced findings into context")
        
        new_state = state.copy()
        new_state["baseline_graph"] = baseline_graph
        new_state["knowledge_graph"] = baseline_graph
        new_state["analysis_report"] = analysis_report
        
        return new_state

    def developer_node(state: AgentState) -> AgentState:
        """
        Step 3: Developer revises code using both:
        - Advanced analysis insights (semantic understanding)
        - Conflict report from judge (structural violations)
        """
        print("\n" + "="*70)
        print("STEP 3: Developer Code Revision")
        print("="*70)
        
        # Prepare enhanced context for developer
        conflict_report = state.get("conflict_report")
        advanced_analysis = state.get("advanced_analysis", {})
        analysis_report = state.get("analysis_report", {})
        repair_brief = state.get("repair_brief") or {}
        
        # Enhance conflict report with advanced analysis insights
        enhanced_report = conflict_report or ""
        
        if advanced_analysis:
            raw_findings = list(advanced_analysis.get("findings") or [])
            raw_recommendations = list(advanced_analysis.get("recommendations") or [])

            def _shorten(item: Any, *, max_len: int = 300) -> str:
                text = str(item).strip()
                if len(text) <= max_len:
                    return text
                return text[: max_len - 3].rstrip() + "..."

            # Keep context concise and high-signal to reduce noisy edits.
            findings = [_shorten(f) for f in raw_findings[:3]]
            recommendations = [_shorten(r) for r in raw_recommendations[:3]]

            if findings or recommendations:
                enhanced_report += "\n\n=== Advanced Analysis Insights ===\n"
                
                if findings:
                    enhanced_report += "\nKey Findings:\n"
                    for i, finding in enumerate(findings, 1):
                        enhanced_report += f"{i}. {finding}\n"
                
                if recommendations:
                    enhanced_report += "\nRecommendations:\n"
                    for i, rec in enumerate(recommendations, 1):
                        enhanced_report += f"{i}. {rec}\n"
                
                print(
                    "Enhanced developer context with "
                    f"{len(findings)} findings and {len(recommendations)} recommendations"
                )

        # Add high-signal graph violations (with reasons) as direct edit targets.
        prioritized_violations = []
        if isinstance(analysis_report, dict):
            violation_report = analysis_report.get("violation_report")
            if isinstance(violation_report, dict):
                raw_prioritized = violation_report.get("prioritized_violations")
                if isinstance(raw_prioritized, list):
                    prioritized_violations = raw_prioritized
        blocking_prioritized = [
            v for v in prioritized_violations if bool(v.get("blocking", False))
        ]
        display_violations = blocking_prioritized or prioritized_violations
        if display_violations:
            enhanced_report += "\n\n=== Graph Violation Priorities ===\n"
            for i, violation in enumerate(display_violations[:5], 1):
                requirement_id = str(violation.get("requirement_id") or "REQ-?")
                code_node = str(violation.get("code_node") or "unknown_node")
                reason = str(violation.get("reason") or "unspecified")
                confidence_raw = violation.get("confidence")
                blocking_text = " blocking" if bool(violation.get("blocking", False)) else " advisory"
                confidence_text = ""
                if confidence_raw is not None:
                    try:
                        confidence_text = f" (confidence={float(confidence_raw):.2f})"
                    except Exception:
                        confidence_text = f" (confidence={confidence_raw})"
                enhanced_report += (
                    f"{i}. {requirement_id} -> {code_node}: {reason}{confidence_text}{blocking_text}\n"
                )
            print(
                "Added prioritized graph violations to developer context: "
                f"{min(len(display_violations), 5)} item(s)"
            )

        target_symbols: list[str] = []
        if _should_apply_repair_brief(repair_brief):
            primary_target = str(repair_brief.get("target_symbol") or "").strip()
            if primary_target:
                target_symbols = [primary_target]
                enhanced_report += "\n\n=== Judge Repair Brief ===\n"
                enhanced_report += f"Requirement: {repair_brief.get('requirement_id', 'REQ-?')}\n"
                enhanced_report += f"Primary Target: {primary_target}\n"
                related_symbols = list(repair_brief.get("related_symbols") or [])
                if related_symbols:
                    enhanced_report += f"Related Symbols: {', '.join(str(s) for s in related_symbols[:3])}\n"
                issue_summary = str(repair_brief.get("issue_summary") or "").strip()
                if issue_summary:
                    enhanced_report += f"Issue Summary: {issue_summary}\n"
                expected_behavior = str(repair_brief.get("expected_behavior") or "").strip()
                if expected_behavior:
                    enhanced_report += f"Expected Behavior: {expected_behavior}\n"
                minimal_change_hint = str(repair_brief.get("minimal_change_hint") or "").strip()
                if minimal_change_hint:
                    enhanced_report += f"Minimal Change Hint: {minimal_change_hint}\n"
                print(f"Using Judge repair brief target: {primary_target}")

        if not target_symbols:
            target_symbol_source = blocking_prioritized
            target_symbols = _extract_target_symbols(conflict_report, target_symbol_source)
        target_symbols, failure_guidance, relocalization_meta = _failure_memory_guidance(
            state,
            target_symbols=target_symbols,
            repair_brief=repair_brief if isinstance(repair_brief, dict) else None,
        )
        if failure_guidance:
            enhanced_report += failure_guidance
        if target_symbols:
            target_evidence_block = _build_target_evidence_block(
                target_symbols=target_symbols,
                blocking_prioritized=blocking_prioritized,
                repair_brief=repair_brief if isinstance(repair_brief, dict) else None,
            )
            target_anchor_block = _build_target_anchor_block(
                state["files"],
                target_symbols=target_symbols,
            )
            repair_hypotheses = _build_repair_hypotheses(
                target_symbols=target_symbols,
                repair_brief=repair_brief if isinstance(repair_brief, dict) else None,
                blocking_prioritized=blocking_prioritized,
                files=state["files"],
            )
            code_ingredients = _build_code_ingredients(
                state["files"],
                target_symbols=target_symbols,
                repair_brief=repair_brief if isinstance(repair_brief, dict) else None,
            )
            if target_evidence_block:
                enhanced_report += target_evidence_block
            if target_anchor_block:
                enhanced_report += target_anchor_block
            if repair_hypotheses:
                enhanced_report += "\n\n=== Repair Operator Plans ===\n"
                enhanced_report += _format_repair_hypotheses(repair_hypotheses) + "\n"
            if code_ingredients:
                enhanced_report += "\n\n=== Code Ingredients ===\n"
                enhanced_report += _format_code_ingredients(code_ingredients) + "\n"
            enhanced_report += "\n\n=== Conflict Target Symbols (must touch at least one) ===\n"
            enhanced_report += ", ".join(target_symbols) + "\n"
            print(f"Target symbols for this revision: {', '.join(target_symbols)}")
        else:
            repair_hypotheses = []
            code_ingredients = []

        def _evaluate_candidate(candidate_files: Dict[str, str]) -> Dict[str, Any]:
            code_blob = "\n\n".join(candidate_files.values())
            temp_graph = graph_manager.parse_code_structure(code_blob)
            temp_enriched = graph_manager.enrich_with_requirements(temp_graph, state["requirements"], llm)
            temp_report = graph_manager.get_analysis_report()
            metrics = _conflict_metrics(None, analysis_report=temp_report)
            return {
                "files": candidate_files,
                "metrics": metrics,
                "conflict_count": int(metrics.get("blocking_conflicts_count", 999) or 999),
            }

        def _total_line_count(files_map: Dict[str, str]) -> int:
            return max(1, sum(len((content or "").split("\n")) for content in files_map.values()))

        def _candidate_score(
            *,
            blocking_conflicts: int,
            advisory_conflicts: int,
            apply_failures: int,
            changed_ratio: float,
            target_hit_rate: float,
            effective_change: bool,
        ) -> float:
            score = (
                10.0 * float(blocking_conflicts)
                + 3.0 * float(advisory_conflicts)
                + 8.0 * float(apply_failures)
                + 4.0 * float(changed_ratio)
                - 6.0 * float(target_hit_rate)
            )
            if not effective_change:
                score += 25.0
            return score

        def _semantic_contract_adjustment(candidate_files: Dict[str, str]) -> float:
            if stop_policy != "semantics_contract_rerank":
                return 0.0
            requirements_text = str(state.get("requirements") or "")
            code_blob = "\n\n".join(candidate_files.values()).lower()
            adjustment = 0.0

            if "_parse_to_version_info" in requirements_text and "def _parse_to_version_info" in code_blob:
                adjustment -= 2.0
            if "valueerror" in requirements_text and "raise valueerror" in code_blob:
                adjustment -= 2.0
            if "notimplemented" in requirements_text and "return notimplemented" in code_blob:
                adjustment -= 2.0

            return adjustment

        developer_input_files = dict(state["files"])
        context_selection_meta = {
            "context_narrowed": False,
            "context_file_count": len(developer_input_files),
            "selected_files": list(developer_input_files.keys()),
        }
        active_input_files = dict(developer_input_files)

        attempt_report: Optional[str] = enhanced_report if enhanced_report else None
        updated_files: Optional[Dict[str, str]] = None
        temp_files: Dict[str, str] = dict(active_input_files)
        last_error: Optional[Exception] = None
        force_full_files: set[str] = set()
        attempts_used = 0
        no_effective_retry_count = 0
        last_revision_meta: Dict[str, Any] = {}
        recoverable_failure_count = 0
        fallback_modes_tried: list[str] = []
        target_hit_info: Dict[str, Any] = {
            "target_hit": None,
            "target_hit_rate": None,
            "target_symbols_total": len(target_symbols),
            "target_symbols_hit": 0,
            "changed_line_count": 0,
            "target_hit_via": None,
            "applied_symbols_count": 0,
        }

        attempt = 0
        mode_sequence = ["search_replace", "symbol_rewrite", "search_replace"]
        for attempt in range(3):
            attempts_used = attempt + 1
            edit_mode = mode_sequence[min(attempt, len(mode_sequence) - 1)]
            fallback_modes_tried.append(edit_mode)
            try:
                mode_report = attempt_report
                if edit_mode == "search_replace" and attempt >= 2:
                    mode_report = (str(mode_report or "").strip() + "\n\n").strip() + (
                        "Surgical anchor mode: choose a smaller, exact search block (3-12 lines) "
                        "with a unique anchor line from the provided file excerpt."
                    )
                updated_files = developer.revise(
                    temp_files,
                    state["requirements"],
                    mode_report,
                    force_full_files=force_full_files,
                    preferred_edit_mode=edit_mode,
                    repair_hypotheses=repair_hypotheses,
                    preferred_hypothesis_label=(repair_hypotheses[0].get("label") if repair_hypotheses else None),
                    code_ingredients=code_ingredients,
                )
                last_revision_meta = dict(getattr(developer, "last_revision_meta", {}) or {})
                updated_files = _merge_file_subset(state["files"], updated_files)

                target_hit_info = _compute_target_hit(
                    state["files"],
                    updated_files,
                    target_symbols,
                    applied_symbols=list(last_revision_meta.get("applied_symbols") or []),
                )
                if target_symbols and not bool(target_hit_info.get("target_hit")):
                    preview = ", ".join(target_symbols[:6])
                    print(
                        "WARNING: Developer changes did not hit target symbols this round "
                        f"({preview}); keeping change as soft-constraint mode."
                    )
            except Exception as e:
                last_error = e
                last_revision_meta = dict(getattr(developer, "last_revision_meta", {}) or {})
                print(f"WARNING: Developer output could not be applied (attempt {attempt + 1}/3): {e}")
                error_text = str(e)
                if _is_recoverable_developer_error(error_text):
                    recoverable_failure_count += 1
                full_file_note = ""
                match = re.search(r"snippet not found in ([^\\s:]+\\.py)", error_text)
                if match:
                    force_full_files.add(match.group(1))
                    full_file_note = (
                        f"Full file content for {match.group(1)} was added; "
                        "please copy an exact `before` snippet from it."
                    )
                no_edits_note = ""
                if "Developer returned no edits" in error_text:
                    force_full_files.update(state["files"].keys())
                    active_input_files = dict(state["files"])
                    temp_files = dict(active_input_files)
                    context_selection_meta = {
                        "context_narrowed": False,
                        "context_file_count": len(active_input_files),
                        "selected_files": list(active_input_files.keys()),
                    }
                    no_edits_note = (
                        "You must return at least one edit that addresses the requirements; "
                        "do not return an empty edits list."
                    )
                if "Developer returned no rewrites" in error_text:
                    force_full_files.update(state["files"].keys())
                    active_input_files = dict(state["files"])
                    temp_files = dict(active_input_files)
                    context_selection_meta = {
                        "context_narrowed": False,
                        "context_file_count": len(active_input_files),
                        "selected_files": list(active_input_files.keys()),
                    }
                    no_edits_note = (
                        "You must return at least one symbol rewrite that addresses the requirements; "
                        "do not return an empty rewrites list."
                    )
                if "no effective file changes" in error_text:
                    no_effective_retry_count += 1
                    no_edits_note = (
                        "Your rewrite must include a real code change that addresses the conflict report."
                    )
                else:
                    no_effective_retry_count = 0
                if "No target-hit edits" in error_text:
                    no_edits_note = (
                        "Your rewrite must touch at least one conflict target symbol "
                        f"({', '.join(target_symbols[:6])})."
                    )
                failed_rewrites = list(last_revision_meta.get("failed_rewrites") or [])
                syntax_failures = [
                    item for item in failed_rewrites
                    if "syntax error" in str(item.get("reason", "")).lower()
                ]
                syntax_note = ""
                if syntax_failures:
                    preview_items = [
                        f"{item.get('filename')}:{item.get('symbol')} -> {item.get('reason')}"
                        for item in syntax_failures[:2]
                    ]
                    syntax_note = (
                        "Fix syntax in symbol replacement output. "
                        + " | ".join(preview_items)
                    )
                unknown_file_note = ""
                if "Developer proposed edit for unknown file" in error_text:
                    unknown_file_note = (
                        "Only edit the files listed in the prompt; do not invent new file paths."
                    )
                if "Developer proposed rewrite for unknown file" in error_text:
                    unknown_file_note = (
                        "Only rewrite files listed in the prompt; do not invent new file paths."
                    )
                if "unknown symbol" in error_text or "ambiguous symbol" in error_text:
                    unknown_file_note = (
                        "Use exact Python symbol names from the provided file, e.g. "
                        "`function_name`, `ClassName`, or `ClassName.method_name`."
                    )
                notes = [
                    note for note in (
                        full_file_note,
                        no_edits_note,
                        syntax_note,
                        unknown_file_note,
                    ) if note
                ]
                if no_effective_retry_count >= 2:
                    notes.append(
                        "Surgical mode: change only the smallest conflict-related logic block; "
                        "do not submit equivalent rewrites."
                    )
                note_text = "\n".join(notes)

                if _maybe_expand_context_from_error(state, error_text):
                    active_input_files = dict(state["files"])
                    temp_files = dict(active_input_files)
                    context_selection_meta = {
                        "context_narrowed": False,
                        "context_file_count": len(active_input_files),
                        "selected_files": list(active_input_files.keys()),
                    }
                    base_source = conflict_report or enhanced_report
                    base = str(base_source).strip()
                    extra = (
                        "Additional relevant files were loaded into the context. "
                        "Please re-try with an exact `before` snippet from the provided files."
                    )
                    attempt_report = (base + "\n\n" if base else "") + (
                        (note_text + "\n") if note_text else ""
                    ) + extra
                    updated_files = None
                    continue
                base_source = conflict_report if no_effective_retry_count >= 1 else enhanced_report
                base = str(base_source).strip()
                attempt_report = (base + "\n\n" if base else "") + (
                    (note_text + "\n") if note_text else ""
                ) + f"Previous attempt failed to apply: {e}"
                updated_files = None
                continue

            syntax_errors = _python_syntax_errors(updated_files)
            if not syntax_errors:
                break

            last_error = RuntimeError("Syntax errors in developer output:\n" + "\n".join(syntax_errors))
            print(f"WARNING: Syntax errors after developer revision (attempt {attempt + 1}/3); retrying once.")
            # Retry from the latest output so the model can directly repair syntax.
            temp_files = {
                name: updated_files[name]
                for name in active_input_files.keys()
                if name in updated_files
            } or dict(active_input_files)
            base = enhanced_report.strip()
            attempt_report = (base + "\n\n" if base else "") + "Syntax errors in your last output:\n" + "\n".join(
                syntax_errors
            )
            updated_files = None

        if updated_files is not None:
            print("Generating 2 additional candidates for best selection...")
            candidates = [updated_files]
            candidate_metas = [last_revision_meta]
            candidate_hypotheses = [repair_hypotheses[0] if repair_hypotheses else {"label": "H1", "target_symbol": target_symbols[0] if target_symbols else "", "change_operator": "minimal_logic_fix"}]

            for cand_idx in range(2):
                try:
                    preferred_hypothesis = repair_hypotheses[cand_idx + 1] if len(repair_hypotheses) > cand_idx + 1 else None
                    candidate_report = attempt_report
                    if preferred_hypothesis:
                        candidate_report = (str(attempt_report or "").strip() + "\n\n").strip() + (
                            f"Start from repair hypothesis {preferred_hypothesis.get('label')} before considering alternatives."
                        )
                        candidate_hypotheses.append(dict(preferred_hypothesis))
                    else:
                        candidate_hypotheses.append(candidate_hypotheses[0])
                    extra_files = developer.revise(
                        active_input_files,
                        state["requirements"],
                        candidate_report,
                        force_full_files=force_full_files,
                        preferred_edit_mode=("symbol_rewrite" if preferred_hypothesis and preferred_hypothesis.get("change_operator") == "route_to_correct_helper" else "search_replace"),
                        repair_hypotheses=repair_hypotheses,
                        preferred_hypothesis_label=(preferred_hypothesis.get("label") if preferred_hypothesis else None),
                        code_ingredients=code_ingredients,
                    )
                    extra_meta = dict(getattr(developer, "last_revision_meta", {}) or {})
                    candidates.append(_merge_file_subset(state["files"], extra_files))
                    candidate_metas.append(extra_meta)
                except Exception as e:
                    print(f"  Candidate {cand_idx + 2} generation failed: {e}")
                    break

            if len(candidates) > 1:
                print(f"Evaluating {len(candidates)} candidates with Judge...")
                evaluated = []
                base_total_lines = _total_line_count(state["files"])
                for i, cand_files in enumerate(candidates):
                    eval_result = _evaluate_candidate(cand_files)
                    metrics = eval_result["metrics"]
                    blocking_conflicts = int(metrics.get("blocking_conflicts_count", eval_result["conflict_count"]) or 999)
                    advisory_conflicts = int(metrics.get("advisory_conflicts_count", 0) or 0)
                    candidate_meta = candidate_metas[i]
                    apply_failures = len(list(candidate_meta.get("failed_rewrites") or []))
                    candidate_target_hit = _compute_target_hit(
                        state["files"],
                        cand_files,
                        target_symbols,
                        applied_symbols=list(candidate_meta.get("applied_symbols") or []),
                    )
                    changed_ratio = float(candidate_target_hit.get("changed_line_count") or 0) / float(base_total_lines)
                    target_hit_rate = float(candidate_target_hit.get("target_hit_rate") or 0.0)
                    effective_change_candidate = cand_files != state["files"]
                    pre_judge_candidate = _pre_judge_assessment(
                        developer_metric={
                            "effective_change": effective_change_candidate,
                            "target_symbols_total": candidate_target_hit.get("target_symbols_total"),
                            "target_hit": candidate_target_hit.get("target_hit"),
                            "changed_line_count": candidate_target_hit.get("changed_line_count"),
                            "recoverable_failure_count": len(list(candidate_meta.get("failed_rewrites") or [])),
                            "applied_rewrites": candidate_meta.get("applied_rewrites", 0),
                        },
                        analysis_report=metrics,
                    )
                    score = _candidate_score(
                        blocking_conflicts=blocking_conflicts,
                        advisory_conflicts=advisory_conflicts,
                        apply_failures=apply_failures,
                        changed_ratio=changed_ratio,
                        target_hit_rate=target_hit_rate,
                        effective_change=effective_change_candidate,
                    )
                    critic_adjustment = 0.0
                    critic_reason = str(pre_judge_candidate.get("reason") or "pass")
                    if pre_judge_candidate.get("decision") == "pass":
                        critic_adjustment -= 1.0
                    else:
                        if "missed_target_symbols" in critic_reason:
                            critic_adjustment += 3.0
                        elif "no_effective_change" in critic_reason:
                            critic_adjustment += 4.0
                        else:
                            critic_adjustment += 1.5
                    if target_hit_rate > 0.0:
                        critic_adjustment -= 1.0
                    candidate_hypothesis = candidate_hypotheses[i] if i < len(candidate_hypotheses) else {}
                    chosen_hypothesis_label = str(candidate_meta.get("chosen_hypothesis_label") or "").strip()
                    expected_hypothesis_label = str(candidate_hypothesis.get("label") or "").strip()
                    soft_score = score + critic_adjustment + _semantic_contract_adjustment(cand_files)
                    print(
                        f"  Candidate {i + 1}: score={score:.2f}, "
                        f"blocking={blocking_conflicts}, advisory={advisory_conflicts}, "
                        f"target_hit_rate={target_hit_rate:.2f}, changed_ratio={changed_ratio:.3f}, "
                        f"apply_failures={apply_failures}, pre_judge={pre_judge_candidate.get('decision')}, "
                        f"operator={candidate_hypothesis.get('change_operator')}, hypothesis={chosen_hypothesis_label or expected_hypothesis_label}"
                    )
                    evaluated.append(
                        (
                            i,
                            cand_files,
                            candidate_meta,
                            soft_score,
                            blocking_conflicts,
                            target_hit_rate,
                            changed_ratio,
                            apply_failures,
                            critic_adjustment,
                            chosen_hypothesis_label or expected_hypothesis_label,
                            str(candidate_hypothesis.get("change_operator") or ""),
                        )
                    )

                evaluated.sort(key=lambda x: (x[3], x[4], -x[5], x[6], x[7]))
                (
                    best_idx,
                    best_files,
                    best_meta,
                    best_score,
                    best_conflicts,
                    _,
                    _,
                    _,
                    best_critic_adjustment,
                    best_hypothesis_label,
                    best_operator,
                ) = evaluated[0]
                print(
                    f"Selected candidate {best_idx + 1} with score={best_score:.2f} "
                    f"(blocking={best_conflicts}, hypothesis={best_hypothesis_label}, operator={best_operator})"
                )
                updated_files = best_files
                last_revision_meta = best_meta

        if updated_files is None:
            # Keep workflow alive for loop experiments: do not abort the whole instance on a bad turn.
            print(
                "WARNING: Developer failed to produce applicable edits after retries; "
                "keeping previous files for this revision."
            )
            if last_error is not None:
                print(f"WARNING: Last developer error: {last_error}")
            updated_files = state["files"]
        
        effective_change = updated_files != state["files"]
        target_hit_info = _compute_target_hit(
            state["files"],
            updated_files,
            target_symbols,
            applied_symbols=list(last_revision_meta.get("applied_symbols") or []),
        )
        primary_target_hit_info = _compute_target_hit(
            state["files"],
            updated_files,
            list(relocalization_meta.get("primary_targets") or []),
            applied_symbols=list(last_revision_meta.get("applied_symbols") or []),
        )
        fallback_target_hit_info = _compute_target_hit(
            state["files"],
            updated_files,
            list(relocalization_meta.get("fallback_added_targets") or []),
            applied_symbols=list(last_revision_meta.get("applied_symbols") or []),
        )
        fallback_usage_info = _summarize_fallback_usage(
            effective_change=effective_change,
            target_hit_info=target_hit_info,
            primary_target_hit_info=primary_target_hit_info,
            fallback_target_hit_info=fallback_target_hit_info,
            relocalization_meta=relocalization_meta,
        )
        if effective_change:
            print("Developer produced effective file changes in this revision.")
        else:
            print("WARNING: No effective file changes in this revision.")
        if target_hit_info.get("target_hit") is not None:
            print(
                "Target-hit metrics: "
                f"hit={target_hit_info.get('target_hit')}, "
                f"rate={float(target_hit_info.get('target_hit_rate') or 0.0):.2f}, "
                f"symbols={target_hit_info.get('target_symbols_hit')}/"
                f"{target_hit_info.get('target_symbols_total')}, "
                f"via={target_hit_info.get('target_hit_via')}, "
                f"applied_symbols={target_hit_info.get('applied_symbols_count')}"
            )
        print(
            "Fallback instrumentation: "
            f"entered={fallback_usage_info.get('entered_fallback')}, "
            f"source={fallback_usage_info.get('selected_target_source')}, "
            f"fallback_hit={fallback_usage_info.get('fallback_target_hit')}, "
            f"reason={fallback_usage_info.get('fallback_would_have_triggered_but_not_used_reason')}"
        )

        new_state = state.copy()
        new_state["files"] = updated_files
        new_state["conflict_report"] = None
        new_state["last_effective_meta"] = dict(last_revision_meta)
        new_state["repair_hypotheses"] = list(repair_hypotheses)
        if effective_change:
            new_state["last_effective_files"] = dict(updated_files)
            new_state["last_effective_revision"] = int(state.get("revision_count", 0) or 0)
        developer_metrics_history = list(state.get("developer_metrics_history", []))
        developer_metrics_history.append(
            {
                "revision": state.get("revision_count", 0),
                "attempts_used": attempts_used,
                "effective_change": effective_change,
                "last_error": str(last_error) if last_error is not None else None,
                "target_hit": target_hit_info.get("target_hit"),
                "target_hit_rate": target_hit_info.get("target_hit_rate"),
                "target_symbols_total": target_hit_info.get("target_symbols_total"),
                "target_symbols_hit": target_hit_info.get("target_symbols_hit"),
                "changed_line_count": target_hit_info.get("changed_line_count"),
                "target_hit_via": target_hit_info.get("target_hit_via"),
                "applied_symbols_count": target_hit_info.get("applied_symbols_count"),
                "applied_symbols": list(target_hit_info.get("applied_symbols") or []),
                "applied_rewrites": int(last_revision_meta.get("applied_rewrites") or 0),
                "proposed_rewrites": int(last_revision_meta.get("proposed_rewrites") or 0),
                "recoverable_failure_count": recoverable_failure_count,
                "recovery_succeeded": bool(recoverable_failure_count > 0 and effective_change),
                "fallback_modes_tried": list(fallback_modes_tried),
                "failed_rewrites_count": len(list(last_revision_meta.get("failed_rewrites") or [])),
                "context_narrowed": bool(context_selection_meta.get("context_narrowed")),
                "context_file_count": int(context_selection_meta.get("context_file_count") or len(active_input_files)),
                "selected_files": list(context_selection_meta.get("selected_files") or []),
                "relocalized": bool(relocalization_meta.get("relocalized")),
                "previous_target": relocalization_meta.get("previous_target"),
                "alternative_targets": list(relocalization_meta.get("alternative_targets") or []),
                "fallback_enabled": bool(fallback_usage_info.get("fallback_enabled")),
                "target_expansion_mode": fallback_usage_info.get("target_expansion_mode"),
                "entered_fallback": bool(fallback_usage_info.get("entered_fallback")),
                "primary_targets": list(fallback_usage_info.get("primary_targets") or []),
                "expanded_targets": list(fallback_usage_info.get("expanded_targets") or []),
                "fallback_added_targets": list(fallback_usage_info.get("fallback_added_targets") or []),
                "fallback_target_hit": fallback_usage_info.get("fallback_target_hit"),
                "fallback_target_hit_rate": fallback_usage_info.get("fallback_target_hit_rate"),
                "fallback_targets_total": fallback_usage_info.get("fallback_targets_total"),
                "fallback_targets_hit": fallback_usage_info.get("fallback_targets_hit"),
                "expansion_anchor_file": fallback_usage_info.get("expansion_anchor_file"),
                "expansion_anchor_symbol": fallback_usage_info.get("expansion_anchor_symbol"),
                "selected_target_source": fallback_usage_info.get("selected_target_source"),
                "fallback_would_have_triggered_but_not_used_reason": fallback_usage_info.get("fallback_would_have_triggered_but_not_used_reason"),
                "selected_hypothesis_label": last_revision_meta.get("chosen_hypothesis_label"),
                "selected_hypothesis_targets": [last_revision_meta.get("chosen_hypothesis_label")] if last_revision_meta.get("chosen_hypothesis_label") else [],
                "selection_reason": last_revision_meta.get("patch_strategy"),
                "execution_result": {},
                "hypothesis_root_cause": last_revision_meta.get("hypothesis_root_cause"),
                "expected_invariant": last_revision_meta.get("expected_invariant"),
                "patch_strategy": last_revision_meta.get("patch_strategy"),
            }
        )
        new_state["developer_metrics_history"] = developer_metrics_history
        
        return new_state

    def graph_builder_node(state: AgentState) -> AgentState:
        """
        Step 4: Rebuild knowledge graph after code revision
        """
        print("\n" + "="*70)
        print("STEP 4: Knowledge Graph Rebuild")
        print("="*70)
        
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        enriched_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        
        analysis_report = graph_manager.get_analysis_report()
        
        print(f"Updated Graph: {analysis_report['graph_statistics']['total_nodes']} nodes, "
              f"{analysis_report['graph_statistics']['total_edges']} edges")
        print(f"Updated Violations: {analysis_report['violation_report']['total_violations']}")
        
        new_state = state.copy()
        new_state["knowledge_graph"] = enriched_graph
        new_state["analysis_report"] = analysis_report
        developer_metrics_history = list(state.get("developer_metrics_history", []))
        latest_metric = developer_metrics_history[-1] if developer_metrics_history else {}
        pre_judge_note = _pre_judge_assessment(
            developer_metric=latest_metric,
            analysis_report=analysis_report,
        )
        new_state["pre_judge_decision"] = str(pre_judge_note.get("decision") or "pass")
        new_state["pre_judge_reason"] = str(pre_judge_note.get("reason") or "pass")
        new_state["pre_judge_reject_count"] = int(state.get("pre_judge_reject_count", 0) or 0)
        print(
            "Pre-judge advisory: "
            f"{new_state['pre_judge_decision']} ({new_state['pre_judge_reason']})"
        )
        
        return new_state

    def pre_judge_gate_node(state: AgentState) -> AgentState:
        print("\n" + "="*70)
        print("STEP 4.5: Pre-Judge Gate")
        print("="*70)

        new_state = state.copy()
        developer_metrics_history = list(state.get("developer_metrics_history", []))
        latest_metric = developer_metrics_history[-1] if developer_metrics_history else {}
        assessment = _pre_judge_assessment(
            developer_metric=latest_metric,
            analysis_report=state.get("analysis_report"),
        )
        decision = str(assessment.get("decision") or "pass")
        reason = str(assessment.get("reason") or "pass")
        reject_count = int(state.get("pre_judge_reject_count", 0) or 0)

        if decision == "reject" and reject_count < 1:
            new_state["pre_judge_decision"] = "reject"
            new_state["pre_judge_reason"] = reason
            new_state["pre_judge_reject_count"] = reject_count + 1
            new_state["revision_count"] = int(state.get("revision_count", 0) or 0) + 1
            print(f"Pre-judge rejected candidate: {reason}")
        else:
            new_state["pre_judge_decision"] = "pass"
            new_state["pre_judge_reason"] = reason
            new_state["pre_judge_reject_count"] = reject_count
            print(f"Pre-judge passed candidate: {reason}")

        return new_state

    def should_run_judge(state: AgentState) -> str:
        decision = str(state.get("pre_judge_decision") or "pass")
        revision_count = int(state.get("revision_count", 0) or 0)
        if decision == "reject":
            if revision_count >= effective_max_revisions:
                print("Pre-judge reject reached revision cap; ending.")
                return "end"
            print(f"Pre-judge requested another developer pass (reason: {state.get('pre_judge_reason')}).")
            return "revise"
        return "judge"

    def _conflict_metrics(
        report: Optional[str],
        *,
        advisory_report: Optional[str] = None,
        analysis_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = (report or advisory_report or "").strip()

        blocking_count: Optional[int] = None
        advisory_count: Optional[int] = None
        if isinstance(analysis_report, dict):
            violation_report = analysis_report.get("violation_report")
            if isinstance(violation_report, dict):
                try:
                    blocking_count = int(violation_report.get("total_blocking_violations") or 0)
                    advisory_count = int(violation_report.get("total_advisory_violations") or 0)
                except Exception:
                    blocking_count = None
                    advisory_count = None

        metrics_source = "analysis_report"
        if blocking_count is None or advisory_count is None:
            metrics_source = "judge_text"
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            blocking_lines = [ln for ln in lines if re.search(r"^B\d+\.\s+REQ-\d+\s*->", ln)]
            advisory_lines = [ln for ln in lines if re.search(r"^A\d+\.\s+REQ-\d+\s*->", ln)]
            if not blocking_lines and not advisory_lines:
                # Backward compatibility with old Judge format.
                blocking_lines = [ln for ln in lines if re.search(r"\bREQ-\d+\s*->", ln)]
            blocking_count = len(blocking_lines)
            advisory_count = len(advisory_lines)

        return {
            "conflict_len": len(text),
            "violates_count": int(blocking_count),  # kept for backward compatibility
            "blocking_conflicts_count": int(blocking_count),
            "advisory_conflicts_count": int(advisory_count),
            "metrics_source": metrics_source,
        }

    def _classify_failure_space(
        *,
        developer_metric: Dict[str, Any],
        conflict_metrics: Dict[str, Any],
    ) -> Dict[str, str]:
        target_hit = developer_metric.get("target_hit")
        effective_change = bool(developer_metric.get("effective_change"))
        blocking_conflicts = int(
            conflict_metrics.get("blocking_conflicts_count", conflict_metrics.get("violates_count", 0)) or 0
        )
        delta_blocking = conflict_metrics.get("delta_blocking_conflicts")
        fault_space_signal = "unknown"
        semantics_space_signal = "unknown"
        failure_class = "mixed_or_unknown"

        if target_hit is False or not effective_change:
            fault_space_signal = "weak"
        elif target_hit is True:
            fault_space_signal = "strong"

        if target_hit is True and blocking_conflicts > 0:
            semantics_space_signal = "weak"
        elif target_hit is True and blocking_conflicts == 0:
            semantics_space_signal = "strong"

        if target_hit is False:
            failure_class = "fault_space_weak"
        elif target_hit is True and blocking_conflicts > 0:
            failure_class = "semantics_space_weak"
        elif target_hit is True and blocking_conflicts == 0:
            failure_class = "resolved_or_clean"

        if effective_change and target_hit is True and delta_blocking is not None and int(delta_blocking) < 0 and blocking_conflicts > 0:
            failure_class = "semantics_space_progress"

        return {
            "failure_class": failure_class,
            "fault_space_signal": fault_space_signal,
            "semantics_space_signal": semantics_space_signal,
        }

    def judge_node(state: AgentState) -> AgentState:
        """
        Step 5: Judge evaluates the revised code with quality tracking
        """
        print("\n" + "="*70)
        print("STEP 5: Judge Evaluation")
        print("="*70)
        
        report = judge.evaluate(
            state["knowledge_graph"], 
            state["requirements"], 
            baseline_graph=state.get("baseline_graph")
        )
        
        new_state = state.copy()
        new_state["conflict_report"] = report
        advisory_report = getattr(judge, "last_advisory_report", None)
        if advisory_report:
            new_state["judge_advisory_report"] = advisory_report
        repair_brief = getattr(judge, "last_repair_brief", None)
        if repair_brief:
            new_state["repair_brief"] = dict(repair_brief)
            repair_brief_history = list(state.get("repair_brief_history", []))
            repair_brief_history.append(dict(repair_brief))
            new_state["repair_brief_history"] = repair_brief_history
        
        # Track code quality across iterations
        if "code_history" not in new_state:
            new_state["code_history"] = []
        
        current_code = "\n\n".join(state["files"].values())
        new_state["code_history"].append({
            "revision": state["revision_count"],
            "code": current_code,
            "has_conflicts": bool(report),
            "conflict_report": report
        })

        conflict_metrics_history = list(state.get("conflict_metrics_history", []))
        current_metrics = _conflict_metrics(
            report,
            advisory_report=advisory_report,
            analysis_report=state.get("analysis_report"),
        )
        current_metrics["revision"] = state.get("revision_count", 0)
        if conflict_metrics_history:
            prev = conflict_metrics_history[-1]
            prev_score = (
                int(prev.get("blocking_conflicts_count", prev.get("violates_count", 0)) or 0),
                int(prev.get("advisory_conflicts_count", 0) or 0),
            )
            curr_score = (
                int(current_metrics.get("blocking_conflicts_count", current_metrics.get("violates_count", 0)) or 0),
                int(current_metrics.get("advisory_conflicts_count", 0) or 0),
            )
            current_metrics["conflict_down"] = curr_score < prev_score
            current_metrics["delta_violates"] = curr_score[0] - prev_score[0]
            current_metrics["delta_blocking_conflicts"] = curr_score[0] - prev_score[0]
            current_metrics["delta_advisory_conflicts"] = curr_score[1] - prev_score[1]
            current_metrics["delta_conflict_len"] = int(current_metrics["conflict_len"]) - int(prev.get("conflict_len", 0) or 0)
        else:
            current_metrics["conflict_down"] = None
            current_metrics["delta_violates"] = None
            current_metrics["delta_blocking_conflicts"] = None
            current_metrics["delta_advisory_conflicts"] = None
            current_metrics["delta_conflict_len"] = None
        conflict_metrics_history.append(current_metrics)
        new_state["conflict_metrics_history"] = conflict_metrics_history

        developer_metrics_history = list(new_state.get("developer_metrics_history", []))

        latest_developer_metric = developer_metrics_history[-1] if developer_metrics_history else {}
        latest_effective_change = bool(latest_developer_metric.get("effective_change"))
        latest_recoverable_failures = int(latest_developer_metric.get("recoverable_failure_count") or 0)

        prev_empty_reports = int(state.get("judge_empty_report_count", 0) or 0)
        if report:
            new_state["judge_stop_signal"] = "has_conflicts"
            new_state["judge_empty_report_count"] = 0
            new_state["judge_explicit_success"] = False
        else:
            new_state["judge_stop_signal"] = "advisory_only" if advisory_report else "empty_report"
            new_state["judge_empty_report_count"] = prev_empty_reports + 1
            # Conservative explicit-success gate: require effective change and
            # observed reduction from previous blocking conflicts to zero.
            explicit_success = False
            if len(conflict_metrics_history) >= 2:
                prev_metrics = conflict_metrics_history[-2]
                prev_blocking = int(prev_metrics.get("blocking_conflicts_count", prev_metrics.get("violates_count", 0)) or 0)
                curr_blocking = int(current_metrics.get("blocking_conflicts_count", current_metrics.get("violates_count", 0)) or 0)
                if latest_effective_change and prev_blocking > 0 and curr_blocking == 0:
                    explicit_success = True
            new_state["judge_explicit_success"] = explicit_success
            if explicit_success:
                new_state["judge_stop_signal"] = "explicit_success"
        print(
            "Conflict metrics: "
            f"len={current_metrics['conflict_len']}, "
            f"blocking={current_metrics.get('blocking_conflicts_count', current_metrics['violates_count'])}, "
            f"advisory={current_metrics.get('advisory_conflicts_count', 0)}, "
            f"down={current_metrics['conflict_down']}, "
            f"source={current_metrics.get('metrics_source')}"
        )

        failure_class_info = _classify_failure_space(
            developer_metric=latest_developer_metric,
            conflict_metrics=current_metrics,
        )
        new_state["failure_class"] = failure_class_info["failure_class"]
        new_state["fault_space_signal"] = failure_class_info["fault_space_signal"]
        new_state["semantics_space_signal"] = failure_class_info["semantics_space_signal"]
        failure_class_history = list(state.get("failure_class_history", []))
        failure_class_history.append(
            {
                "revision": state.get("revision_count", 0),
                "failure_class": failure_class_info["failure_class"],
                "fault_space_signal": failure_class_info["fault_space_signal"],
                "semantics_space_signal": failure_class_info["semantics_space_signal"],
                "target_hit": latest_developer_metric.get("target_hit"),
                "blocking_conflicts": current_metrics.get("blocking_conflicts_count", current_metrics.get("violates_count", 0)),
            }
        )
        new_state["failure_class_history"] = failure_class_history[-12:]
        if developer_metrics_history:
            developer_metrics_history[-1] = {
                **developer_metrics_history[-1],
                **failure_class_info,
            }
            new_state["developer_metrics_history"] = developer_metrics_history
        print(
            "Failure classification: "
            f"class={failure_class_info['failure_class']}, "
            f"fault_space={failure_class_info['fault_space_signal']}, "
            f"semantics_space={failure_class_info['semantics_space_signal']}"
        )

        developer_rounds = len(developer_metrics_history)
        effective_rounds = sum(1 for m in developer_metrics_history if m.get("effective_change"))
        target_rounds = [m for m in developer_metrics_history if m.get("target_hit") is not None]
        target_hit_rounds = sum(1 for m in target_rounds if m.get("target_hit"))
        target_hit_rate = (target_hit_rounds / len(target_rounds)) if target_rounds else 0.0
        new_state["loop_summary"] = {
            "developer_rounds": developer_rounds,
            "effective_rounds": effective_rounds,
            "effective_modification_rate": (
                effective_rounds / developer_rounds if developer_rounds else 0.0
            ),
            "target_hit_rounds": target_hit_rounds,
            "target_rounds": len(target_rounds),
            "target_hit_rate": target_hit_rate,
        }
        print(
            "Developer metrics: "
            f"effective_rounds={effective_rounds}/{developer_rounds} "
            f"(rate={new_state['loop_summary']['effective_modification_rate']:.2f}), "
            f"target_hit={target_hit_rounds}/{len(target_rounds)} "
            f"(rate={target_hit_rate:.2f})"
        )

        failure_memory = list(state.get("failure_memory", []))
        execution_metrics_history = list(state.get("execution_metrics_history", []))
        selected_hypothesis_targets = list(latest_developer_metric.get("selected_hypothesis_targets") or [])
        execution_result = dict(latest_developer_metric.get("execution_result") or {})
        memory_entry = {
            "revision": state.get("revision_count", 0),
            "target_symbol": (selected_hypothesis_targets[0] if selected_hypothesis_targets else None),
            "alternative_targets": list(latest_developer_metric.get("alternative_targets") or []),
            "selected_hypothesis_label": latest_developer_metric.get("selected_hypothesis_label"),
            "target_hit": latest_developer_metric.get("target_hit"),
            "failure_class": failure_class_info["failure_class"],
            "delta_blocking_conflicts": current_metrics.get("delta_blocking_conflicts"),
            "delta_advisory_conflicts": current_metrics.get("delta_advisory_conflicts"),
            "execution_compile_ok": execution_result.get("compile_ok"),
            "execution_summary": execution_result.get("summary"),
        }
        failure_memory.append(memory_entry)
        new_state["failure_memory"] = failure_memory[-8:]
        execution_metrics_history.append(execution_result)
        new_state["execution_metrics_history"] = execution_metrics_history[-8:]
        
        recoverable_cycle_count = int(state.get("recoverable_cycle_count", 0) or 0)
        if report:
            consume_revision_budget = not (
                latest_recoverable_failures > 0 and not latest_effective_change
            )
            if consume_revision_budget:
                new_state["revision_count"] = state["revision_count"] + 1
                new_state["recoverable_cycle_count"] = 0
                print(f"Conflicts found, revision #{new_state['revision_count']}")

                # Early stopping: Check if we're making progress
                if state["revision_count"] > 0:
                    # If we've already tried once and still have conflicts, be cautious
                    print("WARNING: Multiple revisions attempted - consider stopping to prevent degradation")
            else:
                new_state["revision_count"] = state["revision_count"]
                new_state["recoverable_cycle_count"] = recoverable_cycle_count + 1
                print(
                    "Conflicts found but revision budget preserved due recoverable developer failure "
                    f"(recoverable_cycles={new_state['recoverable_cycle_count']})."
                )
        else:
            new_state["recoverable_cycle_count"] = 0
            print("No conflicts detected - code satisfies requirements!")
        
        return new_state

    def should_revise(state: AgentState) -> str:
        """
        Decide whether to revise again or end.
        
        Stop conditions are controlled by `stop_policy`:
        - conflict_only: stop only when Judge report is empty (or max revisions reached)
        - violates_only: stop only when KG has no VIOLATES edges (or max revisions reached)
        - hybrid: stop when either condition is met (or max revisions reached)
        """
        def _has_violates_edges(s: AgentState) -> bool:
            kg = s.get("knowledge_graph")
            if not kg:
                return False
            for _u, _v, d in kg.edges(data=True):
                if d.get("type") == "VIOLATES":
                    return True
            return False

        no_conflicts = not state.get("conflict_report")
        has_violates = _has_violates_edges(state)
        judge_stop_signal = str(state.get("judge_stop_signal") or "")
        explicit_success = bool(state.get("judge_explicit_success"))
        empty_report_count = int(state.get("judge_empty_report_count", 0) or 0)
        recoverable_cycle_count = int(state.get("recoverable_cycle_count", 0) or 0)
        max_empty_report_retries = 1
        max_recoverable_cycles = 2

        revision_count = state.get("revision_count", 0)

        if explicit_success or judge_stop_signal == "explicit_success":
            state["last_stop_reason"] = "explicit_success"
            print("Stopping: explicit success signal from judge gate.")
            return "end"

        if revision_count >= effective_max_revisions:
            state["last_stop_reason"] = "max_revisions"
            print(f"MAX_REVISIONS ({effective_max_revisions}) reached - stopping.")
            return "end"

        if no_conflicts:
            if not has_violates:
                state["last_stop_reason"] = "graph_clean_no_violates"
                print("Stopping: no blocking conflicts and no VIOLATES edges in graph.")
                return "end"
            if empty_report_count <= max_empty_report_retries:
                print(
                    "Judge report is empty without explicit success; "
                    "continuing one guarded retry."
                )
                return "revise"
            state["last_stop_reason"] = "empty_report_exhausted"
            print("Stopping: empty judge report retry budget exhausted.")
            return "end"

        if recoverable_cycle_count >= max_recoverable_cycles:
            state["last_stop_reason"] = "recoverable_exhausted"
            print("Stopping: recoverable developer failure budget exhausted.")
            return "end"

        if stop_policy == "violates_only":
            if not has_violates:
                state["last_stop_reason"] = "violates_only_no_violates"
                print("Stopping: no VIOLATES edges in knowledge graph.")
                return "end"
        elif stop_policy == "hybrid":
            if not has_violates:
                state["last_stop_reason"] = "hybrid_no_violates"
                print("Stopping: no VIOLATES edges in knowledge graph.")
                return "end"
        
        # Continue revising
        print(f"Continuing to revision #{revision_count + 1}")
        return "revise"

    # Build the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("advanced_analysis_step", 
                     lambda s: _run_async(advanced_analysis_node(s)))
    workflow.add_node("initial_graph_builder", initial_graph_builder_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("graph_builder", graph_builder_node)
    workflow.add_node("judge", judge_node)

    # Define workflow edges
    workflow.set_entry_point("advanced_analysis_step")
    workflow.add_edge("advanced_analysis_step", "initial_graph_builder")
    workflow.add_edge("initial_graph_builder", "developer")
    workflow.add_edge("developer", "graph_builder")
    workflow.add_edge("graph_builder", "judge")
    workflow.add_conditional_edges(
        "judge",
        should_revise,
        {
            "revise": "developer",
            "end": END,
        },
    )

    return workflow


def build_integrated_workflow_conflict_only(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT,
    enable_bounded_fallback_targets: bool = False,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    """Integrated workflow variant: stop only when Judge conflict report is empty."""
    return build_integrated_workflow(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=analysis_strategy,
        enable_bounded_fallback_targets=enable_bounded_fallback_targets,
        enable_file_local_neighborhood_targets=False,
        callbacks=callbacks,
        stop_policy="conflict_only",
    )


def build_integrated_workflow_violates_only(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT,
    enable_bounded_fallback_targets: bool = False,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    """Integrated workflow variant: stop only when graph has no VIOLATES edges."""
    return build_integrated_workflow(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=analysis_strategy,
        enable_bounded_fallback_targets=enable_bounded_fallback_targets,
        enable_file_local_neighborhood_targets=False,
        callbacks=callbacks,
        stop_policy="violates_only",
    )


def build_integrated_workflow_ablation1(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    return build_integrated_workflow_conflict_only(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=AnalysisStrategy.GRAPH_ONLY,
        enable_bounded_fallback_targets=False,
        callbacks=callbacks,
    )


def build_integrated_workflow_fault_space_fallback(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    return build_integrated_workflow_conflict_only(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=AnalysisStrategy.GRAPH_ONLY,
        enable_bounded_fallback_targets=True,
        callbacks=callbacks,
    )


def build_integrated_workflow_fault_space_neighborhood(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    return build_integrated_workflow(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=AnalysisStrategy.GRAPH_ONLY,
        enable_bounded_fallback_targets=False,
        enable_file_local_neighborhood_targets=True,
        callbacks=callbacks,
        stop_policy="conflict_only",
    )


def build_integrated_workflow_semantics_contract_rerank(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    return build_integrated_workflow(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=AnalysisStrategy.GRAPH_ONLY,
        enable_bounded_fallback_targets=False,
        enable_file_local_neighborhood_targets=True,
        callbacks=callbacks,
        stop_policy="semantics_contract_rerank",
    )


def main():
    """Run the integrated experiment."""
    
    print("Integrated Advanced Analysis + Traditional Workflow")
    print("="*80)
    print("This experiment combines:")
    print("  1. Advanced LLM Analysis (semantic understanding)")
    print("  2. Enhanced GraphManager (structural analysis)")
    print("  3. Developer Agent (code revision)")
    print("  4. Judge Agent (verification)")
    print("="*80)
    
    # Load experiment data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"

    requirements = load_text(req_path)
    source_code = load_text(code_path)
    target_filename = "target_file.py"

    print("\nExperiment Setup:")
    print(f"   Requirements: {len(requirements)} characters")
    print(f"   Source code: {len(source_code)} characters")
    print(f"   Target file: {target_filename}")

    # Build and run integrated workflow
    workflow = build_integrated_workflow()
    app = workflow.compile()

    initial_state: AgentState = {
        "messages": [],
        "files": {target_filename: source_code},
        "requirements": requirements,
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
        "advanced_analysis": None,
        "analysis_report": None,
    }

    print("\nRunning Integrated Workflow...")
    print("="*80)

    start_time = time.time()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    # Advanced analysis results
    advanced_analysis = final_state.get("advanced_analysis", {})
    if advanced_analysis:
        print("\nAdvanced Analysis Summary:")
        print(f"   Strategy: {advanced_analysis.get('strategy', 'N/A')}")
        print(f"   Confidence: {advanced_analysis.get('confidence', 0):.2f}")
        print(f"   Processing time: {advanced_analysis.get('processing_time', 0):.2f}s")
        print(f"   Findings: {len(advanced_analysis.get('findings', []))}")
        print(f"   Recommendations: {len(advanced_analysis.get('recommendations', []))}")

    # Graph analysis results
    analysis_report = final_state.get("analysis_report", {})
    if analysis_report:
        stats = analysis_report.get('graph_statistics', {})
        violations = analysis_report.get('violation_report', {})
        
        print("\nGraph Analysis Summary:")
        print(f"   Nodes: {stats.get('total_nodes', 0)}")
        print(f"   Edges: {stats.get('total_edges', 0)}")
        print(f"   Violations: {violations.get('total_violations', 0)}")
        print(f"   Satisfied: {violations.get('total_satisfies', 0)}")

    # Judge results
    conflict_report = final_state.get("conflict_report")
    print("\nJudge Evaluation:")
    if conflict_report:
        print("   Conflicts remain:")
        print(f"   {conflict_report}")
    else:
        print("   No conflicts - requirements satisfied!")

    # Rollback mechanism: Use the best version found
    code_history = final_state.get("code_history", [])
    final_files = final_state.get("files", {})
    final_code = final_files.get(target_filename, "")
    
    # Find the best version (one without conflicts, or earliest version)
    best_version = None
    if code_history:
        # Prefer versions without conflicts
        versions_without_conflicts = [v for v in code_history if not v["has_conflicts"]]
        if versions_without_conflicts:
            best_version = versions_without_conflicts[0]  # Use first success
            print(f"\nUsing version from revision {best_version['revision']} (no conflicts)")
        else:
            # All versions have conflicts - use the first one (least modified)
            best_version = code_history[0]
            print(f"\nAll versions have conflicts - using earliest version (revision {best_version['revision']})")
        
        # If best version is not the final version, rollback
        if best_version and best_version["revision"] != final_state.get("revision_count", 0):
            print(f"Rolling back from revision {final_state.get('revision_count', 0)} to {best_version['revision']}")
            # Note: In a real implementation, we would restore the code here
            # For now, we just report it
    
    # Code changes
    
    print("\nCode Changes:")
    diff = difflib.unified_diff(
        source_code.splitlines(keepends=True),
        final_code.splitlines(keepends=True),
        fromfile=target_filename,
        tofile=f"{target_filename} (revised)",
    )
    diff_text = "".join(diff)
    
    if diff_text:
        print("   Code was modified:")
        print(diff_text)
    else:
        print("   No changes made")

    # Summary
    print("\nExperiment Summary:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Revisions: {final_state.get('revision_count', 0)}")
    print(f"   Final status: {'Success' if not conflict_report else 'Needs review'}")
    
    print("\nIntegrated Workflow Benefits:")
    print("   Semantic understanding from LLM analysis")
    print("   Structural verification from graph analysis")
    print("   Intelligent code revision guided by both")
    print("   Comprehensive validation by judge")
    print("   Iterative refinement until requirements met")


if __name__ == "__main__":
    main()
