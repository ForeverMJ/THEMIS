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


def build_integrated_workflow(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT,
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
        if target_symbols:
            enhanced_report += "\n\n=== Conflict Target Symbols (must touch at least one) ===\n"
            enhanced_report += ", ".join(target_symbols) + "\n"
            print(f"Target symbols for this revision: {', '.join(target_symbols)}")
        
        # Developer revises code with enhanced context
        attempt_report: Optional[str] = enhanced_report if enhanced_report else None
        updated_files: Optional[Dict[str, str]] = None
        temp_files: Dict[str, str] = state["files"]
        last_error: Optional[Exception] = None
        force_full_files: set[str] = set()
        attempts_used = 0
        no_effective_retry_count = 0
        last_revision_meta: Dict[str, Any] = {}
        target_hit_info: Dict[str, Any] = {
            "target_hit": None,
            "target_hit_rate": None,
            "target_symbols_total": len(target_symbols),
            "target_symbols_hit": 0,
            "changed_line_count": 0,
            "target_hit_via": None,
            "applied_symbols_count": 0,
        }

        for attempt in range(3):
            attempts_used = attempt + 1
            try:
                updated_files = developer.revise(
                    temp_files,
                    state["requirements"],
                    attempt_report,
                    force_full_files=force_full_files,
                )
                last_revision_meta = dict(getattr(developer, "last_revision_meta", {}) or {})

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
                    no_edits_note = (
                        "You must return at least one edit that addresses the requirements; "
                        "do not return an empty edits list."
                    )
                if "Developer returned no rewrites" in error_text:
                    force_full_files.update(state["files"].keys())
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
            temp_files = updated_files
            base = enhanced_report.strip()
            attempt_report = (base + "\n\n" if base else "") + "Syntax errors in your last output:\n" + "\n".join(
                syntax_errors
            )
            updated_files = None

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

        new_state = state.copy()
        new_state["files"] = updated_files
        new_state["conflict_report"] = None
        new_state["last_effective_meta"] = dict(last_revision_meta)
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
        
        return new_state

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
                    blocking_count = int(violation_report.get("total_blocking_violations"))
                    advisory_count = int(violation_report.get("total_advisory_violations"))
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
        print(
            "Conflict metrics: "
            f"len={current_metrics['conflict_len']}, "
            f"blocking={current_metrics.get('blocking_conflicts_count', current_metrics['violates_count'])}, "
            f"advisory={current_metrics.get('advisory_conflicts_count', 0)}, "
            f"down={current_metrics['conflict_down']}, "
            f"source={current_metrics.get('metrics_source')}"
        )

        developer_metrics_history = list(new_state.get("developer_metrics_history", []))
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
        
        if report:
            new_state["revision_count"] = state["revision_count"] + 1
            print(f"Conflicts found, revision #{new_state['revision_count']}")
            
            # Early stopping: Check if we're making progress
            if state["revision_count"] > 0:
                # If we've already tried once and still have conflicts, be cautious
                print("WARNING: Multiple revisions attempted - consider stopping to prevent degradation")
        else:
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

        revision_count = state.get("revision_count", 0)
        
        if revision_count >= effective_max_revisions:
            print(f"MAX_REVISIONS ({effective_max_revisions}) reached - stopping.")
            return "end"

        if stop_policy == "conflict_only":
            if no_conflicts:
                print("Stopping: Judge conflict report is empty.")
                return "end"
        elif stop_policy == "violates_only":
            if not has_violates:
                print("Stopping: no VIOLATES edges in knowledge graph.")
                return "end"
        else:
            # hybrid policy (backward-compatible default)
            if no_conflicts:
                print("Stopping: Judge conflict report is empty.")
                return "end"
            if not has_violates:
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
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    """Integrated workflow variant: stop only when Judge conflict report is empty."""
    return build_integrated_workflow(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=analysis_strategy,
        callbacks=callbacks,
        stop_policy="conflict_only",
    )


def build_integrated_workflow_violates_only(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    """Integrated workflow variant: stop only when graph has no VIOLATES edges."""
    return build_integrated_workflow(
        llm_model=llm_model,
        max_revisions=max_revisions,
        analysis_model=analysis_model,
        analysis_strategy=analysis_strategy,
        callbacks=callbacks,
        stop_policy="violates_only",
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
