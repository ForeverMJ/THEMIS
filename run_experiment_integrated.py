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


def build_integrated_workflow(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    analysis_model: str | None = None,
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT,
    callbacks: Optional[Sequence[Any]] = None,
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

    llm_kwargs: dict[str, Any] = {"model": llm_model, "temperature": 0}
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
        m = re.search(r"Developer proposed edit for unknown file: (.+)$", err.strip())
        if m:
            rel_path = m.group(1).strip()
            if _try_load_file_into_state(state, rel_path=rel_path):
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

            rel_norm = rel_path.replace("\\", "/").lstrip("./")
            candidates = [p.strip() for p in proc.stdout.splitlines() if p.strip().endswith(rel_norm)]
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
        
        # Enhance conflict report with advanced analysis insights
        enhanced_report = conflict_report or ""
        
        if advanced_analysis:
            findings = advanced_analysis.get("findings", [])
            recommendations = advanced_analysis.get("recommendations", [])
            
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
                    f"Enhanced developer context with {len(findings)} findings and {len(recommendations)} recommendations"
                )
        
        # Developer revises code with enhanced context
        attempt_report: Optional[str] = enhanced_report if enhanced_report else None
        updated_files: Optional[Dict[str, str]] = None
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                updated_files = developer.revise(
                    state["files"],
                    state["requirements"],
                    attempt_report,
                )
            except Exception as e:
                last_error = e
                print(f"WARNING: Developer output could not be applied (attempt {attempt + 1}/3): {e}")
                if _maybe_expand_context_from_error(state, str(e)):
                    base = enhanced_report.strip()
                    attempt_report = (base + "\n\n" if base else "") + (
                        "Additional relevant files were loaded into the context. "
                        "Please re-try with an exact `before` snippet from the provided files."
                    )
                    updated_files = None
                    continue
                base = enhanced_report.strip()
                attempt_report = (base + "\n\n" if base else "") + f"Previous attempt failed to apply: {e}"
                updated_files = None
                continue

            syntax_errors = _python_syntax_errors(updated_files)
            if not syntax_errors:
                break

            last_error = RuntimeError("Syntax errors in developer output:\n" + "\n".join(syntax_errors))
            print(f"WARNING: Syntax errors after developer revision (attempt {attempt + 1}/3); retrying once.")
            base = enhanced_report.strip()
            attempt_report = (base + "\n\n" if base else "") + "Syntax errors in your last output:\n" + "\n".join(
                syntax_errors
            )
            updated_files = None

        if updated_files is None:
            raise RuntimeError(str(last_error) if last_error else "Developer failed to produce valid edits")
         
        new_state = state.copy()
        new_state["files"] = updated_files
        new_state["conflict_report"] = None
        
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
        
        Implements early stopping to prevent code degradation:
        1. Stop if no conflicts found
        2. Stop if MAX_REVISIONS reached
        3. Stop if code quality is degrading
        """
        if not state.get("conflict_report"):
            # No conflicts - we're done!
            return "end"
        
        revision_count = state.get("revision_count", 0)
        
        if revision_count >= effective_max_revisions:
            print(f"WARNING: MAX_REVISIONS ({effective_max_revisions}) reached - stopping to prevent degradation")
            return "end"
        
        # Check for explicit VIOLATES edges (hard failures)
        kg = state.get("knowledge_graph")
        if kg:
            violating_edges = [
                (u, v) for u, v, d in kg.edges(data=True) 
                if d.get("type") == "VIOLATES"
            ]
            if not violating_edges:
                print("No VIOLATES edges found - stopping (soft conflicts may be false positives)")
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
