"""
Enhanced version of main.py that uses the Enhanced GraphManager instead of the original GraphManager.
"""

from __future__ import annotations

import ast
from typing import Any, Optional, Sequence

import networkx as nx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.developer import DeveloperAgent
from src.agents.judge import JudgeAgent
from src.enhanced_graph_adapter import EnhancedGraphAdapter
from src.state import AgentState
from dotenv import load_dotenv
# Ensure .env values override any previously set (possibly malformed) env vars
load_dotenv(override=True)


MAX_REVISIONS = 1  # 修正ループの最大回数


def build_workflow(
    llm_model: str = "gpt-5-mini",
    *,
    max_revisions: int | None = None,
    callbacks: Optional[Sequence[Any]] = None,
) -> StateGraph:
    """Build the enhanced workflow using Enhanced GraphManager."""
    # LangGraph のメインワークフローを組み立てる
    effective_max_revisions = MAX_REVISIONS if max_revisions is None else max_revisions
    llm_kwargs: dict[str, Any] = {"model": llm_model, "temperature": 0}
    if callbacks is not None:
        llm_kwargs["callbacks"] = list(callbacks)
    llm = ChatOpenAI(**llm_kwargs)
    graph_manager = EnhancedGraphAdapter()  # Use Enhanced GraphManager
    developer = DeveloperAgent(llm)
    judge = JudgeAgent(llm)

    def _python_syntax_errors(files: dict[str, str]) -> list[str]:
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

    def initial_graph_builder_node(state: AgentState) -> AgentState:
        """初期コードからベースライン KG を構築 (Enhanced GraphManager使用)"""
        print("Building initial knowledge graph with Enhanced GraphManager...")
        
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        baseline_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        
        # Get detailed analysis report
        analysis_report = graph_manager.get_analysis_report()
        
        print(f"   Graph: {analysis_report['graph_statistics']['total_nodes']} nodes, "
              f"{analysis_report['graph_statistics']['total_edges']} edges")
        print(f"   Violations: {analysis_report['violation_report']['total_violations']}")
        print(f"   Dependencies: {analysis_report['dependency_analysis']['nodes_with_dependencies']} nodes with deps")
        
        new_state = state.copy()
        new_state["baseline_graph"] = baseline_graph
        new_state["knowledge_graph"] = baseline_graph
        new_state["analysis_report"] = analysis_report  # Store analysis report
        return new_state

    def developer_node(state: AgentState) -> AgentState:
        """Developer が要件/矛盾レポートを見てコードを修正"""
        print("Developer analyzing and revising code...")
        
        attempt_report = state.get("conflict_report")
        updated_files: Optional[dict[str, str]] = None
        last_error: Optional[Exception] = None

        for attempt in range(2):
            try:
                updated_files = developer.revise(state["files"], state["requirements"], attempt_report)
            except Exception as e:
                last_error = e
                print(f"WARNING: Developer output could not be applied (attempt {attempt + 1}/2): {e}")
                base = (attempt_report or "").strip()
                attempt_report = (base + "\n\n" if base else "") + f"Previous attempt failed to apply: {e}"
                updated_files = None
                continue

            syntax_errors = _python_syntax_errors(updated_files)
            if not syntax_errors:
                break

            last_error = RuntimeError("Syntax errors in developer output:\n" + "\n".join(syntax_errors))
            print(f"WARNING: Syntax errors after developer revision (attempt {attempt + 1}/2); retrying once.")
            base = (attempt_report or "").strip()
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
        """修正後コードから KG を再構築 (Enhanced GraphManager使用)"""
        print("Rebuilding knowledge graph after code revision...")
        
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        enriched_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        
        # Get updated analysis report
        analysis_report = graph_manager.get_analysis_report()
        
        print(f"   Updated Graph: {analysis_report['graph_statistics']['total_nodes']} nodes, "
              f"{analysis_report['graph_statistics']['total_edges']} edges")
        print(f"   Updated Violations: {analysis_report['violation_report']['total_violations']}")
        
        new_state = state.copy()
        new_state["knowledge_graph"] = enriched_graph
        new_state["analysis_report"] = analysis_report
        return new_state

    def judge_node(state: AgentState) -> AgentState:
        """ベースラインと現在の KG を比較し矛盾を判定"""
        print("Judge evaluating code changes...")
        
        report = judge.evaluate(
            state["knowledge_graph"], state["requirements"], baseline_graph=state.get("baseline_graph")
        )
        
        new_state = state.copy()
        new_state["conflict_report"] = report
        if report:
            new_state["revision_count"] = state["revision_count"] + 1
            print(f"   Conflicts found, revision #{new_state['revision_count']}")
        else:
            print("   No conflicts detected")
        
        return new_state

    def should_revise(state: AgentState) -> str:
        """矛盾があれば再度 Developer に戻す"""
        if state.get("conflict_report") and state.get("revision_count", 0) < effective_max_revisions:
            return "revise"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("initial_graph_builder", initial_graph_builder_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("graph_builder", graph_builder_node)
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("initial_graph_builder")
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


def example_run() -> None:
    """Run an example with Enhanced GraphManager."""
    print("Enhanced GraphManager Example Run")
    print("=" * 50)
    
    app = build_workflow().compile()
    initial_state: AgentState = {
        "messages": [],
        "files": {"example.py": "def add(a, b):\n    return a + b\n"},
        "requirements": "Addition must support numeric inputs and return integers.",
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 50)
    print("Final Results:")
    print("Final files:", final_state["files"])
    print("Conflict report:", final_state["conflict_report"])
    
    if "analysis_report" in final_state:
        report = final_state["analysis_report"]
        print("\nFinal Analysis:")
        print(f"   Graph: {report['graph_statistics']['total_nodes']} nodes, {report['graph_statistics']['total_edges']} edges")
        print(f"   Violations: {report['violation_report']['total_violations']}")
        print(f"   Performance: {sum(report['performance_metrics'].values()):.3f}s total")


if __name__ == "__main__":
    example_run()
