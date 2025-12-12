"""
Enhanced version of main.py that uses the Enhanced GraphManager instead of the original GraphManager.
"""

from __future__ import annotations

import networkx as nx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.developer import DeveloperAgent
from src.agents.judge import JudgeAgent
from src.enhanced_graph_adapter import EnhancedGraphAdapter
from src.state import AgentState


MAX_REVISIONS = 3  # 修正ループの最大回数


def build_workflow(llm_model: str = "gpt-5-mini") -> StateGraph:
    """Build the enhanced workflow using Enhanced GraphManager."""
    # LangGraph のメインワークフローを組み立てる
    llm = ChatOpenAI(model=llm_model, temperature=0)
    graph_manager = EnhancedGraphAdapter()  # Use Enhanced GraphManager
    developer = DeveloperAgent(llm)
    judge = JudgeAgent(llm)

    def initial_graph_builder_node(state: AgentState) -> AgentState:
        """初期コードからベースライン KG を構築 (Enhanced GraphManager使用)"""
        print("🔍 Building initial knowledge graph with Enhanced GraphManager...")
        
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        baseline_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        
        # Get detailed analysis report
        analysis_report = graph_manager.get_analysis_report()
        
        print(f"   📊 Graph: {analysis_report['graph_statistics']['total_nodes']} nodes, "
              f"{analysis_report['graph_statistics']['total_edges']} edges")
        print(f"   ⚠️  Violations: {analysis_report['violation_report']['total_violations']}")
        print(f"   🔗 Dependencies: {analysis_report['dependency_analysis']['nodes_with_dependencies']} nodes with deps")
        
        new_state = state.copy()
        new_state["baseline_graph"] = baseline_graph
        new_state["knowledge_graph"] = baseline_graph
        new_state["analysis_report"] = analysis_report  # Store analysis report
        return new_state

    def developer_node(state: AgentState) -> AgentState:
        """Developer が要件/矛盾レポートを見てコードを修正"""
        print("👨‍💻 Developer analyzing and revising code...")
        
        updated_files = developer.revise(
            state["files"], state["requirements"], state.get("conflict_report")
        )
        new_state = state.copy()
        new_state["files"] = updated_files
        new_state["conflict_report"] = None
        return new_state

    def graph_builder_node(state: AgentState) -> AgentState:
        """修正後コードから KG を再構築 (Enhanced GraphManager使用)"""
        print("🔄 Rebuilding knowledge graph after code revision...")
        
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        enriched_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        
        # Get updated analysis report
        analysis_report = graph_manager.get_analysis_report()
        
        print(f"   📊 Updated Graph: {analysis_report['graph_statistics']['total_nodes']} nodes, "
              f"{analysis_report['graph_statistics']['total_edges']} edges")
        print(f"   ⚠️  Updated Violations: {analysis_report['violation_report']['total_violations']}")
        
        new_state = state.copy()
        new_state["knowledge_graph"] = enriched_graph
        new_state["analysis_report"] = analysis_report
        return new_state

    def judge_node(state: AgentState) -> AgentState:
        """ベースラインと現在の KG を比較し矛盾を判定"""
        print("⚖️  Judge evaluating code changes...")
        
        report = judge.evaluate(
            state["knowledge_graph"], state["requirements"], baseline_graph=state.get("baseline_graph")
        )
        
        new_state = state.copy()
        new_state["conflict_report"] = report
        if report:
            new_state["revision_count"] = state["revision_count"] + 1
            print(f"   🔍 Conflicts found, revision #{new_state['revision_count']}")
        else:
            print("   ✅ No conflicts detected")
        
        return new_state

    def should_revise(state: AgentState) -> str:
        """矛盾があれば再度 Developer に戻す"""
        if state.get("conflict_report") and state.get("revision_count", 0) < MAX_REVISIONS:
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
    print("🚀 Enhanced GraphManager Example Run")
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
    print("📋 Final Results:")
    print("Final files:", final_state["files"])
    print("Conflict report:", final_state["conflict_report"])
    
    if "analysis_report" in final_state:
        report = final_state["analysis_report"]
        print(f"\n📊 Final Analysis:")
        print(f"   Graph: {report['graph_statistics']['total_nodes']} nodes, {report['graph_statistics']['total_edges']} edges")
        print(f"   Violations: {report['violation_report']['total_violations']}")
        print(f"   Performance: {sum(report['performance_metrics'].values()):.3f}s total")


if __name__ == "__main__":
    example_run()