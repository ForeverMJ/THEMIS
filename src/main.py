from __future__ import annotations

import networkx as nx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.developer import DeveloperAgent
from src.agents.judge import JudgeAgent
from src.graph_manager import GraphManager
from src.state import AgentState
from dotenv import load_dotenv
load_dotenv()


MAX_REVISIONS = 1  # 修正ループの最大回数


def build_workflow(llm_model: str = "gpt-5-mini") -> StateGraph:
    # LangGraph のメインワークフローを組み立てる
    llm = ChatOpenAI(model=llm_model, temperature=0)
    graph_manager = GraphManager()
    developer = DeveloperAgent(llm)
    judge = JudgeAgent(llm)

    def initial_graph_builder_node(state: AgentState) -> AgentState:
        # 初期コードからベースライン KG を構築
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        baseline_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        new_state = state.copy()
        new_state["baseline_graph"] = baseline_graph
        # Keep initial knowledge_graph as baseline for reference if needed.
        new_state["knowledge_graph"] = baseline_graph
        return new_state

    def developer_node(state: AgentState) -> AgentState:
        # Developer が要件/矛盾レポートを見てコードを修正
        updated_files = developer.revise(
            state["files"], state["requirements"], state.get("conflict_report")
        )
        new_state = state.copy()
        new_state["files"] = updated_files
        new_state["conflict_report"] = None
        return new_state

    def graph_builder_node(state: AgentState) -> AgentState:
        # 修正後コードから KG を再構築
        code_blob = "\n\n".join(state["files"].values())
        structural_graph = graph_manager.parse_code_structure(code_blob)
        enriched_graph = graph_manager.enrich_with_requirements(
            structural_graph, state["requirements"], llm
        )
        new_state = state.copy()
        new_state["knowledge_graph"] = enriched_graph
        return new_state

    def judge_node(state: AgentState) -> AgentState:
        # ベースラインと現在の KG を比較し矛盾を判定
        report = judge.evaluate(
            state["knowledge_graph"], state["requirements"], baseline_graph=state.get("baseline_graph")
        )
        new_state = state.copy()
        new_state["conflict_report"] = report
        if report:
            new_state["revision_count"] = state["revision_count"] + 1
        return new_state

    def should_revise(state: AgentState) -> str:
        # 矛盾があれば再度 Developer に戻す
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
    app = build_workflow()
    initial_state: AgentState = {
        "messages": [],
        "files": {"example.py": "def add(a, b):\n    return a + b\n"},
        "requirements": "Addition must support numeric inputs and return integers.",
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }
    final_state = app.compile().invoke(initial_state)
    print("Final files:", final_state["files"])
    print("Conflict report:", final_state["conflict_report"])


if __name__ == "__main__":
    example_run()
