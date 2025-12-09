from __future__ import annotations

from typing import Dict, List

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.state import AgentState


class FileEdit(BaseModel):
    filename: str
    content: str


class CodebaseUpdate(BaseModel):
    files: List[FileEdit]


def build_app(llm_model: str = "gpt-4o-mini") -> StateGraph:
    llm_generator = ChatOpenAI(model=llm_model, temperature=0).with_structured_output(CodebaseUpdate)
    llm_critic = ChatOpenAI(model=llm_model, temperature=0)

    def generator(state: AgentState) -> AgentState:
        critique = state.get("conflict_report") or ""
        files_text = "\n\n".join(f"## {name}\n{content}" for name, content in state["files"].items())
        prompt = f"""You are a coder. Fix the code. If there is a critique, address it.

Requirements:
{state['requirements']}

Critique:
{critique if critique else "None"}

Current Files:
{files_text}

Return the full updated files in JSON with fields 'files': [{{'filename': str, 'content': str}}].
"""
        result: CodebaseUpdate = llm_generator.invoke(prompt)
        updated_files = {f.filename: f.content for f in result.files}
        new_state = state.copy()
        new_state["files"] = updated_files
        return new_state

    def critic(state: AgentState) -> AgentState:
        files_text = "\n\n".join(f"## {name}\n{content}" for name, content in state["files"].items())
        prompt = f"""You are a reviewer. Check if the code satisfies the requirements. If it is acceptable, reply with APPROVE.
Otherwise, provide a short critique explaining what is wrong.

Requirements:
{state['requirements']}

Code:
{files_text}
"""
        result = llm_critic.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        new_state = state.copy()
        if content.strip().upper().startswith("APPROVE"):
            new_state["conflict_report"] = None
        else:
            new_state["conflict_report"] = content.strip()
        # Track loops
        new_state["revision_count"] = state.get("revision_count", 0) + 1
        return new_state

    def should_continue(state: AgentState) -> str:
        if state.get("conflict_report") and state.get("revision_count", 0) <= 3:
            return "revise"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("generator", generator)
    graph.add_node("critic", critic)
    graph.set_entry_point("generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "revise": "generator",
            "end": END,
        },
    )
    return graph.compile()


app = build_app()
