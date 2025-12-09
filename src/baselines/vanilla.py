from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.state import AgentState


def build_app(llm_model: str = "gpt-4o-mini"):
    # ループなしのシンプルな直接プロンプトベースライン
    llm = ChatOpenAI(model=llm_model, temperature=0)

    def simple_fixer(state: AgentState) -> AgentState:
        # 要件とコードを提示して一度だけ修正を生成
        code_blob = "\n\n".join(state["files"].values())
        prompt = (
            "Here is the code and the requirement. Fix the code. Return the full content.\n\n"
            f"Requirements:\n{state['requirements']}\n\n"
            f"Code:\n{code_blob}\n"
        )
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)

        updated_files = {name: content for name in state["files"]}
        new_state = state.copy()
        new_state["files"] = updated_files
        return new_state

    graph = StateGraph(AgentState)
    graph.add_node("simple_fixer", simple_fixer)
    graph.set_entry_point("simple_fixer")
    graph.add_edge("simple_fixer", END)
    return graph.compile()


app = build_app()
