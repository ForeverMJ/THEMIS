from __future__ import annotations

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate


class JudgeAgent:
    """Analyses the knowledge graph to detect inconsistencies against requirements."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a code auditor. Review the edges and determine if any requirement is violated "
                    "or logically inconsistent with the code. Be concise and actionable.",
                ),
                (
                    "user",
                    "Requirements:\n{requirements}\n\nBaseline graph edges:\n{baseline_edges}\n\n"
                    "Current graph edges:\n{edges}\n\n"
                    "Report any inconsistencies and propose fixes. If none, answer 'OK'.",
                ),
            ]
        )

    def _hard_check(self, graph: nx.DiGraph) -> str | None:
        violating = [
            (u, v, d) for u, v, d in graph.edges(data=True) if d.get("type") == "VIOLATES"
        ]
        if violating:
            parts = ["Detected explicit VIOLATES edges:"]
            parts.extend(f"{u} -> {v}" for u, v, _ in violating)
            return "\n".join(parts)
        return None

    def _soft_check(
        self, graph: nx.DiGraph, requirements: str, baseline_graph: nx.DiGraph | None
    ) -> str:
        edge_descriptions = [
            f"{u} -[{d.get('type')}]-> {v}" for u, v, d in graph.edges(data=True)
        ]
        baseline_descriptions = (
            [
                f"{u} -[{d.get('type')}]-> {v}"
                for u, v, d in baseline_graph.edges(data=True)
            ]
            if baseline_graph is not None
            else []
        )
        chain = self.prompt | self.llm
        result = chain.invoke(
            {
                "requirements": requirements,
                "edges": "\n".join(edge_descriptions),
                "baseline_edges": "\n".join(baseline_descriptions),
            }
        )
        return result.content if hasattr(result, "content") else str(result)

    def evaluate(
        self, graph: nx.DiGraph, requirements: str, baseline_graph: nx.DiGraph | None = None
    ) -> str | None:
        hard_result = self._hard_check(graph)
        if hard_result:
            return hard_result

        soft_result = self._soft_check(graph, requirements, baseline_graph).strip()
        if soft_result.upper() == "OK":
            return None
        return soft_result
