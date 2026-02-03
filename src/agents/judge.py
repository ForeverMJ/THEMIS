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
                    "or logically inconsistent with the code. Be EXTREMELY SPECIFIC and actionable.\n\n"
                    "CRITICAL: Your feedback quality directly impacts fix success rate.\n"
                    "- Include EXACT function/variable names\n"
                    "- Reference SPECIFIC test cases from requirements\n"
                    "- Point to the EXACT line or operation causing the issue\n"
                    "- Suggest the MINIMAL change needed (prefer 1-line fixes)",
                ),
                (
                    "user",
                    "Requirements:\n{requirements}\n\nBaseline graph edges:\n{baseline_edges}\n\n"
                    "Current graph edges:\n{edges}\n\n"
                    "Report any inconsistencies with SPECIFIC details:\n"
                    "1. EXACT function/variable name with the issue\n"
                    "2. SPECIFIC test case from requirements that fails\n"
                    "3. PRECISE line or operation to fix\n"
                    "4. MINIMAL change suggestion (prefer 1-line fixes)\n\n"
                    "If no issues found, answer 'OK'.",
                ),
            ]
        )

    def _hard_check(self, graph: nx.DiGraph) -> str | None:
        # 明示的な VIOLATES エッジがあるかチェック（第一段階）
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
        # ソフトチェック：エッジをテキスト化して LLM に最終判断を委ねる
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
        content = result.content if hasattr(result, "content") else result
        return self._coerce_text(content)

    @staticmethod
    def _coerce_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text is not None:
                        parts.append(str(text))
                    continue
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text is not None:
                    parts.append(str(text))
            if parts:
                return "\n".join(parts)
            return "\n".join(str(item) for item in value)
        if isinstance(value, dict):
            if "text" in value:
                return str(value["text"])
            if "content" in value:
                return str(value["content"])
        return str(value)

    def evaluate(
        self, graph: nx.DiGraph, requirements: str, baseline_graph: nx.DiGraph | None = None
    ) -> str | None:
        # ハードチェック→ソフトチェックの順で矛盾を判定
        hard_result = self._hard_check(graph)
        if hard_result:
            return hard_result

        soft_result = self._soft_check(graph, requirements, baseline_graph).strip()
        if soft_result.upper() == "OK":
            return None
        return soft_result
