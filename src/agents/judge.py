from __future__ import annotations

from typing import Any

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate


class JudgeAgent:
    """Analyses the knowledge graph to detect inconsistencies against requirements."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.last_advisory_report: str | None = None
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

    @staticmethod
    def _edge_meta(edge_data: dict[str, Any]) -> tuple[str | None, float | None, list[str]]:
        payload = edge_data.get("data")

        reason = edge_data.get("reason")
        confidence = edge_data.get("confidence")
        evidence_tags = edge_data.get("evidence_tags")
        if evidence_tags is None:
            evidence_tags = []

        if reason is None and payload is not None:
            reason = getattr(payload, "reason", None)
        if confidence is None and payload is not None:
            confidence = getattr(payload, "confidence", None)
        if not evidence_tags and payload is not None:
            evidence_tags = getattr(payload, "evidence_tags", []) or []

        reason_text = str(reason).strip() if reason is not None else None
        if not reason_text:
            reason_text = None

        conf_value = None
        if confidence is not None:
            try:
                conf_value = float(confidence)
            except Exception:
                conf_value = None

        tags: list[str] = []
        if isinstance(evidence_tags, list):
            tags = [str(tag).strip() for tag in evidence_tags if str(tag).strip()]
        elif evidence_tags is not None:
            text = str(evidence_tags).strip()
            if text:
                tags = [text]

        return reason_text, conf_value, tags

    @staticmethod
    def _format_line(prefix: str, idx: int, requirement: str, code_node: str, reason: str | None, confidence: float | None, tags: list[str]) -> str:
        reason_text = reason or "unspecified"
        conf_text = f"{confidence:.2f}" if confidence is not None else "n/a"
        tags_text = ", ".join(tags) if tags else "none"
        return (
            f"{prefix}{idx}. {requirement} -> {code_node} | "
            f"reason: {reason_text} | confidence: {conf_text} | evidence: {tags_text}"
        )

    def _hard_check(self, graph: nx.DiGraph) -> str | None:
        blocking: list[tuple[str, str, str | None, float | None, list[str]]] = []
        advisory: list[tuple[str, str, str | None, float | None, list[str]]] = []

        for u, v, d in graph.edges(data=True):
            edge_type = str(d.get("type") or "").upper()
            if edge_type not in {"VIOLATES", "ADVISORY"}:
                continue
            reason, confidence, tags = self._edge_meta(d)
            item = (str(u), str(v), reason, confidence, tags)
            if edge_type == "VIOLATES":
                blocking.append(item)
            else:
                advisory.append(item)

        def _sort_key(item: tuple[str, str, str | None, float | None, list[str]]) -> tuple[float, str, str]:
            conf = item[3] if item[3] is not None else -1.0
            return (-conf, item[0], item[1])

        blocking.sort(key=_sort_key)
        advisory.sort(key=_sort_key)

        advisory_parts: list[str] = []
        if advisory:
            advisory_parts.append(f"Advisory conflicts (non-blocking): {len(advisory)}")
            for idx, (u, v, reason, confidence, tags) in enumerate(advisory[:12], start=1):
                advisory_parts.append(self._format_line("A", idx, u, v, reason, confidence, tags))
            if len(advisory) > 12:
                advisory_parts.append(f"... {len(advisory) - 12} more advisory conflicts omitted")
        self.last_advisory_report = "\n".join(advisory_parts) if advisory_parts else None

        if not blocking:
            return None

        parts: list[str] = [
            f"Detected explicit graph conflicts: blocking={len(blocking)}, advisory={len(advisory)}",
            "Blocking conflicts (must fix):",
        ]
        for idx, (u, v, reason, confidence, tags) in enumerate(blocking[:20], start=1):
            parts.append(self._format_line("B", idx, u, v, reason, confidence, tags))
        if len(blocking) > 20:
            parts.append(f"... {len(blocking) - 20} more blocking conflicts omitted")
        if advisory_parts:
            parts.extend(["", *advisory_parts])
        parts.append("Completion gate: no blocking conflicts in graph.")
        return "\n".join(parts)

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
        # The graph is the primary source of truth; only blocking graph conflicts keep the loop alive.
        hard_result = self._hard_check(graph)
        if hard_result:
            return hard_result

        soft_result = self._soft_check(graph, requirements, baseline_graph).strip()
        if soft_result and soft_result.upper() != "OK":
            if self.last_advisory_report:
                self.last_advisory_report = self.last_advisory_report + "\n\nLLM advisory:\n" + soft_result
            else:
                self.last_advisory_report = "LLM advisory:\n" + soft_result
        return None

