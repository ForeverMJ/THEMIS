from __future__ import annotations

from typing import Any

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RepairBrief(BaseModel):
    requirement_id: str = ""
    target_symbol: str = ""
    related_symbols: list[str] = Field(default_factory=list)
    issue_summary: str = ""
    expected_behavior: str = ""
    minimal_change_hint: str = ""
    blocking: bool = False
    confidence: float = 0.0
    source: str = "graph"


class JudgeAgent:
    """Analyses the knowledge graph to detect inconsistencies against requirements."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.last_advisory_report: str | None = None
        self.last_repair_brief: dict[str, Any] | None = None
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
        self.coach_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are preparing a repair brief for a coding agent. "
                    "Pick the SINGLE most actionable target symbol. "
                    "Prefer an exact Python method or function over a broad class name when justified. "
                    "Keep the guidance minimal, precise, and directly editable.",
                ),
                (
                    "user",
                    "Requirements:\n{requirements}\n\n"
                    "Primary conflict:\n{primary_conflict}\n\n"
                    "Supporting conflicts:\n{supporting_conflicts}\n\n"
                    "Baseline repair brief:\n{baseline_brief}\n\n"
                    "Return a structured repair brief with:\n"
                    "- requirement_id\n"
                    "- target_symbol\n"
                    "- related_symbols\n"
                    "- issue_summary\n"
                    "- expected_behavior\n"
                    "- minimal_change_hint\n"
                    "- blocking\n"
                    "- confidence\n",
                ),
            ]
        )
        self.coach_llm = None
        if llm is not None and hasattr(llm, "with_structured_output"):
            self.coach_llm = llm.with_structured_output(RepairBrief)

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

    def _collect_conflicts(
        self, graph: nx.DiGraph
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[str],
    ]:
        blocking: list[dict[str, Any]] = []
        advisory: list[dict[str, Any]] = []

        for u, v, d in graph.edges(data=True):
            edge_type = str(d.get("type") or "").upper()
            if edge_type not in {"VIOLATES", "ADVISORY"}:
                continue
            reason, confidence, tags = self._edge_meta(d)
            item = {
                "requirement_id": str(u),
                "code_node": str(v),
                "reason": reason,
                "confidence": confidence,
                "tags": tags,
                "edge_type": edge_type,
            }
            if edge_type == "VIOLATES":
                blocking.append(item)
            else:
                advisory.append(item)

        def _sort_key(item: dict[str, Any]) -> tuple[float, str, str]:
            conf = item["confidence"] if item["confidence"] is not None else -1.0
            return (-conf, item["requirement_id"], item["code_node"])

        blocking.sort(key=_sort_key)
        advisory.sort(key=_sort_key)

        advisory_parts: list[str] = []
        if advisory:
            advisory_parts.append(f"Advisory conflicts (non-blocking): {len(advisory)}")
            for idx, item in enumerate(advisory[:12], start=1):
                advisory_parts.append(
                    self._format_line(
                        "A",
                        idx,
                        item["requirement_id"],
                        item["code_node"],
                        item["reason"],
                        item["confidence"],
                        item["tags"],
                    )
                )
            if len(advisory) > 12:
                advisory_parts.append(f"... {len(advisory) - 12} more advisory conflicts omitted")
        self.last_advisory_report = "\n".join(advisory_parts) if advisory_parts else None
        return blocking, advisory, advisory_parts

    def _build_repair_brief(
        self,
        blocking: list[dict[str, Any]],
        advisory: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        primary = blocking[0] if blocking else (advisory[0] if advisory else None)
        if primary is None:
            return None

        related_symbols: list[str] = []
        seen: set[str] = {str(primary["code_node"])}
        primary_requirement = str(primary["requirement_id"])

        for item in blocking + advisory:
            symbol = str(item["code_node"])
            if symbol in seen:
                continue
            if str(item["requirement_id"]) == primary_requirement:
                related_symbols.append(symbol)
                seen.add(symbol)
            if len(related_symbols) >= 3:
                break

        target_symbol = str(primary["code_node"])
        reason_text = str(primary.get("reason") or "unspecified")
        blocking_flag = bool(blocking) and primary in blocking
        confidence_value = primary.get("confidence")
        try:
            confidence = float(confidence_value) if confidence_value is not None else 0.0
        except Exception:
            confidence = 0.0

        if "." in target_symbol:
            minimal_hint = f"Edit the smallest logic block inside `{target_symbol}` tied to `{primary_requirement}`."
        elif related_symbols:
            minimal_hint = (
                f"Start from the smallest method related to `{target_symbol}`, "
                f"especially `{related_symbols[0]}`."
            )
        else:
            minimal_hint = f"Make the smallest logic change inside `{target_symbol}` for `{primary_requirement}`."

        brief = RepairBrief(
            requirement_id=primary_requirement,
            target_symbol=target_symbol,
            related_symbols=related_symbols,
            issue_summary=reason_text,
            expected_behavior=(
                f"Update `{target_symbol}` so `{primary_requirement}` is satisfied "
                f"without introducing unrelated edits."
            ),
            minimal_change_hint=minimal_hint,
            blocking=blocking_flag,
            confidence=confidence,
            source="graph",
        )
        return brief.model_dump()

    def _maybe_refine_repair_brief(
        self,
        *,
        requirements: str,
        blocking: list[dict[str, Any]],
        advisory: list[dict[str, Any]],
        baseline_brief: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.coach_llm is None or baseline_brief is None:
            return baseline_brief
        if not blocking:
            return baseline_brief

        primary = blocking[0] if blocking else (advisory[0] if advisory else None)
        if primary is None:
            return baseline_brief

        primary_conflict = self._format_line(
            "B" if blocking else "A",
            1,
            str(primary["requirement_id"]),
            str(primary["code_node"]),
            primary.get("reason"),
            primary.get("confidence"),
            list(primary.get("tags") or []),
        )

        support_items = (blocking[1:3] + advisory[:3])[:4]
        supporting_conflicts = []
        for idx, item in enumerate(support_items, start=1):
            supporting_conflicts.append(
                self._format_line(
                    "S",
                    idx,
                    str(item["requirement_id"]),
                    str(item["code_node"]),
                    item.get("reason"),
                    item.get("confidence"),
                    list(item.get("tags") or []),
                )
            )

        try:
            chain = self.coach_prompt | self.coach_llm
            refined: RepairBrief = chain.invoke(
                {
                    "requirements": requirements,
                    "primary_conflict": primary_conflict,
                    "supporting_conflicts": "\n".join(supporting_conflicts) or "(none)",
                    "baseline_brief": baseline_brief,
                }
            )
            payload = refined.model_dump()
            if not str(payload.get("target_symbol") or "").strip():
                return baseline_brief
            if not str(payload.get("expected_behavior") or "").strip():
                payload["expected_behavior"] = baseline_brief.get("expected_behavior", "")
            if not str(payload.get("minimal_change_hint") or "").strip():
                payload["minimal_change_hint"] = baseline_brief.get("minimal_change_hint", "")
            payload["source"] = "llm"
            return payload
        except Exception:
            return baseline_brief

    def _hard_check(self, graph: nx.DiGraph) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        blocking, advisory, advisory_parts = self._collect_conflicts(graph)

        if not blocking:
            return None, blocking, advisory

        parts: list[str] = [
            f"Detected explicit graph conflicts: blocking={len(blocking)}, advisory={len(advisory)}",
            "Blocking conflicts (must fix):",
        ]
        for idx, item in enumerate(blocking[:20], start=1):
            parts.append(
                self._format_line(
                    "B",
                    idx,
                    item["requirement_id"],
                    item["code_node"],
                    item["reason"],
                    item["confidence"],
                    item["tags"],
                )
            )
        if len(blocking) > 20:
            parts.append(f"... {len(blocking) - 20} more blocking conflicts omitted")
        if advisory_parts:
            parts.extend(["", *advisory_parts])
        parts.append("Completion gate: no blocking conflicts in graph.")
        return "\n".join(parts), blocking, advisory

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
        hard_result, blocking, advisory = self._hard_check(graph)
        baseline_brief = self._build_repair_brief(blocking, advisory)
        self.last_repair_brief = self._maybe_refine_repair_brief(
            requirements=requirements,
            blocking=blocking,
            advisory=advisory,
            baseline_brief=baseline_brief,
        )
        if hard_result:
            return hard_result

        soft_result = self._soft_check(graph, requirements, baseline_graph).strip()
        if soft_result and soft_result.upper() != "OK":
            if self.last_advisory_report:
                self.last_advisory_report = self.last_advisory_report + "\n\nLLM advisory:\n" + soft_result
            else:
                self.last_advisory_report = "LLM advisory:\n" + soft_result
        return None
