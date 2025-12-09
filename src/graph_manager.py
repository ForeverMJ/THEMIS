from __future__ import annotations

import ast
from typing import List, Tuple

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class RequirementEdge(BaseModel):
    requirement: str = Field(..., description="Identifier of the requirement node")
    target: str = Field(..., description="Target code node name")
    relation: str = Field(..., description="Relation type, e.g., SATISFIES or VIOLATES")


class GraphAugmentation(BaseModel):
    requirement_nodes: List[str] = Field(
        default_factory=list, description="Requirement node identifiers to add"
    )
    edges: List[RequirementEdge] = Field(
        default_factory=list, description="Edges from requirements to code nodes"
    )


class GraphManager:
    """Builds and augments a knowledge graph combining code structure and requirements."""

    def parse_code_structure(self, code: str) -> nx.DiGraph:
        graph = nx.DiGraph()
        tree = ast.parse(code)

        node_names: List[str] = []
        call_edges: List[Tuple[str, str]] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.current_context: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                func_name = ".".join(self.current_context + [node.name])
                summary = ast.get_docstring(node) or ""
                graph.add_node(
                    func_name,
                    type="function",
                    name=func_name,
                    code_summary=summary.strip(),
                )
                node_names.append(func_name)
                self.current_context.append(node.name)
                self.generic_visit(node)
                self.current_context.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                class_name = ".".join(self.current_context + [node.name])
                graph.add_node(
                    class_name,
                    type="class",
                    name=class_name,
                    code_summary=ast.get_docstring(node) or "",
                )
                node_names.append(class_name)
                self.current_context.append(node.name)
                self.generic_visit(node)
                self.current_context.pop()

            def visit_Call(self, node: ast.Call) -> None:
                caller = ".".join(self.current_context) if self.current_context else None
                current_class = self.current_context[0] if self.current_context else None
                callee = self._resolve_call_name(node.func, current_class)
                if caller and callee:
                    call_edges.append((caller, callee))
                self.generic_visit(node)

            def _resolve_call_name(self, func: ast.AST, current_class: str | None) -> str | None:
                if isinstance(func, ast.Name):
                    return func.id
                if isinstance(func, ast.Attribute):
                    parts: List[str] = []
                    while isinstance(func, ast.Attribute):
                        parts.append(func.attr)
                        func = func.value
                    if isinstance(func, ast.Name):
                        parts.append(func.id)
                        resolved = ".".join(reversed(parts))
                        if func.id == "self" and current_class:
                            # Qualify self-bound calls with the class name.
                            return f"{current_class}.{resolved.split('.', 1)[1] if '.' in resolved else resolved.split('.', 1)[0]}"
                        return resolved
                return None

        Visitor().visit(tree)

        for caller, callee in call_edges:
            graph.add_edge(caller, callee, type="CALLS")

        return graph

    def enrich_with_requirements(
        self, graph: nx.DiGraph, requirements: str, llm
    ) -> nx.DiGraph:
        """Augment graph with requirement nodes and semantic edges decided by the LLM."""
        node_list = [
            {"id": n, "type": graph.nodes[n].get("type"), "summary": graph.nodes[n].get("code_summary", "")}
            for n in graph.nodes
        ]

        parser = JsonOutputParser(pydantic_object=GraphAugmentation)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are updating a knowledge graph that links requirements to code elements. "
                    "Add requirement nodes and connect them to code nodes with SATISFIES or VIOLATES edges.",
                ),
                (
                    "user",
                    "Requirements:\n{requirements}\n\nCode nodes:\n{nodes}\n\n"
                    "Return JSON matching the schema: {format_instructions}",
                ),
            ]
        )

        chain = prompt | llm | parser
        raw = chain.invoke(
            {"requirements": requirements, "nodes": node_list, "format_instructions": parser.get_format_instructions()}
        )
        augmentation: GraphAugmentation = (
            raw if isinstance(raw, GraphAugmentation) else GraphAugmentation.model_validate(raw)
        )

        for req in augmentation.requirement_nodes:
            graph.add_node(req, type="requirement", name=req, code_summary=requirements)

        for edge in augmentation.edges:
            if not graph.has_node(edge.requirement):
                graph.add_node(edge.requirement, type="requirement", name=edge.requirement)
            graph.add_edge(edge.requirement, edge.target, type=edge.relation.upper())

        return graph
