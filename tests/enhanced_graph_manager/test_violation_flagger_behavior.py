import networkx as nx

from src.enhanced_graph_manager.models import ClassNode, RequirementNode
from src.enhanced_graph_manager.violation_flagger import ViolationFlagger


def _build_requirement_graph(*, requirement_text: str, code_node: str, relevance: float) -> nx.DiGraph:
    graph = nx.DiGraph()
    requirement = RequirementNode(
        id="REQ-001",
        text=requirement_text,
        priority=1,
        testable=True,
    )
    graph.add_node(requirement.id, type="requirement", data=requirement)
    graph.add_node(
        code_node,
        type="class",
        data=ClassNode(name=code_node, bases=[], methods=[], docstring=None, line_number=1),
    )
    graph.add_edge(
        requirement.id,
        code_node,
        type="MAPS_TO",
        context=f"relevance:{relevance:.2f}",
    )
    return graph


def test_generic_unknown_does_not_escalate_to_blocking_violation():
    flagger = ViolationFlagger()
    graph = _build_requirement_graph(
        requirement_text="The failing behavior is in `ForeignKey.check` for message wording.",
        code_node="ForeignKey",
        relevance=0.50,
    )

    reports = flagger.analyze_requirement_satisfaction(graph)
    report = reports[0]

    assert report.status == "UNKNOWN"
    assert report.blocking is False

    edges = flagger.flag_potential_violations(graph)
    assert len(edges) == 1
    assert edges[0].status == "ADVISORY"
    assert "Potential requirement mismatch" not in edges[0].reason


def test_specific_unknown_can_still_escalate_when_signal_is_strong():
    flagger = ViolationFlagger()

    assert flagger._should_escalate_unknown(
        reason="Exact branch comparison uses the wrong relation check in ForeignKey.check.",
        evidence_score=2.6,
        evidence_tags=[
            "specific_reason",
            "requirement_symbol_overlap",
            "mapping_relevance:0.70",
            "symbol_specific",
        ],
    ) is True
