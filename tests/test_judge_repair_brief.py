import networkx as nx

from src.agents.judge import JudgeAgent


def test_judge_generates_repair_brief_for_blocking_conflict():
    graph = nx.DiGraph()
    graph.add_edge(
        "REQ-001",
        "ForeignKey",
        type="VIOLATES",
        reason="ForeignKey.check still uses the wrong branch for the intermediary model case.",
        confidence=0.82,
        evidence_tags=["specific_reason", "symbol_specific"],
    )
    graph.add_edge(
        "REQ-001",
        "ManyToManyField._check_relationship_model",
        type="ADVISORY",
        reason="Related validation path may also need alignment.",
        confidence=0.61,
        evidence_tags=["specific_reason", "symbol_specific"],
    )

    judge = JudgeAgent(llm=None)
    report = judge.evaluate(graph, "ForeignKey.check must validate the relationship model correctly.")

    assert report is not None
    assert "REQ-001 -> ForeignKey" in report

    repair_brief = judge.last_repair_brief
    assert repair_brief is not None
    assert repair_brief["requirement_id"] == "REQ-001"
    assert repair_brief["target_symbol"] == "ForeignKey"
    assert repair_brief["blocking"] is True
    assert "ManyToManyField._check_relationship_model" in repair_brief["related_symbols"]
