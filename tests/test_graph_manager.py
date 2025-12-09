import networkx as nx

from src.graph_manager import GraphManager


def test_parse_code_structure_extracts_nodes_and_calls():
    code = '''
class Service:
    def public(self):
        return self._helper()

    def _helper(self):
        return compute()


def compute():
    return 42
'''
    manager = GraphManager()
    graph = manager.parse_code_structure(code)

    expected_nodes = {
        "Service",
        "Service.public",
        "Service._helper",
        "compute",
    }
    assert set(graph.nodes) == expected_nodes

    def edge_set(g: nx.DiGraph):
        return {(u, v, d.get("type")) for u, v, d in g.edges(data=True)}

    assert edge_set(graph) == {
        ("Service.public", "Service._helper", "CALLS"),
        ("Service._helper", "compute", "CALLS"),
    }


class DummyLLM:
    """Minimal LLM stub that returns a fixed JSON augmentation."""

    def __call__(self, _input):
        return """
        {
            "requirement_nodes": ["REQ:compute-int"],
            "edges": [
                {"requirement": "REQ:compute-int", "target": "compute", "relation": "SATISFIES"}
            ]
        }
        """


def test_enrich_with_requirements_adds_requirement_nodes_and_edges():
    code = "def compute():\n    return 1\n"
    manager = GraphManager()
    graph = manager.parse_code_structure(code)

    llm = DummyLLM()
    enriched = manager.enrich_with_requirements(graph, "compute must return int", llm)

    assert "REQ:compute-int" in enriched.nodes
    assert enriched.nodes["REQ:compute-int"]["type"] == "requirement"
    assert ("REQ:compute-int", "compute") in enriched.edges
    assert enriched.edges[("REQ:compute-int", "compute")]["type"] == "SATISFIES"
