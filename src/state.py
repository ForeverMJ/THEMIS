from typing import Dict, List, Optional, TypedDict

import networkx as nx
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    messages: List[BaseMessage]
    files: Dict[str, str]
    requirements: str
    knowledge_graph: nx.DiGraph
    baseline_graph: Optional[nx.DiGraph]
    conflict_report: Optional[str]
    revision_count: int
