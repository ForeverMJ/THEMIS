from typing import Any, Dict, List, NotRequired, Optional, TypedDict

import networkx as nx
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    # 会話履歴（LLM へ渡すメッセージ）
    messages: List[BaseMessage]
    # ファイル名 -> コード本文
    files: Dict[str, str]
    # 現在の要件テキスト
    requirements: str
    # 最新のナレッジグラフ
    knowledge_graph: nx.DiGraph
    # ベースライン（初期コード）から構築したグラフ
    baseline_graph: Optional[nx.DiGraph]
    # Judge が検出した矛盾の内容
    conflict_report: Optional[str]
    # 修正ループ回数
    revision_count: int

    # Optional extras used by some workflows / runners.
    repo_root: NotRequired[str]
    advanced_analysis: NotRequired[Any]
    advanced_usage: NotRequired[Dict[str, Any]]
    analysis_report: NotRequired[Dict[str, Any]]
    code_history: NotRequired[List[Dict[str, Any]]]
