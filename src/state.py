from typing import Any, Dict, List, Optional, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

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
    developer_metrics_history: NotRequired[List[Dict[str, Any]]]
    conflict_metrics_history: NotRequired[List[Dict[str, Any]]]
    loop_summary: NotRequired[Dict[str, Any]]
    judge_advisory_report: NotRequired[str]
    repair_brief: NotRequired[Dict[str, Any]]
    repair_hypotheses: NotRequired[List[Dict[str, Any]]]
    repair_brief_history: NotRequired[List[Dict[str, Any]]]
    last_effective_files: NotRequired[Dict[str, str]]
    last_effective_revision: NotRequired[int]
    last_effective_meta: NotRequired[Dict[str, Any]]
    judge_stop_signal: NotRequired[str]
    judge_empty_report_count: NotRequired[int]
    judge_explicit_success: NotRequired[bool]
    recoverable_cycle_count: NotRequired[int]
    last_stop_reason: NotRequired[str]
    pre_judge_decision: NotRequired[str]
    pre_judge_reason: NotRequired[str]
    pre_judge_reject_count: NotRequired[int]
    failure_class: NotRequired[str]
    fault_space_signal: NotRequired[str]
    semantics_space_signal: NotRequired[str]
    failure_class_history: NotRequired[List[Dict[str, Any]]]
    failure_memory: NotRequired[List[Dict[str, Any]]]
    execution_metrics_history: NotRequired[List[Dict[str, Any]]]
