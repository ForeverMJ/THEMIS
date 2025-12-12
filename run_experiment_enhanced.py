"""
Enhanced version of run_experiment.py that uses the Enhanced GraphManager.

This script demonstrates the Enhanced GraphManager's capabilities on the 
separability matrix bug from the experiment data.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import networkx as nx

from src.main_enhanced import build_workflow
from src.state import AgentState


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    """Run the enhanced experiment."""
    print("🚀 Enhanced GraphManager Experiment")
    print("=" * 60)
    
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"

    requirements = load_text(req_path)
    source_code = load_text(code_path)

    target_filename = "target_file.py"

    print("📋 Experiment Setup:")
    print(f"   Requirements file: {req_path}")
    print(f"   Source code file: {code_path}")
    print(f"   Source code length: {len(source_code)} characters")
    print(f"   Requirements length: {len(requirements)} characters")

    workflow = build_workflow()
    app = workflow.compile()

    initial_state: AgentState = {
        "messages": [],
        "files": {target_filename: source_code},
        "requirements": requirements,
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }

    print(f"\n🔄 Running Enhanced GraphManager Workflow...")
    print("-" * 60)

    # LangGraph アプリを実行（再帰制限を広めに設定）
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print("-" * 60)
    print("📊 Experiment Results:")

    conflict_report = final_state.get("conflict_report")
    final_files = final_state.get("files", {})
    final_code = final_files.get(target_filename, "")
    analysis_report = final_state.get("analysis_report", {})

    print(f"\n⚖️  Conflict Report:")
    if conflict_report:
        print(conflict_report)
    else:
        print("   ✅ No conflicts detected")

    print(f"\n📈 Enhanced GraphManager Analysis:")
    if analysis_report:
        stats = analysis_report.get('graph_statistics', {})
        violations = analysis_report.get('violation_report', {})
        deps = analysis_report.get('dependency_analysis', {})
        perf = analysis_report.get('performance_metrics', {})
        
        print(f"   📊 Graph Statistics:")
        print(f"      • Total nodes: {stats.get('total_nodes', 0)}")
        print(f"      • Total edges: {stats.get('total_edges', 0)}")
        print(f"      • Node types: {stats.get('node_types', {})}")
        
        print(f"   ⚠️  Violation Analysis:")
        print(f"      • Total violations: {violations.get('total_violations', 0)}")
        print(f"      • Satisfies requirements: {violations.get('total_satisfies', 0)}")
        print(f"      • Unknown status: {violations.get('total_unknown', 0)}")
        
        print(f"   🔗 Dependency Analysis:")
        print(f"      • Nodes with dependencies: {deps.get('nodes_with_dependencies', 0)}")
        print(f"      • Dependency ratio: {deps.get('dependency_ratio', 0):.2%}")
        
        print(f"   ⏱️  Performance Metrics:")
        total_time = sum(perf.values()) if perf else 0
        print(f"      • Total analysis time: {total_time:.3f}s")
        for operation, time_taken in perf.items():
            print(f"      • {operation}: {time_taken:.3f}s")
        
        # Show top violations if any
        if violations.get('prioritized_violations'):
            print(f"\n🔍 Top Violations:")
            for i, violation in enumerate(violations['prioritized_violations'][:3], 1):
                print(f"      {i}. {violation['requirement_id']} → {violation['code_node']}")
                print(f"         Status: {violation['status']}, Confidence: {violation['confidence']:.2f}")
                print(f"         Reason: {violation['reason']}")

    print(f"\n📝 Code Changes:")
    diff = difflib.unified_diff(
        source_code.splitlines(keepends=True),
        final_code.splitlines(keepends=True),
        fromfile=target_filename,
        tofile=f"{target_filename} (revised)",
    )
    diff_text = "".join(diff)
    
    if diff_text:
        print("   Code was modified:")
        print(diff_text)
    else:
        print("   ✅ No changes made to the code")

    print(f"\n🎯 Experiment Summary:")
    print(f"   • Revisions made: {final_state.get('revision_count', 0)}")
    print(f"   • Final status: {'✅ Success' if not conflict_report else '⚠️ Conflicts remain'}")
    
    if analysis_report:
        health = analysis_report.get('health_status', {})
        print(f"   • System health: {health.get('status', 'unknown')}")

    print(f"\n✨ Enhanced GraphManager provided:")
    print(f"   🔍 Precise structural analysis")
    print(f"   📝 Automated requirement mapping")
    print(f"   🔗 Comprehensive dependency tracking")
    print(f"   ⚠️  Intelligent violation detection")
    print(f"   📊 Detailed performance metrics")


if __name__ == "__main__":
    main()