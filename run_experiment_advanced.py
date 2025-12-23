"""
Advanced Code Analysis Experiment Runner

This script demonstrates the Advanced Code Analysis system's capabilities
on the experiment data, comparing different analysis strategies and showing
detailed insights from the LLM-driven semantic understanding system.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

# Import the unified adapter for advanced analysis
try:
    from src.enhanced_graph_adapter import (
        EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions, UnifiedAnalysisResult
    )
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print("⚠️  Advanced analysis not available - falling back to basic analysis")

# Import traditional workflow for comparison
from src.main_enhanced import build_workflow
from src.state import AgentState


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def load_experiment_case(case_name: str = None) -> tuple[str, str, str]:
    """Load experiment data from specified case or default."""
    base = Path(__file__).parent
    
    if case_name:
        case_dir = base / "experiment_data" / case_name
        req_path = case_dir / "issue.txt"
        code_path = case_dir / "source_code.py"
        answer_path = case_dir / "Answer.txt"
    else:
        req_path = base / "experiment_data" / "issue.txt"
        code_path = base / "experiment_data" / "source_code.py"
        answer_path = base / "experiment_data" / "Answer.txt"
    
    requirements = load_text(req_path)
    source_code = load_text(code_path)
    
    # Load expected answer if available
    expected_answer = ""
    if answer_path.exists():
        expected_answer = load_text(answer_path)
    
    return requirements, source_code, expected_answer


async def run_advanced_analysis(
    issue_text: str, 
    source_code: str, 
    target_filename: str,
    strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT
) -> Optional[UnifiedAnalysisResult]:
    """Run advanced code analysis using the new system."""
    
    if not ADVANCED_ANALYSIS_AVAILABLE:
        print("❌ Advanced analysis system not available")
        return None
    
    print(f"🔍 Running Advanced Analysis (Strategy: {strategy.value})")
    print("-" * 60)
    
    try:
        # Initialize the adapter
        adapter = EnhancedGraphAdapter()
        
        # Check system status
        status = adapter.get_system_status()
        print(f"📊 System Status:")
        print(f"   Available systems: {list(status['systems_initialized'].keys())}")
        print(f"   Available strategies: {status['available_strategies']}")
        
        # Create temporary file for analysis
        temp_file = Path(target_filename)
        temp_file.write_text(source_code, encoding='utf-8')
        
        try:
            # Configure analysis options
            options = AnalysisOptions(
                strategy=strategy,
                confidence_threshold=0.6,
                include_requirements=True,
                debug_mode=True,
                max_context_tokens=8000
            )
            
            # Run the analysis
            start_time = time.time()
            result = await adapter.analyze(
                issue_text=issue_text,
                target_files=[target_filename],
                requirements_text=None,  # Issue text contains the requirements
                options=options
            )
            analysis_time = time.time() - start_time
            
            print(f"⏱️  Analysis completed in {analysis_time:.2f}s")
            return result
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
    
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_traditional_analysis(
    requirements: str, 
    source_code: str, 
    target_filename: str
) -> Dict:
    """Run traditional analysis for comparison."""
    
    print(f"🔄 Running Traditional Enhanced GraphManager Analysis")
    print("-" * 60)
    
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

    start_time = time.time()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})
    analysis_time = time.time() - start_time
    
    print(f"⏱️  Traditional analysis completed in {analysis_time:.2f}s")
    
    return {
        'final_state': final_state,
        'analysis_time': analysis_time,
        'conflict_report': final_state.get("conflict_report"),
        'final_files': final_state.get("files", {}),
        'analysis_report': final_state.get("analysis_report", {})
    }


def print_advanced_analysis_result(result: UnifiedAnalysisResult):
    """Print detailed advanced analysis results."""
    
    print(f"📋 Advanced Analysis Results:")
    print(f"   Strategy Used: {result.strategy_used.value}")
    print(f"   Success: {'✅' if result.success else '❌'}")
    print(f"   Processing Time: {result.processing_time:.2f}s")
    print(f"   Confidence Score: {result.confidence_score:.2f}")
    
    if result.error_message:
        print(f"   ❌ Error: {result.error_message}")
        return
    
    print(f"\n🔍 Primary Findings ({len(result.primary_findings)}):")
    for i, finding in enumerate(result.primary_findings, 1):
        print(f"   {i}. {finding}")
    
    print(f"\n💡 Recommendations ({len(result.recommendations)}):")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Show advanced analysis details
    if result.has_advanced_analysis():
        print(f"\n🧠 Advanced LLM Analysis Available:")
        if hasattr(result, 'bug_classification'):
            print(f"   Bug Type: {getattr(result, 'bug_classification', 'Unknown')}")
        if hasattr(result, 'reasoning_chain'):
            chain = getattr(result, 'reasoning_chain', [])
            if chain:
                print(f"   Reasoning Steps: {len(chain)}")
                for i, step in enumerate(chain[:3], 1):  # Show first 3 steps
                    print(f"      {i}. {step}")
    
    # Show graph analysis details
    if result.has_graph_analysis():
        print(f"\n📊 Graph Analysis Available:")
        if result.graph_statistics:
            stats = result.graph_statistics
            print(f"   Graph: {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
            print(f"   Node types: {stats.get('node_types', {})}")
        
        if result.violation_report:
            violations = result.violation_report.get('total_violations', 0)
            satisfies = result.violation_report.get('total_satisfies', 0)
            print(f"   Requirements: {satisfies} satisfied, {violations} violations")
            
            # Show top violations
            if result.violation_report.get('prioritized_violations'):
                print(f"   Top Violations:")
                for violation in result.violation_report['prioritized_violations'][:3]:
                    print(f"      • {violation['requirement_id']} → {violation['code_node']}")
                    print(f"        Status: {violation['status']}, Confidence: {violation['confidence']:.2f}")


def print_traditional_analysis_result(result: Dict):
    """Print traditional analysis results."""
    
    print(f"📋 Traditional Analysis Results:")
    print(f"   Processing Time: {result['analysis_time']:.2f}s")
    
    conflict_report = result['conflict_report']
    if conflict_report:
        print(f"   ⚠️  Conflicts: {conflict_report}")
    else:
        print(f"   ✅ No conflicts detected")
    
    analysis_report = result['analysis_report']
    if analysis_report:
        stats = analysis_report.get('graph_statistics', {})
        violations = analysis_report.get('violation_report', {})
        
        print(f"\n📊 Graph Statistics:")
        print(f"   Total nodes: {stats.get('total_nodes', 0)}")
        print(f"   Total edges: {stats.get('total_edges', 0)}")
        print(f"   Node types: {stats.get('node_types', {})}")
        
        print(f"\n⚠️  Violation Analysis:")
        print(f"   Total violations: {violations.get('total_violations', 0)}")
        print(f"   Satisfies requirements: {violations.get('total_satisfies', 0)}")


def compare_results(
    advanced_result: Optional[UnifiedAnalysisResult],
    traditional_result: Dict,
    original_code: str,
    target_filename: str
):
    """Compare results from both analysis methods."""
    
    print(f"\n🔄 Analysis Comparison:")
    print("=" * 60)
    
    # Time comparison
    advanced_time = advanced_result.processing_time if advanced_result else 0
    traditional_time = traditional_result['analysis_time']
    
    print(f"⏱️  Processing Time:")
    print(f"   Advanced Analysis: {advanced_time:.2f}s")
    print(f"   Traditional Analysis: {traditional_time:.2f}s")
    print(f"   Speed difference: {abs(advanced_time - traditional_time):.2f}s")
    
    # Confidence comparison
    if advanced_result and advanced_result.success:
        print(f"\n🎯 Confidence Scores:")
        print(f"   Advanced Analysis: {advanced_result.confidence_score:.2f}")
        print(f"   Traditional Analysis: N/A (rule-based)")
    
    # Findings comparison
    print(f"\n🔍 Findings Comparison:")
    if advanced_result and advanced_result.success:
        print(f"   Advanced Findings: {len(advanced_result.primary_findings)}")
        print(f"   Advanced Recommendations: {len(advanced_result.recommendations)}")
    else:
        print(f"   Advanced Analysis: Failed or unavailable")
    
    traditional_conflicts = 1 if traditional_result['conflict_report'] else 0
    print(f"   Traditional Conflicts: {traditional_conflicts}")
    
    # Code changes comparison
    print(f"\n📝 Code Changes:")
    traditional_final_code = traditional_result['final_files'].get(target_filename, original_code)
    
    traditional_changed = traditional_final_code != original_code
    print(f"   Traditional Analysis: {'Modified code' if traditional_changed else 'No changes'}")
    print(f"   Advanced Analysis: {'Provides recommendations' if advanced_result and advanced_result.success else 'No output'}")
    
    if traditional_changed:
        print(f"\n📄 Traditional Analysis Code Diff:")
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            traditional_final_code.splitlines(keepends=True),
            fromfile=target_filename,
            tofile=f"{target_filename} (traditional)",
        )
        diff_text = "".join(diff)
        print(diff_text or "(no changes)")


async def run_experiment_case(case_name: str = None):
    """Run experiment on a specific case."""
    
    print(f"🧪 Running Experiment Case: {case_name or 'default'}")
    print("=" * 80)
    
    # Load experiment data
    try:
        requirements, source_code, expected_answer = load_experiment_case(case_name)
    except FileNotFoundError as e:
        print(f"❌ Failed to load experiment case: {e}")
        return
    
    target_filename = "target_file.py"
    
    print(f"📋 Experiment Setup:")
    print(f"   Case: {case_name or 'default'}")
    print(f"   Source code length: {len(source_code)} characters")
    print(f"   Requirements length: {len(requirements)} characters")
    print(f"   Expected answer available: {'✅' if expected_answer else '❌'}")
    
    if expected_answer:
        print(f"   Expected answer preview: {expected_answer[:200]}...")
    
    print(f"\n📝 Issue Description:")
    print(f"   {requirements[:300]}...")
    
    # Run both analyses
    print(f"\n" + "=" * 80)
    
    # Advanced analysis using the adapter's own strategy selection
    advanced_result = None
    if ADVANCED_ANALYSIS_AVAILABLE:
        print(f"\n{'='*20} AUTO-SELECT ANALYSIS {'='*20}")
        advanced_result = await run_advanced_analysis(
            requirements,
            source_code,
            target_filename,
            AnalysisStrategy.AUTO_SELECT
        )
        if advanced_result:
            print_advanced_analysis_result(advanced_result)
    
    # Traditional analysis
    print(f"\n{'='*20} TRADITIONAL ANALYSIS {'='*20}")
    traditional_result = run_traditional_analysis(requirements, source_code, target_filename)
    print_traditional_analysis_result(traditional_result)
    
    # Compare results
    if advanced_result:
        compare_results(advanced_result, traditional_result, source_code, target_filename)
    
    # Summary
    print(f"\n🎯 Experiment Summary:")
    print("=" * 80)
    
    if advanced_result:
        print(f"   Advanced Analysis:")
        print(f"      Strategy used: {advanced_result.strategy_used.value}")
        print(f"      Success: {'Yes' if advanced_result.success else 'No'}")
        print(f"      Confidence: {advanced_result.confidence_score:.2f}")
        print(f"      Processing time: {advanced_result.processing_time:.2f}s")
    
    print(f"   Traditional Analysis:")
    print(f"      Processing time: {traditional_result['analysis_time']:.2f}s")
    print(f"      Conflicts detected: {'Yes' if traditional_result['conflict_report'] else 'No'}")
    print(f"      Code modified: {'Yes' if traditional_result['final_files'].get(target_filename) != source_code else 'No'}")
    
    if expected_answer:
        print(f"\n📊 Expected Answer Comparison:")
        print(f"   Expected: {expected_answer[:100]}...")
        print(f"   Note: Manual comparison needed for accuracy assessment")


async def main():
    """Main experiment runner."""
    
    print("🚀 Advanced Code Analysis Experiment Runner")
    print("=" * 80)
    
    # Check available experiment cases
    base = Path(__file__).parent / "experiment_data"
    available_cases = []
    
    for case_dir in base.iterdir():
        if case_dir.is_dir() and (case_dir / "issue.txt").exists():
            available_cases.append(case_dir.name)
    
    print(f"📁 Available experiment cases: {available_cases}")
    
    # Run default case
    await run_experiment_case()
    
    print(f"\n✨ All experiments completed!")
    print(f"💡 The Advanced Code Analysis system provides:")
    print(f"   🧠 LLM-driven semantic understanding")
    print(f"   🔍 Intelligent bug classification")
    print(f"   📊 Multi-round reasoning and verification")
    print(f"   🎯 Context-aware analysis strategies")
    print(f"   📈 Confidence scoring and evidence chains")


if __name__ == "__main__":
    asyncio.run(main())
