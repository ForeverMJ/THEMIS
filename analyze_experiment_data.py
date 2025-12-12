#!/usr/bin/env python3
"""
Analyze experiment data using Enhanced GraphManager without LLM dependencies.

This script demonstrates the Enhanced GraphManager's analysis capabilities
on the separability matrix bug from the experiment data.
"""

from pathlib import Path
from src.enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
from src.enhanced_graph_manager.logger import set_log_level


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def main():
    """Analyze the experiment data using Enhanced GraphManager."""
    print("🚀 Enhanced GraphManager Analysis of Experiment Data")
    print("=" * 60)
    
    # Set log level for cleaner output
    set_log_level("WARNING")
    
    # Load experiment data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"

    requirements = load_text(req_path)
    source_code = load_text(code_path)

    print("📋 Experiment Data:")
    print(f"   Requirements file: {req_path}")
    print(f"   Source code file: {code_path}")
    print(f"   Source code: {len(source_code)} characters, {len(source_code.splitlines())} lines")
    print(f"   Requirements: {len(requirements)} characters")
    
    # Show a snippet of the issue
    print(f"\n📝 Issue Summary:")
    issue_lines = requirements.split('\n')[:3]
    for line in issue_lines:
        if line.strip():
            print(f"   {line.strip()}")
    print("   ...")

    # Create Enhanced GraphManager
    manager = EnhancedGraphManager()
    
    print(f"\n🔍 Running Enhanced GraphManager Analysis...")
    print("-" * 60)
    
    # Run complete analysis workflow
    results = manager.analyze_complete_workflow(source_code, requirements)
    
    if results['success']:
        print(f"✅ Analysis completed successfully in {results['execution_time']:.3f} seconds")
        
        # Display comprehensive results
        stats = results['graph_statistics']
        deps = results['dependency_analysis']
        violations = results['violation_report']
        metrics = results['performance_metrics']
        
        print(f"\n📊 Graph Statistics:")
        print(f"   • Total nodes: {stats['total_nodes']}")
        print(f"   • Total edges: {stats['total_edges']}")
        print(f"   • Graph density: {stats['density']:.3f}")
        print(f"   • Node types: {dict(stats['node_types'])}")
        print(f"   • Edge types: {dict(stats['edge_types'])}")
        
        print(f"\n🔗 Dependency Analysis:")
        print(f"   • Nodes with dependencies: {deps['nodes_with_dependencies']}/{deps['total_nodes']}")
        print(f"   • Dependency ratio: {deps['dependency_ratio']:.2%}")
        
        if deps['most_dependent_nodes']:
            print(f"   • Most dependent nodes:")
            for i, node_info in enumerate(deps['most_dependent_nodes'][:5], 1):
                print(f"     {i}. {node_info['node']} ({node_info['dependency_count']} dependencies)")
        
        print(f"\n⚠️  Violation Analysis:")
        print(f"   • Total reports: {violations['total_reports']}")
        print(f"   • Violations: {violations['total_violations']}")
        print(f"   • Satisfies: {violations['total_satisfies']}")
        print(f"   • Unknown: {violations['total_unknown']}")
        
        if violations['prioritized_violations']:
            print(f"\n🔍 Top Violations (by priority):")
            for i, violation in enumerate(violations['prioritized_violations'][:5], 1):
                print(f"   {i}. {violation['requirement_id']} → {violation['code_node']}")
                print(f"      Status: {violation['status']}")
                print(f"      Severity: {violation['severity']}, Confidence: {violation['confidence']:.2f}")
                print(f"      Reason: {violation['reason']}")
                print()
        
        print(f"⏱️  Performance Breakdown:")
        for operation, time_taken in metrics.items():
            print(f"   • {operation}: {time_taken:.3f}s")
        
        # Analyze specific code elements
        print(f"\n🔬 Code Structure Analysis:")
        graph = manager.get_graph()
        
        # Count different types of nodes
        functions = [n for n, d in graph.nodes(data=True) if d.get('type') == 'function']
        classes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'class']
        variables = [n for n, d in graph.nodes(data=True) if d.get('type') == 'variable']
        requirements = [n for n, d in graph.nodes(data=True) if d.get('type') == 'requirement']
        
        print(f"   • Functions found: {len(functions)}")
        if functions:
            print(f"     - {', '.join(functions[:5])}")
            if len(functions) > 5:
                print(f"     - ... and {len(functions) - 5} more")
        
        print(f"   • Classes found: {len(classes)}")
        if classes:
            print(f"     - {', '.join(classes)}")
        
        print(f"   • Variables found: {len(variables)}")
        if variables:
            print(f"     - {', '.join(variables[:3])}")
            if len(variables) > 3:
                print(f"     - ... and {len(variables) - 3} more")
        
        print(f"   • Requirements extracted: {len(requirements)}")
        
        # Show requirement-code mappings
        print(f"\n🔗 Requirement-Code Mappings:")
        mapping_edges = [(s, t, d) for s, t, d in graph.edges(data=True) 
                        if d.get('type') in ['MAPS_TO', 'VIOLATES', 'SATISFIES']]
        
        if mapping_edges:
            for source, target, edge_data in mapping_edges[:10]:  # Show first 10
                edge_type = edge_data.get('type', 'unknown')
                print(f"   • {source} --{edge_type}--> {target}")
            
            if len(mapping_edges) > 10:
                print(f"   • ... and {len(mapping_edges) - 10} more mappings")
        else:
            print("   • No requirement mappings found")
        
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
    
    # Health check
    health = manager.health_check()
    print(f"\n🏥 System Health:")
    print(f"   • Status: {health['status']}")
    print(f"   • Graph size: {health['graph_nodes']} nodes, {health['graph_edges']} edges")
    
    if 'warnings' in health:
        print(f"   • Warnings: {health['warnings']}")
    
    print(f"\n✨ Enhanced GraphManager Analysis Complete!")
    print(f"\nThe Enhanced GraphManager successfully:")
    print(f"   🔍 Extracted {stats['total_nodes']} code elements from {len(source_code.splitlines())} lines")
    print(f"   📝 Identified {len(requirements)} requirements from the issue description")
    print(f"   🔗 Traced {deps['nodes_with_dependencies']} dependency relationships")
    print(f"   ⚠️  Detected {violations['total_violations']} potential violations")
    print(f"   ⏱️  Completed analysis in {results['execution_time']:.3f} seconds")


if __name__ == "__main__":
    main()