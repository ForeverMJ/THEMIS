"""
Example usage of the integrated Advanced Code Analysis system.

This script demonstrates how to use both the standalone AdvancedCodeAnalyzer
and the unified EnhancedGraphAdapter for analyzing code issues.
"""

import asyncio
import tempfile
import os
from pathlib import Path

from .advanced_code_analyzer import AdvancedCodeAnalyzer
from .config import AdvancedAnalysisConfig, LLMConfig
from .graph_manager_integration import GraphManagerIntegration, IntegrationConfig

# Import the unified adapter
try:
    from ..enhanced_graph_adapter import (
        EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
    )
    UNIFIED_ADAPTER_AVAILABLE = True
except ImportError:
    UNIFIED_ADAPTER_AVAILABLE = False


async def main():
    """Main example function."""
    print("🔍 Advanced Code Analysis System - Integration Example")
    print("=" * 60)
    
    # Create configuration
    config = AdvancedAnalysisConfig(
        llm=LLMConfig(
            provider="mock",  # Using mock for demo
            model_name="demo-model",
            max_completion_tokens=4000,
            temperature=0.1
        )
    )
    # Fix the context token limit
    config.analysis.max_context_tokens = 3000
    
    # Initialize analyzer
    print("\n📋 Initializing Advanced Code Analyzer...")
    analyzer = AdvancedCodeAnalyzer(config)
    
    # Validate configuration
    issues = analyzer.validate_configuration()
    if issues:
        print(f"⚠️  Configuration issues: {issues}")
    else:
        print("✅ Configuration validated successfully")
    
    # Test connection
    print("\n🔗 Testing LLM connection...")
    connection_ok = await analyzer.test_connection()
    print(f"{'✅' if connection_ok else '❌'} Connection test: {'Passed' if connection_ok else 'Failed'}")
    
    # Create sample code with bugs
    sample_code = '''
def calculate_total(items):
    """Calculate total price of items."""
    total = 0
    for item in items:
        if item.price = None:  # Bug: assignment instead of comparison
            continue
        total = total + item.price
    return total

def process_orders(orders):
    """Process a list of orders."""
    processed = []
    for order in orders:
        # Bug: constant assignment instead of increment
        order.status = "processed"
        order.processed_count = 1  # Should be += 1
        processed.append(order)
    return processed

class OrderManager:
    def __init__(self):
        self.orders = []
        
    def add_order(self, order):
        # Bug: potential index out of bounds
        self.orders[len(self.orders)] = order  # Should use append()
        
    def get_total_value(self):
        total = 0
        for order in self.orders:
            total += calculate_total(order.items)
        return total
'''
    
    # Write sample code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        temp_file = f.name
    
    try:
        # Set up mock responses for realistic demo
        mock_responses = [
            # Bug classification response
            '{"category": "LOGIC_ERROR", "subcategory": "assignment_error", "confidence": 0.85, "characteristics": ["assignment_operator", "comparison_error", "conditional_logic"], "reasoning": "Multiple assignment vs comparison errors detected in conditional statements"}',
            
            # Semantic extraction response
            '{"technical_concepts": [{"term": "assignment", "confidence": 0.9}, {"term": "comparison", "confidence": 0.8}, {"term": "increment", "confidence": 0.7}], "function_names": [{"name": "calculate_total", "confidence": 0.9}, {"name": "process_orders", "confidence": 0.8}, {"name": "add_order", "confidence": 0.7}], "variable_names": [{"name": "total", "confidence": 0.8}, {"name": "processed_count", "confidence": 0.7}], "class_names": [{"name": "OrderManager", "confidence": 0.9}], "error_patterns": [{"pattern": "assignment in condition", "confidence": 0.9}, {"pattern": "constant assignment", "confidence": 0.7}], "api_calls": [], "problem_summary": "Multiple logic errors involving assignment operators and array access", "overall_confidence": 0.82}',
            
            # Concept mapping response
            'Based on the extracted concepts, the primary issues are located in the calculate_total function (assignment vs comparison) and process_orders function (increment logic). The OrderManager.add_order method also has a potential array bounds issue.',
            
            # Multi-round reasoning responses
            'Initial analysis identifies three distinct bug patterns: 1) Assignment operator used instead of comparison in calculate_total function line 5, 2) Constant assignment instead of increment in process_orders function line 15, 3) Array bounds violation in OrderManager.add_order method line 26.',
            
            'Verification confirms these are genuine bugs that would cause runtime errors or incorrect behavior. The assignment vs comparison bug would cause a syntax error, the increment bug would reset the counter instead of incrementing, and the array access bug would cause IndexError.',
            
            'Final analysis: Three critical bugs identified with high confidence. Primary fix locations are calculate_total (line 5: change = to ==), process_orders (line 15: change = 1 to += 1), and add_order (line 26: use append() instead of direct index assignment).'
        ]
        
        if hasattr(analyzer.llm_interface.provider, 'set_responses'):
            analyzer.llm_interface.provider.set_responses(mock_responses)
        
        # Example 1: Basic analysis
        print("\n🔍 Example 1: Basic Code Analysis")
        print("-" * 40)
        
        issue_description = "The code has several bugs including assignment vs comparison errors and incorrect increment logic"
        
        print(f"Issue: {issue_description}")
        print(f"Analyzing file: {temp_file}")
        
        result = await analyzer.analyze(
            issue_text=issue_description,
            target_files=[temp_file]
        )
        
        print(f"\n📊 Analysis Results:")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   LLM calls made: {result.total_llm_calls}")
        print(f"   Tokens used: {result.total_tokens_used}")
        
        print(f"\n🎯 Primary Analysis:")
        print(f"   Bug location: {result.primary_analysis.bug_location}")
        print(f"   Root cause: {result.primary_analysis.root_cause}")
        print(f"   Fix suggestion: {result.primary_analysis.fix_suggestion}")
        print(f"   Confidence: {result.primary_analysis.confidence:.2f}")
        
        if result.alternative_solutions:
            print(f"\n🔄 Alternative solutions: {len(result.alternative_solutions)}")
        
        if result.conflicts_detected:
            print(f"⚠️  Conflicts detected: {len(result.conflicts_detected)}")
        
        # Example 2: Focused analysis
        print("\n🔍 Example 2: Focused Analysis")
        print("-" * 40)
        
        # Reset mock responses for second analysis
        focused_responses = [
            '{"category": "LOGIC_ERROR", "subcategory": "assignment_error", "confidence": 0.9, "characteristics": ["assignment_operator"], "reasoning": "Assignment operator used in conditional statement"}',
            '{"technical_concepts": [{"term": "assignment", "confidence": 0.95}], "function_names": [{"name": "calculate_total", "confidence": 0.95}], "variable_names": [], "class_names": [], "error_patterns": [{"pattern": "assignment in condition", "confidence": 0.95}], "api_calls": [], "problem_summary": "Assignment vs comparison error in calculate_total", "overall_confidence": 0.9}',
            'Focused analysis on calculate_total function confirms assignment operator used instead of comparison operator.',
            'The bug is specifically in the condition check where = is used instead of ==.',
            'High confidence fix: change "if item.price = None:" to "if item.price == None:" or "if item.price is None:"'
        ]
        
        if hasattr(analyzer.llm_interface.provider, 'set_responses'):
            analyzer.llm_interface.provider.set_responses(focused_responses)
        
        focused_issue = "The calculate_total function has an assignment vs comparison bug"
        focus_elements = ["calculate_total", "item.price", "assignment"]
        
        print(f"Issue: {focused_issue}")
        print(f"Focus elements: {focus_elements}")
        
        focused_result = await analyzer.analyze(
            issue_text=focused_issue,
            target_files=[temp_file],
            focus_elements=focus_elements
        )
        
        print(f"\n📊 Focused Analysis Results:")
        print(f"   Processing time: {focused_result.processing_time:.2f}s")
        print(f"   Bug location: {focused_result.primary_analysis.bug_location}")
        print(f"   Root cause: {focused_result.primary_analysis.root_cause}")
        print(f"   Confidence: {focused_result.primary_analysis.confidence:.2f}")
        
        # Performance statistics
        print("\n📈 Performance Statistics")
        print("-" * 40)
        
        stats = analyzer.get_performance_stats()
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Successful analyses: {stats['successful_analyses']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
        print(f"   Active sessions: {stats['active_sessions']}")
        
        # Example 3: Unified Adapter (if available)
        if UNIFIED_ADAPTER_AVAILABLE:
            print("\n🔍 Example 3: Unified Adapter Integration")
            print("-" * 40)
            
            try:
                # Initialize unified adapter
                unified_adapter = EnhancedGraphAdapter()
                
                # Check available strategies
                status = unified_adapter.get_system_status()
                print(f"Available strategies: {status['available_strategies']}")
                
                # Run analysis with auto-strategy selection
                options = AnalysisOptions(
                    strategy=AnalysisStrategy.AUTO_SELECT,
                    confidence_threshold=0.6,
                    include_requirements=True
                )
                
                requirements_text = """
                The system SHALL validate input parameters correctly.
                WHEN processing items THEN the system SHALL handle null values appropriately.
                WHEN incrementing counters THEN the system SHALL use proper increment operations.
                The system SHALL prevent array bounds violations.
                """
                
                unified_result = await unified_adapter.analyze(
                    issue_text=issue_description,
                    target_files=[temp_file],
                    requirements_text=requirements_text,
                    options=options
                )
                
                print(f"\n📊 Unified Analysis Results:")
                print(f"   Strategy used: {unified_result.strategy_used.value}")
                print(f"   Success: {'✅' if unified_result.success else '❌'}")
                print(f"   Processing time: {unified_result.processing_time:.2f}s")
                print(f"   Confidence: {unified_result.confidence_score:.2f}")
                
                print(f"\n🎯 Primary Findings:")
                for i, finding in enumerate(unified_result.primary_findings, 1):
                    print(f"   {i}. {finding}")
                
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(unified_result.recommendations, 1):
                    print(f"   {i}. {rec}")
                
                # Show integration benefits
                if unified_result.has_graph_analysis():
                    print(f"\n📈 Graph Analysis Benefits:")
                    if unified_result.graph_statistics:
                        stats = unified_result.graph_statistics
                        print(f"   Code structure: {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
                    
                    if unified_result.violation_report:
                        violations = unified_result.violation_report.get('total_violations', 0)
                        print(f"   Requirement violations: {violations}")
                
            except Exception as e:
                print(f"❌ Unified adapter example failed: {e}")
        
        else:
            print("\n⚠️  Unified adapter not available (Enhanced GraphManager not found)")
        
        # Example 4: Direct Integration Layer (if available)
        try:
            print("\n🔍 Example 4: Direct Integration Layer")
            print("-" * 40)
            
            # Initialize integration layer directly
            integration_config = IntegrationConfig(
                enable_graph_context_enhancement=True,
                enable_dependency_aware_analysis=True,
                fallback_to_basic_analysis=True
            )
            
            # This will work even if Enhanced GraphManager is not available
            # due to the fallback mechanism
            print("Integration layer configuration created")
            print(f"Graph context enhancement: {integration_config.enable_graph_context_enhancement}")
            print(f"Dependency aware analysis: {integration_config.enable_dependency_aware_analysis}")
            print(f"Fallback enabled: {integration_config.fallback_to_basic_analysis}")
            
        except Exception as e:
            print(f"❌ Integration layer example failed: {e}")
        
        print("\n✅ Integration example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            os.unlink(temp_file)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(main())