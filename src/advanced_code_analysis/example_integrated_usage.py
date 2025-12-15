"""
Example usage of the integrated Advanced Code Analysis and Enhanced GraphManager system.

This script demonstrates how to use the unified interface to perform code analysis
with different strategies and configuration options.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

# Import the unified adapter
from ..enhanced_graph_adapter import (
    EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions, UnifiedAnalysisResult
)

# Import configuration classes
from .config import AdvancedAnalysisConfig
try:
    from ..enhanced_graph_manager.config import EnhancedGraphManagerConfig
    GRAPH_MANAGER_AVAILABLE = True
except ImportError:
    GRAPH_MANAGER_AVAILABLE = False
    EnhancedGraphManagerConfig = None


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_integrated_analysis():
    """Example of integrated analysis using both systems."""
    
    print("=== Integrated Advanced Code Analysis Example ===\n")
    
    # Initialize the adapter with default configurations
    try:
        adapter = EnhancedGraphAdapter()
        print("✓ Enhanced Graph Adapter initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize adapter: {e}")
        return
    
    # Check system status
    status = adapter.get_system_status()
    print(f"Available systems: {status['systems_initialized']}")
    print(f"Available strategies: {status['available_strategies']}\n")
    
    # Example issue and code files
    issue_text = """
    There's a bug in the user authentication function. When a user tries to log in
    with an invalid password, the system should return an error message, but instead
    it's throwing an exception and crashing the application. The issue seems to be
    in the validate_password function where we're not properly handling the case
    when the password hash comparison fails.
    """
    
    requirements_text = """
    The system SHALL validate user credentials securely.
    WHEN a user provides invalid credentials THEN the system SHALL return an appropriate error message.
    WHEN password validation fails THEN the system SHALL NOT throw unhandled exceptions.
    The system SHALL log authentication attempts for security monitoring.
    """
    
    # Use example files from the project
    target_files = [
        "src/main.py",  # Example file
    ]
    
    # Filter to existing files
    existing_files = [f for f in target_files if Path(f).exists()]
    if not existing_files:
        print("No target files found, using current file as example")
        existing_files = [__file__]
    
    print(f"Analyzing files: {existing_files}")
    print(f"Issue: {issue_text[:100]}...")
    print(f"Requirements provided: {bool(requirements_text)}\n")
    
    # Example 1: Auto-select strategy
    print("--- Example 1: Auto-select Strategy ---")
    try:
        options = AnalysisOptions(
            strategy=AnalysisStrategy.AUTO_SELECT,
            confidence_threshold=0.6,
            include_requirements=True
        )
        
        result = await adapter.analyze(
            issue_text=issue_text,
            target_files=existing_files,
            requirements_text=requirements_text,
            options=options
        )
        
        print_analysis_result(result)
        
    except Exception as e:
        print(f"✗ Auto-select analysis failed: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Force integrated analysis (if available)
    if 'integrated' in status['available_strategies']:
        print("--- Example 2: Integrated Analysis ---")
        try:
            options = AnalysisOptions(
                strategy=AnalysisStrategy.INTEGRATED,
                confidence_threshold=0.7,
                debug_mode=True
            )
            
            result = await adapter.analyze(
                issue_text=issue_text,
                target_files=existing_files,
                requirements_text=requirements_text,
                options=options
            )
            
            print_analysis_result(result)
            
        except Exception as e:
            print(f"✗ Integrated analysis failed: {e}")
        
        print("\n" + "="*60 + "\n")
    
    # Example 3: Advanced analysis only
    if 'advanced_only' in status['available_strategies']:
        print("--- Example 3: Advanced Analysis Only ---")
        try:
            options = AnalysisOptions(
                strategy=AnalysisStrategy.ADVANCED_ONLY,
                max_context_tokens=4000,
                confidence_threshold=0.5
            )
            
            result = await adapter.analyze(
                issue_text=issue_text,
                target_files=existing_files,
                requirements_text=requirements_text,
                options=options
            )
            
            print_analysis_result(result)
            
        except Exception as e:
            print(f"✗ Advanced analysis failed: {e}")
        
        print("\n" + "="*60 + "\n")
    
    # Example 4: Graph analysis only (if available)
    if 'graph_only' in status['available_strategies']:
        print("--- Example 4: Graph Analysis Only ---")
        try:
            options = AnalysisOptions(
                strategy=AnalysisStrategy.GRAPH_ONLY,
                include_requirements=True
            )
            
            result = await adapter.analyze(
                issue_text=issue_text,
                target_files=existing_files,
                requirements_text=requirements_text,
                options=options
            )
            
            print_analysis_result(result)
            
        except Exception as e:
            print(f"✗ Graph analysis failed: {e}")
    
    # Show performance summary
    print("\n--- Performance Summary ---")
    final_status = adapter.get_system_status()
    if final_status['performance_metrics']:
        for strategy, metrics in final_status['performance_metrics'].items():
            print(f"{strategy}: {metrics['successful_runs']}/{metrics['total_runs']} successful, "
                  f"avg confidence: {metrics['average_confidence']:.2f}, "
                  f"avg time: {metrics['average_processing_time']:.2f}s")


def print_analysis_result(result: UnifiedAnalysisResult):
    """Print analysis result in a formatted way."""
    
    print(f"Strategy Used: {result.strategy_used.value}")
    print(f"Success: {'✓' if result.success else '✗'}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
        return
    
    print("\nPrimary Findings:")
    for i, finding in enumerate(result.primary_findings, 1):
        print(f"  {i}. {finding}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Show additional details based on analysis type
    if result.has_advanced_analysis():
        print("\n[Advanced Analysis Available]")
    
    if result.has_graph_analysis():
        print("\n[Graph Analysis Available]")
        if result.graph_statistics:
            stats = result.graph_statistics
            print(f"  Graph: {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
        
        if result.violation_report:
            violations = result.violation_report.get('total_violations', 0)
            if violations > 0:
                print(f"  Violations: {violations} requirement violations found")


async def example_configuration_management():
    """Example of configuration management and system validation."""
    
    print("\n=== Configuration Management Example ===\n")
    
    # Load configuration from file
    try:
        config = AdvancedAnalysisConfig.from_file(
            "src/advanced_code_analysis/example_integration_config.json"
        )
        print("✓ Configuration loaded from file")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        config = AdvancedAnalysisConfig()
        print("Using default configuration")
    
    # Initialize adapter with custom configuration
    try:
        adapter = EnhancedGraphAdapter(advanced_config=config)
        print("✓ Adapter initialized with custom configuration")
    except Exception as e:
        print(f"✗ Failed to initialize with custom config: {e}")
        return
    
    # Validate systems
    print("\nValidating systems...")
    validation_results = await adapter.validate_systems()
    
    for system, issues in validation_results.items():
        if issues:
            print(f"✗ {system}: {len(issues)} issues found")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"    - {issue}")
        else:
            print(f"✓ {system}: No issues found")
    
    # Update configuration at runtime
    print("\nUpdating configuration...")
    try:
        await adapter.configure_systems(
            integration={
                'enable_graph_context_enhancement': False,
                'fallback_confidence_threshold': 0.5
            },
            advanced={
                'debug_mode': True
            }
        )
        print("✓ Configuration updated successfully")
    except Exception as e:
        print(f"✗ Configuration update failed: {e}")
    
    # Clear caches
    print("\nClearing caches...")
    adapter.clear_caches()
    print("✓ Caches cleared")


async def main():
    """Main example function."""
    
    print("Enhanced Graph Adapter Integration Examples")
    print("=" * 50)
    
    # Run integrated analysis example
    await example_integrated_analysis()
    
    # Run configuration management example
    await example_configuration_management()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())