#!/usr/bin/env python3
"""
Demonstration of the Advanced Code Analysis and Enhanced GraphManager integration.

This script shows how the integration provides unified access to both systems
with intelligent strategy selection and configuration options.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from advanced_code_analysis.config import AdvancedAnalysisConfig, IntegrationConfig
from advanced_code_analysis.advanced_code_analyzer import AdvancedCodeAnalyzer

# Try to import the unified adapter
try:
    from enhanced_graph_adapter import (
        EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
    )
    UNIFIED_ADAPTER_AVAILABLE = True
except ImportError:
    UNIFIED_ADAPTER_AVAILABLE = False

# Try to import Enhanced GraphManager
try:
    from enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
    from enhanced_graph_manager.config import EnhancedGraphManagerConfig
    ENHANCED_GRAPH_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_GRAPH_MANAGER_AVAILABLE = False


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


async def demonstrate_integration():
    """Demonstrate the integration capabilities."""
    
    print_header("Advanced Code Analysis Integration Demo")
    
    # Check system availability
    print_section("System Availability Check")
    print(f"Enhanced GraphManager Available: {'✅' if ENHANCED_GRAPH_MANAGER_AVAILABLE else '❌'}")
    print(f"Unified Adapter Available: {'✅' if UNIFIED_ADAPTER_AVAILABLE else '❌'}")
    
    # Demonstrate configuration
    print_section("Configuration Management")
    
    # Create advanced analysis config with integration settings
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"  # Use mock for demo
    config.llm.model_name = "demo-model"
    
    # Configure integration settings
    config.integration.enable_graph_context_enhancement = True
    config.integration.enable_dependency_aware_analysis = True
    config.integration.fallback_to_basic_analysis = True
    
    print("✅ Advanced Analysis Configuration:")
    print(f"   LLM Provider: {config.llm.provider}")
    print(f"   Integration Enabled: {config.integrate_with_enhanced_graph_manager}")
    print(f"   Graph Context Enhancement: {config.integration.enable_graph_context_enhancement}")
    print(f"   Fallback Enabled: {config.integration.fallback_to_basic_analysis}")
    
    # Demonstrate standalone Advanced Code Analyzer
    print_section("Standalone Advanced Code Analyzer")
    
    try:
        analyzer = AdvancedCodeAnalyzer(config)
        print("✅ Advanced Code Analyzer initialized")
        
        # Validate configuration
        issues = analyzer.validate_configuration()
        if issues:
            print(f"⚠️  Configuration issues: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"   - {issue}")
        else:
            print("✅ Configuration validated successfully")
        
        # Get performance stats
        stats = analyzer.get_performance_stats()
        print(f"📊 Performance Stats:")
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Failed to initialize Advanced Code Analyzer: {e}")
    
    # Demonstrate Enhanced GraphManager (if available)
    if ENHANCED_GRAPH_MANAGER_AVAILABLE:
        print_section("Enhanced GraphManager")
        
        try:
            graph_config = EnhancedGraphManagerConfig()
            graph_manager = EnhancedGraphManager(graph_config)
            print("✅ Enhanced GraphManager initialized")
            
            # Health check
            health = graph_manager.health_check()
            print(f"🏥 Health Status: {health.get('status', 'unknown')}")
            
            # Show configuration
            print(f"📋 Configuration:")
            print(f"   Max nodes: {graph_config.max_nodes}")
            print(f"   Max edges: {graph_config.max_edges}")
            print(f"   LLM model: {graph_config.llm_model}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Enhanced GraphManager: {e}")
    
    else:
        print_section("Enhanced GraphManager")
        print("❌ Enhanced GraphManager not available")
        print("   This is expected if the enhanced_graph_manager module is not installed")
    
    # Demonstrate Unified Adapter (if available)
    if UNIFIED_ADAPTER_AVAILABLE:
        print_section("Unified Adapter")
        
        try:
            adapter = EnhancedGraphAdapter(advanced_config=config)
            print("✅ Unified Adapter initialized")
            
            # Get system status
            status = adapter.get_system_status()
            print(f"🔧 System Status:")
            print(f"   Systems initialized: {status['systems_initialized']}")
            print(f"   Available strategies: {status['available_strategies']}")
            
            # Validate systems
            validation_results = await adapter.validate_systems()
            print(f"✅ System Validation:")
            for system, issues in validation_results.items():
                if issues:
                    print(f"   {system}: {len(issues)} issues")
                else:
                    print(f"   {system}: ✅ No issues")
            
        except Exception as e:
            print(f"❌ Failed to initialize Unified Adapter: {e}")
    
    else:
        print_section("Unified Adapter")
        print("❌ Unified Adapter not available")
        print("   This is expected if the enhanced_graph_manager module is not available")
    
    # Demonstrate configuration serialization
    print_section("Configuration Serialization")
    
    try:
        # Convert to dictionary
        config_dict = config.to_dict()
        print("✅ Configuration serialized to dictionary")
        print(f"   Keys: {list(config_dict.keys())}")
        
        # Restore from dictionary
        restored_config = AdvancedAnalysisConfig.from_dict(config_dict)
        print("✅ Configuration restored from dictionary")
        
        # Verify integration settings preserved
        assert restored_config.integration.enable_graph_context_enhancement == config.integration.enable_graph_context_enhancement
        print("✅ Integration settings preserved during serialization")
        
    except Exception as e:
        print(f"❌ Configuration serialization failed: {e}")
    
    # Show integration benefits
    print_section("Integration Benefits")
    
    benefits = [
        "🔄 Unified interface for both analysis systems",
        "🎯 Intelligent strategy selection based on problem type",
        "📈 Enhanced context using graph structure information",
        "🔍 Cross-validation of LLM findings with structural analysis",
        "⚡ Fallback mechanisms for robust operation",
        "🛠️ Flexible configuration options for different use cases",
        "📊 Performance tracking and optimization",
        "🔧 Runtime configuration updates without restart"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    # Show usage scenarios
    print_section("Usage Scenarios")
    
    scenarios = [
        ("🐛 Bug Analysis", "Use integrated analysis for complex logic errors with requirements"),
        ("📋 Code Review", "Use graph analysis for structural issues and dependency problems"),
        ("🔍 Performance Issues", "Use advanced analysis for algorithmic and optimization problems"),
        ("📐 Architecture Validation", "Use graph analysis for requirement compliance checking"),
        ("🚀 Auto-Selection", "Let the system choose the best strategy based on context")
    ]
    
    for scenario, description in scenarios:
        print(f"   {scenario}: {description}")
    
    print_header("Integration Demo Completed Successfully!")
    
    print("\n💡 Next Steps:")
    print("   1. Configure your LLM API keys in the configuration")
    print("   2. Try the example_integrated_usage.py script for hands-on examples")
    print("   3. Use the unified adapter in your own code analysis workflows")
    print("   4. Customize integration settings based on your specific needs")


if __name__ == "__main__":
    asyncio.run(demonstrate_integration())