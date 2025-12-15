"""
Quick test script for the Advanced Code Analysis system.

This script provides a simple way to test the new advanced analysis capabilities
on the experiment data with minimal setup.
"""

import asyncio
import sys
from pathlib import Path

# Import the unified adapter
try:
    from src.enhanced_graph_adapter import (
        EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
    )
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    print("❌ Advanced analysis system not available")
    print("Make sure you have installed all dependencies and the system is properly configured")
    sys.exit(1)


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


async def quick_test():
    """Run a quick test of the advanced analysis system."""
    
    print("🚀 Quick Test - Advanced Code Analysis System")
    print("=" * 60)
    
    # Load default experiment data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"
    
    try:
        issue_text = load_text(req_path)
        source_code = load_text(code_path)
    except FileNotFoundError as e:
        print(f"❌ Could not load experiment data: {e}")
        print("Make sure experiment_data/issue.txt and experiment_data/source_code.py exist")
        return
    
    print(f"📋 Test Data:")
    print(f"   Issue: {issue_text[:100]}...")
    print(f"   Code length: {len(source_code)} characters")
    
    # Initialize the system
    print(f"\n🔧 Initializing Advanced Analysis System...")
    try:
        adapter = EnhancedGraphAdapter()
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Check system status
    status = adapter.get_system_status()
    print(f"📊 System Status:")
    print(f"   Available systems: {list(status['systems_initialized'].keys())}")
    print(f"   Available strategies: {status['available_strategies']}")
    
    # Create temporary file
    target_filename = "temp_test_file.py"
    temp_file = Path(target_filename)
    temp_file.write_text(source_code, encoding='utf-8')
    
    try:
        # Run analysis with auto-select strategy
        print(f"\n🔍 Running Analysis...")
        options = AnalysisOptions(
            strategy=AnalysisStrategy.AUTO_SELECT,
            confidence_threshold=0.5,
            debug_mode=True
        )
        
        result = await adapter.analyze(
            issue_text=issue_text,
            target_files=[target_filename],
            options=options
        )
        
        # Display results
        print(f"\n📋 Analysis Results:")
        print(f"   Strategy Used: {result.strategy_used.value}")
        print(f"   Success: {'✅' if result.success else '❌'}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        print(f"   Confidence Score: {result.confidence_score:.2f}")
        
        if result.error_message:
            print(f"   ❌ Error: {result.error_message}")
        else:
            print(f"\n🔍 Findings ({len(result.primary_findings)}):")
            for i, finding in enumerate(result.primary_findings, 1):
                print(f"   {i}. {finding}")
            
            print(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"   {i}. {rec}")
            
            # Show additional details if available
            if result.has_advanced_analysis():
                print(f"\n🧠 Advanced LLM Analysis: Available")
            
            if result.has_graph_analysis():
                print(f"📊 Graph Analysis: Available")
                if result.graph_statistics:
                    stats = result.graph_statistics
                    print(f"   Graph: {stats.get('total_nodes', 0)} nodes, {stats.get('total_edges', 0)} edges")
        
        print(f"\n✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


async def test_all_strategies():
    """Test all available analysis strategies."""
    
    print("\n🧪 Testing All Available Strategies")
    print("=" * 60)
    
    # Load test data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"
    
    try:
        issue_text = load_text(req_path)
        source_code = load_text(code_path)
    except FileNotFoundError:
        print("❌ Test data not available")
        return
    
    # Initialize system
    adapter = EnhancedGraphAdapter()
    status = adapter.get_system_status()
    
    strategies_to_test = []
    if 'advanced_only' in status['available_strategies']:
        strategies_to_test.append(AnalysisStrategy.ADVANCED_ONLY)
    if 'graph_only' in status['available_strategies']:
        strategies_to_test.append(AnalysisStrategy.GRAPH_ONLY)
    if 'integrated' in status['available_strategies']:
        strategies_to_test.append(AnalysisStrategy.INTEGRATED)
    
    if not strategies_to_test:
        print("❌ No strategies available for testing")
        return
    
    # Create temporary file
    target_filename = "temp_strategy_test.py"
    temp_file = Path(target_filename)
    temp_file.write_text(source_code, encoding='utf-8')
    
    results = {}
    
    try:
        for strategy in strategies_to_test:
            print(f"\n🔍 Testing {strategy.value}...")
            
            options = AnalysisOptions(
                strategy=strategy,
                confidence_threshold=0.5
            )
            
            try:
                result = await adapter.analyze(
                    issue_text=issue_text,
                    target_files=[target_filename],
                    options=options
                )
                
                results[strategy.value] = result
                print(f"   {'✅' if result.success else '❌'} {strategy.value}: "
                      f"Confidence {result.confidence_score:.2f}, "
                      f"Time {result.processing_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ {strategy.value}: Failed - {e}")
        
        # Summary
        print(f"\n📊 Strategy Comparison:")
        successful_results = [(name, result) for name, result in results.items() if result.success]
        
        if successful_results:
            best_strategy, best_result = max(successful_results, key=lambda x: x[1].confidence_score)
            print(f"   Best strategy: {best_strategy} (confidence: {best_result.confidence_score:.2f})")
            
            avg_confidence = sum(result.confidence_score for _, result in successful_results) / len(successful_results)
            avg_time = sum(result.processing_time for _, result in successful_results) / len(successful_results)
            print(f"   Average confidence: {avg_confidence:.2f}")
            print(f"   Average time: {avg_time:.2f}s")
        else:
            print("   No strategies succeeded")
            
    finally:
        if temp_file.exists():
            temp_file.unlink()


async def main():
    """Main test function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all-strategies":
        await test_all_strategies()
    else:
        await quick_test()
    
    print(f"\n💡 Usage Tips:")
    print(f"   • Run 'python run_quick_test.py' for a basic test")
    print(f"   • Run 'python run_quick_test.py --all-strategies' to test all strategies")
    print(f"   • Run 'python run_experiment_advanced.py' for comprehensive analysis")
    print(f"   • Check 'demo_integration.py' for system integration demo")


if __name__ == "__main__":
    asyncio.run(main())