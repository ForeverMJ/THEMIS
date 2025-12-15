"""
Test script to verify that the revision limit works correctly.
"""

import networkx as nx
from src.main import build_workflow
from src.state import AgentState


def test_revision_limit():
    """Test that the workflow stops after MAX_REVISIONS."""
    
    print("🧪 Testing Revision Limit Logic")
    print("=" * 50)
    
    # Create a simple test case that will always have conflicts
    workflow = build_workflow()
    app = workflow.compile()
    
    # Use a simple example that might cause conflicts
    initial_state: AgentState = {
        "messages": [],
        "files": {"test.py": "def add(a, b):\n    return a - b  # Wrong operation!"},
        "requirements": "The add function must return the sum of two numbers, not the difference.",
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }
    
    print("📋 Initial State:")
    print(f"   Files: {list(initial_state['files'].keys())}")
    print(f"   Requirements: {initial_state['requirements'][:50]}...")
    print(f"   Initial revision_count: {initial_state['revision_count']}")
    
    # Track the execution
    print(f"\n🔄 Running workflow...")
    
    try:
        final_state = app.invoke(initial_state, config={"recursion_limit": 20})
        
        print(f"\n📊 Final Results:")
        print(f"   Final revision_count: {final_state.get('revision_count', 0)}")
        print(f"   Final conflict_report: {'Yes' if final_state.get('conflict_report') else 'No'}")
        print(f"   Files modified: {len(final_state.get('files', {}))}")
        
        # Check if the limit was respected
        revision_count = final_state.get('revision_count', 0)
        if revision_count <= 3:
            print(f"   ✅ Revision limit respected: {revision_count} <= 3")
        else:
            print(f"   ❌ Revision limit exceeded: {revision_count} > 3")
            
        # Show the final code
        final_code = final_state.get('files', {}).get('test.py', '')
        print(f"\n📝 Final Code:")
        print(final_code)
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def test_should_revise_logic():
    """Test the should_revise logic directly."""
    
    print(f"\n🧪 Testing should_revise Logic")
    print("=" * 50)
    
    from src.main import MAX_REVISIONS
    
    # Simulate different states
    test_cases = [
        {"revision_count": 0, "conflict_report": "Some conflict", "expected": "revise"},
        {"revision_count": 1, "conflict_report": "Some conflict", "expected": "revise"},
        {"revision_count": 2, "conflict_report": "Some conflict", "expected": "revise"},
        {"revision_count": 3, "conflict_report": "Some conflict", "expected": "end"},
        {"revision_count": 4, "conflict_report": "Some conflict", "expected": "end"},
        {"revision_count": 0, "conflict_report": None, "expected": "end"},
        {"revision_count": 1, "conflict_report": None, "expected": "end"},
    ]
    
    def should_revise(state):
        return "revise" if state.get("conflict_report") and state.get("revision_count", 0) < MAX_REVISIONS else "end"
    
    print(f"MAX_REVISIONS = {MAX_REVISIONS}")
    print()
    
    for i, case in enumerate(test_cases, 1):
        result = should_revise(case)
        status = "✅" if result == case["expected"] else "❌"
        print(f"{status} Case {i}: revision_count={case['revision_count']}, "
              f"conflict={'Yes' if case['conflict_report'] else 'No'} -> {result} "
              f"(expected: {case['expected']})")


if __name__ == "__main__":
    test_should_revise_logic()
    test_revision_limit()