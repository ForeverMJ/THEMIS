"""
Debug version of the experiment to track revision count and workflow execution.
"""

import asyncio
import time
from pathlib import Path
import networkx as nx

from src.main_enhanced import build_workflow, MAX_REVISIONS
from src.state import AgentState


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


class DebugWorkflowWrapper:
    """Wrapper to add debugging to the workflow execution."""
    
    def __init__(self, app):
        self.app = app
        self.step_count = 0
        self.revision_history = []
    
    def invoke(self, initial_state, config=None):
        """Invoke with debugging."""
        print(f"🐛 Debug: Starting workflow execution")
        print(f"   MAX_REVISIONS = {MAX_REVISIONS}")
        print(f"   Initial revision_count = {initial_state.get('revision_count', 0)}")
        print(f"   Recursion limit = {config.get('recursion_limit', 'default') if config else 'default'}")
        
        # Hook into the state updates to track progress
        current_state = initial_state
        
        try:
            final_state = self.app.invoke(initial_state, config=config)
            
            print(f"\n🐛 Debug: Workflow completed")
            print(f"   Final revision_count = {final_state.get('revision_count', 0)}")
            print(f"   Final conflict_report = {'Yes' if final_state.get('conflict_report') else 'No'}")
            
            return final_state
            
        except Exception as e:
            print(f"\n🐛 Debug: Workflow failed with error: {e}")
            raise


def debug_traditional_analysis():
    """Run traditional analysis with debugging."""
    
    print("🐛 Debug Traditional Analysis")
    print("=" * 60)
    
    # Load experiment data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"
    
    try:
        requirements = load_text(req_path)
        source_code = load_text(code_path)
    except FileNotFoundError as e:
        print(f"❌ Could not load experiment data: {e}")
        return
    
    target_filename = "target_file.py"
    
    print(f"📋 Experiment Setup:")
    print(f"   Requirements length: {len(requirements)} characters")
    print(f"   Source code length: {len(source_code)} characters")
    print(f"   Target filename: {target_filename}")
    
    # Build workflow
    workflow = build_workflow()
    app = workflow.compile()
    debug_app = DebugWorkflowWrapper(app)
    
    initial_state: AgentState = {
        "messages": [],
        "files": {target_filename: source_code},
        "requirements": requirements,
        "knowledge_graph": nx.DiGraph(),
        "baseline_graph": None,
        "conflict_report": None,
        "revision_count": 0,
    }
    
    print(f"\n🔄 Running Traditional Analysis...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Use a lower recursion limit to catch infinite loops faster
        final_state = debug_app.invoke(initial_state, config={"recursion_limit": 15})
        analysis_time = time.time() - start_time
        
        print(f"\n📊 Analysis Results:")
        print(f"   Processing time: {analysis_time:.2f}s")
        print(f"   Final revision_count: {final_state.get('revision_count', 0)}")
        print(f"   Final conflict_report: {'Present' if final_state.get('conflict_report') else 'None'}")
        
        if final_state.get('conflict_report'):
            print(f"   Conflict details: {final_state['conflict_report'][:200]}...")
        
        # Check if revision limit was respected
        revision_count = final_state.get('revision_count', 0)
        if revision_count <= MAX_REVISIONS:
            print(f"   ✅ Revision limit respected: {revision_count} <= {MAX_REVISIONS}")
        else:
            print(f"   ❌ Revision limit exceeded: {revision_count} > {MAX_REVISIONS}")
            print(f"   🚨 This indicates a bug in the workflow logic!")
        
        # Show code changes
        original_code = source_code
        final_code = final_state.get('files', {}).get(target_filename, original_code)
        
        if final_code != original_code:
            print(f"\n📝 Code was modified:")
            print(f"   Original length: {len(original_code)} characters")
            print(f"   Final length: {len(final_code)} characters")
        else:
            print(f"\n📝 No code changes made")
            
    except Exception as e:
        analysis_time = time.time() - start_time
        print(f"\n❌ Analysis failed after {analysis_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()


def test_should_revise_logic():
    """Test the should_revise logic with different scenarios."""
    
    print(f"\n🧪 Testing should_revise Logic")
    print("=" * 60)
    
    # Import the actual function logic
    def should_revise(state):
        return "revise" if state.get("conflict_report") and state.get("revision_count", 0) < MAX_REVISIONS else "end"
    
    test_cases = [
        {"revision_count": 0, "conflict_report": "Some conflict", "expected": "revise"},
        {"revision_count": 1, "conflict_report": "Some conflict", "expected": "revise"},
        {"revision_count": 2, "conflict_report": "Some conflict", "expected": "revise"},
        {"revision_count": 3, "conflict_report": "Some conflict", "expected": "end"},
        {"revision_count": 4, "conflict_report": "Some conflict", "expected": "end"},
        {"revision_count": 0, "conflict_report": None, "expected": "end"},
        {"revision_count": 1, "conflict_report": "", "expected": "end"},
    ]
    
    print(f"MAX_REVISIONS = {MAX_REVISIONS}")
    print()
    
    all_passed = True
    for i, case in enumerate(test_cases, 1):
        result = should_revise(case)
        passed = result == case["expected"]
        status = "✅" if passed else "❌"
        
        if not passed:
            all_passed = False
        
        print(f"{status} Case {i}: revision_count={case['revision_count']}, "
              f"conflict={'Yes' if case['conflict_report'] else 'No'} -> {result} "
              f"(expected: {case['expected']})")
    
    if all_passed:
        print(f"\n✅ All test cases passed - logic is correct")
    else:
        print(f"\n❌ Some test cases failed - there may be a logic error")


def main():
    """Main debug function."""
    
    print("🐛 Advanced Code Analysis Debug Tool")
    print("=" * 80)
    
    # Test the logic first
    test_should_revise_logic()
    
    # Then run the actual analysis
    debug_traditional_analysis()
    
    print(f"\n💡 Debug Tips:")
    print(f"   • If revision_count exceeds {MAX_REVISIONS}, there's a bug in the workflow")
    print(f"   • Check if conflict_report is being cleared properly")
    print(f"   • Look for any state copying issues")
    print(f"   • Verify that should_revise logic is being called correctly")


if __name__ == "__main__":
    main()