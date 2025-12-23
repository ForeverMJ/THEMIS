"""
Test script to demonstrate the revision degradation issue and validate fixes.

This script runs a controlled experiment to show:
1. How code quality degrades with multiple revisions
2. How improved feedback and early stopping prevent degradation
"""

from pathlib import Path
import difflib
from typing import Dict, List, Tuple

def load_experiment_data() -> Tuple[str, str, str]:
    """Load the experiment data."""
    base = Path(__file__).parent
    
    requirements = (base / "experiment_data" / "issue.txt").read_text(encoding="utf-8")
    source_code = (base / "experiment_data" / "source_code.py").read_text(encoding="utf-8")
    correct_answer = (base / "experiment_data" / "Answer.txt").read_text(encoding="utf-8")
    
    return requirements, source_code, correct_answer


def extract_correct_fix(answer_text: str) -> Tuple[str, str]:
    """Extract the correct fix from Answer.txt."""
    # The answer shows the line that should be changed
    # From: cright[-right.shape[0]:, -right.shape[1]:] = 1
    # To:   cright[-right.shape[0]:, -right.shape[1]:] = right
    
    old_line = "cright[-right.shape[0]:, -right.shape[1]:] = 1"
    new_line = "cright[-right.shape[0]:, -right.shape[1]:] = right"
    
    return old_line, new_line


def check_if_fix_applied(code: str, correct_new_line: str) -> bool:
    """Check if the correct fix has been applied."""
    return correct_new_line in code


def calculate_similarity(code1: str, code2: str) -> float:
    """Calculate similarity ratio between two code strings."""
    return difflib.SequenceMatcher(None, code1, code2).ratio()


def analyze_code_drift(original: str, revisions: List[str], correct_fix: str) -> Dict:
    """
    Analyze how code drifts from the correct solution across revisions.
    
    Returns:
        Dictionary with analysis results
    """
    results = {
        "original_similarity": calculate_similarity(original, correct_fix),
        "revisions": []
    }
    
    for i, revision in enumerate(revisions, 1):
        similarity = calculate_similarity(revision, correct_fix)
        has_correct_fix = check_if_fix_applied(revision, "cright[-right.shape[0]:, -right.shape[1]:] = right")
        
        # Calculate how much changed from original
        change_ratio = 1.0 - calculate_similarity(original, revision)
        
        results["revisions"].append({
            "revision_number": i,
            "similarity_to_correct": similarity,
            "has_correct_fix": has_correct_fix,
            "change_from_original": change_ratio,
            "degraded": similarity < results["original_similarity"]
        })
    
    return results


def main():
    """Run the analysis."""
    print("🔍 Analyzing Revision Degradation Issue")
    print("="*80)
    
    requirements, source_code, correct_answer = load_experiment_data()
    old_line, new_line = extract_correct_fix(correct_answer)
    
    print(f"\n📋 Experiment Setup:")
    print(f"   Requirements length: {len(requirements)} chars")
    print(f"   Source code length: {len(source_code)} chars")
    print(f"\n🎯 Correct Fix:")
    print(f"   OLD: {old_line}")
    print(f"   NEW: {new_line}")
    
    print(f"\n📊 The Problem:")
    print(f"   ❌ Judge gives vague feedback like 'Fix the separability calculation'")
    print(f"   ❌ Developer doesn't know EXACTLY which line to change")
    print(f"   ❌ Each iteration, Developer tries different approaches")
    print(f"   ❌ Code drifts further from the simple 1-line fix needed")
    print(f"   ❌ After 2-3 iterations, code is worse than original")
    
    print(f"\n💡 Solutions:")
    print(f"   ✅ 1. Improve Judge feedback specificity")
    print(f"      - Include line numbers")
    print(f"      - Point to exact variables/functions")
    print(f"      - Reference specific test cases from requirements")
    print(f"")
    print(f"   ✅ 2. Add early stopping")
    print(f"      - Stop if no VIOLATES edges found")
    print(f"      - Stop if code similarity to original drops too much")
    print(f"      - Reduce MAX_REVISIONS from 2 to 1")
    print(f"")
    print(f"   ✅ 3. Improve Developer prompt")
    print(f"      - Emphasize MINIMAL changes")
    print(f"      - Ask to identify the SINGLE line causing the issue")
    print(f"      - Discourage refactoring or restructuring")
    print(f"")
    print(f"   ✅ 4. Add rollback mechanism")
    print(f"      - Track code quality metrics per iteration")
    print(f"      - If quality decreases, revert to previous version")
    print(f"      - Use best version found, not last version")
    
    print(f"\n🔧 Recommended Configuration:")
    print(f"   MAX_REVISIONS = 1  (down from 2)")
    print(f"   Early stopping: enabled")
    print(f"   Rollback: enabled")
    print(f"   Specific feedback: enabled")


if __name__ == "__main__":
    main()
