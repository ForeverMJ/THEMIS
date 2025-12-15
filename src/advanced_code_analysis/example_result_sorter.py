"""
Example usage of the Result Sorting and Output Engine.

This script demonstrates how to use the ResultSorter class to rank candidate
solutions, format comprehensive outputs, and validate analysis results.
"""

import asyncio
import logging
from typing import List

from .models import (
    AnalysisResult, ReasoningChain, ReasoningStep, EvidenceChain,
    ContextWindow, BugType, BugCategory
)
from .result_sorter import (
    ResultSorter, SortingCriteria, OutputFormat, 
    CodeImpactMetrics, QualityMetrics
)
from .config import AdvancedAnalysisConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_analysis_results() -> List[AnalysisResult]:
    """Create sample analysis results for demonstration."""
    
    # Result 1: High confidence, specific location
    reasoning_chain_1 = ReasoningChain()
    reasoning_chain_1.add_step(ReasoningStep(
        step_number=1,
        description="Initial analysis of variable assignment",
        confidence=0.9,
        evidence=["Variable 'count' assigned constant instead of variable", "Line 42 shows assignment error"]
    ))
    reasoning_chain_1.add_step(ReasoningStep(
        step_number=2,
        description="Verification of assignment pattern",
        confidence=0.85,
        evidence=["Pattern matches common assignment error", "Similar issues in codebase"]
    ))
    reasoning_chain_1.calculate_overall_confidence()
    
    result_1 = AnalysisResult(
        bug_location="src/utils/counter.py:42",
        root_cause="Variable 'count' is assigned constant value 0 instead of incrementing",
        fix_suggestion="Change 'count = 0' to 'count += 1' on line 42",
        confidence=0.88,
        reasoning_chain=reasoning_chain_1,
        supporting_evidence=[
            "Assignment pattern analysis shows constant assignment",
            "Variable usage context indicates increment operation expected",
            "Similar pattern found in 3 other functions"
        ]
    )
    
    # Result 2: Moderate confidence, broader scope
    reasoning_chain_2 = ReasoningChain()
    reasoning_chain_2.add_step(ReasoningStep(
        step_number=1,
        description="API parameter validation analysis",
        confidence=0.7,
        evidence=["Missing parameter validation in API endpoint", "Potential null pointer access"]
    ))
    reasoning_chain_2.add_step(ReasoningStep(
        step_number=2,
        description="Impact assessment of missing validation",
        confidence=0.65,
        evidence=["Could cause runtime errors", "Security implications identified"]
    ))
    reasoning_chain_2.calculate_overall_confidence()
    
    result_2 = AnalysisResult(
        bug_location="src/api/handlers.py:process_request method",
        root_cause="Missing parameter validation allows null values to propagate",
        fix_suggestion="Add parameter validation at method entry and return appropriate error responses",
        confidence=0.72,
        reasoning_chain=reasoning_chain_2,
        supporting_evidence=[
            "API endpoint accepts null parameters without validation",
            "Downstream code assumes non-null values",
            "Error logs show NullPointerException occurrences"
        ]
    )
    
    # Result 3: Lower confidence, complex issue
    reasoning_chain_3 = ReasoningChain()
    reasoning_chain_3.add_step(ReasoningStep(
        step_number=1,
        description="Performance bottleneck analysis",
        confidence=0.6,
        evidence=["Nested loop structure identified", "O(n²) complexity detected"]
    ))
    reasoning_chain_3.calculate_overall_confidence()
    
    result_3 = AnalysisResult(
        bug_location="src/algorithms/search.py:find_duplicates method",
        root_cause="Inefficient nested loop algorithm causing performance degradation",
        fix_suggestion="Refactor to use hash-based approach or optimize loop structure",
        confidence=0.58,
        reasoning_chain=reasoning_chain_3,
        supporting_evidence=[
            "Nested loop creates O(n²) time complexity",
            "Performance tests show degradation with large datasets"
        ]
    )
    
    return [result_1, result_2, result_3]


def create_sample_context() -> ContextWindow:
    """Create sample context window for analysis."""
    return ContextWindow(
        target_code="""
def process_items(items):
    count = 0  # Line 42 - potential issue here
    for item in items:
        if item.is_valid():
            # count should be incremented here
            process_item(item)
    return count
""",
        related_functions=["process_item", "validate_item", "count_valid_items"],
        class_hierarchy={"ItemProcessor": ["BaseProcessor"], "Item": []},
        module_dependencies=["utils", "validators", "processors"],
        domain_concepts=["item processing", "validation", "counting"]
    )


def create_sample_bug_type() -> BugType:
    """Create sample bug type classification."""
    return BugType(
        category=BugCategory.LOGIC_ERROR,
        subcategory="assignment_error",
        confidence=0.85,
        characteristics=["variable_assignment", "increment_operation", "counter_logic"]
    )


async def demonstrate_result_sorting():
    """Demonstrate result sorting functionality."""
    print("=== Result Sorting and Output Engine Demo ===\n")
    
    # Initialize components
    config = AdvancedAnalysisConfig()
    result_sorter = ResultSorter(config)
    
    # Create sample data
    analysis_results = create_sample_analysis_results()
    context = create_sample_context()
    bug_type = create_sample_bug_type()
    issue_description = "Counter variable not incrementing properly in item processing loop"
    
    print(f"Created {len(analysis_results)} sample analysis results")
    print("Original order (by creation):")
    for i, result in enumerate(analysis_results):
        print(f"  {i+1}. {result.bug_location} (confidence: {result.confidence:.2f})")
    print()
    
    # Sort candidates using different criteria
    print("=== Sorting Results ===")
    
    # Default sorting (all criteria)
    ranked_results = result_sorter.sort_candidates(analysis_results, context)
    
    print("Ranked results (all criteria):")
    for result in ranked_results:
        print(f"  Rank {result.rank}: {result.analysis.bug_location}")
        print(f"    Confidence: {result.analysis.confidence:.2f}")
        print(f"    Composite Score: {result.composite_score:.3f}")
        print(f"    Impact Score: {result.impact_metrics.calculate_overall_impact():.3f}")
        print(f"    Quality Score: {result.quality_metrics.calculate_overall_quality():.3f}")
        print(f"    Rationale: {result.ranking_rationale}")
        print()
    
    # Sort by confidence only
    confidence_ranked = result_sorter.sort_candidates(
        analysis_results, context, [SortingCriteria.CONFIDENCE]
    )
    
    print("Ranked by confidence only:")
    for result in confidence_ranked:
        print(f"  Rank {result.rank}: {result.analysis.bug_location} "
              f"(confidence: {result.analysis.confidence:.2f}, "
              f"score: {result.composite_score:.3f})")
    print()
    
    # Demonstrate output formatting
    print("=== Output Formatting ===")
    
    # Detailed format
    formatted_output = result_sorter.format_output(
        ranked_results, issue_description, bug_type, OutputFormat.DETAILED
    )
    
    print("Formatted output summary:")
    print(f"Title: {formatted_output.title}")
    print(f"Summary: {formatted_output.summary}")
    print(f"Number of results: {len(formatted_output.ranked_results)}")
    print(f"Evidence chains: {len(formatted_output.evidence_chains)}")
    print()
    
    # JSON format
    json_output = formatted_output.to_json()
    print("JSON output (first 200 characters):")
    print(json_output[:200] + "..." if len(json_output) > 200 else json_output)
    print()
    
    # Markdown format
    markdown_output = formatted_output.to_markdown()
    print("Markdown output (first 500 characters):")
    print(markdown_output[:500] + "..." if len(markdown_output) > 500 else markdown_output)
    print()
    
    # Demonstrate result validation
    print("=== Result Validation ===")
    
    validation_report = result_sorter.validate_results(ranked_results)
    
    print("Validation Report:")
    print(f"  Valid: {validation_report['is_valid']}")
    print(f"  Issues: {len(validation_report['issues'])}")
    print(f"  Warnings: {len(validation_report['warnings'])}")
    
    if validation_report['issues']:
        print("  Issues found:")
        for issue in validation_report['issues']:
            print(f"    - {issue}")
    
    if validation_report['warnings']:
        print("  Warnings:")
        for warning in validation_report['warnings']:
            print(f"    - {warning}")
    
    if validation_report['quality_scores']:
        print("  Quality Scores:")
        for metric, value in validation_report['quality_scores'].items():
            print(f"    - {metric}: {value:.3f}")
    
    if validation_report['recommendations']:
        print("  Recommendations:")
        for rec in validation_report['recommendations']:
            print(f"    - {rec}")
    print()
    
    # Demonstrate evidence chain building
    print("=== Evidence Chain Display ===")
    
    top_result = ranked_results[0]
    evidence_display = result_sorter.build_evidence_chain_display(top_result.analysis)
    
    print("Evidence chain for top result:")
    print(evidence_display)
    print()
    
    # Demonstrate different sorting criteria combinations
    print("=== Different Sorting Strategies ===")
    
    strategies = [
        ([SortingCriteria.CONFIDENCE], "Confidence Only"),
        ([SortingCriteria.CODE_IMPACT], "Code Impact Only"),
        ([SortingCriteria.EVIDENCE_STRENGTH], "Evidence Strength Only"),
        ([SortingCriteria.CONFIDENCE, SortingCriteria.CODE_IMPACT], "Confidence + Impact"),
        ([SortingCriteria.REASONING_QUALITY, SortingCriteria.EVIDENCE_STRENGTH], "Quality + Evidence")
    ]
    
    for criteria, name in strategies:
        strategy_results = result_sorter.sort_candidates(analysis_results, context, criteria)
        print(f"{name}:")
        for result in strategy_results:
            print(f"  {result.rank}. {result.analysis.bug_location} "
                  f"(score: {result.composite_score:.3f})")
        print()


def demonstrate_impact_analysis():
    """Demonstrate code impact analysis functionality."""
    print("=== Code Impact Analysis Demo ===\n")
    
    config = AdvancedAnalysisConfig()
    result_sorter = ResultSorter(config)
    context = create_sample_context()
    
    # Create different types of fixes to show impact analysis
    fixes = [
        AnalysisResult(
            bug_location="line 42",
            root_cause="Simple assignment error",
            fix_suggestion="Change count = 0 to count += 1",
            confidence=0.9,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["Simple fix"]
        ),
        AnalysisResult(
            bug_location="entire class",
            root_cause="Architecture issue",
            fix_suggestion="Refactor the entire class structure and redesign the API",
            confidence=0.7,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["Major refactor needed"]
        ),
        AnalysisResult(
            bug_location="method signature",
            root_cause="Missing parameter",
            fix_suggestion="Add new method parameter and update all callers",
            confidence=0.8,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["API change required"]
        )
    ]
    
    print("Impact analysis for different fix types:")
    for i, fix in enumerate(fixes):
        impact_metrics = result_sorter._analyze_code_impact(fix, context)
        
        print(f"\nFix {i+1}: {fix.fix_suggestion}")
        print(f"  Lines affected: {impact_metrics.lines_affected}")
        print(f"  Functions affected: {impact_metrics.functions_affected}")
        print(f"  Risk score: {impact_metrics.risk_score:.2f}")
        print(f"  Implementation effort: {impact_metrics.implementation_effort:.2f}")
        print(f"  Dependency impact: {impact_metrics.dependency_impact:.2f}")
        print(f"  Overall impact: {impact_metrics.calculate_overall_impact():.3f}")


def demonstrate_quality_assessment():
    """Demonstrate quality assessment functionality."""
    print("=== Quality Assessment Demo ===\n")
    
    config = AdvancedAnalysisConfig()
    result_sorter = ResultSorter(config)
    
    analysis_results = create_sample_analysis_results()
    
    print("Quality assessment for each result:")
    for i, result in enumerate(analysis_results):
        quality_metrics = result_sorter._assess_quality(result)
        
        print(f"\nResult {i+1}: {result.bug_location}")
        print(f"  Reasoning completeness: {quality_metrics.reasoning_completeness:.2f}")
        print(f"  Evidence strength: {quality_metrics.evidence_strength:.2f}")
        print(f"  Consistency score: {quality_metrics.consistency_score:.2f}")
        print(f"  Specificity score: {quality_metrics.specificity_score:.2f}")
        print(f"  Actionability score: {quality_metrics.actionability_score:.2f}")
        print(f"  Overall quality: {quality_metrics.calculate_overall_quality():.3f}")


async def main():
    """Run all demonstrations."""
    try:
        await demonstrate_result_sorting()
        demonstrate_impact_analysis()
        demonstrate_quality_assessment()
        
        print("=== Demo Complete ===")
        print("The Result Sorting and Output Engine provides:")
        print("- Comprehensive candidate solution ranking")
        print("- Multiple output formats (detailed, JSON, Markdown)")
        print("- Code impact analysis and quality assessment")
        print("- Result validation and quality checks")
        print("- Evidence chain construction and display")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())