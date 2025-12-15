"""
Example usage of the MultiRoundReasoner class.

This module demonstrates how to use the multi-round reasoning engine
for iterative code analysis with convergence strategies, self-verification,
and conflict resolution.
"""

import asyncio
import json
from typing import List

from .multi_round_reasoner import MultiRoundReasoner
from .models import (
    BugType, BugCategory, AnalysisStrategy, PromptTemplate,
    ContextWindow, AnalysisResult, Conflict, ReasoningChain, ReasoningStep
)
from .llm_interface import LLMInterface, MockProvider
from .config import AdvancedAnalysisConfig, LLMConfig, AnalysisConfig


async def example_basic_multi_round_reasoning():
    """Example of basic multi-round reasoning."""
    print("=== Basic Multi-Round Reasoning Example ===")
    
    # Setup configuration
    config = AdvancedAnalysisConfig(
        llm=LLMConfig(provider="mock"),
        analysis=AnalysisConfig(
            max_reasoning_rounds=3,
            confidence_threshold=0.8,
            enable_multi_round_reasoning=True
        )
    )
    
    # Create LLM interface with mock responses
    llm_interface = LLMInterface(config.llm)
    
    # Set up mock responses for multi-round reasoning
    if isinstance(llm_interface.provider, MockProvider):
        mock_responses = [
            # Initial analysis response
            json.dumps({
                "hypothesis": "Assignment operator used instead of comparison",
                "bug_location": "line 5: if x = 10",
                "fix_suggestion": "Change '=' to '==' for comparison",
                "confidence": 0.7,
                "evidence": ["syntax suggests assignment", "context suggests comparison needed"]
            }),
            # Verification response
            json.dumps({
                "is_consistent": True,
                "consistency_score": 0.85,
                "issues_found": [],
                "alternative_explanations": ["could be intentional assignment"],
                "missing_considerations": ["check if assignment was intended"],
                "confidence_adjustment": 0.05
            }),
            # Refinement response
            json.dumps({
                "refined_hypothesis": "Unintentional assignment in conditional",
                "updated_location": "line 5: if x = 10",
                "improved_fix": "Change '=' to '==' and add parentheses for clarity: if (x == 10)",
                "confidence": 0.85,
                "new_evidence": ["conditional context confirms comparison intent"],
                "addressed_issues": ["clarified intent vs assignment"]
            })
        ]
        llm_interface.provider.set_responses(mock_responses)
    
    # Create reasoner
    reasoner = MultiRoundReasoner(llm_interface, config)
    
    # Setup problem context
    issue_description = """
    The code has a bug in the conditional statement. The program is supposed to check
    if x equals 10, but it's not working as expected.
    """
    
    context = ContextWindow(
        target_code="""
def check_value(x):
    if x = 10:  # Bug: assignment instead of comparison
        return "Found ten"
    return "Not ten"
""",
        related_functions=["validate_input", "process_result"],
        class_hierarchy={"ValueChecker": ["BaseChecker"]},
        module_dependencies=["typing", "logging"]
    )
    
    bug_type = BugType(
        category=BugCategory.LOGIC_ERROR,
        subcategory="assignment_error",
        confidence=0.9,
        characteristics=["conditional", "assignment", "syntax"]
    )
    
    strategy = AnalysisStrategy(
        strategy_name="logic_error_analysis",
        prompt_template=PromptTemplate(
            template_id="logic_analysis",
            content="Analyze this logic error: {issue}"
        ),
        context_requirements=["target_code", "related_functions"],
        verification_steps=["syntax_check", "logic_check"],
        max_rounds=3,
        confidence_threshold=0.8
    )
    
    # Perform multi-round reasoning
    print("Starting multi-round reasoning...")
    resolved_analysis = await reasoner.multi_round_reasoning(
        issue_description, context, bug_type, strategy, "confidence_based"
    )
    
    # Display results
    print(f"\nResolution Method: {resolved_analysis.resolution_method}")
    print(f"Final Confidence: {resolved_analysis.final_result.confidence:.2f}")
    print(f"Bug Location: {resolved_analysis.final_result.bug_location}")
    print(f"Root Cause: {resolved_analysis.final_result.root_cause}")
    print(f"Fix Suggestion: {resolved_analysis.final_result.fix_suggestion}")
    print(f"Reasoning Rounds: {len(resolved_analysis.final_result.reasoning_chain.steps)}")
    
    # Show reasoning chain
    print("\nReasoning Chain:")
    for i, step in enumerate(resolved_analysis.final_result.reasoning_chain.steps):
        print(f"  Step {step.step_number}: {step.description}")
        print(f"    Confidence: {step.confidence:.2f}")
        print(f"    Evidence: {', '.join(step.evidence[:2])}...")
    
    return resolved_analysis


async def example_conflict_resolution():
    """Example of conflict resolution between different analyses."""
    print("\n=== Conflict Resolution Example ===")
    
    # Setup configuration
    config = AdvancedAnalysisConfig(
        llm=LLMConfig(provider="mock"),
        analysis=AnalysisConfig(enable_conflict_detection=True)
    )
    
    # Create LLM interface
    llm_interface = LLMInterface(config.llm)
    
    # Set up mock response for conflict resolution
    if isinstance(llm_interface.provider, MockProvider):
        resolution_response = json.dumps({
            "chosen_analysis": "Analysis 2 with type error explanation",
            "reasoning": "Analysis 2 provides more specific error type and better fix",
            "synthesized_elements": [
                "combines syntax awareness from Analysis 1",
                "uses type-specific fix from Analysis 2"
            ],
            "confidence": 0.88,
            "resolution_method": "evidence_weighted"
        })
        llm_interface.provider.set_responses([resolution_response])
    
    # Create reasoner
    reasoner = MultiRoundReasoner(llm_interface, config)
    
    # Create conflicting analyses
    analysis1 = AnalysisResult(
        bug_location="line 3: x = '10'",
        root_cause="String assignment instead of integer",
        fix_suggestion="Change '10' to 10",
        confidence=0.75,
        reasoning_chain=ReasoningChain(),
        supporting_evidence=["string literal detected", "type mismatch likely"]
    )
    
    analysis2 = AnalysisResult(
        bug_location="line 3: x = '10'", 
        root_cause="Type error - string used where integer expected",
        fix_suggestion="Convert string to int: x = int('10')",
        confidence=0.82,
        reasoning_chain=ReasoningChain(),
        supporting_evidence=["type analysis shows string", "context requires integer"]
    )
    
    # Create conflict
    conflict = Conflict(
        conflict_type="root_cause_disagreement",
        description="Different explanations for the same bug location",
        conflicting_analyses=[analysis1, analysis2],
        resolution_strategy="llm_mediated"
    )
    
    # Resolve conflicts
    print("Resolving conflicts between analyses...")
    resolved_analysis = await reasoner.resolve_conflicts(
        [conflict], 
        "Variable has wrong type causing comparison issues"
    )
    
    # Display resolution results
    print(f"\nResolution Method: {resolved_analysis.resolution_method}")
    print(f"Resolution Confidence: {resolved_analysis.resolution_confidence:.2f}")
    print(f"Final Bug Location: {resolved_analysis.final_result.bug_location}")
    print(f"Final Root Cause: {resolved_analysis.final_result.root_cause}")
    print(f"Final Fix: {resolved_analysis.final_result.fix_suggestion}")
    print(f"Discarded Alternatives: {len(resolved_analysis.discarded_alternatives)}")
    
    # Show synthesized evidence
    print("\nSynthesized Evidence:")
    for evidence in resolved_analysis.final_result.supporting_evidence:
        if "combines" in evidence or "uses" in evidence:
            print(f"  - {evidence}")
    
    return resolved_analysis


async def example_convergence_strategies():
    """Example of different convergence strategies."""
    print("\n=== Convergence Strategies Example ===")
    
    # Setup configuration
    config = AdvancedAnalysisConfig(
        llm=LLMConfig(provider="mock"),
        analysis=AnalysisConfig(
            max_reasoning_rounds=5,
            confidence_threshold=0.8
        )
    )
    
    # Create LLM interface
    llm_interface = LLMInterface(config.llm)
    
    # Create reasoner
    reasoner = MultiRoundReasoner(llm_interface, config)
    
    # Test different convergence strategies
    strategies = ["confidence_based", "evidence_based", "conservative"]
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name} convergence strategy:")
        
        strategy_config = reasoner.convergence_strategies[strategy_name]
        print(f"  Confidence Threshold: {strategy_config.confidence_threshold}")
        print(f"  Max Rounds: {strategy_config.max_rounds}")
        print(f"  Improvement Threshold: {strategy_config.improvement_threshold}")
        print(f"  Stability Rounds: {strategy_config.stability_rounds}")
        
        # Simulate convergence check with mock rounds
        from .multi_round_reasoner import RoundResult
        from unittest.mock import Mock
        
        # Create mock rounds with increasing confidence
        mock_rounds = []
        confidences = [0.5, 0.65, 0.75, 0.82, 0.83]
        
        for i, conf in enumerate(confidences):
            mock_analysis = Mock()
            mock_analysis.confidence = conf
            
            round_result = RoundResult(
                round_number=i + 1,
                analysis=mock_analysis,
                confidence_change=conf - (confidences[i-1] if i > 0 else 0.0)
            )
            mock_rounds.append(round_result)
        
        # Test convergence
        converged = reasoner._check_convergence(mock_rounds, strategy_config)
        print(f"  Converged with {len(mock_rounds)} rounds: {converged}")


async def example_evidence_chain_building():
    """Example of building comprehensive evidence chains."""
    print("\n=== Evidence Chain Building Example ===")
    
    # Setup
    config = AdvancedAnalysisConfig(llm=LLMConfig(provider="mock"))
    llm_interface = LLMInterface(config.llm)
    reasoner = MultiRoundReasoner(llm_interface, config)
    
    # Create analysis with detailed reasoning chain
    reasoning_chain = ReasoningChain()
    
    # Add multiple reasoning steps
    steps_data = [
        {
            "description": "Initial syntax analysis",
            "confidence": 0.6,
            "evidence": ["assignment operator detected", "conditional context"]
        },
        {
            "description": "Semantic analysis",
            "confidence": 0.75,
            "evidence": ["comparison intent inferred", "variable usage pattern"]
        },
        {
            "description": "Context validation",
            "confidence": 0.85,
            "evidence": ["surrounding code confirms comparison", "function purpose analysis"]
        }
    ]
    
    for i, step_data in enumerate(steps_data):
        step = ReasoningStep(
            step_number=i + 1,
            description=step_data["description"],
            confidence=step_data["confidence"],
            evidence=step_data["evidence"],
            input_data={"step": i + 1},
            output_data={"analysis": f"step_{i+1}_result"}
        )
        reasoning_chain.add_step(step)
    
    # Create analysis result
    analysis = AnalysisResult(
        bug_location="line 5: if x = 10",
        root_cause="Assignment operator in conditional expression",
        fix_suggestion="Replace '=' with '==' for comparison",
        confidence=0.85,
        reasoning_chain=reasoning_chain,
        supporting_evidence=[
            "syntax error pattern matches assignment-for-comparison",
            "code context strongly suggests comparison intent",
            "similar patterns found in codebase corrections"
        ]
    )
    
    # Build evidence chain
    evidence_chain = reasoner.build_evidence_chain(analysis)
    
    # Display evidence chain
    print("Evidence Chain:")
    print(f"  Total Evidence Items: {len(evidence_chain.evidence_items)}")
    print(f"  Source Locations: {len(evidence_chain.source_locations)}")
    print(f"  Reasoning Path Steps: {len(evidence_chain.reasoning_path)}")
    
    print("\nEvidence Items:")
    for i, (evidence, source, weight) in enumerate(zip(
        evidence_chain.evidence_items,
        evidence_chain.source_locations, 
        evidence_chain.confidence_weights
    )):
        print(f"  {i+1}. {evidence}")
        print(f"     Source: {source}")
        print(f"     Weight: {weight:.2f}")
    
    print("\nReasoning Path:")
    for i, path_item in enumerate(evidence_chain.reasoning_path):
        print(f"  {i+1}. {path_item}")
    
    return evidence_chain


async def main():
    """Run all examples."""
    print("Multi-Round Reasoner Examples")
    print("=" * 50)
    
    try:
        # Run examples
        await example_basic_multi_round_reasoning()
        await example_conflict_resolution()
        await example_convergence_strategies()
        await example_evidence_chain_building()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())