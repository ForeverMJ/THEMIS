"""
Example usage of the Conflict Detection and Handling Engine.

This module demonstrates how to use the ConflictDetector class to detect
and resolve conflicts between AST analysis and LLM judgments, handle
multi-strategy reasoning, and collect adaptive context for low confidence
scenarios.
"""

import asyncio
import logging
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_code_analysis.conflict_detector import ConflictDetector, ConflictType, ConflictSeverity
from advanced_code_analysis.models import (
    AnalysisResult, ReasoningChain, ReasoningStep, ContextWindow,
    BugType, BugCategory, AnalysisStrategy, PromptTemplate
)
from advanced_code_analysis.llm_interface import LLMInterface
from advanced_code_analysis.enhanced_ast_analyzer import EnhancedASTAnalyzer
from advanced_code_analysis.multi_round_reasoner import MultiRoundReasoner
from advanced_code_analysis.config import AdvancedAnalysisConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_conflict_detection():
    """Demonstrate basic conflict detection functionality."""
    print("=== Conflict Detection Demo ===")
    
    # Create mock LLM interface for demonstration
    class MockLLMInterface:
        async def generate(self, prompt, **kwargs):
            from .llm_interface import LLMResponse
            return LLMResponse(
                content='{"reasoning": "Mock resolution", "confidence": 0.8, "resolved_location": "line 42"}',
                usage={"total_tokens": 100},
                model="mock-model",
                finish_reason="stop",
                response_time=0.1
            )
    
    # Create mock config
    class MockConfig:
        def __init__(self):
            self.analysis = MockAnalysisConfig()
    
    class MockAnalysisConfig:
        def __init__(self):
            self.confidence_threshold = 0.7
            self.max_reasoning_rounds = 3
    
    # Initialize components
    llm_interface = MockLLMInterface()
    config = MockConfig()
    conflict_detector = ConflictDetector(llm_interface, config)
    
    # Create sample analyses with conflicts
    llm_analysis = create_sample_llm_analysis()
    ast_analysis = create_sample_ast_analysis()
    
    print(f"LLM Analysis: {llm_analysis.bug_location} - {llm_analysis.root_cause}")
    print(f"AST Analysis: {ast_analysis.bug_location} - {ast_analysis.root_cause}")
    
    # Detect conflicts
    conflicts = await conflict_detector.detect_conflicts(
        llm_analysis=llm_analysis,
        ast_analysis=ast_analysis,
        code=SAMPLE_CODE,
        context=create_sample_context()
    )
    
    print(f"\nDetected {len(conflicts)} conflicts:")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  {i}. {conflict.conflict_type.value} ({conflict.severity.value})")
        print(f"     {conflict.description}")
        print(f"     Confidence: {conflict.detection_confidence:.2f}")
    
    return conflicts


async def demonstrate_conflict_resolution():
    """Demonstrate conflict resolution strategies."""
    print("\n=== Conflict Resolution Demo ===")
    
    # Setup (reuse from detection demo)
    class MockLLMInterface:
        async def generate(self, prompt, **kwargs):
            from .llm_interface import LLMResponse
            return LLMResponse(
                content='{"reasoning": "Resolved through mediation", "confidence": 0.85, "resolved_location": "line 42", "synthesized_analysis": "Combined analysis result"}',
                usage={"total_tokens": 150},
                model="mock-model",
                finish_reason="stop",
                response_time=0.1
            )
    
    class MockConfig:
        def __init__(self):
            self.analysis = MockAnalysisConfig()
    
    class MockAnalysisConfig:
        def __init__(self):
            self.confidence_threshold = 0.7
            self.max_reasoning_rounds = 3
    
    llm_interface = MockLLMInterface()
    config = MockConfig()
    conflict_detector = ConflictDetector(llm_interface, config)
    
    # Get conflicts from previous demo
    conflicts = await demonstrate_conflict_detection()
    
    if conflicts:
        print(f"\nResolving {len(conflicts)} conflicts...")
        
        # Handle conflicts
        resolution = await conflict_detector.handle_conflicts(
            conflicts=conflicts,
            original_issue="Sample bug report: Function crashes with null pointer",
            code=SAMPLE_CODE,
            context=create_sample_context()
        )
        
        print(f"Resolution Method: {resolution.resolution_method}")
        print(f"Final Location: {resolution.final_result.bug_location}")
        print(f"Final Cause: {resolution.final_result.root_cause}")
        print(f"Resolution Confidence: {resolution.resolution_confidence:.2f}")
        print(f"Notes: {resolution.resolution_notes}")
        
        return resolution
    else:
        print("No conflicts to resolve")
        return None


async def demonstrate_adaptive_context_collection():
    """Demonstrate adaptive context collection for low confidence scenarios."""
    print("\n=== Adaptive Context Collection Demo ===")
    
    # Setup
    class MockLLMInterface:
        async def generate(self, prompt, **kwargs):
            from .llm_interface import LLMResponse
            return LLMResponse(
                content='{"additional_context": "Extended analysis context"}',
                usage={"total_tokens": 80},
                model="mock-model",
                finish_reason="stop",
                response_time=0.1
            )
    
    class MockConfig:
        def __init__(self):
            self.analysis = MockAnalysisConfig()
    
    class MockAnalysisConfig:
        def __init__(self):
            self.confidence_threshold = 0.7
            self.max_reasoning_rounds = 3
    
    llm_interface = MockLLMInterface()
    config = MockConfig()
    conflict_detector = ConflictDetector(llm_interface, config)
    
    # Create low confidence analysis
    low_confidence_analysis = create_sample_llm_analysis()
    low_confidence_analysis.confidence = 0.4  # Low confidence triggers adaptive collection
    
    print(f"Low confidence analysis: {low_confidence_analysis.confidence:.2f}")
    
    # Collect adaptive context
    adaptive_context = await conflict_detector.collect_adaptive_context(
        analysis=low_confidence_analysis,
        original_context=create_sample_context(),
        code=SAMPLE_CODE
    )
    
    print(f"Adaptive context collected:")
    print(f"  Additional functions: {len(adaptive_context.additional_functions)}")
    print(f"  Extended dependencies: {len(adaptive_context.extended_dependencies)}")
    print(f"  Similar patterns: {len(adaptive_context.similar_code_patterns)}")
    print(f"  Domain info keys: {list(adaptive_context.domain_specific_info.keys())}")
    print(f"  Collection confidence: {adaptive_context.collection_confidence:.2f}")
    
    return adaptive_context


async def demonstrate_multi_strategy_reasoning():
    """Demonstrate multi-strategy reasoning for conflict resolution."""
    print("\n=== Multi-Strategy Reasoning Demo ===")
    
    # Setup
    class MockLLMInterface:
        async def generate(self, prompt, **kwargs):
            from .llm_interface import LLMResponse
            return LLMResponse(
                content='{"strategy_result": "Multi-strategy analysis", "confidence": 0.9}',
                usage={"total_tokens": 120},
                model="mock-model",
                finish_reason="stop",
                response_time=0.1
            )
    
    class MockConfig:
        def __init__(self):
            self.analysis = MockAnalysisConfig()
    
    class MockAnalysisConfig:
        def __init__(self):
            self.confidence_threshold = 0.7
            self.max_reasoning_rounds = 3
    
    llm_interface = MockLLMInterface()
    config = MockConfig()
    conflict_detector = ConflictDetector(llm_interface, config)
    
    # Create conflicts for multi-strategy reasoning
    conflicts = await demonstrate_conflict_detection()
    
    if conflicts:
        print(f"Applying multi-strategy reasoning to {len(conflicts)} conflicts...")
        
        # Apply multiple strategies
        strategy_results = await conflict_detector.multi_strategy_reasoning(
            conflicts=conflicts,
            original_issue="Complex bug requiring multiple analysis approaches",
            context=create_sample_context()
        )
        
        print(f"Generated {len(strategy_results)} strategy results:")
        for i, result in enumerate(strategy_results, 1):
            print(f"  Strategy {i}:")
            print(f"    Location: {result.bug_location}")
            print(f"    Confidence: {result.confidence:.2f}")
            print(f"    Evidence count: {len(result.supporting_evidence)}")
        
        return strategy_results
    else:
        print("No conflicts available for multi-strategy reasoning")
        return []


def create_sample_llm_analysis() -> AnalysisResult:
    """Create a sample LLM analysis result."""
    reasoning_chain = ReasoningChain()
    
    step1 = ReasoningStep(
        step_number=1,
        description="LLM initial analysis",
        confidence=0.8,
        evidence=["Detected null pointer access pattern", "Function lacks null validation"]
    )
    reasoning_chain.add_step(step1)
    
    return AnalysisResult(
        bug_location="line 42 in function process_data",
        root_cause="Null pointer dereference when accessing user.name without validation",
        fix_suggestion="Add null check: if (user != null && user.name != null) before access",
        confidence=0.8,
        reasoning_chain=reasoning_chain,
        supporting_evidence=[
            "LLM identified null pointer access pattern",
            "Missing validation in function signature",
            "Similar pattern found in codebase"
        ]
    )


def create_sample_ast_analysis() -> AnalysisResult:
    """Create a sample AST analysis result."""
    reasoning_chain = ReasoningChain()
    
    step1 = ReasoningStep(
        step_number=1,
        description="AST structural analysis",
        confidence=0.6,
        evidence=["AST detected type inconsistency", "Variable usage before definition"]
    )
    reasoning_chain.add_step(step1)
    
    return AnalysisResult(
        bug_location="line 45 in method validate_user",
        root_cause="Type mismatch between expected string and actual object type",
        fix_suggestion="Convert object to string: str(user.name) or use proper type checking",
        confidence=0.6,
        reasoning_chain=reasoning_chain,
        supporting_evidence=[
            "AST found type inconsistency in assignment",
            "Variable used before proper type validation",
            "Function signature expects different type"
        ]
    )


def create_sample_context() -> ContextWindow:
    """Create a sample context window."""
    return ContextWindow(
        target_code=SAMPLE_CODE,
        related_functions=["validate_user", "process_data", "get_user_info"],
        class_hierarchy={"User": ["BaseUser"], "UserProcessor": []},
        module_dependencies=["typing", "dataclasses", "logging"],
        domain_concepts=["user_validation", "data_processing", "error_handling"]
    )


# Sample code for demonstration
SAMPLE_CODE = '''
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

def process_data(user: User) -> str:
    # Potential null pointer issue here
    return f"Processing {user.name}"

def validate_user(user_data: dict) -> User:
    # Type mismatch potential here
    name = user_data.get("name")
    email = user_data.get("email")
    
    if name and email:
        return User(name, email)
    return None

def get_user_info(user_id: int) -> dict:
    # Simplified user lookup
    users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"}
    }
    return users.get(user_id, {})

# Main processing function
def main():
    user_data = get_user_info(1)
    user = validate_user(user_data)
    result = process_data(user)  # Potential issue if user is None
    print(result)
'''


async def run_all_demonstrations():
    """Run all conflict detector demonstrations."""
    print("🔍 Advanced Code Analysis - Conflict Detection and Handling Demo")
    print("=" * 70)
    
    try:
        # Run all demonstrations
        await demonstrate_conflict_detection()
        await demonstrate_conflict_resolution()
        await demonstrate_adaptive_context_collection()
        await demonstrate_multi_strategy_reasoning()
        
        print("\n" + "=" * 70)
        print("✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logger.exception("Demonstration failed")


if __name__ == "__main__":
    asyncio.run(run_all_demonstrations())