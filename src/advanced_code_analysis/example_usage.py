#!/usr/bin/env python3
"""
Example usage of the Advanced Code Analysis system.

This script demonstrates how to use the core components of the
Advanced Code Analysis system for bug detection and analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_code_analysis import (
    AdvancedAnalysisConfig,
    LLMInterface,
    BugType,
    BugCategory,
    ContextWindow,
    AnalysisResult,
    PromptTemplate
)


async def example_bug_analysis():
    """Example of analyzing a simple bug using the system."""
    print("🔍 Advanced Code Analysis Example")
    print("=" * 50)
    
    # 1. Setup configuration
    print("1. Setting up configuration...")
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"  # Use mock for demo
    config.analysis.confidence_threshold = 0.7
    
    # 2. Create LLM interface
    print("2. Creating LLM interface...")
    llm = LLMInterface(config.llm)
    
    # Set up mock responses for demonstration
    mock_responses = [
        """I've analyzed the code and found a logic error:

**Bug Type**: Logic Error - Assignment Issue
**Location**: Line 2 in function `calculate_total`
**Problem**: The variable `total` is assigned a constant value `100` instead of calculating the sum of `price` and `tax`.
**Root Cause**: Hardcoded assignment instead of computation
**Fix**: Change `total = 100` to `total = price + tax`
**Confidence**: 0.92""",
        
        """Verification confirms the analysis:
- The function parameters `price` and `tax` are unused
- Static analysis shows no computation performed
- The return value is always constant regardless of inputs
- This is definitely a logic error requiring the suggested fix"""
    ]
    
    llm.provider.set_responses(mock_responses)
    
    # 3. Prepare code context
    print("3. Preparing code context...")
    buggy_code = """
def calculate_total(price, tax):
    total = 100  # BUG: Should be price + tax
    return total

# Usage
result = calculate_total(50, 5)
print(f"Total: {result}")  # Always prints 100
"""
    
    context = ContextWindow(
        target_code=buggy_code,
        related_functions=["calculate_total"],
        domain_concepts=["arithmetic", "function_parameters", "return_values"]
    )
    
    print(f"   Context prepared: {context.token_count} tokens")
    
    # 4. Create analysis prompt
    print("4. Creating analysis prompt...")
    analysis_template = PromptTemplate(
        template_id="bug_analysis_v1",
        content="""Analyze the following code for potential bugs:

Code:
```python
{code}
```

Problem Description: {problem_description}

Please identify:
1. Bug type and category
2. Exact location of the issue
3. Root cause analysis
4. Suggested fix
5. Confidence level (0.0-1.0)

Format your response clearly with sections for each point.""",
        placeholders=["code", "problem_description"]
    )
    
    # 5. Perform initial analysis
    print("5. Performing initial analysis...")
    response = await llm.generate(
        analysis_template,
        template_vars={
            "code": context.target_code,
            "problem_description": "Function returns incorrect result - always returns 100 regardless of input parameters"
        }
    )
    
    print(f"   Analysis completed: {len(response.content)} characters")
    print(f"   Response preview: {response.content[:100]}...")
    
    # 6. Parse and structure results (simplified for demo)
    print("6. Structuring analysis results...")
    
    # In a real implementation, this would use NLP to parse the LLM response
    bug_type = BugType(
        category=BugCategory.LOGIC_ERROR,
        subcategory="assignment_error",
        confidence=0.92,
        characteristics=["hardcoded_value", "unused_parameters"]
    )
    
    # Create structured analysis result
    analysis_result = AnalysisResult(
        bug_location="Line 2, function calculate_total",
        root_cause="Hardcoded assignment instead of computation using function parameters",
        fix_suggestion="Change 'total = 100' to 'total = price + tax'",
        confidence=0.92,
        reasoning_chain=None,  # Would be populated in full implementation
        supporting_evidence=[
            "Parameters price and tax are unused",
            "Function always returns constant value",
            "Static analysis confirms no computation"
        ]
    )
    
    # 7. Verification round (optional)
    print("7. Performing verification...")
    verification_prompt = PromptTemplate(
        template_id="verification_v1",
        content="""Please verify this bug analysis:

Original Code:
{code}

Analysis:
- Bug Type: {bug_type}
- Location: {location}
- Fix: {fix}

Is this analysis correct? Provide additional evidence or corrections.""",
        placeholders=["code", "bug_type", "location", "fix"]
    )
    
    verification_response = await llm.generate(
        verification_prompt,
        template_vars={
            "code": context.target_code,
            "bug_type": bug_type.category.value,
            "location": analysis_result.bug_location,
            "fix": analysis_result.fix_suggestion
        }
    )
    
    print(f"   Verification completed: {len(verification_response.content)} characters")
    
    # 8. Display final results
    print("\n" + "=" * 50)
    print("📋 ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Bug Category: {bug_type.category.value}")
    print(f"Subcategory: {bug_type.subcategory}")
    print(f"Confidence: {bug_type.confidence:.2f}")
    print(f"Location: {analysis_result.bug_location}")
    print(f"Root Cause: {analysis_result.root_cause}")
    print(f"Fix Suggestion: {analysis_result.fix_suggestion}")
    print(f"Overall Confidence: {analysis_result.confidence:.2f}")
    
    print("\nSupporting Evidence:")
    for i, evidence in enumerate(analysis_result.supporting_evidence, 1):
        print(f"  {i}. {evidence}")
    
    print(f"\nToken Usage: {response.usage['total_tokens']} tokens")
    print(f"Analysis Time: {response.response_time:.2f}s")
    
    return analysis_result


async def main():
    """Main demonstration function."""
    try:
        result = await example_bug_analysis()
        print("\n✅ Example completed successfully!")
        print(f"Final confidence: {result.confidence:.2f}")
        return 0
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)