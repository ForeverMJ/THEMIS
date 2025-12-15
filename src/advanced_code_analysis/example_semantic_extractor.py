"""
Example usage of the SemanticExtractor class.

This script demonstrates how to use the semantic information extraction engine
to analyze problem descriptions and extract key technical information.
"""

import asyncio
import json
from .semantic_extractor import SemanticExtractor, InformationType
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface


async def main():
    """Demonstrate SemanticExtractor usage."""
    
    # Create configuration
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"  # Use mock provider for demo
    
    # Create LLM interface
    llm = LLMInterface(config.llm)
    
    # Set up mock responses for demonstration
    if hasattr(llm.provider, 'set_responses'):
        mock_responses = [
            # Response for technical concept extraction
            json.dumps({
                "technical_concepts": [
                    {"term": "IndexError", "confidence": 0.9, "context": "list index out of range"},
                    {"term": "list comprehension", "confidence": 0.7, "context": "data processing"}
                ],
                "function_names": [
                    {"name": "process_items", "confidence": 0.8, "context": "main processing function"},
                    {"name": "validate_data", "confidence": 0.7, "context": "validation step"}
                ],
                "variable_names": [
                    {"name": "item_list", "confidence": 0.8, "context": "input data"},
                    {"name": "index", "confidence": 0.9, "context": "loop variable"}
                ],
                "class_names": [],
                "error_patterns": [
                    {"pattern": "list index out of range", "confidence": 0.9, "type": "IndexError"}
                ],
                "api_calls": [],
                "overall_confidence": 0.85
            }),
            # Response for summary generation
            json.dumps({
                "problem_type": "boundary_condition_error",
                "key_components": ["process_items", "item_list", "index"],
                "technical_concepts": ["IndexError", "list comprehension"],
                "code_elements": ["process_items", "validate_data", "item_list", "index"],
                "error_indicators": ["IndexError", "list index out of range"],
                "confidence_score": 0.85,
                "reasoning": "The error occurs when accessing list elements beyond the valid index range"
            })
        ]
        llm.provider.set_responses(mock_responses)
    
    # Create semantic extractor
    extractor = SemanticExtractor(config, llm)
    
    # Example problem description
    problem_text = """
    The function process_items(item_list) is failing with an IndexError: list index out of range.
    This happens when the loop variable index exceeds the length of item_list.
    The validate_data() function should check the list bounds before processing.
    The error occurs in the list comprehension [item_list[i] for i in range(len(item_list) + 1)].
    """
    
    print("=== Semantic Information Extraction Demo ===")
    print(f"Problem Description:\n{problem_text}\n")
    
    # Extract information
    print("Extracting semantic information...")
    result = await extractor.extract_information(problem_text)
    
    print(f"Extraction completed in {result.processing_time:.2f} seconds")
    print(f"Overall confidence: {result.overall_confidence:.2f}\n")
    
    # Display extracted items by type
    print("=== Extracted Information ===")
    
    for info_type in InformationType:
        items = result.get_items_by_type(info_type)
        if items:
            print(f"\n{info_type.value.replace('_', ' ').title()}:")
            for item in items:
                print(f"  - {item.content} (confidence: {item.confidence:.2f})")
                if item.context:
                    print(f"    Context: {item.context}")
    
    # Display structured summary
    print(f"\n=== Structured Summary ===")
    summary = result.structured_summary
    print(f"Problem Type: {summary.problem_type}")
    print(f"Confidence Score: {summary.confidence_score:.2f}")
    
    if summary.key_components:
        print(f"Key Components: {', '.join(summary.key_components)}")
    
    if summary.technical_concepts:
        print(f"Technical Concepts: {', '.join(summary.technical_concepts)}")
    
    if summary.code_elements:
        print(f"Code Elements: {', '.join(summary.code_elements)}")
    
    if summary.error_indicators:
        print(f"Error Indicators: {', '.join(summary.error_indicators)}")
    
    # Display reasoning chain
    print(f"\n=== Reasoning Chain ===")
    for i, step in enumerate(result.reasoning_chain.steps, 1):
        print(f"Step {i}: {step.description}")
        print(f"  Confidence: {step.confidence:.2f}")
        if step.output_data:
            print(f"  Output: {step.output_data}")
    
    # Demonstrate filtering by confidence
    print(f"\n=== High Confidence Items (>= 0.8) ===")
    high_confidence_items = result.get_high_confidence_items(0.8)
    for item in high_confidence_items:
        print(f"  - {item.info_type.value}: {item.content} ({item.confidence:.2f})")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())