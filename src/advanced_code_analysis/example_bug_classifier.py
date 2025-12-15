"""
Example usage of the BugClassifier for intelligent problem classification.

This script demonstrates how to use the BugClassifier to classify different
types of software bugs and select appropriate analysis strategies.
"""

import asyncio
import json
from pathlib import Path

from .bug_classifier import BugClassifier
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface


async def demonstrate_bug_classification():
    """Demonstrate bug classification with various example issues."""
    
    # Create configuration (using mock provider for demonstration)
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"
    
    # Create LLM interface
    llm = LLMInterface(config.llm)
    
    # Set up mock responses for different bug types
    if hasattr(llm.provider, 'set_responses'):
        mock_responses = [
            # Logic error classification
            '{"category": "LOGIC_ERROR", "subcategory": "conditional_logic", "confidence": 0.85, "characteristics": ["wrong_condition", "if_statement"], "reasoning": "The issue involves incorrect conditional logic in an if statement"}',
            
            # API issue classification  
            '{"category": "API_ISSUE", "subcategory": "parameter_error", "confidence": 0.9, "characteristics": ["wrong_parameters", "api_call"], "reasoning": "The API is being called with incorrect parameter types"}',
            
            # Performance issue classification
            '{"category": "PERFORMANCE", "subcategory": "algorithmic_complexity", "confidence": 0.8, "characteristics": ["inefficient_loop", "O(n^2)"], "reasoning": "The nested loop creates O(n^2) complexity causing performance issues"}',
            
            # Boundary condition classification
            '{"category": "BOUNDARY_CONDITION", "subcategory": "null_check", "confidence": 0.95, "characteristics": ["null_pointer", "missing_validation"], "reasoning": "Missing null check causes null pointer exception"}',
            
            # Strategy selection responses (8 total responses needed)
            '{"selected_strategy": "logic_error_deep_analysis", "reasoning": "Best for conditional logic analysis", "confidence": 0.9}',
            '{"selected_strategy": "api_issue_analysis", "reasoning": "Specialized for API parameter issues", "confidence": 0.85}', 
            '{"selected_strategy": "performance_analysis", "reasoning": "Optimized for algorithmic complexity analysis", "confidence": 0.8}',
            '{"selected_strategy": "boundary_condition_analysis", "reasoning": "Specialized for null checks and validation", "confidence": 0.95}'
        ]
        llm.provider.set_responses(mock_responses)
    
    # Create bug classifier
    classifier = BugClassifier(config, llm)
    
    # Example bug reports to classify
    bug_examples = [
        {
            "title": "Logic Error Example",
            "description": "The function returns True when it should return False. The if condition is checking x > 5 but it should be x >= 5.",
            "code": """
def check_threshold(x):
    if x > 5:  # Bug: should be x >= 5
        return True
    return False
"""
        },
        {
            "title": "API Issue Example", 
            "description": "The REST API call is failing with a 400 error. We're passing an integer but the API expects a string.",
            "code": """
response = requests.post('/api/users', {
    'user_id': 123,  # Bug: should be '123' (string)
    'name': 'John'
})
"""
        },
        {
            "title": "Performance Issue Example",
            "description": "The function is very slow with large datasets. It takes several seconds to process 1000 items.",
            "code": """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):  # Bug: O(n^2) complexity
            if items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates
"""
        },
        {
            "title": "Boundary Condition Example",
            "description": "The application crashes with a NullPointerException when processing empty input.",
            "code": """
def process_data(data):
    return data.strip().upper()  # Bug: no null check for data
"""
        }
    ]
    
    print("🔍 Bug Classification Demonstration")
    print("=" * 50)
    
    for i, example in enumerate(bug_examples, 1):
        print(f"\n📋 Example {i}: {example['title']}")
        print(f"Description: {example['description']}")
        print(f"Code:\n{example['code']}")
        
        # Classify the bug
        try:
            result = await classifier.classify_bug_type(
                example['description'], 
                example['code']
            )
            
            print(f"\n🎯 Classification Result:")
            print(f"  Category: {result.bug_type.category.value}")
            print(f"  Subcategory: {result.bug_type.subcategory}")
            print(f"  Confidence: {result.bug_type.confidence:.2f}")
            print(f"  Characteristics: {', '.join(result.bug_type.characteristics)}")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Processing Time: {result.processing_time:.3f}s")
            
            # Select analysis strategy
            strategy = await classifier.select_analysis_strategy(result.bug_type)
            print(f"\n🔧 Selected Strategy:")
            print(f"  Strategy: {strategy.strategy_name}")
            print(f"  Max Rounds: {strategy.max_rounds}")
            print(f"  Confidence Threshold: {strategy.confidence_threshold}")
            print(f"  Context Requirements: {', '.join(strategy.context_requirements)}")
            
            # Get appropriate prompt template
            template = classifier.get_prompt_template(result.bug_type)
            print(f"\n📝 Prompt Template:")
            print(f"  Template ID: {template.template_id}")
            print(f"  Placeholders: {', '.join(template.placeholders)}")
            
        except Exception as e:
            print(f"❌ Error classifying bug: {e}")
        
        print("-" * 50)
    
    # Show classification statistics
    print(f"\n📊 Classification Statistics:")
    stats = classifier.get_classification_stats()
    print(f"  Total Feedback: {stats['total_feedback']}")
    print(f"  Overall Accuracy: {stats['accuracy']:.2f}")


async def demonstrate_feedback_learning():
    """Demonstrate how the classifier learns from feedback."""
    
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"
    llm = LLMInterface(config.llm)
    classifier = BugClassifier(config, llm)
    
    print("\n🎓 Feedback Learning Demonstration")
    print("=" * 50)
    
    # Simulate some classification feedback
    from .models import ClassificationFeedback, BugType, BugCategory
    
    feedback_examples = [
        ClassificationFeedback(
            original_classification=BugType(
                category=BugCategory.LOGIC_ERROR,
                confidence=0.7
            ),
            correct_classification=BugType(
                category=BugCategory.API_ISSUE,
                confidence=0.9
            ),
            issue_text="API returns 404 error",
            feedback_notes="This was actually an API configuration issue, not logic error"
        ),
        ClassificationFeedback(
            original_classification=BugType(
                category=BugCategory.PERFORMANCE,
                confidence=0.6
            ),
            correct_classification=BugType(
                category=BugCategory.BOUNDARY_CONDITION,
                confidence=0.85
            ),
            issue_text="Slow processing with large files",
            feedback_notes="The real issue was missing input size validation"
        )
    ]
    
    print("Adding feedback to improve classification accuracy...")
    
    for i, feedback in enumerate(feedback_examples, 1):
        classifier.update_classification_model(feedback)
        print(f"  {i}. {feedback.original_classification.category.value} → "
              f"{feedback.correct_classification.category.value}")
        print(f"     Note: {feedback.feedback_notes}")
    
    # Show updated statistics
    stats = classifier.get_classification_stats()
    print(f"\n📊 Updated Statistics:")
    print(f"  Total Feedback: {stats['total_feedback']}")
    print(f"  Overall Accuracy: {stats['accuracy']:.2f}")
    
    if 'category_stats' in stats:
        print(f"  Category Breakdown:")
        for category, category_stats in stats['category_stats'].items():
            print(f"    {category}: {category_stats['correct']}/{category_stats['total']} "
                  f"({category_stats['accuracy']:.2f})")


def demonstrate_template_and_strategy_libraries():
    """Demonstrate the template and strategy libraries."""
    
    print("\n📚 Template and Strategy Libraries")
    print("=" * 50)
    
    # Demonstrate prompt template library
    from .bug_classifier import PromptTemplateLibrary, AnalysisStrategyLibrary
    
    template_lib = PromptTemplateLibrary()
    print("📝 Available Prompt Templates:")
    for template_id, template in template_lib.templates.items():
        print(f"  - {template_id}: {len(template.placeholders)} placeholders")
    
    # Demonstrate strategy library
    strategy_lib = AnalysisStrategyLibrary()
    print(f"\n🔧 Available Analysis Strategies:")
    for strategy_name, strategy in strategy_lib.strategies.items():
        print(f"  - {strategy_name}:")
        print(f"    Max Rounds: {strategy.max_rounds}")
        print(f"    Confidence Threshold: {strategy.confidence_threshold}")
        print(f"    Context Requirements: {len(strategy.context_requirements)}")
    
    # Show strategies by category
    print(f"\n🏷️ Strategies by Bug Category:")
    from .models import BugCategory
    
    for category in BugCategory:
        strategies = strategy_lib.get_strategies_for_category(category)
        strategy_names = [s.strategy_name for s in strategies]
        print(f"  {category.value}: {', '.join(strategy_names)}")


async def main():
    """Main demonstration function."""
    print("🚀 Advanced Code Analysis - Bug Classifier Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        await demonstrate_bug_classification()
        await demonstrate_feedback_learning()
        demonstrate_template_and_strategy_libraries()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"\nThe BugClassifier provides:")
        print(f"  • Intelligent bug type classification using LLM")
        print(f"  • Automatic analysis strategy selection")
        print(f"  • Extensible prompt template library")
        print(f"  • Learning from classification feedback")
        print(f"  • Comprehensive error handling and fallbacks")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())