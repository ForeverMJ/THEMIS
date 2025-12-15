"""
Example usage of the PatternMatcher for detecting predefined bug patterns.

This example demonstrates how to use the PatternMatcher to:
1. Detect common bug patterns in code
2. Match against stored bug patterns
3. Get specialized prompts for different bug categories
4. Adapt to new domains
"""

import asyncio
from pathlib import Path

from src.advanced_code_analysis.pattern_matcher import PatternMatcher, PatternRule, DomainPattern
from src.advanced_code_analysis.models import (
    BugPattern, BugCategory, ContextWindow, DependencyContext, 
    DomainKnowledge, PromptTemplate
)
from src.advanced_code_analysis.config import AdvancedAnalysisConfig, LLMConfig
from src.advanced_code_analysis.llm_interface import LLMInterface


def create_sample_context() -> ContextWindow:
    """Create a sample context window for testing."""
    return ContextWindow(
        target_code="",
        related_functions=["calculate_sum", "process_data", "validate_input"],
        class_hierarchy={"BaseClass": ["DerivedClass1", "DerivedClass2"]},
        module_dependencies=["math", "json", "requests"],
        domain_concepts=["authentication", "validation", "processing"],
        dependency_context=DependencyContext(
            function_signatures={
                "calculate_sum": "def calculate_sum(numbers: List[int]) -> int",
                "process_data": "def process_data(data: Dict[str, Any]) -> bool"
            },
            class_methods={
                "DataProcessor": ["process", "validate", "save"]
            },
            import_statements=["import json", "from typing import List, Dict, Any"]
        ),
        domain_knowledge=DomainKnowledge(
            domain_name="data_processing",
            terminology={"validation": "checking data integrity", "processing": "transforming data"},
            common_patterns=["input validation", "error handling", "data transformation"]
        )
    )


def example_basic_pattern_detection():
    """Example of basic pattern detection using regex rules."""
    print("=== Basic Pattern Detection Example ===")
    
    # Create configuration
    config = AdvancedAnalysisConfig()
    
    # Create pattern matcher (without LLM for this example)
    matcher = PatternMatcher(config)
    
    # Sample code with various bug patterns
    buggy_code = """
def process_user_data(user_id, data):
    # Assignment error: assigning constant instead of computed value
    result = 0
    
    # Self assignment error
    user_id = user_id
    
    # Assignment in comparison (should be ==)
    if user_id = 123:
        print("Admin user")
    
    # Off-by-one error in loop
    for i in range(len(data) + 1):
        print(data[i])  # This will cause IndexError
    
    # Unclosed file handle
    file = open("data.txt", "r")
    content = file.read()
    # Missing file.close() or context manager
    
    return result
"""
    
    # Create context
    context = create_sample_context()
    context.target_code = buggy_code
    
    # Detect patterns
    detected_patterns = matcher.detect_patterns(buggy_code, context)
    
    print(f"Detected {len(detected_patterns)} potential issues:")
    for rule, confidence in detected_patterns:
        print(f"  - {rule.name} (confidence: {confidence:.2f})")
        print(f"    Category: {rule.bug_category.value}")
        print(f"    Description: {rule.description}")
        print()


def example_stored_pattern_matching():
    """Example of matching against stored bug patterns."""
    print("=== Stored Pattern Matching Example ===")
    
    config = AdvancedAnalysisConfig()
    matcher = PatternMatcher(config)
    
    # Create and store some sample bug patterns
    sample_patterns = [
        BugPattern(
            pattern_id="assignment_zero_instead_of_sum",
            problem_signature="variable assigned zero instead of calculated sum",
            code_pattern="result = 0",
            fix_pattern="result = sum(values)",
            success_rate=0.85,
            applicable_domains=["data_processing", "mathematics"]
        ),
        BugPattern(
            pattern_id="missing_file_close",
            problem_signature="file opened but not closed properly",
            code_pattern="file = open(...)",
            fix_pattern="with open(...) as file:",
            success_rate=0.92,
            applicable_domains=["file_io", "general"]
        )
    ]
    
    for pattern in sample_patterns:
        matcher.store_pattern(pattern)
    
    # Test pattern matching
    issue_descriptions = [
        "The result variable is always zero instead of the calculated sum",
        "File handle not closed causing resource leak",
        "Database connection timeout error"
    ]
    
    for issue in issue_descriptions:
        matched_patterns = matcher.match_bug_patterns(issue, "")
        print(f"Issue: {issue}")
        print(f"Matched patterns: {len(matched_patterns)}")
        for pattern in matched_patterns:
            print(f"  - {pattern.pattern_id} (success rate: {pattern.success_rate:.2f})")
        print()


def example_specialized_prompts():
    """Example of getting specialized prompts for different bug categories."""
    print("=== Specialized Prompts Example ===")
    
    config = AdvancedAnalysisConfig()
    matcher = PatternMatcher(config)
    
    # Sample code with assignment error
    code_with_assignment_error = """
def calculate_total(items):
    total = 0  # This should be calculated, not hardcoded
    for item in items:
        # Missing: total += item.price
        pass
    return total
"""
    
    # Get specialized prompt for logic errors
    prompt = matcher.get_specialized_prompt(
        BugCategory.LOGIC_ERROR,
        code=code_with_assignment_error,
        context="Function should calculate sum of item prices"
    )
    
    if prompt:
        print("Specialized prompt for LOGIC_ERROR:")
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    else:
        print("No specialized prompt available for LOGIC_ERROR")
    
    print()
    
    # Get specialized prompt for parameter errors
    code_with_param_error = """
def process_data(data, format_type, validate=True):
    pass

# Wrong number of parameters
process_data("test_data")  # Missing format_type parameter
"""
    
    prompt = matcher.get_specialized_prompt(
        BugCategory.API_ISSUE,
        code=code_with_param_error,
        function_signatures="def process_data(data, format_type, validate=True)"
    )
    
    if prompt:
        print("Specialized prompt for API_ISSUE:")
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    else:
        print("No specialized prompt available for API_ISSUE")


def example_domain_adaptation():
    """Example of domain adaptation for new code patterns."""
    print("=== Domain Adaptation Example ===")
    
    config = AdvancedAnalysisConfig()
    matcher = PatternMatcher(config)
    
    # Sample code from different domains
    web_code = """
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    response = requests.get('https://api.example.com/users')
    return jsonify(response.json())

class UserController:
    def authenticate(self, token):
        pass
    
    def authorize(self, user, resource):
        pass
"""
    
    data_science_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

class DataProcessor:
    def preprocess(self, dataframe):
        pass
    
    def feature_engineering(self, features):
        pass
"""
    
    # Test domain adaptation
    contexts = [
        (web_code, "Web Development Code"),
        (data_science_code, "Data Science Code")
    ]
    
    for code, description in contexts:
        print(f"\n{description}:")
        context = create_sample_context()
        context.target_code = code
        
        domain_pattern = matcher.adapt_to_domain(code, context)
        
        print(f"  Domain: {domain_pattern.domain_name}")
        print(f"  Keywords: {', '.join(list(domain_pattern.keywords)[:5])}")
        print(f"  Common functions: {', '.join(list(domain_pattern.common_functions)[:5])}")
        print(f"  Typical imports: {', '.join(list(domain_pattern.typical_imports))}")


async def example_semantic_pattern_detection():
    """Example of semantic pattern detection using LLM."""
    print("=== Semantic Pattern Detection Example ===")
    
    # Create configuration with mock LLM for testing
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"
    
    llm_interface = LLMInterface(config.llm)
    
    # Set up mock responses
    if hasattr(llm_interface.provider, 'set_responses'):
        mock_response = '''
{
  "patterns": [
    {
      "type": "assignment_error",
      "confidence": 0.8,
      "explanation": "Variable 'result' assigned constant 0 instead of computed sum",
      "lines": [3]
    },
    {
      "type": "resource_error", 
      "confidence": 0.9,
      "explanation": "File opened but not properly closed with context manager",
      "lines": [8]
    }
  ]
}
'''
        llm_interface.provider.set_responses([mock_response])
    
    # Create pattern matcher with LLM
    matcher = PatternMatcher(config, llm_interface)
    
    # Sample code for semantic analysis
    code = """
def process_file(filename):
    result = 0  # Should be calculated
    
    file = open(filename, 'r')
    data = file.read()
    # Missing file.close()
    
    return result
"""
    
    context = create_sample_context()
    context.target_code = code
    
    # Detect patterns (this will use both regex and semantic detection)
    detected_patterns = matcher.detect_patterns(code, context)
    
    print(f"Detected {len(detected_patterns)} patterns:")
    for rule, confidence in detected_patterns:
        print(f"  - {rule.name} (confidence: {confidence:.2f})")
        print(f"    Category: {rule.bug_category.value}")
        print(f"    Description: {rule.description}")


def example_pattern_guidance():
    """Example of creating pattern guidance from detected patterns."""
    print("=== Pattern Guidance Example ===")
    
    config = AdvancedAnalysisConfig()
    matcher = PatternMatcher(config)
    
    # Create sample matched patterns
    matched_patterns = [
        BugPattern(
            pattern_id="file_not_closed",
            problem_signature="file handle not properly closed",
            code_pattern="file = open(...)",
            fix_pattern="with open(...) as file:",
            success_rate=0.9,
            applicable_domains=["file_io"]
        )
    ]
    
    # Create sample detected rules
    detected_rules = [
        (PatternRule(
            rule_id="unclosed_file",
            name="Unclosed File Handle",
            description="File opened but not properly closed",
            pattern_regex="",
            bug_category=BugCategory.RESOURCE_MANAGEMENT
        ), 0.85)
    ]
    
    # Create guidance
    guidance = matcher.create_pattern_guidance(matched_patterns, detected_rules)
    
    print(f"Pattern Guidance:")
    print(f"  Confidence: {guidance.confidence:.2f}")
    print(f"  Suggested approach: {guidance.suggested_approach}")
    print(f"  Relevant context: {', '.join(guidance.relevant_context)}")
    print(f"  Matched patterns: {len(guidance.matched_patterns)}")


def main():
    """Run all examples."""
    print("Pattern Matcher Examples")
    print("=" * 50)
    
    example_basic_pattern_detection()
    print("\n" + "=" * 50 + "\n")
    
    example_stored_pattern_matching()
    print("\n" + "=" * 50 + "\n")
    
    example_specialized_prompts()
    print("\n" + "=" * 50 + "\n")
    
    example_domain_adaptation()
    print("\n" + "=" * 50 + "\n")
    
    # Run async example
    asyncio.run(example_semantic_pattern_detection())
    print("\n" + "=" * 50 + "\n")
    
    example_pattern_guidance()


if __name__ == "__main__":
    main()