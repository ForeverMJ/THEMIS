"""
Example usage of the Enhanced AST Analysis Engine.

This example demonstrates how to use the EnhancedASTAnalyzer to perform
LLM-guided AST analysis, including error pattern detection, function call
validation, and data flow tracing.
"""

from enhanced_ast_analyzer import (
    EnhancedASTAnalyzer,
    SuspiciousRegion,
    ErrorPattern,
    FunctionCallValidation,
    DataFlowTrace
)
from models import (
    AnalysisResult,
    ReasoningChain,
    ReasoningStep,
    ContextWindow
)


def example_basic_analysis():
    """Example of basic enhanced AST analysis."""
    print("=== Basic Enhanced AST Analysis ===")
    
    # Sample code with various issues
    code = """
def calculate_discount(price, customer_type):
    # Bug 1: Constant assignment that might be wrong
    discount_rate = 10  # Should this be 0.10?
    
    # Bug 2: Assignment instead of comparison
    if customer_type = "premium":  # Should be ==
        discount_rate = 0.15
    elif customer_type == "regular":
        discount_rate = 0.05
    
    # Bug 3: Using undefined variable
    final_price = price * (1 - discount_rate) + tax_amount
    
    return final_price

def process_order(items):
    total = 0
    for item in items:
        # Bug 4: Wrong number of arguments
        discount_price = calculate_discount(item.price)  # Missing customer_type
        total += discount_price
    
    return total
"""
    
    # Initialize analyzer
    analyzer = EnhancedASTAnalyzer()
    
    # 1. Detect error patterns
    print("\n1. Error Pattern Detection:")
    patterns = analyzer.detect_error_patterns(code)
    
    for pattern in patterns:
        print(f"  - {pattern.pattern_type} at {pattern.location}")
        print(f"    Description: {pattern.description}")
        print(f"    Severity: {pattern.severity}, Confidence: {pattern.confidence:.2f}")
        if pattern.suggested_fix:
            print(f"    Suggested fix: {pattern.suggested_fix}")
        print()
    
    # 2. Validate function calls
    print("2. Function Call Validation:")
    validations = analyzer.validate_function_calls(code)
    
    for validation in validations:
        status = "✓ Valid" if validation.is_valid else "✗ Invalid"
        print(f"  - {validation.function_name} at {validation.call_location}: {status}")
        if not validation.is_valid:
            for issue in validation.issues:
                print(f"    Issue: {issue}")
        if validation.expected_signature:
            print(f"    Expected: {validation.expected_signature}")
        print(f"    Actual: {validation.actual_call}")
        print()
    
    # 3. Trace variable data flow
    print("3. Data Flow Tracing:")
    key_variables = ["discount_rate", "tax_amount", "total"]
    traces = analyzer.trace_variable_data_flow(code, key_variables)
    
    for var_name, trace in traces.items():
        print(f"  Variable: {var_name}")
        print(f"    Definitions: {len(trace.definition_points)}")
        print(f"    Usages: {len(trace.usage_points)}")
        print(f"    Modifications: {len(trace.modifications)}")
        
        if trace.potential_issues:
            print("    Issues:")
            for issue in trace.potential_issues:
                print(f"      - {issue}")
        print()


def example_suspicious_region_analysis():
    """Example of analyzing LLM-identified suspicious regions."""
    print("=== Suspicious Region Analysis ===")
    
    code = """
def validate_user_input(user_data):
    # Region 1: Suspicious validation logic
    if user_data.age = 18:  # Assignment instead of comparison
        is_adult = True
    else:
        is_adult = False
    
    # Region 2: Potential security issue
    query = "SELECT * FROM users WHERE id = " + user_data.id  # SQL injection risk
    
    # Region 3: Resource management issue
    file_handle = open(user_data.filename)
    content = file_handle.read()
    # Missing file_handle.close()
    
    return is_adult, content
"""
    
    # Create suspicious regions (as would be identified by LLM)
    regions = [
        SuspiciousRegion(
            start_line=3,
            end_line=3,
            reason="Assignment operator used in if condition",
            confidence=0.9,
            code_snippet="if user_data.age = 18:",
            suggested_focus=["comparison_operator", "conditional_logic"]
        ),
        SuspiciousRegion(
            start_line=8,
            end_line=8,
            reason="String concatenation in SQL query - potential injection",
            confidence=0.8,
            code_snippet='query = "SELECT * FROM users WHERE id = " + user_data.id',
            suggested_focus=["sql_injection", "input_sanitization"]
        ),
        SuspiciousRegion(
            start_line=11,
            end_line=13,
            reason="File opened but not explicitly closed",
            confidence=0.7,
            code_snippet="file_handle = open(user_data.filename)\ncontent = file_handle.read()",
            suggested_focus=["resource_management", "file_handling"]
        )
    ]
    
    # Initialize analyzer
    analyzer = EnhancedASTAnalyzer()
    
    # Analyze suspicious regions
    print("Analyzing suspicious regions:")
    results = analyzer.analyze_suspicious_regions(code, regions)
    
    for i, result in enumerate(results, 1):
        print(f"\nRegion {i} Analysis:")
        print(f"  Location: {result.bug_location}")
        print(f"  Root cause: {result.root_cause}")
        print(f"  Fix suggestion: {result.fix_suggestion}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        print("  Supporting evidence:")
        for evidence in result.supporting_evidence:
            print(f"    - {evidence}")
        
        print("  Reasoning chain:")
        for step in result.reasoning_chain.steps:
            print(f"    Step {step.step_number}: {step.description}")
            print(f"      Confidence: {step.confidence:.2f}")


def example_llm_integration():
    """Example of integrating AST analysis with LLM results."""
    print("=== LLM Integration Example ===")
    
    code = """
def process_payment(amount, card_number):
    # Simulate LLM-identified issue
    fee = 2.50  # LLM suspects this should be percentage
    
    if amount > 100:
        fee = amount * 0.025  # 2.5% fee for large amounts
    
    total = amount + fee
    
    # LLM identified potential issue with card validation
    if len(card_number) = 16:  # Assignment instead of comparison
        return process_card_payment(total, card_number)
    else:
        return {"error": "Invalid card number"}

def process_card_payment(amount, card):
    # Simulate processing
    return {"status": "success", "amount": amount}
"""
    
    # Simulate LLM analysis result
    llm_analysis = AnalysisResult(
        bug_location="line 9",
        root_cause="Assignment operator (=) used instead of comparison (==) in conditional",
        fix_suggestion="Change = to == for proper comparison",
        confidence=0.85,
        reasoning_chain=ReasoningChain(),
        supporting_evidence=[
            "LLM detected assignment in conditional context",
            "Pattern matches common programming error",
            "Syntax suggests comparison intent"
        ]
    )
    
    # Add reasoning steps
    step1 = ReasoningStep(
        step_number=1,
        description="Analyzed conditional statement syntax",
        confidence=0.9,
        evidence=["Single = operator in if statement", "Context suggests comparison"]
    )
    step2 = ReasoningStep(
        step_number=2,
        description="Compared with similar patterns in codebase",
        confidence=0.8,
        evidence=["Other conditionals use == operator", "Consistent pattern expected"]
    )
    
    llm_analysis.reasoning_chain.add_step(step1)
    llm_analysis.reasoning_chain.add_step(step2)
    llm_analysis.reasoning_chain.calculate_overall_confidence()
    
    # Initialize analyzer and integrate with LLM analysis
    analyzer = EnhancedASTAnalyzer()
    
    print("Original LLM Analysis:")
    print(f"  Location: {llm_analysis.bug_location}")
    print(f"  Root cause: {llm_analysis.root_cause}")
    print(f"  Confidence: {llm_analysis.confidence:.2f}")
    print(f"  Evidence count: {len(llm_analysis.supporting_evidence)}")
    
    # Integrate AST analysis
    integrated_result = analyzer.integrate_with_llm_analysis(code, llm_analysis)
    
    print("\nIntegrated Analysis Result:")
    print(f"  Location: {integrated_result.bug_location}")
    print(f"  Root cause: {integrated_result.root_cause}")
    print(f"  Fix suggestion: {integrated_result.fix_suggestion}")
    print(f"  Confidence: {integrated_result.confidence:.2f}")
    print(f"  Evidence count: {len(integrated_result.supporting_evidence)}")
    
    print("\nAdditional AST Evidence:")
    ast_evidence = integrated_result.supporting_evidence[len(llm_analysis.supporting_evidence):]
    for evidence in ast_evidence:
        print(f"  - {evidence}")


def example_complex_workflow():
    """Example of complete analysis workflow."""
    print("=== Complex Analysis Workflow ===")
    
    code = """
class UserManager:
    def __init__(self, database):
        self.db = database
        self.cache = {}
    
    def get_user(self, user_id):
        # Check cache first
        if user_id in self.cache:
            return self.cache[user_id]
        
        # Bug: SQL injection vulnerability
        query = f"SELECT * FROM users WHERE id = {user_id}"
        user = self.db.execute(query)
        
        # Bug: Cache might grow indefinitely
        self.cache[user_id] = user
        
        return user
    
    def update_user(self, user_id, data):
        # Bug: No validation of data
        user = self.get_user(user_id)
        
        if user:
            # Bug: Assignment instead of comparison
            if data.age = 0:  # Should be ==
                data.age = None
            
            # Bug: Wrong number of arguments
            result = self.db.update(user_id, data, "extra_param")
            return result
        
        return None
    
    def delete_user(self, user_id):
        # Bug: Using undefined variable
        if user_id in active_users:  # active_users not defined
            return {"error": "Cannot delete active user"}
        
        return self.db.delete(user_id)
"""
    
    analyzer = EnhancedASTAnalyzer()
    
    print("1. Comprehensive Error Pattern Detection:")
    patterns = analyzer.detect_error_patterns(code)
    
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        print(f"  {pattern.pattern_type} at {pattern.location}: {pattern.description}")
    
    print(f"\nPattern Summary: {dict(pattern_counts)}")
    
    print("\n2. Method Call Validation:")
    validations = analyzer.validate_function_calls(code, ["get_user", "update", "delete"])
    
    valid_calls = sum(1 for v in validations if v.is_valid)
    invalid_calls = len(validations) - valid_calls
    
    print(f"  Valid calls: {valid_calls}")
    print(f"  Invalid calls: {invalid_calls}")
    
    for validation in validations:
        if not validation.is_valid:
            print(f"    ✗ {validation.function_name}: {', '.join(validation.issues)}")
    
    print("\n3. Critical Variable Flow Analysis:")
    critical_vars = ["user_id", "data", "active_users", "query"]
    traces = analyzer.trace_variable_data_flow(code, critical_vars)
    
    for var_name, trace in traces.items():
        if trace.potential_issues:
            print(f"  {var_name}: {len(trace.potential_issues)} issues")
            for issue in trace.potential_issues[:2]:  # Show first 2 issues
                print(f"    - {issue}")
    
    print("\n4. Security and Quality Metrics:")
    security_patterns = [p for p in patterns if "injection" in p.description.lower() or "security" in p.description.lower()]
    quality_patterns = [p for p in patterns if p.severity in ["high", "critical"]]
    
    print(f"  Security issues: {len(security_patterns)}")
    print(f"  High/Critical issues: {len(quality_patterns)}")
    print(f"  Overall code health: {'Poor' if len(quality_patterns) > 3 else 'Fair' if len(quality_patterns) > 1 else 'Good'}")


if __name__ == "__main__":
    print("Enhanced AST Analyzer Examples")
    print("=" * 50)
    
    try:
        example_basic_analysis()
        print("\n" + "=" * 50)
        
        example_suspicious_region_analysis()
        print("\n" + "=" * 50)
        
        example_llm_integration()
        print("\n" + "=" * 50)
        
        example_complex_workflow()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()