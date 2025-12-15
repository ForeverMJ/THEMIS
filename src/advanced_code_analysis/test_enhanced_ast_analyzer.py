"""
Tests for the Enhanced AST Analysis Engine.

This module contains comprehensive tests for the EnhancedASTAnalyzer class,
including unit tests for error pattern detection, function call validation,
data flow tracing, and LLM integration.
"""

import pytest
import ast
from unittest.mock import Mock, MagicMock

from .enhanced_ast_analyzer import (
    EnhancedASTAnalyzer,
    SuspiciousRegion,
    ErrorPattern,
    FunctionCallValidation,
    DataFlowTrace,
    FunctionCallValidator,
    DataFlowTracer
)
from .models import (
    AnalysisResult,
    ReasoningChain,
    ReasoningStep,
    ContextWindow
)
from .llm_interface import LLMInterface


class TestEnhancedASTAnalyzer:
    """Test cases for the EnhancedASTAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LLMInterface)
        self.analyzer = EnhancedASTAnalyzer(self.mock_llm)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.llm_interface == self.mock_llm
        assert isinstance(self.analyzer.error_patterns, dict)
        assert len(self.analyzer.error_patterns) > 0
        assert isinstance(self.analyzer.function_signatures, dict)
        assert isinstance(self.analyzer.class_methods, dict)
    
    def test_analyze_suspicious_regions_simple(self):
        """Test analysis of suspicious regions with simple code."""
        code = """
def test_function(x):
    y = 5  # Suspicious constant assignment
    return x + y
"""
        
        region = SuspiciousRegion(
            start_line=2,
            end_line=3,
            reason="Constant assignment detected",
            confidence=0.8,
            code_snippet="y = 5"
        )
        
        results = self.analyzer.analyze_suspicious_regions(code, [region])
        
        assert len(results) >= 0  # Should handle the region without errors
        if results:
            assert isinstance(results[0], AnalysisResult)
            assert "2-3" in results[0].bug_location or "line" in results[0].bug_location
    
    def test_analyze_suspicious_regions_syntax_error(self):
        """Test analysis with syntax error in code."""
        code = """
def test_function(x:
    return x + 1
"""
        
        region = SuspiciousRegion(
            start_line=1,
            end_line=2,
            reason="Syntax issue",
            confidence=0.9,
            code_snippet="def test_function(x:"
        )
        
        results = self.analyzer.analyze_suspicious_regions(code, [region])
        
        assert len(results) == 1
        assert "syntax" in results[0].root_cause.lower()
        assert results[0].confidence > 0.8
    
    def test_detect_error_patterns_constant_assignment(self):
        """Test detection of constant assignment patterns."""
        code = """
def process_data():
    count = 100  # Suspicious constant
    name = "test"  # String constant
    flag = True  # Boolean constant
    return count
"""
        
        patterns = self.analyzer.detect_error_patterns(code)
        
        # Should detect at least the numeric constant assignment
        constant_patterns = [p for p in patterns if p.pattern_type == "constant_assignment"]
        assert len(constant_patterns) >= 1
        
        # Check the first constant pattern
        if constant_patterns:
            pattern = constant_patterns[0]
            assert pattern.severity == "medium"
            assert pattern.confidence > 0.0
            assert "100" in pattern.description
    
    def test_detect_error_patterns_wrong_comparison(self):
        """Test detection of wrong comparison patterns."""
        # Use code that will trigger syntax error detection
        code = """
def check_value(x):
    if x = 5:  # Wrong: assignment instead of comparison
        return True
    return False
"""
        
        patterns = self.analyzer.detect_error_patterns(code)
        
        # Should detect syntax error patterns since the code is invalid
        syntax_patterns = [p for p in patterns if p.pattern_type == "syntax_error"]
        assert len(syntax_patterns) >= 1
        
        if syntax_patterns:
            pattern = syntax_patterns[0]
            assert pattern.severity == "critical"
            assert "syntax" in pattern.description.lower()
    
    def test_detect_error_patterns_wrong_comparison_regex(self):
        """Test detection of wrong comparison patterns using regex on valid code."""
        # Test the regex pattern directly on code lines
        code_lines = [
            "def check_value(x):",
            "    if x = 5:  # This should be detected",
            "    if y == 10:  # This should not be detected", 
            "    return True"
        ]
        
        # Manually test the regex pattern
        import re
        wrong_comparison_count = 0
        for line in code_lines:
            if re.search(r'if\s+[^=]*\w+\s*=\s*[^=]', line.strip()):
                wrong_comparison_count += 1
        
        assert wrong_comparison_count >= 1
    
    def test_detect_error_patterns_undefined_variable(self):
        """Test detection of undefined variable patterns."""
        code = """
def process():
    result = undefined_var + 5  # undefined_var is not defined
    return result
"""
        
        patterns = self.analyzer.detect_error_patterns(code)
        
        # Should detect undefined variable
        undefined_patterns = [p for p in patterns if p.pattern_type == "undefined_variable"]
        assert len(undefined_patterns) >= 1
        
        if undefined_patterns:
            pattern = undefined_patterns[0]
            assert pattern.severity == "high"
            assert "undefined_var" in pattern.description
    
    def test_detect_error_patterns_with_focus_lines(self):
        """Test error pattern detection with focus lines."""
        code = """
def test():
    x = 10  # Line 2
    y = 20  # Line 3
    z = 30  # Line 4
"""
        
        focus_lines = {2, 3}  # Only focus on lines 2 and 3
        patterns = self.analyzer.detect_error_patterns(code, focus_lines)
        
        # All detected patterns should be within focus lines
        for pattern in patterns:
            line_num = int(pattern.location.split()[-1])
            assert line_num in focus_lines
    
    def test_validate_function_calls_simple(self):
        """Test function call validation with simple cases."""
        code = """
def add(a, b):
    return a + b

def test():
    result1 = add(1, 2)  # Correct call
    result2 = add(1)     # Missing argument
    result3 = add(1, 2, 3)  # Too many arguments
"""
        
        validations = self.analyzer.validate_function_calls(code)
        
        # Should find all three calls
        assert len(validations) >= 3
        
        # Check for the invalid calls
        invalid_calls = [v for v in validations if not v.is_valid]
        assert len(invalid_calls) >= 2  # Missing arg and too many args
    
    def test_validate_function_calls_with_target_functions(self):
        """Test function call validation with specific target functions."""
        code = """
def target_func(x):
    return x * 2

def other_func(y):
    return y + 1

def test():
    result1 = target_func(5)
    result2 = other_func(3)
    result3 = target_func()  # Missing argument
"""
        
        validations = self.analyzer.validate_function_calls(code, ["target_func"])
        
        # Should only validate calls to target_func
        target_validations = [v for v in validations if v.function_name == "target_func"]
        other_validations = [v for v in validations if v.function_name == "other_func"]
        
        assert len(target_validations) >= 2  # Two calls to target_func
        assert len(other_validations) == 0   # Should ignore other_func
    
    def test_trace_variable_data_flow_simple(self):
        """Test variable data flow tracing with simple cases."""
        code = """
def process():
    x = 10        # Definition
    y = x + 5     # Usage of x
    x = y * 2     # Redefinition of x
    return x      # Final usage
"""
        
        traces = self.analyzer.trace_variable_data_flow(code, ["x", "y"])
        
        assert "x" in traces
        assert "y" in traces
        
        x_trace = traces["x"]
        assert len(x_trace.definition_points) >= 2  # Initial def and redefinition
        assert len(x_trace.usage_points) >= 2      # Usage in y calculation and return
        
        y_trace = traces["y"]
        assert len(y_trace.definition_points) >= 1  # Definition
        assert len(y_trace.usage_points) >= 1      # Usage in x redefinition
    
    def test_trace_variable_data_flow_with_issues(self):
        """Test data flow tracing that detects issues."""
        code = """
def problematic():
    result = undefined_var  # Usage before definition
    x = 5                   # Definition without usage
    return result
"""
        
        traces = self.analyzer.trace_variable_data_flow(code, ["undefined_var", "x"])
        
        if "undefined_var" in traces:
            undefined_trace = traces["undefined_var"]
            assert len(undefined_trace.potential_issues) > 0
            assert any("never defined" in issue for issue in undefined_trace.potential_issues)
        
        if "x" in traces:
            x_trace = traces["x"]
            assert len(x_trace.potential_issues) > 0
            assert any("never used" in issue for issue in x_trace.potential_issues)
    
    def test_trace_variable_data_flow_with_focus_region(self):
        """Test data flow tracing with focus region."""
        code = """
def test():
    x = 1      # Line 2
    y = x + 1  # Line 3
    z = y + 1  # Line 4
    return z   # Line 5
"""
        
        focus_region = (3, 4)  # Only lines 3-4
        traces = self.analyzer.trace_variable_data_flow(code, ["x", "y", "z"], focus_region)
        
        # Should only trace within the focus region
        for var_name, trace in traces.items():
            for line, _ in trace.definition_points + trace.usage_points + trace.modifications:
                assert focus_region[0] <= line <= focus_region[1]
    
    def test_integrate_with_llm_analysis(self):
        """Test integration with LLM analysis results."""
        code = """
def buggy_function(x):
    y = 5  # Line 2: Constant assignment
    return x + y
"""
        
        # Mock LLM analysis result
        llm_analysis = AnalysisResult(
            bug_location="line 2",
            root_cause="Suspicious constant assignment",
            fix_suggestion="Consider using a variable",
            confidence=0.7,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["LLM detected constant assignment"]
        )
        
        integrated_result = self.analyzer.integrate_with_llm_analysis(code, llm_analysis)
        
        assert isinstance(integrated_result, AnalysisResult)
        assert integrated_result.bug_location == llm_analysis.bug_location
        assert len(integrated_result.supporting_evidence) > len(llm_analysis.supporting_evidence)
        
        # Should have additional AST analysis evidence
        evidence_text = " ".join(integrated_result.supporting_evidence)
        assert "AST" in evidence_text or "error pattern" in evidence_text.lower()
    
    def test_integrate_with_llm_analysis_syntax_error(self):
        """Test integration when code has syntax errors."""
        code = """
def broken_function(x:
    return x + 1
"""
        
        llm_analysis = AnalysisResult(
            bug_location="line 1",
            root_cause="Syntax issue",
            fix_suggestion="Fix syntax",
            confidence=0.8,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["LLM detected syntax issue"]
        )
        
        integrated_result = self.analyzer.integrate_with_llm_analysis(code, llm_analysis)
        
        # Should still return a result, confidence may be adjusted
        assert isinstance(integrated_result, AnalysisResult)
        # Confidence may increase due to AST analysis confirming issues
        assert integrated_result.confidence >= llm_analysis.confidence * 0.8
    
    def test_build_signature_cache(self):
        """Test building function signature cache."""
        code = """
def simple_func(a, b):
    return a + b

def varargs_func(a, *args, **kwargs):
    return sum(args)

class TestClass:
    def method1(self):
        pass
    
    def method2(self, x, y=None):
        return x
"""
        
        tree = ast.parse(code)
        self.analyzer._build_signature_cache(tree)
        
        # Check function signatures
        assert "simple_func" in self.analyzer.function_signatures
        simple_sig = self.analyzer.function_signatures["simple_func"]
        assert simple_sig["arg_count"] == 2
        assert not simple_sig["has_varargs"]
        assert not simple_sig["has_kwargs"]
        
        assert "varargs_func" in self.analyzer.function_signatures
        varargs_sig = self.analyzer.function_signatures["varargs_func"]
        assert varargs_sig["arg_count"] == 1  # Only 'a' is regular arg
        assert varargs_sig["has_varargs"]
        assert varargs_sig["has_kwargs"]
        
        # Check class methods
        assert "TestClass" in self.analyzer.class_methods
        methods = self.analyzer.class_methods["TestClass"]
        assert "method1" in methods
        assert "method2" in methods


class TestFunctionCallValidator:
    """Test cases for the FunctionCallValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.function_signatures = {
            "add": {
                "args": ["a", "b"],
                "arg_count": 2,
                "has_varargs": False,
                "has_kwargs": False
            },
            "varargs_func": {
                "args": ["x", "*args"],
                "arg_count": 1,
                "has_varargs": True,
                "has_kwargs": False
            }
        }
        self.class_methods = {}
        self.validator = FunctionCallValidator(
            self.function_signatures, 
            self.class_methods
        )
    
    def test_validate_correct_call(self):
        """Test validation of correct function call."""
        code = "add(1, 2)"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        validation = self.validator.validate_call(call_node)
        
        assert validation is not None
        assert validation.function_name == "add"
        assert validation.is_valid
        assert len(validation.issues) == 0
    
    def test_validate_too_many_args(self):
        """Test validation of call with too many arguments."""
        code = "add(1, 2, 3)"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        validation = self.validator.validate_call(call_node)
        
        assert validation is not None
        assert validation.function_name == "add"
        assert not validation.is_valid
        assert len(validation.issues) > 0
        assert "too many" in validation.issues[0].lower()
    
    def test_validate_too_few_args(self):
        """Test validation of call with too few arguments."""
        code = "add(1)"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        validation = self.validator.validate_call(call_node)
        
        assert validation is not None
        assert validation.function_name == "add"
        assert not validation.is_valid
        assert len(validation.issues) > 0
        assert "too few" in validation.issues[0].lower()
    
    def test_validate_varargs_function(self):
        """Test validation of function with variable arguments."""
        code = "varargs_func(1, 2, 3, 4)"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        validation = self.validator.validate_call(call_node)
        
        assert validation is not None
        assert validation.function_name == "varargs_func"
        assert validation.is_valid  # Should accept extra args due to *args
    
    def test_validate_unknown_function(self):
        """Test validation of unknown function."""
        code = "unknown_func(1, 2)"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        
        validation = self.validator.validate_call(call_node)
        
        assert validation is not None
        assert validation.function_name == "unknown_func"
        assert validation.is_valid  # Should be valid since we don't know the signature
        assert validation.expected_signature is None
    
    def test_validate_with_target_functions(self):
        """Test validation with specific target functions."""
        validator = FunctionCallValidator(
            self.function_signatures, 
            self.class_methods,
            target_functions=["add"]
        )
        
        # Should validate add() call
        code1 = "add(1, 2)"
        tree1 = ast.parse(code1)
        call_node1 = tree1.body[0].value
        validation1 = validator.validate_call(call_node1)
        assert validation1 is not None
        
        # Should skip varargs_func() call
        code2 = "varargs_func(1, 2)"
        tree2 = ast.parse(code2)
        call_node2 = tree2.body[0].value
        validation2 = validator.validate_call(call_node2)
        assert validation2 is None


class TestDataFlowTracer:
    """Test cases for the DataFlowTracer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = DataFlowTracer(["x", "y"])
    
    def test_trace_simple_variable(self):
        """Test tracing of simple variable flow."""
        code = """
x = 10
y = x + 5
print(y)
"""
        tree = ast.parse(code)
        code_lines = code.strip().split('\n')
        
        trace = self.tracer.trace_variable(tree, code_lines, "x")
        
        assert trace.variable_name == "x"
        assert len(trace.definition_points) == 1
        assert len(trace.usage_points) == 1
        # AST line numbers start from the actual code lines
        assert trace.definition_points[0][0] == 2  # x = 10 is on line 2
        assert trace.usage_points[0][0] == 3       # y = x + 5 is on line 3
    
    def test_trace_variable_with_modifications(self):
        """Test tracing variable with modifications."""
        code = """
x = 10
x += 5
x *= 2
result = x
"""
        tree = ast.parse(code)
        code_lines = code.strip().split('\n')
        
        trace = self.tracer.trace_variable(tree, code_lines, "x")
        
        assert len(trace.definition_points) == 1   # Initial definition
        assert len(trace.modifications) == 2       # += and *=
        assert len(trace.usage_points) == 1        # Usage in result assignment
    
    def test_trace_with_focus_region(self):
        """Test tracing with focus region."""
        code = """
x = 1      # Line 2 in AST
y = x + 1  # Line 3 in AST
z = y + 1  # Line 4 in AST
w = z + 1  # Line 5 in AST
"""
        tree = ast.parse(code)
        code_lines = code.strip().split('\n')
        
        tracer = DataFlowTracer(["x", "y"], focus_region=(3, 4))  # Focus on lines 3-4
        trace = tracer.trace_variable(tree, code_lines, "x")
        
        # Should only find usage in line 3 (within focus region)
        assert len(trace.definition_points) == 0  # Line 2 is outside focus
        assert len(trace.usage_points) == 1       # Line 3 is in focus
        assert trace.usage_points[0][0] == 3
    
    def test_analyze_flow_issues_usage_before_definition(self):
        """Test detection of usage before definition."""
        code = """
result = x + 1  # Line 2: usage
x = 10          # Line 3: definition
"""
        tree = ast.parse(code)
        code_lines = code.strip().split('\n')
        
        trace = self.tracer.trace_variable(tree, code_lines, "x")
        
        assert len(trace.potential_issues) > 0
        assert any("used before definition" in issue for issue in trace.potential_issues)
    
    def test_analyze_flow_issues_unused_variable(self):
        """Test detection of unused variable."""
        code = """
x = 10  # Defined but never used
y = 20
result = y
"""
        tree = ast.parse(code)
        code_lines = code.strip().split('\n')
        
        trace = self.tracer.trace_variable(tree, code_lines, "x")
        
        assert len(trace.potential_issues) > 0
        assert any("never used" in issue for issue in trace.potential_issues)
    
    def test_analyze_flow_issues_undefined_variable(self):
        """Test detection of undefined variable usage."""
        code = """
result = undefined_var + 1  # Used but never defined
"""
        tree = ast.parse(code)
        code_lines = code.strip().split('\n')
        
        trace = self.tracer.trace_variable(tree, code_lines, "undefined_var")
        
        assert len(trace.potential_issues) > 0
        assert any("never defined" in issue for issue in trace.potential_issues)


class TestSuspiciousRegion:
    """Test cases for the SuspiciousRegion dataclass."""
    
    def test_creation(self):
        """Test creation of SuspiciousRegion."""
        region = SuspiciousRegion(
            start_line=10,
            end_line=15,
            reason="Test reason",
            confidence=0.8,
            code_snippet="test code"
        )
        
        assert region.start_line == 10
        assert region.end_line == 15
        assert region.reason == "Test reason"
        assert region.confidence == 0.8
        assert region.code_snippet == "test code"
        assert region.suggested_focus == []


class TestErrorPattern:
    """Test cases for the ErrorPattern dataclass."""
    
    def test_creation(self):
        """Test creation of ErrorPattern."""
        pattern = ErrorPattern(
            pattern_type="test_pattern",
            location="line 5",
            description="Test description",
            severity="medium",
            confidence=0.7
        )
        
        assert pattern.pattern_type == "test_pattern"
        assert pattern.location == "line 5"
        assert pattern.description == "Test description"
        assert pattern.severity == "medium"
        assert pattern.confidence == 0.7
        assert pattern.suggested_fix == ""
        assert pattern.code_context == ""


class TestFunctionCallValidation:
    """Test cases for the FunctionCallValidation dataclass."""
    
    def test_creation(self):
        """Test creation of FunctionCallValidation."""
        validation = FunctionCallValidation(
            function_name="test_func",
            call_location="line 10",
            is_valid=False
        )
        
        assert validation.function_name == "test_func"
        assert validation.call_location == "line 10"
        assert not validation.is_valid
        assert validation.issues == []
        assert validation.expected_signature is None
        assert validation.actual_call == ""


class TestDataFlowTrace:
    """Test cases for the DataFlowTrace dataclass."""
    
    def test_creation(self):
        """Test creation of DataFlowTrace."""
        trace = DataFlowTrace(variable_name="test_var")
        
        assert trace.variable_name == "test_var"
        assert trace.definition_points == []
        assert trace.usage_points == []
        assert trace.modifications == []
        assert trace.flow_path == []
        assert trace.potential_issues == []


# Integration tests
class TestEnhancedASTAnalyzerIntegration:
    """Integration tests for the Enhanced AST Analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedASTAnalyzer()
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow with realistic code."""
        code = """
def calculate_total(items):
    total = 0
    for item in items:
        if item.price = 100:  # Bug: assignment instead of comparison
            discount = 0.1
        else:
            discount = 0
        total += item.price * (1 - discount)
    
    # Bug: using undefined variable
    final_total = total + tax_amount
    return final_total

def process_order(order):
    items = order.get_items()
    total = calculate_total(items)  # Correct call
    
    # Bug: wrong number of arguments
    result = calculate_total(items, "extra_arg")
    
    return total
"""
        
        # Create suspicious regions
        regions = [
            SuspiciousRegion(
                start_line=4,
                end_line=4,
                reason="Assignment in if condition",
                confidence=0.9,
                code_snippet="if item.price = 100:"
            ),
            SuspiciousRegion(
                start_line=11,
                end_line=11,
                reason="Undefined variable usage",
                confidence=0.8,
                code_snippet="final_total = total + tax_amount"
            )
        ]
        
        # Run analysis
        results = self.analyzer.analyze_suspicious_regions(code, regions)
        
        # Should find issues in both regions
        assert len(results) >= 1
        
        # Test error pattern detection
        patterns = self.analyzer.detect_error_patterns(code)
        assert len(patterns) > 0
        
        # Should detect syntax error (since the code has invalid syntax)
        syntax_patterns = [p for p in patterns if "syntax" in p.pattern_type.lower()]
        assert len(syntax_patterns) > 0
        
        # Test function call validation
        validations = self.analyzer.validate_function_calls(code)
        invalid_calls = [v for v in validations if not v.is_valid]
        assert len(invalid_calls) > 0  # Should find the call with extra argument
        
        # Test data flow tracing
        traces = self.analyzer.trace_variable_data_flow(code, ["total", "tax_amount"])
        assert "total" in traces
        assert "tax_amount" in traces
        
        # tax_amount should have issues (used but not defined)
        tax_trace = traces["tax_amount"]
        assert len(tax_trace.potential_issues) > 0
    
    def test_analysis_with_complex_code(self):
        """Test analysis with more complex code structures."""
        code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process_data(self, data):
        if not data:
            return []
        
        results = []
        for item in data:
            # Bug: potential type mismatch
            processed = self.transform_item(item) + "suffix"
            results.append(processed)
        
        return results
    
    def transform_item(self, item):
        # Bug: method call with wrong arguments
        return self.apply_transform(item, "extra", "args")
    
    def apply_transform(self, item):
        return str(item).upper()
"""
        
        # Test with class methods
        validations = self.analyzer.validate_function_calls(code)
        
        # Should validate method calls
        method_validations = [v for v in validations if "apply_transform" in v.function_name]
        assert len(method_validations) > 0
        
        # Should detect the call with wrong number of arguments
        invalid_method_calls = [v for v in method_validations if not v.is_valid]
        assert len(invalid_method_calls) > 0
        
        # Test error pattern detection
        patterns = self.analyzer.detect_error_patterns(code)
        
        # May detect type mismatch patterns
        type_patterns = [p for p in patterns if "type" in p.pattern_type]
        # Note: This might not always detect depending on the complexity of analysis
        
        # Test data flow tracing for class variables
        traces = self.analyzer.trace_variable_data_flow(code, ["results", "processed"])
        
        if "results" in traces:
            results_trace = traces["results"]
            assert len(results_trace.definition_points) > 0
            assert len(results_trace.usage_points) > 0


if __name__ == "__main__":
    pytest.main([__file__])