"""Property-based tests for call relationship correctness."""

import ast
import pytest
from hypothesis import given, strategies as st, assume
from typing import List, Tuple

from src.enhanced_graph_manager.structural_extractor import StructuralExtractor
from src.enhanced_graph_manager.models import CallEdge


# Hypothesis strategies for generating valid Python identifiers and code
def valid_identifier():
    """Generate valid Python identifiers."""
    import keyword
    return st.text(
        min_size=1, 
        max_size=20,
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    ).filter(lambda x: x.isidentifier() and not x.startswith('_') and not x[0].isdigit() and not keyword.iskeyword(x))


def valid_function_name():
    """Generate valid function names (starting with lowercase)."""
    return valid_identifier().filter(lambda x: x[0].islower())


def valid_class_name():
    """Generate valid class names (starting with uppercase)."""
    return valid_identifier().filter(lambda x: x[0].isupper())


def generate_function_with_calls(caller_name: str, callees: List[str]) -> str:
    """Generate Python function code with specified function calls."""
    call_statements = []
    for callee in callees:
        call_statements.append(f"    {callee}()")
    
    if not call_statements:
        call_statements.append("    pass")
    
    return f"""def {caller_name}():
{chr(10).join(call_statements)}
"""


def generate_class_with_method_calls(class_name: str, method_name: str, callees: List[str]) -> str:
    """Generate Python class with method that makes calls."""
    call_statements = []
    for callee in callees:
        if callee.startswith("self."):
            call_statements.append(f"        {callee}()")
        else:
            call_statements.append(f"        {callee}()")
    
    if not call_statements:
        call_statements.append("        pass")
    
    return f"""class {class_name}:
    def {method_name}(self):
{chr(10).join(call_statements)}
"""


@given(
    caller_name=valid_function_name(),
    callees=st.lists(
        valid_function_name(),
        min_size=1,
        max_size=5,
        unique=True
    )
)
def test_function_call_relationship_correctness(caller_name, callees):
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For any code containing function calls, the system should create correct
    CALLS edges between caller and callee functions.
    """
    # Ensure caller is different from all callees
    assume(caller_name not in callees)
    
    # Generate function code with calls
    code = generate_function_with_calls(caller_name, callees)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract call edges using the structural extractor
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(code))
    
    # Property: Should extract exactly the same number of calls as defined
    assert len(call_edges) == len(callees), \
        f"Expected {len(callees)} call edges, got {len(call_edges)}"
    
    # Property: All call edges should be CallEdge instances
    for edge in call_edges:
        assert isinstance(edge, CallEdge), f"Expected CallEdge, got {type(edge)}"
    
    # Property: All calls should have the correct caller
    for edge in call_edges:
        assert edge.caller == caller_name, \
            f"Expected caller '{caller_name}', got '{edge.caller}'"
    
    # Property: All expected callees should be present
    actual_callees = [edge.callee for edge in call_edges]
    assert set(actual_callees) == set(callees), \
        f"Expected callees {set(callees)}, got {set(actual_callees)}"
    
    # Property: Line numbers should be positive
    for edge in call_edges:
        assert edge.line_number > 0, f"Expected positive line number, got {edge.line_number}"


@given(
    class_name=valid_class_name(),
    method_name=valid_function_name(),
    self_methods=st.lists(
        valid_function_name(),
        min_size=1,
        max_size=3,
        unique=True
    ),
    external_functions=st.lists(
        valid_function_name(),
        min_size=0,
        max_size=2,
        unique=True
    )
)
def test_method_call_relationship_correctness(class_name, method_name, self_methods, external_functions):
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For any class method containing calls to self methods and external functions,
    the system should create correct CALLS edges with proper naming.
    """
    # Ensure method name is different from self methods and external functions
    assume(method_name not in self_methods)
    assume(method_name not in external_functions)
    assume(not set(self_methods).intersection(set(external_functions)))
    
    # Create list of all callees with proper prefixes
    self_callees = [f"self.{method}" for method in self_methods]
    all_callees = self_callees + external_functions
    
    # Generate class code with method calls
    code = generate_class_with_method_calls(class_name, method_name, all_callees)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract call edges using the structural extractor
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(code))
    
    # Property: Should extract exactly the same number of calls as defined
    expected_call_count = len(self_methods) + len(external_functions)
    assert len(call_edges) == expected_call_count, \
        f"Expected {expected_call_count} call edges, got {len(call_edges)}"
    
    # Property: All calls should have the correct caller (method name)
    for edge in call_edges:
        assert edge.caller == method_name, \
            f"Expected caller '{method_name}', got '{edge.caller}'"
    
    # Property: Self method calls should be properly prefixed
    actual_callees = [edge.callee for edge in call_edges]
    assert set(actual_callees) == set(all_callees), \
        f"Expected callees {set(all_callees)}, got {set(actual_callees)}"
    
    # Property: Self method calls should have self. prefix
    self_call_edges = [edge for edge in call_edges if edge.callee.startswith("self.")]
    assert len(self_call_edges) == len(self_methods), \
        f"Expected {len(self_methods)} self method calls, got {len(self_call_edges)}"


@given(
    functions=st.lists(
        st.tuples(
            valid_function_name(),
            st.lists(valid_function_name(), min_size=0, max_size=3, unique=True)
        ),
        min_size=2,
        max_size=4,
        unique_by=lambda x: x[0]  # Ensure unique function names
    )
)
def test_multiple_function_call_relationships(functions):
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For code with multiple functions making calls, the system should correctly
    track all call relationships without confusion between callers.
    """
    # Generate code with multiple functions
    code_parts = []
    expected_edges = []
    
    for caller_name, callees in functions:
        # Ensure caller doesn't call itself or other callers
        filtered_callees = [callee for callee in callees 
                          if callee != caller_name and 
                          callee not in [func_name for func_name, _ in functions]]
        
        if filtered_callees:
            code_parts.append(generate_function_with_calls(caller_name, filtered_callees))
            for callee in filtered_callees:
                expected_edges.append((caller_name, callee))
        else:
            # Add function with no calls
            code_parts.append(f"def {caller_name}():\n    pass")
    
    code = "\n\n".join(code_parts)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract call edges using the structural extractor
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(code))
    
    # Property: Should extract exactly the expected number of calls
    assert len(call_edges) == len(expected_edges), \
        f"Expected {len(expected_edges)} call edges, got {len(call_edges)}"
    
    # Property: All expected call relationships should be present
    actual_edges = [(edge.caller, edge.callee) for edge in call_edges]
    assert set(actual_edges) == set(expected_edges), \
        f"Expected edges {set(expected_edges)}, got {set(actual_edges)}"


def test_chained_method_calls():
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For chained method calls like obj.method1().method2(), the system should
    handle complex call patterns appropriately.
    """
    code = """
def complex_caller():
    obj.method1()
    obj.method2().chain()
    self.helper()
    nested_obj.attr.method()
"""
    
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(code))
    
    # Property: Should extract call edges for identifiable patterns
    # Note: Complex chained calls may be simplified to the first identifiable part
    assert len(call_edges) >= 2, f"Expected at least 2 call edges, got {len(call_edges)}"
    
    # Property: All edges should have correct caller
    for edge in call_edges:
        assert edge.caller == "complex_caller", \
            f"Expected caller 'complex_caller', got '{edge.caller}'"
    
    # Property: Should identify some of the method calls
    callees = [edge.callee for edge in call_edges]
    assert "obj.method1" in callees or "obj.method2" in callees or "self.helper" in callees


def test_no_function_calls():
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For functions with no calls, the system should return an empty list.
    """
    code = """
def function_with_no_calls():
    x = 42
    y = x + 10
    return y

def another_function():
    pass
"""
    
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(code))
    
    # Property: Should extract no call edges
    assert len(call_edges) == 0, f"Expected 0 call edges for functions with no calls, got {len(call_edges)}"


def test_recursive_function_calls():
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For recursive function calls, the system should correctly identify
    self-referential call relationships.
    """
    code = """
def recursive_function(n):
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)

def mutually_recursive_a(n):
    if n <= 0:
        return 1
    return mutually_recursive_b(n - 1)

def mutually_recursive_b(n):
    if n <= 0:
        return 0
    return mutually_recursive_a(n - 1)
"""
    
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(code))
    
    # Property: Should extract recursive call edges
    assert len(call_edges) >= 3, f"Expected at least 3 call edges, got {len(call_edges)}"
    
    # Property: Should identify recursive call
    recursive_edges = [edge for edge in call_edges 
                      if edge.caller == "recursive_function" and edge.callee == "recursive_function"]
    assert len(recursive_edges) == 1, f"Expected 1 recursive edge, got {len(recursive_edges)}"
    
    # Property: Should identify mutual recursion
    mutual_edges = [(edge.caller, edge.callee) for edge in call_edges 
                   if "mutually_recursive" in edge.caller and "mutually_recursive" in edge.callee]
    assert len(mutual_edges) >= 2, f"Expected at least 2 mutual recursive edges, got {len(mutual_edges)}"


def test_empty_code_call_extraction():
    """
    **Feature: enhanced-graph-manager, Property 4: 调用关系正确性**
    **Validates: Requirements 1.4**
    
    For empty Python code, call extraction should return an empty list.
    """
    extractor = StructuralExtractor()
    call_edges = extractor.extract_call_edges(ast.parse(""))
    
    # Property: Empty code should result in no call edges
    assert len(call_edges) == 0, f"Expected 0 call edges for empty code, got {len(call_edges)}"