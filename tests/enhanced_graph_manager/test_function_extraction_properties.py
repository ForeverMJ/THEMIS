"""Property-based tests for function extraction completeness."""

import ast
import pytest
from hypothesis import given, strategies as st, assume
from typing import List

from src.enhanced_graph_manager.structural_extractor import StructuralExtractor
from src.enhanced_graph_manager.models import FunctionNode


# Hypothesis strategies for generating valid Python identifiers and code
def valid_identifier():
    """Generate valid Python identifiers."""
    return st.text(
        min_size=1, 
        max_size=20,
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    ).filter(lambda x: x.isidentifier() and not x.startswith('_') and not x[0].isdigit())


def valid_function_name():
    """Generate valid function names."""
    return valid_identifier().filter(lambda x: not x[0].isupper())


def valid_parameter_list():
    """Generate valid parameter lists."""
    return st.lists(
        valid_identifier(),
        min_size=0,
        max_size=5,
        unique=True
    )


def simple_function_body():
    """Generate simple function bodies."""
    return st.sampled_from([
        "pass",
        "return None",
        "return 42",
        'return "hello"',
        "x = 1\nreturn x",
        "print('test')",
    ])


def generate_function_code(name: str, params: List[str], body: str, 
                          has_return_type: bool = False, 
                          has_docstring: bool = False) -> str:
    """Generate Python function code."""
    param_str = ", ".join(params)
    
    return_annotation = " -> int" if has_return_type else ""
    
    docstring = '    """Test function docstring."""\n' if has_docstring else ""
    
    # Ensure proper indentation for body
    indented_body = "\n".join(f"    {line}" for line in body.split("\n"))
    
    return f"""def {name}({param_str}){return_annotation}:
{docstring}{indented_body}
"""


@given(
    name=valid_function_name(),
    params=valid_parameter_list(),
    body=simple_function_body(),
    has_return_type=st.booleans(),
    has_docstring=st.booleans()
)
def test_function_extraction_completeness(name, params, body, has_return_type, has_docstring):
    """
    **Feature: enhanced-graph-manager, Property 1: 函数提取完整性**
    **Validates: Requirements 1.1**
    
    For any Python code containing function definitions, structural extraction
    should create a FunctionNode for each function with correct name, parameters,
    and return type information.
    """
    # Generate function code
    code = generate_function_code(name, params, body, has_return_type, has_docstring)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract functions using the structural extractor
    extractor = StructuralExtractor()
    functions = extractor.extract_functions(ast.parse(code))
    
    # Property: Should extract exactly one function
    assert len(functions) == 1, f"Expected 1 function, got {len(functions)}"
    
    func = functions[0]
    
    # Property: Function should be a FunctionNode instance
    assert isinstance(func, FunctionNode), f"Expected FunctionNode, got {type(func)}"
    
    # Property: Function name should match exactly
    assert func.name == name, f"Expected name '{name}', got '{func.name}'"
    
    # Property: Function parameters should match exactly
    assert func.args == params, f"Expected args {params}, got {func.args}"
    
    # Property: Return type should be correctly extracted
    if has_return_type:
        assert func.return_type == "int", f"Expected return type 'int', got '{func.return_type}'"
    else:
        assert func.return_type is None, f"Expected no return type, got '{func.return_type}'"
    
    # Property: Docstring should be correctly extracted
    if has_docstring:
        assert func.docstring == "Test function docstring.", f"Expected docstring, got '{func.docstring}'"
    else:
        assert func.docstring is None, f"Expected no docstring, got '{func.docstring}'"
    
    # Property: Line number should be positive
    assert func.line_number > 0, f"Expected positive line number, got {func.line_number}"


@given(
    functions=st.lists(
        st.tuples(
            valid_function_name(),
            valid_parameter_list(),
            simple_function_body()
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]  # Ensure unique function names
    )
)
def test_multiple_function_extraction_completeness(functions):
    """
    **Feature: enhanced-graph-manager, Property 1: 函数提取完整性**
    **Validates: Requirements 1.1**
    
    For any Python code containing multiple function definitions, structural 
    extraction should create a FunctionNode for each function.
    """
    # Generate code with multiple functions
    code_parts = []
    expected_names = []
    
    for name, params, body in functions:
        func_code = generate_function_code(name, params, body)
        code_parts.append(func_code)
        expected_names.append(name)
    
    code = "\n\n".join(code_parts)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract functions using the structural extractor
    extractor = StructuralExtractor()
    extracted_functions = extractor.extract_functions(ast.parse(code))
    
    # Property: Should extract exactly the same number of functions as defined
    assert len(extracted_functions) == len(functions), \
        f"Expected {len(functions)} functions, got {len(extracted_functions)}"
    
    # Property: All function names should be present
    extracted_names = [func.name for func in extracted_functions]
    assert set(extracted_names) == set(expected_names), \
        f"Expected names {set(expected_names)}, got {set(extracted_names)}"
    
    # Property: Each extracted function should be a FunctionNode
    for func in extracted_functions:
        assert isinstance(func, FunctionNode), f"Expected FunctionNode, got {type(func)}"
        assert func.line_number > 0, f"Expected positive line number, got {func.line_number}"


@given(
    class_name=valid_identifier().filter(lambda x: x[0].isupper()),
    method_names=st.lists(
        valid_function_name(),
        min_size=1,
        max_size=3,
        unique=True
    )
)
def test_method_extraction_completeness(class_name, method_names):
    """
    **Feature: enhanced-graph-manager, Property 1: 函数提取完整性**
    **Validates: Requirements 1.1**
    
    For any Python code containing class methods, structural extraction
    should create a FunctionNode for each method including class methods.
    """
    # Generate class with methods
    method_codes = []
    for method_name in method_names:
        method_code = f"""    def {method_name}(self):
        pass"""
        method_codes.append(method_code)
    
    code = f"""class {class_name}:
{chr(10).join(method_codes)}
"""
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract functions using the structural extractor
    extractor = StructuralExtractor()
    functions = extractor.extract_functions(ast.parse(code))
    
    # Property: Should extract exactly the same number of methods as defined
    assert len(functions) == len(method_names), \
        f"Expected {len(method_names)} methods, got {len(functions)}"
    
    # Property: All method names should be present
    extracted_names = [func.name for func in functions]
    assert set(extracted_names) == set(method_names), \
        f"Expected method names {set(method_names)}, got {set(extracted_names)}"
    
    # Property: Each method should have 'self' as first parameter
    for func in functions:
        assert isinstance(func, FunctionNode), f"Expected FunctionNode, got {type(func)}"
        assert len(func.args) >= 1, f"Expected at least 1 argument (self), got {len(func.args)}"
        assert func.args[0] == "self", f"Expected first argument to be 'self', got '{func.args[0]}'"


def test_empty_code_extraction():
    """
    **Feature: enhanced-graph-manager, Property 1: 函数提取完整性**
    **Validates: Requirements 1.1**
    
    For empty Python code, structural extraction should return an empty list.
    """
    extractor = StructuralExtractor()
    functions = extractor.extract_functions(ast.parse(""))
    
    # Property: Empty code should result in no functions
    assert len(functions) == 0, f"Expected 0 functions for empty code, got {len(functions)}"


def test_code_without_functions_extraction():
    """
    **Feature: enhanced-graph-manager, Property 1: 函数提取完整性**
    **Validates: Requirements 1.1**
    
    For Python code without function definitions, structural extraction 
    should return an empty list.
    """
    code = """
x = 42
y = "hello"
print(x + len(y))
"""
    
    extractor = StructuralExtractor()
    functions = extractor.extract_functions(ast.parse(code))
    
    # Property: Code without functions should result in no functions
    assert len(functions) == 0, f"Expected 0 functions for code without functions, got {len(functions)}"