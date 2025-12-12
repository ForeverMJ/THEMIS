"""Property-based tests for member variable extraction completeness."""

import ast
import pytest
from hypothesis import given, strategies as st, assume
from typing import List, Tuple

from src.enhanced_graph_manager.structural_extractor import StructuralExtractor
from src.enhanced_graph_manager.models import VariableNode


# Hypothesis strategies for generating valid Python identifiers and code
def valid_identifier():
    """Generate valid Python identifiers."""
    import keyword
    return st.text(
        min_size=1, 
        max_size=20,
        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    ).filter(lambda x: x.isidentifier() and not x.startswith('_') and not x[0].isdigit() and not keyword.iskeyword(x))


def valid_class_name():
    """Generate valid class names (starting with uppercase)."""
    return valid_identifier().filter(lambda x: x[0].isupper())


def valid_method_name():
    """Generate valid method names (starting with lowercase)."""
    return valid_identifier().filter(lambda x: x[0].islower())


def valid_variable_name():
    """Generate valid variable names (starting with lowercase)."""
    return valid_identifier().filter(lambda x: x[0].islower())


def simple_value():
    """Generate simple Python values for assignment."""
    return st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(min_size=0, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz '),
        st.lists(st.integers(min_value=0, max_value=10), min_size=0, max_size=3),
        st.booleans(),
        st.none()
    )


def generate_class_with_variables(class_name: str, methods_with_vars: List[Tuple[str, List[Tuple[str, any]]]]) -> str:
    """Generate Python class code with member variable assignments."""
    method_codes = []
    
    for method_name, variables in methods_with_vars:
        var_assignments = []
        for var_name, value in variables:
            if isinstance(value, str):
                var_assignments.append(f"        self.{var_name} = \"{value}\"")
            elif isinstance(value, list):
                var_assignments.append(f"        self.{var_name} = {value}")
            elif value is None:
                var_assignments.append(f"        self.{var_name} = None")
            else:
                var_assignments.append(f"        self.{var_name} = {value}")
        
        if var_assignments:
            method_code = f"""    def {method_name}(self):
{chr(10).join(var_assignments)}"""
        else:
            method_code = f"""    def {method_name}(self):
        pass"""
        
        method_codes.append(method_code)
    
    return f"""class {class_name}:
{chr(10).join(method_codes)}
"""


@given(
    class_name=valid_class_name(),
    method_name=valid_method_name(),
    variables=st.lists(
        st.tuples(valid_variable_name(), simple_value()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]  # Ensure unique variable names
    )
)
def test_member_variable_extraction_completeness(class_name, method_name, variables):
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For any class containing self.xxx member variable definitions, structural
    extraction should create a VariableNode for each member variable.
    """
    # Generate class code with member variables
    code = generate_class_with_variables(class_name, [(method_name, variables)])
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract variables using the structural extractor
    extractor = StructuralExtractor()
    extracted_variables = extractor.extract_variables(ast.parse(code))
    
    # Property: Should extract exactly the same number of variables as defined
    assert len(extracted_variables) == len(variables), \
        f"Expected {len(variables)} variables, got {len(extracted_variables)}"
    
    # Property: All variable names should be present with self. prefix
    expected_names = [f"self.{var_name}" for var_name, _ in variables]
    extracted_names = [var.name for var in extracted_variables]
    assert set(extracted_names) == set(expected_names), \
        f"Expected variable names {set(expected_names)}, got {set(extracted_names)}"
    
    # Property: Each extracted variable should be a VariableNode
    for var in extracted_variables:
        assert isinstance(var, VariableNode), f"Expected VariableNode, got {type(var)}"
        
        # Property: Variable should be defined in the correct method
        expected_defined_in = f"{class_name}.{method_name}"
        assert var.defined_in == expected_defined_in, \
            f"Expected defined_in '{expected_defined_in}', got '{var.defined_in}'"
        
        # Property: Line number should be positive
        assert var.line_number > 0, f"Expected positive line number, got {var.line_number}"


@given(
    class_name=valid_class_name(),
    methods_with_vars=st.lists(
        st.tuples(
            valid_method_name(),
            st.lists(
                st.tuples(valid_variable_name(), simple_value()),
                min_size=1,
                max_size=3,
                unique_by=lambda x: x[0]
            )
        ),
        min_size=1,
        max_size=3,
        unique_by=lambda x: x[0]  # Ensure unique method names
    )
)
def test_multiple_method_variable_extraction_completeness(class_name, methods_with_vars):
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For any class with multiple methods containing member variables, structural
    extraction should correctly associate each variable with its defining method.
    """
    # Generate class code with multiple methods and variables
    code = generate_class_with_variables(class_name, methods_with_vars)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract variables using the structural extractor
    extractor = StructuralExtractor()
    extracted_variables = extractor.extract_variables(ast.parse(code))
    
    # Calculate expected total variables
    total_expected_vars = sum(len(variables) for _, variables in methods_with_vars)
    
    # Property: Should extract exactly the total number of variables
    assert len(extracted_variables) == total_expected_vars, \
        f"Expected {total_expected_vars} variables, got {len(extracted_variables)}"
    
    # Property: Each variable should be correctly associated with its method
    for method_name, variables in methods_with_vars:
        expected_defined_in = f"{class_name}.{method_name}"
        
        # Find variables defined in this method
        method_vars = [var for var in extracted_variables if var.defined_in == expected_defined_in]
        
        # Property: Should have correct number of variables for this method
        assert len(method_vars) == len(variables), \
            f"Expected {len(variables)} variables in {method_name}, got {len(method_vars)}"
        
        # Property: Variable names should match
        expected_names = [f"self.{var_name}" for var_name, _ in variables]
        actual_names = [var.name for var in method_vars]
        assert set(actual_names) == set(expected_names), \
            f"Expected variables {set(expected_names)} in {method_name}, got {set(actual_names)}"


@given(
    class_name=valid_class_name(),
    init_vars=st.lists(
        st.tuples(valid_variable_name(), simple_value()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]
    ),
    other_method_vars=st.lists(
        st.tuples(valid_variable_name(), simple_value()),
        min_size=1,
        max_size=3,
        unique_by=lambda x: x[0]
    )
)
def test_init_vs_other_method_variable_extraction(class_name, init_vars, other_method_vars):
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For any class with variables defined in __init__ and other methods, structural
    extraction should correctly distinguish between them.
    """
    # Ensure no variable name conflicts between methods
    init_var_names = [name for name, _ in init_vars]
    other_var_names = [name for name, _ in other_method_vars]
    assume(not set(init_var_names).intersection(set(other_var_names)))
    
    # Generate class code with variables in both __init__ and another method
    methods_with_vars = [
        ("__init__", init_vars),
        ("update_data", other_method_vars)
    ]
    code = generate_class_with_variables(class_name, methods_with_vars)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract variables using the structural extractor
    extractor = StructuralExtractor()
    extracted_variables = extractor.extract_variables(ast.parse(code))
    
    # Property: Should extract variables from both methods
    init_defined_in = f"{class_name}.__init__"
    other_defined_in = f"{class_name}.update_data"
    
    init_extracted_vars = [var for var in extracted_variables if var.defined_in == init_defined_in]
    other_extracted_vars = [var for var in extracted_variables if var.defined_in == other_defined_in]
    
    # Property: Correct number of variables in each method
    assert len(init_extracted_vars) == len(init_vars), \
        f"Expected {len(init_vars)} variables in __init__, got {len(init_extracted_vars)}"
    assert len(other_extracted_vars) == len(other_method_vars), \
        f"Expected {len(other_method_vars)} variables in update_data, got {len(other_extracted_vars)}"
    
    # Property: Variable names should be correct for each method
    expected_init_names = [f"self.{name}" for name, _ in init_vars]
    expected_other_names = [f"self.{name}" for name, _ in other_method_vars]
    
    actual_init_names = [var.name for var in init_extracted_vars]
    actual_other_names = [var.name for var in other_extracted_vars]
    
    assert set(actual_init_names) == set(expected_init_names)
    assert set(actual_other_names) == set(expected_other_names)


def test_no_member_variables_extraction():
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For classes with no member variable definitions, structural extraction
    should return an empty list.
    """
    code = """
class EmptyClass:
    def method1(self):
        x = 42  # Local variable, not self.x
        return x
    
    def method2(self):
        pass
"""
    
    extractor = StructuralExtractor()
    variables = extractor.extract_variables(ast.parse(code))
    
    # Property: Should extract no variables
    assert len(variables) == 0, f"Expected 0 variables for class without member vars, got {len(variables)}"


def test_complex_member_variable_patterns():
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For classes with various member variable assignment patterns, structural
    extraction should handle different assignment types.
    """
    code = """
class ComplexClass:
    def __init__(self):
        self.simple = 42
        self.string_val = "hello"
        self.list_val = [1, 2, 3]
        self.dict_val = {"key": "value"}
        self.none_val = None
        self.bool_val = True
    
    def set_computed(self):
        self.computed = self.simple + 10
        self.method_result = self.get_value()
    
    def get_value(self):
        return 100
"""
    
    extractor = StructuralExtractor()
    variables = extractor.extract_variables(ast.parse(code))
    
    # Property: Should extract all member variables
    expected_var_names = {
        "self.simple", "self.string_val", "self.list_val", 
        "self.dict_val", "self.none_val", "self.bool_val",
        "self.computed", "self.method_result"
    }
    
    actual_var_names = {var.name for var in variables}
    assert actual_var_names == expected_var_names, \
        f"Expected variables {expected_var_names}, got {actual_var_names}"
    
    # Property: Variables should be correctly associated with methods
    init_vars = [var for var in variables if var.defined_in == "ComplexClass.__init__"]
    computed_vars = [var for var in variables if var.defined_in == "ComplexClass.set_computed"]
    
    assert len(init_vars) == 6, f"Expected 6 variables in __init__, got {len(init_vars)}"
    assert len(computed_vars) == 2, f"Expected 2 variables in set_computed, got {len(computed_vars)}"


def test_nested_class_member_variables():
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For nested classes with member variables, structural extraction should
    correctly handle variables in both outer and inner classes.
    """
    code = """
class OuterClass:
    def __init__(self):
        self.outer_var = "outer"
    
    class InnerClass:
        def __init__(self):
            self.inner_var = "inner"
        
        def set_inner_data(self):
            self.inner_data = 42
    
    def set_outer_data(self):
        self.outer_data = 100
"""
    
    extractor = StructuralExtractor()
    variables = extractor.extract_variables(ast.parse(code))
    
    # Property: Should extract variables from both classes
    expected_var_names = {
        "self.outer_var", "self.outer_data", 
        "self.inner_var", "self.inner_data"
    }
    
    actual_var_names = {var.name for var in variables}
    assert actual_var_names == expected_var_names, \
        f"Expected variables {expected_var_names}, got {actual_var_names}"
    
    # Property: Variables should be correctly associated with their classes
    outer_vars = [var for var in variables if "OuterClass" in var.defined_in and "InnerClass" not in var.defined_in]
    inner_vars = [var for var in variables if "InnerClass" in var.defined_in]
    
    assert len(outer_vars) == 2, f"Expected 2 outer class variables, got {len(outer_vars)}"
    assert len(inner_vars) == 2, f"Expected 2 inner class variables, got {len(inner_vars)}"


def test_empty_code_variable_extraction():
    """
    **Feature: enhanced-graph-manager, Property 3: 成员变量提取完整性**
    **Validates: Requirements 1.3**
    
    For empty Python code, structural extraction should return an empty list.
    """
    extractor = StructuralExtractor()
    variables = extractor.extract_variables(ast.parse(""))
    
    # Property: Empty code should result in no variables
    assert len(variables) == 0, f"Expected 0 variables for empty code, got {len(variables)}"