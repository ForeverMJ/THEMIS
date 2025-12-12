"""Property-based tests for variable definition relationship correctness."""

import ast
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import List, Tuple

from src.enhanced_graph_manager.structural_extractor import StructuralExtractor
from src.enhanced_graph_manager.models import DependencyEdge


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


def generate_class_with_variable_definitions(class_name: str, methods_with_vars: List[Tuple[str, List[Tuple[str, any]]]]) -> str:
    """Generate Python class code with member variable definitions."""
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
def test_variable_definition_relationship_correctness(class_name, method_name, variables):
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For any variable definition, the system should create correct DEFINED_IN
    edges linking variables to their definition locations.
    """
    # Generate class code with variable definitions
    code = generate_class_with_variable_definitions(class_name, [(method_name, variables)])
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract definition edges using the structural extractor
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Property: Should extract exactly the same number of definition edges as variables
    assert len(definition_edges) == len(variables), \
        f"Expected {len(variables)} definition edges, got {len(definition_edges)}"
    
    # Property: All definition edges should be DependencyEdge instances
    for edge in definition_edges:
        assert isinstance(edge, DependencyEdge), f"Expected DependencyEdge, got {type(edge)}"
    
    # Property: All edges should have DEFINED_IN dependency type
    for edge in definition_edges:
        assert edge.dependency_type == "DEFINED_IN", \
            f"Expected dependency_type 'DEFINED_IN', got '{edge.dependency_type}'"
    
    # Property: All variables should be linked to the correct method
    expected_target = f"{class_name}.{method_name}"
    for edge in definition_edges:
        assert edge.target == expected_target, \
            f"Expected target '{expected_target}', got '{edge.target}'"
    
    # Property: All expected variable names should be present with self. prefix
    expected_sources = [f"self.{var_name}" for var_name, _ in variables]
    actual_sources = [edge.source for edge in definition_edges]
    assert set(actual_sources) == set(expected_sources), \
        f"Expected sources {set(expected_sources)}, got {set(actual_sources)}"
    
    # Property: Context should indicate class scope
    for edge in definition_edges:
        assert edge.context == f"class:{class_name}", \
            f"Expected context 'class:{class_name}', got '{edge.context}'"


@given(
    class_name=valid_class_name(),
    methods_with_vars=st.lists(
        st.tuples(
            valid_method_name(),
            st.lists(
                st.tuples(valid_variable_name(), simple_value()),
                min_size=1,
                max_size=2,
                unique_by=lambda x: x[0]
            )
        ),
        min_size=2,
        max_size=3,
        unique_by=lambda x: x[0]  # Ensure unique method names
    )
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_multiple_method_definition_relationships(class_name, methods_with_vars):
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For classes with multiple methods defining variables, the system should
    correctly associate each variable with its defining method.
    """
    # Generate class code with multiple methods and variables
    code = generate_class_with_variable_definitions(class_name, methods_with_vars)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract definition edges using the structural extractor
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Calculate expected total definition edges
    total_expected_edges = sum(len(variables) for _, variables in methods_with_vars)
    
    # Property: Should extract exactly the total number of definition edges
    assert len(definition_edges) == total_expected_edges, \
        f"Expected {total_expected_edges} definition edges, got {len(definition_edges)}"
    
    # Property: Each variable should be correctly associated with its method
    for method_name, variables in methods_with_vars:
        expected_target = f"{class_name}.{method_name}"
        
        # Find definition edges for this method
        method_edges = [edge for edge in definition_edges if edge.target == expected_target]
        
        # Property: Should have correct number of edges for this method
        assert len(method_edges) == len(variables), \
            f"Expected {len(variables)} definition edges for {method_name}, got {len(method_edges)}"
        
        # Property: Variable names should match
        expected_sources = [f"self.{var_name}" for var_name, _ in variables]
        actual_sources = [edge.source for edge in method_edges]
        assert set(actual_sources) == set(expected_sources), \
            f"Expected sources {set(expected_sources)} for {method_name}, got {set(actual_sources)}"


@given(
    class_name=valid_class_name(),
    init_vars=st.lists(
        st.tuples(valid_variable_name(), simple_value()),
        min_size=1,
        max_size=4,
        unique_by=lambda x: x[0]
    ),
    other_method_vars=st.lists(
        st.tuples(valid_variable_name(), simple_value()),
        min_size=1,
        max_size=3,
        unique_by=lambda x: x[0]
    )
)
def test_init_vs_other_method_definition_relationships(class_name, init_vars, other_method_vars):
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For classes with variables defined in __init__ and other methods, the system
    should correctly distinguish between definition locations.
    """
    # Ensure no variable name conflicts between methods
    init_var_names = [name for name, _ in init_vars]
    other_var_names = [name for name, _ in other_method_vars]
    assume(not set(init_var_names).intersection(set(other_var_names)))
    
    # Generate class code with variables in both __init__ and another method
    methods_with_vars = [
        ("__init__", init_vars),
        ("update_state", other_method_vars)
    ]
    code = generate_class_with_variable_definitions(class_name, methods_with_vars)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract definition edges using the structural extractor
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Property: Should extract definition edges from both methods
    init_target = f"{class_name}.__init__"
    other_target = f"{class_name}.update_state"
    
    init_edges = [edge for edge in definition_edges if edge.target == init_target]
    other_edges = [edge for edge in definition_edges if edge.target == other_target]
    
    # Property: Correct number of edges for each method
    assert len(init_edges) == len(init_vars), \
        f"Expected {len(init_vars)} definition edges for __init__, got {len(init_edges)}"
    assert len(other_edges) == len(other_method_vars), \
        f"Expected {len(other_method_vars)} definition edges for update_state, got {len(other_edges)}"
    
    # Property: Variable names should be correct for each method
    expected_init_sources = [f"self.{name}" for name, _ in init_vars]
    expected_other_sources = [f"self.{name}" for name, _ in other_method_vars]
    
    actual_init_sources = [edge.source for edge in init_edges]
    actual_other_sources = [edge.source for edge in other_edges]
    
    assert set(actual_init_sources) == set(expected_init_sources)
    assert set(actual_other_sources) == set(expected_other_sources)


def test_complex_variable_definition_patterns():
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For classes with various variable definition patterns, the system should
    handle different assignment contexts correctly.
    """
    code = """
class ComplexClass:
    def __init__(self):
        self.basic_var = 42
        self.string_var = "hello"
        self.list_var = [1, 2, 3]
    
    def conditional_definitions(self):
        if True:
            self.conditional_var = "conditional"
        
        for i in range(3):
            self.loop_var = i
    
    def computed_definitions(self):
        self.computed_var = self.basic_var + 10
        self.method_result = self.get_value()
    
    def get_value(self):
        return 100
"""
    
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Property: Should extract all variable definitions
    expected_var_names = {
        "self.basic_var", "self.string_var", "self.list_var",
        "self.conditional_var", "self.loop_var",
        "self.computed_var", "self.method_result"
    }
    
    actual_var_names = {edge.source for edge in definition_edges}
    assert actual_var_names == expected_var_names, \
        f"Expected variables {expected_var_names}, got {actual_var_names}"
    
    # Property: Variables should be correctly associated with methods
    init_edges = [edge for edge in definition_edges if edge.target == "ComplexClass.__init__"]
    conditional_edges = [edge for edge in definition_edges if edge.target == "ComplexClass.conditional_definitions"]
    computed_edges = [edge for edge in definition_edges if edge.target == "ComplexClass.computed_definitions"]
    
    assert len(init_edges) == 3, f"Expected 3 variables in __init__, got {len(init_edges)}"
    assert len(conditional_edges) == 2, f"Expected 2 variables in conditional_definitions, got {len(conditional_edges)}"
    assert len(computed_edges) == 2, f"Expected 2 variables in computed_definitions, got {len(computed_edges)}"


def test_nested_class_variable_definitions():
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For nested classes with variable definitions, the system should correctly
    handle variables in both outer and inner classes.
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
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Property: Should extract definition edges from both classes
    expected_var_names = {
        "self.outer_var", "self.outer_data", 
        "self.inner_var", "self.inner_data"
    }
    
    actual_var_names = {edge.source for edge in definition_edges}
    assert actual_var_names == expected_var_names, \
        f"Expected variables {expected_var_names}, got {actual_var_names}"
    
    # Property: Variables should be correctly associated with their classes
    outer_edges = [edge for edge in definition_edges if "OuterClass" in edge.target and "InnerClass" not in edge.target]
    inner_edges = [edge for edge in definition_edges if "InnerClass" in edge.target]
    
    assert len(outer_edges) == 2, f"Expected 2 outer class variables, got {len(outer_edges)}"
    assert len(inner_edges) == 2, f"Expected 2 inner class variables, got {len(inner_edges)}"
    
    # Property: Context should reflect the correct class
    for edge in outer_edges:
        assert edge.context == "class:OuterClass", \
            f"Expected context 'class:OuterClass', got '{edge.context}'"
    
    for edge in inner_edges:
        assert edge.context == "class:InnerClass", \
            f"Expected context 'class:InnerClass', got '{edge.context}'"


def test_no_variable_definitions():
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For classes with no member variable definitions, the system should
    return an empty list.
    """
    code = """
class EmptyClass:
    def method1(self):
        x = 42  # Local variable, not self.x
        return x
    
    def method2(self):
        pass

def standalone_function():
    y = 100  # Not a member variable
    return y
"""
    
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Property: Should extract no definition edges
    assert len(definition_edges) == 0, \
        f"Expected 0 definition edges for code without member variables, got {len(definition_edges)}"


def test_global_scope_variable_definitions():
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For functions outside classes that define variables, the system should
    handle global scope correctly.
    """
    code = """
def global_function():
    # This should not create definition edges since it's not self.xxx
    x = 42
    y = "hello"

class TestClass:
    def method(self):
        self.member_var = 100  # This should create a definition edge
"""
    
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(code))
    
    # Property: Should only extract definition edges for member variables
    assert len(definition_edges) == 1, \
        f"Expected 1 definition edge for member variable, got {len(definition_edges)}"
    
    edge = definition_edges[0]
    assert edge.source == "self.member_var", \
        f"Expected source 'self.member_var', got '{edge.source}'"
    assert edge.target == "TestClass.method", \
        f"Expected target 'TestClass.method', got '{edge.target}'"
    assert edge.context == "class:TestClass", \
        f"Expected context 'class:TestClass', got '{edge.context}'"


def test_empty_code_definition_extraction():
    """
    **Feature: enhanced-graph-manager, Property 6: 变量定义关系正确性**
    **Validates: Requirements 1.6**
    
    For empty Python code, definition extraction should return an empty list.
    """
    extractor = StructuralExtractor()
    definition_edges = extractor.extract_definition_edges(ast.parse(""))
    
    # Property: Empty code should result in no definition edges
    assert len(definition_edges) == 0, \
        f"Expected 0 definition edges for empty code, got {len(definition_edges)}"