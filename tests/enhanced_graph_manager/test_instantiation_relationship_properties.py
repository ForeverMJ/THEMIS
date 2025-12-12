"""Property-based tests for instantiation relationship correctness."""

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


def generate_function_with_instantiations(function_name: str, class_names: List[str]) -> str:
    """Generate Python function code with class instantiations."""
    instantiation_statements = []
    for i, class_name in enumerate(class_names):
        var_name = f"obj{i}"
        instantiation_statements.append(f"    {var_name} = {class_name}()")
    
    if not instantiation_statements:
        instantiation_statements.append("    pass")
    
    return f"""def {function_name}():
{chr(10).join(instantiation_statements)}
"""


def generate_method_with_instantiations(class_name: str, method_name: str, instantiated_classes: List[str]) -> str:
    """Generate Python class method with class instantiations."""
    instantiation_statements = []
    for i, instantiated_class in enumerate(instantiated_classes):
        var_name = f"obj{i}"
        instantiation_statements.append(f"        {var_name} = {instantiated_class}()")
    
    if not instantiation_statements:
        instantiation_statements.append("        pass")
    
    return f"""class {class_name}:
    def {method_name}(self):
{chr(10).join(instantiation_statements)}
"""


@given(
    function_name=valid_function_name(),
    class_names=st.lists(
        valid_class_name(),
        min_size=1,
        max_size=5,
        unique=True
    )
)
def test_function_instantiation_relationship_correctness(function_name, class_names):
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    For any code containing class instantiations, the system should create
    correct INSTANTIATES edges between the instantiating function and classes.
    """
    # Generate function code with instantiations
    code = generate_function_with_instantiations(function_name, class_names)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract instantiation edges using the structural extractor
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should extract exactly the same number of instantiations as defined
    assert len(instantiation_edges) == len(class_names), \
        f"Expected {len(class_names)} instantiation edges, got {len(instantiation_edges)}"
    
    # Property: All instantiation edges should be CallEdge instances
    for edge in instantiation_edges:
        assert isinstance(edge, CallEdge), f"Expected CallEdge, got {type(edge)}"
    
    # Property: All instantiations should have the correct caller (function)
    for edge in instantiation_edges:
        assert edge.caller == function_name, \
            f"Expected caller '{function_name}', got '{edge.caller}'"
    
    # Property: All expected classes should be instantiated
    actual_callees = [edge.callee for edge in instantiation_edges]
    assert set(actual_callees) == set(class_names), \
        f"Expected instantiated classes {set(class_names)}, got {set(actual_callees)}"
    
    # Property: Line numbers should be positive
    for edge in instantiation_edges:
        assert edge.line_number > 0, f"Expected positive line number, got {edge.line_number}"


@given(
    class_name=valid_class_name(),
    method_name=valid_function_name(),
    instantiated_classes=st.lists(
        valid_class_name(),
        min_size=1,
        max_size=4,
        unique=True
    )
)
def test_method_instantiation_relationship_correctness(class_name, method_name, instantiated_classes):
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    For any class method containing instantiations, the system should create
    correct INSTANTIATES edges with the method as the caller.
    """
    # Ensure the containing class is not in the instantiated classes
    assume(class_name not in instantiated_classes)
    
    # Generate class method code with instantiations
    code = generate_method_with_instantiations(class_name, method_name, instantiated_classes)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract instantiation edges using the structural extractor
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should extract exactly the same number of instantiations as defined
    assert len(instantiation_edges) == len(instantiated_classes), \
        f"Expected {len(instantiated_classes)} instantiation edges, got {len(instantiation_edges)}"
    
    # Property: All instantiations should have the correct caller (method name)
    for edge in instantiation_edges:
        assert edge.caller == method_name, \
            f"Expected caller '{method_name}', got '{edge.caller}'"
    
    # Property: All expected classes should be instantiated
    actual_callees = [edge.callee for edge in instantiation_edges]
    assert set(actual_callees) == set(instantiated_classes), \
        f"Expected instantiated classes {set(instantiated_classes)}, got {set(actual_callees)}"


@given(
    functions=st.lists(
        st.tuples(
            valid_function_name(),
            st.lists(valid_class_name(), min_size=0, max_size=3, unique=True)
        ),
        min_size=2,
        max_size=4,
        unique_by=lambda x: x[0]  # Ensure unique function names
    )
)
def test_multiple_function_instantiation_relationships(functions):
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    For code with multiple functions making instantiations, the system should
    correctly track all instantiation relationships without confusion.
    """
    # Generate code with multiple functions
    code_parts = []
    expected_edges = []
    
    for function_name, class_names in functions:
        if class_names:
            code_parts.append(generate_function_with_instantiations(function_name, class_names))
            for class_name in class_names:
                expected_edges.append((function_name, class_name))
        else:
            # Add function with no instantiations
            code_parts.append(f"def {function_name}():\n    pass")
    
    code = "\n\n".join(code_parts)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract instantiation edges using the structural extractor
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should extract exactly the expected number of instantiations
    assert len(instantiation_edges) == len(expected_edges), \
        f"Expected {len(expected_edges)} instantiation edges, got {len(instantiation_edges)}"
    
    # Property: All expected instantiation relationships should be present
    actual_edges = [(edge.caller, edge.callee) for edge in instantiation_edges]
    assert set(actual_edges) == set(expected_edges), \
        f"Expected edges {set(expected_edges)}, got {set(actual_edges)}"


def test_instantiation_vs_function_call_distinction():
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    The system should correctly distinguish between class instantiations
    (uppercase names) and function calls (lowercase names).
    """
    code = """
def test_function():
    # These should be detected as instantiations (uppercase)
    obj1 = MyClass()
    obj2 = AnotherClass()
    
    # These should NOT be detected as instantiations (lowercase)
    result1 = my_function()
    result2 = another_function()
    
    # Mixed case - should detect the uppercase ones
    obj3 = ThirdClass()
    result3 = third_function()
"""
    
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should only detect uppercase class instantiations
    expected_classes = {"MyClass", "AnotherClass", "ThirdClass"}
    actual_classes = {edge.callee for edge in instantiation_edges}
    
    assert actual_classes == expected_classes, \
        f"Expected instantiated classes {expected_classes}, got {actual_classes}"
    
    # Property: Should not detect lowercase function calls as instantiations
    lowercase_functions = {"my_function", "another_function", "third_function"}
    assert not any(edge.callee in lowercase_functions for edge in instantiation_edges), \
        "Lowercase function calls should not be detected as instantiations"


def test_qualified_class_instantiation():
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    The system should handle qualified class names (module.ClassName) correctly.
    """
    code = """
def create_objects():
    # Simple class instantiation
    obj1 = SimpleClass()
    
    # Qualified class instantiation (should detect the class name part)
    obj2 = module.QualifiedClass()
    obj3 = package.submodule.AnotherClass()
    
    # Mixed with function calls
    result = module.function_call()  # Should not be detected
"""
    
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should detect both simple and qualified class instantiations
    actual_classes = {edge.callee for edge in instantiation_edges}
    
    # Should include simple class and the class part of qualified names
    assert "SimpleClass" in actual_classes, "Should detect simple class instantiation"
    
    # For qualified names, should detect the class name part
    qualified_classes = [callee for callee in actual_classes if callee in {"QualifiedClass", "AnotherClass"}]
    assert len(qualified_classes) >= 1, "Should detect at least one qualified class instantiation"
    
    # Should not detect lowercase function calls
    assert "function_call" not in actual_classes, "Should not detect function calls as instantiations"


def test_constructor_with_arguments():
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    The system should detect class instantiations regardless of constructor arguments.
    """
    code = """
def create_objects():
    # No arguments
    obj1 = EmptyClass()
    
    # Positional arguments
    obj2 = ClassWithArgs(1, 2, 3)
    
    # Keyword arguments
    obj3 = ClassWithKwargs(name="test", value=42)
    
    # Mixed arguments
    obj4 = ComplexClass(1, name="test", items=[1, 2, 3])
"""
    
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should detect all class instantiations regardless of arguments
    expected_classes = {"EmptyClass", "ClassWithArgs", "ClassWithKwargs", "ComplexClass"}
    actual_classes = {edge.callee for edge in instantiation_edges}
    
    assert actual_classes == expected_classes, \
        f"Expected classes {expected_classes}, got {actual_classes}"
    
    # Property: All should have the same caller
    for edge in instantiation_edges:
        assert edge.caller == "create_objects", \
            f"Expected caller 'create_objects', got '{edge.caller}'"


def test_no_instantiations():
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    For functions with no class instantiations, the system should return an empty list.
    """
    code = """
def function_with_no_instantiations():
    x = 42
    y = x + 10
    result = some_function()  # Function call, not instantiation
    return result

def another_function():
    pass
"""
    
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should extract no instantiation edges
    assert len(instantiation_edges) == 0, \
        f"Expected 0 instantiation edges for functions with no instantiations, got {len(instantiation_edges)}"


def test_nested_class_instantiation():
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    The system should handle instantiations within nested contexts correctly.
    """
    code = """
def outer_function():
    obj1 = OuterClass()
    
    def inner_function():
        obj2 = InnerClass()
        return obj2
    
    if True:
        obj3 = ConditionalClass()
    
    for i in range(3):
        obj4 = LoopClass()
    
    return inner_function()
"""
    
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(code))
    
    # Property: Should detect instantiations in different contexts
    expected_classes = {"OuterClass", "InnerClass", "ConditionalClass", "LoopClass"}
    actual_classes = {edge.callee for edge in instantiation_edges}
    
    assert actual_classes == expected_classes, \
        f"Expected classes {expected_classes}, got {actual_classes}"
    
    # Property: Should correctly identify the calling functions
    outer_edges = [edge for edge in instantiation_edges if edge.caller == "outer_function"]
    inner_edges = [edge for edge in instantiation_edges if edge.caller == "inner_function"]
    
    # outer_function should have instantiations (OuterClass, ConditionalClass, LoopClass)
    assert len(outer_edges) >= 1, "outer_function should have at least one instantiation"
    
    # inner_function should have InnerClass instantiation
    inner_classes = {edge.callee for edge in inner_edges}
    assert "InnerClass" in inner_classes, "inner_function should instantiate InnerClass"


def test_empty_code_instantiation_extraction():
    """
    **Feature: enhanced-graph-manager, Property 5: 实例化关系正确性**
    **Validates: Requirements 1.5**
    
    For empty Python code, instantiation extraction should return an empty list.
    """
    extractor = StructuralExtractor()
    instantiation_edges = extractor.extract_instantiation_edges(ast.parse(""))
    
    # Property: Empty code should result in no instantiation edges
    assert len(instantiation_edges) == 0, \
        f"Expected 0 instantiation edges for empty code, got {len(instantiation_edges)}"