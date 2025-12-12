"""Property-based tests for class extraction completeness."""

import ast
import pytest
from hypothesis import given, strategies as st, assume
from typing import List

from src.enhanced_graph_manager.structural_extractor import StructuralExtractor
from src.enhanced_graph_manager.models import ClassNode


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


def valid_base_class_list():
    """Generate valid base class lists."""
    return st.lists(
        valid_class_name(),
        min_size=0,
        max_size=3,
        unique=True
    )


def valid_method_list():
    """Generate valid method lists."""
    return st.lists(
        valid_method_name(),
        min_size=0,
        max_size=5,
        unique=True
    )


def generate_class_code(name: str, bases: List[str], methods: List[str], 
                       has_docstring: bool = False) -> str:
    """Generate Python class code."""
    base_str = f"({', '.join(bases)})" if bases else ""
    
    docstring = '    """Test class docstring."""\n' if has_docstring else ""
    
    # Generate method definitions
    method_codes = []
    for method_name in methods:
        method_codes.append(f"    def {method_name}(self):\n        pass")
    
    # If no methods, add pass statement
    if not method_codes and not has_docstring:
        method_codes.append("    pass")
    
    methods_str = "\n".join(method_codes) if method_codes else ""
    
    return f"""class {name}{base_str}:
{docstring}{methods_str}
"""


@given(
    name=valid_class_name(),
    bases=valid_base_class_list(),
    methods=valid_method_list(),
    has_docstring=st.booleans()
)
def test_class_extraction_completeness(name, bases, methods, has_docstring):
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For any Python code containing class definitions, structural extraction
    should create a ClassNode for each class with correct name and inheritance
    relationships.
    """
    # Generate class code
    code = generate_class_code(name, bases, methods, has_docstring)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract classes using the structural extractor
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Should extract exactly one class
    assert len(classes) == 1, f"Expected 1 class, got {len(classes)}"
    
    cls = classes[0]
    
    # Property: Class should be a ClassNode instance
    assert isinstance(cls, ClassNode), f"Expected ClassNode, got {type(cls)}"
    
    # Property: Class name should match exactly
    assert cls.name == name, f"Expected name '{name}', got '{cls.name}'"
    
    # Property: Base classes should match exactly
    assert cls.bases == bases, f"Expected bases {bases}, got {cls.bases}"
    
    # Property: Method names should match exactly
    assert set(cls.methods) == set(methods), f"Expected methods {set(methods)}, got {set(cls.methods)}"
    
    # Property: Docstring should be correctly extracted
    if has_docstring:
        assert cls.docstring == "Test class docstring.", f"Expected docstring, got '{cls.docstring}'"
    else:
        assert cls.docstring is None, f"Expected no docstring, got '{cls.docstring}'"
    
    # Property: Line number should be positive
    assert cls.line_number > 0, f"Expected positive line number, got {cls.line_number}"


@given(
    classes=st.lists(
        st.tuples(
            valid_class_name(),
            valid_base_class_list(),
            valid_method_list()
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]  # Ensure unique class names
    )
)
def test_multiple_class_extraction_completeness(classes):
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For any Python code containing multiple class definitions, structural 
    extraction should create a ClassNode for each class.
    """
    # Generate code with multiple classes
    code_parts = []
    expected_names = []
    
    for name, bases, methods in classes:
        class_code = generate_class_code(name, bases, methods)
        code_parts.append(class_code)
        expected_names.append(name)
    
    code = "\n\n".join(code_parts)
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract classes using the structural extractor
    extractor = StructuralExtractor()
    extracted_classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Should extract exactly the same number of classes as defined
    assert len(extracted_classes) == len(classes), \
        f"Expected {len(classes)} classes, got {len(extracted_classes)}"
    
    # Property: All class names should be present
    extracted_names = [cls.name for cls in extracted_classes]
    assert set(extracted_names) == set(expected_names), \
        f"Expected names {set(expected_names)}, got {set(extracted_names)}"
    
    # Property: Each extracted class should be a ClassNode
    for cls in extracted_classes:
        assert isinstance(cls, ClassNode), f"Expected ClassNode, got {type(cls)}"
        assert cls.line_number > 0, f"Expected positive line number, got {cls.line_number}"


@given(
    parent_class=valid_class_name(),
    child_class=valid_class_name(),
    additional_bases=st.lists(valid_class_name(), min_size=0, max_size=2, unique=True)
)
def test_inheritance_extraction_completeness(parent_class, child_class, additional_bases):
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For any Python code containing class inheritance, structural extraction
    should correctly capture inheritance relationships.
    """
    # Ensure child class name is different from parent and additional bases
    assume(child_class != parent_class)
    assume(child_class not in additional_bases)
    assume(parent_class not in additional_bases)
    
    # Create list of all base classes
    all_bases = [parent_class] + additional_bases
    
    # Generate inheritance code
    code = generate_class_code(child_class, all_bases, ["child_method"])
    
    # Ensure the generated code is syntactically valid
    try:
        ast.parse(code)
    except SyntaxError:
        assume(False)  # Skip invalid code
    
    # Extract classes using the structural extractor
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Should extract exactly one class
    assert len(classes) == 1, f"Expected 1 class, got {len(classes)}"
    
    cls = classes[0]
    
    # Property: Class should have correct inheritance
    assert cls.name == child_class, f"Expected class name '{child_class}', got '{cls.name}'"
    assert set(cls.bases) == set(all_bases), f"Expected bases {set(all_bases)}, got {set(cls.bases)}"
    
    # Property: Should preserve order of base classes
    assert cls.bases == all_bases, f"Expected bases in order {all_bases}, got {cls.bases}"


def test_empty_class_extraction():
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For Python code with empty classes, structural extraction should
    correctly extract the class with empty methods list.
    """
    code = """
class EmptyClass:
    pass
"""
    
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Should extract exactly one class
    assert len(classes) == 1, f"Expected 1 class, got {len(classes)}"
    
    cls = classes[0]
    
    # Property: Class should have correct attributes
    assert cls.name == "EmptyClass"
    assert cls.bases == []
    assert cls.methods == []
    assert cls.docstring is None
    assert cls.line_number > 0


def test_class_with_only_docstring_extraction():
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For Python code with classes containing only docstrings, structural 
    extraction should correctly extract the docstring.
    """
    code = """
class DocumentedClass:
    '''This class has only a docstring.'''
"""
    
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Should extract exactly one class
    assert len(classes) == 1, f"Expected 1 class, got {len(classes)}"
    
    cls = classes[0]
    
    # Property: Class should have correct attributes
    assert cls.name == "DocumentedClass"
    assert cls.bases == []
    assert cls.methods == []
    assert cls.docstring == "This class has only a docstring."
    assert cls.line_number > 0


def test_nested_class_extraction():
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For Python code with nested classes, structural extraction should
    extract both outer and inner classes.
    """
    code = """
class OuterClass:
    def outer_method(self):
        pass
    
    class InnerClass:
        def inner_method(self):
            pass
"""
    
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Should extract both classes
    assert len(classes) == 2, f"Expected 2 classes, got {len(classes)}"
    
    class_names = [cls.name for cls in classes]
    assert "OuterClass" in class_names
    assert "InnerClass" in class_names
    
    # Property: Each class should have correct methods
    for cls in classes:
        if cls.name == "OuterClass":
            assert "outer_method" in cls.methods
        elif cls.name == "InnerClass":
            assert "inner_method" in cls.methods


def test_empty_code_class_extraction():
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For empty Python code, structural extraction should return an empty list.
    """
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(""))
    
    # Property: Empty code should result in no classes
    assert len(classes) == 0, f"Expected 0 classes for empty code, got {len(classes)}"


def test_code_without_classes_extraction():
    """
    **Feature: enhanced-graph-manager, Property 2: 类提取完整性**
    **Validates: Requirements 1.2**
    
    For Python code without class definitions, structural extraction 
    should return an empty list.
    """
    code = """
def function():
    pass

x = 42
y = "hello"
"""
    
    extractor = StructuralExtractor()
    classes = extractor.extract_classes(ast.parse(code))
    
    # Property: Code without classes should result in no classes
    assert len(classes) == 0, f"Expected 0 classes for code without classes, got {len(classes)}"