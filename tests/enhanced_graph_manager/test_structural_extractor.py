"""Tests for the StructuralExtractor class."""

import pytest
import networkx as nx

from src.enhanced_graph_manager.structural_extractor import StructuralExtractor
from src.enhanced_graph_manager.models import FunctionNode, ClassNode, VariableNode


class TestStructuralExtractor:
    """Test the StructuralExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = StructuralExtractor()
    
    def test_extract_simple_function(self):
        """Test extraction of a simple function."""
        code = """
def hello_world():
    print("Hello, World!")
"""
        functions = self.extractor.extract_functions(ast.parse(code))
        
        assert len(functions) == 1
        func = functions[0]
        assert func.name == "hello_world"
        assert func.args == []
        assert func.return_type is None
        assert func.line_number == 2
    
    def test_extract_function_with_args_and_return_type(self):
        """Test extraction of function with arguments and return type."""
        code = """
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers together.'''
    return a + b
"""
        functions = self.extractor.extract_functions(ast.parse(code))
        
        assert len(functions) == 1
        func = functions[0]
        assert func.name == "add_numbers"
        assert func.args == ["a", "b"]
        assert func.return_type == "int"
        assert func.docstring == "Add two numbers together."
        assert func.line_number == 2
    
    def test_extract_simple_class(self):
        """Test extraction of a simple class."""
        code = """
class MyClass:
    '''A simple test class.'''
    
    def method1(self):
        pass
    
    def method2(self):
        pass
"""
        classes = self.extractor.extract_classes(ast.parse(code))
        
        assert len(classes) == 1
        cls = classes[0]
        assert cls.name == "MyClass"
        assert cls.bases == []
        assert set(cls.methods) == {"method1", "method2"}
        assert cls.docstring == "A simple test class."
        assert cls.line_number == 2
    
    def test_extract_class_with_inheritance(self):
        """Test extraction of class with inheritance."""
        code = """
class ChildClass(ParentClass, Mixin):
    def child_method(self):
        pass
"""
        classes = self.extractor.extract_classes(ast.parse(code))
        
        assert len(classes) == 1
        cls = classes[0]
        assert cls.name == "ChildClass"
        assert cls.bases == ["ParentClass", "Mixin"]
        assert cls.methods == ["child_method"]
    
    def test_extract_member_variables(self):
        """Test extraction of member variables (self.xxx patterns)."""
        code = """
class TestClass:
    def __init__(self):
        self.name = "test"
        self.value = 42
        self.items = []
    
    def set_data(self):
        self.data = "some data"
"""
        variables = self.extractor.extract_variables(ast.parse(code))
        
        # Should find 4 member variables
        assert len(variables) == 4
        
        var_names = [var.name for var in variables]
        assert "self.name" in var_names
        assert "self.value" in var_names
        assert "self.items" in var_names
        assert "self.data" in var_names
        
        # Check that they're properly associated with their defining methods
        for var in variables:
            if var.name in ["self.name", "self.value", "self.items"]:
                assert var.defined_in == "TestClass.__init__"
            elif var.name == "self.data":
                assert var.defined_in == "TestClass.set_data"
    
    def test_extract_call_edges(self):
        """Test extraction of function call relationships."""
        code = """
def caller():
    callee1()
    obj.method()
    self.helper()

def callee1():
    pass
"""
        call_edges = self.extractor.extract_call_edges(ast.parse(code))
        
        assert len(call_edges) == 3
        
        callers = [edge.caller for edge in call_edges]
        callees = [edge.callee for edge in call_edges]
        
        assert all(caller == "caller" for caller in callers)
        assert "callee1" in callees
        assert "obj.method" in callees
        assert "self.helper" in callees
    
    def test_extract_instantiation_edges(self):
        """Test extraction of class instantiation relationships."""
        code = """
def create_objects():
    obj1 = MyClass()
    obj2 = AnotherClass(param=1)
    result = some_function()  # This should not be detected as instantiation
"""
        instantiation_edges = self.extractor.extract_instantiation_edges(ast.parse(code))
        
        assert len(instantiation_edges) == 2
        
        callees = [edge.callee for edge in instantiation_edges]
        assert "MyClass" in callees
        assert "AnotherClass" in callees
        assert "some_function" not in callees  # lowercase function should not be detected
    
    def test_extract_definition_edges(self):
        """Test extraction of variable definition relationships."""
        code = """
class TestClass:
    def __init__(self):
        self.name = "test"
        self.value = 42
    
    def update(self):
        self.status = "updated"
"""
        definition_edges = self.extractor.extract_definition_edges(ast.parse(code))
        
        assert len(definition_edges) == 3
        
        # Check that variables are linked to their defining methods
        for edge in definition_edges:
            assert edge.dependency_type == "DEFINED_IN"
            assert edge.context == "class:TestClass"
            
            if edge.source == "self.name" or edge.source == "self.value":
                assert edge.target == "TestClass.__init__"
            elif edge.source == "self.status":
                assert edge.target == "TestClass.update"
    
    def test_extract_complete_structure(self):
        """Test complete structure extraction integration."""
        code = """
class Calculator:
    '''A simple calculator class.'''
    
    def __init__(self):
        self.result = 0
    
    def add(self, value: int) -> int:
        '''Add a value to the result.'''
        self.result += value
        return self.result
    
    def multiply(self, value: int) -> int:
        self.result *= value
        return self.result
    
    def reset(self):
        self.result = 0
        self.clear_history()
    
    def clear_history(self):
        pass

def create_calculator():
    calc = Calculator()
    return calc
"""
        graph = self.extractor.extract_structure(code)
        
        # Check that we have the expected nodes
        assert graph.has_node("Calculator")
        assert graph.has_node("add")
        assert graph.has_node("multiply")
        assert graph.has_node("reset")
        assert graph.has_node("clear_history")
        assert graph.has_node("create_calculator")
        assert graph.has_node("self.result")
        
        # Check node types
        assert graph.nodes["Calculator"]["type"] == "class"
        assert graph.nodes["add"]["type"] == "function"
        assert graph.nodes["self.result"]["type"] == "variable"
        
        # Check some edges exist
        assert graph.has_edge("reset", "self.clear_history")  # CALLS edge
        assert graph.has_edge("create_calculator", "Calculator")  # INSTANTIATES edge
        assert graph.has_edge("self.result", "Calculator.__init__")  # DEFINED_IN edge
    
    def test_syntax_error_handling(self):
        """Test that syntax errors are properly handled."""
        invalid_code = """
def broken_function(
    # Missing closing parenthesis and colon
"""
        with pytest.raises(SyntaxError):
            self.extractor.extract_structure(invalid_code)


# Import ast module for the tests
import ast