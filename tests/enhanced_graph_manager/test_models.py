"""Tests for Enhanced GraphManager data models."""

import pytest
from hypothesis import given, strategies as st

from src.enhanced_graph_manager.models import (
    FunctionNode,
    ClassNode,
    VariableNode,
    RequirementNode,
    CallEdge,
    DependencyEdge,
    ViolationEdge,
)


class TestDataModels:
    """Test the core data model classes."""
    
    def test_function_node_creation(self):
        """Test basic FunctionNode creation."""
        node = FunctionNode(
            name="test_func",
            args=["self", "param1"],
            return_type="str",
            docstring="Test function",
            line_number=10
        )
        assert node.name == "test_func"
        assert node.args == ["self", "param1"]
        assert node.return_type == "str"
        assert node.docstring == "Test function"
        assert node.line_number == 10
    
    def test_class_node_creation(self):
        """Test basic ClassNode creation."""
        node = ClassNode(
            name="TestClass",
            bases=["BaseClass"],
            methods=["method1", "method2"],
            docstring="Test class",
            line_number=5
        )
        assert node.name == "TestClass"
        assert node.bases == ["BaseClass"]
        assert node.methods == ["method1", "method2"]
        assert node.docstring == "Test class"
        assert node.line_number == 5
    
    def test_variable_node_creation(self):
        """Test basic VariableNode creation."""
        node = VariableNode(
            name="self.test_var",
            var_type="int",
            defined_in="TestClass.method1",
            line_number=15
        )
        assert node.name == "self.test_var"
        assert node.var_type == "int"
        assert node.defined_in == "TestClass.method1"
        assert node.line_number == 15
    
    def test_requirement_node_creation(self):
        """Test basic RequirementNode creation."""
        node = RequirementNode(
            id="REQ-001",
            text="System shall validate input",
            priority=1,
            testable=True
        )
        assert node.id == "REQ-001"
        assert node.text == "System shall validate input"
        assert node.priority == 1
        assert node.testable is True
    
    def test_call_edge_creation(self):
        """Test basic CallEdge creation."""
        edge = CallEdge(
            caller="func1",
            callee="func2",
            line_number=20
        )
        assert edge.caller == "func1"
        assert edge.callee == "func2"
        assert edge.line_number == 20
    
    def test_dependency_edge_creation(self):
        """Test basic DependencyEdge creation."""
        edge = DependencyEdge(
            source="var1",
            target="var2",
            dependency_type="DEPENDS_ON",
            context="function_scope"
        )
        assert edge.source == "var1"
        assert edge.target == "var2"
        assert edge.dependency_type == "DEPENDS_ON"
        assert edge.context == "function_scope"
    
    def test_violation_edge_creation(self):
        """Test basic ViolationEdge creation."""
        edge = ViolationEdge(
            requirement="REQ-001",
            code_node="func1",
            status="VIOLATES",
            reason="Missing validation",
            confidence=0.85
        )
        assert edge.requirement == "REQ-001"
        assert edge.code_node == "func1"
        assert edge.status == "VIOLATES"
        assert edge.reason == "Missing validation"
        assert edge.confidence == 0.85