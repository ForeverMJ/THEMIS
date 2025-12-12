"""Property-based tests for API return value correctness."""

import pytest
from hypothesis import given, strategies as st
import networkx as nx

from src.enhanced_graph_manager import EnhancedGraphManager
from src.enhanced_graph_manager.models import (
    FunctionNode,
    ClassNode,
    VariableNode,
    RequirementNode,
)


# Hypothesis strategies for generating test data
function_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'))
class_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'))
variable_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_.'))
line_numbers = st.integers(min_value=1, max_value=10000)
priorities = st.integers(min_value=1, max_value=10)


@given(
    name=function_names,
    args=st.lists(function_names, min_size=0, max_size=10),
    return_type=st.one_of(st.none(), function_names),
    docstring=st.one_of(st.none(), st.text(max_size=200)),
    line_number=line_numbers
)
def test_function_node_api_correctness(name, args, return_type, docstring, line_number):
    """
    **Feature: enhanced-graph-manager, Property 15: API返回值正确性**
    **Validates: Requirements 5.1**
    
    For any valid function node data, creating a FunctionNode should return
    an object with the correct type and all specified attributes accessible.
    """
    node = FunctionNode(
        name=name,
        args=args,
        return_type=return_type,
        docstring=docstring,
        line_number=line_number
    )
    
    # Verify the node is of correct type
    assert isinstance(node, FunctionNode)
    
    # Verify all attributes are correctly set and accessible
    assert node.name == name
    assert node.args == args
    assert node.return_type == return_type
    assert node.docstring == docstring
    assert node.line_number == line_number
    
    # Verify args is a list
    assert isinstance(node.args, list)
    
    # Verify line_number is an integer
    assert isinstance(node.line_number, int)


@given(
    name=class_names,
    bases=st.lists(class_names, min_size=0, max_size=5),
    methods=st.lists(function_names, min_size=0, max_size=20),
    docstring=st.one_of(st.none(), st.text(max_size=200)),
    line_number=line_numbers
)
def test_class_node_api_correctness(name, bases, methods, docstring, line_number):
    """
    **Feature: enhanced-graph-manager, Property 15: API返回值正确性**
    **Validates: Requirements 5.1**
    
    For any valid class node data, creating a ClassNode should return
    an object with the correct type and all specified attributes accessible.
    """
    node = ClassNode(
        name=name,
        bases=bases,
        methods=methods,
        docstring=docstring,
        line_number=line_number
    )
    
    # Verify the node is of correct type
    assert isinstance(node, ClassNode)
    
    # Verify all attributes are correctly set and accessible
    assert node.name == name
    assert node.bases == bases
    assert node.methods == methods
    assert node.docstring == docstring
    assert node.line_number == line_number
    
    # Verify bases and methods are lists
    assert isinstance(node.bases, list)
    assert isinstance(node.methods, list)
    
    # Verify line_number is an integer
    assert isinstance(node.line_number, int)


@given(
    name=variable_names,
    var_type=st.one_of(st.none(), function_names),
    defined_in=function_names,
    line_number=line_numbers
)
def test_variable_node_api_correctness(name, var_type, defined_in, line_number):
    """
    **Feature: enhanced-graph-manager, Property 15: API返回值正确性**
    **Validates: Requirements 5.1**
    
    For any valid variable node data, creating a VariableNode should return
    an object with the correct type and all specified attributes accessible.
    """
    node = VariableNode(
        name=name,
        var_type=var_type,
        defined_in=defined_in,
        line_number=line_number
    )
    
    # Verify the node is of correct type
    assert isinstance(node, VariableNode)
    
    # Verify all attributes are correctly set and accessible
    assert node.name == name
    assert node.var_type == var_type
    assert node.defined_in == defined_in
    assert node.line_number == line_number
    
    # Verify line_number is an integer
    assert isinstance(node.line_number, int)


@given(
    id=function_names,
    text=st.text(min_size=1, max_size=500),
    priority=priorities,
    testable=st.booleans()
)
def test_requirement_node_api_correctness(id, text, priority, testable):
    """
    **Feature: enhanced-graph-manager, Property 15: API返回值正确性**
    **Validates: Requirements 5.1**
    
    For any valid requirement node data, creating a RequirementNode should return
    an object with the correct type and all specified attributes accessible.
    """
    node = RequirementNode(
        id=id,
        text=text,
        priority=priority,
        testable=testable
    )
    
    # Verify the node is of correct type
    assert isinstance(node, RequirementNode)
    
    # Verify all attributes are correctly set and accessible
    assert node.id == id
    assert node.text == text
    assert node.priority == priority
    assert node.testable == testable
    
    # Verify priority is an integer
    assert isinstance(node.priority, int)
    
    # Verify testable is a boolean
    assert isinstance(node.testable, bool)


def test_enhanced_graph_manager_api_correctness():
    """
    **Feature: enhanced-graph-manager, Property 15: API返回值正确性**
    **Validates: Requirements 5.1**
    
    For any EnhancedGraphManager instance, the API methods should return
    objects of the correct types as specified in the interface.
    """
    manager = EnhancedGraphManager()
    
    # Test get_graph returns NetworkX DiGraph
    graph = manager.get_graph()
    assert isinstance(graph, nx.DiGraph)
    
    # Test extract_structure returns NetworkX DiGraph
    result = manager.extract_structure("def test(): pass")
    assert isinstance(result, nx.DiGraph)
    
    # Test inject_semantics returns NetworkX DiGraph
    result = manager.inject_semantics("test requirement")
    assert isinstance(result, nx.DiGraph)
    
    # Test trace_dependencies returns dict
    result = manager.trace_dependencies()
    assert isinstance(result, dict)
    
    # Test flag_violations returns list
    result = manager.flag_violations()
    assert isinstance(result, list)
    
    # Test serialize_graph returns dict
    result = manager.serialize_graph()
    assert isinstance(result, dict)