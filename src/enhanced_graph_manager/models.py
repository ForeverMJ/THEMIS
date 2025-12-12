"""Core data models for the Enhanced GraphManager system."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FunctionNode:
    """Represents a function in the code graph."""
    name: str
    args: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    line_number: int


@dataclass
class ClassNode:
    """Represents a class in the code graph."""
    name: str
    bases: List[str]
    methods: List[str]
    docstring: Optional[str]
    line_number: int


@dataclass
class VariableNode:
    """Represents a variable (particularly self.xxx member variables) in the code graph."""
    name: str
    var_type: Optional[str]
    defined_in: str
    line_number: int


@dataclass
class RequirementNode:
    """Represents a requirement extracted from issue text."""
    id: str
    text: str
    priority: int
    testable: bool


@dataclass
class CallEdge:
    """Represents a function call relationship."""
    caller: str
    callee: str
    line_number: int


@dataclass
class DependencyEdge:
    """Represents various dependency relationships in code."""
    source: str
    target: str
    dependency_type: str  # DEPENDS_ON, USES_VAR, DEFINED_IN
    context: str


@dataclass
class ViolationEdge:
    """Represents requirement satisfaction or violation relationships."""
    requirement: str
    code_node: str
    status: str  # SATISFIES, VIOLATES
    reason: str
    confidence: float