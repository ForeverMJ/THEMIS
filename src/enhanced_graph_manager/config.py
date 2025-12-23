"""Configuration settings for Enhanced GraphManager."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnhancedGraphManagerConfig:
    """Configuration class for Enhanced GraphManager settings."""
    
    # Structural extraction settings
    max_ast_depth: int = 100
    extract_docstrings: bool = True
    extract_type_hints: bool = True
    
    # Semantic injection settings
    llm_model: str = "gpt-5-mini"
    max_requirement_length: int = 1500
    requirement_priority_threshold: int = 5
    
    # Dependency tracing settings
    max_dependency_depth: int = 50
    trace_external_imports: bool = False
    
    # Violation flagging settings
    violation_confidence_threshold: float = 0.7
    max_violations_per_requirement: int = 10
    
    # Graph settings
    max_nodes: int = 10000
    max_edges: int = 50000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "max_ast_depth": self.max_ast_depth,
            "extract_docstrings": self.extract_docstrings,
            "extract_type_hints": self.extract_type_hints,
            "llm_model": self.llm_model,
            "max_requirement_length": self.max_requirement_length,
            "requirement_priority_threshold": self.requirement_priority_threshold,
            "max_dependency_depth": self.max_dependency_depth,
            "trace_external_imports": self.trace_external_imports,
            "violation_confidence_threshold": self.violation_confidence_threshold,
            "max_violations_per_requirement": self.max_violations_per_requirement,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnhancedGraphManagerConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)