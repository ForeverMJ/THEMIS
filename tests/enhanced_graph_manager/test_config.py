"""Tests for Enhanced GraphManager configuration."""

import pytest
from src.enhanced_graph_manager.config import EnhancedGraphManagerConfig


class TestEnhancedGraphManagerConfig:
    """Test the configuration class."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = EnhancedGraphManagerConfig()
        
        assert config.max_ast_depth == 100
        assert config.extract_docstrings is True
        assert config.extract_type_hints is True
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.max_requirement_length == 1000
        assert config.requirement_priority_threshold == 5
        assert config.max_dependency_depth == 50
        assert config.trace_external_imports is False
        assert config.violation_confidence_threshold == 0.7
        assert config.max_violations_per_requirement == 10
        assert config.max_nodes == 10000
        assert config.max_edges == 50000
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = EnhancedGraphManagerConfig(
            max_ast_depth=200,
            llm_model="gpt-4",
            violation_confidence_threshold=0.8
        )
        
        assert config.max_ast_depth == 200
        assert config.llm_model == "gpt-4"
        assert config.violation_confidence_threshold == 0.8
        # Other values should remain default
        assert config.extract_docstrings is True
        assert config.max_nodes == 10000
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = EnhancedGraphManagerConfig(max_ast_depth=150)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["max_ast_depth"] == 150
        assert config_dict["extract_docstrings"] is True
        assert len(config_dict) == 12  # All config fields
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "max_ast_depth": 300,
            "llm_model": "gpt-4-turbo",
            "violation_confidence_threshold": 0.9,
            "extract_docstrings": False,
            "extract_type_hints": True,
            "max_requirement_length": 2000,
            "requirement_priority_threshold": 3,
            "max_dependency_depth": 75,
            "trace_external_imports": True,
            "max_violations_per_requirement": 15,
            "max_nodes": 20000,
            "max_edges": 100000,
        }
        
        config = EnhancedGraphManagerConfig.from_dict(config_dict)
        
        assert config.max_ast_depth == 300
        assert config.llm_model == "gpt-4-turbo"
        assert config.violation_confidence_threshold == 0.9
        assert config.extract_docstrings is False
        assert config.trace_external_imports is True
        assert config.max_nodes == 20000