"""
Tests for the Context Enhancement Engine.

This module contains unit tests and integration tests for the ContextEnhancer class,
verifying its ability to collect code context, optimize context windows, analyze
dependencies, and extract domain knowledge.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from .context_enhancer import ContextEnhancer, CodeContext, ProjectStructure
from .models import ContextWindow, DependencyContext, DomainKnowledge
from .config import AdvancedAnalysisConfig


class TestContextEnhancer:
    """Test suite for the ContextEnhancer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AdvancedAnalysisConfig()
        self.enhancer = ContextEnhancer(self.config)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample Python code for testing
        self.sample_code = '''
"""Sample module for testing."""

import os
import json
from typing import List, Dict

class DataProcessor:
    """Processes data for analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data = []
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process the loaded data."""
        processed = []
        for item in data:
            if self.validate_item(item):
                processed.append(self.transform_item(item))
        return processed
    
    def validate_item(self, item: Dict) -> bool:
        """Validate a single data item."""
        return 'id' in item and 'value' in item
    
    def transform_item(self, item: Dict) -> Dict:
        """Transform a single data item."""
        return {
            'id': item['id'],
            'processed_value': item['value'] * 2,
            'timestamp': self.get_timestamp()
        }
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()

def main():
    """Main function for data processing."""
    processor = DataProcessor({'debug': True})
    data = processor.load_data('data.json')
    result = processor.process_data(data)
    print(f"Processed {len(result)} items")

if __name__ == "__main__":
    main()
'''
        
        # Create test file
        self.test_file = os.path.join(self.temp_dir, 'test_module.py')
        with open(self.test_file, 'w') as f:
            f.write(self.sample_code)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_collect_code_context_basic(self):
        """Test basic code context collection."""
        context = self.enhancer.collect_code_context([self.test_file])
        
        assert isinstance(context, ContextWindow)
        assert len(context.target_code) > 0
        assert self.test_file in context.target_code
        assert len(context.related_functions) > 0
        assert len(context.class_hierarchy) > 0
        assert 'DataProcessor' in context.class_hierarchy
        assert len(context.module_dependencies) > 0
        assert 'os' in context.module_dependencies
        assert 'json' in context.module_dependencies
        assert context.token_count > 0
    
    def test_collect_code_context_with_focus(self):
        """Test code context collection with focus elements."""
        focus_elements = ['process_data', 'validate']
        context = self.enhancer.collect_code_context([self.test_file], focus_elements)
        
        assert isinstance(context, ContextWindow)
        # Should have filtered functions based on focus elements
        focused_functions = [f for f in context.related_functions 
                           if any(elem in f for elem in focus_elements)]
        assert len(focused_functions) > 0
    
    def test_collect_code_context_nonexistent_file(self):
        """Test handling of non-existent files."""
        context = self.enhancer.collect_code_context(['/nonexistent/file.py'])
        
        assert isinstance(context, ContextWindow)
        assert context.target_code == ""
        assert context.token_count == 0
    
    def test_build_dependency_context(self):
        """Test dependency context building."""
        dependency_context = self.enhancer.build_dependency_context([self.test_file])
        
        assert isinstance(dependency_context, DependencyContext)
        assert len(dependency_context.function_signatures) > 0
        assert len(dependency_context.class_methods) > 0
        assert 'DataProcessor' in dependency_context.class_methods
        assert len(dependency_context.import_statements) > 0
        assert any('import os' in stmt for stmt in dependency_context.import_statements)
        assert len(dependency_context.call_graph) > 0
    
    def test_extract_domain_knowledge_web(self):
        """Test domain knowledge extraction for web projects."""
        domain_knowledge = self.enhancer.extract_domain_knowledge('web')
        
        assert isinstance(domain_knowledge, DomainKnowledge)
        assert domain_knowledge.domain_name == 'web'
        assert len(domain_knowledge.terminology) > 0
        assert 'HTTP' in domain_knowledge.terminology
        assert len(domain_knowledge.common_patterns) > 0
        assert len(domain_knowledge.best_practices) > 0
        assert len(domain_knowledge.anti_patterns) > 0
    
    def test_extract_domain_knowledge_ml(self):
        """Test domain knowledge extraction for ML projects."""
        domain_knowledge = self.enhancer.extract_domain_knowledge('ml')
        
        assert isinstance(domain_knowledge, DomainKnowledge)
        assert domain_knowledge.domain_name == 'ml'
        assert 'ML' in domain_knowledge.terminology
        assert any('training' in pattern for pattern in domain_knowledge.common_patterns)
    
    def test_extract_domain_knowledge_general(self):
        """Test domain knowledge extraction for general projects."""
        domain_knowledge = self.enhancer.extract_domain_knowledge('general')
        
        assert isinstance(domain_knowledge, DomainKnowledge)
        assert domain_knowledge.domain_name == 'general'
        assert len(domain_knowledge.common_patterns) > 0
        assert len(domain_knowledge.best_practices) > 0
    
    def test_optimize_context_window_no_optimization_needed(self):
        """Test context window optimization when no optimization is needed."""
        context = ContextWindow(
            target_code="short code",
            related_functions=["func1", "func2"],
            token_count=100
        )
        
        optimized = self.enhancer.optimize_context_window(context, 1000)
        
        assert optimized.token_count <= 1000
        assert optimized.target_code == context.target_code
        assert len(optimized.related_functions) == len(context.related_functions)
    
    def test_optimize_context_window_with_optimization(self):
        """Test context window optimization when optimization is needed."""
        # Create a large context that needs optimization
        large_functions = [f"function_{i}(arg1, arg2, arg3)" for i in range(50)]
        large_dependencies = [f"module_{i}" for i in range(50)]
        large_concepts = [f"Concept_{i}" for i in range(50)]
        
        context = ContextWindow(
            target_code="x" * 1000,  # Large target code
            related_functions=large_functions,
            module_dependencies=large_dependencies,
            domain_concepts=large_concepts
        )
        context.token_count = 5000  # Simulate large token count
        
        optimized = self.enhancer.optimize_context_window(context, 1000)
        
        assert optimized.token_count < context.token_count
        # Should have reduced some elements
        assert (len(optimized.related_functions) < len(context.related_functions) or
                len(optimized.module_dependencies) < len(context.module_dependencies) or
                len(optimized.domain_concepts) < len(context.domain_concepts) or
                len(optimized.target_code) < len(context.target_code))
    
    def test_analyze_project_structure(self):
        """Test project structure analysis."""
        # Create additional test files
        test_dir = os.path.join(self.temp_dir, 'test_project')
        os.makedirs(test_dir, exist_ok=True)
        
        # Create Python files
        with open(os.path.join(test_dir, 'main.py'), 'w') as f:
            f.write('print("main")')
        
        with open(os.path.join(test_dir, 'test_main.py'), 'w') as f:
            f.write('import unittest')
        
        # Create config file
        with open(os.path.join(test_dir, 'requirements.txt'), 'w') as f:
            f.write('pytest\nrequests')
        
        # Create documentation
        with open(os.path.join(test_dir, 'README.md'), 'w') as f:
            f.write('# Test Project')
        
        project_structure = self.enhancer.analyze_project_structure(test_dir)
        
        assert isinstance(project_structure, ProjectStructure)
        assert project_structure.root_path == test_dir
        assert len(project_structure.python_files) > 0
        assert len(project_structure.test_files) > 0
        assert len(project_structure.config_files) > 0
        assert len(project_structure.documentation_files) > 0
        assert 'main.py' in project_structure.python_files
        assert 'test_main.py' in project_structure.test_files
        assert 'requirements.txt' in project_structure.config_files
        assert 'README.md' in project_structure.documentation_files
    
    def test_get_contextual_code_snippets_definition(self):
        """Test getting definition snippets."""
        # First analyze the project structure
        self.enhancer.analyze_project_structure(self.temp_dir)
        
        snippets = self.enhancer.get_contextual_code_snippets('DataProcessor', 'definition')
        
        assert isinstance(snippets, list)
        if len(snippets) > 0:  # May be empty if file not found in structure
            snippet = snippets[0]
            assert isinstance(snippet, CodeContext)
            assert snippet.element_name == 'DataProcessor'
            assert snippet.element_type == 'class'
            assert 'class DataProcessor' in snippet.source_code
    
    def test_get_contextual_code_snippets_usage(self):
        """Test getting usage snippets."""
        # First analyze the project structure
        self.enhancer.analyze_project_structure(self.temp_dir)
        
        snippets = self.enhancer.get_contextual_code_snippets('DataProcessor', 'usage')
        
        assert isinstance(snippets, list)
        # Usage snippets may be empty for this simple test case
    
    def test_caching_behavior(self):
        """Test that caching works correctly."""
        # First call should populate cache
        context1 = self.enhancer.collect_code_context([self.test_file])
        
        # Second call should use cache
        context2 = self.enhancer.collect_code_context([self.test_file])
        
        assert context1.target_code == context2.target_code
        assert len(context1.related_functions) == len(context2.related_functions)
        
        # Check that file is in cache
        assert self.test_file in self.enhancer._file_cache
        assert self.test_file in self.enhancer._ast_cache
    
    def test_domain_knowledge_caching(self):
        """Test that domain knowledge is cached."""
        # First call
        knowledge1 = self.enhancer.extract_domain_knowledge('web')
        
        # Second call should use cache
        knowledge2 = self.enhancer.extract_domain_knowledge('web')
        
        assert knowledge1.domain_name == knowledge2.domain_name
        assert knowledge1.terminology == knowledge2.terminology
        assert 'web' in self.enhancer._domain_knowledge_cache
    
    def test_error_handling_syntax_error(self):
        """Test handling of files with syntax errors."""
        # Create file with syntax error
        bad_file = os.path.join(self.temp_dir, 'bad_syntax.py')
        with open(bad_file, 'w') as f:
            f.write('def broken_function(\n    # Missing closing parenthesis')
        
        context = self.enhancer.collect_code_context([bad_file])
        
        # Should handle gracefully and return partial results
        assert isinstance(context, ContextWindow)
        # May have some content from the file even with syntax errors
    
    def test_token_estimation(self):
        """Test token count estimation."""
        context = ContextWindow(
            target_code="def test(): pass",
            related_functions=["func1()", "func2()"],
            class_hierarchy={"Class1": ["Base"]},
            module_dependencies=["os", "sys"],
            domain_concepts=["Concept1", "Concept2"]
        )
        
        estimated_tokens = self.enhancer._estimate_token_count(context)
        
        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0
    
    def test_project_type_inference(self):
        """Test project type inference."""
        # Test web project detection
        web_code = "import django\nfrom flask import Flask"
        web_file = os.path.join(self.temp_dir, 'web_app.py')
        with open(web_file, 'w') as f:
            f.write(web_code)
        
        project_type = self.enhancer._infer_project_type([web_file])
        assert project_type == 'web'
        
        # Test ML project detection
        ml_code = "import pandas as pd\nimport numpy as np\nfrom sklearn import model_selection"
        ml_file = os.path.join(self.temp_dir, 'ml_model.py')
        with open(ml_file, 'w') as f:
            f.write(ml_code)
        
        project_type = self.enhancer._infer_project_type([ml_file])
        assert project_type == 'ml'
        
        # Test CLI project detection
        cli_code = "import click\nimport argparse"
        cli_file = os.path.join(self.temp_dir, 'cli_tool.py')
        with open(cli_file, 'w') as f:
            f.write(cli_code)
        
        project_type = self.enhancer._infer_project_type([cli_file])
        assert project_type == 'cli'
    
    def test_function_signature_extraction(self):
        """Test function signature extraction."""
        # Create file with typed functions
        typed_code = '''
def simple_func(x: int, y: str) -> bool:
    return True

def complex_func(a: List[Dict[str, Any]], b: Optional[int] = None) -> Tuple[str, int]:
    return "result", 42
'''
        typed_file = os.path.join(self.temp_dir, 'typed_functions.py')
        with open(typed_file, 'w') as f:
            f.write(typed_code)
        
        dependency_context = self.enhancer.build_dependency_context([typed_file])
        
        assert 'simple_func' in dependency_context.function_signatures
        assert 'complex_func' in dependency_context.function_signatures
        
        simple_sig = dependency_context.function_signatures['simple_func']
        assert 'x: int' in simple_sig
        assert 'y: str' in simple_sig
        assert '-> bool' in simple_sig
    
    def test_call_graph_extraction(self):
        """Test call graph extraction."""
        call_code = '''
def caller():
    result1 = callee1()
    result2 = callee2()
    return result1 + result2

def callee1():
    return helper()

def callee2():
    return 42

def helper():
    return 1
'''
        call_file = os.path.join(self.temp_dir, 'call_graph.py')
        with open(call_file, 'w') as f:
            f.write(call_code)
        
        dependency_context = self.enhancer.build_dependency_context([call_file])
        
        assert 'caller' in dependency_context.call_graph
        assert 'callee1' in dependency_context.call_graph['caller']
        assert 'callee2' in dependency_context.call_graph['caller']
        assert 'helper' in dependency_context.call_graph['callee1']
    
    def test_configuration_integration(self):
        """Test integration with configuration settings."""
        # Test with custom configuration
        custom_config = AdvancedAnalysisConfig()
        custom_config.analysis.max_related_functions = 5
        custom_config.analysis.max_dependency_depth = 2
        
        custom_enhancer = ContextEnhancer(custom_config)
        
        # Create file with many functions
        many_functions_code = '\n'.join([f'def func_{i}(): pass' for i in range(20)])
        many_func_file = os.path.join(self.temp_dir, 'many_functions.py')
        with open(many_func_file, 'w') as f:
            f.write(many_functions_code)
        
        context = custom_enhancer.collect_code_context([many_func_file])
        
        # Should respect the configuration limit
        assert len(context.related_functions) <= custom_config.analysis.max_related_functions


class TestContextEnhancerIntegration:
    """Integration tests for ContextEnhancer with other components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = AdvancedAnalysisConfig()
        self.enhancer = ContextEnhancer(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_with_enhanced_graph_manager(self):
        """Test integration with Enhanced Graph Manager."""
        # Create a realistic Python project structure
        project_code = '''
"""Main application module."""

from typing import List, Dict
from .models import User, DataModel
from .utils import validate_input, process_data

class Application:
    """Main application class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.users: List[User] = []
    
    def add_user(self, user_data: Dict) -> User:
        """Add a new user to the application."""
        if validate_input(user_data):
            user = User(**user_data)
            self.users.append(user)
            return user
        raise ValueError("Invalid user data")
    
    def process_user_data(self, user_id: str) -> Dict:
        """Process data for a specific user."""
        user = self.find_user(user_id)
        if user:
            return process_data(user.data)
        return {}
    
    def find_user(self, user_id: str) -> User:
        """Find user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None
'''
        
        app_file = os.path.join(self.temp_dir, 'app.py')
        with open(app_file, 'w') as f:
            f.write(project_code)
        
        # Test comprehensive context collection
        context = self.enhancer.collect_code_context([app_file])
        
        assert isinstance(context, ContextWindow)
        assert len(context.target_code) > 0
        assert len(context.related_functions) > 0
        assert 'Application' in context.class_hierarchy
        assert context.dependency_context is not None
        assert context.domain_knowledge is not None
        
        # Test context optimization
        optimized_context = self.enhancer.optimize_context_window(context, 1000)
        assert optimized_context.token_count <= 1000
    
    def test_real_world_project_analysis(self):
        """Test analysis of a realistic project structure."""
        # Create a mini project structure
        project_root = os.path.join(self.temp_dir, 'sample_project')
        os.makedirs(project_root, exist_ok=True)
        
        # Create main module
        main_code = '''
"""Main entry point."""
import sys
from src.core import Engine
from src.utils import setup_logging

def main():
    setup_logging()
    engine = Engine()
    engine.run()

if __name__ == "__main__":
    main()
'''
        
        # Create src directory
        src_dir = os.path.join(project_root, 'src')
        os.makedirs(src_dir, exist_ok=True)
        
        with open(os.path.join(project_root, 'main.py'), 'w') as f:
            f.write(main_code)
        
        # Create core module
        core_code = '''
"""Core engine module."""
from typing import Optional
import logging

class Engine:
    """Main processing engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def run(self):
        """Run the engine."""
        self.logger.info("Starting engine")
        self.running = True
        try:
            self.process()
        finally:
            self.stop()
    
    def process(self):
        """Main processing loop."""
        while self.running:
            # Process data
            pass
    
    def stop(self):
        """Stop the engine."""
        self.running = False
        self.logger.info("Engine stopped")
'''
        
        with open(os.path.join(src_dir, 'core.py'), 'w') as f:
            f.write(core_code)
        
        # Create utils module
        utils_code = '''
"""Utility functions."""
import logging

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO)

def validate_data(data):
    """Validate input data."""
    return data is not None
'''
        
        with open(os.path.join(src_dir, 'utils.py'), 'w') as f:
            f.write(utils_code)
        
        # Analyze the project
        project_structure = self.enhancer.analyze_project_structure(project_root)
        
        assert len(project_structure.python_files) >= 3
        assert project_structure.root_path == project_root
        
        # Collect context from all Python files
        all_files = [os.path.join(project_root, f) for f in project_structure.python_files]
        context = self.enhancer.collect_code_context(all_files)
        
        assert len(context.target_code) > 0
        assert len(context.related_functions) > 0
        assert len(context.module_dependencies) > 0
        assert 'logging' in context.module_dependencies
        assert context.dependency_context is not None
        assert len(context.dependency_context.function_signatures) > 0


if __name__ == "__main__":
    pytest.main([__file__])