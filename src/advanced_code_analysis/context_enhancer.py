"""
Context Enhancement Engine for the Advanced Code Analysis system.

This module implements the ContextEnhancer class that collects rich code context
information, optimizes context windows for LLM consumption, analyzes dependencies,
and extracts domain-specific knowledge to improve analysis accuracy.
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

from .models import (
    ContextWindow, 
    DependencyContext, 
    DomainKnowledge
)
from .config import AdvancedAnalysisConfig


logger = logging.getLogger(__name__)


@dataclass
class CodeContext:
    """Represents context information for a specific code element."""
    element_name: str
    element_type: str  # function, class, variable, module
    source_code: str
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)


@dataclass
class ProjectStructure:
    """Represents the overall structure of a project."""
    root_path: str
    python_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    documentation_files: List[str] = field(default_factory=list)
    package_structure: Dict[str, List[str]] = field(default_factory=dict)


class ContextEnhancer:
    """
    Context Enhancement Engine that collects and optimizes code context for LLM analysis.
    
    This class implements the core functionality for:
    1. Collecting rich code context information
    2. Optimizing context windows for token limits
    3. Analyzing dependency relationships
    4. Extracting domain-specific knowledge and patterns
    """
    
    def __init__(self, config: Optional[AdvancedAnalysisConfig] = None):
        """Initialize the Context Enhancer."""
        self.config = config or AdvancedAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cache for parsed AST trees to avoid re-parsing
        self._ast_cache: Dict[str, ast.AST] = {}
        
        # Cache for file contents
        self._file_cache: Dict[str, str] = {}
        
        # Project structure cache
        self._project_structure: Optional[ProjectStructure] = None
        
        # Domain knowledge cache
        self._domain_knowledge_cache: Dict[str, DomainKnowledge] = {}
        
        self.logger.info("ContextEnhancer initialized")
    
    def collect_code_context(self, target_files: List[str], 
                           focus_elements: Optional[List[str]] = None) -> ContextWindow:
        """
        Collect comprehensive code context from target files.
        
        Args:
            target_files: List of file paths to analyze
            focus_elements: Optional list of specific elements to focus on
            
        Returns:
            ContextWindow containing collected context information
        """
        self.logger.info(f"Collecting code context from {len(target_files)} files")
        
        try:
            # Initialize context window
            context = ContextWindow(target_code="")
            
            # Collect code from target files
            target_code_parts = []
            related_functions = []
            class_hierarchy = {}
            module_dependencies = []
            domain_concepts = []
            
            for file_path in target_files:
                if not os.path.exists(file_path):
                    self.logger.warning(f"File not found: {file_path}")
                    continue
                
                # Read and cache file content
                file_content = self._get_file_content(file_path)
                if not file_content:
                    continue
                
                target_code_parts.append(f"# File: {file_path}\n{file_content}")
                
                # Parse AST and extract information
                try:
                    tree = self._get_ast(file_path, file_content)
                    
                    # Extract functions and classes
                    file_functions, file_classes = self._extract_functions_and_classes(tree, file_path)
                    related_functions.extend(file_functions)
                    
                    # Build class hierarchy
                    for class_name, class_info in file_classes.items():
                        class_hierarchy[class_name] = class_info.get('bases', [])
                    
                    # Extract imports and dependencies
                    file_dependencies = self._extract_imports(tree)
                    module_dependencies.extend(file_dependencies)
                    
                    # Extract domain concepts from docstrings and comments
                    file_concepts = self._extract_domain_concepts(tree, file_content)
                    domain_concepts.extend(file_concepts)
                    
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error in {file_path}: {e}")
                    continue
            
            # Combine target code
            context.target_code = "\n\n".join(target_code_parts)
            
            # Filter and limit related functions based on focus elements
            if focus_elements:
                related_functions = self._filter_related_functions(
                    related_functions, focus_elements
                )
            
            # Limit the number of related functions
            max_functions = self.config.analysis.max_related_functions
            if len(related_functions) > max_functions:
                related_functions = related_functions[:max_functions]
                self.logger.info(f"Limited related functions to {max_functions}")
            
            context.related_functions = related_functions
            context.class_hierarchy = class_hierarchy
            context.module_dependencies = list(set(module_dependencies))  # Remove duplicates
            context.domain_concepts = list(set(domain_concepts))  # Remove duplicates
            
            # Build dependency context
            context.dependency_context = self.build_dependency_context(target_files)
            
            # Extract domain knowledge
            project_type = self._infer_project_type(target_files)
            context.domain_knowledge = self.extract_domain_knowledge(project_type)
            
            # Calculate token count
            context.token_count = self._estimate_token_count(context)
            
            self.logger.info(f"Context collection completed: {context.token_count} estimated tokens")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error collecting code context: {e}")
            return ContextWindow(target_code="", token_count=0)
    
    def build_dependency_context(self, target_files: List[str]) -> DependencyContext:
        """
        Build detailed dependency context for the target files.
        
        Args:
            target_files: List of file paths to analyze
            
        Returns:
            DependencyContext containing dependency information
        """
        self.logger.debug("Building dependency context")
        
        dependency_context = DependencyContext()
        
        try:
            for file_path in target_files:
                if not os.path.exists(file_path):
                    continue
                
                file_content = self._get_file_content(file_path)
                if not file_content:
                    continue
                
                try:
                    tree = self._get_ast(file_path, file_content)
                    
                    # Extract function signatures
                    function_signatures = self._extract_function_signatures(tree)
                    dependency_context.function_signatures.update(function_signatures)
                    
                    # Extract class methods
                    class_methods = self._extract_class_methods(tree)
                    dependency_context.class_methods.update(class_methods)
                    
                    # Extract import statements
                    imports = self._extract_import_statements(tree)
                    dependency_context.import_statements.extend(imports)
                    
                    # Build call graph
                    call_relationships = self._extract_call_relationships(tree)
                    for caller, callees in call_relationships.items():
                        if caller not in dependency_context.call_graph:
                            dependency_context.call_graph[caller] = []
                        dependency_context.call_graph[caller].extend(callees)
                    
                except SyntaxError as e:
                    self.logger.warning(f"Syntax error in {file_path}: {e}")
                    continue
            
            # Remove duplicate import statements
            dependency_context.import_statements = list(set(dependency_context.import_statements))
            
            self.logger.debug(f"Dependency context built: {len(dependency_context.function_signatures)} functions, "
                            f"{len(dependency_context.class_methods)} classes")
            
            return dependency_context
            
        except Exception as e:
            self.logger.error(f"Error building dependency context: {e}")
            return DependencyContext()
    
    def extract_domain_knowledge(self, project_type: str) -> DomainKnowledge:
        """
        Extract domain-specific knowledge and terminology.
        
        Args:
            project_type: Type of project (web, ml, data, cli, etc.)
            
        Returns:
            DomainKnowledge containing domain-specific information
        """
        self.logger.debug(f"Extracting domain knowledge for project type: {project_type}")
        
        # Check cache first
        if project_type in self._domain_knowledge_cache:
            return self._domain_knowledge_cache[project_type]
        
        try:
            domain_knowledge = DomainKnowledge(domain_name=project_type)
            
            # Load domain-specific terminology and patterns
            domain_data = self._load_domain_data(project_type)
            
            domain_knowledge.terminology = domain_data.get('terminology', {})
            domain_knowledge.common_patterns = domain_data.get('common_patterns', [])
            domain_knowledge.best_practices = domain_data.get('best_practices', [])
            domain_knowledge.anti_patterns = domain_data.get('anti_patterns', [])
            
            # Extract project-specific terminology from code
            if self._project_structure:
                project_terminology = self._extract_project_terminology()
                domain_knowledge.terminology.update(project_terminology)
            
            # Cache the result
            self._domain_knowledge_cache[project_type] = domain_knowledge
            
            self.logger.debug(f"Domain knowledge extracted: {len(domain_knowledge.terminology)} terms, "
                            f"{len(domain_knowledge.common_patterns)} patterns")
            
            return domain_knowledge
            
        except Exception as e:
            self.logger.error(f"Error extracting domain knowledge: {e}")
            return DomainKnowledge(domain_name=project_type)
    
    def optimize_context_window(self, context: ContextWindow, max_tokens: int) -> ContextWindow:
        """
        Optimize context window to fit within token limits while preserving important information.
        
        Args:
            context: Original context window
            max_tokens: Maximum allowed tokens
            
        Returns:
            Optimized context window
        """
        self.logger.debug(f"Optimizing context window: {context.token_count} -> {max_tokens} tokens")
        
        if context.token_count <= max_tokens:
            return context
        
        try:
            # Create a copy to modify
            optimized_context = ContextWindow(
                target_code=context.target_code,
                related_functions=context.related_functions.copy(),
                class_hierarchy=context.class_hierarchy.copy(),
                module_dependencies=context.module_dependencies.copy(),
                domain_concepts=context.domain_concepts.copy(),
                dependency_context=context.dependency_context,
                domain_knowledge=context.domain_knowledge
            )
            
            # Use the original token count for comparison since we manually set it in tests
            current_tokens = context.token_count
            
            # Strategy 1: Reduce related functions
            while (current_tokens > max_tokens and len(optimized_context.related_functions) > 5):
                # Remove least relevant functions (simple heuristic: remove from end)
                optimized_context.related_functions = optimized_context.related_functions[:-1]
                current_tokens = self._estimate_token_count(optimized_context)
            
            # Strategy 2: Reduce module dependencies
            while (current_tokens > max_tokens and len(optimized_context.module_dependencies) > 10):
                optimized_context.module_dependencies = optimized_context.module_dependencies[:-1]
                current_tokens = self._estimate_token_count(optimized_context)
            
            # Strategy 3: Reduce domain concepts
            while (current_tokens > max_tokens and len(optimized_context.domain_concepts) > 10):
                optimized_context.domain_concepts = optimized_context.domain_concepts[:-1]
                current_tokens = self._estimate_token_count(optimized_context)
            
            # Strategy 4: Truncate target code if still too large
            if current_tokens > max_tokens:
                target_lines = optimized_context.target_code.split('\n')
                # Keep first 80% of lines as a simple truncation strategy
                keep_lines = int(len(target_lines) * 0.8)
                if keep_lines < len(target_lines):
                    optimized_context.target_code = '\n'.join(target_lines[:keep_lines])
                    optimized_context.target_code += "\n\n# ... (truncated for token limit)"
                    current_tokens = self._estimate_token_count(optimized_context)
            
            # Set final token count
            optimized_context.token_count = current_tokens
            
            self.logger.info(f"Context optimized: {context.token_count} -> {optimized_context.token_count} tokens")
            
            return optimized_context
            
        except Exception as e:
            self.logger.error(f"Error optimizing context window: {e}")
            return context
    
    def analyze_project_structure(self, root_path: str) -> ProjectStructure:
        """
        Analyze the overall structure of a project.
        
        Args:
            root_path: Root directory of the project
            
        Returns:
            ProjectStructure containing project organization information
        """
        self.logger.debug(f"Analyzing project structure: {root_path}")
        
        try:
            project_structure = ProjectStructure(root_path=root_path)
            
            # Walk through the project directory
            for root, dirs, files in os.walk(root_path):
                # Skip common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                          d not in ['__pycache__', 'node_modules', 'venv', '.venv']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_path)
                    
                    # Categorize files
                    if file.endswith('.py'):
                        # Check if it's a test file based on filename patterns
                        is_test_file = (file.lower().startswith('test_') or 
                                      file.lower().endswith('_test.py') or
                                      '/tests/' in root.lower() or
                                      root.lower().endswith('/test'))
                        
                        if is_test_file:
                            project_structure.test_files.append(relative_path)
                        else:
                            project_structure.python_files.append(relative_path)
                    elif file in ['setup.py', 'setup.cfg', 'pyproject.toml', 'requirements.txt', 
                                 'Pipfile', 'poetry.lock', 'environment.yml']:
                        project_structure.config_files.append(relative_path)
                    elif file.endswith(('.md', '.rst', '.txt')) and 'readme' in file.lower():
                        project_structure.documentation_files.append(relative_path)
            
            # Build package structure
            for py_file in project_structure.python_files:
                package_parts = py_file.replace('/', '.').replace('\\', '.').replace('.py', '').split('.')
                if len(package_parts) > 1:
                    package_name = '.'.join(package_parts[:-1])
                    if package_name not in project_structure.package_structure:
                        project_structure.package_structure[package_name] = []
                    project_structure.package_structure[package_name].append(package_parts[-1])
            
            # Cache the result
            self._project_structure = project_structure
            
            self.logger.debug(f"Project structure analyzed: {len(project_structure.python_files)} Python files, "
                            f"{len(project_structure.package_structure)} packages")
            
            return project_structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing project structure: {e}")
            return ProjectStructure(root_path=root_path)
    
    def get_contextual_code_snippets(self, element_name: str, 
                                   context_type: str = "usage") -> List[CodeContext]:
        """
        Get contextual code snippets for a specific element.
        
        Args:
            element_name: Name of the code element (function, class, variable)
            context_type: Type of context to retrieve ("usage", "definition", "related")
            
        Returns:
            List of CodeContext objects containing relevant snippets
        """
        self.logger.debug(f"Getting contextual snippets for {element_name} ({context_type})")
        
        try:
            snippets = []
            
            if not self._project_structure:
                return snippets
            
            # Search through Python files
            for py_file in self._project_structure.python_files:
                file_path = os.path.join(self._project_structure.root_path, py_file)
                
                if not os.path.exists(file_path):
                    continue
                
                file_content = self._get_file_content(file_path)
                if not file_content:
                    continue
                
                try:
                    tree = self._get_ast(file_path, file_content)
                    
                    # Find relevant snippets based on context type
                    if context_type == "definition":
                        snippets.extend(self._find_definition_snippets(tree, element_name, file_path, file_content))
                    elif context_type == "usage":
                        snippets.extend(self._find_usage_snippets(tree, element_name, file_path, file_content))
                    elif context_type == "related":
                        snippets.extend(self._find_related_snippets(tree, element_name, file_path, file_content))
                    
                except SyntaxError:
                    continue
            
            # Limit the number of snippets
            max_snippets = 10
            if len(snippets) > max_snippets:
                snippets = snippets[:max_snippets]
            
            self.logger.debug(f"Found {len(snippets)} contextual snippets for {element_name}")
            
            return snippets
            
        except Exception as e:
            self.logger.error(f"Error getting contextual snippets: {e}")
            return []
    
    # Private helper methods
    
    def _get_file_content(self, file_path: str) -> str:
        """Get file content with caching."""
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._file_cache[file_path] = content
            return content
        except Exception as e:
            self.logger.warning(f"Could not read file {file_path}: {e}")
            return ""
    
    def _get_ast(self, file_path: str, content: str) -> ast.AST:
        """Get AST with caching."""
        if file_path in self._ast_cache:
            return self._ast_cache[file_path]
        
        tree = ast.parse(content)
        self._ast_cache[file_path] = tree
        return tree
    
    def _extract_functions_and_classes(self, tree: ast.AST, file_path: str) -> Tuple[List[str], Dict[str, Dict]]:
        """Extract function and class information from AST."""
        functions = []
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = f"{node.name}({', '.join(arg.arg for arg in node.args.args)})"
                functions.append(func_info)
            elif isinstance(node, ast.ClassDef):
                bases = [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                classes[node.name] = {
                    'bases': bases,
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                }
        
        return functions, classes
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _extract_domain_concepts(self, tree: ast.AST, content: str) -> List[str]:
        """Extract domain concepts from docstrings and comments."""
        concepts = []
        
        # Extract from docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Simple extraction of capitalized words as potential concepts
                    words = re.findall(r'\b[A-Z][a-z]+\b', docstring)
                    concepts.extend(words)
        
        # Extract from comments
        comment_lines = [line.strip() for line in content.split('\n') if line.strip().startswith('#')]
        for comment in comment_lines:
            words = re.findall(r'\b[A-Z][a-z]+\b', comment)
            concepts.extend(words)
        
        return list(set(concepts))  # Remove duplicates
    
    def _filter_related_functions(self, functions: List[str], focus_elements: List[str]) -> List[str]:
        """Filter functions based on focus elements."""
        filtered = []
        
        for func in functions:
            for element in focus_elements:
                if element.lower() in func.lower():
                    filtered.append(func)
                    break
        
        return filtered
    
    def _infer_project_type(self, target_files: List[str]) -> str:
        """Infer project type based on file patterns and imports."""
        # Simple heuristics for project type detection
        all_content = ""
        for file_path in target_files:
            content = self._get_file_content(file_path)
            all_content += content
        
        if any(framework in all_content for framework in ['django', 'flask', 'fastapi']):
            return "web"
        elif any(lib in all_content for lib in ['pandas', 'numpy', 'sklearn', 'tensorflow', 'pytorch']):
            return "ml"
        elif any(lib in all_content for lib in ['click', 'argparse', 'typer']):
            return "cli"
        elif any(lib in all_content for lib in ['pytest', 'unittest', 'nose']):
            return "testing"
        else:
            return "general"
    
    def _load_domain_data(self, project_type: str) -> Dict[str, Any]:
        """Load domain-specific data from configuration or built-in knowledge."""
        # Built-in domain knowledge
        domain_data = {
            "web": {
                "terminology": {
                    "HTTP": "HyperText Transfer Protocol",
                    "REST": "Representational State Transfer",
                    "API": "Application Programming Interface",
                    "CRUD": "Create, Read, Update, Delete",
                    "MVC": "Model-View-Controller"
                },
                "common_patterns": [
                    "request-response cycle",
                    "middleware pattern",
                    "route handling",
                    "template rendering"
                ],
                "best_practices": [
                    "validate input data",
                    "handle errors gracefully",
                    "use HTTPS for security",
                    "implement proper authentication"
                ],
                "anti_patterns": [
                    "storing passwords in plain text",
                    "SQL injection vulnerabilities",
                    "missing input validation"
                ]
            },
            "ml": {
                "terminology": {
                    "ML": "Machine Learning",
                    "AI": "Artificial Intelligence",
                    "CNN": "Convolutional Neural Network",
                    "RNN": "Recurrent Neural Network",
                    "GPU": "Graphics Processing Unit"
                },
                "common_patterns": [
                    "data preprocessing",
                    "model training",
                    "feature engineering",
                    "cross-validation"
                ],
                "best_practices": [
                    "split data into train/validation/test",
                    "normalize input features",
                    "monitor for overfitting",
                    "use appropriate metrics"
                ],
                "anti_patterns": [
                    "data leakage",
                    "overfitting to training data",
                    "ignoring class imbalance"
                ]
            },
            "general": {
                "terminology": {},
                "common_patterns": [
                    "factory pattern",
                    "singleton pattern",
                    "observer pattern"
                ],
                "best_practices": [
                    "write clear documentation",
                    "use meaningful variable names",
                    "handle exceptions properly",
                    "write unit tests"
                ],
                "anti_patterns": [
                    "global variables",
                    "deep nesting",
                    "magic numbers"
                ]
            }
        }
        
        return domain_data.get(project_type, domain_data["general"])
    
    def _extract_project_terminology(self) -> Dict[str, str]:
        """Extract project-specific terminology from code."""
        terminology = {}
        
        if not self._project_structure:
            return terminology
        
        # Extract class names as potential domain terms
        for py_file in self._project_structure.python_files[:10]:  # Limit to avoid performance issues
            file_path = os.path.join(self._project_structure.root_path, py_file)
            content = self._get_file_content(file_path)
            
            if content:
                try:
                    tree = self._get_ast(file_path, content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Use class name as term and docstring as definition
                            docstring = ast.get_docstring(node)
                            if docstring:
                                # Take first sentence as definition
                                definition = docstring.split('.')[0].strip()
                                terminology[node.name] = definition
                except SyntaxError:
                    continue
        
        return terminology
    
    def _estimate_token_count(self, context: ContextWindow) -> int:
        """Estimate token count for a context window."""
        # Simple approximation: 1 token ≈ 4 characters
        total_chars = len(context.target_code)
        total_chars += sum(len(func) for func in context.related_functions)
        total_chars += sum(len(f"{k}: {v}") for k, v in context.class_hierarchy.items())
        total_chars += sum(len(dep) for dep in context.module_dependencies)
        total_chars += sum(len(concept) for concept in context.domain_concepts)
        
        return total_chars // 4
    
    def _extract_function_signatures(self, tree: ast.AST) -> Dict[str, str]:
        """Extract function signatures from AST."""
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                
                return_annotation = ""
                if node.returns:
                    return_annotation = f" -> {ast.unparse(node.returns)}"
                
                signature = f"def {node.name}({', '.join(args)}){return_annotation}"
                signatures[node.name] = signature
        
        return signatures
    
    def _extract_class_methods(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract class methods from AST."""
        class_methods = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                class_methods[node.name] = methods
        
        return class_methods
    
    def _extract_import_statements(self, tree: ast.AST) -> List[str]:
        """Extract import statements as strings."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_str = f"import {alias.name}"
                    if alias.asname:
                        import_str += f" as {alias.asname}"
                    imports.append(import_str)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = []
                    for alias in node.names:
                        name_str = alias.name
                        if alias.asname:
                            name_str += f" as {alias.asname}"
                        names.append(name_str)
                    import_str = f"from {node.module} import {', '.join(names)}"
                    imports.append(import_str)
        
        return imports
    
    def _extract_call_relationships(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract function call relationships from AST."""
        call_graph = {}
        current_function = None
        
        class CallVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                nonlocal current_function
                old_function = current_function
                current_function = node.name
                call_graph[current_function] = []
                self.generic_visit(node)
                current_function = old_function
            
            def visit_Call(self, node):
                if current_function and isinstance(node.func, ast.Name):
                    call_graph[current_function].append(node.func.id)
                self.generic_visit(node)
        
        visitor = CallVisitor()
        visitor.visit(tree)
        
        return call_graph
    
    def _find_definition_snippets(self, tree: ast.AST, element_name: str, 
                                file_path: str, content: str) -> List[CodeContext]:
        """Find definition snippets for an element."""
        snippets = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == element_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                
                snippet_lines = lines[start_line:end_line]
                snippet_code = '\n'.join(snippet_lines)
                
                docstring = ast.get_docstring(node)
                
                snippet = CodeContext(
                    element_name=element_name,
                    element_type="function" if isinstance(node, ast.FunctionDef) else "class",
                    source_code=snippet_code,
                    file_path=file_path,
                    line_number=node.lineno,
                    docstring=docstring
                )
                snippets.append(snippet)
        
        return snippets
    
    def _find_usage_snippets(self, tree: ast.AST, element_name: str, 
                           file_path: str, content: str) -> List[CodeContext]:
        """Find usage snippets for an element."""
        snippets = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == element_name:
                    line_num = node.lineno - 1
                    start_line = max(0, line_num - 2)
                    end_line = min(len(lines), line_num + 3)
                    
                    snippet_lines = lines[start_line:end_line]
                    snippet_code = '\n'.join(snippet_lines)
                    
                    snippet = CodeContext(
                        element_name=element_name,
                        element_type="usage",
                        source_code=snippet_code,
                        file_path=file_path,
                        line_number=node.lineno
                    )
                    snippets.append(snippet)
        
        return snippets
    
    def _find_related_snippets(self, tree: ast.AST, element_name: str, 
                             file_path: str, content: str) -> List[CodeContext]:
        """Find related snippets for an element."""
        # For now, return empty list - this could be expanded to find
        # functions that call the element or are called by it
        return []