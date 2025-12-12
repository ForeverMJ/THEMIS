"""Structural extraction engine for Enhanced GraphManager."""

import ast
from typing import List, Dict, Set, Optional, Tuple
import networkx as nx

from .models import (
    FunctionNode,
    ClassNode,
    VariableNode,
    CallEdge,
    DependencyEdge,
)


class StructuralExtractor:
    """
    Extracts precise structural information from Python code using AST analysis.
    
    This class implements the structural extraction engine that performs
    fact-based code analysis without interpretation, focusing on:
    - Function definitions with parameters and return types
    - Class definitions with inheritance relationships
    - Member variable definitions (self.xxx patterns)
    - Function call relationships
    - Class instantiation relationships
    - Variable definition relationships
    """
    
    def __init__(self):
        """Initialize the structural extractor."""
        self.current_class = None
        self.current_function = None
        self.scope_stack = []
        
    def extract_structure(self, code: str) -> nx.DiGraph:
        """
        Extract complete structural information from Python code.
        
        Args:
            code: Python source code to analyze
            
        Returns:
            NetworkX DiGraph containing all extracted structural information
            
        Raises:
            SyntaxError: If the code cannot be parsed
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse code: {e}")
        
        graph = nx.DiGraph()
        
        # Extract all structural components
        functions = self.extract_functions(tree)
        classes = self.extract_classes(tree)
        variables = self.extract_variables(tree)
        call_edges = self.extract_call_edges(tree)
        instantiation_edges = self.extract_instantiation_edges(tree)
        definition_edges = self.extract_definition_edges(tree)
        
        # Add nodes to graph
        for func in functions:
            graph.add_node(func.name, type='function', data=func)
            
        for cls in classes:
            graph.add_node(cls.name, type='class', data=cls)
            
        for var in variables:
            graph.add_node(var.name, type='variable', data=var)
        
        # Add edges to graph
        for edge in call_edges:
            graph.add_edge(edge.caller, edge.callee, type='CALLS', data=edge)
            
        for edge in instantiation_edges:
            graph.add_edge(edge.caller, edge.callee, type='INSTANTIATES', data=edge)
            
        for edge in definition_edges:
            graph.add_edge(edge.source, edge.target, type='DEFINED_IN', data=edge)
        
        return graph
    
    def extract_functions(self, tree: ast.AST) -> List[FunctionNode]:
        """
        Extract function definitions from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of FunctionNode objects representing all functions
        """
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract function arguments
                args = []
                for arg in node.args.args:
                    args.append(arg.arg)
                
                # Handle *args and **kwargs
                if node.args.vararg:
                    args.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")
                
                # Extract return type annotation
                return_type = None
                if node.returns:
                    return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else None
                
                # Extract docstring
                docstring = None
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value
                
                function_node = FunctionNode(
                    name=node.name,
                    args=args,
                    return_type=return_type,
                    docstring=docstring,
                    line_number=node.lineno
                )
                functions.append(function_node)
        
        return functions
    
    def extract_classes(self, tree: ast.AST) -> List[ClassNode]:
        """
        Extract class definitions from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of ClassNode objects representing all classes
        """
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        # Handle qualified names like module.BaseClass
                        bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))
                
                # Extract method names
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                
                # Extract docstring
                docstring = None
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value
                
                class_node = ClassNode(
                    name=node.name,
                    bases=bases,
                    methods=methods,
                    docstring=docstring,
                    line_number=node.lineno
                )
                classes.append(class_node)
        
        return classes
    
    def extract_variables(self, tree: ast.AST) -> List[VariableNode]:
        """
        Extract member variable definitions (self.xxx patterns) from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of VariableNode objects representing member variables
        """
        variables = []
        current_class = None
        current_method = None
        
        class VariableVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                nonlocal current_class
                old_class = current_class
                current_class = node.name
                self.generic_visit(node)
                current_class = old_class
            
            def visit_FunctionDef(self, node):
                nonlocal current_method
                old_method = current_method
                current_method = f"{current_class}.{node.name}" if current_class else node.name
                self.generic_visit(node)
                current_method = old_method
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_Assign(self, node):
                # Look for self.xxx assignments
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (isinstance(target.value, ast.Name) and 
                            target.value.id == 'self' and 
                            current_class and current_method):
                            
                            var_name = f"self.{target.attr}"
                            
                            # Try to infer type from assignment
                            var_type = None
                            if isinstance(node.value, ast.Constant):
                                var_type = type(node.value.value).__name__
                            elif isinstance(node.value, ast.Name):
                                var_type = "reference"
                            elif isinstance(node.value, ast.Call):
                                var_type = "call_result"
                            
                            variable_node = VariableNode(
                                name=var_name,
                                var_type=var_type,
                                defined_in=current_method,
                                line_number=node.lineno
                            )
                            variables.append(variable_node)
                
                self.generic_visit(node)
        
        visitor = VariableVisitor()
        visitor.visit(tree)
        
        return variables
    
    def extract_call_edges(self, tree: ast.AST) -> List[CallEdge]:
        """
        Extract function call relationships from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of CallEdge objects representing function calls
        """
        call_edges = []
        current_function = None
        
        class CallVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                nonlocal current_function
                old_function = current_function
                current_function = node.name
                self.generic_visit(node)
                current_function = old_function
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_Call(self, node):
                if current_function:
                    callee = None
                    
                    # Handle different call patterns
                    if isinstance(node.func, ast.Name):
                        callee = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        # Handle method calls like obj.method() or self.method()
                        if isinstance(node.func.value, ast.Name):
                            if node.func.value.id == 'self':
                                callee = f"self.{node.func.attr}"
                            else:
                                callee = f"{node.func.value.id}.{node.func.attr}"
                        else:
                            callee = node.func.attr
                    
                    if callee:
                        call_edge = CallEdge(
                            caller=current_function,
                            callee=callee,
                            line_number=node.lineno
                        )
                        call_edges.append(call_edge)
                
                self.generic_visit(node)
        
        visitor = CallVisitor()
        visitor.visit(tree)
        
        return call_edges
    
    def extract_instantiation_edges(self, tree: ast.AST) -> List[CallEdge]:
        """
        Extract class instantiation relationships from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of CallEdge objects representing class instantiations
        """
        instantiation_edges = []
        current_function = None
        
        class InstantiationVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                nonlocal current_function
                old_function = current_function
                current_function = node.name
                self.generic_visit(node)
                current_function = old_function
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_Call(self, node):
                if current_function:
                    # Check if this is a class instantiation (calling a class name)
                    class_name = None
                    
                    if isinstance(node.func, ast.Name):
                        # Simple class instantiation like MyClass()
                        # We assume it's a class if it starts with uppercase
                        if node.func.id[0].isupper():
                            class_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        # Qualified class instantiation like module.MyClass()
                        if node.func.attr[0].isupper():
                            class_name = node.func.attr
                    
                    if class_name:
                        instantiation_edge = CallEdge(
                            caller=current_function,
                            callee=class_name,
                            line_number=node.lineno
                        )
                        instantiation_edges.append(instantiation_edge)
                
                self.generic_visit(node)
        
        visitor = InstantiationVisitor()
        visitor.visit(tree)
        
        return instantiation_edges
    
    def extract_definition_edges(self, tree: ast.AST) -> List[DependencyEdge]:
        """
        Extract variable definition relationships from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of DependencyEdge objects representing DEFINED_IN relationships
        """
        definition_edges = []
        current_class = None
        current_method = None
        
        class DefinitionVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                nonlocal current_class
                old_class = current_class
                current_class = node.name
                self.generic_visit(node)
                current_class = old_class
            
            def visit_FunctionDef(self, node):
                nonlocal current_method
                old_method = current_method
                current_method = f"{current_class}.{node.name}" if current_class else node.name
                self.generic_visit(node)
                current_method = old_method
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_Assign(self, node):
                if current_method:
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if (isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                
                                var_name = f"self.{target.attr}"
                                
                                definition_edge = DependencyEdge(
                                    source=var_name,
                                    target=current_method,
                                    dependency_type="DEFINED_IN",
                                    context=f"class:{current_class}" if current_class else "global"
                                )
                                definition_edges.append(definition_edge)
                
                self.generic_visit(node)
        
        visitor = DefinitionVisitor()
        visitor.visit(tree)
        
        return definition_edges