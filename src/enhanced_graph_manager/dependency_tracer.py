"""Dependency tracing engine for Enhanced GraphManager."""

import ast
from typing import List, Dict, Set, Optional, Tuple
import networkx as nx

from .models import DependencyEdge


class DependencyTracer:
    """
    Traces definition-usage chains and dependency relationships in code.
    
    This class implements the dependency tracing engine that:
    - Creates DEPENDS_ON and USES_VAR edges
    - Builds definition-usage chains
    - Resolves self references
    - Maintains transitive dependency relationships
    """
    
    def __init__(self):
        """Initialize the dependency tracer."""
        self.variable_definitions = {}  # var_name -> defining_node
        self.variable_usages = {}       # var_name -> [using_nodes]
        
    def trace_dependencies(self, code: str, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Trace all dependency relationships in the code and add them to the graph.
        
        Args:
            code: Python source code to analyze
            graph: Existing graph to enhance with dependencies
            
        Returns:
            Enhanced graph with dependency edges
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return graph
        
        # Clear previous analysis
        self.variable_definitions.clear()
        self.variable_usages.clear()
        
        # Build definition-usage chains
        self._build_definition_usage_chains(tree)
        
        # Create dependency edges
        dependency_edges = self._create_dependency_edges()
        
        # Add dependency edges to graph
        for edge in dependency_edges:
            # Try to find matching nodes in the graph
            source_node = self._find_matching_node(edge.source, graph)
            target_node = self._find_matching_node(edge.target, graph)
            
            if source_node and target_node:
                graph.add_edge(
                    source_node,
                    target_node,
                    type=edge.dependency_type,
                    data=edge
                )
        
        # Resolve self references
        self._resolve_self_references(graph)
        
        # Build transitive dependencies
        self._build_transitive_dependencies(graph)
        
        return graph
    
    def find_definition_usage_chains(self, variable_name: str) -> List[Tuple[str, str]]:
        """
        Find definition-usage chains for a specific variable.
        
        Args:
            variable_name: Name of the variable to trace
            
        Returns:
            List of (definition_node, usage_node) tuples
        """
        chains = []
        
        if variable_name in self.variable_definitions:
            definition_node = self.variable_definitions[variable_name]
            usage_nodes = self.variable_usages.get(variable_name, [])
            
            for usage_node in usage_nodes:
                chains.append((definition_node, usage_node))
        
        return chains
    
    def get_transitive_dependencies(self, graph: nx.DiGraph, node: str) -> Set[str]:
        """
        Get all transitive dependencies for a node.
        
        Args:
            graph: Graph to analyze
            node: Node to find dependencies for
            
        Returns:
            Set of nodes that the given node transitively depends on
        """
        if not graph.has_node(node):
            return set()
        
        dependencies = set()
        visited = set()
        
        def _collect_dependencies(current_node):
            if current_node in visited:
                return
            visited.add(current_node)
            
            # Find direct dependencies
            for _, target, edge_data in graph.edges(current_node, data=True):
                if edge_data.get('type') in ['DEPENDS_ON', 'USES_VAR']:
                    dependencies.add(target)
                    _collect_dependencies(target)
        
        _collect_dependencies(node)
        return dependencies
    
    def _build_definition_usage_chains(self, tree: ast.AST):
        """Build definition-usage chains from AST."""
        current_class = None
        current_function = None
        
        # Create a visitor instance with access to our data structures
        visitor = self._create_dependency_visitor()
        
        visitor.visit(tree)
    
    def _create_dependency_edges(self) -> List[DependencyEdge]:
        """Create dependency edges from definition-usage analysis."""
        edges = []
        
        # Create USES_VAR edges (usage -> definition)
        for var_name, usage_nodes in self.variable_usages.items():
            if var_name in self.variable_definitions:
                definition_node = self.variable_definitions[var_name]
                
                for usage_node in usage_nodes:
                    if usage_node != definition_node:  # Don't create self-loops
                        edge = DependencyEdge(
                            source=usage_node,
                            target=definition_node,
                            dependency_type="USES_VAR",
                            context=f"variable:{var_name}"
                        )
                        edges.append(edge)
        
        # Create DEPENDS_ON edges (definition -> usage for data flow)
        for var_name, definition_node in self.variable_definitions.items():
            usage_nodes = self.variable_usages.get(var_name, [])
            
            for usage_node in usage_nodes:
                if usage_node != definition_node:  # Don't create self-loops
                    edge = DependencyEdge(
                        source=definition_node,
                        target=usage_node,
                        dependency_type="DEPENDS_ON",
                        context=f"provides:{var_name}"
                    )
                    edges.append(edge)
        
        return edges
    
    def _resolve_self_references(self, graph: nx.DiGraph):
        """Resolve self.method() calls to actual method definitions."""
        # Find all self.method calls and link them to method definitions
        for node_name, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'function':
                # Check if this function makes self calls
                for _, target, edge_data in graph.edges(node_name, data=True):
                    if (edge_data.get('type') == 'CALLS' and 
                        target.startswith('self.')):
                        
                        # Extract method name
                        method_name = target[5:]  # Remove 'self.'
                        
                        # Find the actual method node
                        actual_method = None
                        for potential_node, potential_data in graph.nodes(data=True):
                            if (potential_data.get('type') == 'function' and 
                                potential_node.endswith(f".{method_name}")):
                                actual_method = potential_node
                                break
                        
                        if actual_method and actual_method != node_name:
                            # Add resolved self reference edge
                            graph.add_edge(
                                node_name,
                                actual_method,
                                type='SELF_CALLS',
                                data=DependencyEdge(
                                    source=node_name,
                                    target=actual_method,
                                    dependency_type="SELF_CALLS",
                                    context=f"resolves:{target}"
                                )
                            )
    
    def _build_transitive_dependencies(self, graph: nx.DiGraph):
        """Build and cache transitive dependency relationships."""
        # This could be optimized with caching for large graphs
        # For now, we'll compute them on-demand in get_transitive_dependencies
        pass
    
    def analyze_dependency_impact(self, graph: nx.DiGraph, changed_node: str) -> Set[str]:
        """
        Analyze the impact of changes to a node on other nodes.
        
        Args:
            graph: Graph to analyze
            changed_node: Node that has changed
            
        Returns:
            Set of nodes that might be affected by the change
        """
        affected_nodes = set()
        
        if not graph.has_node(changed_node):
            return affected_nodes
        
        # Find all nodes that depend on the changed node
        for source, target, edge_data in graph.edges(data=True):
            if (target == changed_node and 
                edge_data.get('type') in ['DEPENDS_ON', 'USES_VAR', 'CALLS']):
                affected_nodes.add(source)
        
        # Also find transitive dependencies
        visited = set()
        
        def _find_transitive_impact(node):
            if node in visited:
                return
            visited.add(node)
            
            for source, target, edge_data in graph.edges(data=True):
                if (target == node and 
                    edge_data.get('type') in ['DEPENDS_ON', 'USES_VAR', 'CALLS']):
                    affected_nodes.add(source)
                    _find_transitive_impact(source)
        
        _find_transitive_impact(changed_node)
        
        return affected_nodes
    
    def _create_dependency_visitor(self):
        """Create a dependency visitor with proper context handling."""
        current_class = None
        current_function = None
        
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self, tracer):
                self.tracer = tracer
            
            def visit_ClassDef(self, node):
                nonlocal current_class
                old_class = current_class
                current_class = node.name
                self.generic_visit(node)
                current_class = old_class
            
            def visit_FunctionDef(self, node):
                nonlocal current_function
                old_function = current_function
                current_function = f"{current_class}.{node.name}" if current_class else node.name
                self.generic_visit(node)
                current_function = old_function
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_Assign(self, node):
                if current_function:
                    # Handle variable definitions
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if (isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                var_name = f"self.{target.attr}"
                                self.tracer.variable_definitions[var_name] = current_function
                        elif isinstance(target, ast.Name):
                            var_name = target.id
                            self.tracer.variable_definitions[var_name] = current_function
                
                # Check for variable usage in the assignment value
                self._check_variable_usage(node.value, current_function)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and current_function:
                    # This is a variable usage
                    var_name = node.id
                    if var_name not in self.tracer.variable_usages:
                        self.tracer.variable_usages[var_name] = []
                    if current_function not in self.tracer.variable_usages[var_name]:
                        self.tracer.variable_usages[var_name].append(current_function)
                
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == 'self' and 
                    isinstance(node.ctx, ast.Load) and 
                    current_function):
                    # This is a self.attribute usage
                    var_name = f"self.{node.attr}"
                    if var_name not in self.tracer.variable_usages:
                        self.tracer.variable_usages[var_name] = []
                    if current_function not in self.tracer.variable_usages[var_name]:
                        self.tracer.variable_usages[var_name].append(current_function)
                
                self.generic_visit(node)
            
            def _check_variable_usage(self, node, function_context):
                """Recursively check for variable usage in an expression."""
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    var_name = node.id
                    if var_name not in self.tracer.variable_usages:
                        self.tracer.variable_usages[var_name] = []
                    if function_context not in self.tracer.variable_usages[var_name]:
                        self.tracer.variable_usages[var_name].append(function_context)
                elif isinstance(node, ast.Attribute):
                    if (isinstance(node.value, ast.Name) and 
                        node.value.id == 'self'):
                        var_name = f"self.{node.attr}"
                        if var_name not in self.tracer.variable_usages:
                            self.tracer.variable_usages[var_name] = []
                        if function_context not in self.tracer.variable_usages[var_name]:
                            self.tracer.variable_usages[var_name].append(function_context)
                
                # Recursively check child nodes
                for child in ast.iter_child_nodes(node):
                    self._check_variable_usage(child, function_context)
        
        return DependencyVisitor(self)
    
    def _find_matching_node(self, node_name: str, graph: nx.DiGraph) -> Optional[str]:
        """Find a matching node in the graph for the given node name."""
        # Direct match
        if graph.has_node(node_name):
            return node_name
        
        # Try to match function names without class prefix
        if '.' in node_name:
            simple_name = node_name.split('.')[-1]  # Get last part after dot
            if graph.has_node(simple_name):
                return simple_name
        
        # Try to match with class prefix
        for graph_node in graph.nodes():
            if graph_node.endswith(f".{node_name}") or graph_node == node_name:
                return graph_node
        
        return None