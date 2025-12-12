"""Enhanced GraphManager main class implementing the four core engines."""

import ast
import time
from typing import List, Dict, Set, Optional, Any
import networkx as nx

from .models import (
    FunctionNode,
    ClassNode,
    VariableNode,
    RequirementNode,
    CallEdge,
    DependencyEdge,
    ViolationEdge,
)
from .structural_extractor import StructuralExtractor
from .semantic_injector import SemanticInjector
from .dependency_tracer import DependencyTracer
from .violation_flagger import ViolationFlagger
from .logger import get_logger
from .config import EnhancedGraphManagerConfig


class EnhancedGraphManager:
    """
    Enhanced GraphManager that combines structural extraction, semantic injection,
    dependency tracing, and violation flagging for comprehensive code analysis.
    """
    
    def __init__(self, config: Optional[EnhancedGraphManagerConfig] = None):
        """Initialize the Enhanced GraphManager."""
        self.config = config or EnhancedGraphManagerConfig()
        self.logger = get_logger()
        self.graph = nx.DiGraph()
        self.structural_extractor = StructuralExtractor()
        self.semantic_injector = SemanticInjector()
        self.dependency_tracer = DependencyTracer()
        self.violation_flagger = ViolationFlagger()
        self._last_code = None  # Store last analyzed code for dependency tracing
        self._performance_metrics = {}  # Store performance metrics
        
        self.logger.info("Enhanced GraphManager initialized")
    
    def _measure_performance(self, operation_name: str):
        """Decorator to measure operation performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self._performance_metrics[operation_name] = execution_time
                    self.logger.debug(f"{operation_name} completed in {execution_time:.3f}s")
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error(f"{operation_name} failed after {execution_time:.3f}s: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for all operations."""
        return self._performance_metrics.copy()
        
    def extract_structure(self, code: str) -> nx.DiGraph:
        """
        Extract structural information from Python code using AST analysis.
        
        Args:
            code: Python source code to analyze
            
        Returns:
            NetworkX DiGraph containing extracted structure
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting structural extraction")
            
            # Check code size limits
            if len(code) > self.config.max_nodes * 100:  # Rough estimate
                self.logger.warning(f"Code size ({len(code)} chars) may exceed processing limits")
            
            self.graph = self.structural_extractor.extract_structure(code)
            self._last_code = code  # Store for dependency tracing
            
            execution_time = time.time() - start_time
            self._performance_metrics['extract_structure'] = execution_time
            
            self.logger.info(f"Structural extraction completed: {self.graph.number_of_nodes()} nodes, "
                           f"{self.graph.number_of_edges()} edges in {execution_time:.3f}s")
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in code: {str(e)}")
            self.graph = nx.DiGraph()
            self._last_code = None
        except Exception as e:
            self.logger.error(f"Unexpected error during structural extraction: {str(e)}")
            self.graph = nx.DiGraph()
            self._last_code = None
            
        return self.graph
    
    def inject_semantics(self, requirements_text: str) -> nx.DiGraph:
        """
        Inject semantic requirements into the graph using rule-based analysis.
        
        Args:
            requirements_text: Text containing requirements to analyze
            
        Returns:
            Enhanced graph with requirement nodes
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting semantic injection")
            
            # Check requirements text size
            if len(requirements_text) > self.config.max_requirement_length:
                self.logger.warning(f"Requirements text length ({len(requirements_text)}) exceeds recommended limit")
                requirements_text = requirements_text[:self.config.max_requirement_length]
            
            # Decompose requirements text into atomic requirements
            requirements = self.semantic_injector.decompose_requirements(requirements_text)
            self.logger.debug(f"Decomposed into {len(requirements)} requirements")
            
            # Map requirements to existing code nodes
            mapping_edges = self.semantic_injector.map_requirements_to_code(requirements, self.graph)
            self.logger.debug(f"Created {len(mapping_edges)} requirement mappings")
            
            # Inject requirements and mappings into the graph
            self.graph = self.semantic_injector.inject_requirements_into_graph(
                requirements, mapping_edges, self.graph
            )
            
            execution_time = time.time() - start_time
            self._performance_metrics['inject_semantics'] = execution_time
            
            self.logger.info(f"Semantic injection completed: {len(requirements)} requirements "
                           f"injected in {execution_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error during semantic injection: {str(e)}")
            # Return current graph state on error
            
        return self.graph
    
    def trace_dependencies(self) -> Dict[str, Set[str]]:
        """
        Trace definition-usage chains and dependency relationships.
        
        Returns:
            Dictionary mapping nodes to their dependencies
        """
        start_time = time.time()
        
        if self._last_code is None:
            self.logger.warning("No code available for dependency tracing")
            return {}
        
        try:
            self.logger.info("Starting dependency tracing")
            
            # Check graph size limits
            if self.graph.number_of_nodes() > self.config.max_nodes:
                self.logger.warning(f"Graph size ({self.graph.number_of_nodes()}) exceeds recommended limit")
            
            # Trace dependencies and enhance the graph
            self.graph = self.dependency_tracer.trace_dependencies(self._last_code, self.graph)
            
            # Build dependency mapping with depth limit
            dependency_map = {}
            for node_name in self.graph.nodes():
                dependencies = self.dependency_tracer.get_transitive_dependencies(self.graph, node_name)
                # Limit dependency depth to prevent performance issues
                if len(dependencies) > self.config.max_dependency_depth:
                    self.logger.warning(f"Node {node_name} has {len(dependencies)} dependencies, "
                                      f"exceeding limit of {self.config.max_dependency_depth}")
                dependency_map[node_name] = dependencies
            
            execution_time = time.time() - start_time
            self._performance_metrics['trace_dependencies'] = execution_time
            
            self.logger.info(f"Dependency tracing completed: {len(dependency_map)} nodes analyzed "
                           f"in {execution_time:.3f}s")
            
            return dependency_map
            
        except Exception as e:
            self.logger.error(f"Error during dependency tracing: {str(e)}")
            return {}
    
    def flag_violations(self) -> List[ViolationEdge]:
        """
        Detect and flag potential requirement violations.
        
        Returns:
            List of violation edges with details
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting violation flagging")
            
            # Flag potential violations
            violation_edges = self.violation_flagger.flag_potential_violations(self.graph)
            
            # Limit number of violations to prevent performance issues
            if len(violation_edges) > self.config.max_violations_per_requirement * 10:
                self.logger.warning(f"Large number of violations ({len(violation_edges)}) detected, "
                                  f"limiting to top violations")
                violation_edges = violation_edges[:self.config.max_violations_per_requirement * 10]
            
            # Add violation edges to the graph
            added_edges = 0
            for edge in violation_edges:
                if self.graph.has_node(edge.requirement) and self.graph.has_node(edge.code_node):
                    self.graph.add_edge(
                        edge.requirement,
                        edge.code_node,
                        type=edge.status,  # VIOLATES or SATISFIES
                        data=edge
                    )
                    added_edges += 1
            
            execution_time = time.time() - start_time
            self._performance_metrics['flag_violations'] = execution_time
            
            self.logger.info(f"Violation flagging completed: {len(violation_edges)} violations found, "
                           f"{added_edges} edges added in {execution_time:.3f}s")
            
            return violation_edges
            
        except Exception as e:
            self.logger.error(f"Error during violation flagging: {str(e)}")
            return []
    
    def get_graph(self) -> nx.DiGraph:
        """
        Get the current knowledge graph.
        
        Returns:
            The NetworkX DiGraph representing the knowledge graph
        """
        return self.graph
    
    def get_violation_report(self) -> Dict[str, Any]:
        """
        Get a structured violation report.
        
        Returns:
            Dictionary containing violation analysis results
        """
        try:
            reports = self.violation_flagger.analyze_requirement_satisfaction(self.graph)
            prioritized_violations = self.violation_flagger.prioritize_violations(reports)
            
            return {
                'total_reports': len(reports),
                'total_violations': len([r for r in reports if r.status == 'VIOLATES']),
                'total_satisfies': len([r for r in reports if r.status == 'SATISFIES']),
                'total_unknown': len([r for r in reports if r.status == 'UNKNOWN']),
                'prioritized_violations': [
                    {
                        'requirement_id': r.requirement_id,
                        'code_node': r.code_node,
                        'status': r.status,
                        'reason': r.reason,
                        'confidence': r.confidence,
                        'severity': r.severity
                    }
                    for r in prioritized_violations
                ],
                'all_reports': [
                    {
                        'requirement_id': r.requirement_id,
                        'code_node': r.code_node,
                        'status': r.status,
                        'reason': r.reason,
                        'confidence': r.confidence,
                        'severity': r.severity
                    }
                    for r in reports
                ]
            }
        except Exception as e:
            return {
                'error': str(e),
                'total_reports': 0,
                'total_violations': 0,
                'total_satisfies': 0,
                'total_unknown': 0,
                'prioritized_violations': [],
                'all_reports': []
            }
    
    def get_dependency_analysis(self) -> Dict[str, Any]:
        """
        Get dependency analysis results.
        
        Returns:
            Dictionary containing dependency information
        """
        try:
            dependencies = self.trace_dependencies()
            
            # Calculate dependency statistics
            total_nodes = len(dependencies)
            nodes_with_deps = len([node for node, deps in dependencies.items() if deps])
            
            # Find most dependent nodes
            most_dependent = sorted(
                [(node, len(deps)) for node, deps in dependencies.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'total_nodes': total_nodes,
                'nodes_with_dependencies': nodes_with_deps,
                'dependency_ratio': nodes_with_deps / total_nodes if total_nodes > 0 else 0,
                'most_dependent_nodes': [
                    {'node': node, 'dependency_count': count}
                    for node, count in most_dependent
                ],
                'all_dependencies': dependencies
            }
        except Exception as e:
            return {
                'error': str(e),
                'total_nodes': 0,
                'nodes_with_dependencies': 0,
                'dependency_ratio': 0,
                'most_dependent_nodes': [],
                'all_dependencies': {}
            }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dictionary containing graph statistics
        """
        try:
            # Count nodes by type
            node_types = {}
            for node_name, node_data in self.graph.nodes(data=True):
                node_type = node_data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Count edges by type
            edge_types = {}
            for source, target, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get('type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            return {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'node_types': node_types,
                'edge_types': edge_types,
                'is_directed': self.graph.is_directed(),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0
            }
        except Exception as e:
            return {
                'error': str(e),
                'total_nodes': 0,
                'total_edges': 0,
                'node_types': {},
                'edge_types': {},
                'is_directed': True,
                'density': 0
            }
    
    def serialize_graph(self) -> Dict[str, Any]:
        """
        Serialize the graph for persistence or exchange.
        
        Returns:
            Serialized graph data
        """
        try:
            # Convert NetworkX graph to a serializable format
            serialized_data = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'directed': self.graph.is_directed(),
                    'node_count': self.graph.number_of_nodes(),
                    'edge_count': self.graph.number_of_edges()
                }
            }
            
            # Serialize nodes
            for node_name, node_data in self.graph.nodes(data=True):
                node_info = {
                    'id': node_name,
                    'type': node_data.get('type', 'unknown'),
                    'data': self._serialize_node_data(node_data.get('data'))
                }
                serialized_data['nodes'].append(node_info)
            
            # Serialize edges
            for source, target, edge_data in self.graph.edges(data=True):
                edge_info = {
                    'source': source,
                    'target': target,
                    'type': edge_data.get('type', 'unknown'),
                    'data': self._serialize_edge_data(edge_data.get('data'))
                }
                serialized_data['edges'].append(edge_info)
            
            return serialized_data
            
        except Exception as e:
            return {
                'error': str(e),
                'nodes': [],
                'edges': [],
                'metadata': {}
            }
    
    def deserialize_graph(self, data: Dict[str, Any]) -> bool:
        """
        Deserialize graph data to restore the graph state.
        
        Args:
            data: Serialized graph data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear current graph
            self.graph.clear()
            
            # Restore nodes
            for node_info in data.get('nodes', []):
                node_id = node_info['id']
                node_type = node_info.get('type', 'unknown')
                node_data = self._deserialize_node_data(node_info.get('data'))
                
                self.graph.add_node(node_id, type=node_type, data=node_data)
            
            # Restore edges
            for edge_info in data.get('edges', []):
                source = edge_info['source']
                target = edge_info['target']
                edge_type = edge_info.get('type', 'unknown')
                edge_data = self._deserialize_edge_data(edge_info.get('data'))
                
                if self.graph.has_node(source) and self.graph.has_node(target):
                    self.graph.add_edge(source, target, type=edge_type, data=edge_data)
            
            return True
            
        except Exception as e:
            # Could add logging here in the future
            return False
    
    def analyze_code_impact(self, changed_nodes: Set[str]) -> Dict[str, Any]:
        """
        Analyze the impact of code changes on requirements and dependencies.
        
        Args:
            changed_nodes: Set of node names that have changed
            
        Returns:
            Dictionary containing impact analysis
        """
        try:
            # Find affected dependencies
            affected_nodes = set()
            for changed_node in changed_nodes:
                if self.graph.has_node(changed_node):
                    impact_nodes = self.dependency_tracer.analyze_dependency_impact(
                        self.graph, changed_node
                    )
                    affected_nodes.update(impact_nodes)
            
            # Update violation status for affected requirements
            updated_violations = self.violation_flagger.update_violation_status(
                self.graph, changed_nodes
            )
            
            return {
                'changed_nodes': list(changed_nodes),
                'affected_nodes': list(affected_nodes),
                'total_affected': len(affected_nodes),
                'updated_violations': [
                    {
                        'requirement_id': r.requirement_id,
                        'code_node': r.code_node,
                        'status': r.status,
                        'reason': r.reason,
                        'confidence': r.confidence,
                        'severity': r.severity
                    }
                    for r in updated_violations
                ]
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'changed_nodes': list(changed_nodes),
                'affected_nodes': [],
                'total_affected': 0,
                'updated_violations': []
            }
    
    def _serialize_node_data(self, data) -> Optional[Dict[str, Any]]:
        """Serialize node data to a dictionary."""
        if data is None:
            return None
        
        try:
            # Convert dataclass to dictionary
            if hasattr(data, '__dict__'):
                return {
                    'type': type(data).__name__,
                    'attributes': data.__dict__
                }
            else:
                return {'value': str(data)}
        except Exception:
            return {'value': str(data)}
    
    def _serialize_edge_data(self, data) -> Optional[Dict[str, Any]]:
        """Serialize edge data to a dictionary."""
        if data is None:
            return None
        
        try:
            # Convert dataclass to dictionary
            if hasattr(data, '__dict__'):
                return {
                    'type': type(data).__name__,
                    'attributes': data.__dict__
                }
            else:
                return {'value': str(data)}
        except Exception:
            return {'value': str(data)}
    
    def _deserialize_node_data(self, data: Optional[Dict[str, Any]]):
        """Deserialize node data from a dictionary."""
        if data is None:
            return None
        
        try:
            if 'type' in data and 'attributes' in data:
                # This is a simplified deserialization
                # In a full implementation, we would reconstruct the actual objects
                return data['attributes']
            else:
                return data.get('value')
        except Exception:
            return None
    
    def _deserialize_edge_data(self, data: Optional[Dict[str, Any]]):
        """Deserialize edge data from a dictionary."""
        if data is None:
            return None
        
        try:
            if 'type' in data and 'attributes' in data:
                # This is a simplified deserialization
                # In a full implementation, we would reconstruct the actual objects
                return data['attributes']
            else:
                return data.get('value')
        except Exception:
            return None
    def analyze_complete_workflow(self, code: str, requirements_text: str) -> Dict[str, Any]:
        """
        Run the complete analysis workflow: structure -> semantics -> dependencies -> violations.
        
        Args:
            code: Python source code to analyze
            requirements_text: Requirements text to inject
            
        Returns:
            Dictionary containing complete analysis results
        """
        workflow_start = time.time()
        
        try:
            self.logger.info("Starting complete workflow analysis")
            
            # Step 1: Extract structure
            graph = self.extract_structure(code)
            
            # Step 2: Inject semantics
            enhanced_graph = self.inject_semantics(requirements_text)
            
            # Step 3: Trace dependencies
            dependencies = self.trace_dependencies()
            
            # Step 4: Flag violations
            violations = self.flag_violations()
            
            # Compile results
            workflow_time = time.time() - workflow_start
            self._performance_metrics['complete_workflow'] = workflow_time
            
            results = {
                'success': True,
                'execution_time': workflow_time,
                'graph_statistics': self.get_graph_statistics(),
                'dependency_analysis': self.get_dependency_analysis(),
                'violation_report': self.get_violation_report(),
                'performance_metrics': self.get_performance_metrics()
            }
            
            self.logger.info(f"Complete workflow analysis finished in {workflow_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Complete workflow analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - workflow_start,
                'performance_metrics': self.get_performance_metrics()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the Enhanced GraphManager.
        
        Returns:
            Dictionary containing health status
        """
        try:
            health_status = {
                'status': 'healthy',
                'graph_nodes': self.graph.number_of_nodes(),
                'graph_edges': self.graph.number_of_edges(),
                'has_code': self._last_code is not None,
                'config': {
                    'max_nodes': self.config.max_nodes,
                    'max_edges': self.config.max_edges,
                    'max_dependency_depth': self.config.max_dependency_depth
                },
                'performance_metrics': self._performance_metrics
            }
            
            # Check for potential issues
            warnings = []
            
            if self.graph.number_of_nodes() > self.config.max_nodes * 0.8:
                warnings.append(f"Graph approaching node limit ({self.graph.number_of_nodes()}/{self.config.max_nodes})")
            
            if self.graph.number_of_edges() > self.config.max_edges * 0.8:
                warnings.append(f"Graph approaching edge limit ({self.graph.number_of_edges()}/{self.config.max_edges})")
            
            if warnings:
                health_status['warnings'] = warnings
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def reset(self):
        """Reset the Enhanced GraphManager to initial state."""
        try:
            self.logger.info("Resetting Enhanced GraphManager")
            self.graph.clear()
            self._last_code = None
            self._performance_metrics.clear()
            
            # Reset component states
            self.semantic_injector.requirement_counter = 0
            self.dependency_tracer.variable_definitions.clear()
            self.dependency_tracer.variable_usages.clear()
            
            self.logger.info("Enhanced GraphManager reset completed")
            
        except Exception as e:
            self.logger.error(f"Error during reset: {str(e)}")
            raise