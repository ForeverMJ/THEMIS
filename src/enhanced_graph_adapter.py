"""
Adapter to integrate Enhanced GraphManager with the existing experiment framework.

This adapter provides compatibility between the Enhanced GraphManager and the 
original GraphManager interface used in the experiment system.
"""

import networkx as nx
from typing import Any

from src.enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
from src.enhanced_graph_manager.logger import set_log_level


class EnhancedGraphAdapter:
    """
    Adapter class that provides the same interface as the original GraphManager
    but uses the Enhanced GraphManager internally.
    """
    
    def __init__(self):
        """Initialize the Enhanced GraphManager adapter."""
        # Set log level to reduce noise during experiments
        set_log_level("ERROR")
        
        self.enhanced_manager = EnhancedGraphManager()
    
    def parse_code_structure(self, code: str) -> nx.DiGraph:
        """
        Parse code structure using Enhanced GraphManager.
        
        Args:
            code: Python source code to analyze
            
        Returns:
            NetworkX DiGraph containing extracted structure
        """
        # Use Enhanced GraphManager to extract structure
        graph = self.enhanced_manager.extract_structure(code)
        
        # Convert to format expected by original system
        converted_graph = nx.DiGraph()
        
        for node_name, node_data in graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            
            # Skip requirement nodes in structural parsing
            if node_type == 'requirement':
                continue
            
            # Extract summary from node data
            summary = ""
            if 'data' in node_data and hasattr(node_data['data'], 'docstring'):
                summary = node_data['data'].docstring or ""
            
            converted_graph.add_node(
                node_name,
                type=node_type,
                name=node_name,
                code_summary=summary
            )
        
        # Copy edges, focusing on CALLS relationships
        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('type', 'unknown')
            
            # Only include structural edges in the base graph
            if edge_type in ['CALLS', 'INSTANTIATES']:
                if converted_graph.has_node(source) and converted_graph.has_node(target):
                    converted_graph.add_edge(source, target, type=edge_type)
        
        return converted_graph
    
    def enrich_with_requirements(self, graph: nx.DiGraph, requirements: str, llm=None) -> nx.DiGraph:
        """
        Enrich graph with requirements using Enhanced GraphManager.
        
        Args:
            graph: Base structural graph
            requirements: Requirements text to inject
            llm: LLM instance (not used in Enhanced GraphManager)
            
        Returns:
            Enhanced graph with requirement nodes and relationships
        """
        # First, restore the graph in Enhanced GraphManager
        self.enhanced_manager.graph = graph.copy()
        
        # Inject semantics using Enhanced GraphManager
        enhanced_graph = self.enhanced_manager.inject_semantics(requirements)
        
        # Trace dependencies
        self.enhanced_manager.trace_dependencies()
        
        # Flag violations
        violations = self.enhanced_manager.flag_violations()
        
        # Convert back to expected format
        converted_graph = graph.copy()
        
        # Add requirement nodes
        for node_name, node_data in enhanced_graph.nodes(data=True):
            if node_data.get('type') == 'requirement':
                req_data = node_data.get('data')
                if req_data:
                    converted_graph.add_node(
                        node_name,
                        type='requirement',
                        name=node_name,
                        code_summary=req_data.text
                    )
        
        # Add requirement edges with proper relation types
        for source, target, edge_data in enhanced_graph.edges(data=True):
            edge_type = edge_data.get('type', 'unknown')
            
            if edge_type in ['VIOLATES', 'SATISFIES', 'MAPS_TO']:
                # Convert MAPS_TO to SATISFIES for compatibility
                relation_type = 'SATISFIES' if edge_type == 'MAPS_TO' else edge_type
                
                if converted_graph.has_node(source) and converted_graph.has_node(target):
                    converted_graph.add_edge(source, target, type=relation_type)
        
        # Add violation edges from the violation analysis
        for violation in violations:
            if (converted_graph.has_node(violation.requirement) and 
                converted_graph.has_node(violation.code_node)):
                converted_graph.add_edge(
                    violation.requirement,
                    violation.code_node,
                    type=violation.status  # VIOLATES or SATISFIES
                )
        
        return converted_graph
    
    def get_analysis_report(self) -> dict:
        """
        Get detailed analysis report from Enhanced GraphManager.
        
        Returns:
            Dictionary containing comprehensive analysis results
        """
        return {
            'graph_statistics': self.enhanced_manager.get_graph_statistics(),
            'dependency_analysis': self.enhanced_manager.get_dependency_analysis(),
            'violation_report': self.enhanced_manager.get_violation_report(),
            'performance_metrics': self.enhanced_manager.get_performance_metrics(),
            'health_status': self.enhanced_manager.health_check()
        }