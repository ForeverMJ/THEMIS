"""Semantic injection engine for Enhanced GraphManager."""

import re
from typing import List, Dict, Set, Optional, Tuple
import networkx as nx

from .models import RequirementNode, DependencyEdge


class SemanticInjector:
    """
    Injects semantic requirements into the graph using rule-based analysis.
    
    This class implements the semantic injection engine that processes
    issue text and maps requirements to relevant code components:
    - Decomposes complex issue text into atomic requirements
    - Maps requirements to related code nodes
    - Maintains requirement hierarchy and traceability
    """
    
    def __init__(self):
        """Initialize the semantic injector."""
        self.requirement_counter = 0
        
    def decompose_requirements(self, issue_text: str) -> List[RequirementNode]:
        """
        Decompose complex issue text into atomic requirement nodes.
        
        Args:
            issue_text: Text containing requirements to analyze
            
        Returns:
            List of RequirementNode objects representing atomic requirements
        """
        requirements = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(issue_text)
        
        for sentence in sentences:
            # Skip empty or very short sentences
            if len(sentence.strip()) < 10:
                continue
                
            # Check if sentence contains requirement indicators
            if self._is_requirement_sentence(sentence):
                requirement = self._create_requirement_node(sentence)
                requirements.append(requirement)
        
        return requirements
    
    def map_requirements_to_code(self, requirements: List[RequirementNode], 
                                graph: nx.DiGraph) -> List[DependencyEdge]:
        """
        Map requirements to relevant code components in the graph.
        
        Args:
            requirements: List of requirement nodes to map
            graph: NetworkX graph containing code structure
            
        Returns:
            List of DependencyEdge objects representing requirement mappings
        """
        mapping_edges = []
        
        for requirement in requirements:
            # Find relevant code nodes for this requirement
            relevant_nodes = self._find_relevant_code_nodes(requirement, graph)
            
            # Create mapping edges
            for node_name, relevance_score in relevant_nodes:
                if relevance_score > 0.3:  # Threshold for relevance
                    edge = DependencyEdge(
                        source=requirement.id,
                        target=node_name,
                        dependency_type="MAPS_TO",
                        context=f"relevance:{relevance_score:.2f}"
                    )
                    mapping_edges.append(edge)
        
        return mapping_edges
    
    def inject_requirements_into_graph(self, requirements: List[RequirementNode],
                                     mapping_edges: List[DependencyEdge],
                                     graph: nx.DiGraph) -> nx.DiGraph:
        """
        Inject requirement nodes and mappings into the existing graph.
        
        Args:
            requirements: List of requirement nodes to inject
            mapping_edges: List of mapping edges
            graph: Existing graph to enhance
            
        Returns:
            Enhanced graph with requirement nodes and mappings
        """
        # Add requirement nodes to graph
        for requirement in requirements:
            graph.add_node(
                requirement.id,
                type='requirement',
                data=requirement
            )
        
        # Add mapping edges to graph
        for edge in mapping_edges:
            if graph.has_node(edge.source) and graph.has_node(edge.target):
                graph.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.dependency_type,
                    data=edge
                )
        
        return graph
    
    def analyze_requirement_relevance(self, requirement: RequirementNode,
                                    code_node_name: str, 
                                    code_node_data: dict) -> float:
        """
        Analyze relevance between a requirement and a code node.
        
        Args:
            requirement: Requirement node to analyze
            code_node_name: Name of the code node
            code_node_data: Data associated with the code node
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        relevance_score = 0.0
        requirement_text = requirement.text.lower()
        
        # Check name similarity
        if code_node_name.lower() in requirement_text:
            relevance_score += 0.4
        
        # Check for keyword matches
        code_keywords = self._extract_keywords_from_code_node(code_node_name, code_node_data)
        requirement_keywords = self._extract_keywords_from_text(requirement_text)
        
        common_keywords = code_keywords.intersection(requirement_keywords)
        if common_keywords:
            relevance_score += min(0.5, len(common_keywords) * 0.1)
        
        # Check for functional relationships
        if self._has_functional_relationship(requirement_text, code_node_name, code_node_data):
            relevance_score += 0.3
        
        return min(1.0, relevance_score)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting using periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_requirement_sentence(self, sentence: str) -> bool:
        """Check if a sentence contains requirement indicators."""
        requirement_indicators = [
            'should', 'must', 'shall', 'need', 'require', 'want', 'expect',
            'implement', 'add', 'create', 'build', 'develop', 'fix', 'update',
            'improve', 'enhance', 'support', 'handle', 'process', 'manage'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in requirement_indicators)
    
    def _create_requirement_node(self, sentence: str) -> RequirementNode:
        """Create a requirement node from a sentence."""
        self.requirement_counter += 1
        
        # Determine priority based on keywords
        priority = self._determine_priority(sentence)
        
        # Determine if testable
        testable = self._is_testable_requirement(sentence)
        
        return RequirementNode(
            id=f"REQ-{self.requirement_counter:03d}",
            text=sentence.strip(),
            priority=priority,
            testable=testable
        )
    
    def _determine_priority(self, sentence: str) -> int:
        """Determine priority of a requirement (1=high, 5=low)."""
        high_priority_keywords = ['critical', 'urgent', 'must', 'essential', 'required']
        medium_priority_keywords = ['should', 'important', 'need']
        
        sentence_lower = sentence.lower()
        
        if any(keyword in sentence_lower for keyword in high_priority_keywords):
            return 1
        elif any(keyword in sentence_lower for keyword in medium_priority_keywords):
            return 3
        else:
            return 5
    
    def _is_testable_requirement(self, sentence: str) -> bool:
        """Determine if a requirement is testable."""
        non_testable_keywords = [
            'user-friendly', 'intuitive', 'easy', 'simple', 'clean', 'beautiful',
            'maintainable', 'readable', 'elegant', 'performance', 'fast', 'slow'
        ]
        
        sentence_lower = sentence.lower()
        return not any(keyword in sentence_lower for keyword in non_testable_keywords)
    
    def _find_relevant_code_nodes(self, requirement: RequirementNode,
                                 graph: nx.DiGraph) -> List[Tuple[str, float]]:
        """Find code nodes relevant to a requirement."""
        relevant_nodes = []
        
        for node_name, node_data in graph.nodes(data=True):
            # Skip requirement nodes
            if node_data.get('type') == 'requirement':
                continue
                
            relevance_score = self.analyze_requirement_relevance(
                requirement, node_name, node_data
            )
            
            if relevance_score > 0:
                relevant_nodes.append((node_name, relevance_score))
        
        # Sort by relevance score (descending)
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return relevant_nodes
    
    def _extract_keywords_from_code_node(self, node_name: str, node_data: dict) -> Set[str]:
        """Extract keywords from a code node."""
        keywords = set()
        
        # Add node name parts
        name_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', node_name)
        keywords.update(part.lower() for part in name_parts)
        
        # Add keywords from node data
        if 'data' in node_data:
            data_obj = node_data['data']
            
            # Extract from docstring if available
            if hasattr(data_obj, 'docstring') and data_obj.docstring:
                doc_keywords = self._extract_keywords_from_text(data_obj.docstring)
                keywords.update(doc_keywords)
            
            # Extract from method names if it's a class
            if hasattr(data_obj, 'methods'):
                for method in data_obj.methods:
                    method_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', method)
                    keywords.update(part.lower() for part in method_parts)
        
        return keywords
    
    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        
        return keywords
    
    def _has_functional_relationship(self, requirement_text: str, 
                                   code_node_name: str, 
                                   code_node_data: dict) -> bool:
        """Check if there's a functional relationship between requirement and code."""
        # Check for action words that might relate to function names
        action_words = [
            'create', 'build', 'make', 'generate', 'produce',
            'update', 'modify', 'change', 'edit', 'alter',
            'delete', 'remove', 'clear', 'clean',
            'get', 'fetch', 'retrieve', 'obtain',
            'set', 'assign', 'configure', 'setup',
            'validate', 'check', 'verify', 'test',
            'process', 'handle', 'manage', 'control'
        ]
        
        requirement_lower = requirement_text.lower()
        node_name_lower = code_node_name.lower()
        
        # Check if any action word in requirement matches function name pattern
        for action in action_words:
            if action in requirement_lower and action in node_name_lower:
                return True
        
        return False