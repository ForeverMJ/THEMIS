"""Violation flagging engine for Enhanced GraphManager."""

from typing import List, Dict, Set, Optional, Tuple
import networkx as nx
from dataclasses import dataclass

from .models import RequirementNode, ViolationEdge


@dataclass
class ViolationReport:
    """Represents a violation report with details."""
    requirement_id: str
    code_node: str
    status: str  # SATISFIES, VIOLATES, UNKNOWN
    reason: str
    confidence: float
    severity: int  # 1=critical, 5=low


class ViolationFlagger:
    """
    Detects and flags potential requirement violations in code.
    
    This class implements the violation flagging engine that:
    - Analyzes requirement-code relationships
    - Assigns SATISFIES/VIOLATES status
    - Provides actionable violation information
    - Prioritizes violations by severity and impact
    """
    
    def __init__(self):
        """Initialize the violation flagger."""
        self.violation_patterns = self._initialize_violation_patterns()
        
    def analyze_requirement_satisfaction(self, graph: nx.DiGraph) -> List[ViolationReport]:
        """
        Analyze requirement satisfaction across the entire graph.
        
        Args:
            graph: Graph containing requirements and code nodes
            
        Returns:
            List of ViolationReport objects
        """
        reports = []
        
        # Find all requirement nodes
        requirement_nodes = []
        for node_name, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'requirement':
                requirement_nodes.append((node_name, node_data.get('data')))
        
        # Analyze each requirement
        for req_node_name, req_data in requirement_nodes:
            if req_data:
                # Find related code nodes
                related_code_nodes = self._find_related_code_nodes(req_node_name, graph)
                
                # Analyze satisfaction for each related code node
                for code_node in related_code_nodes:
                    report = self._analyze_single_requirement_satisfaction(
                        req_data, code_node, graph
                    )
                    reports.append(report)
        
        return reports
    
    def flag_potential_violations(self, graph: nx.DiGraph) -> List[ViolationEdge]:
        """
        Flag potential violations and create violation edges.
        
        Args:
            graph: Graph to analyze for violations
            
        Returns:
            List of ViolationEdge objects representing violations
        """
        violation_edges = []
        reports = self.analyze_requirement_satisfaction(graph)
        
        for report in reports:
            if report.status in ['VIOLATES', 'UNKNOWN']:
                edge = ViolationEdge(
                    requirement=report.requirement_id,
                    code_node=report.code_node,
                    status=report.status,
                    reason=report.reason,
                    confidence=report.confidence
                )
                violation_edges.append(edge)
        
        return violation_edges
    
    def prioritize_violations(self, reports: List[ViolationReport]) -> List[ViolationReport]:
        """
        Prioritize violations by severity and impact.
        
        Args:
            reports: List of violation reports to prioritize
            
        Returns:
            Sorted list of violation reports (highest priority first)
        """
        # Filter to only violations
        violations = [r for r in reports if r.status == 'VIOLATES']
        
        # Sort by severity (lower number = higher priority) and confidence (higher = more priority)
        violations.sort(key=lambda r: (r.severity, -r.confidence))
        
        return violations
    
    def update_violation_status(self, graph: nx.DiGraph, 
                              changed_nodes: Set[str]) -> List[ViolationReport]:
        """
        Update violation status for nodes that have changed.
        
        Args:
            graph: Updated graph
            changed_nodes: Set of nodes that have been modified
            
        Returns:
            List of updated violation reports
        """
        updated_reports = []
        
        # Find requirements that might be affected by the changed nodes
        affected_requirements = self._find_affected_requirements(graph, changed_nodes)
        
        # Re-analyze affected requirements
        for req_node_name, req_data in affected_requirements:
            related_code_nodes = self._find_related_code_nodes(req_node_name, graph)
            
            for code_node in related_code_nodes:
                if code_node in changed_nodes:
                    report = self._analyze_single_requirement_satisfaction(
                        req_data, code_node, graph
                    )
                    updated_reports.append(report)
        
        return updated_reports
    
    def _initialize_violation_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns for detecting common violations."""
        return {
            'missing_validation': {
                'keywords': ['validate', 'check', 'verify', 'input'],
                'required_functions': ['validate', 'check', 'verify'],
                'severity': 2
            },
            'missing_error_handling': {
                'keywords': ['error', 'exception', 'handle', 'graceful'],
                'required_patterns': ['try', 'except', 'catch'],
                'severity': 2
            },
            'missing_authentication': {
                'keywords': ['authenticate', 'login', 'auth', 'credential'],
                'required_functions': ['authenticate', 'login', 'verify'],
                'severity': 1
            },
            'missing_crud_operations': {
                'keywords': ['create', 'read', 'update', 'delete', 'add', 'remove'],
                'required_functions': ['create', 'get', 'update', 'delete'],
                'severity': 3
            },
            'performance_issues': {
                'keywords': ['fast', 'quick', 'performance', 'efficient'],
                'anti_patterns': ['nested_loops', 'inefficient_search'],
                'severity': 4
            }
        }
    
    def _find_related_code_nodes(self, requirement_node: str, graph: nx.DiGraph) -> List[str]:
        """Find code nodes related to a requirement."""
        related_nodes = []
        
        # Find nodes connected by MAPS_TO edges
        for source, target, edge_data in graph.edges(requirement_node, data=True):
            if edge_data.get('type') == 'MAPS_TO':
                related_nodes.append(target)
        
        # If no direct mappings, find nodes with similar names/keywords
        if not related_nodes:
            req_data = graph.nodes[requirement_node].get('data')
            if req_data:
                related_nodes = self._find_nodes_by_keywords(req_data.text, graph)
        
        return related_nodes
    
    def _find_nodes_by_keywords(self, requirement_text: str, graph: nx.DiGraph) -> List[str]:
        """Find nodes that match keywords in the requirement text."""
        keywords = requirement_text.lower().split()
        matching_nodes = []
        
        for node_name, node_data in graph.nodes(data=True):
            if node_data.get('type') in ['function', 'class']:
                node_name_lower = node_name.lower()
                
                # Check if any keyword appears in the node name
                for keyword in keywords:
                    if len(keyword) > 3 and keyword in node_name_lower:
                        matching_nodes.append(node_name)
                        break
        
        return matching_nodes
    
    def _analyze_single_requirement_satisfaction(self, requirement: RequirementNode,
                                               code_node: str,
                                               graph: nx.DiGraph) -> ViolationReport:
        """Analyze satisfaction of a single requirement against a code node."""
        
        # Get code node data
        code_node_data = graph.nodes.get(code_node, {})
        
        # Determine violation type based on requirement text
        violation_type = self._classify_requirement_type(requirement.text)
        
        # Check for satisfaction based on violation type
        status, reason, confidence = self._check_requirement_satisfaction(
            requirement, code_node, code_node_data, violation_type, graph
        )
        
        # Determine severity
        severity = self._determine_violation_severity(requirement, violation_type)
        
        return ViolationReport(
            requirement_id=requirement.id,
            code_node=code_node,
            status=status,
            reason=reason,
            confidence=confidence,
            severity=severity
        )
    
    def _classify_requirement_type(self, requirement_text: str) -> str:
        """Classify the type of requirement based on text analysis."""
        text_lower = requirement_text.lower()
        
        for pattern_name, pattern_data in self.violation_patterns.items():
            keywords = pattern_data.get('keywords', [])
            if any(keyword in text_lower for keyword in keywords):
                return pattern_name
        
        return 'general'
    
    def _check_requirement_satisfaction(self, requirement: RequirementNode,
                                      code_node: str,
                                      code_node_data: dict,
                                      violation_type: str,
                                      graph: nx.DiGraph) -> Tuple[str, str, float]:
        """Check if a requirement is satisfied by a code node."""
        
        if violation_type == 'missing_validation':
            return self._check_validation_requirement(requirement, code_node, graph)
        elif violation_type == 'missing_authentication':
            return self._check_authentication_requirement(requirement, code_node, graph)
        elif violation_type == 'missing_crud_operations':
            return self._check_crud_requirement(requirement, code_node, graph)
        elif violation_type == 'missing_error_handling':
            return self._check_error_handling_requirement(requirement, code_node, graph)
        else:
            return self._check_general_requirement(requirement, code_node, code_node_data)
    
    def _check_validation_requirement(self, requirement: RequirementNode,
                                    code_node: str, graph: nx.DiGraph) -> Tuple[str, str, float]:
        """Check validation-related requirements."""
        # Look for validation functions in the graph
        validation_functions = []
        for node_name, node_data in graph.nodes(data=True):
            if (node_data.get('type') == 'function' and 
                any(keyword in node_name.lower() for keyword in ['validate', 'check', 'verify'])):
                validation_functions.append(node_name)
        
        if validation_functions:
            # Check if the code node is connected to validation functions
            connected_to_validation = False
            for _, target, edge_data in graph.edges(code_node, data=True):
                if target in validation_functions:
                    connected_to_validation = True
                    break
            
            if connected_to_validation:
                return "SATISFIES", f"Connected to validation function", 0.8
            else:
                return "VIOLATES", f"No connection to validation functions: {validation_functions}", 0.7
        else:
            return "VIOLATES", "No validation functions found in codebase", 0.9
    
    def _check_authentication_requirement(self, requirement: RequirementNode,
                                        code_node: str, graph: nx.DiGraph) -> Tuple[str, str, float]:
        """Check authentication-related requirements."""
        auth_functions = []
        for node_name, node_data in graph.nodes(data=True):
            if (node_data.get('type') == 'function' and 
                any(keyword in node_name.lower() for keyword in ['auth', 'login', 'credential'])):
                auth_functions.append(node_name)
        
        if auth_functions:
            return "SATISFIES", f"Authentication functions found: {auth_functions}", 0.8
        else:
            return "VIOLATES", "No authentication functions found", 0.9
    
    def _check_crud_requirement(self, requirement: RequirementNode,
                              code_node: str, graph: nx.DiGraph) -> Tuple[str, str, float]:
        """Check CRUD operation requirements."""
        crud_operations = {'create': False, 'read': False, 'update': False, 'delete': False}
        
        for node_name, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'function':
                name_lower = node_name.lower()
                if any(op in name_lower for op in ['create', 'add', 'new']):
                    crud_operations['create'] = True
                if any(op in name_lower for op in ['get', 'read', 'fetch', 'find']):
                    crud_operations['read'] = True
                if any(op in name_lower for op in ['update', 'modify', 'edit', 'change']):
                    crud_operations['update'] = True
                if any(op in name_lower for op in ['delete', 'remove', 'destroy']):
                    crud_operations['delete'] = True
        
        missing_operations = [op for op, exists in crud_operations.items() if not exists]
        
        if not missing_operations:
            return "SATISFIES", "All CRUD operations implemented", 0.9
        elif len(missing_operations) <= 2:
            return "VIOLATES", f"Missing CRUD operations: {missing_operations}", 0.7
        else:
            return "VIOLATES", f"Most CRUD operations missing: {missing_operations}", 0.9
    
    def _check_error_handling_requirement(self, requirement: RequirementNode,
                                        code_node: str, graph: nx.DiGraph) -> Tuple[str, str, float]:
        """Check error handling requirements."""
        # This is a simplified check - in a real implementation, 
        # we would analyze the AST for try/except blocks
        error_keywords = ['error', 'exception', 'handle', 'graceful']
        req_text_lower = requirement.text.lower()
        
        if any(keyword in req_text_lower for keyword in error_keywords):
            # Look for error handling patterns in function names
            error_handling_functions = []
            for node_name, node_data in graph.nodes(data=True):
                if (node_data.get('type') == 'function' and 
                    any(keyword in node_name.lower() for keyword in ['handle', 'error', 'exception'])):
                    error_handling_functions.append(node_name)
            
            if error_handling_functions:
                return "SATISFIES", f"Error handling functions found: {error_handling_functions}", 0.7
            else:
                return "VIOLATES", "No explicit error handling functions found", 0.8
        
        return "UNKNOWN", "Cannot determine error handling satisfaction", 0.3
    
    def _check_general_requirement(self, requirement: RequirementNode,
                                 code_node: str, code_node_data: dict) -> Tuple[str, str, float]:
        """Check general requirements using basic heuristics."""
        # Basic keyword matching
        req_keywords = set(requirement.text.lower().split())
        node_keywords = set(code_node.lower().split('_') + code_node.lower().split('.'))
        
        common_keywords = req_keywords.intersection(node_keywords)
        
        if common_keywords:
            confidence = min(0.8, len(common_keywords) * 0.2)
            return "SATISFIES", f"Keyword match: {common_keywords}", confidence
        else:
            return "UNKNOWN", "No clear relationship found", 0.2
    
    def _determine_violation_severity(self, requirement: RequirementNode, 
                                    violation_type: str) -> int:
        """Determine the severity of a violation."""
        # Use requirement priority as base severity
        base_severity = requirement.priority
        
        # Adjust based on violation type
        if violation_type in self.violation_patterns:
            pattern_severity = self.violation_patterns[violation_type].get('severity', 3)
            # Take the more severe (lower number) of the two
            return min(base_severity, pattern_severity)
        
        return base_severity
    
    def _find_affected_requirements(self, graph: nx.DiGraph, 
                                  changed_nodes: Set[str]) -> List[Tuple[str, RequirementNode]]:
        """Find requirements that might be affected by changed nodes."""
        affected_requirements = []
        
        for node_name, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'requirement':
                req_data = node_data.get('data')
                if req_data:
                    # Check if this requirement is connected to any changed nodes
                    for _, target, edge_data in graph.edges(node_name, data=True):
                        if target in changed_nodes and edge_data.get('type') == 'MAPS_TO':
                            affected_requirements.append((node_name, req_data))
                            break
        
        return affected_requirements