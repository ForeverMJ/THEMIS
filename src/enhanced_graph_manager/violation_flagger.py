"""Violation flagging engine for Enhanced GraphManager."""

from typing import List, Dict, Set, Optional, Tuple
import networkx as nx
from dataclasses import dataclass, field
import re

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
    blocking: bool = False
    evidence_score: float = 0.0
    evidence_tags: List[str] = field(default_factory=list)


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
        self.generic_reason_patterns = [
            "no explicit error handling functions found",
            "no authentication functions found",
            "no validation functions found",
            "no connection to validation functions",
            "missing crud operations",
            "most crud operations missing",
            "cannot determine",
            "no clear relationship found",
        ]
        self.low_signal_tokens = {
            "set",
            "error",
            "errors",
            "field",
            "fields",
            "model",
            "models",
            "name",
            "names",
            "value",
            "values",
            "type",
            "types",
            "related",
            "self",
            "none",
            "get",
            "add",
            "update",
            "delete",
            "remove",
            "check",
            "create",
            "_",
        }
        
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
                edge_status = "ADVISORY"
                if report.status == "VIOLATES" and report.blocking:
                    edge_status = "VIOLATES"
                edge = ViolationEdge(
                    requirement=report.requirement_id,
                    code_node=report.code_node,
                    status=edge_status,
                    reason=report.reason,
                    confidence=report.confidence,
                    blocking=report.blocking,
                    evidence_score=report.evidence_score,
                    evidence_tags=list(report.evidence_tags or []),
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
        
        # Sort by blocking first, then severity (lower=more important), confidence, and evidence score.
        violations.sort(
            key=lambda r: (
                0 if r.blocking else 1,
                r.severity,
                -r.confidence,
                -float(r.evidence_score or 0.0),
            )
        )
        
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

        blocking = False
        evidence_score = 0.0
        evidence_tags: List[str] = []
        if status in ("VIOLATES", "UNKNOWN"):
            blocking, evidence_score, evidence_tags = self._assess_violation_signal(
                requirement=requirement,
                code_node=code_node,
                reason=reason,
                confidence=confidence,
                graph=graph,
            )
            # Escalate high-signal UNKNOWN findings into actionable violations.
            if status == "UNKNOWN":
                if self._should_escalate_unknown(
                    reason=reason,
                    evidence_score=evidence_score,
                    evidence_tags=evidence_tags,
                ):
                    status = "VIOLATES"
                    reason = f"Potential requirement mismatch (escalated from UNKNOWN): {reason}"
                    confidence = max(confidence, 0.55)
                else:
                    blocking = False
        
        return ViolationReport(
            requirement_id=requirement.id,
            code_node=code_node,
            status=status,
            reason=reason,
            confidence=confidence,
            severity=severity,
            blocking=blocking,
            evidence_score=evidence_score,
            evidence_tags=evidence_tags,
        )

    def _classify_requirement_type(self, requirement_text: str) -> str:
        """Classify the type of requirement based on text analysis."""
        text_lower = requirement_text.lower()

        # Message/hint wording issues are common in SWE tasks and are not equivalent
        # to missing runtime error-handling logic.
        message_markers = [
            "error message",
            "warning message",
            "hint",
            "message text",
            "display message",
            "wording",
        ]
        if any(marker in text_lower for marker in message_markers):
            return "general"
        
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

    def _assess_violation_signal(
        self,
        *,
        requirement: RequirementNode,
        code_node: str,
        reason: str,
        confidence: float,
        graph: nx.DiGraph,
    ) -> Tuple[bool, float, List[str]]:
        """Estimate whether a violation should be blocking or advisory."""
        score = 0.0
        tags: List[str] = []
        mapping_relevance = 0.0

        if confidence >= 0.78:
            score += 1.0
            tags.append("confidence>=0.78")
        elif confidence >= 0.65:
            score += 0.5
            tags.append("confidence>=0.65")

        if self._is_specific_reason(reason):
            score += 1.0
            tags.append("specific_reason")
        else:
            score -= 0.4
            tags.append("generic_reason")
        specific_reason = "specific_reason" in tags

        req_tokens = self._extract_signal_tokens(requirement.text)
        node_tokens = self._extract_signal_tokens(code_node.replace(".", " ").replace("_", " "))
        overlap = req_tokens.intersection(node_tokens)
        if overlap:
            score += 1.0
            tags.append("requirement_symbol_overlap")

        mapping_relevance = self._mapping_relevance(requirement.id, code_node, graph)
        if mapping_relevance >= 0.45:
            score += 1.0
            tags.append(f"mapping_relevance:{mapping_relevance:.2f}")
        elif mapping_relevance > 0:
            score += 0.4
            tags.append(f"mapping_relevance:{mapping_relevance:.2f}")

        if self._is_specific_symbol(code_node):
            score += 0.6
            tags.append("symbol_specific")

        strong_anchor = bool(overlap) or mapping_relevance >= 0.45 or specific_reason
        blocking = (score >= 2.0 and strong_anchor) or (
            score >= 2.2 and self._is_specific_symbol(code_node)
        )
        return blocking, score, tags

    def _should_escalate_unknown(
        self,
        *,
        reason: str,
        evidence_score: float,
        evidence_tags: List[str],
    ) -> bool:
        """Promote UNKNOWN only when the rationale is concrete enough to edit against."""
        if not self._is_specific_reason(reason):
            return False

        has_overlap = "requirement_symbol_overlap" in evidence_tags
        has_relevance = any(tag.startswith("mapping_relevance:") for tag in evidence_tags)
        has_specific_symbol = "symbol_specific" in evidence_tags

        strong_anchor = (has_overlap and has_relevance) or (has_specific_symbol and has_relevance)
        return evidence_score >= 2.4 and strong_anchor

    def _is_specific_reason(self, reason: str) -> bool:
        reason_l = reason.strip().lower()
        if len(reason_l) < 20:
            return False
        return not any(pattern in reason_l for pattern in self.generic_reason_patterns)

    def _extract_signal_tokens(self, text: str) -> Set[str]:
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower()))
        return {token for token in tokens if token not in self.low_signal_tokens}

    def _is_specific_symbol(self, symbol: str) -> bool:
        symbol_l = symbol.strip().lower()
        if len(symbol_l) < 4:
            return False
        if symbol_l in self.low_signal_tokens:
            return False
        return "." in symbol or "_" in symbol or any(ch.isupper() for ch in symbol[1:])

    def _mapping_relevance(self, requirement_id: str, code_node: str, graph: nx.DiGraph) -> float:
        edge_data = graph.get_edge_data(requirement_id, code_node, default=None)
        if not edge_data:
            return 0.0

        context = edge_data.get("context")
        if not context:
            payload = edge_data.get("data")
            context = getattr(payload, "context", "") if payload is not None else ""
        if not isinstance(context, str):
            return 0.0

        m = re.search(r"relevance:([0-9]*\.?[0-9]+)", context)
        if not m:
            return 0.0
        try:
            return float(m.group(1))
        except Exception:
            return 0.0
    
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
