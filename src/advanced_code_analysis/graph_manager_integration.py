"""
Integration interface between Advanced Code Analysis and Enhanced GraphManager.

This module provides the integration layer that allows the Advanced Code Analysis
system to work seamlessly with the existing Enhanced GraphManager, enhancing
rather than replacing its functionality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx

from .models import (
    AnalysisResult, BugType, ContextWindow, ReasoningChain,
    VerificationResult
)
from .config import AdvancedAnalysisConfig
from .advanced_code_analyzer import AdvancedCodeAnalyzer, AnalysisRequest, ComprehensiveAnalysisResult

# Import Enhanced GraphManager components
try:
    from ..enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
    from ..enhanced_graph_manager.config import EnhancedGraphManagerConfig
    from ..enhanced_graph_manager.models import (
        FunctionNode, ClassNode, VariableNode, RequirementNode,
        CallEdge, DependencyEdge, ViolationEdge
    )
    ENHANCED_GRAPH_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_GRAPH_MANAGER_AVAILABLE = False
    # Create placeholder classes for type hints
    class EnhancedGraphManager:
        pass
    class EnhancedGraphManagerConfig:
        pass


logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for Advanced Analysis - Enhanced GraphManager integration."""
    
    # Integration mode settings
    enable_graph_context_enhancement: bool = True
    enable_semantic_requirement_mapping: bool = True
    enable_dependency_aware_analysis: bool = True
    enable_violation_guided_analysis: bool = True
    
    # Analysis strategy selection
    use_graph_for_bug_classification: bool = True
    use_graph_for_concept_mapping: bool = True
    use_graph_for_pattern_matching: bool = True
    
    # Performance settings
    max_graph_nodes_for_analysis: int = 1000
    max_dependency_depth_for_context: int = 5
    enable_parallel_graph_analysis: bool = False
    
    # Fallback settings
    fallback_to_basic_analysis: bool = True
    fallback_confidence_threshold: float = 0.3
    
    # Output integration
    merge_analysis_results: bool = True
    preserve_graph_structure: bool = True
    add_analysis_nodes_to_graph: bool = False


@dataclass
class IntegratedAnalysisResult:
    """Analysis result that combines Advanced Analysis with Graph Manager data."""
    
    # Advanced analysis results
    advanced_analysis: ComprehensiveAnalysisResult
    
    # Graph manager results
    graph_statistics: Optional[Dict[str, Any]] = None
    dependency_analysis: Optional[Dict[str, Any]] = None
    violation_report: Optional[Dict[str, Any]] = None
    
    # Integration metadata
    integration_method: str = "hybrid"
    graph_nodes_analyzed: int = 0
    graph_enhanced_confidence: float = 0.0
    
    # Combined insights
    graph_supported_findings: List[str] = field(default_factory=list)
    graph_contradicted_findings: List[str] = field(default_factory=list)
    additional_context_from_graph: List[str] = field(default_factory=list)


class GraphManagerIntegration:
    """
    Integration layer between Advanced Code Analysis and Enhanced GraphManager.
    
    This class provides methods to:
    1. Use graph structure to enhance context for LLM analysis
    2. Map semantic requirements to graph nodes
    3. Use dependency information for better concept mapping
    4. Validate LLM findings against graph structure
    5. Provide unified analysis results
    """
    
    def __init__(self, 
                 advanced_analyzer: AdvancedCodeAnalyzer,
                 graph_manager: Optional[EnhancedGraphManager] = None,
                 integration_config: Optional[IntegrationConfig] = None):
        """Initialize the integration layer."""
        
        if not ENHANCED_GRAPH_MANAGER_AVAILABLE:
            raise ImportError("Enhanced GraphManager is not available for integration")
        
        self.advanced_analyzer = advanced_analyzer
        self.graph_manager = graph_manager or EnhancedGraphManager()
        self.integration_config = integration_config or IntegrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cache for graph analysis results
        self._graph_cache: Dict[str, Any] = {}
        self._last_code_hash: Optional[str] = None
        
        self.logger.info("GraphManagerIntegration initialized")
    
    async def analyze_with_graph_enhancement(self,
                                           issue_text: str,
                                           target_files: List[str],
                                           requirements_text: Optional[str] = None,
                                           **options) -> IntegratedAnalysisResult:
        """
        Perform integrated analysis using both Advanced Analysis and Graph Manager.
        
        Args:
            issue_text: Description of the problem to analyze
            target_files: List of files to analyze
            requirements_text: Optional requirements text for semantic injection
            **options: Additional analysis options
            
        Returns:
            IntegratedAnalysisResult with combined insights
        """
        self.logger.info("Starting integrated analysis")
        
        try:
            # Step 1: Prepare graph analysis
            graph_results = await self._prepare_graph_analysis(
                target_files, requirements_text
            )
            
            # Step 2: Enhance context using graph information
            enhanced_context = await self._enhance_context_with_graph(
                issue_text, target_files, graph_results
            )
            
            # Step 3: Perform advanced analysis with graph-enhanced context
            advanced_result = await self.advanced_analyzer.analyze(
                issue_text=issue_text,
                target_files=target_files,
                code_context=enhanced_context,
                **options
            )
            
            # Step 4: Validate and cross-reference with graph findings
            integrated_result = await self._integrate_analysis_results(
                advanced_result, graph_results
            )
            
            self.logger.info("Integrated analysis completed successfully")
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"Integrated analysis failed: {e}")
            
            # Fallback to basic advanced analysis if integration fails
            if self.integration_config.fallback_to_basic_analysis:
                self.logger.info("Falling back to basic advanced analysis")
                advanced_result = await self.advanced_analyzer.analyze(
                    issue_text=issue_text,
                    target_files=target_files,
                    **options
                )
                
                return IntegratedAnalysisResult(
                    advanced_analysis=advanced_result,
                    integration_method="fallback"
                )
            else:
                raise
    
    async def _prepare_graph_analysis(self,
                                    target_files: List[str],
                                    requirements_text: Optional[str] = None) -> Dict[str, Any]:
        """Prepare graph analysis by extracting structure and injecting semantics."""
        
        try:
            # Read code from target files
            code_content = ""
            for file_path in target_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content += f"\n# File: {file_path}\n"
                        code_content += f.read()
                        code_content += "\n"
                except (IOError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Could not read file {file_path}: {e}")
            
            if not code_content.strip():
                self.logger.warning("No code content available for graph analysis")
                return {}
            
            # Check if we need to update graph analysis
            code_hash = str(hash(code_content))
            if (code_hash == self._last_code_hash and 
                code_hash in self._graph_cache):
                self.logger.debug("Using cached graph analysis results")
                return self._graph_cache[code_hash]
            
            # Extract structure
            self.logger.debug("Extracting code structure")
            graph = self.graph_manager.extract_structure(code_content)
            
            # Inject semantics if requirements provided
            if requirements_text and self.integration_config.enable_semantic_requirement_mapping:
                self.logger.debug("Injecting semantic requirements")
                graph = self.graph_manager.inject_semantics(requirements_text)
            
            # Trace dependencies
            if self.integration_config.enable_dependency_aware_analysis:
                self.logger.debug("Tracing dependencies")
                dependencies = self.graph_manager.trace_dependencies()
            else:
                dependencies = {}
            
            # Flag violations
            violations = []
            if (requirements_text and 
                self.integration_config.enable_violation_guided_analysis):
                self.logger.debug("Flagging violations")
                violations = self.graph_manager.flag_violations()
            
            # Compile results
            results = {
                'graph': graph,
                'dependencies': dependencies,
                'violations': violations,
                'graph_statistics': self.graph_manager.get_graph_statistics(),
                'dependency_analysis': self.graph_manager.get_dependency_analysis(),
                'violation_report': self.graph_manager.get_violation_report()
            }
            
            # Cache results
            self._graph_cache[code_hash] = results
            self._last_code_hash = code_hash
            
            return results
            
        except Exception as e:
            self.logger.error(f"Graph analysis preparation failed: {e}")
            return {}
    
    async def _enhance_context_with_graph(self,
                                        issue_text: str,
                                        target_files: List[str],
                                        graph_results: Dict[str, Any]) -> str:
        """Enhance context using graph structure and dependency information."""
        
        if not graph_results or not self.integration_config.enable_graph_context_enhancement:
            return ""
        
        try:
            enhanced_context_parts = []
            
            # Add graph statistics context
            if 'graph_statistics' in graph_results:
                stats = graph_results['graph_statistics']
                enhanced_context_parts.append(
                    f"Code Structure Overview:\n"
                    f"- Total nodes: {stats.get('total_nodes', 0)}\n"
                    f"- Total edges: {stats.get('total_edges', 0)}\n"
                    f"- Node types: {stats.get('node_types', {})}\n"
                )
            
            # Add dependency context
            if 'dependency_analysis' in graph_results:
                dep_analysis = graph_results['dependency_analysis']
                most_dependent = dep_analysis.get('most_dependent_nodes', [])
                if most_dependent:
                    enhanced_context_parts.append(
                        f"Key Dependencies:\n"
                        + "\n".join([
                            f"- {node['node']}: {node['dependency_count']} dependencies"
                            for node in most_dependent[:5]
                        ])
                    )
            
            # Add violation context if available
            if 'violation_report' in graph_results:
                violation_report = graph_results['violation_report']
                if violation_report.get('total_violations', 0) > 0:
                    enhanced_context_parts.append(
                        f"Potential Issues Detected:\n"
                        f"- {violation_report['total_violations']} violations found\n"
                        f"- {violation_report['total_satisfies']} requirements satisfied\n"
                    )
            
            # Add specific node information relevant to the issue
            graph = graph_results.get('graph')
            if graph and hasattr(graph, 'nodes'):
                relevant_nodes = self._find_relevant_nodes(issue_text, graph)
                if relevant_nodes:
                    enhanced_context_parts.append(
                        f"Relevant Code Elements:\n"
                        + "\n".join([f"- {node}" for node in relevant_nodes[:10]])
                    )
            
            return "\n\n".join(enhanced_context_parts)
            
        except Exception as e:
            self.logger.error(f"Context enhancement failed: {e}")
            return ""
    
    def _find_relevant_nodes(self, issue_text: str, graph: nx.DiGraph) -> List[str]:
        """Find graph nodes relevant to the issue description."""
        
        relevant_nodes = []
        issue_lower = issue_text.lower()
        
        # Extract keywords from issue text
        keywords = []
        for word in issue_lower.split():
            # Clean word and add if it looks like a code identifier
            clean_word = word.strip('.,!?()[]{}":;')
            if len(clean_word) > 2 and ('_' in clean_word or clean_word.isalnum()):
                keywords.append(clean_word)
        
        # Find nodes that match keywords
        for node_name in graph.nodes():
            node_name_lower = node_name.lower()
            
            # Direct match
            if any(keyword in node_name_lower for keyword in keywords):
                relevant_nodes.append(node_name)
            
            # Partial match for function/class names
            elif any(keyword in node_name_lower.replace('_', '') for keyword in keywords):
                relevant_nodes.append(node_name)
        
        return relevant_nodes
    
    async def _integrate_analysis_results(self,
                                        advanced_result: ComprehensiveAnalysisResult,
                                        graph_results: Dict[str, Any]) -> IntegratedAnalysisResult:
        """Integrate advanced analysis results with graph manager findings."""
        
        try:
            # Create integrated result
            integrated_result = IntegratedAnalysisResult(
                advanced_analysis=advanced_result,
                graph_statistics=graph_results.get('graph_statistics'),
                dependency_analysis=graph_results.get('dependency_analysis'),
                violation_report=graph_results.get('violation_report'),
                integration_method="hybrid",
                graph_nodes_analyzed=graph_results.get('graph_statistics', {}).get('total_nodes', 0)
            )
            
            # Cross-validate findings
            await self._cross_validate_findings(integrated_result, graph_results)
            
            # Enhance confidence based on graph support
            await self._calculate_graph_enhanced_confidence(integrated_result, graph_results)
            
            # Add additional context from graph
            await self._extract_additional_graph_context(integrated_result, graph_results)
            
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"Result integration failed: {e}")
            # Return basic integration on failure
            return IntegratedAnalysisResult(
                advanced_analysis=advanced_result,
                integration_method="basic"
            )
    
    async def _cross_validate_findings(self,
                                     integrated_result: IntegratedAnalysisResult,
                                     graph_results: Dict[str, Any]) -> None:
        """Cross-validate advanced analysis findings with graph structure."""
        
        try:
            primary_analysis = integrated_result.advanced_analysis.primary_analysis
            graph = graph_results.get('graph')
            
            if not graph or not hasattr(graph, 'nodes'):
                return
            
            # Check if identified bug location exists in graph
            bug_location = primary_analysis.bug_location
            if bug_location:
                # Look for exact or partial matches in graph nodes
                matching_nodes = [
                    node for node in graph.nodes()
                    if bug_location.lower() in node.lower() or node.lower() in bug_location.lower()
                ]
                
                if matching_nodes:
                    integrated_result.graph_supported_findings.append(
                        f"Bug location '{bug_location}' confirmed in code structure"
                    )
                else:
                    integrated_result.graph_contradicted_findings.append(
                        f"Bug location '{bug_location}' not found in code structure"
                    )
            
            # Check if suggested fix targets are valid
            fix_suggestion = primary_analysis.fix_suggestion
            if fix_suggestion:
                # Extract potential function/class names from fix suggestion
                potential_targets = []
                for word in fix_suggestion.split():
                    clean_word = word.strip('.,!?()[]{}":;')
                    if '_' in clean_word or (clean_word.isalnum() and len(clean_word) > 2):
                        potential_targets.append(clean_word)
                
                for target in potential_targets:
                    matching_nodes = [
                        node for node in graph.nodes()
                        if target.lower() in node.lower()
                    ]
                    if matching_nodes:
                        integrated_result.graph_supported_findings.append(
                            f"Fix target '{target}' found in code structure"
                        )
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
    
    async def _calculate_graph_enhanced_confidence(self,
                                                 integrated_result: IntegratedAnalysisResult,
                                                 graph_results: Dict[str, Any]) -> None:
        """Calculate enhanced confidence based on graph support."""
        
        try:
            base_confidence = integrated_result.advanced_analysis.primary_analysis.confidence
            
            # Start with base confidence
            enhanced_confidence = base_confidence
            
            # Boost confidence for graph-supported findings
            support_boost = len(integrated_result.graph_supported_findings) * 0.05
            contradiction_penalty = len(integrated_result.graph_contradicted_findings) * 0.1
            
            enhanced_confidence = min(1.0, enhanced_confidence + support_boost - contradiction_penalty)
            
            # Additional boost if violations align with findings
            violation_report = graph_results.get('violation_report', {})
            if violation_report.get('total_violations', 0) > 0:
                # If we found violations and advanced analysis found issues, boost confidence
                enhanced_confidence = min(1.0, enhanced_confidence + 0.1)
            
            integrated_result.graph_enhanced_confidence = enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            integrated_result.graph_enhanced_confidence = integrated_result.advanced_analysis.primary_analysis.confidence
    
    async def _extract_additional_graph_context(self,
                                              integrated_result: IntegratedAnalysisResult,
                                              graph_results: Dict[str, Any]) -> None:
        """Extract additional context insights from graph analysis."""
        
        try:
            # Add dependency insights
            dependency_analysis = graph_results.get('dependency_analysis', {})
            most_dependent = dependency_analysis.get('most_dependent_nodes', [])
            
            if most_dependent:
                integrated_result.additional_context_from_graph.append(
                    f"High-dependency components that may be affected: "
                    + ", ".join([node['node'] for node in most_dependent[:3]])
                )
            
            # Add violation insights
            violation_report = graph_results.get('violation_report', {})
            prioritized_violations = violation_report.get('prioritized_violations', [])
            
            if prioritized_violations:
                high_severity_violations = [
                    v for v in prioritized_violations[:5]
                    if v.get('severity') in ['high', 'critical']
                ]
                
                if high_severity_violations:
                    integrated_result.additional_context_from_graph.append(
                        f"High-severity requirement violations detected in: "
                        + ", ".join([v['code_node'] for v in high_severity_violations])
                    )
            
            # Add structural insights
            graph_stats = graph_results.get('graph_statistics', {})
            node_types = graph_stats.get('node_types', {})
            
            if node_types:
                integrated_result.additional_context_from_graph.append(
                    f"Code structure: {node_types.get('function', 0)} functions, "
                    f"{node_types.get('class', 0)} classes, "
                    f"{node_types.get('variable', 0)} variables"
                )
            
        except Exception as e:
            self.logger.error(f"Additional context extraction failed: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of the integration system."""
        
        return {
            'enhanced_graph_manager_available': ENHANCED_GRAPH_MANAGER_AVAILABLE,
            'integration_config': {
                'graph_context_enhancement': self.integration_config.enable_graph_context_enhancement,
                'semantic_requirement_mapping': self.integration_config.enable_semantic_requirement_mapping,
                'dependency_aware_analysis': self.integration_config.enable_dependency_aware_analysis,
                'violation_guided_analysis': self.integration_config.enable_violation_guided_analysis,
            },
            'cache_status': {
                'cached_analyses': len(self._graph_cache),
                'last_code_hash': self._last_code_hash is not None,
            },
            'graph_manager_status': self.graph_manager.health_check() if self.graph_manager else None,
            'advanced_analyzer_status': self.advanced_analyzer.get_performance_stats()
        }
    
    async def configure_integration(self, **config_updates) -> None:
        """Update integration configuration."""
        
        for key, value in config_updates.items():
            if hasattr(self.integration_config, key):
                setattr(self.integration_config, key, value)
                self.logger.info(f"Updated integration config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown integration config key: {key}")
        
        # Clear cache when configuration changes
        self._graph_cache.clear()
        self._last_code_hash = None
    
    def clear_cache(self) -> None:
        """Clear the integration cache."""
        self._graph_cache.clear()
        self._last_code_hash = None
        self.logger.info("Integration cache cleared")
    
    async def validate_integration(self) -> List[str]:
        """Validate the integration setup and return any issues."""
        
        issues = []
        
        # Check Enhanced GraphManager availability
        if not ENHANCED_GRAPH_MANAGER_AVAILABLE:
            issues.append("Enhanced GraphManager is not available")
        
        # Check graph manager health
        if self.graph_manager:
            try:
                health = self.graph_manager.health_check()
                if health.get('status') != 'healthy':
                    issues.append(f"Graph manager unhealthy: {health.get('error', 'unknown')}")
            except Exception as e:
                issues.append(f"Graph manager health check failed: {e}")
        
        # Check advanced analyzer
        try:
            analyzer_issues = self.advanced_analyzer.validate_configuration()
            issues.extend(analyzer_issues)
        except Exception as e:
            issues.append(f"Advanced analyzer validation failed: {e}")
        
        # Check LLM connection
        try:
            connection_ok = await self.advanced_analyzer.test_connection()
            if not connection_ok:
                issues.append("LLM connection test failed")
        except Exception as e:
            issues.append(f"LLM connection test error: {e}")
        
        return issues