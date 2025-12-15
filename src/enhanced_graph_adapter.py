"""
Enhanced Graph Adapter - Unified Interface for Advanced Analysis and Graph Management

This module provides a unified interface that combines the Advanced Code Analysis
system with the Enhanced GraphManager, allowing users to choose their analysis
strategy and seamlessly switch between different approaches.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import networkx as nx

# Import Advanced Code Analysis components
from src.advanced_code_analysis.advanced_code_analyzer import AdvancedCodeAnalyzer
from src.advanced_code_analysis.config import AdvancedAnalysisConfig
from src.advanced_code_analysis.graph_manager_integration import (
    GraphManagerIntegration, IntegratedAnalysisResult, IntegrationConfig
)

# Import Enhanced GraphManager components
try:
    from src.enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
    from src.enhanced_graph_manager.config import EnhancedGraphManagerConfig
    ENHANCED_GRAPH_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_GRAPH_MANAGER_AVAILABLE = False
    EnhancedGraphManager = None
    EnhancedGraphManagerConfig = None


logger = logging.getLogger(__name__)


class AnalysisStrategy(Enum):
    """Available analysis strategies."""
    ADVANCED_ONLY = "advanced_only"
    GRAPH_ONLY = "graph_only"
    INTEGRATED = "integrated"
    AUTO_SELECT = "auto_select"


@dataclass
class AnalysisOptions:
    """Options for configuring analysis behavior."""
    strategy: AnalysisStrategy = AnalysisStrategy.AUTO_SELECT
    include_requirements: bool = True
    max_context_tokens: int = 8000
    confidence_threshold: float = 0.7
    enable_caching: bool = True
    parallel_processing: bool = False
    debug_mode: bool = False


@dataclass
class UnifiedAnalysisResult:
    """Unified result that can contain results from either or both systems."""
    
    # Analysis metadata
    strategy_used: AnalysisStrategy
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None
    
    # Advanced analysis results (if available)
    advanced_result: Optional[Any] = None
    
    # Graph manager results (if available)
    graph_statistics: Optional[Dict[str, Any]] = None
    dependency_analysis: Optional[Dict[str, Any]] = None
    violation_report: Optional[Dict[str, Any]] = None
    
    # Integrated results (if available)
    integrated_result: Optional[IntegratedAnalysisResult] = None
    
    # Unified output
    primary_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    def get_best_recommendation(self) -> Optional[str]:
        """Get the highest confidence recommendation."""
        if not self.recommendations:
            return None
        return self.recommendations[0]  # Assuming sorted by confidence
    
    def has_advanced_analysis(self) -> bool:
        """Check if advanced analysis results are available."""
        return self.advanced_result is not None or self.integrated_result is not None
    
    def has_graph_analysis(self) -> bool:
        """Check if graph analysis results are available."""
        return (self.graph_statistics is not None or 
                self.dependency_analysis is not None or 
                self.violation_report is not None)


class EnhancedGraphAdapter:
    """
    Unified adapter that provides a single interface for both Advanced Code Analysis
    and Enhanced GraphManager systems, with intelligent strategy selection.
    """
    
    def __init__(self, 
                 advanced_config: Optional[AdvancedAnalysisConfig] = None,
                 graph_config: Optional[EnhancedGraphManagerConfig] = None,
                 integration_config: Optional[IntegrationConfig] = None):
        """Initialize the Enhanced Graph Adapter."""
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize configurations
        self.advanced_config = advanced_config or AdvancedAnalysisConfig()
        self.graph_config = graph_config or (EnhancedGraphManagerConfig() if ENHANCED_GRAPH_MANAGER_AVAILABLE else None)
        self.integration_config = integration_config or IntegrationConfig()
        
        # Initialize systems
        self.advanced_analyzer: Optional[AdvancedCodeAnalyzer] = None
        self.graph_manager: Optional[EnhancedGraphManager] = None
        self.integration_layer: Optional[GraphManagerIntegration] = None
        
        # Performance tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[AnalysisStrategy, Dict[str, float]] = {}
        
        # Initialize systems
        self._initialize_systems()
        
        self.logger.info("EnhancedGraphAdapter initialized")
    
    def _initialize_systems(self) -> None:
        """Initialize available analysis systems."""
        
        # Initialize Advanced Code Analyzer
        try:
            self.advanced_analyzer = AdvancedCodeAnalyzer(self.advanced_config)
            self.logger.info("Advanced Code Analyzer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Advanced Code Analyzer: {e}")
        
        # Initialize Enhanced GraphManager if available
        if ENHANCED_GRAPH_MANAGER_AVAILABLE and self.graph_config:
            try:
                self.graph_manager = EnhancedGraphManager(self.graph_config)
                self.logger.info("Enhanced GraphManager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Enhanced GraphManager: {e}")
        
        # Initialize integration layer if both systems are available
        if (self.advanced_analyzer and self.graph_manager and 
            self.advanced_config.integrate_with_enhanced_graph_manager):
            try:
                self.integration_layer = GraphManagerIntegration(
                    self.advanced_analyzer,
                    self.graph_manager,
                    self.integration_config
                )
                self.logger.info("Integration layer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize integration layer: {e}")
    
    async def analyze(self,
                     issue_text: str,
                     target_files: List[str],
                     requirements_text: Optional[str] = None,
                     options: Optional[AnalysisOptions] = None) -> UnifiedAnalysisResult:
        """
        Perform unified analysis using the specified or auto-selected strategy.
        
        Args:
            issue_text: Description of the problem to analyze
            target_files: List of files to analyze
            requirements_text: Optional requirements text
            options: Analysis options and strategy selection
            
        Returns:
            UnifiedAnalysisResult containing results from selected strategy
        """
        
        options = options or AnalysisOptions()
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"Starting unified analysis with strategy: {options.strategy}")
        
        try:
            # Select analysis strategy
            selected_strategy = await self._select_strategy(
                issue_text, target_files, requirements_text, options
            )
            
            # Execute analysis based on selected strategy
            if selected_strategy == AnalysisStrategy.INTEGRATED:
                result = await self._run_integrated_analysis(
                    issue_text, target_files, requirements_text, options
                )
            elif selected_strategy == AnalysisStrategy.ADVANCED_ONLY:
                result = await self._run_advanced_analysis(
                    issue_text, target_files, requirements_text, options
                )
            elif selected_strategy == AnalysisStrategy.GRAPH_ONLY:
                result = await self._run_graph_analysis(
                    issue_text, target_files, requirements_text, options
                )
            else:
                raise ValueError(f"Unsupported strategy: {selected_strategy}")
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            result.processing_time = processing_time
            result.strategy_used = selected_strategy
            
            # Update performance tracking
            self._update_performance_tracking(selected_strategy, result)
            
            self.logger.info(f"Analysis completed in {processing_time:.2f}s using {selected_strategy}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            return UnifiedAnalysisResult(
                strategy_used=options.strategy,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _select_strategy(self,
                             issue_text: str,
                             target_files: List[str],
                             requirements_text: Optional[str],
                             options: AnalysisOptions) -> AnalysisStrategy:
        """Select the best analysis strategy based on context and availability."""
        
        # If strategy is explicitly specified and not AUTO_SELECT, use it
        if options.strategy != AnalysisStrategy.AUTO_SELECT:
            # Validate that the requested strategy is available
            if options.strategy == AnalysisStrategy.INTEGRATED and not self.integration_layer:
                self.logger.warning("Integrated analysis not available, falling back to advanced")
                return AnalysisStrategy.ADVANCED_ONLY
            elif options.strategy == AnalysisStrategy.GRAPH_ONLY and not self.graph_manager:
                self.logger.warning("Graph analysis not available, falling back to advanced")
                return AnalysisStrategy.ADVANCED_ONLY
            elif options.strategy == AnalysisStrategy.ADVANCED_ONLY and not self.advanced_analyzer:
                raise ValueError("Advanced analysis not available")
            
            return options.strategy
        
        # Auto-select strategy based on context and system availability
        
        # Prefer integrated analysis if available and requirements are provided
        if (self.integration_layer and requirements_text and 
            len(target_files) <= self.integration_config.max_graph_nodes_for_analysis):
            return AnalysisStrategy.INTEGRATED
        
        # Use advanced analysis for complex issues or when LLM reasoning is needed
        if self.advanced_analyzer:
            # Check if issue seems complex (heuristic)
            complex_indicators = [
                'logic error', 'algorithm', 'performance', 'optimization',
                'bug', 'issue', 'problem', 'error', 'exception'
            ]
            
            if any(indicator in issue_text.lower() for indicator in complex_indicators):
                return AnalysisStrategy.ADVANCED_ONLY
        
        # Use graph analysis for structural analysis or when advanced analysis is not available
        if self.graph_manager and requirements_text:
            return AnalysisStrategy.GRAPH_ONLY
        
        # Default to advanced analysis if available
        if self.advanced_analyzer:
            return AnalysisStrategy.ADVANCED_ONLY
        
        # Fallback to graph analysis
        if self.graph_manager:
            return AnalysisStrategy.GRAPH_ONLY
        
        raise ValueError("No analysis systems available")
    
    async def _run_integrated_analysis(self,
                                     issue_text: str,
                                     target_files: List[str],
                                     requirements_text: Optional[str],
                                     options: AnalysisOptions) -> UnifiedAnalysisResult:
        """Run integrated analysis using both systems."""
        
        if not self.integration_layer:
            raise ValueError("Integration layer not available")
        
        try:
            integrated_result = await self.integration_layer.analyze_with_graph_enhancement(
                issue_text=issue_text,
                target_files=target_files,
                requirements_text=requirements_text
            )
            
            # Extract unified findings
            primary_findings = []
            recommendations = []
            
            if integrated_result.advanced_analysis.primary_analysis:
                analysis = integrated_result.advanced_analysis.primary_analysis
                primary_findings.append(f"Bug Location: {analysis.bug_location}")
                primary_findings.append(f"Root Cause: {analysis.root_cause}")
                recommendations.append(analysis.fix_suggestion)
            
            # Add graph-supported findings
            primary_findings.extend(integrated_result.graph_supported_findings)
            
            # Add additional context from graph
            primary_findings.extend(integrated_result.additional_context_from_graph)
            
            return UnifiedAnalysisResult(
                strategy_used=AnalysisStrategy.INTEGRATED,
                processing_time=0.0,  # Will be set by caller
                integrated_result=integrated_result,
                graph_statistics=integrated_result.graph_statistics,
                dependency_analysis=integrated_result.dependency_analysis,
                violation_report=integrated_result.violation_report,
                primary_findings=primary_findings,
                recommendations=recommendations,
                confidence_score=integrated_result.graph_enhanced_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Integrated analysis failed: {e}")
            raise
    
    async def _run_advanced_analysis(self,
                                   issue_text: str,
                                   target_files: List[str],
                                   requirements_text: Optional[str],
                                   options: AnalysisOptions) -> UnifiedAnalysisResult:
        """Run advanced LLM-based analysis only."""
        
        if not self.advanced_analyzer:
            raise ValueError("Advanced analyzer not available")
        
        try:
            advanced_result = await self.advanced_analyzer.analyze(
                issue_text=issue_text,
                target_files=target_files,
                code_context=requirements_text
            )
            
            # Extract unified findings
            primary_findings = []
            recommendations = []
            
            if advanced_result.primary_analysis:
                analysis = advanced_result.primary_analysis
                primary_findings.append(f"Bug Location: {analysis.bug_location}")
                primary_findings.append(f"Root Cause: {analysis.root_cause}")
                recommendations.append(analysis.fix_suggestion)
                
                # Add supporting evidence
                primary_findings.extend(analysis.supporting_evidence)
            
            return UnifiedAnalysisResult(
                strategy_used=AnalysisStrategy.ADVANCED_ONLY,
                processing_time=0.0,  # Will be set by caller
                advanced_result=advanced_result,
                primary_findings=primary_findings,
                recommendations=recommendations,
                confidence_score=advanced_result.primary_analysis.confidence if advanced_result.primary_analysis else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Advanced analysis failed: {e}")
            raise
    
    async def _run_graph_analysis(self,
                                issue_text: str,
                                target_files: List[str],
                                requirements_text: Optional[str],
                                options: AnalysisOptions) -> UnifiedAnalysisResult:
        """Run graph-based structural analysis only."""
        
        if not self.graph_manager:
            raise ValueError("Graph manager not available")
        
        try:
            # Read code content
            code_content = ""
            for file_path in target_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content += f"\n# File: {file_path}\n"
                        code_content += f.read()
                        code_content += "\n"
                except (IOError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Could not read file {file_path}: {e}")
            
            # Run complete workflow
            if requirements_text:
                workflow_result = self.graph_manager.analyze_complete_workflow(
                    code_content, requirements_text
                )
            else:
                # Run structure extraction only
                self.graph_manager.extract_structure(code_content)
                workflow_result = {
                    'success': True,
                    'graph_statistics': self.graph_manager.get_graph_statistics(),
                    'dependency_analysis': self.graph_manager.get_dependency_analysis(),
                    'violation_report': self.graph_manager.get_violation_report()
                }
            
            # Extract unified findings
            primary_findings = []
            recommendations = []
            
            # Add graph statistics
            if workflow_result.get('graph_statistics'):
                stats = workflow_result['graph_statistics']
                primary_findings.append(
                    f"Code Structure: {stats.get('total_nodes', 0)} nodes, "
                    f"{stats.get('total_edges', 0)} edges"
                )
            
            # Add violation findings
            if workflow_result.get('violation_report'):
                violation_report = workflow_result['violation_report']
                if violation_report.get('total_violations', 0) > 0:
                    primary_findings.append(
                        f"Found {violation_report['total_violations']} requirement violations"
                    )
                    
                    # Add specific violations as recommendations
                    for violation in violation_report.get('prioritized_violations', [])[:3]:
                        recommendations.append(
                            f"Address violation in {violation['code_node']}: {violation['reason']}"
                        )
            
            # Add dependency findings
            if workflow_result.get('dependency_analysis'):
                dep_analysis = workflow_result['dependency_analysis']
                most_dependent = dep_analysis.get('most_dependent_nodes', [])
                if most_dependent:
                    primary_findings.append(
                        f"High-dependency components: {', '.join([n['node'] for n in most_dependent[:3]])}"
                    )
            
            # Calculate confidence based on analysis completeness
            confidence = 0.5  # Base confidence for structural analysis
            if workflow_result.get('violation_report', {}).get('total_violations', 0) > 0:
                confidence += 0.3  # Boost for finding violations
            if workflow_result.get('dependency_analysis', {}).get('total_nodes', 0) > 0:
                confidence += 0.2  # Boost for dependency analysis
            
            return UnifiedAnalysisResult(
                strategy_used=AnalysisStrategy.GRAPH_ONLY,
                processing_time=0.0,  # Will be set by caller
                graph_statistics=workflow_result.get('graph_statistics'),
                dependency_analysis=workflow_result.get('dependency_analysis'),
                violation_report=workflow_result.get('violation_report'),
                primary_findings=primary_findings,
                recommendations=recommendations,
                confidence_score=min(1.0, confidence)
            )
            
        except Exception as e:
            self.logger.error(f"Graph analysis failed: {e}")
            raise
    
    def _update_performance_tracking(self,
                                   strategy: AnalysisStrategy,
                                   result: UnifiedAnalysisResult) -> None:
        """Update performance tracking for strategy selection optimization."""
        
        # Record analysis in history
        self.analysis_history.append({
            'strategy': strategy,
            'success': result.success,
            'confidence': result.confidence_score,
            'processing_time': result.processing_time,
            'findings_count': len(result.primary_findings),
            'recommendations_count': len(result.recommendations)
        })
        
        # Update strategy performance metrics
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_runs': 0,
                'successful_runs': 0,
                'average_confidence': 0.0,
                'average_processing_time': 0.0
            }
        
        metrics = self.strategy_performance[strategy]
        metrics['total_runs'] += 1
        
        if result.success:
            metrics['successful_runs'] += 1
        
        # Update averages
        total_runs = metrics['total_runs']
        metrics['average_confidence'] = (
            (metrics['average_confidence'] * (total_runs - 1) + result.confidence_score) / total_runs
        )
        metrics['average_processing_time'] = (
            (metrics['average_processing_time'] * (total_runs - 1) + result.processing_time) / total_runs
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all systems."""
        
        status = {
            'enhanced_graph_manager_available': ENHANCED_GRAPH_MANAGER_AVAILABLE,
            'systems_initialized': {
                'advanced_analyzer': self.advanced_analyzer is not None,
                'graph_manager': self.graph_manager is not None,
                'integration_layer': self.integration_layer is not None
            },
            'available_strategies': self._get_available_strategies(),
            'performance_metrics': self.strategy_performance,
            'analysis_history_count': len(self.analysis_history)
        }
        
        # Add system-specific status
        if self.advanced_analyzer:
            status['advanced_analyzer_stats'] = self.advanced_analyzer.get_performance_stats()
        
        if self.graph_manager:
            status['graph_manager_health'] = self.graph_manager.health_check()
        
        if self.integration_layer:
            status['integration_status'] = self.integration_layer.get_integration_status()
        
        return status
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available analysis strategies."""
        
        strategies = []
        
        if self.advanced_analyzer:
            strategies.append(AnalysisStrategy.ADVANCED_ONLY.value)
        
        if self.graph_manager:
            strategies.append(AnalysisStrategy.GRAPH_ONLY.value)
        
        if self.integration_layer:
            strategies.append(AnalysisStrategy.INTEGRATED.value)
        
        if len(strategies) > 1:
            strategies.append(AnalysisStrategy.AUTO_SELECT.value)
        
        return strategies
    
    async def configure_systems(self, **config_updates) -> None:
        """Update system configurations."""
        
        # Update advanced analyzer config
        if 'advanced' in config_updates and self.advanced_analyzer:
            for key, value in config_updates['advanced'].items():
                if hasattr(self.advanced_config, key):
                    setattr(self.advanced_config, key, value)
        
        # Update graph manager config
        if 'graph' in config_updates and self.graph_manager:
            for key, value in config_updates['graph'].items():
                if hasattr(self.graph_config, key):
                    setattr(self.graph_config, key, value)
        
        # Update integration config
        if 'integration' in config_updates and self.integration_layer:
            await self.integration_layer.configure_integration(**config_updates['integration'])
        
        self.logger.info("System configurations updated")
    
    async def validate_systems(self) -> Dict[str, List[str]]:
        """Validate all systems and return any issues."""
        
        validation_results = {}
        
        # Validate advanced analyzer
        if self.advanced_analyzer:
            try:
                issues = self.advanced_analyzer.validate_configuration()
                validation_results['advanced_analyzer'] = issues
            except Exception as e:
                validation_results['advanced_analyzer'] = [f"Validation failed: {e}"]
        
        # Validate graph manager
        if self.graph_manager:
            try:
                health = self.graph_manager.health_check()
                if health.get('status') != 'healthy':
                    validation_results['graph_manager'] = [health.get('error', 'Unhealthy status')]
                else:
                    validation_results['graph_manager'] = []
            except Exception as e:
                validation_results['graph_manager'] = [f"Health check failed: {e}"]
        
        # Validate integration layer
        if self.integration_layer:
            try:
                issues = await self.integration_layer.validate_integration()
                validation_results['integration_layer'] = issues
            except Exception as e:
                validation_results['integration_layer'] = [f"Validation failed: {e}"]
        
        return validation_results
    
    def clear_caches(self) -> None:
        """Clear all system caches."""
        
        if self.integration_layer:
            self.integration_layer.clear_cache()
        
        # Clear performance tracking if requested
        self.analysis_history.clear()
        
        self.logger.info("All caches cleared")

    # ------------------------------------------------------------------
    # Compatibility helpers for main_enhanced.py workflow
    # ------------------------------------------------------------------
    def parse_code_structure(self, code: str) -> nx.DiGraph:
        """
        Thin wrapper to mirror the legacy GraphManager interface.
        Uses EnhancedGraphManager to extract the structural graph.
        """
        if not self.graph_manager:
            raise AttributeError("EnhancedGraphManager is not available")
        return self.graph_manager.extract_structure(code)

    def enrich_with_requirements(self,
                                 graph: Optional[nx.DiGraph],
                                 requirements: str,
                                 llm: Any = None) -> nx.DiGraph:
        """
        Mirror legacy enrich_with_requirements API.
        Injects requirements, traces dependencies, and flags violations.
        """
        if not self.graph_manager:
            raise AttributeError("EnhancedGraphManager is not available")

        # If an external graph was passed in, keep it in sync.
        if graph is not None:
            self.graph_manager.graph = graph

        self.graph_manager.inject_semantics(requirements)
        self.graph_manager.trace_dependencies()
        self.graph_manager.flag_violations()
        return self.graph_manager.get_graph()

    def get_analysis_report(self) -> Dict[str, Any]:
        """
        Provide a consolidated analysis report for logging/inspection.
        """
        if not self.graph_manager:
            raise AttributeError("EnhancedGraphManager is not available")

        return {
            "graph_statistics": self.graph_manager.get_graph_statistics(),
            "dependency_analysis": self.graph_manager.get_dependency_analysis(),
            "violation_report": self.graph_manager.get_violation_report(),
            "performance_metrics": self.graph_manager.get_performance_metrics(),
        }
