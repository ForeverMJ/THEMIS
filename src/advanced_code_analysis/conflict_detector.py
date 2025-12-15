"""
Conflict Detection and Handling Engine for Advanced Code Analysis.

This module implements the ConflictDetector class that detects inconsistencies
between AST analysis and LLM judgments, provides conflict marking and handling
mechanisms, implements multi-strategy reasoning for conflict resolution, and
adds adaptive information collection for low confidence scenarios.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .models import (
    AnalysisResult, VerificationResult, Conflict, ReasoningChain, 
    ReasoningStep, EvidenceChain, ResolvedAnalysis, ContextWindow,
    BugType, AnalysisStrategy, PromptTemplate
)
from .llm_interface import LLMInterface, LLMResponse
from .enhanced_ast_analyzer import EnhancedASTAnalyzer, ErrorPattern, FunctionCallValidation, DataFlowTrace
from .multi_round_reasoner import MultiRoundReasoner
from .config import AdvancedAnalysisConfig


logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can be detected."""
    LOCATION_MISMATCH = "location_mismatch"
    CONFIDENCE_DISCREPANCY = "confidence_discrepancy"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"
    FIX_INCONSISTENCY = "fix_inconsistency"
    AST_LLM_DISAGREEMENT = "ast_llm_disagreement"
    MULTIPLE_ROOT_CAUSES = "multiple_root_causes"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConflictMarker:
    """Marker for identified conflicts with metadata."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    affected_analyses: List[AnalysisResult] = field(default_factory=list)
    detection_confidence: float = 0.0
    resolution_priority: int = 1  # 1=highest, 5=lowest
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResolutionStrategy:
    """Strategy for resolving specific types of conflicts."""
    strategy_name: str
    applicable_conflicts: List[ConflictType]
    confidence_threshold: float
    max_attempts: int = 3
    requires_additional_context: bool = False
    prompt_template: Optional[PromptTemplate] = None


@dataclass
class AdaptiveContext:
    """Additional context collected adaptively for low confidence scenarios."""
    original_context: ContextWindow
    additional_functions: List[str] = field(default_factory=list)
    extended_dependencies: List[str] = field(default_factory=list)
    similar_code_patterns: List[str] = field(default_factory=list)
    domain_specific_info: Dict[str, Any] = field(default_factory=dict)
    collection_confidence: float = 0.0


class ConflictDetector:
    """
    Conflict detection and handling engine for advanced code analysis.
    
    This class implements:
    1. Detection of inconsistencies between AST and LLM analysis
    2. Conflict marking and categorization
    3. Multi-strategy reasoning for conflict resolution
    4. Adaptive information collection for low confidence scenarios
    """
    
    def __init__(self, llm_interface: LLMInterface, config: AdvancedAnalysisConfig,
                 ast_analyzer: Optional[EnhancedASTAnalyzer] = None,
                 multi_round_reasoner: Optional[MultiRoundReasoner] = None):
        """Initialize the conflict detector."""
        self.llm = llm_interface
        self.config = config
        self.ast_analyzer = ast_analyzer or EnhancedASTAnalyzer(llm_interface)
        self.multi_round_reasoner = multi_round_reasoner or MultiRoundReasoner(llm_interface, config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize resolution strategies
        self.resolution_strategies = self._initialize_resolution_strategies()
        
        # Conflict detection thresholds
        self.confidence_threshold = config.analysis.confidence_threshold
        self.location_similarity_threshold = 0.7
        self.evidence_contradiction_threshold = 0.6
        
        # Adaptive context collection settings
        self.low_confidence_threshold = 0.5
        self.context_expansion_factor = 1.5
    
    def _initialize_resolution_strategies(self) -> Dict[str, ResolutionStrategy]:
        """Initialize resolution strategies for different conflict types."""
        strategies = {}
        
        # Location mismatch strategy
        strategies["location_reconciliation"] = ResolutionStrategy(
            strategy_name="location_reconciliation",
            applicable_conflicts=[ConflictType.LOCATION_MISMATCH],
            confidence_threshold=0.7,
            max_attempts=2,
            requires_additional_context=True,
            prompt_template=PromptTemplate(
                template_id="location_reconciliation",
                content="""
Resolve the following location mismatch between analyses:

Issue: {issue_description}
LLM Analysis Location: {llm_location}
AST Analysis Location: {ast_location}
Code Context: {code_context}

Please determine the most accurate bug location by:
1. Examining the code context around both locations
2. Considering the nature of the bug type
3. Evaluating the evidence supporting each location

Provide your resolution as JSON:
{{
    "resolved_location": "...",
    "reasoning": "...",
    "confidence": 0.0,
    "supporting_evidence": ["...", "..."]
}}
""",
                placeholders=["issue_description", "llm_location", "ast_location", "code_context"]
            )
        )
        
        # Confidence discrepancy strategy
        strategies["confidence_calibration"] = ResolutionStrategy(
            strategy_name="confidence_calibration",
            applicable_conflicts=[ConflictType.CONFIDENCE_DISCREPANCY],
            confidence_threshold=0.6,
            max_attempts=3,
            requires_additional_context=False,
            prompt_template=PromptTemplate(
                template_id="confidence_calibration",
                content="""
Calibrate confidence levels for conflicting analyses:

Issue: {issue_description}
Analysis 1: {analysis1} (Confidence: {confidence1})
Analysis 2: {analysis2} (Confidence: {confidence2})

Consider:
1. Quality and quantity of supporting evidence
2. Consistency of reasoning chains
3. Alignment with known bug patterns

Provide calibrated confidence as JSON:
{{
    "calibrated_confidence": 0.0,
    "primary_analysis": "analysis1|analysis2",
    "reasoning": "...",
    "confidence_factors": ["...", "..."]
}}
""",
                placeholders=["issue_description", "analysis1", "analysis2", "confidence1", "confidence2"]
            )
        )
        
        # AST-LLM disagreement strategy
        strategies["ast_llm_mediation"] = ResolutionStrategy(
            strategy_name="ast_llm_mediation",
            applicable_conflicts=[ConflictType.AST_LLM_DISAGREEMENT],
            confidence_threshold=0.8,
            max_attempts=2,
            requires_additional_context=True,
            prompt_template=PromptTemplate(
                template_id="ast_llm_mediation",
                content="""
Mediate between AST analysis and LLM judgment:

Issue: {issue_description}
LLM Analysis: {llm_analysis}
AST Findings: {ast_findings}
Code Context: {code_context}

The AST analysis and LLM judgment disagree. Please:
1. Evaluate the technical accuracy of AST findings
2. Assess the contextual understanding of LLM analysis
3. Determine which approach is more reliable for this specific case

Provide mediation as JSON:
{{
    "preferred_approach": "ast|llm|hybrid",
    "reasoning": "...",
    "synthesized_analysis": "...",
    "confidence": 0.0,
    "integration_notes": ["...", "..."]
}}
""",
                placeholders=["issue_description", "llm_analysis", "ast_findings", "code_context"]
            )
        )
        
        return strategies
    
    async def detect_conflicts(self, analyses: List[AnalysisResult], 
                             verification_result: Optional[VerificationResult] = None,
                             context: Optional[ContextWindow] = None) -> List[ConflictMarker]:
        """
        Detect conflicts between different analysis results.
        
        Args:
            analyses: List of analysis results to compare
            verification_result: Optional verification result
            context: Optional context window
            
        Returns:
            List of detected conflict markers
        """
        self.logger.info(f"Starting conflict detection for {len(analyses)} analyses")
        conflicts = []
        
        try:
            if len(analyses) < 2:
                # Need at least 2 analyses to detect conflicts
                return conflicts
            
            # Compare analyses pairwise
            for i in range(len(analyses)):
                for j in range(i + 1, len(analyses)):
                    analysis1, analysis2 = analyses[i], analyses[j]
                    
                    # Detect location mismatches
                    location_conflicts = self._detect_location_conflicts(analysis1, analysis2)
                    conflicts.extend(location_conflicts)
                    
                    # Detect confidence discrepancies
                    confidence_conflicts = self._detect_confidence_conflicts(analysis1, analysis2)
                    conflicts.extend(confidence_conflicts)
                    
                    # Detect evidence contradictions
                    evidence_conflicts = self._detect_evidence_conflicts(analysis1, analysis2)
                    conflicts.extend(evidence_conflicts)
                    
                    # Detect fix inconsistencies
                    fix_conflicts = self._detect_fix_conflicts(analysis1, analysis2)
                    conflicts.extend(fix_conflicts)
            
            # Check verification result for additional conflicts
            if verification_result and not verification_result.is_consistent:
                for conflict in verification_result.conflicts:
                    conflict_marker = ConflictMarker(
                        conflict_id=f"verification_{len(conflicts)}",
                        conflict_type=ConflictType.EVIDENCE_CONTRADICTION,
                        severity=ConflictSeverity.MEDIUM,
                        description=conflict.description,
                        affected_analyses=analyses,
                        detection_confidence=0.8
                    )
                    conflicts.append(conflict_marker)
            
            # Detect internal consistency issues
            for analysis in analyses:
                internal_conflicts = self._detect_internal_conflicts(analysis)
                conflicts.extend(internal_conflicts)
            
            # Prioritize conflicts by severity and impact
            conflicts = self._prioritize_conflicts(conflicts)
            
            self.logger.info(f"Detected {len(conflicts)} conflicts")
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Error in conflict detection: {e}")
            return []
    
    async def handle_conflicts(self, conflicts: List[ConflictMarker], 
                             original_issue: str,
                             code: Optional[str] = None,
                             context: Optional[ContextWindow] = None) -> ResolvedAnalysis:
        """
        Handle and resolve detected conflicts using multi-strategy reasoning.
        
        Args:
            conflicts: List of conflict markers to resolve
            original_issue: Original issue description
            code: Optional source code
            context: Optional context window
            
        Returns:
            Resolved analysis after conflict handling
        """
        self.logger.info(f"Handling {len(conflicts)} conflicts")
        
        if not conflicts:
            # No conflicts to handle, return the primary analysis
            primary_analysis = conflicts[0].affected_analyses[0] if conflicts else None
            if not primary_analysis:
                raise ValueError("No analysis available for conflict resolution")
            
            return ResolvedAnalysis(
                final_result=primary_analysis,
                resolution_method="no_conflicts",
                resolution_confidence=primary_analysis.confidence,
                resolution_notes="No conflicts detected"
            )
        
        try:
            # Group conflicts by type for efficient resolution
            conflict_groups = self._group_conflicts_by_type(conflicts)
            
            resolved_analyses = []
            
            # Resolve each group of conflicts
            for conflict_type, conflict_list in conflict_groups.items():
                self.logger.info(f"Resolving {len(conflict_list)} {conflict_type.value} conflicts")
                
                group_resolution = await self._resolve_conflict_group(
                    conflict_type, conflict_list, original_issue, code, context
                )
                
                if group_resolution:
                    resolved_analyses.append(group_resolution)
            
            # If multiple resolutions, perform final integration
            if len(resolved_analyses) > 1:
                final_resolution = await self._integrate_resolutions(
                    resolved_analyses, original_issue, context
                )
            elif resolved_analyses:
                final_resolution = resolved_analyses[0]
            else:
                # Fallback: use the highest confidence analysis from conflicts
                all_analyses = []
                for conflict in conflicts:
                    all_analyses.extend(conflict.affected_analyses)
                
                if all_analyses:
                    best_analysis = max(all_analyses, key=lambda a: a.confidence)
                    final_resolution = ResolvedAnalysis(
                        final_result=best_analysis,
                        resolution_method="fallback_highest_confidence",
                        resolution_confidence=best_analysis.confidence,
                        resolution_notes="Used fallback resolution due to conflict handling failure"
                    )
                else:
                    raise ValueError("No analyses available for resolution")
            
            self.logger.info(f"Conflicts resolved with confidence: {final_resolution.resolution_confidence:.2f}")
            return final_resolution
            
        except Exception as e:
            self.logger.error(f"Error in conflict handling: {e}")
            # Return a fallback resolution
            return self._create_fallback_resolution(conflicts, str(e))
    
    async def resolve_conflicts(self, conflicts: List[ConflictMarker], 
                              context: Optional[ContextWindow] = None) -> Optional[ResolvedAnalysis]:
        """
        Resolve detected conflicts and return a unified analysis.
        
        Args:
            conflicts: List of detected conflicts
            context: Optional context window for additional information
            
        Returns:
            Resolved analysis or None if resolution fails
        """
        if not conflicts:
            return None
        
        try:
            # Collect all affected analyses
            all_analyses = []
            for conflict in conflicts:
                all_analyses.extend(conflict.affected_analyses)
            
            # Remove duplicates
            unique_analyses = []
            seen_ids = set()
            for analysis in all_analyses:
                analysis_id = id(analysis)  # Use object id as unique identifier
                if analysis_id not in seen_ids:
                    unique_analyses.append(analysis)
                    seen_ids.add(analysis_id)
            
            if not unique_analyses:
                return None
            
            # If only one analysis, return it as resolved
            if len(unique_analyses) == 1:
                return ResolvedAnalysis(
                    final_result=unique_analyses[0],
                    resolution_method="single_analysis",
                    resolution_confidence=unique_analyses[0].confidence,
                    resolution_notes="Only one analysis available"
                )
            
            # Use multi-round reasoner to resolve conflicts
            if self.multi_round_reasoner:
                try:
                    # Create a conflict description for the reasoner
                    conflict_descriptions = [f"{c.conflict_type.value}: {c.description}" for c in conflicts]
                    issue_description = f"Conflicts detected: {'; '.join(conflict_descriptions)}"
                    
                    resolved = await self.multi_round_reasoner.resolve_conflicts(conflicts, issue_description)
                    return resolved
                except Exception as e:
                    self.logger.warning(f"Multi-round conflict resolution failed: {e}")
            
            # Fallback: choose highest confidence analysis
            best_analysis = max(unique_analyses, key=lambda a: a.confidence)
            discarded = [a for a in unique_analyses if a != best_analysis]
            
            return ResolvedAnalysis(
                final_result=best_analysis,
                resolution_method="highest_confidence_fallback",
                discarded_alternatives=discarded,
                resolution_confidence=best_analysis.confidence * 0.8,  # Reduce confidence due to conflicts
                resolution_notes=f"Selected highest confidence analysis from {len(unique_analyses)} options"
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving conflicts: {e}")
            return None

    async def collect_adaptive_context(self, analysis: AnalysisResult, 
                                     original_context: ContextWindow,
                                     code: Optional[str] = None) -> AdaptiveContext:
        """
        Collect additional context adaptively when confidence is low.
        
        Args:
            analysis: Analysis result with low confidence
            original_context: Original context window
            code: Optional source code for analysis
            
        Returns:
            Adaptive context with additional information
        """
        self.logger.info(f"Collecting adaptive context for low confidence analysis ({analysis.confidence:.2f})")
        
        adaptive_context = AdaptiveContext(original_context=original_context)
        
        try:
            # Expand function context if code is available
            if code and original_context.related_functions:
                additional_functions = await self._collect_additional_functions(
                    code, original_context.related_functions, analysis
                )
                adaptive_context.additional_functions = additional_functions
            
            # Collect extended dependencies
            if original_context.dependency_context:
                extended_deps = await self._collect_extended_dependencies(
                    original_context.dependency_context, analysis
                )
                adaptive_context.extended_dependencies = extended_deps
            
            # Find similar code patterns
            if code:
                similar_patterns = await self._find_similar_patterns(code, analysis)
                adaptive_context.similar_code_patterns = similar_patterns
            
            # Collect domain-specific information
            domain_info = await self._collect_domain_info(analysis, original_context)
            adaptive_context.domain_specific_info = domain_info
            
            # Calculate collection confidence
            adaptive_context.collection_confidence = self._calculate_collection_confidence(
                adaptive_context
            )
            
            self.logger.info(f"Adaptive context collected with confidence: {adaptive_context.collection_confidence:.2f}")
            return adaptive_context
            
        except Exception as e:
            self.logger.error(f"Error in adaptive context collection: {e}")
            adaptive_context.collection_confidence = 0.1
            return adaptive_context
    
    async def multi_strategy_reasoning(self, conflicts: List[ConflictMarker],
                                     original_issue: str,
                                     context: Optional[ContextWindow] = None) -> List[AnalysisResult]:
        """
        Apply multiple reasoning strategies to resolve conflicts.
        
        Args:
            conflicts: List of conflicts to resolve
            original_issue: Original issue description
            context: Optional context window
            
        Returns:
            List of analysis results from different strategies
        """
        self.logger.info(f"Applying multi-strategy reasoning to {len(conflicts)} conflicts")
        
        strategy_results = []
        
        try:
            # Strategy 1: Conservative approach (high confidence threshold)
            conservative_result = await self._apply_conservative_strategy(
                conflicts, original_issue, context
            )
            if conservative_result:
                strategy_results.append(conservative_result)
            
            # Strategy 2: Evidence-based approach
            evidence_result = await self._apply_evidence_strategy(
                conflicts, original_issue, context
            )
            if evidence_result:
                strategy_results.append(evidence_result)
            
            # Strategy 3: Pattern-matching approach
            pattern_result = await self._apply_pattern_strategy(
                conflicts, original_issue, context
            )
            if pattern_result:
                strategy_results.append(pattern_result)
            
            # Strategy 4: Hybrid AST-LLM approach
            hybrid_result = await self._apply_hybrid_strategy(
                conflicts, original_issue, context
            )
            if hybrid_result:
                strategy_results.append(hybrid_result)
            
            self.logger.info(f"Multi-strategy reasoning produced {len(strategy_results)} results")
            return strategy_results
            
        except Exception as e:
            self.logger.error(f"Error in multi-strategy reasoning: {e}")
            return []
    
    # Private helper methods
    
    async def _generate_ast_analysis(self, llm_analysis: AnalysisResult, 
                                   code: str, context: ContextWindow) -> Optional[AnalysisResult]:
        """Generate AST analysis for comparison with LLM analysis."""
        try:
            return self.ast_analyzer.integrate_with_llm_analysis(code, llm_analysis, context)
        except Exception as e:
            self.logger.warning(f"Failed to generate AST analysis: {e}")
            return None
    
    def _detect_location_conflicts(self, llm_analysis: AnalysisResult, 
                                 ast_analysis: AnalysisResult) -> List[ConflictMarker]:
        """Detect conflicts in bug location identification."""
        conflicts = []
        
        # Compare bug locations
        llm_location = llm_analysis.bug_location.lower()
        ast_location = ast_analysis.bug_location.lower()
        
        # Simple similarity check (can be enhanced with more sophisticated matching)
        similarity = self._calculate_location_similarity(llm_location, ast_location)
        
        if similarity < self.location_similarity_threshold:
            conflict = ConflictMarker(
                conflict_id=f"location_mismatch_{datetime.now().timestamp()}",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.HIGH,
                description=f"Location mismatch: LLM says '{llm_location}', AST says '{ast_location}'",
                affected_analyses=[llm_analysis, ast_analysis],
                detection_confidence=1.0 - similarity,
                resolution_priority=1,
                metadata={
                    "llm_location": llm_location,
                    "ast_location": ast_location,
                    "similarity_score": similarity
                }
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_confidence_conflicts(self, llm_analysis: AnalysisResult, 
                                   ast_analysis: AnalysisResult) -> List[ConflictMarker]:
        """Detect conflicts in confidence levels."""
        conflicts = []
        
        confidence_diff = abs(llm_analysis.confidence - ast_analysis.confidence)
        
        # If confidence difference is significant
        if confidence_diff > 0.3:
            severity = ConflictSeverity.HIGH if confidence_diff > 0.5 else ConflictSeverity.MEDIUM
            
            conflict = ConflictMarker(
                conflict_id=f"confidence_discrepancy_{datetime.now().timestamp()}",
                conflict_type=ConflictType.CONFIDENCE_DISCREPANCY,
                severity=severity,
                description=f"Significant confidence difference: LLM={llm_analysis.confidence:.2f}, AST={ast_analysis.confidence:.2f}",
                affected_analyses=[llm_analysis, ast_analysis],
                detection_confidence=min(1.0, confidence_diff * 2),
                resolution_priority=2,
                metadata={
                    "llm_confidence": llm_analysis.confidence,
                    "ast_confidence": ast_analysis.confidence,
                    "difference": confidence_diff
                }
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_evidence_conflicts(self, llm_analysis: AnalysisResult, 
                                 ast_analysis: AnalysisResult) -> List[ConflictMarker]:
        """Detect conflicts in supporting evidence."""
        conflicts = []
        
        # Check for contradictory evidence
        llm_evidence = set(llm_analysis.supporting_evidence)
        ast_evidence = set(ast_analysis.supporting_evidence)
        
        # Look for explicit contradictions (simple keyword-based approach)
        contradiction_keywords = [
            ("no error", "error found"),
            ("valid", "invalid"),
            ("correct", "incorrect"),
            ("working", "broken")
        ]
        
        contradictions_found = []
        for llm_ev in llm_evidence:
            for ast_ev in ast_evidence:
                for pos_kw, neg_kw in contradiction_keywords:
                    if (pos_kw in llm_ev.lower() and neg_kw in ast_ev.lower()) or \
                       (neg_kw in llm_ev.lower() and pos_kw in ast_ev.lower()):
                        contradictions_found.append((llm_ev, ast_ev))
        
        if contradictions_found:
            conflict = ConflictMarker(
                conflict_id=f"evidence_contradiction_{datetime.now().timestamp()}",
                conflict_type=ConflictType.EVIDENCE_CONTRADICTION,
                severity=ConflictSeverity.MEDIUM,
                description=f"Found {len(contradictions_found)} evidence contradictions",
                affected_analyses=[llm_analysis, ast_analysis],
                detection_confidence=min(1.0, len(contradictions_found) * 0.3),
                resolution_priority=3,
                metadata={
                    "contradictions": contradictions_found,
                    "llm_evidence_count": len(llm_evidence),
                    "ast_evidence_count": len(ast_evidence)
                }
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_fix_conflicts(self, llm_analysis: AnalysisResult, 
                            ast_analysis: AnalysisResult) -> List[ConflictMarker]:
        """Detect conflicts in fix suggestions."""
        conflicts = []
        
        llm_fix = llm_analysis.fix_suggestion.lower()
        ast_fix = ast_analysis.fix_suggestion.lower()
        
        # Check for contradictory fix suggestions
        fix_similarity = self._calculate_text_similarity(llm_fix, ast_fix)
        
        if fix_similarity < 0.3:  # Very different fixes
            conflict = ConflictMarker(
                conflict_id=f"fix_inconsistency_{datetime.now().timestamp()}",
                conflict_type=ConflictType.FIX_INCONSISTENCY,
                severity=ConflictSeverity.MEDIUM,
                description="Inconsistent fix suggestions between LLM and AST analysis",
                affected_analyses=[llm_analysis, ast_analysis],
                detection_confidence=1.0 - fix_similarity,
                resolution_priority=4,
                metadata={
                    "llm_fix": llm_analysis.fix_suggestion,
                    "ast_fix": ast_analysis.fix_suggestion,
                    "similarity_score": fix_similarity
                }
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_internal_conflicts(self, analysis: AnalysisResult) -> List[ConflictMarker]:
        """Detect internal consistency issues within a single analysis."""
        conflicts = []
        
        # Check for multiple root causes mentioned
        root_cause_lower = analysis.root_cause.lower()
        cause_indicators = ["because", "due to", "caused by", "result of"]
        cause_count = sum(1 for indicator in cause_indicators if indicator in root_cause_lower)
        
        if cause_count > 2:  # Multiple potential causes mentioned
            conflict = ConflictMarker(
                conflict_id=f"multiple_causes_{datetime.now().timestamp()}",
                conflict_type=ConflictType.MULTIPLE_ROOT_CAUSES,
                severity=ConflictSeverity.LOW,
                description="Multiple potential root causes mentioned in analysis",
                affected_analyses=[analysis],
                detection_confidence=min(1.0, (cause_count - 2) * 0.2),
                resolution_priority=5,
                metadata={
                    "cause_indicators_found": cause_count,
                    "root_cause_text": analysis.root_cause
                }
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _prioritize_conflicts(self, conflicts: List[ConflictMarker]) -> List[ConflictMarker]:
        """Prioritize conflicts by severity and resolution priority."""
        # Sort by severity (critical first) then by priority (1 = highest)
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3
        }
        
        return sorted(conflicts, key=lambda c: (severity_order[c.severity], c.resolution_priority))
    
    def _group_conflicts_by_type(self, conflicts: List[ConflictMarker]) -> Dict[ConflictType, List[ConflictMarker]]:
        """Group conflicts by their type for efficient resolution."""
        groups = {}
        for conflict in conflicts:
            if conflict.conflict_type not in groups:
                groups[conflict.conflict_type] = []
            groups[conflict.conflict_type].append(conflict)
        return groups
    
    async def _resolve_conflict_group(self, conflict_type: ConflictType, 
                                    conflicts: List[ConflictMarker],
                                    original_issue: str,
                                    code: Optional[str] = None,
                                    context: Optional[ContextWindow] = None) -> Optional[ResolvedAnalysis]:
        """Resolve a group of conflicts of the same type."""
        # Find appropriate resolution strategy
        strategy = None
        for strat in self.resolution_strategies.values():
            if conflict_type in strat.applicable_conflicts:
                strategy = strat
                break
        
        if not strategy:
            self.logger.warning(f"No resolution strategy found for conflict type: {conflict_type}")
            return None
        
        try:
            # Collect all affected analyses
            all_analyses = []
            for conflict in conflicts:
                all_analyses.extend(conflict.affected_analyses)
            
            # Remove duplicates while preserving order
            unique_analyses = []
            seen = set()
            for analysis in all_analyses:
                analysis_id = id(analysis)
                if analysis_id not in seen:
                    unique_analyses.append(analysis)
                    seen.add(analysis_id)
            
            if not unique_analyses:
                return None
            
            # Apply resolution strategy
            if strategy.prompt_template and len(unique_analyses) >= 2:
                resolved_analysis = await self._apply_llm_resolution(
                    strategy, conflicts, unique_analyses, original_issue, context
                )
            else:
                # Fallback: use highest confidence analysis
                resolved_analysis = max(unique_analyses, key=lambda a: a.confidence)
            
            return ResolvedAnalysis(
                final_result=resolved_analysis,
                resolution_method=strategy.strategy_name,
                discarded_alternatives=[a for a in unique_analyses if a != resolved_analysis],
                resolution_confidence=resolved_analysis.confidence,
                resolution_notes=f"Resolved {len(conflicts)} {conflict_type.value} conflicts"
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving conflict group {conflict_type}: {e}")
            return None
    
    async def _apply_llm_resolution(self, strategy: ResolutionStrategy,
                                  conflicts: List[ConflictMarker],
                                  analyses: List[AnalysisResult],
                                  original_issue: str,
                                  context: Optional[ContextWindow] = None) -> AnalysisResult:
        """Apply LLM-based resolution using the strategy's prompt template."""
        if not strategy.prompt_template:
            raise ValueError("Strategy has no prompt template")
        
        # Prepare template variables based on conflict type
        template_vars = {"issue_description": original_issue}
        
        if strategy.strategy_name == "location_reconciliation":
            template_vars.update({
                "llm_location": analyses[0].bug_location,
                "ast_location": analyses[1].bug_location if len(analyses) > 1 else "N/A",
                "code_context": self._format_context(context) if context else "N/A"
            })
        elif strategy.strategy_name == "confidence_calibration":
            template_vars.update({
                "analysis1": self._format_analysis_summary(analyses[0]),
                "analysis2": self._format_analysis_summary(analyses[1]) if len(analyses) > 1 else "N/A",
                "confidence1": analyses[0].confidence,
                "confidence2": analyses[1].confidence if len(analyses) > 1 else 0.0
            })
        elif strategy.strategy_name == "ast_llm_mediation":
            template_vars.update({
                "llm_analysis": self._format_analysis_summary(analyses[0]),
                "ast_findings": self._format_analysis_summary(analyses[1]) if len(analyses) > 1 else "N/A",
                "code_context": self._format_context(context) if context else "N/A"
            })
        
        # Generate resolution
        response = await self.llm.generate(
            strategy.prompt_template,
            template_vars=template_vars,
            max_tokens=1500,
            temperature=0.1
        )
        
        # Parse response and create resolved analysis
        resolution_data = self._parse_resolution_response(response.content)
        
        # Use the best analysis as base and update with resolution
        base_analysis = max(analyses, key=lambda a: a.confidence)
        
        resolved_analysis = AnalysisResult(
            bug_location=resolution_data.get("resolved_location", base_analysis.bug_location),
            root_cause=resolution_data.get("reasoning", base_analysis.root_cause),
            fix_suggestion=resolution_data.get("synthesized_analysis", base_analysis.fix_suggestion),
            confidence=resolution_data.get("confidence", base_analysis.confidence),
            reasoning_chain=base_analysis.reasoning_chain,
            supporting_evidence=base_analysis.supporting_evidence + 
                              resolution_data.get("supporting_evidence", [])
        )
        
        return resolved_analysis
    
    def _parse_resolution_response(self, response: str) -> Dict[str, Any]:
        """Parse resolution response from LLM."""
        try:
            import json
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "reasoning": response,
                "confidence": 0.5,
                "supporting_evidence": ["LLM resolution applied"]
            }
    
    def _calculate_location_similarity(self, loc1: str, loc2: str) -> float:
        """Calculate similarity between two location strings."""
        # Simple similarity based on common words and line numbers
        import re
        
        # Extract line numbers
        lines1 = re.findall(r'\d+', loc1)
        lines2 = re.findall(r'\d+', loc2)
        
        # Check for line number overlap
        line_overlap = len(set(lines1) & set(lines2)) / max(len(set(lines1) | set(lines2)), 1)
        
        # Check for word overlap
        words1 = set(loc1.lower().split())
        words2 = set(loc2.lower().split())
        word_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        
        # Combine metrics
        return (line_overlap + word_overlap) / 2
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _format_context(self, context: Optional[ContextWindow]) -> str:
        """Format context window for LLM consumption."""
        if not context:
            return "No context available"
        
        formatted = f"Target Code:\n{context.target_code[:500]}...\n\n"
        
        if context.related_functions:
            formatted += f"Related Functions: {', '.join(context.related_functions[:3])}\n"
        
        return formatted
    
    def _format_analysis_summary(self, analysis: AnalysisResult) -> str:
        """Format analysis result as a summary."""
        return f"Location: {analysis.bug_location}, Cause: {analysis.root_cause[:100]}..., Fix: {analysis.fix_suggestion[:100]}..."
    
    def _create_fallback_resolution(self, conflicts: List[ConflictMarker], error: str) -> ResolvedAnalysis:
        """Create fallback resolution when conflict handling fails."""
        # Find the analysis with highest confidence
        all_analyses = []
        for conflict in conflicts:
            all_analyses.extend(conflict.affected_analyses)
        
        if all_analyses:
            best_analysis = max(all_analyses, key=lambda a: a.confidence)
        else:
            # Create minimal fallback analysis
            best_analysis = AnalysisResult(
                bug_location="Unknown",
                root_cause="Conflict resolution failed",
                fix_suggestion="Manual investigation required",
                confidence=0.1,
                reasoning_chain=ReasoningChain(),
                supporting_evidence=[f"Error: {error}"]
            )
        
        return ResolvedAnalysis(
            final_result=best_analysis,
            resolution_method="fallback",
            resolution_confidence=best_analysis.confidence,
            resolution_notes=f"Fallback resolution due to error: {error}"
        )
    
    # Adaptive context collection methods
    
    async def _collect_additional_functions(self, code: str, existing_functions: List[str], 
                                          analysis: AnalysisResult) -> List[str]:
        """Collect additional functions that might be relevant."""
        # This is a simplified implementation - can be enhanced with more sophisticated analysis
        additional_functions = []
        
        try:
            import ast
            tree = ast.parse(code)
            
            # Find all function definitions
            all_functions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    all_functions.append(node.name)
            
            # Add functions not already in the context
            for func in all_functions:
                if func not in existing_functions and len(additional_functions) < 5:
                    additional_functions.append(func)
        
        except Exception as e:
            self.logger.warning(f"Error collecting additional functions: {e}")
        
        return additional_functions
    
    async def _collect_extended_dependencies(self, dep_context, analysis: AnalysisResult) -> List[str]:
        """Collect extended dependencies beyond the original context."""
        # Simplified implementation
        extended_deps = []
        
        if hasattr(dep_context, 'import_statements'):
            # Add more import statements that might be relevant
            for imp in dep_context.import_statements[:3]:
                if imp not in extended_deps:
                    extended_deps.append(f"Extended: {imp}")
        
        return extended_deps
    
    async def _find_similar_patterns(self, code: str, analysis: AnalysisResult) -> List[str]:
        """Find similar code patterns that might be relevant."""
        # Simplified pattern matching
        patterns = []
        
        # Look for similar variable names or function calls mentioned in analysis
        analysis_text = f"{analysis.root_cause} {analysis.fix_suggestion}".lower()
        
        # Extract potential identifiers from analysis
        import re
        identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', analysis_text)
        
        for identifier in identifiers[:3]:
            if identifier in code.lower():
                patterns.append(f"Pattern found: {identifier} usage in code")
        
        return patterns
    
    async def _collect_domain_info(self, analysis: AnalysisResult, 
                                 context: ContextWindow) -> Dict[str, Any]:
        """Collect domain-specific information."""
        domain_info = {}
        
        # Extract potential domain from context
        if context.domain_concepts:
            domain_info["concepts"] = context.domain_concepts[:3]
        
        # Add analysis-specific domain information
        domain_info["analysis_domain"] = "code_analysis"
        domain_info["confidence_factors"] = ["ast_integration", "llm_reasoning"]
        
        return domain_info
    
    def _calculate_collection_confidence(self, adaptive_context: AdaptiveContext) -> float:
        """Calculate confidence in the collected adaptive context."""
        confidence_factors = []
        
        if adaptive_context.additional_functions:
            confidence_factors.append(0.2)
        
        if adaptive_context.extended_dependencies:
            confidence_factors.append(0.2)
        
        if adaptive_context.similar_code_patterns:
            confidence_factors.append(0.3)
        
        if adaptive_context.domain_specific_info:
            confidence_factors.append(0.3)
        
        return sum(confidence_factors) if confidence_factors else 0.1
    
    # Multi-strategy reasoning methods
    
    async def _apply_conservative_strategy(self, conflicts: List[ConflictMarker],
                                         original_issue: str,
                                         context: Optional[ContextWindow] = None) -> Optional[AnalysisResult]:
        """Apply conservative strategy with high confidence threshold."""
        # Find the analysis with highest confidence that meets conservative threshold
        all_analyses = []
        for conflict in conflicts:
            all_analyses.extend(conflict.affected_analyses)
        
        conservative_threshold = 0.8
        conservative_analyses = [a for a in all_analyses if a.confidence >= conservative_threshold]
        
        if conservative_analyses:
            return max(conservative_analyses, key=lambda a: a.confidence)
        
        return None
    
    async def _apply_evidence_strategy(self, conflicts: List[ConflictMarker],
                                     original_issue: str,
                                     context: Optional[ContextWindow] = None) -> Optional[AnalysisResult]:
        """Apply evidence-based strategy focusing on supporting evidence quality."""
        all_analyses = []
        for conflict in conflicts:
            all_analyses.extend(conflict.affected_analyses)
        
        if not all_analyses:
            return None
        
        # Score analyses based on evidence quality
        def evidence_score(analysis):
            evidence_count = len(analysis.supporting_evidence)
            evidence_quality = sum(1 for ev in analysis.supporting_evidence if len(ev) > 20)
            return evidence_count * 0.3 + evidence_quality * 0.7
        
        return max(all_analyses, key=evidence_score)
    
    async def _apply_pattern_strategy(self, conflicts: List[ConflictMarker],
                                    original_issue: str,
                                    context: Optional[ContextWindow] = None) -> Optional[AnalysisResult]:
        """Apply pattern-matching strategy."""
        # Simplified pattern matching - can be enhanced with actual pattern database
        all_analyses = []
        for conflict in conflicts:
            all_analyses.extend(conflict.affected_analyses)
        
        if not all_analyses:
            return None
        
        # Prefer analyses that mention common bug patterns
        pattern_keywords = ["null pointer", "index out of bounds", "type error", "undefined variable"]
        
        def pattern_score(analysis):
            text = f"{analysis.root_cause} {analysis.fix_suggestion}".lower()
            return sum(1 for keyword in pattern_keywords if keyword in text)
        
        pattern_analyses = [a for a in all_analyses if pattern_score(a) > 0]
        
        if pattern_analyses:
            return max(pattern_analyses, key=lambda a: (pattern_score(a), a.confidence))
        
        return max(all_analyses, key=lambda a: a.confidence)
    
    async def _apply_hybrid_strategy(self, conflicts: List[ConflictMarker],
                                   original_issue: str,
                                   context: Optional[ContextWindow] = None) -> Optional[AnalysisResult]:
        """Apply hybrid AST-LLM strategy."""
        # Find analyses that have both AST and LLM components
        all_analyses = []
        for conflict in conflicts:
            all_analyses.extend(conflict.affected_analyses)
        
        if not all_analyses:
            return None
        
        # Prefer analyses with more comprehensive evidence (indicating AST integration)
        def hybrid_score(analysis):
            evidence_diversity = len(set(ev.split(':')[0] for ev in analysis.supporting_evidence))
            reasoning_steps = len(analysis.reasoning_chain.steps)
            return evidence_diversity * 0.4 + reasoning_steps * 0.6
        
        return max(all_analyses, key=lambda a: (hybrid_score(a), a.confidence))
    
    async def _integrate_resolutions(self, resolutions: List[ResolvedAnalysis],
                                   original_issue: str,
                                   context: Optional[ContextWindow] = None) -> ResolvedAnalysis:
        """Integrate multiple resolutions into a final resolution."""
        if len(resolutions) == 1:
            return resolutions[0]
        
        # Use multi-round reasoner to integrate multiple resolutions
        conflicts = []
        for i, resolution in enumerate(resolutions):
            conflict = Conflict(
                conflict_type="resolution_integration",
                description=f"Resolution {i+1} from {resolution.resolution_method}",
                conflicting_analyses=[resolution.final_result]
            )
            conflicts.append(conflict)
        
        try:
            final_resolution = await self.multi_round_reasoner.resolve_conflicts(conflicts, original_issue)
            return final_resolution
        except Exception as e:
            self.logger.error(f"Error integrating resolutions: {e}")
            # Fallback: return highest confidence resolution
            return max(resolutions, key=lambda r: r.resolution_confidence)