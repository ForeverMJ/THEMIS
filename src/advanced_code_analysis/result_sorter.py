"""
Result sorting and output engine for the Advanced Code Analysis system.

This module implements the ResultSorter class that handles candidate solution ranking
based on code impact analysis, comprehensive output formatting with reasoning processes
and confidence evaluation, evidence chain construction and display, and result
validation and quality checks.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .models import (
    AnalysisResult, ReasoningChain, ReasoningStep, EvidenceChain,
    ContextWindow, BugType, Conflict, VerificationResult
)
from .config import AdvancedAnalysisConfig


logger = logging.getLogger(__name__)


class SortingCriteria(Enum):
    """Criteria for sorting candidate solutions."""
    CONFIDENCE = "confidence"
    CODE_IMPACT = "code_impact"
    EVIDENCE_STRENGTH = "evidence_strength"
    REASONING_QUALITY = "reasoning_quality"
    IMPLEMENTATION_COMPLEXITY = "implementation_complexity"


class OutputFormat(Enum):
    """Available output formats."""
    DETAILED = "detailed"
    SUMMARY = "summary"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class CodeImpactMetrics:
    """Metrics for assessing code impact of a solution."""
    lines_affected: int = 0
    functions_affected: int = 0
    classes_affected: int = 0
    modules_affected: int = 0
    dependency_impact: float = 0.0  # 0.0 = no impact, 1.0 = high impact
    risk_score: float = 0.0  # 0.0 = low risk, 1.0 = high risk
    implementation_effort: float = 0.0  # 0.0 = easy, 1.0 = complex
    
    def calculate_overall_impact(self) -> float:
        """Calculate overall impact score (0.0 = low impact, 1.0 = high impact)."""
        # Weighted combination of different impact factors
        weights = {
            'lines': 0.2,
            'functions': 0.25,
            'classes': 0.2,
            'modules': 0.15,
            'dependency': 0.1,
            'risk': 0.1
        }
        
        # Normalize metrics to 0-1 scale
        normalized_lines = min(self.lines_affected / 50.0, 1.0)  # 50+ lines = high impact
        normalized_functions = min(self.functions_affected / 10.0, 1.0)  # 10+ functions = high impact
        normalized_classes = min(self.classes_affected / 5.0, 1.0)  # 5+ classes = high impact
        normalized_modules = min(self.modules_affected / 3.0, 1.0)  # 3+ modules = high impact
        
        overall_impact = (
            weights['lines'] * normalized_lines +
            weights['functions'] * normalized_functions +
            weights['classes'] * normalized_classes +
            weights['modules'] * normalized_modules +
            weights['dependency'] * self.dependency_impact +
            weights['risk'] * self.risk_score
        )
        
        return min(overall_impact, 1.0)


@dataclass
class QualityMetrics:
    """Quality metrics for analysis results."""
    reasoning_completeness: float = 0.0  # How complete is the reasoning chain
    evidence_strength: float = 0.0  # Strength of supporting evidence
    consistency_score: float = 0.0  # Internal consistency of the analysis
    specificity_score: float = 0.0  # How specific vs. vague the analysis is
    actionability_score: float = 0.0  # How actionable the fix suggestion is
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall quality score."""
        return (
            self.reasoning_completeness * 0.25 +
            self.evidence_strength * 0.25 +
            self.consistency_score * 0.2 +
            self.specificity_score * 0.15 +
            self.actionability_score * 0.15
        )


@dataclass
class RankedResult:
    """Analysis result with ranking information."""
    analysis: AnalysisResult
    rank: int
    impact_metrics: CodeImpactMetrics
    quality_metrics: QualityMetrics
    composite_score: float
    ranking_rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rank': self.rank,
            'analysis': {
                'bug_location': self.analysis.bug_location,
                'root_cause': self.analysis.root_cause,
                'fix_suggestion': self.analysis.fix_suggestion,
                'confidence': self.analysis.confidence,
                'supporting_evidence': self.analysis.supporting_evidence
            },
            'impact_metrics': {
                'lines_affected': self.impact_metrics.lines_affected,
                'functions_affected': self.impact_metrics.functions_affected,
                'classes_affected': self.impact_metrics.classes_affected,
                'modules_affected': self.impact_metrics.modules_affected,
                'dependency_impact': self.impact_metrics.dependency_impact,
                'risk_score': self.impact_metrics.risk_score,
                'implementation_effort': self.impact_metrics.implementation_effort,
                'overall_impact': self.impact_metrics.calculate_overall_impact()
            },
            'quality_metrics': {
                'reasoning_completeness': self.quality_metrics.reasoning_completeness,
                'evidence_strength': self.quality_metrics.evidence_strength,
                'consistency_score': self.quality_metrics.consistency_score,
                'specificity_score': self.quality_metrics.specificity_score,
                'actionability_score': self.quality_metrics.actionability_score,
                'overall_quality': self.quality_metrics.calculate_overall_quality()
            },
            'composite_score': self.composite_score,
            'ranking_rationale': self.ranking_rationale
        }


@dataclass
class SortedResults:
    """Results from sorting and ranking candidates."""
    ranked_candidates: List[AnalysisResult]
    quality_assessment: Dict[str, Any]
    sorting_metadata: Dict[str, Any]


@dataclass
class FormattedOutput:
    """Formatted analysis output with complete information."""
    title: str
    summary: str
    ranked_results: List[RankedResult]
    reasoning_process: str
    evidence_chains: Dict[int, str]  # rank -> formatted evidence chain
    quality_assessment: str
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        output_dict = {
            'title': self.title,
            'summary': self.summary,
            'ranked_results': [result.to_dict() for result in self.ranked_results],
            'reasoning_process': self.reasoning_process,
            'evidence_chains': self.evidence_chains,
            'quality_assessment': self.quality_assessment,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }
        return json.dumps(output_dict, indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        md_lines = [
            f"# {self.title}",
            "",
            f"## Summary",
            self.summary,
            "",
            f"## Analysis Results",
            ""
        ]
        
        for result in self.ranked_results:
            md_lines.extend([
                f"### Rank {result.rank}: {result.analysis.bug_location}",
                "",
                f"**Root Cause:** {result.analysis.root_cause}",
                "",
                f"**Fix Suggestion:** {result.analysis.fix_suggestion}",
                "",
                f"**Confidence:** {result.analysis.confidence:.2f}",
                "",
                f"**Impact Score:** {result.impact_metrics.calculate_overall_impact():.2f}",
                "",
                f"**Quality Score:** {result.quality_metrics.calculate_overall_quality():.2f}",
                "",
                f"**Rationale:** {result.ranking_rationale}",
                "",
                "**Evidence:**",
                ""
            ])
            
            for evidence in result.analysis.supporting_evidence:
                md_lines.append(f"- {evidence}")
            
            md_lines.extend(["", "---", ""])
        
        md_lines.extend([
            f"## Reasoning Process",
            self.reasoning_process,
            "",
            f"## Quality Assessment", 
            self.quality_assessment,
            "",
            f"## Recommendations",
            ""
        ])
        
        for rec in self.recommendations:
            md_lines.append(f"- {rec}")
        
        return "\n".join(md_lines)


class ResultSorter:
    """
    Result sorting and output engine for candidate solution ranking and formatting.
    
    This class implements:
    1. Code impact analysis for solution ranking
    2. Comprehensive output formatting with reasoning processes
    3. Evidence chain construction and display
    4. Result validation and quality assessment
    """
    
    def __init__(self, config: AdvancedAnalysisConfig):
        """Initialize the result sorter."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sorting weights for different criteria
        self.sorting_weights = {
            SortingCriteria.CONFIDENCE: 0.3,
            SortingCriteria.CODE_IMPACT: 0.25,  # Lower impact = higher rank
            SortingCriteria.EVIDENCE_STRENGTH: 0.2,
            SortingCriteria.REASONING_QUALITY: 0.15,
            SortingCriteria.IMPLEMENTATION_COMPLEXITY: 0.1  # Lower complexity = higher rank
        }
    
    async def sort_and_rank_candidates(self, candidates: List[AnalysisResult], 
                                      context: Optional[ContextWindow] = None) -> 'SortedResults':
        """
        Sort and rank candidate solutions with comprehensive analysis.
        
        Args:
            candidates: List of analysis results to sort and rank
            context: Optional context window for impact analysis
            
        Returns:
            SortedResults object containing ranked candidates and quality assessment
        """
        try:
            if not candidates:
                return SortedResults(
                    ranked_candidates=[],
                    quality_assessment={},
                    sorting_metadata={}
                )
            
            # Use default context if none provided
            if context is None:
                context = ContextWindow(target_code="", token_count=0)
            
            # Sort candidates using existing method
            ranked_results = self.sort_candidates(candidates, context)
            
            # Extract just the analysis results in ranked order
            ranked_candidates = [result.analysis for result in ranked_results]
            
            # Generate quality assessment
            quality_assessment = {}
            if ranked_results:
                total_confidence = sum(r.analysis.confidence for r in ranked_results)
                avg_confidence = total_confidence / len(ranked_results)
                
                quality_assessment = {
                    'average_confidence': avg_confidence,
                    'total_candidates': len(ranked_results),
                    'high_confidence_count': sum(1 for r in ranked_results if r.analysis.confidence > 0.8),
                    'best_score': ranked_results[0].composite_score if ranked_results else 0.0
                }
            
            # Generate sorting metadata
            sorting_metadata = {
                'sorting_criteria': ['confidence', 'code_impact', 'evidence_strength'],
                'timestamp': datetime.now().isoformat(),
                'total_processed': len(candidates)
            }
            
            return SortedResults(
                ranked_candidates=ranked_candidates,
                quality_assessment=quality_assessment,
                sorting_metadata=sorting_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in sort_and_rank_candidates: {e}")
            return SortedResults(
                ranked_candidates=candidates,  # Return unsorted as fallback
                quality_assessment={'error': str(e)},
                sorting_metadata={'error': True}
            )

    def sort_candidates(self, candidates: List[AnalysisResult], 
                       context: ContextWindow,
                       sorting_criteria: List[SortingCriteria] = None) -> List[RankedResult]:
        """
        Sort candidate solutions based on multiple criteria.
        
        Args:
            candidates: List of analysis results to sort
            context: Code context for impact analysis
            sorting_criteria: Criteria to use for sorting (default: all criteria)
            
        Returns:
            List of ranked results sorted by composite score
        """
        self.logger.info(f"Sorting {len(candidates)} candidate solutions")
        
        if not candidates:
            return []
        
        if sorting_criteria is None:
            sorting_criteria = list(SortingCriteria)
        
        ranked_results = []
        
        for i, candidate in enumerate(candidates):
            # Calculate impact metrics
            impact_metrics = self._analyze_code_impact(candidate, context)
            
            # Calculate quality metrics
            quality_metrics = self._assess_quality(candidate)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                candidate, impact_metrics, quality_metrics, sorting_criteria
            )
            
            # Generate ranking rationale
            rationale = self._generate_ranking_rationale(
                candidate, impact_metrics, quality_metrics, composite_score
            )
            
            ranked_result = RankedResult(
                analysis=candidate,
                rank=0,  # Will be set after sorting
                impact_metrics=impact_metrics,
                quality_metrics=quality_metrics,
                composite_score=composite_score,
                ranking_rationale=rationale
            )
            
            ranked_results.append(ranked_result)
        
        # Sort by composite score (higher is better)
        ranked_results.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign final ranks
        for i, result in enumerate(ranked_results):
            result.rank = i + 1
        
        self.logger.info(f"Candidates sorted. Top result has score: {ranked_results[0].composite_score:.3f}")
        return ranked_results
    
    def format_output(self, ranked_results: List[RankedResult],
                     issue_description: str,
                     bug_type: BugType,
                     output_format: OutputFormat = OutputFormat.DETAILED) -> FormattedOutput:
        """
        Format analysis results into comprehensive output.
        
        Args:
            ranked_results: Sorted analysis results
            issue_description: Original issue description
            bug_type: Classified bug type
            output_format: Desired output format
            
        Returns:
            Formatted output with complete analysis information
        """
        self.logger.info(f"Formatting output for {len(ranked_results)} results in {output_format.value} format")
        
        # Generate title and summary
        title = f"Advanced Code Analysis Results - {bug_type.category.value.replace('_', ' ').title()}"
        summary = self._generate_summary(ranked_results, issue_description, bug_type)
        
        # Build reasoning process description
        reasoning_process = self._build_reasoning_process_description(ranked_results)
        
        # Build evidence chains for each result
        evidence_chains = {}
        for result in ranked_results:
            evidence_chains[result.rank] = self._format_evidence_chain(result.analysis)
        
        # Generate quality assessment
        quality_assessment = self._generate_quality_assessment(ranked_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(ranked_results, bug_type)
        
        # Prepare metadata
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'bug_type': bug_type.category.value,
            'bug_confidence': bug_type.confidence,
            'total_candidates': len(ranked_results),
            'top_confidence': ranked_results[0].analysis.confidence if ranked_results else 0.0,
            'analysis_version': '1.0'
        }
        
        formatted_output = FormattedOutput(
            title=title,
            summary=summary,
            ranked_results=ranked_results,
            reasoning_process=reasoning_process,
            evidence_chains=evidence_chains,
            quality_assessment=quality_assessment,
            recommendations=recommendations,
            metadata=metadata
        )
        
        self.logger.info("Output formatting completed")
        return formatted_output
    
    def validate_results(self, ranked_results: List[RankedResult]) -> Dict[str, Any]:
        """
        Validate analysis results and perform quality checks.
        
        Args:
            ranked_results: Results to validate
            
        Returns:
            Validation report with issues and quality metrics
        """
        self.logger.info(f"Validating {len(ranked_results)} analysis results")
        
        validation_report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'quality_scores': {},
            'recommendations': []
        }
        
        if not ranked_results:
            validation_report['is_valid'] = False
            validation_report['issues'].append("No analysis results to validate")
            return validation_report
        
        # Check confidence levels
        low_confidence_count = sum(1 for r in ranked_results if r.analysis.confidence < 0.5)
        if low_confidence_count == len(ranked_results):
            validation_report['warnings'].append("All results have low confidence (<0.5)")
        
        # Check for missing information
        for i, result in enumerate(ranked_results):
            if not result.analysis.bug_location or result.analysis.bug_location.strip() == "":
                validation_report['issues'].append(f"Result {i+1}: Missing bug location")
            
            if not result.analysis.root_cause or result.analysis.root_cause.strip() == "":
                validation_report['issues'].append(f"Result {i+1}: Missing root cause")
            
            if not result.analysis.fix_suggestion or result.analysis.fix_suggestion.strip() == "":
                validation_report['issues'].append(f"Result {i+1}: Missing fix suggestion")
            
            if len(result.analysis.supporting_evidence) == 0:
                validation_report['warnings'].append(f"Result {i+1}: No supporting evidence")
        
        # Check reasoning chain completeness
        incomplete_reasoning = 0
        for result in ranked_results:
            if len(result.analysis.reasoning_chain.steps) < 2:
                incomplete_reasoning += 1
        
        if incomplete_reasoning > 0:
            validation_report['warnings'].append(
                f"{incomplete_reasoning} results have incomplete reasoning chains"
            )
        
        # Calculate overall quality scores
        if ranked_results:
            avg_confidence = sum(r.analysis.confidence for r in ranked_results) / len(ranked_results)
            avg_quality = sum(r.quality_metrics.calculate_overall_quality() for r in ranked_results) / len(ranked_results)
            
            validation_report['quality_scores'] = {
                'average_confidence': avg_confidence,
                'average_quality': avg_quality,
                'top_result_confidence': ranked_results[0].analysis.confidence,
                'confidence_spread': max(r.analysis.confidence for r in ranked_results) - 
                                   min(r.analysis.confidence for r in ranked_results)
            }
        
        # Generate recommendations based on validation
        if validation_report['quality_scores'].get('average_confidence', 0) < 0.6:
            validation_report['recommendations'].append(
                "Consider additional analysis rounds to improve confidence"
            )
        
        if validation_report['quality_scores'].get('confidence_spread', 0) > 0.5:
            validation_report['recommendations'].append(
                "Large confidence spread detected - consider conflict resolution"
            )
        
        # Set overall validity
        if validation_report['issues']:
            validation_report['is_valid'] = False
        
        self.logger.info(f"Validation completed. Valid: {validation_report['is_valid']}, "
                        f"Issues: {len(validation_report['issues'])}, "
                        f"Warnings: {len(validation_report['warnings'])}")
        
        return validation_report
    
    def build_evidence_chain_display(self, analysis: AnalysisResult) -> str:
        """
        Build comprehensive evidence chain display for an analysis result.
        
        Args:
            analysis: Analysis result to build evidence chain for
            
        Returns:
            Formatted evidence chain string
        """
        if not analysis.evidence_chain:
            # Build evidence chain from reasoning steps and supporting evidence
            evidence_chain = EvidenceChain()
            
            # Add evidence from reasoning steps
            for step in analysis.reasoning_chain.steps:
                for evidence_item in step.evidence:
                    evidence_chain.add_evidence(
                        evidence=evidence_item,
                        source=f"Step {step.step_number}: {step.description}",
                        weight=step.confidence
                    )
            
            # Add supporting evidence
            for evidence_item in analysis.supporting_evidence:
                evidence_chain.add_evidence(
                    evidence=evidence_item,
                    source="Supporting Evidence",
                    weight=1.0
                )
            
            analysis.evidence_chain = evidence_chain
        
        return self._format_evidence_chain(analysis)
    
    # Private helper methods
    
    def _analyze_code_impact(self, analysis: AnalysisResult, context: ContextWindow) -> CodeImpactMetrics:
        """Analyze the code impact of implementing the suggested fix."""
        impact_metrics = CodeImpactMetrics()
        
        # Estimate lines affected based on fix suggestion complexity
        fix_text = analysis.fix_suggestion.lower()
        
        # Simple heuristics for impact estimation
        if any(keyword in fix_text for keyword in ['refactor', 'restructure', 'redesign']):
            impact_metrics.lines_affected = 50
            impact_metrics.functions_affected = 5
            impact_metrics.risk_score = 0.8
            impact_metrics.implementation_effort = 0.9
        elif any(keyword in fix_text for keyword in ['add method', 'new function', 'create class']):
            impact_metrics.lines_affected = 20
            impact_metrics.functions_affected = 2
            impact_metrics.risk_score = 0.4
            impact_metrics.implementation_effort = 0.6
        elif any(keyword in fix_text for keyword in ['change', 'modify', 'update']):
            impact_metrics.lines_affected = 10
            impact_metrics.functions_affected = 1
            impact_metrics.risk_score = 0.3
            impact_metrics.implementation_effort = 0.4
        else:
            # Simple fix
            impact_metrics.lines_affected = 5
            impact_metrics.functions_affected = 1
            impact_metrics.risk_score = 0.2
            impact_metrics.implementation_effort = 0.2
        
        # Adjust based on context complexity
        if context.class_hierarchy and len(context.class_hierarchy) > 5:
            impact_metrics.dependency_impact = 0.6
        elif context.module_dependencies and len(context.module_dependencies) > 3:
            impact_metrics.dependency_impact = 0.4
        else:
            impact_metrics.dependency_impact = 0.2
        
        return impact_metrics
    
    def _assess_quality(self, analysis: AnalysisResult) -> QualityMetrics:
        """Assess the quality of an analysis result."""
        quality_metrics = QualityMetrics()
        
        # Reasoning completeness based on reasoning chain
        if len(analysis.reasoning_chain.steps) >= 3:
            quality_metrics.reasoning_completeness = 0.9
        elif len(analysis.reasoning_chain.steps) >= 2:
            quality_metrics.reasoning_completeness = 0.7
        else:
            quality_metrics.reasoning_completeness = 0.4
        
        # Evidence strength based on amount and diversity of evidence
        evidence_count = len(analysis.supporting_evidence)
        if evidence_count >= 5:
            quality_metrics.evidence_strength = 0.9
        elif evidence_count >= 3:
            quality_metrics.evidence_strength = 0.7
        elif evidence_count >= 1:
            quality_metrics.evidence_strength = 0.5
        else:
            quality_metrics.evidence_strength = 0.2
        
        # Consistency score from reasoning chain confidence
        quality_metrics.consistency_score = analysis.reasoning_chain.overall_confidence
        
        # Specificity based on detail level
        specificity_indicators = [
            len(analysis.bug_location) > 20,
            len(analysis.root_cause) > 50,
            len(analysis.fix_suggestion) > 30,
            'line' in analysis.bug_location.lower(),
            'function' in analysis.bug_location.lower()
        ]
        quality_metrics.specificity_score = sum(specificity_indicators) / len(specificity_indicators)
        
        # Actionability based on fix suggestion detail
        actionability_indicators = [
            any(keyword in analysis.fix_suggestion.lower() 
                for keyword in ['change', 'add', 'remove', 'replace', 'modify']),
            len(analysis.fix_suggestion.split()) > 10,
            'line' in analysis.fix_suggestion.lower(),
            any(keyword in analysis.fix_suggestion.lower() 
                for keyword in ['should', 'need', 'must', 'replace'])
        ]
        quality_metrics.actionability_score = sum(actionability_indicators) / len(actionability_indicators)
        
        return quality_metrics
    
    def _calculate_composite_score(self, analysis: AnalysisResult,
                                 impact_metrics: CodeImpactMetrics,
                                 quality_metrics: QualityMetrics,
                                 criteria: List[SortingCriteria]) -> float:
        """Calculate composite score for ranking."""
        score = 0.0
        
        for criterion in criteria:
            weight = self.sorting_weights.get(criterion, 0.0)
            
            if criterion == SortingCriteria.CONFIDENCE:
                score += weight * analysis.confidence
            elif criterion == SortingCriteria.CODE_IMPACT:
                # Lower impact is better, so invert the score
                impact_score = 1.0 - impact_metrics.calculate_overall_impact()
                score += weight * impact_score
            elif criterion == SortingCriteria.EVIDENCE_STRENGTH:
                score += weight * quality_metrics.evidence_strength
            elif criterion == SortingCriteria.REASONING_QUALITY:
                score += weight * quality_metrics.calculate_overall_quality()
            elif criterion == SortingCriteria.IMPLEMENTATION_COMPLEXITY:
                # Lower complexity is better, so invert the score
                complexity_score = 1.0 - impact_metrics.implementation_effort
                score += weight * complexity_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_ranking_rationale(self, analysis: AnalysisResult,
                                  impact_metrics: CodeImpactMetrics,
                                  quality_metrics: QualityMetrics,
                                  composite_score: float) -> str:
        """Generate human-readable rationale for ranking."""
        rationale_parts = []
        
        # Confidence assessment
        if analysis.confidence >= 0.8:
            rationale_parts.append("High confidence analysis")
        elif analysis.confidence >= 0.6:
            rationale_parts.append("Moderate confidence analysis")
        else:
            rationale_parts.append("Lower confidence analysis")
        
        # Impact assessment
        overall_impact = impact_metrics.calculate_overall_impact()
        if overall_impact <= 0.3:
            rationale_parts.append("low implementation impact")
        elif overall_impact <= 0.6:
            rationale_parts.append("moderate implementation impact")
        else:
            rationale_parts.append("high implementation impact")
        
        # Quality assessment
        overall_quality = quality_metrics.calculate_overall_quality()
        if overall_quality >= 0.8:
            rationale_parts.append("excellent reasoning quality")
        elif overall_quality >= 0.6:
            rationale_parts.append("good reasoning quality")
        else:
            rationale_parts.append("basic reasoning quality")
        
        # Evidence assessment
        if quality_metrics.evidence_strength >= 0.8:
            rationale_parts.append("strong supporting evidence")
        elif quality_metrics.evidence_strength >= 0.5:
            rationale_parts.append("adequate supporting evidence")
        else:
            rationale_parts.append("limited supporting evidence")
        
        return f"Ranked based on: {', '.join(rationale_parts)} (composite score: {composite_score:.3f})"
    
    def _generate_summary(self, ranked_results: List[RankedResult],
                         issue_description: str, bug_type: BugType) -> str:
        """Generate summary of analysis results."""
        if not ranked_results:
            return "No analysis results available."
        
        top_result = ranked_results[0]
        
        summary_parts = [
            f"Analysis of {bug_type.category.value.replace('_', ' ')} issue completed.",
            f"Found {len(ranked_results)} potential solution(s).",
            f"Top recommendation: {top_result.analysis.bug_location}",
            f"with confidence {top_result.analysis.confidence:.2f}",
            f"and composite score {top_result.composite_score:.3f}."
        ]
        
        if len(ranked_results) > 1:
            confidence_range = (
                min(r.analysis.confidence for r in ranked_results),
                max(r.analysis.confidence for r in ranked_results)
            )
            summary_parts.append(
                f"Confidence range: {confidence_range[0]:.2f} - {confidence_range[1]:.2f}."
            )
        
        return " ".join(summary_parts)
    
    def _build_reasoning_process_description(self, ranked_results: List[RankedResult]) -> str:
        """Build description of the reasoning process."""
        if not ranked_results:
            return "No reasoning process available."
        
        process_parts = [
            "## Reasoning Process Overview",
            "",
            f"The analysis process evaluated {len(ranked_results)} candidate solution(s) using multiple criteria:",
            "",
            "### Evaluation Criteria:",
            "- **Confidence**: LLM confidence in the analysis",
            "- **Code Impact**: Estimated impact of implementing the fix",
            "- **Evidence Strength**: Quality and quantity of supporting evidence", 
            "- **Reasoning Quality**: Completeness and consistency of reasoning",
            "- **Implementation Complexity**: Difficulty of implementing the solution",
            "",
            "### Process Steps:",
            "1. Initial analysis and classification",
            "2. Multi-round reasoning and refinement",
            "3. Evidence collection and validation",
            "4. Impact assessment and quality evaluation",
            "5. Composite scoring and ranking",
            ""
        ]
        
        # Add details about top result's reasoning
        if ranked_results:
            top_result = ranked_results[0]
            process_parts.extend([
                f"### Top Result Reasoning ({len(top_result.analysis.reasoning_chain.steps)} steps):",
                ""
            ])
            
            for i, step in enumerate(top_result.analysis.reasoning_chain.steps):
                process_parts.append(
                    f"{i+1}. **{step.description}** (confidence: {step.confidence:.2f})"
                )
                if step.evidence:
                    process_parts.append(f"   - Evidence: {'; '.join(step.evidence[:2])}")
                process_parts.append("")
        
        return "\n".join(process_parts)
    
    def _format_evidence_chain(self, analysis: AnalysisResult) -> str:
        """Format evidence chain for display."""
        if not analysis.evidence_chain:
            return "No evidence chain available."
        
        evidence_parts = [
            "### Evidence Chain",
            ""
        ]
        
        # Group evidence by source
        evidence_by_source = {}
        for i, evidence in enumerate(analysis.evidence_chain.evidence_items):
            source = analysis.evidence_chain.source_locations[i]
            weight = analysis.evidence_chain.confidence_weights[i]
            
            if source not in evidence_by_source:
                evidence_by_source[source] = []
            evidence_by_source[source].append((evidence, weight))
        
        # Format evidence by source
        for source, evidence_list in evidence_by_source.items():
            evidence_parts.append(f"**{source}:**")
            for evidence, weight in evidence_list:
                evidence_parts.append(f"- {evidence} (weight: {weight:.2f})")
            evidence_parts.append("")
        
        # Add reasoning path if available
        if analysis.evidence_chain.reasoning_path:
            evidence_parts.extend([
                "**Reasoning Path:**",
                ""
            ])
            for step in analysis.evidence_chain.reasoning_path:
                evidence_parts.append(f"- {step}")
        
        return "\n".join(evidence_parts)
    
    def _generate_quality_assessment(self, ranked_results: List[RankedResult]) -> str:
        """Generate overall quality assessment."""
        if not ranked_results:
            return "No results to assess."
        
        # Calculate aggregate metrics
        avg_confidence = sum(r.analysis.confidence for r in ranked_results) / len(ranked_results)
        avg_quality = sum(r.quality_metrics.calculate_overall_quality() for r in ranked_results) / len(ranked_results)
        
        high_confidence_count = sum(1 for r in ranked_results if r.analysis.confidence >= 0.7)
        
        assessment_parts = [
            f"**Overall Quality Assessment:**",
            f"- Average confidence: {avg_confidence:.2f}",
            f"- Average quality score: {avg_quality:.2f}",
            f"- High confidence results: {high_confidence_count}/{len(ranked_results)}",
            ""
        ]
        
        # Quality interpretation
        if avg_confidence >= 0.8 and avg_quality >= 0.8:
            assessment_parts.append("**Assessment: Excellent** - High confidence and quality across results.")
        elif avg_confidence >= 0.6 and avg_quality >= 0.6:
            assessment_parts.append("**Assessment: Good** - Solid confidence and quality levels.")
        elif avg_confidence >= 0.4 or avg_quality >= 0.4:
            assessment_parts.append("**Assessment: Fair** - Moderate confidence or quality, consider additional analysis.")
        else:
            assessment_parts.append("**Assessment: Poor** - Low confidence and quality, manual review recommended.")
        
        return "\n".join(assessment_parts)
    
    def _generate_recommendations(self, ranked_results: List[RankedResult], 
                                bug_type: BugType) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if not ranked_results:
            recommendations.append("No analysis results available - manual investigation required")
            return recommendations
        
        top_result = ranked_results[0]
        
        # Confidence-based recommendations
        if top_result.analysis.confidence >= 0.8:
            recommendations.append("High confidence result - proceed with implementation")
        elif top_result.analysis.confidence >= 0.6:
            recommendations.append("Moderate confidence - consider additional validation before implementation")
        else:
            recommendations.append("Low confidence - perform manual review and additional analysis")
        
        # Impact-based recommendations
        impact_score = top_result.impact_metrics.calculate_overall_impact()
        if impact_score >= 0.7:
            recommendations.append("High impact change - thorough testing and staged rollout recommended")
        elif impact_score >= 0.4:
            recommendations.append("Moderate impact change - standard testing procedures apply")
        else:
            recommendations.append("Low impact change - minimal testing required")
        
        # Multiple candidates recommendations
        if len(ranked_results) > 1:
            confidence_spread = (max(r.analysis.confidence for r in ranked_results) - 
                               min(r.analysis.confidence for r in ranked_results))
            if confidence_spread > 0.3:
                recommendations.append("Multiple viable solutions with varying confidence - consider hybrid approach")
        
        # Bug type specific recommendations
        if bug_type.category.value == "logic_error":
            recommendations.append("Logic error detected - verify fix with comprehensive test cases")
        elif bug_type.category.value == "api_issue":
            recommendations.append("API issue detected - check for breaking changes and update documentation")
        elif bug_type.category.value == "performance":
            recommendations.append("Performance issue detected - benchmark before and after implementation")
        
        return recommendations