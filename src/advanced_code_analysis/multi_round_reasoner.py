"""
Multi-round reasoning engine for the Advanced Code Analysis system.

This module implements the MultiRoundReasoner class that executes multi-step
reasoning and verification, handles convergence strategies for ambiguous problems,
implements self-verification mechanisms, and provides conflict detection and
resolution capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .models import (
    AnalysisResult, VerificationResult, Conflict, ReasoningChain, 
    ReasoningStep, EvidenceChain, ResolvedAnalysis, ContextWindow,
    BugType, AnalysisStrategy, PromptTemplate
)
from .llm_interface import LLMInterface, LLMResponse
from .config import AdvancedAnalysisConfig


logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result of a single reasoning round."""
    round_number: int
    analysis: AnalysisResult
    confidence_change: float
    new_evidence: List[str] = field(default_factory=list)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    reasoning_quality: float = 0.0


@dataclass
class ConvergenceStrategy:
    """Strategy for multi-round reasoning convergence."""
    name: str
    confidence_threshold: float
    max_rounds: int
    improvement_threshold: float = 0.05  # Minimum improvement per round
    stability_rounds: int = 2  # Rounds with stable results before convergence


class MultiRoundReasoner:
    """
    Multi-round reasoning engine that executes iterative analysis and verification.
    
    This class implements:
    1. Multi-step reasoning with convergence strategies
    2. Self-verification mechanisms for consistency checking
    3. Conflict detection and resolution
    4. Evidence chain building and validation
    """
    
    def __init__(self, llm_interface: LLMInterface, config: AdvancedAnalysisConfig):
        """Initialize the multi-round reasoner."""
        self.llm = llm_interface
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize convergence strategies
        self.convergence_strategies = {
            "confidence_based": ConvergenceStrategy(
                name="confidence_based",
                confidence_threshold=config.analysis.confidence_threshold,
                max_rounds=config.analysis.max_reasoning_rounds,
                improvement_threshold=0.05,
                stability_rounds=2
            ),
            "evidence_based": ConvergenceStrategy(
                name="evidence_based", 
                confidence_threshold=0.8,
                max_rounds=config.analysis.max_reasoning_rounds,
                improvement_threshold=0.03,
                stability_rounds=3
            ),
            "conservative": ConvergenceStrategy(
                name="conservative",
                confidence_threshold=0.9,
                max_rounds=config.analysis.max_reasoning_rounds + 2,
                improvement_threshold=0.02,
                stability_rounds=3
            )
        }
        
        # Prompt templates for different reasoning phases
        self.prompt_templates = self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize prompt templates for different reasoning phases."""
        return {
            "initial_analysis": PromptTemplate(
                template_id="initial_analysis",
                content="""
Analyze the following code issue and provide your initial assessment:

Issue Description: {issue_description}
Code Context: {code_context}
Bug Type: {bug_type}

Please provide:
1. Your initial hypothesis about the root cause
2. The most likely location of the bug
3. A preliminary fix suggestion
4. Your confidence level (0.0-1.0)
5. Key evidence supporting your analysis

Format your response as JSON:
{{
    "hypothesis": "...",
    "bug_location": "...", 
    "fix_suggestion": "...",
    "confidence": 0.0,
    "evidence": ["...", "..."]
}}
""",
                placeholders=["issue_description", "code_context", "bug_type"]
            ),
            
            "verification": PromptTemplate(
                template_id="verification",
                content="""
Please verify the following analysis for internal consistency and accuracy:

Original Issue: {issue_description}
Analysis Result: {analysis_result}
Supporting Evidence: {evidence}

Check for:
1. Logical consistency in the reasoning
2. Whether the evidence supports the conclusion
3. Potential alternative explanations
4. Missing considerations or edge cases

Provide your verification as JSON:
{{
    "is_consistent": true/false,
    "consistency_score": 0.0,
    "issues_found": ["...", "..."],
    "alternative_explanations": ["...", "..."],
    "missing_considerations": ["...", "..."],
    "confidence_adjustment": 0.0
}}
""",
                placeholders=["issue_description", "analysis_result", "evidence"]
            ),
            
            "refinement": PromptTemplate(
                template_id="refinement", 
                content="""
Based on the previous analysis and verification feedback, refine your analysis:

Original Analysis: {previous_analysis}
Verification Feedback: {verification_feedback}
Additional Context: {additional_context}

Please provide a refined analysis addressing the feedback:

{{
    "refined_hypothesis": "...",
    "updated_location": "...",
    "improved_fix": "...",
    "confidence": 0.0,
    "new_evidence": ["...", "..."],
    "addressed_issues": ["...", "..."]
}}
""",
                placeholders=["previous_analysis", "verification_feedback", "additional_context"]
            ),
            
            "conflict_resolution": PromptTemplate(
                template_id="conflict_resolution",
                content="""
Resolve the following conflicting analyses:

Issue: {issue_description}
Conflicting Analyses:
{conflicting_analyses}

Please provide a resolution that:
1. Identifies the most likely correct analysis
2. Explains why other analyses are less likely
3. Synthesizes the best elements from each analysis

Resolution:
{{
    "chosen_analysis": "...",
    "reasoning": "...",
    "synthesized_elements": ["...", "..."],
    "confidence": 0.0,
    "resolution_method": "..."
}}
""",
                placeholders=["issue_description", "conflicting_analyses"]
            )
        }
    
    async def initial_analysis(self, issue: str, context: ContextWindow, 
                             bug_type: BugType, strategy: AnalysisStrategy) -> AnalysisResult:
        """
        Perform initial analysis of the issue.
        
        Args:
            issue: The issue description
            context: Code context window
            bug_type: Classified bug type
            strategy: Analysis strategy to use
            
        Returns:
            Initial analysis result
        """
        self.logger.info("Starting initial analysis")
        
        try:
            # Prepare context for LLM
            template_vars = {
                "issue_description": issue,
                "code_context": self._format_context(context),
                "bug_type": f"{bug_type.category.value} (confidence: {bug_type.confidence:.2f})"
            }
            
            # Generate initial analysis
            response = await self.llm.generate(
                self.prompt_templates["initial_analysis"],
                template_vars=template_vars,
                max_completion_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            analysis_data = self._parse_json_response(response.content)
            
            # Create reasoning chain
            reasoning_chain = ReasoningChain()
            initial_step = ReasoningStep(
                step_number=1,
                description="Initial analysis",
                input_data={"issue": issue, "bug_type": bug_type.category.value},
                output_data=analysis_data,
                confidence=analysis_data.get("confidence", 0.0),
                evidence=analysis_data.get("evidence", []),
                timestamp=datetime.now().isoformat()
            )
            reasoning_chain.add_step(initial_step)
            
            # Create analysis result
            result = AnalysisResult(
                bug_location=analysis_data.get("bug_location", "Unknown"),
                root_cause=analysis_data.get("hypothesis", "Unknown"),
                fix_suggestion=analysis_data.get("fix_suggestion", "No suggestion"),
                confidence=analysis_data.get("confidence", 0.0),
                reasoning_chain=reasoning_chain,
                supporting_evidence=analysis_data.get("evidence", [])
            )
            
            self.logger.info(f"Initial analysis completed with confidence: {result.confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in initial analysis: {e}")
            # Return a minimal result to allow the process to continue
            return self._create_fallback_analysis(issue, str(e))
    
    async def verify_analysis(self, result: AnalysisResult, issue: str) -> VerificationResult:
        """
        Verify analysis result for internal consistency.
        
        Args:
            result: Analysis result to verify
            issue: Original issue description
            
        Returns:
            Verification result with consistency check
        """
        self.logger.info("Starting analysis verification")
        
        try:
            # Prepare verification prompt
            template_vars = {
                "issue_description": issue,
                "analysis_result": self._format_analysis_result(result),
                "evidence": "; ".join(result.supporting_evidence)
            }
            
            # Generate verification
            response = await self.llm.generate(
                self.prompt_templates["verification"],
                template_vars=template_vars,
                max_completion_tokens=1500,
                temperature=0.0  # Use deterministic verification
            )
            
            # Parse verification response
            verification_data = self._parse_json_response(response.content)
            
            # Create conflicts if issues found
            conflicts = []
            if verification_data.get("issues_found"):
                conflict = Conflict(
                    conflict_type="consistency_issue",
                    description="Internal consistency issues detected",
                    conflicting_analyses=[result],
                    resolution_strategy="refinement_needed"
                )
                conflicts.append(conflict)
            
            # Create verification result
            verification_result = VerificationResult(
                is_consistent=verification_data.get("is_consistent", False),
                conflicts=conflicts,
                confidence_adjustment=verification_data.get("confidence_adjustment", 0.0),
                additional_evidence=verification_data.get("alternative_explanations", []),
                verification_notes=f"Consistency score: {verification_data.get('consistency_score', 0.0)}"
            )
            
            self.logger.info(f"Verification completed: consistent={verification_result.is_consistent}")
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Error in verification: {e}")
            return VerificationResult(
                is_consistent=False,
                verification_notes=f"Verification failed: {e}"
            )
    
    async def reason_multi_round(self, issue: str, context: ContextWindow,
                               strategy: AnalysisStrategy, 
                               initial_candidates: Optional[List[Any]] = None) -> List[RoundResult]:
        """
        Perform multi-round reasoning and return list of round results.
        
        Args:
            issue: Problem description
            context: Code context window
            strategy: Analysis strategy to use
            initial_candidates: Optional initial candidate matches
            
        Returns:
            List of RoundResult objects from each reasoning round
        """
        try:
            # Infer bug type from strategy or use default
            from .models import BugCategory
            bug_type = BugType(category=BugCategory.LOGIC_ERROR, confidence=0.5)
            
            # Perform multi-round reasoning
            resolved_analysis = await self.multi_round_reasoning(
                issue, context, bug_type, strategy
            )
            
            # Convert to RoundResult format
            round_results = []
            if resolved_analysis and resolved_analysis.final_result:
                round_result = RoundResult(
                    round_number=1,
                    analysis=resolved_analysis.final_result,
                    confidence_change=0.0,
                    new_evidence=[],
                    convergence_metrics={},
                    reasoning_quality=resolved_analysis.resolution_confidence
                )
                round_results.append(round_result)
            
            return round_results
            
        except Exception as e:
            self.logger.error(f"Multi-round reasoning failed: {e}")
            return []

    async def multi_round_reasoning(self, issue: str, context: ContextWindow,
                                  bug_type: BugType, strategy: AnalysisStrategy,
                                  convergence_strategy: str = "confidence_based") -> ResolvedAnalysis:
        """
        Execute multi-round reasoning with convergence strategy.
        
        Args:
            issue: Issue description
            context: Code context
            bug_type: Bug type classification
            strategy: Analysis strategy
            convergence_strategy: Strategy for convergence ("confidence_based", "evidence_based", "conservative")
            
        Returns:
            Resolved analysis after multi-round reasoning
        """
        self.logger.info(f"Starting multi-round reasoning with {convergence_strategy} strategy")
        
        conv_strategy = self.convergence_strategies.get(convergence_strategy, 
                                                      self.convergence_strategies["confidence_based"])
        
        # Track reasoning rounds
        rounds: List[RoundResult] = []
        current_analysis = await self.initial_analysis(issue, context, bug_type, strategy)
        
        # Initial round
        initial_round = RoundResult(
            round_number=1,
            analysis=current_analysis,
            confidence_change=current_analysis.confidence,
            convergence_metrics={"initial": True}
        )
        rounds.append(initial_round)
        
        # Multi-round refinement
        for round_num in range(2, conv_strategy.max_rounds + 1):
            self.logger.info(f"Starting reasoning round {round_num}")
            
            # Verify current analysis
            verification = await self.verify_analysis(current_analysis, issue)
            
            # Check if we need refinement
            if verification.is_consistent and len(verification.conflicts) == 0:
                # Check convergence
                if self._check_convergence(rounds, conv_strategy):
                    self.logger.info(f"Convergence achieved at round {round_num}")
                    break
            
            # Refine analysis
            refined_analysis = await self._refine_analysis(
                current_analysis, verification, issue, context
            )
            
            # Calculate metrics for this round
            confidence_change = refined_analysis.confidence - current_analysis.confidence
            round_result = RoundResult(
                round_number=round_num,
                analysis=refined_analysis,
                confidence_change=confidence_change,
                new_evidence=refined_analysis.supporting_evidence,
                convergence_metrics=self._calculate_convergence_metrics(
                    current_analysis, refined_analysis
                )
            )
            rounds.append(round_result)
            
            current_analysis = refined_analysis
            
            # Early termination if confidence is very high
            if current_analysis.confidence >= 0.95:
                self.logger.info(f"High confidence achieved at round {round_num}")
                break
        
        # Build final evidence chain
        evidence_chain = self._build_evidence_chain(rounds)
        
        # Create resolved analysis
        resolved_analysis = ResolvedAnalysis(
            final_result=current_analysis,
            resolution_method=f"multi_round_{convergence_strategy}",
            resolution_confidence=current_analysis.confidence,
            resolution_notes=f"Completed {len(rounds)} reasoning rounds"
        )
        
        self.logger.info(f"Multi-round reasoning completed with final confidence: {current_analysis.confidence:.2f}")
        return resolved_analysis
    
    async def resolve_conflicts(self, conflicts: List[Conflict], issue: str) -> ResolvedAnalysis:
        """
        Resolve conflicts between different analysis results.
        
        Args:
            conflicts: List of conflicts to resolve
            issue: Original issue description
            
        Returns:
            Resolved analysis after conflict resolution
        """
        self.logger.info(f"Resolving {len(conflicts)} conflicts")
        
        if not conflicts:
            raise ValueError("No conflicts to resolve")
        
        # Collect all conflicting analyses
        all_analyses = []
        for conflict in conflicts:
            all_analyses.extend(conflict.conflicting_analyses)
        
        if not all_analyses:
            raise ValueError("No analyses found in conflicts")
        
        try:
            # Format conflicting analyses for LLM
            formatted_analyses = []
            for i, analysis in enumerate(all_analyses):
                formatted_analyses.append(f"""
Analysis {i+1}:
- Location: {analysis.bug_location}
- Cause: {analysis.root_cause}
- Fix: {analysis.fix_suggestion}
- Confidence: {analysis.confidence:.2f}
- Evidence: {'; '.join(analysis.supporting_evidence)}
""")
            
            template_vars = {
                "issue_description": issue,
                "conflicting_analyses": "\n".join(formatted_analyses)
            }
            
            # Generate conflict resolution
            response = await self.llm.generate(
                self.prompt_templates["conflict_resolution"],
                template_vars=template_vars,
                max_completion_tokens=2000,
                temperature=0.1
            )
            
            # Parse resolution
            resolution_data = self._parse_json_response(response.content)
            
            # Find the chosen analysis or create a new one
            chosen_idx = self._find_chosen_analysis(resolution_data.get("chosen_analysis", ""), all_analyses)
            
            if chosen_idx is not None:
                base_analysis = all_analyses[chosen_idx]
            else:
                # Create new synthesized analysis
                base_analysis = all_analyses[0]  # Use first as base
            
            # Create resolved analysis
            resolved_result = AnalysisResult(
                bug_location=base_analysis.bug_location,
                root_cause=resolution_data.get("reasoning", base_analysis.root_cause),
                fix_suggestion=base_analysis.fix_suggestion,
                confidence=resolution_data.get("confidence", base_analysis.confidence),
                reasoning_chain=base_analysis.reasoning_chain,
                supporting_evidence=base_analysis.supporting_evidence + 
                                  resolution_data.get("synthesized_elements", [])
            )
            
            # Add resolution step to reasoning chain
            resolution_step = ReasoningStep(
                step_number=len(resolved_result.reasoning_chain.steps) + 1,
                description="Conflict resolution",
                input_data={"conflicts": len(conflicts)},
                output_data=resolution_data,
                confidence=resolution_data.get("confidence", 0.0),
                evidence=resolution_data.get("synthesized_elements", []),
                timestamp=datetime.now().isoformat()
            )
            resolved_result.reasoning_chain.add_step(resolution_step)
            
            resolved_analysis = ResolvedAnalysis(
                final_result=resolved_result,
                resolution_method=resolution_data.get("resolution_method", "llm_mediated"),
                discarded_alternatives=[a for i, a in enumerate(all_analyses) if i != chosen_idx],
                resolution_confidence=resolution_data.get("confidence", 0.0),
                resolution_notes=resolution_data.get("reasoning", "")
            )
            
            self.logger.info(f"Conflicts resolved with confidence: {resolved_analysis.resolution_confidence:.2f}")
            return resolved_analysis
            
        except Exception as e:
            self.logger.error(f"Error in conflict resolution: {e}")
            # Fallback: return highest confidence analysis
            best_analysis = max(all_analyses, key=lambda a: a.confidence)
            return ResolvedAnalysis(
                final_result=best_analysis,
                resolution_method="fallback_highest_confidence",
                discarded_alternatives=[a for a in all_analyses if a != best_analysis],
                resolution_confidence=best_analysis.confidence,
                resolution_notes=f"Fallback resolution due to error: {e}"
            )
    
    def build_evidence_chain(self, analysis: AnalysisResult) -> EvidenceChain:
        """
        Build comprehensive evidence chain for an analysis result.
        
        Args:
            analysis: Analysis result to build evidence for
            
        Returns:
            Evidence chain with reasoning path
        """
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
                source="Supporting evidence",
                weight=1.0
            )
        
        # Build reasoning path
        evidence_chain.reasoning_path = [
            f"Step {step.step_number}: {step.description} (confidence: {step.confidence:.2f})"
            for step in analysis.reasoning_chain.steps
        ]
        
        return evidence_chain
    
    # Helper methods
    
    def _format_context(self, context: ContextWindow) -> str:
        """Format context window for LLM consumption."""
        formatted = f"Target Code:\n{context.target_code}\n\n"
        
        if context.related_functions:
            formatted += f"Related Functions: {', '.join(context.related_functions[:5])}\n"
        
        if context.class_hierarchy:
            formatted += f"Class Hierarchy: {str(context.class_hierarchy)}\n"
        
        if context.module_dependencies:
            formatted += f"Dependencies: {', '.join(context.module_dependencies[:3])}\n"
        
        return formatted
    
    def _format_analysis_result(self, result: AnalysisResult) -> str:
        """Format analysis result for verification."""
        return f"""
Location: {result.bug_location}
Root Cause: {result.root_cause}
Fix Suggestion: {result.fix_suggestion}
Confidence: {result.confidence:.2f}
"""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with error handling."""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try parsing entire response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            # Return minimal structure
            return {
                "hypothesis": "Parse error",
                "bug_location": "Unknown",
                "fix_suggestion": "Unable to parse response",
                "confidence": 0.1,
                "evidence": [f"JSON parse error: {e}"]
            }
    
    def _create_fallback_analysis(self, issue: str, error: str) -> AnalysisResult:
        """Create fallback analysis when initial analysis fails."""
        reasoning_chain = ReasoningChain()
        fallback_step = ReasoningStep(
            step_number=1,
            description="Fallback analysis due to error",
            input_data={"issue": issue, "error": error},
            output_data={"fallback": True},
            confidence=0.1,
            evidence=[f"Analysis failed: {error}"],
            timestamp=datetime.now().isoformat()
        )
        reasoning_chain.add_step(fallback_step)
        
        return AnalysisResult(
            bug_location="Unknown - analysis failed",
            root_cause=f"Unable to analyze due to error: {error}",
            fix_suggestion="Manual investigation required",
            confidence=0.1,
            reasoning_chain=reasoning_chain,
            supporting_evidence=[f"Error: {error}"]
        )
    
    async def _refine_analysis(self, current_analysis: AnalysisResult, 
                             verification: VerificationResult,
                             issue: str, context: ContextWindow) -> AnalysisResult:
        """Refine analysis based on verification feedback."""
        try:
            template_vars = {
                "previous_analysis": self._format_analysis_result(current_analysis),
                "verification_feedback": verification.verification_notes,
                "additional_context": "; ".join(verification.additional_evidence)
            }
            
            response = await self.llm.generate(
                self.prompt_templates["refinement"],
                template_vars=template_vars,
                max_completion_tokens=2000,
                temperature=0.1
            )
            
            refinement_data = self._parse_json_response(response.content)
            
            # Create new reasoning step
            refinement_step = ReasoningStep(
                step_number=len(current_analysis.reasoning_chain.steps) + 1,
                description="Analysis refinement",
                input_data={"verification_feedback": verification.verification_notes},
                output_data=refinement_data,
                confidence=refinement_data.get("confidence", current_analysis.confidence),
                evidence=refinement_data.get("new_evidence", []),
                timestamp=datetime.now().isoformat()
            )
            
            # Update reasoning chain
            new_reasoning_chain = ReasoningChain(
                steps=current_analysis.reasoning_chain.steps + [refinement_step],
                confidence_scores=current_analysis.reasoning_chain.confidence_scores + [refinement_step.confidence],
                evidence_links=current_analysis.reasoning_chain.evidence_links,
                final_conclusion=refinement_data.get("refined_hypothesis", current_analysis.root_cause)
            )
            new_reasoning_chain.calculate_overall_confidence()
            
            # Create refined analysis
            refined_analysis = AnalysisResult(
                bug_location=refinement_data.get("updated_location", current_analysis.bug_location),
                root_cause=refinement_data.get("refined_hypothesis", current_analysis.root_cause),
                fix_suggestion=refinement_data.get("improved_fix", current_analysis.fix_suggestion),
                confidence=refinement_data.get("confidence", current_analysis.confidence),
                reasoning_chain=new_reasoning_chain,
                supporting_evidence=current_analysis.supporting_evidence + 
                                  refinement_data.get("new_evidence", [])
            )
            
            return refined_analysis
            
        except Exception as e:
            self.logger.error(f"Error in refinement: {e}")
            return current_analysis  # Return original if refinement fails
    
    def _check_convergence(self, rounds: List[RoundResult], 
                          strategy: ConvergenceStrategy) -> bool:
        """Check if reasoning has converged based on strategy."""
        if len(rounds) < 2:
            return False
        
        current_round = rounds[-1]
        
        # Check confidence threshold
        if current_round.analysis.confidence >= strategy.confidence_threshold:
            # Check stability over recent rounds
            if len(rounds) >= strategy.stability_rounds:
                recent_rounds = rounds[-strategy.stability_rounds:]
                confidence_variance = self._calculate_confidence_variance(recent_rounds)
                
                if confidence_variance < 0.01:  # Very stable
                    return True
        
        # Check improvement threshold
        if len(rounds) >= 3:
            recent_improvements = [r.confidence_change for r in rounds[-2:]]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            
            if avg_improvement < strategy.improvement_threshold:
                return True
        
        return False
    
    def _calculate_convergence_metrics(self, prev_analysis: AnalysisResult, 
                                     curr_analysis: AnalysisResult) -> Dict[str, float]:
        """Calculate convergence metrics between two analyses."""
        metrics = {}
        
        # Confidence change
        metrics["confidence_change"] = curr_analysis.confidence - prev_analysis.confidence
        
        # Location stability (simple string comparison)
        metrics["location_stability"] = 1.0 if prev_analysis.bug_location == curr_analysis.bug_location else 0.0
        
        # Evidence growth
        prev_evidence_count = len(prev_analysis.supporting_evidence)
        curr_evidence_count = len(curr_analysis.supporting_evidence)
        metrics["evidence_growth"] = curr_evidence_count - prev_evidence_count
        
        return metrics
    
    def _calculate_confidence_variance(self, rounds: List[RoundResult]) -> float:
        """Calculate variance in confidence across rounds."""
        confidences = [r.analysis.confidence for r in rounds]
        if len(confidences) < 2:
            return 0.0
        
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        return variance
    
    def _build_evidence_chain(self, rounds: List[RoundResult]) -> EvidenceChain:
        """Build evidence chain from reasoning rounds."""
        evidence_chain = EvidenceChain()
        
        for round_result in rounds:
            analysis = round_result.analysis
            
            # Add evidence from this round
            for evidence_item in analysis.supporting_evidence:
                evidence_chain.add_evidence(
                    evidence=evidence_item,
                    source=f"Round {round_result.round_number}",
                    weight=analysis.confidence
                )
        
        # Build reasoning path
        evidence_chain.reasoning_path = [
            f"Round {r.round_number}: confidence={r.analysis.confidence:.2f}, "
            f"change={r.confidence_change:+.2f}"
            for r in rounds
        ]
        
        return evidence_chain
    
    def _find_chosen_analysis(self, chosen_description: str, 
                            analyses: List[AnalysisResult]) -> Optional[int]:
        """Find which analysis was chosen based on description."""
        # Simple matching based on keywords in the description
        chosen_lower = chosen_description.lower()
        
        for i, analysis in enumerate(analyses):
            # Check if analysis location or cause appears in chosen description
            if (analysis.bug_location.lower() in chosen_lower or 
                analysis.root_cause.lower() in chosen_lower):
                return i
        
        return None  # No clear match found