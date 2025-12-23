"""
Advanced Code Analyzer - Main Integration Class

This module implements the main AdvancedCodeAnalyzer class that integrates all
analysis engines into a complete system for intelligent code analysis and bug detection.
It provides the complete analysis flow from problem classification to final output,
with error handling, retry mechanisms, and performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    AnalysisResult, BugType, ContextWindow, ReasoningChain, 
    VerificationResult, ResolvedAnalysis, EvidenceChain
)
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface, LLMError, LLMTimeoutError, LLMRateLimitError
from .bug_classifier import BugClassifier, ClassificationResult
from .semantic_extractor import SemanticExtractor, ExtractionResult
from .context_enhancer import ContextEnhancer
from .concept_mapper import ConceptMapper, ConceptMappingResult
from .pattern_matcher import PatternMatcher
from .multi_round_reasoner import MultiRoundReasoner, RoundResult
from .enhanced_ast_analyzer import EnhancedASTAnalyzer
from .conflict_detector import ConflictDetector, ConflictMarker
from .result_sorter import ResultSorter


logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """Request for code analysis with all necessary information."""
    issue_text: str
    target_files: List[str]
    code_context: Optional[str] = None
    focus_elements: Optional[List[str]] = None
    analysis_options: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high, critical


@dataclass
class AnalysisSession:
    """Analysis session tracking state and progress."""
    session_id: str
    request: AnalysisRequest
    start_time: float
    current_stage: str = "initialization"
    progress: float = 0.0
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveAnalysisResult:
    """Complete analysis result with all components and metadata."""
    # Core results
    primary_analysis: AnalysisResult
    alternative_solutions: List[AnalysisResult] = field(default_factory=list)
    
    # Analysis components
    bug_classification: Optional[ClassificationResult] = None
    semantic_extraction: Optional[ExtractionResult] = None
    concept_mapping: Optional[ConceptMappingResult] = None
    pattern_matches: Optional[List[Any]] = None
    reasoning_rounds: List[RoundResult] = field(default_factory=list)
    
    # Quality and validation
    verification_result: Optional[VerificationResult] = None
    conflicts_detected: List[ConflictMarker] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    session_info: Optional[AnalysisSession] = None
    processing_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    
    def get_best_solution(self) -> AnalysisResult:
        """Get the highest confidence solution."""
        all_solutions = [self.primary_analysis] + self.alternative_solutions
        return max(all_solutions, key=lambda x: x.confidence)


class AdvancedCodeAnalyzer:
    """
    Main Advanced Code Analyzer that integrates all analysis engines.
    
    This class orchestrates the complete analysis workflow:
    1. Problem classification and information extraction
    2. Context enhancement and concept mapping
    3. Pattern matching and multi-round reasoning
    4. Conflict detection and resolution
    5. Result sorting and output generation
    """
    
    def __init__(self, config: Optional[AdvancedAnalysisConfig] = None):
        """Initialize the Advanced Code Analyzer."""
        self.config = config or AdvancedAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM interface
        self.llm_interface = LLMInterface(self.config.llm)
        
        # Initialize all analysis engines
        self._initialize_engines()
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_analyses = 0
        self.average_processing_time = 0.0
        
        # Session tracking
        self.active_sessions: Dict[str, AnalysisSession] = {}
        
        self.logger.info("AdvancedCodeAnalyzer initialized successfully")
    
    def _initialize_engines(self) -> None:
        """Initialize all analysis engines with proper dependencies."""
        try:
            # Core engines
            self.bug_classifier = BugClassifier(self.config, self.llm_interface)
            self.semantic_extractor = SemanticExtractor(self.config, self.llm_interface)
            self.context_enhancer = ContextEnhancer(self.config)
            
            # Advanced engines
            self.concept_mapper = ConceptMapper(
                self.config, self.llm_interface, self.context_enhancer
            )
            self.pattern_matcher = PatternMatcher(self.config, self.llm_interface)
            self.multi_round_reasoner = MultiRoundReasoner(self.llm_interface, self.config)
            
            # Analysis and validation engines
            self.ast_analyzer = EnhancedASTAnalyzer(self.llm_interface)
            self.conflict_detector = ConflictDetector(
                self.llm_interface, self.config, self.ast_analyzer, self.multi_round_reasoner
            )
            self.result_sorter = ResultSorter(self.config)
            
            self.logger.info("All analysis engines initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analysis engines: {e}")
            raise
    
    async def analyze(self, issue_text: str, target_files: List[str],
                     code_context: Optional[str] = None,
                     focus_elements: Optional[List[str]] = None,
                     **options) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive code analysis.
        
        Args:
            issue_text: Description of the problem to analyze
            target_files: List of files to analyze
            code_context: Optional additional code context
            focus_elements: Optional list of specific elements to focus on
            **options: Additional analysis options
            
        Returns:
            ComprehensiveAnalysisResult containing complete analysis
        """
        # Create analysis request and session
        request = AnalysisRequest(
            issue_text=issue_text,
            target_files=target_files,
            code_context=code_context,
            focus_elements=focus_elements,
            analysis_options=options
        )
        
        session = AnalysisSession(
            session_id=f"analysis_{int(time.time())}_{len(self.active_sessions)}",
            request=request,
            start_time=time.time()
        )
        
        self.active_sessions[session.session_id] = session
        
        try:
            return await self._execute_analysis_pipeline(session)
        except Exception as e:
            self.logger.error(f"Analysis failed for session {session.session_id}: {e}")
            session.errors.append(str(e))
            raise
        finally:
            # Clean up session
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    async def _execute_analysis_pipeline(self, session: AnalysisSession) -> ComprehensiveAnalysisResult:
        """Execute the complete analysis pipeline."""
        self.logger.info(f"Starting analysis pipeline for session {session.session_id}")
        
        result = ComprehensiveAnalysisResult(
            primary_analysis=AnalysisResult(
                bug_location="",
                root_cause="",
                fix_suggestion="",
                confidence=0.0,
                reasoning_chain=ReasoningChain()
            ),
            session_info=session
        )
        
        try:
            # Stage 1: Problem Classification and Information Extraction
            session.current_stage = "classification_and_extraction"
            session.progress = 0.1
            
            classification_result, extraction_result = await self._classify_and_extract(session)
            result.bug_classification = classification_result
            result.semantic_extraction = extraction_result
            
            # Stage 2: Context Enhancement
            session.current_stage = "context_enhancement"
            session.progress = 0.2
            
            context_window = await self._enhance_context(session, extraction_result)
            session.intermediate_results['context_window'] = context_window
            
            # Stage 3: Concept Mapping
            session.current_stage = "concept_mapping"
            session.progress = 0.3
            
            mapping_result = await self._perform_concept_mapping(
                session, classification_result, extraction_result, context_window
            )
            result.concept_mapping = mapping_result
            
            # Stage 4: Pattern Matching
            session.current_stage = "pattern_matching"
            session.progress = 0.4
            
            pattern_matches = await self._match_patterns(
                session, classification_result.bug_type, context_window
            )
            result.pattern_matches = pattern_matches
            
            # Stage 5: Multi-round Reasoning
            session.current_stage = "multi_round_reasoning"
            session.progress = 0.5
            
            reasoning_results = await self._perform_multi_round_reasoning(
                session, classification_result, context_window, mapping_result
            )
            result.reasoning_rounds = reasoning_results
            result.primary_analysis = reasoning_results[-1].analysis if reasoning_results else result.primary_analysis
            
            # Stage 6: AST Analysis and Validation
            session.current_stage = "ast_analysis"
            session.progress = 0.7
            
            await self._perform_ast_analysis(session, result)
            
            # Stage 7: Conflict Detection and Resolution
            session.current_stage = "conflict_detection"
            session.progress = 0.8
            
            conflicts = await self._detect_and_resolve_conflicts(session, result)
            result.conflicts_detected = conflicts
            
            # Stage 8: Result Sorting and Quality Assessment
            session.current_stage = "result_sorting"
            session.progress = 0.9
            
            await self._sort_and_assess_results(session, result)
            
            # Stage 9: Final Validation and Output
            session.current_stage = "final_validation"
            session.progress = 1.0
            
            await self._final_validation(session, result)
            
            # Update performance metrics
            result.processing_time = time.time() - session.start_time
            self._update_performance_metrics(result)
            
            self.logger.info(f"Analysis pipeline completed for session {session.session_id} "
                           f"in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            session.errors.append(f"Pipeline execution failed: {str(e)}")
            raise
    
    async def _classify_and_extract(self, session: AnalysisSession) -> tuple[ClassificationResult, ExtractionResult]:
        """Perform bug classification and semantic information extraction."""
        try:
            # Run classification and extraction in parallel for efficiency
            classification_task = self.bug_classifier.classify_bug_type(
                session.request.issue_text, 
                session.request.code_context
            )
            
            extraction_task = self.semantic_extractor.extract_information(
                session.request.issue_text
            )
            
            classification_result, extraction_result = await asyncio.gather(
                classification_task, extraction_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(classification_result, Exception):
                self.logger.error(f"Classification failed: {classification_result}")
                session.errors.append(f"Classification failed: {str(classification_result)}")
                # Create fallback classification
                from .models import BugCategory
                classification_result = ClassificationResult(
                    bug_type=BugType(category=BugCategory.LOGIC_ERROR, confidence=0.1),
                    reasoning="Classification failed, using fallback"
                )
            
            if isinstance(extraction_result, Exception):
                self.logger.error(f"Extraction failed: {extraction_result}")
                session.errors.append(f"Extraction failed: {str(extraction_result)}")
                # Create fallback extraction
                from .semantic_extractor import StructuredSummary
                extraction_result = ExtractionResult(
                    extracted_items=[],
                    structured_summary=StructuredSummary(
                        problem_type="unknown",
                        key_components=[],
                        technical_concepts=[],
                        code_elements=[],
                        error_indicators=[],
                        confidence_score=0.0
                    ),
                    reasoning_chain=ReasoningChain(),
                    overall_confidence=0.0,
                    processing_time=0.0
                )
            
            return classification_result, extraction_result
            
        except Exception as e:
            self.logger.error(f"Classification and extraction failed: {e}")
            raise
    
    async def _enhance_context(self, session: AnalysisSession, 
                              extraction_result: ExtractionResult) -> ContextWindow:
        """Enhance code context for analysis."""
        try:
            # Extract focus elements from semantic extraction
            focus_elements = session.request.focus_elements or []
            
            # Add extracted function and class names as focus elements
            for item in extraction_result.extracted_items:
                if item.info_type.value in ['function_name', 'class_name', 'variable_name']:
                    focus_elements.append(item.content)
            
            # Collect context
            context_window = self.context_enhancer.collect_code_context(
                session.request.target_files,
                focus_elements
            )
            
            # Optimize context window for token limits
            max_completion_tokens = self.config.analysis.max_context_tokens
            if context_window.token_count > max_completion_tokens:
                context_window = self.context_enhancer.optimize_context_window(
                    context_window, max_completion_tokens
                )
            
            return context_window
            
        except Exception as e:
            self.logger.error(f"Context enhancement failed: {e}")
            # Return minimal context window
            return ContextWindow(target_code="", token_count=0)
    
    async def _perform_concept_mapping(self, session: AnalysisSession,
                                     classification_result: ClassificationResult,
                                     extraction_result: ExtractionResult,
                                     context_window: ContextWindow) -> ConceptMappingResult:
        """Perform concept mapping to locate relevant code."""
        try:
            # Extract search terms from semantic extraction
            search_terms = []
            for item in extraction_result.extracted_items:
                if item.confidence > 0.5:  # Only use high-confidence items
                    search_terms.append(item.content)
            
            # Perform concept mapping
            mapping_result = await self.concept_mapper.map_concepts_to_code(
                session.request.issue_text,
                search_terms,
                context_window,
                classification_result.bug_type
            )
            
            return mapping_result
            
        except Exception as e:
            self.logger.error(f"Concept mapping failed: {e}")
            # Return empty mapping result
            from .concept_mapper import ConceptMappingResult
            return ConceptMappingResult()
    
    async def _match_patterns(self, session: AnalysisSession,
                            bug_type: BugType,
                            context_window: ContextWindow) -> List[Any]:
        """Match predefined bug patterns."""
        try:
            pattern_matches = await self.pattern_matcher.match_patterns(
                session.request.issue_text,
                context_window,
                bug_type
            )
            
            return pattern_matches
            
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {e}")
            return []
    
    async def _perform_multi_round_reasoning(self, session: AnalysisSession,
                                           classification_result: ClassificationResult,
                                           context_window: ContextWindow,
                                           mapping_result: ConceptMappingResult) -> List[RoundResult]:
        """Perform multi-round reasoning for analysis refinement."""
        try:
            # Select analysis strategy based on classification
            strategy = await self.bug_classifier.select_analysis_strategy(
                classification_result.bug_type
            )
            
            # Perform multi-round reasoning
            reasoning_results = await self.multi_round_reasoner.reason_multi_round(
                session.request.issue_text,
                context_window,
                strategy,
                initial_candidates=mapping_result.primary_matches
            )
            
            return reasoning_results
            
        except Exception as e:
            self.logger.error(f"Multi-round reasoning failed: {e}")
            return []
    
    async def _perform_ast_analysis(self, session: AnalysisSession,
                                  result: ComprehensiveAnalysisResult) -> None:
        """Perform enhanced AST analysis on identified regions."""
        try:
            if not result.primary_analysis.bug_location:
                return
            
            # Identify suspicious regions from LLM analysis
            suspicious_regions = await self.ast_analyzer.identify_suspicious_regions(
                result.primary_analysis.bug_location,
                session.request.target_files
            )
            
            # Get the code content for AST analysis
            code_content = ""
            if session.request.target_files:
                try:
                    with open(session.request.target_files[0], 'r', encoding='utf-8') as f:
                        code_content = f.read()
                except (IOError, UnicodeDecodeError):
                    code_content = ""
            
            # Perform targeted AST analysis
            ast_results = self.ast_analyzer.analyze_suspicious_regions(
                code_content,
                suspicious_regions,
                session.intermediate_results.get('context_window')
            )
            
            # Update analysis with AST findings
            if ast_results:
                result.primary_analysis.supporting_evidence.extend(
                    [f"AST Analysis: {finding}" for finding in ast_results]
                )
            
        except Exception as e:
            self.logger.error(f"AST analysis failed: {e}")
            session.warnings.append(f"AST analysis failed: {str(e)}")
    
    async def _detect_and_resolve_conflicts(self, session: AnalysisSession,
                                          result: ComprehensiveAnalysisResult) -> List[ConflictMarker]:
        """Detect and resolve conflicts between different analysis results."""
        try:
            # Collect all analysis results for conflict detection
            all_analyses = [result.primary_analysis] + result.alternative_solutions
            
            # Detect conflicts
            conflicts = await self.conflict_detector.detect_conflicts(
                all_analyses,
                result.verification_result
            )
            
            # Resolve conflicts if any are found
            if conflicts:
                resolved_analysis = await self.conflict_detector.resolve_conflicts(
                    conflicts,
                    session.intermediate_results.get('context_window')
                )
                
                if resolved_analysis:
                    result.primary_analysis = resolved_analysis.final_result
                    result.alternative_solutions.extend(resolved_analysis.discarded_alternatives)
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}")
            return []
    
    async def _sort_and_assess_results(self, session: AnalysisSession,
                                     result: ComprehensiveAnalysisResult) -> None:
        """Sort results and assess quality metrics."""
        try:
            # Collect all candidate solutions
            all_candidates = [result.primary_analysis] + result.alternative_solutions
            
            # Sort and rank candidates
            sorted_results = await self.result_sorter.sort_and_rank_candidates(
                all_candidates,
                session.intermediate_results.get('context_window')
            )
            
            # Update result with sorted candidates
            if sorted_results.ranked_candidates:
                result.primary_analysis = sorted_results.ranked_candidates[0]
                result.alternative_solutions = sorted_results.ranked_candidates[1:]
            
            # Store quality metrics
            result.quality_metrics = sorted_results.quality_assessment
            
        except Exception as e:
            self.logger.error(f"Result sorting failed: {e}")
            session.warnings.append(f"Result sorting failed: {str(e)}")
    
    async def _final_validation(self, session: AnalysisSession,
                              result: ComprehensiveAnalysisResult) -> None:
        """Perform final validation and quality checks."""
        try:
            # Validate that we have a meaningful result
            if not result.primary_analysis.bug_location or not result.primary_analysis.fix_suggestion:
                session.warnings.append("Analysis did not produce specific location or fix suggestion")
            
            # Check confidence thresholds
            if result.primary_analysis.confidence < self.config.analysis.confidence_threshold:
                session.warnings.append(
                    f"Analysis confidence ({result.primary_analysis.confidence:.2f}) "
                    f"below threshold ({self.config.analysis.confidence_threshold})"
                )
            
            # Validate reasoning chain completeness
            if len(result.primary_analysis.reasoning_chain.steps) == 0:
                session.warnings.append("No reasoning steps recorded")
            
            # Count total LLM calls and tokens
            result.total_llm_calls = self._count_llm_calls(result)
            result.total_tokens_used = self._estimate_total_tokens(result)
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {e}")
            session.warnings.append(f"Final validation failed: {str(e)}")
    
    def _count_llm_calls(self, result: ComprehensiveAnalysisResult) -> int:
        """Count total LLM calls made during analysis."""
        # This is a simplified count - in a real implementation,
        # we would track this throughout the pipeline
        call_count = 0
        
        if result.bug_classification:
            call_count += 1  # Classification call
        
        if result.semantic_extraction:
            call_count += 2  # Extraction + refinement calls
        
        if result.reasoning_rounds:
            call_count += len(result.reasoning_rounds)  # One call per round
        
        return call_count
    
    def _estimate_total_tokens(self, result: ComprehensiveAnalysisResult) -> int:
        """Estimate total tokens used during analysis."""
        # Simplified estimation based on text lengths
        total_tokens = 0
        
        if result.session_info:
            # Input tokens
            input_text = result.session_info.request.issue_text
            if result.session_info.request.code_context:
                input_text += result.session_info.request.code_context
            total_tokens += len(input_text) // 4  # Rough approximation
            
            # Output tokens (reasoning chains, analysis results)
            if result.primary_analysis.reasoning_chain:
                for step in result.primary_analysis.reasoning_chain.steps:
                    total_tokens += len(step.description) // 4
        
        return total_tokens
    
    def _update_performance_metrics(self, result: ComprehensiveAnalysisResult) -> None:
        """Update performance tracking metrics."""
        self.total_analyses += 1
        
        if result.primary_analysis.confidence > self.config.analysis.confidence_threshold:
            self.successful_analyses += 1
        
        # Update average processing time
        if self.total_analyses == 1:
            self.average_processing_time = result.processing_time
        else:
            self.average_processing_time = (
                (self.average_processing_time * (self.total_analyses - 1) + result.processing_time) 
                / self.total_analyses
            )
    
    async def test_connection(self) -> bool:
        """Test connection to LLM provider."""
        try:
            return await self.llm_interface.test_connection()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        success_rate = (self.successful_analyses / self.total_analyses 
                       if self.total_analyses > 0 else 0.0)
        
        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate": success_rate,
            "average_processing_time": self.average_processing_time,
            "active_sessions": len(self.active_sessions)
        }
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active analysis session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "current_stage": session.current_stage,
            "progress": session.progress,
            "elapsed_time": time.time() - session.start_time,
            "errors": session.errors,
            "warnings": session.warnings
        }
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active analysis session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Cancelled analysis session {session_id}")
            return True
        return False
    
    def validate_configuration(self) -> List[str]:
        """Validate system configuration and return any issues."""
        issues = []
        
        # Validate LLM configuration
        llm_issues = self.llm_interface.validate_config()
        issues.extend(llm_issues)
        
        # Validate analysis configuration
        config_issues = self.config.validate()
        issues.extend(config_issues)
        
        # Check if all engines are properly initialized
        required_engines = [
            'bug_classifier', 'semantic_extractor', 'context_enhancer',
            'concept_mapper', 'pattern_matcher', 'multi_round_reasoner',
            'ast_analyzer', 'conflict_detector', 'result_sorter'
        ]
        
        for engine_name in required_engines:
            if not hasattr(self, engine_name) or getattr(self, engine_name) is None:
                issues.append(f"Engine not initialized: {engine_name}")
        
        return issues