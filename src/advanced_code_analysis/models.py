"""
Core data models for the Advanced Code Analysis system.

This module defines all the data structures used throughout the system,
including bug types, analysis strategies, context windows, reasoning chains,
and analysis results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class BugCategory(Enum):
    """Enumeration of bug categories for classification."""
    LOGIC_ERROR = "logic_error"
    API_ISSUE = "api_issue"
    PERFORMANCE = "performance"
    BOUNDARY_CONDITION = "boundary_condition"
    TYPE_ERROR = "type_error"
    CONCURRENCY = "concurrency"
    RESOURCE_MANAGEMENT = "resource_management"
    CONFIGURATION = "configuration"


@dataclass
class BugType:
    """Represents a classified bug type with confidence and characteristics."""
    category: BugCategory
    subcategory: Optional[str] = None
    confidence: float = 0.0
    characteristics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class PromptTemplate:
    """Template for LLM prompts with placeholders."""
    template_id: str
    content: str
    placeholders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisStrategy:
    """Strategy for analyzing specific bug types."""
    strategy_name: str
    prompt_template: PromptTemplate
    context_requirements: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    max_rounds: int = 3
    confidence_threshold: float = 0.8


@dataclass
class DependencyContext:
    """Context information about code dependencies."""
    function_signatures: Dict[str, str] = field(default_factory=dict)
    class_methods: Dict[str, List[str]] = field(default_factory=dict)
    import_statements: List[str] = field(default_factory=list)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DomainKnowledge:
    """Domain-specific knowledge and terminology."""
    domain_name: str
    terminology: Dict[str, str] = field(default_factory=dict)
    common_patterns: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)


@dataclass
class ContextWindow:
    """Code context information window for LLM analysis."""
    target_code: str
    related_functions: List[str] = field(default_factory=list)
    class_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    module_dependencies: List[str] = field(default_factory=list)
    domain_concepts: List[str] = field(default_factory=list)
    dependency_context: Optional[DependencyContext] = None
    domain_knowledge: Optional[DomainKnowledge] = None
    token_count: int = 0
    
    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count == 0:
            # Simple approximation: 1 token ≈ 4 characters
            total_text = self.target_code + " ".join(self.related_functions)
            self.token_count = len(total_text) // 4


@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain."""
    step_number: int
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None


@dataclass
class ReasoningChain:
    """Complete chain of reasoning steps with evidence."""
    steps: List[ReasoningStep] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    evidence_links: List[str] = field(default_factory=list)
    final_conclusion: str = ""
    overall_confidence: float = 0.0
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the chain."""
        self.steps.append(step)
        self.confidence_scores.append(step.confidence)
        
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence based on individual steps."""
        if not self.confidence_scores:
            return 0.0
        # Use geometric mean for conservative confidence estimation, but handle zero scores
        non_zero_scores = [score for score in self.confidence_scores if score > 0.0]
        if not non_zero_scores:
            return 0.0
        
        product = 1.0
        for score in non_zero_scores:
            product *= score
        self.overall_confidence = product ** (1.0 / len(non_zero_scores))
        return self.overall_confidence


@dataclass
class EvidenceChain:
    """Chain of evidence supporting an analysis conclusion."""
    evidence_items: List[str] = field(default_factory=list)
    source_locations: List[str] = field(default_factory=list)
    confidence_weights: List[float] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    
    def add_evidence(self, evidence: str, source: str, weight: float = 1.0) -> None:
        """Add evidence item with source and weight."""
        self.evidence_items.append(evidence)
        self.source_locations.append(source)
        self.confidence_weights.append(weight)


@dataclass
class AnalysisResult:
    """Result of code analysis with location, cause, and fix suggestion."""
    bug_location: str
    root_cause: str
    fix_suggestion: str
    confidence: float
    reasoning_chain: ReasoningChain
    supporting_evidence: List[str] = field(default_factory=list)
    evidence_chain: Optional[EvidenceChain] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class Conflict:
    """Represents a conflict between different analysis results."""
    conflict_type: str
    description: str
    conflicting_analyses: List[AnalysisResult] = field(default_factory=list)
    resolution_strategy: str = ""
    severity: str = "medium"  # low, medium, high
    
    def add_analysis(self, analysis: AnalysisResult) -> None:
        """Add a conflicting analysis result."""
        self.conflicting_analyses.append(analysis)


@dataclass
class VerificationResult:
    """Result of verifying an analysis for consistency."""
    is_consistent: bool
    conflicts: List[Conflict] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    additional_evidence: List[str] = field(default_factory=list)
    verification_notes: str = ""
    
    def add_conflict(self, conflict: Conflict) -> None:
        """Add a detected conflict."""
        self.conflicts.append(conflict)
        self.is_consistent = False


@dataclass
class BugPattern:
    """Learned pattern from successful bug fixes."""
    pattern_id: str
    problem_signature: str
    code_pattern: str
    fix_pattern: str
    success_rate: float
    applicable_domains: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        """Validate success rate."""
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("Success rate must be between 0.0 and 1.0")


@dataclass
class PatternGuidance:
    """Guidance based on matched bug patterns."""
    matched_patterns: List[BugPattern] = field(default_factory=list)
    confidence: float = 0.0
    suggested_approach: str = ""
    relevant_context: List[str] = field(default_factory=list)


@dataclass
class ResolvedAnalysis:
    """Analysis result after conflict resolution."""
    final_result: AnalysisResult
    resolution_method: str
    discarded_alternatives: List[AnalysisResult] = field(default_factory=list)
    resolution_confidence: float = 0.0
    resolution_notes: str = ""


@dataclass
class ClassificationFeedback:
    """Feedback for improving bug classification."""
    original_classification: BugType
    correct_classification: BugType
    issue_text: str
    feedback_notes: str = ""
    timestamp: Optional[str] = None