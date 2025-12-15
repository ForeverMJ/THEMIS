"""
Advanced Code Analysis and Semantic Understanding System

This module provides LLM-driven intelligent semantic understanding to enhance
the existing Enhanced GraphManager system. It focuses on solving core problems
identified in system analysis: inability to understand complex technical requirements,
inability to extract key information from problem descriptions, and inability to
accurately map technical concepts to code components.
"""

from .models import (
    BugType,
    BugCategory,
    AnalysisStrategy,
    ContextWindow,
    ReasoningChain,
    AnalysisResult,
    VerificationResult,
    Conflict,
    BugPattern,
    DependencyContext,
    DomainKnowledge,
    PatternGuidance,
    ResolvedAnalysis,
    EvidenceChain,
    ReasoningStep,
    ClassificationFeedback,
    PromptTemplate
)

from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface
from .bug_classifier import BugClassifier, ClassificationResult, PromptTemplateLibrary, AnalysisStrategyLibrary
from .semantic_extractor import (
    SemanticExtractor, 
    ExtractedInformation, 
    StructuredSummary, 
    ExtractionResult, 
    InformationType
)
from .context_enhancer import ContextEnhancer, CodeContext, ProjectStructure
from .multi_round_reasoner import MultiRoundReasoner, RoundResult, ConvergenceStrategy

__version__ = "1.0.0"
__all__ = [
    "BugType",
    "BugCategory",
    "AnalysisStrategy", 
    "ContextWindow",
    "ReasoningChain",
    "AnalysisResult",
    "VerificationResult",
    "Conflict",
    "BugPattern",
    "DependencyContext",
    "DomainKnowledge",
    "PatternGuidance",
    "ResolvedAnalysis",
    "EvidenceChain",
    "ReasoningStep",
    "ClassificationFeedback",
    "PromptTemplate",
    "AdvancedAnalysisConfig",
    "LLMInterface",
    "BugClassifier",
    "ClassificationResult",
    "PromptTemplateLibrary",
    "AnalysisStrategyLibrary",
    "SemanticExtractor",
    "ExtractedInformation",
    "StructuredSummary",
    "ExtractionResult",
    "InformationType",
    "ContextEnhancer",
    "CodeContext",
    "ProjectStructure",
    "MultiRoundReasoner",
    "RoundResult",
    "ConvergenceStrategy"
]