"""
Tests for the Conflict Detection and Handling Engine.

This module contains unit tests and property-based tests for the ConflictDetector
class, testing conflict detection, resolution strategies, and adaptive context
collection functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from .conflict_detector import (
    ConflictDetector, ConflictMarker, ConflictType, ConflictSeverity,
    ResolutionStrategy, AdaptiveContext
)
from .models import (
    AnalysisResult, ReasoningChain, ReasoningStep, ContextWindow,
    BugType, BugCategory, AnalysisStrategy, PromptTemplate, Conflict
)
from .llm_interface import LLMInterface, LLMResponse
from .enhanced_ast_analyzer import EnhancedASTAnalyzer
from .multi_round_reasoner import MultiRoundReasoner
from .config import AdvancedAnalysisConfig


class TestConflictDetector:
    """Test cases for ConflictDetector class."""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate = AsyncMock(return_value=LLMResponse(
            content='{"reasoning": "test", "confidence": 0.8}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        ))
        return mock_llm
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=AdvancedAnalysisConfig)
        config.analysis = Mock()
        config.analysis.confidence_threshold = 0.7
        config.analysis.max_reasoning_rounds = 3
        return config
    
    @pytest.fixture
    def sample_analysis_result(self):
        """Create sample analysis result."""
        reasoning_chain = ReasoningChain()
        step = ReasoningStep(
            step_number=1,
            description="Initial analysis",
            confidence=0.8,
            evidence=["Sample evidence"]
        )
        reasoning_chain.add_step(step)
        
        return AnalysisResult(
            bug_location="line 42",
            root_cause="Null pointer dereference",
            fix_suggestion="Add null check",
            confidence=0.8,
            reasoning_chain=reasoning_chain,
            supporting_evidence=["Found null pointer access"]
        )
    
    @pytest.fixture
    def conflicting_analysis_result(self):
        """Create conflicting analysis result."""
        reasoning_chain = ReasoningChain()
        step = ReasoningStep(
            step_number=1,
            description="AST analysis",
            confidence=0.6,
            evidence=["AST evidence"]
        )
        reasoning_chain.add_step(step)
        
        return AnalysisResult(
            bug_location="line 45",
            root_cause="Type mismatch error",
            fix_suggestion="Fix type conversion",
            confidence=0.6,
            reasoning_chain=reasoning_chain,
            supporting_evidence=["Type error detected"]
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context window."""
        return ContextWindow(
            target_code="def test_function():\n    x = None\n    return x.value",
            related_functions=["helper_function"],
            module_dependencies=["os", "sys"]
        )
    
    @pytest.fixture
    def conflict_detector(self, mock_llm_interface, mock_config):
        """Create ConflictDetector instance."""
        return ConflictDetector(mock_llm_interface, mock_config)
    
    def test_initialization(self, conflict_detector):
        """Test ConflictDetector initialization."""
        assert conflict_detector is not None
        assert conflict_detector.llm is not None
        assert conflict_detector.config is not None
        assert len(conflict_detector.resolution_strategies) > 0
        assert conflict_detector.confidence_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_detect_location_conflicts(self, conflict_detector, 
                                           sample_analysis_result, 
                                           conflicting_analysis_result):
        """Test detection of location conflicts."""
        conflicts = conflict_detector._detect_location_conflicts(
            sample_analysis_result, conflicting_analysis_result
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.LOCATION_MISMATCH
        assert conflicts[0].severity == ConflictSeverity.HIGH
        assert "line 42" in conflicts[0].metadata["llm_location"]
        assert "line 45" in conflicts[0].metadata["ast_location"]
    
    @pytest.mark.asyncio
    async def test_detect_confidence_conflicts(self, conflict_detector,
                                             sample_analysis_result,
                                             conflicting_analysis_result):
        """Test detection of confidence conflicts."""
        # Modify confidence to create significant difference
        conflicting_analysis_result.confidence = 0.3
        
        conflicts = conflict_detector._detect_confidence_conflicts(
            sample_analysis_result, conflicting_analysis_result
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.CONFIDENCE_DISCREPANCY
        # The difference is 0.5, which should be MEDIUM severity (not HIGH)
        assert conflicts[0].severity == ConflictSeverity.MEDIUM
        assert conflicts[0].metadata["difference"] == 0.5
    
    @pytest.mark.asyncio
    async def test_detect_evidence_conflicts(self, conflict_detector,
                                           sample_analysis_result,
                                           conflicting_analysis_result):
        """Test detection of evidence conflicts."""
        # Create contradictory evidence
        sample_analysis_result.supporting_evidence = ["No error found"]
        conflicting_analysis_result.supporting_evidence = ["Error found in code"]
        
        conflicts = conflict_detector._detect_evidence_conflicts(
            sample_analysis_result, conflicting_analysis_result
        )
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.EVIDENCE_CONTRADICTION
        assert len(conflicts[0].metadata["contradictions"]) > 0
    
    @pytest.mark.asyncio
    async def test_detect_internal_conflicts(self, conflict_detector, sample_analysis_result):
        """Test detection of internal conflicts."""
        # Create analysis with multiple root causes
        sample_analysis_result.root_cause = "This is caused by null pointer because of type error due to missing validation"
        
        conflicts = conflict_detector._detect_internal_conflicts(sample_analysis_result)
        
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.MULTIPLE_ROOT_CAUSES
        assert conflicts[0].severity == ConflictSeverity.LOW
    
    @pytest.mark.asyncio
    async def test_detect_conflicts_integration(self, conflict_detector,
                                              sample_analysis_result,
                                              conflicting_analysis_result,
                                              sample_context):
        """Test full conflict detection integration."""
        conflicts = await conflict_detector.detect_conflicts(
            analyses=[sample_analysis_result, conflicting_analysis_result],
            context=sample_context
        )
        
        # Should detect at least location conflicts
        assert len(conflicts) >= 1
        conflict_types = [c.conflict_type for c in conflicts]
        assert ConflictType.LOCATION_MISMATCH in conflict_types
        # Note: Confidence discrepancy might not be detected if difference is small
    
    @pytest.mark.asyncio
    async def test_handle_conflicts_no_conflicts(self, conflict_detector):
        """Test handling when no conflicts exist."""
        sample_analysis = AnalysisResult(
            bug_location="line 1",
            root_cause="Test cause",
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["Test evidence"]
        )
        
        conflicts = [ConflictMarker(
            conflict_id="test",
            conflict_type=ConflictType.LOCATION_MISMATCH,
            severity=ConflictSeverity.LOW,
            description="Test conflict",
            affected_analyses=[sample_analysis]
        )]
        
        resolution = await conflict_detector.handle_conflicts(
            conflicts, "Test issue"
        )
        
        assert resolution is not None
        assert resolution.final_result is not None
        assert resolution.resolution_method is not None
    
    @pytest.mark.asyncio
    async def test_collect_adaptive_context(self, conflict_detector,
                                          sample_analysis_result,
                                          sample_context):
        """Test adaptive context collection."""
        # Set low confidence to trigger adaptive collection
        sample_analysis_result.confidence = 0.4
        
        adaptive_context = await conflict_detector.collect_adaptive_context(
            sample_analysis_result,
            sample_context,
            code="def test():\n    pass"
        )
        
        assert isinstance(adaptive_context, AdaptiveContext)
        assert adaptive_context.original_context == sample_context
        assert adaptive_context.collection_confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_multi_strategy_reasoning(self, conflict_detector,
                                          sample_analysis_result,
                                          conflicting_analysis_result):
        """Test multi-strategy reasoning."""
        conflicts = [
            ConflictMarker(
                conflict_id="test1",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.HIGH,
                description="Test conflict",
                affected_analyses=[sample_analysis_result, conflicting_analysis_result]
            )
        ]
        
        results = await conflict_detector.multi_strategy_reasoning(
            conflicts, "Test issue"
        )
        
        assert isinstance(results, list)
        # Should have at least one result from the strategies
        assert len(results) >= 1
    
    def test_prioritize_conflicts(self, conflict_detector):
        """Test conflict prioritization."""
        conflicts = [
            ConflictMarker(
                conflict_id="low",
                conflict_type=ConflictType.MULTIPLE_ROOT_CAUSES,
                severity=ConflictSeverity.LOW,
                description="Low priority",
                resolution_priority=5
            ),
            ConflictMarker(
                conflict_id="high",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.CRITICAL,
                description="High priority",
                resolution_priority=1
            ),
            ConflictMarker(
                conflict_id="medium",
                conflict_type=ConflictType.CONFIDENCE_DISCREPANCY,
                severity=ConflictSeverity.MEDIUM,
                description="Medium priority",
                resolution_priority=3
            )
        ]
        
        prioritized = conflict_detector._prioritize_conflicts(conflicts)
        
        assert len(prioritized) == 3
        assert prioritized[0].severity == ConflictSeverity.CRITICAL
        assert prioritized[-1].severity == ConflictSeverity.LOW
    
    def test_group_conflicts_by_type(self, conflict_detector):
        """Test grouping conflicts by type."""
        conflicts = [
            ConflictMarker(
                conflict_id="loc1",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.HIGH,
                description="Location conflict 1"
            ),
            ConflictMarker(
                conflict_id="loc2",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.MEDIUM,
                description="Location conflict 2"
            ),
            ConflictMarker(
                conflict_id="conf1",
                conflict_type=ConflictType.CONFIDENCE_DISCREPANCY,
                severity=ConflictSeverity.LOW,
                description="Confidence conflict"
            )
        ]
        
        groups = conflict_detector._group_conflicts_by_type(conflicts)
        
        assert len(groups) == 2
        assert ConflictType.LOCATION_MISMATCH in groups
        assert ConflictType.CONFIDENCE_DISCREPANCY in groups
        assert len(groups[ConflictType.LOCATION_MISMATCH]) == 2
        assert len(groups[ConflictType.CONFIDENCE_DISCREPANCY]) == 1
    
    def test_calculate_location_similarity(self, conflict_detector):
        """Test location similarity calculation."""
        # Similar locations
        similarity1 = conflict_detector._calculate_location_similarity(
            "line 42 in function test", "line 42 in method test"
        )
        assert similarity1 > 0.5
        
        # Different locations
        similarity2 = conflict_detector._calculate_location_similarity(
            "line 10 in class A", "line 100 in module B"
        )
        assert similarity2 < 0.5
        
        # Identical locations
        similarity3 = conflict_detector._calculate_location_similarity(
            "line 42", "line 42"
        )
        assert similarity3 == 1.0
    
    def test_calculate_text_similarity(self, conflict_detector):
        """Test text similarity calculation."""
        # Similar texts
        similarity1 = conflict_detector._calculate_text_similarity(
            "add null check before access", "add null validation before accessing"
        )
        assert similarity1 > 0.3
        
        # Different texts
        similarity2 = conflict_detector._calculate_text_similarity(
            "fix type error", "remove unused variable"
        )
        assert similarity2 < 0.3
        
        # Identical texts
        similarity3 = conflict_detector._calculate_text_similarity(
            "same text", "same text"
        )
        assert similarity3 == 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_conflict_detection(self, conflict_detector):
        """Test error handling during conflict detection."""
        # Create invalid analysis result that might cause errors
        invalid_analysis = Mock()
        invalid_analysis.bug_location = None
        invalid_analysis.confidence = "invalid"
        
        conflicts = await conflict_detector.detect_conflicts(
            analyses=[invalid_analysis]
        )
        
        # Should handle errors gracefully and return empty list
        assert isinstance(conflicts, list)
    
    @pytest.mark.asyncio
    async def test_fallback_resolution(self, conflict_detector, sample_analysis_result):
        """Test fallback resolution when normal resolution fails."""
        conflicts = [
            ConflictMarker(
                conflict_id="test",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.HIGH,
                description="Test conflict",
                affected_analyses=[sample_analysis_result]
            )
        ]
        
        # Mock the resolution to fail
        with patch.object(conflict_detector, '_resolve_conflict_group', 
                         side_effect=Exception("Test error")):
            resolution = conflict_detector._create_fallback_resolution(
                conflicts, "Test error"
            )
            
            assert resolution is not None
            assert resolution.resolution_method == "fallback"
            assert "Test error" in resolution.resolution_notes


class TestConflictDetectorProperties:
    """Property-based tests for ConflictDetector."""
    
    @pytest.fixture
    def conflict_detector(self):
        """Create ConflictDetector for property tests."""
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate = AsyncMock(return_value=LLMResponse(
            content='{"confidence": 0.8}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        ))
        
        mock_config = Mock(spec=AdvancedAnalysisConfig)
        mock_config.analysis = Mock()
        mock_config.analysis.confidence_threshold = 0.7
        mock_config.analysis.max_reasoning_rounds = 3
        
        return ConflictDetector(mock_llm, mock_config)
    
    def test_property_conflict_detection_consistency(self, conflict_detector):
        """
        **Feature: advanced-code-analysis, Property 15: Conflict detection handling**
        
        Property: For any AST analysis and LLM judgment disagreement, 
        the system should detect and flag conflicts requiring further verification.
        """
        # Create analyses with clear disagreements
        llm_analysis = AnalysisResult(
            bug_location="line 10",
            root_cause="Null pointer error",
            fix_suggestion="Add null check",
            confidence=0.9,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["LLM found null pointer"]
        )
        
        ast_analysis = AnalysisResult(
            bug_location="line 20",
            root_cause="Type mismatch error", 
            fix_suggestion="Fix type conversion",
            confidence=0.8,
            reasoning_chain=ReasoningChain(),
            supporting_evidence=["AST found type error"]
        )
        
        # Detect conflicts
        location_conflicts = conflict_detector._detect_location_conflicts(llm_analysis, ast_analysis)
        confidence_conflicts = conflict_detector._detect_confidence_conflicts(llm_analysis, ast_analysis)
        
        # Property: Disagreements should be detected as conflicts
        assert len(location_conflicts) > 0, "Location disagreement should be detected as conflict"
        
        # Property: Conflicts should be properly categorized
        if location_conflicts:
            assert location_conflicts[0].conflict_type == ConflictType.LOCATION_MISMATCH
            assert location_conflicts[0].severity in [ConflictSeverity.HIGH, ConflictSeverity.MEDIUM]
        
        # Property: Conflict metadata should contain relevant information
        if location_conflicts:
            metadata = location_conflicts[0].metadata
            assert "llm_location" in metadata
            assert "ast_location" in metadata
            assert "similarity_score" in metadata
    
    def test_property_conflict_prioritization(self, conflict_detector):
        """
        Property: For any set of conflicts, critical and high severity conflicts
        should be prioritized over medium and low severity conflicts.
        """
        conflicts = [
            ConflictMarker(
                conflict_id="low1",
                conflict_type=ConflictType.MULTIPLE_ROOT_CAUSES,
                severity=ConflictSeverity.LOW,
                description="Low severity",
                resolution_priority=4
            ),
            ConflictMarker(
                conflict_id="critical1",
                conflict_type=ConflictType.AST_LLM_DISAGREEMENT,
                severity=ConflictSeverity.CRITICAL,
                description="Critical severity",
                resolution_priority=1
            ),
            ConflictMarker(
                conflict_id="medium1",
                conflict_type=ConflictType.EVIDENCE_CONTRADICTION,
                severity=ConflictSeverity.MEDIUM,
                description="Medium severity",
                resolution_priority=3
            ),
            ConflictMarker(
                conflict_id="high1",
                conflict_type=ConflictType.LOCATION_MISMATCH,
                severity=ConflictSeverity.HIGH,
                description="High severity",
                resolution_priority=2
            )
        ]
        
        prioritized = conflict_detector._prioritize_conflicts(conflicts)
        
        # Property: Critical conflicts should come first
        assert prioritized[0].severity == ConflictSeverity.CRITICAL
        
        # Property: Severity ordering should be maintained
        severity_order = [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH, 
                         ConflictSeverity.MEDIUM, ConflictSeverity.LOW]
        
        prev_severity_index = -1
        for conflict in prioritized:
            current_severity_index = severity_order.index(conflict.severity)
            assert current_severity_index >= prev_severity_index, \
                "Conflicts should be ordered by severity (critical to low)"
            prev_severity_index = current_severity_index
    
    def test_property_similarity_calculation_bounds(self, conflict_detector):
        """
        Property: For any two text strings, similarity calculation should
        return a value between 0.0 and 1.0 inclusive.
        """
        test_cases = [
            ("identical text", "identical text"),
            ("completely different", "totally unrelated words"),
            ("", ""),
            ("single", ""),
            ("partial match here", "partial match there"),
            ("line 42 function test", "line 42 method test")
        ]
        
        for text1, text2 in test_cases:
            # Test text similarity
            text_similarity = conflict_detector._calculate_text_similarity(text1, text2)
            assert 0.0 <= text_similarity <= 1.0, \
                f"Text similarity should be in [0,1], got {text_similarity} for '{text1}' vs '{text2}'"
            
            # Test location similarity
            location_similarity = conflict_detector._calculate_location_similarity(text1, text2)
            assert 0.0 <= location_similarity <= 1.0, \
                f"Location similarity should be in [0,1], got {location_similarity} for '{text1}' vs '{text2}'"
    
    def test_property_conflict_grouping_completeness(self, conflict_detector):
        """
        Property: For any list of conflicts, grouping by type should preserve
        all conflicts and group them correctly by their conflict type.
        """
        conflicts = [
            ConflictMarker(
                conflict_id=f"conflict_{i}",
                conflict_type=ConflictType.LOCATION_MISMATCH if i % 2 == 0 else ConflictType.CONFIDENCE_DISCREPANCY,
                severity=ConflictSeverity.MEDIUM,
                description=f"Conflict {i}"
            )
            for i in range(10)
        ]
        
        groups = conflict_detector._group_conflicts_by_type(conflicts)
        
        # Property: All conflicts should be preserved
        total_grouped = sum(len(group) for group in groups.values())
        assert total_grouped == len(conflicts), "All conflicts should be preserved in grouping"
        
        # Property: Conflicts should be grouped correctly by type
        for conflict_type, group in groups.items():
            for conflict in group:
                assert conflict.conflict_type == conflict_type, \
                    f"Conflict {conflict.conflict_id} should be in group {conflict_type}"
        
        # Property: No conflict should appear in multiple groups
        all_grouped_conflicts = []
        for group in groups.values():
            all_grouped_conflicts.extend(group)
        
        conflict_ids = [c.conflict_id for c in all_grouped_conflicts]
        assert len(conflict_ids) == len(set(conflict_ids)), "No conflict should appear in multiple groups"


if __name__ == "__main__":
    pytest.main([__file__])