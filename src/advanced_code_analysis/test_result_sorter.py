"""
Tests for the Result Sorting and Output Engine.

This module contains comprehensive tests for the ResultSorter class,
including unit tests for sorting algorithms, output formatting,
validation, and quality assessment functionality.
"""

import pytest
import json
from typing import List
from unittest.mock import Mock, patch

from .models import (
    AnalysisResult, ReasoningChain, ReasoningStep, EvidenceChain,
    ContextWindow, BugType, BugCategory
)
from .result_sorter import (
    ResultSorter, SortingCriteria, OutputFormat, CodeImpactMetrics,
    QualityMetrics, RankedResult, FormattedOutput
)
from .config import AdvancedAnalysisConfig


class TestCodeImpactMetrics:
    """Test CodeImpactMetrics functionality."""
    
    def test_calculate_overall_impact_low(self):
        """Test overall impact calculation for low impact changes."""
        metrics = CodeImpactMetrics(
            lines_affected=5,
            functions_affected=1,
            classes_affected=0,
            modules_affected=0,
            dependency_impact=0.1,
            risk_score=0.2
        )
        
        impact = metrics.calculate_overall_impact()
        assert 0.0 <= impact <= 1.0
        assert impact < 0.5  # Should be low impact
    
    def test_calculate_overall_impact_high(self):
        """Test overall impact calculation for high impact changes."""
        metrics = CodeImpactMetrics(
            lines_affected=100,
            functions_affected=20,
            classes_affected=10,
            modules_affected=5,
            dependency_impact=0.9,
            risk_score=0.8
        )
        
        impact = metrics.calculate_overall_impact()
        assert 0.0 <= impact <= 1.0
        assert impact > 0.7  # Should be high impact
    
    def test_calculate_overall_impact_capped(self):
        """Test that overall impact is capped at 1.0."""
        metrics = CodeImpactMetrics(
            lines_affected=1000,
            functions_affected=100,
            classes_affected=50,
            modules_affected=20,
            dependency_impact=1.0,
            risk_score=1.0
        )
        
        impact = metrics.calculate_overall_impact()
        assert impact == 1.0


class TestQualityMetrics:
    """Test QualityMetrics functionality."""
    
    def test_calculate_overall_quality(self):
        """Test overall quality calculation."""
        metrics = QualityMetrics(
            reasoning_completeness=0.8,
            evidence_strength=0.7,
            consistency_score=0.9,
            specificity_score=0.6,
            actionability_score=0.8
        )
        
        quality = metrics.calculate_overall_quality()
        assert 0.0 <= quality <= 1.0
        # Should be weighted average: 0.8*0.25 + 0.7*0.25 + 0.9*0.2 + 0.6*0.15 + 0.8*0.15
        expected = 0.8 * 0.25 + 0.7 * 0.25 + 0.9 * 0.2 + 0.6 * 0.15 + 0.8 * 0.15
        assert abs(quality - expected) < 0.01


class TestRankedResult:
    """Test RankedResult functionality."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Create sample analysis result
        reasoning_chain = ReasoningChain()
        reasoning_chain.add_step(ReasoningStep(
            step_number=1,
            description="Test step",
            confidence=0.8
        ))
        
        analysis = AnalysisResult(
            bug_location="test.py:10",
            root_cause="Test cause",
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=reasoning_chain,
            supporting_evidence=["Test evidence"]
        )
        
        impact_metrics = CodeImpactMetrics(lines_affected=5)
        quality_metrics = QualityMetrics(reasoning_completeness=0.8)
        
        ranked_result = RankedResult(
            analysis=analysis,
            rank=1,
            impact_metrics=impact_metrics,
            quality_metrics=quality_metrics,
            composite_score=0.75,
            ranking_rationale="Test rationale"
        )
        
        result_dict = ranked_result.to_dict()
        
        assert result_dict['rank'] == 1
        assert result_dict['composite_score'] == 0.75
        assert result_dict['ranking_rationale'] == "Test rationale"
        assert 'analysis' in result_dict
        assert 'impact_metrics' in result_dict
        assert 'quality_metrics' in result_dict


class TestFormattedOutput:
    """Test FormattedOutput functionality."""
    
    def create_sample_formatted_output(self) -> FormattedOutput:
        """Create sample formatted output for testing."""
        # Create minimal ranked result
        reasoning_chain = ReasoningChain()
        analysis = AnalysisResult(
            bug_location="test.py:10",
            root_cause="Test cause",
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=reasoning_chain
        )
        
        ranked_result = RankedResult(
            analysis=analysis,
            rank=1,
            impact_metrics=CodeImpactMetrics(),
            quality_metrics=QualityMetrics(),
            composite_score=0.75,
            ranking_rationale="Test rationale"
        )
        
        return FormattedOutput(
            title="Test Analysis",
            summary="Test summary",
            ranked_results=[ranked_result],
            reasoning_process="Test process",
            evidence_chains={1: "Test evidence"},
            quality_assessment="Test assessment",
            recommendations=["Test recommendation"],
            metadata={"test": "value"}
        )
    
    def test_to_json(self):
        """Test JSON conversion."""
        output = self.create_sample_formatted_output()
        json_str = output.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['title'] == "Test Analysis"
        assert parsed['summary'] == "Test summary"
        assert len(parsed['ranked_results']) == 1
    
    def test_to_markdown(self):
        """Test Markdown conversion."""
        output = self.create_sample_formatted_output()
        markdown = output.to_markdown()
        
        assert "# Test Analysis" in markdown
        assert "## Summary" in markdown
        assert "## Analysis Results" in markdown
        assert "### Rank 1:" in markdown
        assert "**Root Cause:**" in markdown
        assert "**Fix Suggestion:**" in markdown


class TestResultSorter:
    """Test ResultSorter functionality."""
    
    def create_sample_analysis_results(self) -> List[AnalysisResult]:
        """Create sample analysis results for testing."""
        results = []
        
        for i in range(3):
            reasoning_chain = ReasoningChain()
            reasoning_chain.add_step(ReasoningStep(
                step_number=1,
                description=f"Step {i+1}",
                confidence=0.7 + i * 0.1,
                evidence=[f"Evidence {i+1}"]
            ))
            
            result = AnalysisResult(
                bug_location=f"file{i+1}.py:line{i+1}0",
                root_cause=f"Cause {i+1}",
                fix_suggestion=f"Fix {i+1}",
                confidence=0.6 + i * 0.1,
                reasoning_chain=reasoning_chain,
                supporting_evidence=[f"Evidence {i+1}", f"More evidence {i+1}"]
            )
            results.append(result)
        
        return results
    
    def create_sample_context(self) -> ContextWindow:
        """Create sample context for testing."""
        return ContextWindow(
            target_code="def test(): pass",
            related_functions=["func1", "func2"],
            class_hierarchy={"Class1": []},
            module_dependencies=["module1"]
        )
    
    def test_init(self):
        """Test ResultSorter initialization."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        assert sorter.config == config
        assert len(sorter.sorting_weights) == 5
        assert SortingCriteria.CONFIDENCE in sorter.sorting_weights
    
    def test_sort_candidates_empty(self):
        """Test sorting with empty candidate list."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        
        result = sorter.sort_candidates([], context)
        assert result == []
    
    def test_sort_candidates_single(self):
        """Test sorting with single candidate."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        candidates = self.create_sample_analysis_results()[:1]
        
        result = sorter.sort_candidates(candidates, context)
        
        assert len(result) == 1
        assert result[0].rank == 1
        assert result[0].analysis == candidates[0]
        assert 0.0 <= result[0].composite_score <= 1.0
    
    def test_sort_candidates_multiple(self):
        """Test sorting with multiple candidates."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        candidates = self.create_sample_analysis_results()
        
        result = sorter.sort_candidates(candidates, context)
        
        assert len(result) == 3
        # Should be sorted by composite score (descending)
        for i in range(len(result) - 1):
            assert result[i].composite_score >= result[i + 1].composite_score
        
        # Ranks should be assigned correctly
        for i, ranked_result in enumerate(result):
            assert ranked_result.rank == i + 1
    
    def test_sort_candidates_confidence_only(self):
        """Test sorting with confidence criteria only."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        candidates = self.create_sample_analysis_results()
        
        result = sorter.sort_candidates(
            candidates, context, [SortingCriteria.CONFIDENCE]
        )
        
        assert len(result) == 3
        # Should be sorted by confidence (highest first)
        for i in range(len(result) - 1):
            assert result[i].analysis.confidence >= result[i + 1].analysis.confidence
    
    def test_format_output_detailed(self):
        """Test detailed output formatting."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        candidates = self.create_sample_analysis_results()
        
        ranked_results = sorter.sort_candidates(candidates, context)
        bug_type = BugType(category=BugCategory.LOGIC_ERROR, confidence=0.8)
        
        output = sorter.format_output(
            ranked_results, "Test issue", bug_type, OutputFormat.DETAILED
        )
        
        assert isinstance(output, FormattedOutput)
        assert "Logic Error" in output.title
        assert len(output.ranked_results) == 3
        assert len(output.evidence_chains) == 3
        assert output.summary
        assert output.reasoning_process
        assert output.quality_assessment
        assert output.recommendations
        assert output.metadata
    
    def test_validate_results_empty(self):
        """Test validation with empty results."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        report = sorter.validate_results([])
        
        assert not report['is_valid']
        assert len(report['issues']) > 0
        assert "No analysis results" in report['issues'][0]
    
    def test_validate_results_valid(self):
        """Test validation with valid results."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        candidates = self.create_sample_analysis_results()
        
        ranked_results = sorter.sort_candidates(candidates, context)
        report = sorter.validate_results(ranked_results)
        
        assert 'is_valid' in report
        assert 'issues' in report
        assert 'warnings' in report
        assert 'quality_scores' in report
        assert 'recommendations' in report
    
    def test_validate_results_missing_info(self):
        """Test validation with missing information."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        # Create result with missing information
        reasoning_chain = ReasoningChain()
        analysis = AnalysisResult(
            bug_location="",  # Missing
            root_cause="",    # Missing
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=reasoning_chain
        )
        
        ranked_result = RankedResult(
            analysis=analysis,
            rank=1,
            impact_metrics=CodeImpactMetrics(),
            quality_metrics=QualityMetrics(),
            composite_score=0.75,
            ranking_rationale="Test"
        )
        
        report = sorter.validate_results([ranked_result])
        
        assert not report['is_valid']
        assert len(report['issues']) >= 2  # Missing location and cause
    
    def test_build_evidence_chain_display(self):
        """Test evidence chain display building."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        # Create analysis with evidence chain
        reasoning_chain = ReasoningChain()
        reasoning_chain.add_step(ReasoningStep(
            step_number=1,
            description="Test step",
            confidence=0.8,
            evidence=["Test evidence"]
        ))
        
        evidence_chain = EvidenceChain()
        evidence_chain.add_evidence("Test evidence", "Test source", 0.9)
        
        analysis = AnalysisResult(
            bug_location="test.py:10",
            root_cause="Test cause",
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=reasoning_chain,
            supporting_evidence=["Supporting evidence"],
            evidence_chain=evidence_chain
        )
        
        display = sorter.build_evidence_chain_display(analysis)
        
        assert isinstance(display, str)
        assert "Evidence Chain" in display
        assert "Test evidence" in display
    
    def test_analyze_code_impact_simple_fix(self):
        """Test code impact analysis for simple fixes."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        
        analysis = AnalysisResult(
            bug_location="test.py:10",
            root_cause="Simple error",
            fix_suggestion="Change variable assignment",
            confidence=0.8,
            reasoning_chain=ReasoningChain()
        )
        
        impact = sorter._analyze_code_impact(analysis, context)
        
        assert isinstance(impact, CodeImpactMetrics)
        assert impact.lines_affected > 0
        assert 0.0 <= impact.risk_score <= 1.0
        assert 0.0 <= impact.implementation_effort <= 1.0
    
    def test_analyze_code_impact_complex_fix(self):
        """Test code impact analysis for complex fixes."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        context = self.create_sample_context()
        
        analysis = AnalysisResult(
            bug_location="entire_module.py",
            root_cause="Architecture issue",
            fix_suggestion="Refactor and restructure the entire codebase",
            confidence=0.7,
            reasoning_chain=ReasoningChain()
        )
        
        impact = sorter._analyze_code_impact(analysis, context)
        
        assert impact.lines_affected > 20  # Should detect high impact
        assert impact.risk_score > 0.5
        assert impact.implementation_effort > 0.5
    
    def test_assess_quality_high_quality(self):
        """Test quality assessment for high-quality analysis."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        # Create high-quality analysis
        reasoning_chain = ReasoningChain()
        for i in range(4):  # Multiple reasoning steps
            reasoning_chain.add_step(ReasoningStep(
                step_number=i + 1,
                description=f"Detailed step {i + 1}",
                confidence=0.8,
                evidence=[f"Strong evidence {i + 1}"]
            ))
        
        analysis = AnalysisResult(
            bug_location="src/module/file.py:line 42 in function process_data",  # Specific
            root_cause="Detailed explanation of the root cause with technical details",  # Detailed
            fix_suggestion="Specific fix: change variable assignment from x = 0 to x += 1 on line 42",  # Actionable
            confidence=0.9,
            reasoning_chain=reasoning_chain,
            supporting_evidence=[f"Evidence {i}" for i in range(6)]  # Lots of evidence
        )
        
        quality = sorter._assess_quality(analysis)
        
        assert quality.reasoning_completeness > 0.8
        assert quality.evidence_strength > 0.8
        assert quality.specificity_score > 0.5
        assert quality.actionability_score > 0.5
    
    def test_assess_quality_low_quality(self):
        """Test quality assessment for low-quality analysis."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        # Create low-quality analysis
        reasoning_chain = ReasoningChain()
        reasoning_chain.add_step(ReasoningStep(
            step_number=1,
            description="Basic step",
            confidence=0.3
        ))
        
        analysis = AnalysisResult(
            bug_location="somewhere",  # Vague
            root_cause="error",        # Vague
            fix_suggestion="fix it",   # Not actionable
            confidence=0.3,
            reasoning_chain=reasoning_chain,
            supporting_evidence=[]     # No evidence
        )
        
        quality = sorter._assess_quality(analysis)
        
        assert quality.reasoning_completeness < 0.6
        assert quality.evidence_strength < 0.4
        assert quality.specificity_score < 0.5
        assert quality.actionability_score < 0.5
    
    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        analysis = AnalysisResult(
            bug_location="test.py:10",
            root_cause="Test cause",
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=ReasoningChain()
        )
        
        impact_metrics = CodeImpactMetrics(
            lines_affected=5,
            risk_score=0.2,
            implementation_effort=0.3
        )
        
        quality_metrics = QualityMetrics(
            reasoning_completeness=0.8,
            evidence_strength=0.7,
            consistency_score=0.9,
            specificity_score=0.6,
            actionability_score=0.8
        )
        
        score = sorter._calculate_composite_score(
            analysis, impact_metrics, quality_metrics, list(SortingCriteria)
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_generate_ranking_rationale(self):
        """Test ranking rationale generation."""
        config = AdvancedAnalysisConfig()
        sorter = ResultSorter(config)
        
        analysis = AnalysisResult(
            bug_location="test.py:10",
            root_cause="Test cause",
            fix_suggestion="Test fix",
            confidence=0.8,
            reasoning_chain=ReasoningChain()
        )
        
        impact_metrics = CodeImpactMetrics(risk_score=0.2)
        quality_metrics = QualityMetrics(evidence_strength=0.8)
        
        rationale = sorter._generate_ranking_rationale(
            analysis, impact_metrics, quality_metrics, 0.75
        )
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "confidence" in rationale.lower()
        assert "0.750" in rationale  # Composite score should be included


@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    return AdvancedAnalysisConfig()


@pytest.fixture
def sample_sorter(sample_config):
    """Provide sample ResultSorter for tests."""
    return ResultSorter(sample_config)


class TestIntegration:
    """Integration tests for the complete result sorting workflow."""
    
    def test_complete_workflow(self, sample_sorter):
        """Test complete workflow from candidates to formatted output."""
        # Create test data
        candidates = []
        for i in range(3):
            reasoning_chain = ReasoningChain()
            reasoning_chain.add_step(ReasoningStep(
                step_number=1,
                description=f"Analysis step {i+1}",
                confidence=0.6 + i * 0.1,
                evidence=[f"Evidence {i+1}"]
            ))
            
            candidate = AnalysisResult(
                bug_location=f"file{i+1}.py:line{(i+1)*10}",
                root_cause=f"Root cause {i+1}",
                fix_suggestion=f"Fix suggestion {i+1}",
                confidence=0.6 + i * 0.1,
                reasoning_chain=reasoning_chain,
                supporting_evidence=[f"Evidence {i+1}"]
            )
            candidates.append(candidate)
        
        context = ContextWindow(
            target_code="def test(): pass",
            related_functions=["func1"],
            class_hierarchy={"Class1": []},
            module_dependencies=["module1"]
        )
        
        bug_type = BugType(category=BugCategory.LOGIC_ERROR, confidence=0.8)
        
        # Execute workflow
        ranked_results = sample_sorter.sort_candidates(candidates, context)
        formatted_output = sample_sorter.format_output(
            ranked_results, "Test issue", bug_type
        )
        validation_report = sample_sorter.validate_results(ranked_results)
        
        # Verify results
        assert len(ranked_results) == 3
        assert isinstance(formatted_output, FormattedOutput)
        assert validation_report['is_valid'] or len(validation_report['issues']) == 0
        
        # Test output formats
        json_output = formatted_output.to_json()
        markdown_output = formatted_output.to_markdown()
        
        assert json.loads(json_output)  # Should be valid JSON
        assert "# " in markdown_output   # Should contain markdown headers
    
    def test_edge_cases(self, sample_sorter):
        """Test edge cases and error conditions."""
        context = ContextWindow(target_code="")
        
        # Test with minimal analysis result
        minimal_reasoning = ReasoningChain()
        minimal_analysis = AnalysisResult(
            bug_location="unknown",
            root_cause="unknown",
            fix_suggestion="unknown",
            confidence=0.0,
            reasoning_chain=minimal_reasoning
        )
        
        ranked_results = sample_sorter.sort_candidates([minimal_analysis], context)
        assert len(ranked_results) == 1
        assert ranked_results[0].composite_score >= 0.0
        
        # Test validation with problematic result
        validation_report = sample_sorter.validate_results(ranked_results)
        # Should handle gracefully without crashing
        assert 'is_valid' in validation_report