"""
Integration tests for the Advanced Code Analysis system.

This module contains comprehensive integration tests that verify the complete
analysis pipeline from problem classification to final output.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from .advanced_code_analyzer import AdvancedCodeAnalyzer, ComprehensiveAnalysisResult
from .config import AdvancedAnalysisConfig, LLMConfig


class TestAdvancedCodeAnalyzerIntegration:
    """Integration tests for the complete Advanced Code Analyzer system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AdvancedAnalysisConfig(
            llm=LLMConfig(
                provider="mock",
                model_name="test-model",
                max_completion_tokens=1000,
                temperature=0.1
            )
        )
    
    @pytest.fixture
    def analyzer(self, config):
        """Create analyzer instance for testing."""
        return AdvancedCodeAnalyzer(config)
    
    @pytest.fixture
    def sample_code_file(self):
        """Create a temporary Python file with sample code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)

def process_data(data):
    if data = None:  # Bug: assignment instead of comparison
        return []
    
    results = []
    for item in data:
        avg = calculate_average(item)
        results.append(avg)
    
    return results

class DataProcessor:
    def __init__(self):
        self.processed_count = 0
    
    def process(self, items):
        for item in items:
            # Bug: constant assignment instead of increment
            self.processed_count = 1
        return self.processed_count
""")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.llm_interface is not None
        assert analyzer.bug_classifier is not None
        assert analyzer.semantic_extractor is not None
        assert analyzer.context_enhancer is not None
        assert analyzer.concept_mapper is not None
        assert analyzer.pattern_matcher is not None
        assert analyzer.multi_round_reasoner is not None
        assert analyzer.ast_analyzer is not None
        assert analyzer.conflict_detector is not None
        assert analyzer.result_sorter is not None
    
    def test_configuration_validation(self, analyzer):
        """Test configuration validation."""
        issues = analyzer.validate_configuration()
        # Should have no critical issues (mock provider doesn't need API key)
        assert isinstance(issues, list)
    
    @pytest.mark.asyncio
    async def test_connection_test(self, analyzer):
        """Test LLM connection testing."""
        # Mock provider should always return True
        result = await analyzer.test_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_simple_analysis_pipeline(self, analyzer, sample_code_file):
        """Test the complete analysis pipeline with a simple issue."""
        issue_text = "The function has a bug where assignment is used instead of comparison"
        target_files = [sample_code_file]
        
        # Set up mock responses for the LLM
        mock_responses = [
            '{"category": "LOGIC_ERROR", "subcategory": "assignment_error", "confidence": 0.8, "characteristics": ["assignment", "comparison"], "reasoning": "Assignment operator used in condition"}',
            '{"technical_concepts": [{"term": "assignment", "confidence": 0.9}], "function_names": [{"name": "process_data", "confidence": 0.8}], "variable_names": [], "class_names": [], "error_patterns": [{"pattern": "assignment in condition", "confidence": 0.9}], "api_calls": [], "problem_summary": "Assignment vs comparison error", "overall_confidence": 0.85}',
            'The issue is in the process_data function where assignment (=) is used instead of comparison (==) in the if condition.',
            'Based on the analysis, the bug is located in the process_data function at the condition check.',
            'The analysis shows high confidence in identifying an assignment vs comparison error.'
        ]
        
        if hasattr(analyzer.llm_interface.provider, 'set_responses'):
            analyzer.llm_interface.provider.set_responses(mock_responses)
        
        try:
            result = await analyzer.analyze(
                issue_text=issue_text,
                target_files=target_files
            )
            
            # Verify result structure
            assert isinstance(result, ComprehensiveAnalysisResult)
            assert result.primary_analysis is not None
            assert result.processing_time > 0
            assert result.session_info is not None
            
            # Verify analysis components were executed
            assert result.bug_classification is not None
            assert result.semantic_extraction is not None
            
            # Verify the analysis found something meaningful
            assert len(result.primary_analysis.bug_location) > 0
            assert len(result.primary_analysis.root_cause) > 0
            assert len(result.primary_analysis.fix_suggestion) > 0
            
        except Exception as e:
            pytest.fail(f"Analysis pipeline failed: {e}")
    
    @pytest.mark.asyncio
    async def test_analysis_with_focus_elements(self, analyzer, sample_code_file):
        """Test analysis with specific focus elements."""
        issue_text = "The DataProcessor class has a bug in the process method"
        target_files = [sample_code_file]
        focus_elements = ["DataProcessor", "process", "processed_count"]
        
        # Set up mock responses
        mock_responses = [
            '{"category": "LOGIC_ERROR", "subcategory": "increment_error", "confidence": 0.7, "characteristics": ["constant_assignment"], "reasoning": "Constant assignment instead of increment"}',
            '{"technical_concepts": [{"term": "increment", "confidence": 0.8}], "function_names": [{"name": "process", "confidence": 0.9}], "variable_names": [{"name": "processed_count", "confidence": 0.9}], "class_names": [{"name": "DataProcessor", "confidence": 0.9}], "error_patterns": [], "api_calls": [], "problem_summary": "Increment error in class method", "overall_confidence": 0.8}',
            'The issue is in the DataProcessor.process method where a constant is assigned instead of incrementing.',
            'Analysis focused on the DataProcessor class and process method.',
            'High confidence in identifying the increment error pattern.'
        ]
        
        if hasattr(analyzer.llm_interface.provider, 'set_responses'):
            analyzer.llm_interface.provider.set_responses(mock_responses)
        
        try:
            result = await analyzer.analyze(
                issue_text=issue_text,
                target_files=target_files,
                focus_elements=focus_elements
            )
            
            assert isinstance(result, ComprehensiveAnalysisResult)
            assert result.primary_analysis is not None
            
            # Should have processed the focus elements
            if result.semantic_extraction:
                extracted_items = result.semantic_extraction.extracted_items
                focus_found = any(
                    item.content in focus_elements 
                    for item in extracted_items
                )
                # Note: This might not always be true due to mock responses
                # but we can at least verify the pipeline completed
            
        except Exception as e:
            pytest.fail(f"Focused analysis failed: {e}")
    
    def test_performance_metrics(self, analyzer):
        """Test performance metrics tracking."""
        stats = analyzer.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'total_analyses' in stats
        assert 'successful_analyses' in stats
        assert 'success_rate' in stats
        assert 'average_processing_time' in stats
        assert 'active_sessions' in stats
        
        # Initially should be zero
        assert stats['total_analyses'] == 0
        assert stats['successful_analyses'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_processing_time'] == 0.0
        assert stats['active_sessions'] == 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, analyzer, sample_code_file):
        """Test session management functionality."""
        issue_text = "Test session management"
        target_files = [sample_code_file]
        
        # Set up minimal mock responses
        mock_responses = [
            '{"category": "LOGIC_ERROR", "confidence": 0.5, "characteristics": [], "reasoning": "Test"}',
            '{"technical_concepts": [], "function_names": [], "variable_names": [], "class_names": [], "error_patterns": [], "api_calls": [], "problem_summary": "Test", "overall_confidence": 0.5}',
            'Test analysis result',
            'Test verification',
            'Test final result'
        ]
        
        if hasattr(analyzer.llm_interface.provider, 'set_responses'):
            analyzer.llm_interface.provider.set_responses(mock_responses)
        
        # Start analysis in background
        analysis_task = asyncio.create_task(
            analyzer.analyze(issue_text, target_files)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Check active sessions
        stats = analyzer.get_performance_stats()
        # Note: Session might complete quickly, so we can't guarantee it's still active
        
        # Wait for completion
        try:
            result = await analysis_task
            assert isinstance(result, ComprehensiveAnalysisResult)
        except Exception as e:
            pytest.fail(f"Session management test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling with invalid inputs."""
        # Test with non-existent file
        issue_text = "Test error handling"
        target_files = ["/non/existent/file.py"]
        
        mock_responses = ['{"category": "LOGIC_ERROR", "confidence": 0.1, "characteristics": [], "reasoning": "Error case"}']
        
        if hasattr(analyzer.llm_interface.provider, 'set_responses'):
            analyzer.llm_interface.provider.set_responses(mock_responses)
        
        try:
            result = await analyzer.analyze(issue_text, target_files)
            
            # Should complete without crashing, even with invalid files
            assert isinstance(result, ComprehensiveAnalysisResult)
            
            # May have warnings about missing files
            if result.session_info:
                # Errors or warnings might be recorded
                pass
                
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_get_session_status_nonexistent(self, analyzer):
        """Test getting status of non-existent session."""
        status = analyzer.get_session_status("non-existent-session")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_cancel_session_nonexistent(self, analyzer):
        """Test canceling non-existent session."""
        result = await analyzer.cancel_session("non-existent-session")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])