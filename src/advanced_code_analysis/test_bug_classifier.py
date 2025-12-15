"""
Tests for the BugClassifier implementation.

This module contains unit tests for the intelligent bug classification engine.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from .bug_classifier import BugClassifier, ClassificationResult, PromptTemplateLibrary, AnalysisStrategyLibrary
from .models import BugType, BugCategory, AnalysisStrategy, PromptTemplate
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface, LLMResponse


class TestPromptTemplateLibrary:
    """Test the prompt template library."""
    
    def test_initialization(self):
        """Test that the library initializes with expected templates."""
        library = PromptTemplateLibrary()
        
        # Check that key templates exist
        assert library.get_template("classification") is not None
        assert library.get_template("refinement") is not None
        assert library.get_template("strategy_selection") is not None
        
        # Check template structure
        classification_template = library.get_template("classification")
        assert "issue_text" in classification_template.placeholders
        assert "code_context" in classification_template.placeholders
    
    def test_add_template(self):
        """Test adding a new template."""
        library = PromptTemplateLibrary()
        
        new_template = PromptTemplate(
            template_id="test_template",
            content="Test content: {test_var}",
            placeholders=["test_var"]
        )
        
        library.add_template(new_template)
        retrieved = library.get_template("test_template")
        
        assert retrieved is not None
        assert retrieved.template_id == "test_template"
        assert retrieved.content == "Test content: {test_var}"


class TestAnalysisStrategyLibrary:
    """Test the analysis strategy library."""
    
    def test_initialization(self):
        """Test that the library initializes with expected strategies."""
        library = AnalysisStrategyLibrary()
        
        # Check that key strategies exist
        assert library.get_strategy("logic_error_deep_analysis") is not None
        assert library.get_strategy("api_issue_analysis") is not None
        assert library.get_strategy("performance_analysis") is not None
        assert library.get_strategy("multi_round_general") is not None
    
    def test_get_strategies_for_category(self):
        """Test getting strategies for specific bug categories."""
        library = AnalysisStrategyLibrary()
        
        # Test logic error strategies
        logic_strategies = library.get_strategies_for_category(BugCategory.LOGIC_ERROR)
        assert len(logic_strategies) >= 1
        assert any(s.strategy_name == "logic_error_deep_analysis" for s in logic_strategies)
        
        # Test API issue strategies
        api_strategies = library.get_strategies_for_category(BugCategory.API_ISSUE)
        assert len(api_strategies) >= 1
        assert any(s.strategy_name == "api_issue_analysis" for s in api_strategies)
        
        # Test unknown category falls back to general
        general_strategies = library.get_strategies_for_category(BugCategory.CONCURRENCY)
        assert len(general_strategies) >= 1
        assert any(s.strategy_name == "multi_round_general" for s in general_strategies)


class TestBugClassifier:
    """Test the BugClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AdvancedAnalysisConfig()
        self.mock_llm = Mock(spec=LLMInterface)
        self.classifier = BugClassifier(self.config, self.mock_llm)
    
    @pytest.mark.asyncio
    async def test_classify_bug_type_logic_error(self):
        """Test classification of a logic error."""
        # Mock LLM response
        mock_response = LLMResponse(
            content='{"category": "LOGIC_ERROR", "subcategory": "conditional_logic", "confidence": 0.85, "characteristics": ["wrong_condition", "if_statement"], "reasoning": "The issue involves incorrect conditional logic"}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.5
        )
        self.mock_llm.generate = AsyncMock(return_value=mock_response)
        
        # Test classification
        issue_text = "The if condition is checking the wrong variable, causing the function to return incorrect results."
        result = await self.classifier.classify_bug_type(issue_text)
        
        # Verify result
        assert isinstance(result, ClassificationResult)
        assert result.bug_type.category == BugCategory.LOGIC_ERROR
        assert result.bug_type.subcategory == "conditional_logic"
        assert result.bug_type.confidence == 0.85
        assert "wrong_condition" in result.bug_type.characteristics
        assert "incorrect conditional logic" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_classify_bug_type_api_issue(self):
        """Test classification of an API issue."""
        # Mock LLM response
        mock_response = LLMResponse(
            content='{"category": "API_ISSUE", "subcategory": "parameter_error", "confidence": 0.9, "characteristics": ["wrong_parameters", "api_call"], "reasoning": "The API is being called with incorrect parameters"}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.5
        )
        self.mock_llm.generate = AsyncMock(return_value=mock_response)
        
        # Test classification
        issue_text = "The API call is failing because we're passing the wrong parameter type."
        result = await self.classifier.classify_bug_type(issue_text)
        
        # Verify result
        assert result.bug_type.category == BugCategory.API_ISSUE
        assert result.bug_type.subcategory == "parameter_error"
        assert result.bug_type.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_classify_bug_type_with_refinement(self):
        """Test classification that requires refinement due to low confidence."""
        # Set low confidence threshold to trigger refinement
        self.config.analysis.classification_confidence_threshold = 0.8
        
        # Mock initial low-confidence response
        initial_response = LLMResponse(
            content='{"category": "LOGIC_ERROR", "subcategory": "unknown", "confidence": 0.5, "characteristics": ["unclear"], "reasoning": "Unclear issue"}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.5
        )
        
        # Mock refined response
        refined_response = LLMResponse(
            content='{"category": "BOUNDARY_CONDITION", "subcategory": "null_check", "confidence": 0.9, "characteristics": ["null_pointer", "validation"], "reasoning": "This is a null pointer issue"}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.5
        )
        
        # Set up mock to return different responses for different calls
        self.mock_llm.generate = AsyncMock(side_effect=[initial_response, refined_response])
        
        # Test classification
        issue_text = "The function crashes when null is passed as input."
        result = await self.classifier.classify_bug_type(issue_text)
        
        # Verify that refinement improved the classification
        assert result.bug_type.category == BugCategory.BOUNDARY_CONDITION
        assert result.bug_type.confidence == 0.9
        assert self.mock_llm.generate.call_count == 2  # Initial + refinement
    
    @pytest.mark.asyncio
    async def test_select_analysis_strategy(self):
        """Test strategy selection for different bug types."""
        # Test logic error strategy selection
        logic_bug = BugType(
            category=BugCategory.LOGIC_ERROR,
            subcategory="conditional",
            confidence=0.8,
            characteristics=["if_statement"]
        )
        
        strategy = await self.classifier.select_analysis_strategy(logic_bug)
        assert isinstance(strategy, AnalysisStrategy)
        assert strategy.strategy_name in ["logic_error_deep_analysis", "multi_round_general"]
        
        # Test API issue strategy selection
        api_bug = BugType(
            category=BugCategory.API_ISSUE,
            subcategory="parameters",
            confidence=0.9,
            characteristics=["wrong_params"]
        )
        
        strategy = await self.classifier.select_analysis_strategy(api_bug)
        assert isinstance(strategy, AnalysisStrategy)
        assert strategy.strategy_name in ["api_issue_analysis", "multi_round_general"]
    
    @pytest.mark.asyncio
    async def test_select_analysis_strategy_with_llm(self):
        """Test strategy selection using LLM when multiple strategies are available."""
        # Mock LLM response for strategy selection
        mock_response = LLMResponse(
            content='{"selected_strategy": "logic_error_deep_analysis", "reasoning": "This strategy is best for conditional logic issues", "confidence": 0.9}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.5
        )
        self.mock_llm.generate = AsyncMock(return_value=mock_response)
        
        logic_bug = BugType(
            category=BugCategory.LOGIC_ERROR,
            subcategory="conditional",
            confidence=0.8,
            characteristics=["if_statement"]
        )
        
        strategy = await self.classifier.select_analysis_strategy(logic_bug)
        assert strategy.strategy_name == "logic_error_deep_analysis"
    
    def test_get_prompt_template(self):
        """Test getting appropriate prompt templates for bug types."""
        # Test logic error template
        logic_bug = BugType(category=BugCategory.LOGIC_ERROR, confidence=0.8)
        template = self.classifier.get_prompt_template(logic_bug)
        assert isinstance(template, PromptTemplate)
        
        # Test API issue template
        api_bug = BugType(category=BugCategory.API_ISSUE, confidence=0.8)
        template = self.classifier.get_prompt_template(api_bug)
        assert isinstance(template, PromptTemplate)
        
        # Test unknown category falls back to general template
        unknown_bug = BugType(category=BugCategory.CONCURRENCY, confidence=0.8)
        template = self.classifier.get_prompt_template(unknown_bug)
        assert isinstance(template, PromptTemplate)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in classification."""
        # Mock LLM to raise an exception
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        
        # Test that classification handles errors gracefully
        result = await self.classifier.classify_bug_type("Test issue")
        
        # Should return a default classification
        assert isinstance(result, ClassificationResult)
        assert result.bug_type.category == BugCategory.LOGIC_ERROR
        assert result.bug_type.confidence == 0.1
        assert "classification_failed" in result.bug_type.characteristics
        assert "Classification failed" in result.reasoning
    
    def test_parse_classification_response_valid_json(self):
        """Test parsing valid JSON classification response."""
        response = '{"category": "LOGIC_ERROR", "subcategory": "loop", "confidence": 0.8, "characteristics": ["infinite_loop"], "reasoning": "Loop condition is wrong"}'
        
        result = self.classifier._parse_classification_response(response)
        
        assert result["category"] == "LOGIC_ERROR"
        assert result["subcategory"] == "loop"
        assert result["confidence"] == 0.8
        assert result["characteristics"] == ["infinite_loop"]
    
    def test_parse_classification_response_invalid_json(self):
        """Test parsing invalid JSON classification response."""
        response = "This is not valid JSON"
        
        result = self.classifier._parse_classification_response(response)
        
        # Should return default values
        assert result["category"] == "LOGIC_ERROR"
        assert result["subcategory"] == "unknown"
        assert result["confidence"] == 0.1
        assert "parse_error" in result["characteristics"]
    
    def test_parse_classification_response_unknown_category(self):
        """Test parsing response with unknown category."""
        response = '{"category": "UNKNOWN_CATEGORY", "confidence": 0.8}'
        
        result = self.classifier._parse_classification_response(response)
        
        # Should default to LOGIC_ERROR for unknown categories
        assert result["category"] == "LOGIC_ERROR"
    
    def test_get_classification_stats_empty(self):
        """Test getting stats with no feedback history."""
        stats = self.classifier.get_classification_stats()
        
        assert stats["total_feedback"] == 0
        assert stats["accuracy"] == 0.0
    
    def test_update_classification_model(self):
        """Test updating classification model with feedback."""
        from .models import ClassificationFeedback
        
        original = BugType(category=BugCategory.LOGIC_ERROR, confidence=0.8)
        correct = BugType(category=BugCategory.API_ISSUE, confidence=0.9)
        
        feedback = ClassificationFeedback(
            original_classification=original,
            correct_classification=correct,
            issue_text="Test issue",
            feedback_notes="Should be API issue"
        )
        
        self.classifier.update_classification_model(feedback)
        
        assert len(self.classifier.feedback_history) == 1
        assert self.classifier.feedback_history[0] == feedback


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])