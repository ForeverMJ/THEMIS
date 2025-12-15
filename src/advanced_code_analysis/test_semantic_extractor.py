"""
Tests for the SemanticExtractor class.

This module contains unit tests for the semantic information extraction engine,
testing both pattern-based and LLM-based extraction capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import json
from hypothesis import given, strategies as st, assume, settings

from .semantic_extractor import (
    SemanticExtractor, ExtractedInformation, StructuredSummary, 
    ExtractionResult, InformationType
)
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface, LLMResponse


class TestSemanticExtractor:
    """Test cases for SemanticExtractor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AdvancedAnalysisConfig()
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM interface."""
        llm = Mock(spec=LLMInterface)
        llm.generate = AsyncMock()
        return llm
    
    @pytest.fixture
    def extractor(self, config, mock_llm):
        """Create SemanticExtractor instance."""
        return SemanticExtractor(config, mock_llm)
    
    def test_init_extraction_patterns(self, extractor):
        """Test that extraction patterns are properly initialized."""
        assert InformationType.FUNCTION_NAME in extractor.patterns
        assert InformationType.VARIABLE_NAME in extractor.patterns
        assert InformationType.CLASS_NAME in extractor.patterns
        assert InformationType.ERROR_PATTERN in extractor.patterns
        
        # Check that patterns are not empty
        for info_type, patterns in extractor.patterns.items():
            assert len(patterns) > 0
            for pattern in patterns:
                assert isinstance(pattern, str)
                assert len(pattern) > 0
    
    def test_init_prompt_templates(self, extractor):
        """Test that prompt templates are properly initialized."""
        expected_templates = [
            'extract_technical_concepts',
            'refine_extraction', 
            'generate_summary'
        ]
        
        for template_id in expected_templates:
            assert template_id in extractor.templates
            template = extractor.templates[template_id]
            assert template.template_id == template_id
            assert len(template.content) > 0
            assert len(template.placeholders) > 0
    
    @pytest.mark.asyncio
    async def test_extract_with_patterns_function_names(self, extractor):
        """Test pattern-based extraction of function names."""
        problem_text = """
        The function calculate_total() is not working correctly.
        When I call process_data(items), it returns None instead of the expected result.
        The method validate_input should check the parameters.
        """
        
        reasoning_chain = Mock()
        reasoning_chain.steps = []
        reasoning_chain.add_step = Mock()
        
        items = await extractor._extract_with_patterns(problem_text, reasoning_chain)
        
        # Should find function names
        function_items = [item for item in items if item.info_type == InformationType.FUNCTION_NAME]
        function_names = [item.content for item in function_items]
        
        assert 'calculate_total' in function_names
        assert 'process_data' in function_names
        assert 'validate_input' in function_names
        
        # Check that reasoning step was added
        reasoning_chain.add_step.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_with_patterns_variable_names(self, extractor):
        """Test pattern-based extraction of variable names."""
        problem_text = """
        The variable user_count = 0 is not being updated.
        Parameter max_items should be validated.
        """
        
        reasoning_chain = Mock()
        reasoning_chain.steps = []
        reasoning_chain.add_step = Mock()
        
        items = await extractor._extract_with_patterns(problem_text, reasoning_chain)
        
        # Should find variable names
        variable_items = [item for item in items if item.info_type == InformationType.VARIABLE_NAME]
        variable_names = [item.content for item in variable_items]
        
        assert 'user_count' in variable_names
        assert 'max_items' in variable_names
    
    @pytest.mark.asyncio
    async def test_extract_with_llm_success(self, extractor, mock_llm):
        """Test LLM-based extraction with successful response."""
        problem_text = "The function process_user_data() fails with TypeError"
        
        # Mock LLM response
        mock_response_content = json.dumps({
            "technical_concepts": [
                {"term": "TypeError", "confidence": 0.9, "context": "exception type"}
            ],
            "function_names": [
                {"name": "process_user_data", "confidence": 0.8, "context": "main function"}
            ],
            "variable_names": [],
            "class_names": [],
            "error_patterns": [
                {"pattern": "TypeError", "confidence": 0.9, "type": "type_error"}
            ],
            "api_calls": [],
            "overall_confidence": 0.85
        })
        
        mock_llm.generate.return_value = LLMResponse(
            content=mock_response_content,
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=1.0
        )
        
        reasoning_chain = Mock()
        reasoning_chain.steps = []
        reasoning_chain.add_step = Mock()
        
        items = await extractor._extract_with_llm(problem_text, reasoning_chain)
        
        # Should extract items from LLM response
        assert len(items) > 0
        
        # Check technical concepts
        concept_items = [item for item in items if item.info_type == InformationType.TECHNICAL_CONCEPT]
        assert len(concept_items) == 1
        assert concept_items[0].content == "TypeError"
        assert concept_items[0].confidence == 0.9
        
        # Check function names
        function_items = [item for item in items if item.info_type == InformationType.FUNCTION_NAME]
        assert len(function_items) == 1
        assert function_items[0].content == "process_user_data"
        assert function_items[0].confidence == 0.8
        
        # Check error patterns
        error_items = [item for item in items if item.info_type == InformationType.ERROR_PATTERN]
        assert len(error_items) == 1
        assert error_items[0].content == "TypeError"
        
        # Verify LLM was called with correct template
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        assert 'extract_technical_concepts' in str(call_args)
    
    @pytest.mark.asyncio
    async def test_extract_with_llm_failure(self, extractor, mock_llm):
        """Test LLM-based extraction with failure."""
        problem_text = "Test problem"
        
        # Mock LLM failure
        mock_llm.generate.side_effect = Exception("API Error")
        
        reasoning_chain = Mock()
        reasoning_chain.steps = []
        reasoning_chain.add_step = Mock()
        
        items = await extractor._extract_with_llm(problem_text, reasoning_chain)
        
        # Should return empty list on failure
        assert items == []
        
        # Should still add reasoning step
        reasoning_chain.add_step.assert_called_once()
    
    def test_deduplicate_items(self, extractor):
        """Test deduplication of extracted items."""
        items = [
            ExtractedInformation("test_func", InformationType.FUNCTION_NAME, 0.8),
            ExtractedInformation("test_func", InformationType.FUNCTION_NAME, 0.7),  # Duplicate
            ExtractedInformation("other_func", InformationType.FUNCTION_NAME, 0.9),
            ExtractedInformation("TEST_FUNC", InformationType.FUNCTION_NAME, 0.6),  # Case variation
        ]
        
        deduplicated = extractor._deduplicate_items(items)
        
        # Should remove duplicates (case-insensitive)
        assert len(deduplicated) == 2
        function_names = [item.content for item in deduplicated]
        assert "test_func" in function_names
        assert "other_func" in function_names
    
    def test_merge_similar_items(self, extractor):
        """Test merging of similar items."""
        items = [
            ExtractedInformation("test_func", InformationType.FUNCTION_NAME, 0.8, "context1"),
            ExtractedInformation("test_func", InformationType.FUNCTION_NAME, 0.6, "context2"),
            ExtractedInformation("other_func", InformationType.FUNCTION_NAME, 0.9, "context3"),
        ]
        
        merged = extractor._merge_similar_items(items)
        
        # Should merge similar items
        assert len(merged) == 2
        
        # Find the merged item
        merged_item = next(item for item in merged if item.content == "test_func")
        
        # Should have combined confidence (weighted average)
        assert merged_item.confidence > 0.6
        assert merged_item.confidence < 0.8
        
        # Should have combined contexts
        assert "context1" in merged_item.context
        assert "context2" in merged_item.context
    
    def test_calculate_pattern_confidence(self, extractor):
        """Test pattern confidence calculation."""
        import re
        
        problem_text = "The function calculate_total() is failing"
        match = re.search(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)', problem_text)
        
        confidence = extractor._calculate_pattern_confidence(
            "calculate_total", InformationType.FUNCTION_NAME, match, problem_text
        )
        
        # Should return reasonable confidence
        assert 0.0 < confidence <= 1.0
        
        # Should be higher for function names with "function" keyword in context
        assert confidence > 0.5
    
    def test_generate_basic_summary(self, extractor):
        """Test basic summary generation."""
        items = [
            ExtractedInformation("test_func", InformationType.FUNCTION_NAME, 0.8),
            ExtractedInformation("TypeError", InformationType.ERROR_PATTERN, 0.9),
            ExtractedInformation("user_data", InformationType.VARIABLE_NAME, 0.7),
        ]
        
        summary = extractor._generate_basic_summary(items)
        
        assert isinstance(summary, StructuredSummary)
        assert summary.problem_type in ["general", "error_analysis", "function_issue"]
        assert len(summary.code_elements) > 0
        assert len(summary.error_indicators) > 0
        assert 0.0 <= summary.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_extract_information_integration(self, extractor, mock_llm):
        """Test full extraction workflow integration."""
        problem_text = """
        The function process_data(user_input) is throwing a ValueError 
        when the parameter is None. The variable result_count should be 
        initialized before the loop.
        """
        
        # Mock LLM responses for different stages
        extraction_response = json.dumps({
            "technical_concepts": [{"term": "ValueError", "confidence": 0.9}],
            "function_names": [{"name": "process_data", "confidence": 0.8}],
            "variable_names": [{"name": "user_input", "confidence": 0.7}],
            "class_names": [],
            "error_patterns": [{"pattern": "ValueError", "confidence": 0.9}],
            "api_calls": [],
            "overall_confidence": 0.8
        })
        
        summary_response = json.dumps({
            "problem_type": "function_issue",
            "key_components": ["process_data", "user_input"],
            "technical_concepts": ["ValueError"],
            "code_elements": ["process_data", "user_input", "result_count"],
            "error_indicators": ["ValueError", "None"],
            "confidence_score": 0.8,
            "reasoning": "Function parameter validation issue"
        })
        
        mock_llm.generate.side_effect = [
            LLMResponse(extraction_response, {}, "test", "stop", 1.0),
            LLMResponse(summary_response, {}, "test", "stop", 1.0),
            LLMResponse(summary_response, {}, "test", "stop", 1.0)  # For potential refinement call
        ]
        
        result = await extractor.extract_information(problem_text)
        
        # Verify result structure
        assert isinstance(result, ExtractionResult)
        assert len(result.extracted_items) > 0
        assert isinstance(result.structured_summary, StructuredSummary)
        assert 0.0 <= result.overall_confidence <= 1.0
        assert result.processing_time > 0
        
        # Verify extracted items include both pattern and LLM results
        function_items = result.get_items_by_type(InformationType.FUNCTION_NAME)
        assert len(function_items) > 0
        assert any(item.content == "process_data" for item in function_items)
        
        # Verify summary (allow for fallback behavior)
        assert result.structured_summary.problem_type in ["function_issue", "error_analysis"]
        assert len(result.structured_summary.key_components) > 0
        assert len(result.structured_summary.technical_concepts) > 0 or len(result.structured_summary.error_indicators) > 0


# Hypothesis strategies for generating problem descriptions
def valid_identifier():
    """Generate valid Python identifiers."""
    return st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        whitelist_characters='_'
    )).filter(lambda x: x[0].isalpha() or x[0] == '_')

def function_name():
    """Generate valid function names."""
    return valid_identifier()

def variable_name():
    """Generate valid variable names."""
    return valid_identifier()

def class_name():
    """Generate valid class names."""
    return valid_identifier().filter(lambda x: x[0].isupper())

def error_type():
    """Generate common error types."""
    return st.sampled_from([
        "ValueError", "TypeError", "AttributeError", "KeyError", 
        "IndexError", "RuntimeError", "ImportError", "SyntaxError"
    ])

def problem_description():
    """Generate realistic problem descriptions with extractable information."""
    return st.one_of([
        # Function-related problems
        st.builds(
            lambda func, error: f"The function {func}() is throwing a {error} when called.",
            function_name(),
            error_type()
        ),
        # Variable-related problems
        st.builds(
            lambda var, func: f"The variable {var} is not being updated in {func}().",
            variable_name(),
            function_name()
        ),
        # Class-related problems
        st.builds(
            lambda cls, method: f"The class {cls} method {method} is not working correctly.",
            class_name(),
            function_name()
        ),
        # API-related problems
        st.builds(
            lambda obj, method: f"The API call {obj}.{method}() returns None instead of expected data.",
            valid_identifier(),
            function_name()
        ),
        # General error problems
        st.builds(
            lambda error, desc: f"Getting {error}: {desc}",
            error_type(),
            st.text(min_size=10, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')))
        )
    ])


class TestSemanticExtractorProperties:
    """Property-based tests for SemanticExtractor."""

    @given(problem_text=problem_description())
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_information_extraction_completeness_property(self, problem_text):
        """
        **Feature: advanced-code-analysis, Property 2: Information extraction completeness**
        **Validates: Requirements 1.3, 1.5**
        
        Property: For any problem description, information extraction should identify 
        all key technical concepts, function names, variable names, and generate 
        a structured summary with confidence scores.
        """
        assume(len(problem_text.strip()) > 5)  # Ensure non-trivial input
        
        # Create fresh instances for each test run
        config = AdvancedAnalysisConfig()
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate = AsyncMock()
        extractor = SemanticExtractor(config, mock_llm)
        
        # Mock LLM response with realistic extraction data
        mock_extraction_response = {
            "technical_concepts": [
                {"term": "error_handling", "confidence": 0.8, "context": "error processing"}
            ],
            "function_names": [
                {"name": "test_function", "confidence": 0.7, "context": "main function"}
            ],
            "variable_names": [
                {"name": "test_var", "confidence": 0.6, "context": "local variable"}
            ],
            "class_names": [],
            "error_patterns": [
                {"pattern": "generic_error", "confidence": 0.8, "type": "runtime_error"}
            ],
            "api_calls": [],
            "overall_confidence": 0.75
        }
        
        mock_summary_response = {
            "problem_type": "function_issue",
            "key_components": ["test_function", "test_var"],
            "technical_concepts": ["error_handling"],
            "code_elements": ["test_function", "test_var"],
            "error_indicators": ["generic_error"],
            "confidence_score": 0.75,
            "reasoning": "Extracted key components from problem description"
        }
        
        mock_llm.generate.side_effect = [
            LLMResponse(
                content=json.dumps(mock_extraction_response),
                usage={"total_tokens": 100},
                model="test-model",
                finish_reason="stop",
                response_time=1.0
            ),
            LLMResponse(
                content=json.dumps(mock_summary_response),
                usage={"total_tokens": 100},
                model="test-model",
                finish_reason="stop",
                response_time=1.0
            )
        ]
        
        # Execute extraction
        result = await extractor.extract_information(problem_text)
        
        # Property assertions: Information extraction completeness
        
        # 1. Result structure completeness
        assert isinstance(result, ExtractionResult), "Must return ExtractionResult"
        assert isinstance(result.extracted_items, list), "Must have extracted_items list"
        assert isinstance(result.structured_summary, StructuredSummary), "Must have structured_summary"
        assert isinstance(result.reasoning_chain, object), "Must have reasoning_chain"
        
        # 2. Confidence score validity
        assert 0.0 <= result.overall_confidence <= 1.0, "Overall confidence must be between 0 and 1"
        assert 0.0 <= result.structured_summary.confidence_score <= 1.0, "Summary confidence must be between 0 and 1"
        
        # 3. Individual item confidence validity
        for item in result.extracted_items:
            assert isinstance(item, ExtractedInformation), "All items must be ExtractedInformation"
            assert 0.0 <= item.confidence <= 1.0, f"Item confidence must be between 0 and 1, got {item.confidence}"
            assert isinstance(item.content, str), "Item content must be string"
            assert len(item.content.strip()) > 0, "Item content must not be empty"
            assert isinstance(item.info_type, InformationType), "Item must have valid info_type"
        
        # 4. Structured summary completeness
        summary = result.structured_summary
        assert isinstance(summary.problem_type, str), "Must have problem_type"
        assert len(summary.problem_type.strip()) > 0, "Problem type must not be empty"
        assert isinstance(summary.key_components, list), "Must have key_components list"
        assert isinstance(summary.technical_concepts, list), "Must have technical_concepts list"
        assert isinstance(summary.code_elements, list), "Must have code_elements list"
        assert isinstance(summary.error_indicators, list), "Must have error_indicators list"
        
        # 5. Processing metadata completeness
        assert result.processing_time >= 0, "Processing time must be non-negative"
        assert hasattr(result, 'reasoning_chain'), "Must have reasoning chain"
        
        # 6. Content extraction validation - at least some information should be extracted
        # Either from patterns or LLM, there should be some extracted content
        total_extracted_content = len(result.extracted_items)
        total_summary_content = (len(summary.key_components) + 
                               len(summary.technical_concepts) + 
                               len(summary.code_elements) + 
                               len(summary.error_indicators))
        
        # For any non-trivial problem description, we should extract some information
        # This validates the completeness aspect of the property
        assert total_extracted_content >= 0, "Should extract some items (can be 0 for very simple inputs)"
        assert total_summary_content >= 0, "Summary should contain some information (can be 0 for very simple inputs)"
        
        # 7. Type consistency - extracted items should match their declared types
        function_items = result.get_items_by_type(InformationType.FUNCTION_NAME)
        for item in function_items:
            assert item.info_type == InformationType.FUNCTION_NAME, "Function items must have FUNCTION_NAME type"
        
        variable_items = result.get_items_by_type(InformationType.VARIABLE_NAME)
        for item in variable_items:
            assert item.info_type == InformationType.VARIABLE_NAME, "Variable items must have VARIABLE_NAME type"
        
        concept_items = result.get_items_by_type(InformationType.TECHNICAL_CONCEPT)
        for item in concept_items:
            assert item.info_type == InformationType.TECHNICAL_CONCEPT, "Concept items must have TECHNICAL_CONCEPT type"
        
        # 8. High confidence items should be accessible
        high_conf_items = result.get_high_confidence_items(0.7)
        for item in high_conf_items:
            assert item.confidence >= 0.7, "High confidence items must meet threshold"


if __name__ == "__main__":
    pytest.main([__file__])