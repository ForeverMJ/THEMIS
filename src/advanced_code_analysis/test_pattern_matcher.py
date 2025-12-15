"""
Tests for the PatternMatcher module.

This module contains unit tests and property-based tests for the
pattern matching engine, including pattern detection, domain adaptation,
and specialized prompt generation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from .pattern_matcher import (
    PatternMatcher, PatternRule, RegexPatternDetector, 
    SemanticPatternDetector, DomainPattern
)
from .models import (
    BugPattern, BugCategory, ContextWindow, DependencyContext,
    DomainKnowledge, PromptTemplate, PatternGuidance
)
from .config import AdvancedAnalysisConfig, LLMConfig, StorageConfig
from .llm_interface import LLMInterface, LLMResponse


class TestPatternRule:
    """Test cases for PatternRule class."""
    
    def test_pattern_rule_creation(self):
        """Test creating a pattern rule."""
        rule = PatternRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            pattern_regex=r"\w+\s*=\s*0",
            bug_category=BugCategory.LOGIC_ERROR,
            confidence_weight=0.8
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.bug_category == BugCategory.LOGIC_ERROR
        assert rule.confidence_weight == 0.8
    
    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        rule = PatternRule(
            rule_id="assignment_zero",
            name="Assignment Zero",
            description="Variable assigned zero",
            pattern_regex=r"\w+\s*=\s*0",
            bug_category=BugCategory.LOGIC_ERROR
        )
        
        # Test positive match
        code_with_pattern = "result = 0"
        assert rule.matches(code_with_pattern)
        
        # Test negative match
        code_without_pattern = "result = calculate_sum()"
        assert not rule.matches(code_without_pattern)
    
    def test_invalid_regex_handling(self):
        """Test handling of invalid regex patterns."""
        rule = PatternRule(
            rule_id="invalid_regex",
            name="Invalid Regex",
            description="Rule with invalid regex",
            pattern_regex=r"[invalid",  # Invalid regex
            bug_category=BugCategory.LOGIC_ERROR
        )
        
        # Should not raise exception, should return False
        assert not rule.matches("any code")


class TestRegexPatternDetector:
    """Test cases for RegexPatternDetector class."""
    
    def test_pattern_detection(self):
        """Test basic pattern detection."""
        rules = [
            PatternRule(
                rule_id="assignment_zero",
                name="Assignment Zero",
                description="Variable assigned zero",
                pattern_regex=r"\w+\s*=\s*0",
                bug_category=BugCategory.LOGIC_ERROR,
                confidence_weight=0.8
            ),
            PatternRule(
                rule_id="self_assignment",
                name="Self Assignment",
                description="Variable assigned to itself",
                pattern_regex=r"(\w+)\s*=\s*\1",
                bug_category=BugCategory.LOGIC_ERROR,
                confidence_weight=0.9
            )
        ]
        
        detector = RegexPatternDetector(rules)
        context = ContextWindow(target_code="")
        
        code = """
        result = 0
        x = x
        """
        
        matches = detector.detect_patterns(code, context)
        
        assert len(matches) == 2
        assert matches[0][0].rule_id in ["assignment_zero", "self_assignment"]
        assert matches[1][0].rule_id in ["assignment_zero", "self_assignment"]
    
    def test_confidence_adjustment_with_context(self):
        """Test confidence adjustment based on context requirements."""
        rule = PatternRule(
            rule_id="context_dependent",
            name="Context Dependent",
            description="Rule requiring context",
            pattern_regex=r"function\(\)",
            bug_category=BugCategory.API_ISSUE,
            confidence_weight=1.0,
            context_requirements=["function_signature", "type_info"]
        )
        
        detector = RegexPatternDetector([rule])
        
        # Context with no matching requirements
        context_empty = ContextWindow(target_code="")
        matches_empty = detector.detect_patterns("function()", context_empty)
        assert matches_empty[0][1] == 0.5  # Minimum confidence
        
        # Context with matching requirements
        context_full = ContextWindow(
            target_code="",
            domain_concepts=["function_signature", "type_info"]
        )
        matches_full = detector.detect_patterns("function()", context_full)
        assert matches_full[0][1] == 1.0  # Full confidence


class TestSemanticPatternDetector:
    """Test cases for SemanticPatternDetector class."""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create a mock LLM interface."""
        mock_llm = Mock(spec=LLMInterface)
        mock_response = LLMResponse(
            content=json.dumps({
                "patterns": [
                    {
                        "type": "assignment_error",
                        "confidence": 0.8,
                        "explanation": "Variable assigned constant instead of computed value",
                        "lines": [1]
                    }
                ]
            }),
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        mock_llm.generate = AsyncMock(return_value=mock_response)
        return mock_llm
    
    def test_semantic_detection_creation(self, mock_llm_interface):
        """Test creating semantic pattern detector."""
        detector = SemanticPatternDetector(mock_llm_interface, [])
        assert detector.llm_interface == mock_llm_interface
        assert detector.patterns == []
    
    @pytest.mark.asyncio
    async def test_async_pattern_detection(self, mock_llm_interface):
        """Test async pattern detection."""
        detector = SemanticPatternDetector(mock_llm_interface, [])
        context = ContextWindow(target_code="")
        
        matches = await detector.detect_patterns_async("result = 0", context)
        
        assert len(matches) == 1
        assert matches[0][0].name == "Assignment Error"
        assert matches[0][1] == 0.8
    
    def test_sync_pattern_detection(self, mock_llm_interface):
        """Test synchronous wrapper for pattern detection."""
        detector = SemanticPatternDetector(mock_llm_interface, [])
        context = ContextWindow(target_code="")
        
        matches = detector.detect_patterns("result = 0", context)
        
        assert len(matches) == 1
        assert matches[0][0].name == "Assignment Error"
    
    def test_invalid_json_response_handling(self, mock_llm_interface):
        """Test handling of invalid JSON responses."""
        # Mock invalid JSON response
        mock_response = LLMResponse(
            content="Invalid JSON response",
            usage={"total_tokens": 50},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        mock_llm_interface.generate = AsyncMock(return_value=mock_response)
        
        detector = SemanticPatternDetector(mock_llm_interface, [])
        context = ContextWindow(target_code="")
        
        matches = detector.detect_patterns("code", context)
        assert matches == []


class TestPatternMatcher:
    """Test cases for PatternMatcher class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AdvancedAnalysisConfig()
            config.storage.patterns_db_path = str(Path(temp_dir) / "patterns.json")
            config.storage.cache_dir = str(Path(temp_dir) / "cache")
            yield config
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        mock_llm = Mock(spec=LLMInterface)
        mock_response = LLMResponse(
            content=json.dumps({"patterns": []}),
            usage={"total_tokens": 50},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        mock_llm.generate = AsyncMock(return_value=mock_response)
        return mock_llm
    
    def test_pattern_matcher_initialization(self, temp_config):
        """Test PatternMatcher initialization."""
        matcher = PatternMatcher(temp_config)
        
        assert matcher.config == temp_config
        assert len(matcher.pattern_rules) > 0  # Should have predefined rules
        assert len(matcher.prompt_templates) > 0  # Should have predefined templates
        assert isinstance(matcher.regex_detector, RegexPatternDetector)
    
    def test_pattern_matcher_with_llm(self, temp_config, mock_llm_interface):
        """Test PatternMatcher initialization with LLM."""
        matcher = PatternMatcher(temp_config, mock_llm_interface)
        
        assert matcher.llm_interface == mock_llm_interface
        assert matcher.semantic_detector is not None
    
    def test_detect_patterns_regex_only(self, temp_config):
        """Test pattern detection with regex only."""
        matcher = PatternMatcher(temp_config)
        
        code = """
        result = 0
        x = x
        if user_id = 123:
            pass
        """
        
        context = ContextWindow(target_code=code)
        matches = matcher.detect_patterns(code, context)
        
        assert len(matches) > 0
        # Should be sorted by confidence (highest first)
        if len(matches) > 1:
            assert matches[0][1] >= matches[1][1]
    
    def test_match_bug_patterns(self, temp_config):
        """Test matching against stored bug patterns."""
        # Lower the similarity threshold for testing
        temp_config.analysis.pattern_similarity_threshold = 0.1
        matcher = PatternMatcher(temp_config)
        
        # Store a test pattern
        test_pattern = BugPattern(
            pattern_id="test_pattern",
            problem_signature="variable assigned zero instead of sum",
            code_pattern="result = 0",
            fix_pattern="result = sum(values)",
            success_rate=0.8,
            applicable_domains=["testing"]
        )
        matcher.store_pattern(test_pattern)
        
        # Test matching
        issue_text = "The variable is assigned zero instead of calculating the sum"
        matches = matcher.match_bug_patterns(issue_text, "")
        
        assert len(matches) > 0
        assert matches[0].pattern_id == "test_pattern"
    
    def test_get_specialized_prompt(self, temp_config):
        """Test getting specialized prompts."""
        matcher = PatternMatcher(temp_config)
        
        # Test assignment error prompt
        prompt = matcher.get_specialized_prompt(
            BugCategory.LOGIC_ERROR,
            code="result = 0",
            context="Should calculate sum"
        )
        
        assert prompt is not None
        assert "assignment" in prompt.lower()
        assert "result = 0" in prompt
    
    def test_get_specialized_prompt_missing_vars(self, temp_config):
        """Test specialized prompt with missing template variables."""
        matcher = PatternMatcher(temp_config)
        
        # Test with missing required variable
        prompt = matcher.get_specialized_prompt(
            BugCategory.LOGIC_ERROR,
            code="result = 0"
            # Missing 'context' variable
        )
        
        assert prompt is None  # Should return None for missing variables
    
    def test_adapt_to_domain_web(self, temp_config):
        """Test domain adaptation for web development code."""
        matcher = PatternMatcher(temp_config)
        
        web_code = """
        from flask import Flask, request
        import requests
        
        app = Flask(__name__)
        
        @app.route('/api/users')
        def get_users():
            return jsonify([])
        
        class UserController:
            def authenticate(self, token):
                pass
        """
        
        context = ContextWindow(target_code=web_code)
        domain_pattern = matcher.adapt_to_domain(web_code, context)
        
        assert domain_pattern.domain_name == "web"
        assert "flask" in domain_pattern.typical_imports
        assert "requests" in domain_pattern.typical_imports
        assert "get_users" in domain_pattern.common_functions
        assert "UserController" in domain_pattern.keywords
    
    def test_adapt_to_domain_data_science(self, temp_config):
        """Test domain adaptation for data science code."""
        matcher = PatternMatcher(temp_config)
        
        ds_code = """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        def train_model(data):
            X = data.drop('target', axis=1)
            y = data['target']
            return model
        
        class DataProcessor:
            def preprocess(self, df):
                pass
        """
        
        context = ContextWindow(target_code=ds_code)
        domain_pattern = matcher.adapt_to_domain(ds_code, context)
        
        assert domain_pattern.domain_name == "data_science"
        assert "pandas" in domain_pattern.typical_imports
        assert "numpy" in domain_pattern.typical_imports
        assert "sklearn" in domain_pattern.typical_imports
    
    def test_store_and_load_patterns(self, temp_config):
        """Test storing and loading patterns from disk."""
        # Create first matcher and store pattern
        matcher1 = PatternMatcher(temp_config)
        
        test_pattern = BugPattern(
            pattern_id="persistent_pattern",
            problem_signature="test pattern for persistence",
            code_pattern="test = code",
            fix_pattern="test = fixed_code",
            success_rate=0.75,
            applicable_domains=["testing"]
        )
        
        matcher1.store_pattern(test_pattern)
        
        # Create second matcher and verify pattern is loaded
        matcher2 = PatternMatcher(temp_config)
        
        assert "persistent_pattern" in matcher2.bug_patterns
        loaded_pattern = matcher2.bug_patterns["persistent_pattern"]
        assert loaded_pattern.problem_signature == test_pattern.problem_signature
        assert loaded_pattern.success_rate == test_pattern.success_rate
    
    def test_create_pattern_guidance(self, temp_config):
        """Test creating pattern guidance."""
        matcher = PatternMatcher(temp_config)
        
        # Create test data
        matched_patterns = [
            BugPattern(
                pattern_id="test1",
                problem_signature="test pattern 1",
                code_pattern="code1",
                fix_pattern="fix1",
                success_rate=0.8,
                applicable_domains=["domain1"]
            )
        ]
        
        detected_rules = [
            (PatternRule(
                rule_id="rule1",
                name="Test Rule",
                description="Test rule",
                pattern_regex="",
                bug_category=BugCategory.LOGIC_ERROR,
                context_requirements=["req1"]
            ), 0.7)
        ]
        
        guidance = matcher.create_pattern_guidance(matched_patterns, detected_rules)
        
        assert guidance.confidence > 0
        assert len(guidance.matched_patterns) == 1
        assert "test1" in guidance.suggested_approach
        assert "domain1" in guidance.relevant_context
        assert "req1" in guidance.relevant_context
    
    def test_create_pattern_guidance_empty(self, temp_config):
        """Test creating pattern guidance with no patterns."""
        matcher = PatternMatcher(temp_config)
        
        guidance = matcher.create_pattern_guidance([], [])
        
        assert guidance.confidence == 0.0
        assert guidance.suggested_approach == "No patterns matched"
        assert len(guidance.matched_patterns) == 0
    
    def test_cleanup_patterns(self, temp_config):
        """Test pattern cleanup functionality."""
        # Set low max patterns for testing
        temp_config.analysis.max_stored_patterns = 3
        
        matcher = PatternMatcher(temp_config)
        
        # Add more patterns than the limit
        patterns = [
            BugPattern(
                pattern_id=f"pattern_{i}",
                problem_signature=f"pattern {i}",
                code_pattern=f"code{i}",
                fix_pattern=f"fix{i}",
                success_rate=0.5 + (i * 0.1),  # Varying success rates
                applicable_domains=["test"],
                usage_count=i
            )
            for i in range(5)
        ]
        
        for pattern in patterns:
            matcher.store_pattern(pattern)
        
        # Trigger cleanup
        matcher.cleanup_patterns()
        
        # Should keep only the best patterns (80% of max = 2.4 -> 2 patterns)
        assert len(matcher.bug_patterns) <= int(temp_config.analysis.max_stored_patterns * 0.8) + 1


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestPatternMatcherProperties:
        """Property-based tests for PatternMatcher."""
        
        @given(st.text(min_size=1, max_size=1000))
        def test_pattern_detection_never_crashes(self, code):
            """**Feature: advanced-code-analysis, Property 9: 预定义模式匹配**
            
            For any code input, pattern detection should never crash and should
            return a valid list of matches.
            **Validates: Requirements 3.3**
            """
            config = AdvancedAnalysisConfig()
            with tempfile.TemporaryDirectory() as temp_dir:
                config.storage.patterns_db_path = str(Path(temp_dir) / "patterns.json")
                
                matcher = PatternMatcher(config)
                context = ContextWindow(target_code=code)
                
                # Should not raise any exceptions
                matches = matcher.detect_patterns(code, context)
                
                # Should return a list
                assert isinstance(matches, list)
                
                # Each match should be a tuple of (PatternRule, float)
                for match in matches:
                    assert isinstance(match, tuple)
                    assert len(match) == 2
                    assert isinstance(match[0], PatternRule)
                    assert isinstance(match[1], (int, float))
                    assert 0.0 <= match[1] <= 1.0  # Confidence should be valid
        
        @given(st.text(min_size=1, max_size=500))
        def test_domain_adaptation_consistency(self, code):
            """**Feature: advanced-code-analysis, Property 10: 领域上下文适应**
            
            For any code input, domain adaptation should produce consistent
            results and identify valid domain characteristics.
            **Validates: Requirements 3.4**
            """
            config = AdvancedAnalysisConfig()
            with tempfile.TemporaryDirectory() as temp_dir:
                config.storage.patterns_db_path = str(Path(temp_dir) / "patterns.json")
                
                matcher = PatternMatcher(config)
                context = ContextWindow(target_code=code)
                
                # Should not raise any exceptions
                domain_pattern = matcher.adapt_to_domain(code, context)
                
                # Should return a valid DomainPattern
                assert isinstance(domain_pattern, DomainPattern)
                assert isinstance(domain_pattern.domain_name, str)
                assert isinstance(domain_pattern.keywords, set)
                assert isinstance(domain_pattern.common_functions, set)
                assert isinstance(domain_pattern.typical_imports, set)
                
                # Domain name should not be empty
                assert len(domain_pattern.domain_name) > 0
        
        @given(
            st.text(min_size=1, max_size=200),
            st.text(min_size=1, max_size=200)
        )
        def test_pattern_matching_symmetry(self, issue1, issue2):
            """Test that pattern matching behaves consistently."""
            config = AdvancedAnalysisConfig()
            with tempfile.TemporaryDirectory() as temp_dir:
                config.storage.patterns_db_path = str(Path(temp_dir) / "patterns.json")
                
                matcher = PatternMatcher(config)
                
                # Add a test pattern
                test_pattern = BugPattern(
                    pattern_id="test_symmetry",
                    problem_signature="test pattern",
                    code_pattern="test",
                    fix_pattern="fixed",
                    success_rate=0.5,
                    applicable_domains=["test"]
                )
                matcher.store_pattern(test_pattern)
                
                # Pattern matching should be deterministic
                matches1_first = matcher.match_bug_patterns(issue1, "")
                matches1_second = matcher.match_bug_patterns(issue1, "")
                
                # Same input should produce same results
                assert len(matches1_first) == len(matches1_second)
                if matches1_first:
                    assert matches1_first[0].pattern_id == matches1_second[0].pattern_id

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    pytest.main([__file__])