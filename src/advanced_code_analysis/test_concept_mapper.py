"""
Tests for the ConceptMapper class.

This module contains unit tests and property-based tests for the multi-level
concept mapping engine functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from .concept_mapper import ConceptMapper, MatchResult, SearchCandidate, ConceptMappingResult
from .models import (
    ContextWindow, 
    BugType, 
    BugCategory, 
    DependencyContext,
    DomainKnowledge
)
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface, LLMResponse


class TestConceptMapper:
    """Test suite for ConceptMapper functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AdvancedAnalysisConfig()
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        mock_llm = Mock(spec=LLMInterface)
        mock_llm.generate = AsyncMock()
        return mock_llm
    
    @pytest.fixture
    def mock_context_enhancer(self):
        """Create mock context enhancer."""
        from .context_enhancer import ContextEnhancer
        return Mock(spec=ContextEnhancer)
    
    @pytest.fixture
    def concept_mapper(self, config, mock_llm_interface, mock_context_enhancer):
        """Create ConceptMapper instance for testing."""
        return ConceptMapper(
            config=config,
            llm_interface=mock_llm_interface,
            context_enhancer=mock_context_enhancer
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context window for testing."""
        return ContextWindow(
            target_code="""
def authenticate_user(username, password):
    '''Authenticate user with username and password.'''
    if not username or not password:
        return False
    return check_credentials(username, password)

class UserManager:
    '''Manages user operations.'''
    def create_user(self, username, email):
        return User(username, email)
    
    def delete_user(self, user_id):
        return remove_user_from_db(user_id)

def process_login_request(request):
    username = request.get('username')
    password = request.get('password')
    return authenticate_user(username, password)
""",
            related_functions=[
                "check_credentials(username, password)",
                "remove_user_from_db(user_id)",
                "validate_email(email)"
            ],
            class_hierarchy={
                "UserManager": ["BaseManager"],
                "User": ["BaseModel"]
            },
            module_dependencies=[
                "hashlib",
                "database.models",
                "auth.validators"
            ],
            domain_concepts=[
                "Authentication",
                "User",
                "Login",
                "Password"
            ],
            dependency_context=DependencyContext(
                function_signatures={
                    "authenticate_user": "def authenticate_user(username: str, password: str) -> bool",
                    "check_credentials": "def check_credentials(username: str, password: str) -> bool"
                },
                class_methods={
                    "UserManager": ["create_user", "delete_user"]
                },
                call_graph={
                    "authenticate_user": ["check_credentials"],
                    "process_login_request": ["authenticate_user"]
                }
            )
        )
    
    @pytest.mark.asyncio
    async def test_exact_matching(self, concept_mapper, sample_context):
        """Test exact matching functionality."""
        # Setup
        concepts = ["authenticate_user", "UserManager"]
        issue_description = "Login function not working properly"
        
        # Mock LLM response for conceptual matching
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content='{"matches": []}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify
        assert isinstance(result, ConceptMappingResult)
        assert len(result.primary_matches) >= 2  # Should find exact matches
        
        # Check for exact matches
        exact_matches = [m for m in result.primary_matches if m.match_type == "exact"]
        assert len(exact_matches) >= 2
        
        # Verify specific matches
        match_names = [m.element_name for m in exact_matches]
        assert "authenticate_user" in match_names
        assert "UserManager" in match_names
        
        # Check confidence scores
        for match in exact_matches:
            assert match.confidence == 1.0
            assert match.similarity_score == 1.0
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching(self, concept_mapper, sample_context):
        """Test fuzzy matching functionality."""
        # Setup
        concepts = ["auth", "user_mgr", "login"]  # Partial matches
        issue_description = "Authentication system has bugs"
        
        # Mock LLM response
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content='{"matches": []}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify
        fuzzy_matches = [m for m in result.primary_matches if m.match_type == "fuzzy"]
        assert len(fuzzy_matches) > 0
        
        # Check that fuzzy matches have reasonable confidence
        for match in fuzzy_matches:
            assert 0.2 <= match.confidence < 1.0  # Lower threshold for fuzzy matches
            assert match.similarity_score >= 0.2  # Should have some similarity
    
    @pytest.mark.asyncio
    async def test_conceptual_matching(self, concept_mapper, sample_context):
        """Test conceptual matching using LLM."""
        # Setup
        concepts = ["security", "validation"]
        issue_description = "User login validation is broken"
        
        # Mock LLM response with conceptual matches
        llm_response = '''
{
    "matches": [
        {
            "element_number": 1,
            "confidence": 0.85,
            "explanation": "This function handles user authentication which is related to security"
        },
        {
            "element_number": 3,
            "confidence": 0.75,
            "explanation": "This function processes login requests which involves validation"
        }
    ]
}
'''
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content=llm_response,
            usage={"total_tokens": 200},
            model="test-model",
            finish_reason="stop",
            response_time=0.2
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify
        conceptual_matches = [m for m in result.primary_matches if m.match_type == "conceptual"]
        assert len(conceptual_matches) >= 1
        
        # Check confidence scores
        for match in conceptual_matches:
            assert match.confidence >= concept_mapper.conceptual_match_threshold
            assert len(match.evidence) > 0
    
    @pytest.mark.asyncio
    async def test_hierarchical_search(self, concept_mapper, sample_context):
        """Test hierarchical search strategy."""
        # Setup
        concepts = ["auth", "database"]  # Should match at different levels
        issue_description = "Database authentication issues"
        
        # Mock LLM response
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content='{"matches": []}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify hierarchical search found matches at different levels
        assert len(result.primary_matches) > 0
        
        # Should find matches at module level (database.models)
        module_matches = [m for m in result.primary_matches if m.element_type == "module"]
        assert len(module_matches) > 0
        
        # Should find matches at function level (authenticate_user)
        function_matches = [m for m in result.primary_matches if m.element_type == "function"]
        assert len(function_matches) > 0
    
    @pytest.mark.asyncio
    async def test_alternative_candidate_generation(self, concept_mapper, sample_context):
        """Test generation of alternative candidates."""
        # Setup
        concepts = ["login", "user"]
        issue_description = "Login system crashes when user enters invalid credentials"
        
        # Mock LLM responses
        pattern_response = '''
{
    "suggestions": [
        {
            "location": "error handling in authentication",
            "reasoning": "Crashes often occur due to unhandled exceptions in auth code",
            "confidence": 0.8
        },
        {
            "location": "input validation functions",
            "reasoning": "Invalid credentials suggest input validation issues",
            "confidence": 0.7
        }
    ]
}
'''
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content=pattern_response,
            usage={"total_tokens": 150},
            model="test-model",
            finish_reason="stop",
            response_time=0.15
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify
        assert len(result.alternative_candidates) > 0
        
        # Check candidate properties
        for candidate in result.alternative_candidates:
            assert isinstance(candidate, SearchCandidate)
            assert candidate.confidence > 0.0
            assert len(candidate.reasoning) > 0
            assert len(candidate.location) > 0
    
    @pytest.mark.asyncio
    async def test_evidence_chain_construction(self, concept_mapper, sample_context):
        """Test evidence chain and reasoning path construction."""
        # Setup
        concepts = ["authenticate_user"]
        issue_description = "Authentication function returns wrong result"
        
        # Mock LLM response
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content='{"matches": []}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify evidence chain
        assert result.evidence_chain is not None
        assert len(result.evidence_chain.evidence_items) > 0
        assert len(result.evidence_chain.source_locations) > 0
        assert len(result.evidence_chain.reasoning_path) > 0
        
        # Verify reasoning chain
        assert result.reasoning_chain is not None
        assert len(result.reasoning_chain.steps) >= 4  # Should have 4 main steps
        assert result.overall_confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, concept_mapper, sample_context):
        """Test confidence score calculations."""
        # Setup
        concepts = ["authenticate_user", "nonexistent_function"]
        issue_description = "Function not working"
        
        # Mock LLM response
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content='{"matches": []}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        
        # Execute
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify confidence calculations
        assert 0.0 <= result.overall_confidence <= 1.0
        
        # Should have high confidence for exact match
        exact_matches = [m for m in result.primary_matches if m.match_type == "exact"]
        if exact_matches:
            assert max(m.confidence for m in exact_matches) == 1.0
        
        # Reasoning chain confidence should be calculated
        assert result.reasoning_chain.overall_confidence > 0.0
    
    def test_match_deduplication(self, concept_mapper):
        """Test deduplication of matches."""
        # Create duplicate matches
        matches = [
            MatchResult(
                element_name="test_func",
                element_type="function",
                file_path="test.py",
                line_number=10,
                match_type="exact",
                confidence=1.0
            ),
            MatchResult(
                element_name="test_func",
                element_type="function", 
                file_path="test.py",
                line_number=10,
                match_type="fuzzy",
                confidence=0.8
            ),
            MatchResult(
                element_name="other_func",
                element_type="function",
                file_path="test.py", 
                line_number=20,
                match_type="exact",
                confidence=1.0
            )
        ]
        
        # Execute deduplication
        deduplicated = concept_mapper._deduplicate_matches(matches)
        
        # Verify
        assert len(deduplicated) == 2  # Should remove one duplicate
        names = [m.element_name for m in deduplicated]
        assert "test_func" in names
        assert "other_func" in names
    
    def test_code_element_extraction(self, concept_mapper, sample_context):
        """Test extraction of code elements from context."""
        # Execute
        elements = concept_mapper._extract_code_elements(sample_context)
        
        # Verify structure
        assert "functions" in elements
        assert "classes" in elements
        assert "variables" in elements
        assert "modules" in elements
        
        # Verify function extraction
        function_names = [f['name'] for f in elements['functions']]
        assert "authenticate_user" in function_names
        assert "process_login_request" in function_names
        
        # Verify class extraction
        class_names = [c['name'] for c in elements['classes']]
        assert "UserManager" in class_names
        
        # Verify module extraction
        module_names = [m['name'] for m in elements['modules']]
        assert "hashlib" in module_names
        assert "database.models" in module_names
    
    @pytest.mark.asyncio
    async def test_error_handling(self, concept_mapper):
        """Test error handling in concept mapping."""
        # Setup with invalid context
        invalid_context = ContextWindow(target_code="invalid python syntax !!!")
        concepts = ["test"]
        issue_description = "Test issue"
        
        # Mock LLM response
        concept_mapper.llm_interface.generate.return_value = LLMResponse(
            content='{"matches": []}',
            usage={"total_tokens": 100},
            model="test-model",
            finish_reason="stop",
            response_time=0.1
        )
        
        # Execute - should not raise exception
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, invalid_context
        )
        
        # Verify graceful handling
        assert isinstance(result, ConceptMappingResult)
        # Should still return a result even with invalid input
    
    @pytest.mark.asyncio
    async def test_llm_error_handling(self, concept_mapper, sample_context):
        """Test handling of LLM errors."""
        # Setup
        concepts = ["test"]
        issue_description = "Test issue"
        
        # Mock LLM to raise exception
        concept_mapper.llm_interface.generate.side_effect = Exception("LLM error")
        
        # Execute - should not raise exception
        result = await concept_mapper.map_concepts_to_code(
            issue_description, concepts, sample_context
        )
        
        # Verify graceful handling
        assert isinstance(result, ConceptMappingResult)
        # Should still find non-LLM matches (exact, fuzzy)
        non_conceptual_matches = [m for m in result.primary_matches if m.match_type != "conceptual"]
        # May or may not have matches depending on concepts, but should not crash


# Property-based tests would go here if using hypothesis
# For now, we'll add some additional integration-style tests

class TestConceptMapperIntegration:
    """Integration tests for ConceptMapper with realistic scenarios."""
    
    @pytest.fixture
    def real_config(self):
        """Create realistic configuration."""
        config = AdvancedAnalysisConfig()
        config.llm.provider = "mock"  # Use mock provider for testing
        return config
    
    @pytest.fixture
    def real_concept_mapper(self, real_config):
        """Create ConceptMapper with real dependencies."""
        from .llm_interface import LLMInterface
        from .context_enhancer import ContextEnhancer
        
        llm_interface = LLMInterface(real_config.llm)
        context_enhancer = ContextEnhancer(real_config)
        
        return ConceptMapper(
            config=real_config,
            llm_interface=llm_interface,
            context_enhancer=context_enhancer
        )
    
    @pytest.mark.asyncio
    async def test_realistic_bug_scenario(self, real_concept_mapper):
        """Test with a realistic bug scenario."""
        # Setup realistic scenario
        issue_description = """
        The user authentication system is failing when users try to log in with valid credentials.
        The authenticate_user function seems to return False even for correct username/password combinations.
        This started happening after we updated the password hashing algorithm.
        """
        
        concepts = ["authenticate_user", "password", "hashing", "credentials"]
        
        context = ContextWindow(
            target_code="""
import hashlib
import bcrypt

def authenticate_user(username, password):
    '''Authenticate user with username and password.'''
    user = get_user_by_username(username)
    if not user:
        return False
    
    # Bug: using wrong hashing method
    hashed_password = hashlib.md5(password.encode()).hexdigest()
    return user.password_hash == hashed_password

def hash_password(password):
    '''Hash password using bcrypt.'''
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def get_user_by_username(username):
    '''Get user from database by username.'''
    # Database lookup logic here
    pass
""",
            related_functions=[
                "get_user_by_username(username)",
                "hash_password(password)",
                "verify_password(password, hash)"
            ],
            module_dependencies=["hashlib", "bcrypt", "database.models"],
            domain_concepts=["Authentication", "Hashing", "Security"]
        )
        
        # Set up mock LLM responses
        if hasattr(real_concept_mapper.llm_interface.provider, 'set_responses'):
            real_concept_mapper.llm_interface.provider.set_responses([
                '''
{
    "matches": [
        {
            "element_number": 1,
            "confidence": 0.9,
            "explanation": "This function handles user authentication and contains password hashing logic"
        }
    ]
}
''',
                '''
{
    "suggestions": [
        {
            "location": "password hashing function",
            "reasoning": "The issue mentions password hashing algorithm changes",
            "confidence": 0.85
        }
    ]
}
'''
            ])
        
        # Execute
        result = await real_concept_mapper.map_concepts_to_code(
            issue_description, concepts, context
        )
        
        # Verify realistic results
        assert result.overall_confidence > 0.5
        assert len(result.primary_matches) > 0
        
        # Should find the authenticate_user function
        auth_matches = [m for m in result.primary_matches if "authenticate" in m.element_name.lower()]
        assert len(auth_matches) > 0
        
        # Should have evidence and reasoning
        assert len(result.evidence_chain.evidence_items) > 0
        assert len(result.reasoning_chain.steps) >= 4