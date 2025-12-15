"""
Example usage of the ConceptMapper class.

This script demonstrates how to use the multi-level concept mapping engine
to map problem descriptions to specific code locations.
"""

import asyncio
from typing import List

from .concept_mapper import ConceptMapper
from .models import ContextWindow, BugType, BugCategory, DependencyContext
from .config import AdvancedAnalysisConfig
from .llm_interface import LLMInterface


async def main():
    """Demonstrate ConceptMapper usage with realistic examples."""
    
    print("=== ConceptMapper Example Usage ===\n")
    
    # Setup configuration
    config = AdvancedAnalysisConfig()
    config.llm.provider = "mock"  # Use mock provider for demo
    
    # Create LLM interface
    llm_interface = LLMInterface(config.llm)
    
    # Set up mock responses for demonstration
    if hasattr(llm_interface.provider, 'set_responses'):
        llm_interface.provider.set_responses([
            # Conceptual matching response
            '''
{
    "matches": [
        {
            "element_number": 1,
            "confidence": 0.9,
            "explanation": "This function handles user authentication which is directly related to login security issues"
        },
        {
            "element_number": 3,
            "confidence": 0.8,
            "explanation": "This function processes login requests and may contain validation logic"
        }
    ]
}
''',
            # Pattern matching response
            '''
{
    "suggestions": [
        {
            "location": "password validation functions",
            "reasoning": "Login failures often stem from incorrect password validation logic",
            "confidence": 0.85
        },
        {
            "location": "session management code",
            "reasoning": "Authentication issues can be caused by improper session handling",
            "confidence": 0.75
        }
    ]
}
'''
        ])
    
    # Create ConceptMapper
    concept_mapper = ConceptMapper(config=config, llm_interface=llm_interface)
    
    # Example 1: Authentication Bug Scenario
    print("Example 1: Authentication Bug Analysis")
    print("-" * 40)
    
    issue_description = """
    Users are unable to log in to the system even with correct credentials. 
    The authenticate_user function seems to always return False, and we suspect 
    there's an issue with the password hashing comparison. This started after 
    we updated the security library.
    """
    
    extracted_concepts = [
        "authenticate_user", 
        "password", 
        "hashing", 
        "credentials", 
        "security"
    ]
    
    # Create context with sample code
    context = ContextWindow(
        target_code="""
def authenticate_user(username, password):
    '''Authenticate user with username and password.'''
    user = get_user_by_username(username)
    if not user:
        return False
    
    # Potential bug: using wrong hashing method
    hashed_password = hashlib.md5(password.encode()).hexdigest()
    return user.password_hash == hashed_password

def hash_password(password):
    '''Hash password using bcrypt for security.'''
    import bcrypt
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, stored_hash):
    '''Verify password against stored hash.'''
    import bcrypt
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash)

class UserManager:
    '''Manages user operations and authentication.'''
    
    def login_user(self, username, password):
        if authenticate_user(username, password):
            return create_session(username)
        return None
    
    def create_user(self, username, password, email):
        hashed_pw = hash_password(password)
        return User(username, hashed_pw, email)
""",
        related_functions=[
            "get_user_by_username(username)",
            "create_session(username)",
            "validate_credentials(username, password)"
        ],
        class_hierarchy={
            "UserManager": ["BaseManager"],
            "User": ["BaseModel"]
        },
        module_dependencies=[
            "hashlib",
            "bcrypt", 
            "database.models",
            "auth.session"
        ],
        domain_concepts=[
            "Authentication",
            "Security",
            "Hashing",
            "Session",
            "Credentials"
        ],
        dependency_context=DependencyContext(
            function_signatures={
                "authenticate_user": "def authenticate_user(username: str, password: str) -> bool",
                "hash_password": "def hash_password(password: str) -> bytes",
                "verify_password": "def verify_password(password: str, stored_hash: bytes) -> bool"
            },
            class_methods={
                "UserManager": ["login_user", "create_user"]
            },
            call_graph={
                "authenticate_user": ["get_user_by_username"],
                "login_user": ["authenticate_user", "create_session"],
                "create_user": ["hash_password"]
            }
        )
    )
    
    # Perform concept mapping
    result = await concept_mapper.map_concepts_to_code(
        issue_description, extracted_concepts, context
    )
    
    # Display results
    print(f"Overall Confidence: {result.overall_confidence:.2f}")
    print(f"Search Strategy: {result.search_strategy_used}")
    print()
    
    print("Primary Matches:")
    for i, match in enumerate(result.primary_matches[:5], 1):
        print(f"  {i}. {match.element_name} ({match.element_type})")
        print(f"     Match Type: {match.match_type}")
        print(f"     Confidence: {match.confidence:.2f}")
        print(f"     Location: {match.file_path}:{match.line_number}")
        print(f"     Evidence: {match.evidence[0] if match.evidence else 'N/A'}")
        print()
    
    print("Alternative Candidates:")
    for i, candidate in enumerate(result.alternative_candidates[:3], 1):
        print(f"  {i}. {candidate.location}")
        print(f"     Confidence: {candidate.confidence:.2f}")
        print(f"     Reasoning: {candidate.reasoning}")
        print()
    
    print("Reasoning Chain:")
    for i, step in enumerate(result.reasoning_chain.steps, 1):
        print(f"  Step {i}: {step.description}")
        print(f"          Confidence: {step.confidence:.2f}")
        print(f"          Input: {step.input_data}")
        print(f"          Output: {step.output_data}")
        print()
    
    print("Evidence Chain:")
    for i, evidence in enumerate(result.evidence_chain.evidence_items[:5], 1):
        source = result.evidence_chain.source_locations[i-1]
        weight = result.evidence_chain.confidence_weights[i-1]
        print(f"  {i}. {evidence}")
        print(f"     Source: {source} (weight: {weight:.2f})")
        print()
    
    # Example 2: Performance Issue Scenario
    print("\n" + "="*60)
    print("Example 2: Performance Issue Analysis")
    print("-" * 40)
    
    performance_issue = """
    The application is running very slowly when processing large datasets.
    Users report that the data_processor function takes several minutes to complete
    operations that should finish in seconds. We suspect there might be an 
    inefficient algorithm or unnecessary database queries.
    """
    
    performance_concepts = [
        "data_processor",
        "performance", 
        "algorithm",
        "database",
        "optimization"
    ]
    
    performance_context = ContextWindow(
        target_code="""
def data_processor(dataset):
    '''Process large dataset with transformations.'''
    results = []
    for item in dataset:
        # Potential performance issue: N+1 query problem
        related_data = fetch_related_data(item.id)
        processed_item = transform_data(item, related_data)
        results.append(processed_item)
    return results

def fetch_related_data(item_id):
    '''Fetch related data from database.'''
    # This could be causing performance issues
    return database.query(f"SELECT * FROM related WHERE item_id = {item_id}")

def transform_data(item, related_data):
    '''Transform data with complex calculations.'''
    # Potentially expensive operations
    result = complex_calculation(item.value)
    for related in related_data:
        result += expensive_operation(related.value)
    return result

def optimize_query(dataset_ids):
    '''Optimized version using batch queries.'''
    # Better approach: batch query
    return database.query(f"SELECT * FROM related WHERE item_id IN ({','.join(map(str, dataset_ids))})")
""",
        related_functions=[
            "complex_calculation(value)",
            "expensive_operation(value)",
            "batch_process_data(dataset)"
        ],
        module_dependencies=[
            "database",
            "numpy",
            "pandas"
        ],
        domain_concepts=[
            "Performance",
            "Optimization", 
            "Database",
            "Algorithm"
        ]
    )
    
    # Perform concept mapping for performance issue
    perf_result = await concept_mapper.map_concepts_to_code(
        performance_issue, performance_concepts, performance_context
    )
    
    print(f"Overall Confidence: {perf_result.overall_confidence:.2f}")
    print()
    
    print("Top Matches for Performance Issue:")
    for i, match in enumerate(perf_result.primary_matches[:3], 1):
        print(f"  {i}. {match.element_name} ({match.element_type})")
        print(f"     Confidence: {match.confidence:.2f}")
        print(f"     Evidence: {match.evidence[0] if match.evidence else 'N/A'}")
        print()
    
    print("Performance Optimization Suggestions:")
    for i, candidate in enumerate(perf_result.alternative_candidates[:2], 1):
        print(f"  {i}. {candidate.location}")
        print(f"     Reasoning: {candidate.reasoning}")
        print()


if __name__ == "__main__":
    asyncio.run(main())