"""
Example usage of the Context Enhancement Engine.

This script demonstrates how to use the ContextEnhancer class to collect
rich code context, optimize context windows, and extract domain knowledge.
"""

import os
import tempfile
from pathlib import Path

from .context_enhancer import ContextEnhancer
from .config import AdvancedAnalysisConfig


def create_sample_project():
    """Create a sample project for demonstration."""
    temp_dir = tempfile.mkdtemp()
    project_root = os.path.join(temp_dir, 'sample_web_app')
    os.makedirs(project_root, exist_ok=True)
    
    # Create main application file
    main_code = '''
"""Main web application using Flask."""

from flask import Flask, request, jsonify
from typing import Dict, List, Optional
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

class UserManager:
    """Manages user operations."""
    
    def __init__(self):
        self.users: Dict[str, Dict] = {}
    
    def create_user(self, user_data: Dict) -> Optional[str]:
        """Create a new user."""
        if self.validate_user_data(user_data):
            user_id = f"user_{len(self.users) + 1}"
            self.users[user_id] = user_data
            logger.info(f"Created user {user_id}")
            return user_id
        return None
    
    def validate_user_data(self, data: Dict) -> bool:
        """Validate user data."""
        required_fields = ['name', 'email']
        return all(field in data for field in required_fields)
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        return self.users.get(user_id)

user_manager = UserManager()

@app.route('/users', methods=['POST'])
def create_user():
    """API endpoint to create a user."""
    try:
        user_data = request.get_json()
        user_id = user_manager.create_user(user_data)
        
        if user_id:
            return jsonify({'user_id': user_id, 'status': 'created'}), 201
        else:
            return jsonify({'error': 'Invalid user data'}), 400
    
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id: str):
    """API endpoint to get a user."""
    user = user_manager.get_user(user_id)
    
    if user:
        return jsonify(user), 200
    else:
        return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
'''
    
    with open(os.path.join(project_root, 'app.py'), 'w') as f:
        f.write(main_code)
    
    # Create utilities module
    utils_code = '''
"""Utility functions for the web application."""

import re
from typing import Dict, Any
import hashlib

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize user input data."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            sanitized[key] = re.sub(r'[<>"\']', '', value)
        else:
            sanitized[key] = value
    return sanitized

class DatabaseConnection:
    """Mock database connection for demonstration."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to the database."""
        # Mock connection logic
        self.connected = True
        return True
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a database query."""
        if not self.connected:
            raise ConnectionError("Not connected to database")
        
        # Mock query execution
        return [{'id': 1, 'result': 'mock_data'}]
    
    def close(self):
        """Close database connection."""
        self.connected = False
'''
    
    with open(os.path.join(project_root, 'utils.py'), 'w') as f:
        f.write(utils_code)
    
    # Create test file
    test_code = '''
"""Tests for the web application."""

import unittest
from unittest.mock import patch, MagicMock
import json

from app import app, user_manager
from utils import validate_email, hash_password

class TestUserManager(unittest.TestCase):
    """Test cases for UserManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        user_manager.users.clear()
    
    def test_create_user_valid_data(self):
        """Test creating user with valid data."""
        user_data = {'name': 'John Doe', 'email': 'john@example.com'}
        user_id = user_manager.create_user(user_data)
        
        self.assertIsNotNone(user_id)
        self.assertIn(user_id, user_manager.users)
    
    def test_create_user_invalid_data(self):
        """Test creating user with invalid data."""
        user_data = {'name': 'John Doe'}  # Missing email
        user_id = user_manager.create_user(user_data)
        
        self.assertIsNone(user_id)

class TestAPI(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()
        user_manager.users.clear()
    
    def test_create_user_endpoint(self):
        """Test user creation endpoint."""
        user_data = {'name': 'Jane Doe', 'email': 'jane@example.com'}
        
        response = self.client.post('/users', 
                                  data=json.dumps(user_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertIn('user_id', data)

if __name__ == '__main__':
    unittest.main()
'''
    
    with open(os.path.join(project_root, 'test_app.py'), 'w') as f:
        f.write(test_code)
    
    # Create requirements file
    requirements = '''
flask==2.3.3
pytest==7.4.2
requests==2.31.0
'''
    
    with open(os.path.join(project_root, 'requirements.txt'), 'w') as f:
        f.write(requirements)
    
    # Create README
    readme = '''
# Sample Web Application

This is a sample Flask web application for demonstrating the Context Enhancement Engine.

## Features

- User management API
- Input validation and sanitization
- Database connection utilities
- Comprehensive test suite

## API Endpoints

- POST /users - Create a new user
- GET /users/<user_id> - Get user by ID

## Running the Application

```bash
pip install -r requirements.txt
python app.py
```

## Running Tests

```bash
python -m pytest test_app.py
```
'''
    
    with open(os.path.join(project_root, 'README.md'), 'w') as f:
        f.write(readme)
    
    return project_root


def demonstrate_context_enhancement():
    """Demonstrate the Context Enhancement Engine capabilities."""
    print("🚀 Context Enhancement Engine Demo")
    print("=" * 50)
    
    # Create sample project
    print("\n1. Creating sample web application project...")
    project_root = create_sample_project()
    print(f"   Project created at: {project_root}")
    
    # Initialize ContextEnhancer
    config = AdvancedAnalysisConfig()
    enhancer = ContextEnhancer(config)
    
    # Analyze project structure
    print("\n2. Analyzing project structure...")
    project_structure = enhancer.analyze_project_structure(project_root)
    
    print(f"   📁 Python files: {len(project_structure.python_files)}")
    for file in project_structure.python_files:
        print(f"      - {file}")
    
    print(f"   🧪 Test files: {len(project_structure.test_files)}")
    for file in project_structure.test_files:
        print(f"      - {file}")
    
    print(f"   ⚙️  Config files: {len(project_structure.config_files)}")
    for file in project_structure.config_files:
        print(f"      - {file}")
    
    # Collect code context
    print("\n3. Collecting code context...")
    target_files = [os.path.join(project_root, f) for f in project_structure.python_files]
    context = enhancer.collect_code_context(target_files)
    
    print(f"   📊 Token count: {context.token_count}")
    print(f"   🔧 Related functions: {len(context.related_functions)}")
    print(f"   🏗️  Class hierarchy: {len(context.class_hierarchy)} classes")
    print(f"   📦 Module dependencies: {len(context.module_dependencies)}")
    print(f"   💡 Domain concepts: {len(context.domain_concepts)}")
    
    # Show some examples
    if context.related_functions:
        print(f"\n   Example functions:")
        for func in context.related_functions[:3]:
            print(f"      - {func}")
    
    if context.class_hierarchy:
        print(f"\n   Example classes:")
        for class_name, bases in list(context.class_hierarchy.items())[:3]:
            print(f"      - {class_name} (bases: {bases})")
    
    if context.module_dependencies:
        print(f"\n   Example dependencies:")
        for dep in context.module_dependencies[:5]:
            print(f"      - {dep}")
    
    # Build dependency context
    print("\n4. Building dependency context...")
    dependency_context = enhancer.build_dependency_context(target_files)
    
    print(f"   🔗 Function signatures: {len(dependency_context.function_signatures)}")
    print(f"   🏛️  Class methods: {len(dependency_context.class_methods)}")
    print(f"   📥 Import statements: {len(dependency_context.import_statements)}")
    print(f"   📞 Call graph entries: {len(dependency_context.call_graph)}")
    
    # Show some examples
    if dependency_context.function_signatures:
        print(f"\n   Example function signatures:")
        for name, sig in list(dependency_context.function_signatures.items())[:3]:
            print(f"      - {sig}")
    
    # Extract domain knowledge
    print("\n5. Extracting domain knowledge...")
    domain_knowledge = enhancer.extract_domain_knowledge('web')
    
    print(f"   📚 Domain: {domain_knowledge.domain_name}")
    print(f"   📖 Terminology: {len(domain_knowledge.terminology)} terms")
    print(f"   🎯 Common patterns: {len(domain_knowledge.common_patterns)}")
    print(f"   ✅ Best practices: {len(domain_knowledge.best_practices)}")
    print(f"   ❌ Anti-patterns: {len(domain_knowledge.anti_patterns)}")
    
    # Show some examples
    if domain_knowledge.terminology:
        print(f"\n   Example terminology:")
        for term, definition in list(domain_knowledge.terminology.items())[:3]:
            print(f"      - {term}: {definition}")
    
    if domain_knowledge.common_patterns:
        print(f"\n   Example patterns:")
        for pattern in domain_knowledge.common_patterns[:3]:
            print(f"      - {pattern}")
    
    # Demonstrate context optimization
    print("\n6. Demonstrating context optimization...")
    print(f"   Original context: {context.token_count} tokens")
    
    # Optimize for different token limits
    for limit in [2000, 1000, 500]:
        optimized = enhancer.optimize_context_window(context, limit)
        print(f"   Optimized for {limit} tokens: {optimized.token_count} tokens")
        print(f"      - Functions: {len(optimized.related_functions)}")
        print(f"      - Dependencies: {len(optimized.module_dependencies)}")
        print(f"      - Concepts: {len(optimized.domain_concepts)}")
    
    # Get contextual snippets
    print("\n7. Getting contextual code snippets...")
    snippets = enhancer.get_contextual_code_snippets('UserManager', 'definition')
    
    if snippets:
        snippet = snippets[0]
        print(f"   Found definition for 'UserManager':")
        print(f"      - File: {snippet.file_path}")
        print(f"      - Line: {snippet.line_number}")
        print(f"      - Type: {snippet.element_type}")
        if snippet.docstring:
            print(f"      - Docstring: {snippet.docstring}")
    
    print("\n✨ Context Enhancement Demo Complete!")
    print(f"📁 Sample project available at: {project_root}")
    
    # Cleanup note
    print(f"\n💡 Note: The sample project is in a temporary directory.")
    print(f"   You can explore it manually or it will be cleaned up automatically.")
    
    return project_root


if __name__ == "__main__":
    demonstrate_context_enhancement()