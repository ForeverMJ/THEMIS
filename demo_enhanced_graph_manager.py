#!/usr/bin/env python3
"""
Enhanced GraphManager Demo Script

This script demonstrates the complete functionality of the Enhanced GraphManager,
including structural extraction, semantic injection, dependency tracing, and 
violation flagging.
"""

from src.enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
from src.enhanced_graph_manager.logger import set_log_level

def main():
    """Run the Enhanced GraphManager demonstration."""
    
    print("🚀 Enhanced GraphManager Demo")
    print("=" * 50)
    
    # Set log level for demo
    set_log_level("WARNING")  # Reduce log noise for demo
    
    # Sample Python code to analyze
    sample_code = """
class UserService:
    '''Service for managing user accounts.'''
    
    def __init__(self):
        self.users = {}
        self.active_sessions = []
    
    def create_user(self, username: str, email: str, password: str) -> bool:
        '''Create a new user account.'''
        # Missing validation!
        if username in self.users:
            return False
        
        self.users[username] = {
            'email': email,
            'password': password,
            'created_at': self.get_timestamp()
        }
        return True
    
    def authenticate(self, username: str, password: str) -> bool:
        '''Authenticate user credentials.'''
        user = self.users.get(username)
        if user and user['password'] == password:
            self.active_sessions.append(username)
            return True
        return False
    
    def get_user(self, username: str) -> dict:
        '''Get user information.'''
        return self.users.get(username, {})
    
    def get_timestamp(self) -> str:
        '''Get current timestamp.'''
        import datetime
        return datetime.datetime.now().isoformat()
    
    # Missing: update_user, delete_user methods

def validate_email(email: str) -> bool:
    '''Validate email format.'''
    return '@' in email and '.' in email

def hash_password(password: str) -> str:
    '''Hash password for security.'''
    # Simplified hashing
    return f"hashed_{password}"
"""
    
    # Requirements text
    requirements_text = """
    The system must validate all user input before processing.
    Users should be able to create accounts with unique usernames.
    The application needs to authenticate users securely.
    User passwords must be properly hashed and stored.
    The system should support full CRUD operations for users.
    Email addresses must be validated before account creation.
    The application must handle authentication failures gracefully.
    """
    
    # Create Enhanced GraphManager instance
    manager = EnhancedGraphManager()
    
    print("\n📊 Running Complete Analysis Workflow...")
    
    # Run complete workflow
    results = manager.analyze_complete_workflow(sample_code, requirements_text)
    
    if results['success']:
        print(f"✅ Analysis completed in {results['execution_time']:.3f} seconds")
        
        # Display graph statistics
        stats = results['graph_statistics']
        print(f"\n📈 Graph Statistics:")
        print(f"   • Total nodes: {stats['total_nodes']}")
        print(f"   • Total edges: {stats['total_edges']}")
        print(f"   • Node types: {dict(stats['node_types'])}")
        
        # Display dependency analysis
        deps = results['dependency_analysis']
        print(f"\n🔗 Dependency Analysis:")
        print(f"   • Nodes with dependencies: {deps['nodes_with_dependencies']}/{deps['total_nodes']}")
        print(f"   • Dependency ratio: {deps['dependency_ratio']:.2%}")
        
        if deps['most_dependent_nodes']:
            top_node = deps['most_dependent_nodes'][0]
            print(f"   • Most dependent node: {top_node['node']} ({top_node['dependency_count']} dependencies)")
        
        # Display violation report
        violations = results['violation_report']
        print(f"\n⚠️  Violation Report:")
        print(f"   • Total violations: {violations['total_violations']}")
        print(f"   • Satisfies requirements: {violations['total_satisfies']}")
        print(f"   • Unknown status: {violations['total_unknown']}")
        
        # Show top violations
        if violations['prioritized_violations']:
            print(f"\n🔍 Top Violations:")
            for i, violation in enumerate(violations['prioritized_violations'][:5], 1):
                print(f"   {i}. {violation['requirement_id']} → {violation['code_node']}")
                print(f"      Status: {violation['status']}")
                print(f"      Reason: {violation['reason']}")
                print(f"      Severity: {violation['severity']}, Confidence: {violation['confidence']:.2f}")
                print()
        
        # Performance metrics
        metrics = results['performance_metrics']
        print(f"⏱️  Performance Metrics:")
        for operation, time_taken in metrics.items():
            print(f"   • {operation}: {time_taken:.3f}s")
        
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
    
    # Demonstrate serialization
    print(f"\n💾 Testing Serialization...")
    serialized = manager.serialize_graph()
    print(f"   • Serialized {len(serialized['nodes'])} nodes and {len(serialized['edges'])} edges")
    
    # Health check
    print(f"\n🏥 System Health Check:")
    health = manager.health_check()
    print(f"   • Status: {health['status']}")
    print(f"   • Graph size: {health['graph_nodes']} nodes, {health['graph_edges']} edges")
    
    if 'warnings' in health:
        print(f"   • Warnings: {health['warnings']}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"\nThe Enhanced GraphManager provides:")
    print(f"   🔍 Structural code analysis")
    print(f"   📝 Semantic requirement injection")
    print(f"   🔗 Dependency relationship tracing")
    print(f"   ⚠️  Automated violation detection")
    print(f"   📊 Comprehensive reporting")
    print(f"   💾 Graph serialization/persistence")
    print(f"   🏥 Health monitoring")

if __name__ == "__main__":
    main()