"""
Basic integration tests for the Advanced Code Analysis and Enhanced GraphManager integration.

This module provides simple tests to verify that the integration components
work correctly and can handle various scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from pathlib import Path

from .config import AdvancedAnalysisConfig, IntegrationConfig
from .graph_manager_integration import GraphManagerIntegration, IntegratedAnalysisResult
from .advanced_code_analyzer import AdvancedCodeAnalyzer

# Test if Enhanced GraphManager is available
try:
    from ..enhanced_graph_manager.enhanced_graph_manager import EnhancedGraphManager
    from ..enhanced_graph_manager.config import EnhancedGraphManagerConfig
    ENHANCED_GRAPH_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_GRAPH_MANAGER_AVAILABLE = False


class TestIntegrationConfig:
    """Test integration configuration."""
    
    def test_integration_config_creation(self):
        """Test that integration config can be created with defaults."""
        config = IntegrationConfig()
        
        assert config.enable_graph_context_enhancement is True
        assert config.enable_semantic_requirement_mapping is True
        assert config.enable_dependency_aware_analysis is True
        assert config.fallback_to_basic_analysis is True
        assert config.max_graph_nodes_for_analysis == 1000
    
    def test_integration_config_customization(self):
        """Test that integration config can be customized."""
        config = IntegrationConfig(
            enable_graph_context_enhancement=False,
            max_graph_nodes_for_analysis=500,
            fallback_confidence_threshold=0.5
        )
        
        assert config.enable_graph_context_enhancement is False
        assert config.max_graph_nodes_for_analysis == 500
        assert config.fallback_confidence_threshold == 0.5


class TestAdvancedAnalysisConfig:
    """Test advanced analysis configuration with integration."""
    
    def test_config_with_integration(self):
        """Test that advanced config includes integration settings."""
        config = AdvancedAnalysisConfig()
        
        assert hasattr(config, 'integration')
        assert isinstance(config.integration, IntegrationConfig)
        assert config.integrate_with_enhanced_graph_manager is True
    
    def test_config_serialization_with_integration(self):
        """Test that config can be serialized and deserialized with integration settings."""
        config = AdvancedAnalysisConfig()
        config.integration.enable_graph_context_enhancement = False
        config.integration.max_graph_nodes_for_analysis = 2000
        
        config_dict = config.to_dict()
        
        assert 'integration' in config_dict
        assert config_dict['integration']['enable_graph_context_enhancement'] is False
        assert config_dict['integration']['max_graph_nodes_for_analysis'] == 2000
        
        # Test deserialization
        restored_config = AdvancedAnalysisConfig.from_dict(config_dict)
        assert restored_config.integration.enable_graph_context_enhancement is False
        assert restored_config.integration.max_graph_nodes_for_analysis == 2000


@pytest.mark.skipif(not ENHANCED_GRAPH_MANAGER_AVAILABLE, 
                   reason="Enhanced GraphManager not available")
class TestGraphManagerIntegration:
    """Test GraphManager integration functionality."""
    
    @pytest.fixture
    def mock_advanced_analyzer(self):
        """Create a mock advanced analyzer."""
        analyzer = Mock(spec=AdvancedCodeAnalyzer)
        analyzer.analyze = Mock()
        analyzer.validate_configuration = Mock(return_value=[])
        analyzer.test_connection = Mock(return_value=True)
        return analyzer
    
    @pytest.fixture
    def mock_graph_manager(self):
        """Create a mock graph manager."""
        manager = Mock(spec=EnhancedGraphManager)
        manager.extract_structure = Mock()
        manager.inject_semantics = Mock()
        manager.trace_dependencies = Mock(return_value={})
        manager.flag_violations = Mock(return_value=[])
        manager.get_graph_statistics = Mock(return_value={'total_nodes': 10, 'total_edges': 5})
        manager.get_dependency_analysis = Mock(return_value={'total_nodes': 10})
        manager.get_violation_report = Mock(return_value={'total_violations': 0})
        manager.health_check = Mock(return_value={'status': 'healthy'})
        return manager
    
    def test_integration_initialization(self, mock_advanced_analyzer, mock_graph_manager):
        """Test that integration layer can be initialized."""
        integration = GraphManagerIntegration(
            mock_advanced_analyzer,
            mock_graph_manager
        )
        
        assert integration.advanced_analyzer == mock_advanced_analyzer
        assert integration.graph_manager == mock_graph_manager
        assert isinstance(integration.integration_config, IntegrationConfig)
    
    def test_integration_status(self, mock_advanced_analyzer, mock_graph_manager):
        """Test that integration status can be retrieved."""
        integration = GraphManagerIntegration(
            mock_advanced_analyzer,
            mock_graph_manager
        )
        
        status = integration.get_integration_status()
        
        assert 'enhanced_graph_manager_available' in status
        assert 'integration_config' in status
        assert 'cache_status' in status
        assert status['enhanced_graph_manager_available'] is True
    
    @pytest.mark.asyncio
    async def test_integration_validation(self, mock_advanced_analyzer, mock_graph_manager):
        """Test that integration validation works."""
        integration = GraphManagerIntegration(
            mock_advanced_analyzer,
            mock_graph_manager
        )
        
        # Mock the validation methods
        mock_advanced_analyzer.validate_configuration.return_value = []
        mock_advanced_analyzer.test_connection.return_value = True
        
        issues = await integration.validate_integration()
        
        assert isinstance(issues, list)
        # Should have no issues with healthy mocks
        assert len(issues) == 0
    
    def test_cache_management(self, mock_advanced_analyzer, mock_graph_manager):
        """Test cache management functionality."""
        integration = GraphManagerIntegration(
            mock_advanced_analyzer,
            mock_graph_manager
        )
        
        # Initially cache should be empty
        status = integration.get_integration_status()
        assert status['cache_status']['cached_analyses'] == 0
        
        # Clear cache should work without error
        integration.clear_cache()
        
        # Status should still show empty cache
        status = integration.get_integration_status()
        assert status['cache_status']['cached_analyses'] == 0


class TestIntegrationFallback:
    """Test integration fallback mechanisms when Enhanced GraphManager is not available."""
    
    def test_integration_without_graph_manager(self):
        """Test that integration gracefully handles missing Enhanced GraphManager."""
        # This test should work regardless of whether Enhanced GraphManager is available
        config = IntegrationConfig(fallback_to_basic_analysis=True)
        
        assert config.fallback_to_basic_analysis is True
        assert config.fallback_confidence_threshold == 0.3
    
    def test_config_validation_without_graph_manager(self):
        """Test that configuration validation works without Enhanced GraphManager."""
        config = AdvancedAnalysisConfig()
        
        # Should not raise an error even if Enhanced GraphManager is not available
        issues = config.validate()
        
        # Issues might exist (like missing API key), but should not crash
        assert isinstance(issues, list)


class TestUnifiedAdapter:
    """Test unified adapter functionality."""
    
    def test_unified_adapter_import(self):
        """Test that unified adapter can be imported."""
        try:
            from ..enhanced_graph_adapter import EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
            
            # Test enum values
            assert AnalysisStrategy.ADVANCED_ONLY.value == "advanced_only"
            assert AnalysisStrategy.GRAPH_ONLY.value == "graph_only"
            assert AnalysisStrategy.INTEGRATED.value == "integrated"
            assert AnalysisStrategy.AUTO_SELECT.value == "auto_select"
            
            # Test options creation
            options = AnalysisOptions()
            assert options.strategy == AnalysisStrategy.AUTO_SELECT
            assert options.confidence_threshold == 0.7
            
        except ImportError:
            # If import fails, that's expected when Enhanced GraphManager is not available
            pytest.skip("Unified adapter not available")


# Integration test that can run without Enhanced GraphManager
def test_basic_integration_components():
    """Test that basic integration components can be created."""
    
    # Test configuration
    config = AdvancedAnalysisConfig()
    assert config.integrate_with_enhanced_graph_manager is True
    
    # Test integration config
    integration_config = IntegrationConfig()
    assert integration_config.fallback_to_basic_analysis is True
    
    # Test that we can create an advanced analyzer
    analyzer = AdvancedCodeAnalyzer(config)
    assert analyzer is not None
    
    # Test configuration validation
    issues = analyzer.validate_configuration()
    assert isinstance(issues, list)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])