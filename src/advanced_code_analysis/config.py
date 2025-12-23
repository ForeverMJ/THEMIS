"""
Configuration module for the Advanced Code Analysis system.

This module provides configuration classes for LLM integration,
analysis parameters, and system settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os


@dataclass
class LLMConfig:
    """Configuration for LLM API integration."""
    provider: str = "openai"  # openai, anthropic, local, etc.
    model_name: str = "gpt-4o-mini"  # 稳定的默认模型
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    # 最大トークン数（completion 上限）
    max_completion_tokens: int = 4096
    # 互換性: 旧設定キーで上書きしたい場合に使用
    legacy_max_completion_tokens: Optional[int] = None
    temperature: float = 0.1
    timeout: int = 100
    max_retries: int = 4
    retry_delay: float = 1.0
    
    @property
    def max_tokens(self) -> int:
        """Alias for max_completion_tokens for backward compatibility."""
        return self.max_completion_tokens
    
    def __post_init__(self):
        """Load configuration from environment if not provided."""
        # 从环境变量读取模型配置（支持快速切换）
        env_model = os.getenv("LLM_MODEL")
        env_provider = os.getenv("LLM_PROVIDER")
        
        # 如果环境变量存在，总是使用环境变量的值（优先级更高）
        if env_model:
            self.model_name = env_model
        
        if env_provider:
            self.provider = env_provider
        
        # Load API key from environment if not provided
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # 互換性: 古い設定で max_completion_tokens（legacy）が渡された場合に上書き
        if self.legacy_max_completion_tokens is not None:
            self.max_completion_tokens = self.legacy_max_completion_tokens


@dataclass
class AnalysisConfig:
    """Configuration for analysis behavior."""
    max_context_tokens: int = 8000
    confidence_threshold: float = 0.7
    max_reasoning_rounds: int = 5
    enable_pattern_learning: bool = True
    enable_conflict_detection: bool = True
    enable_multi_round_reasoning: bool = True
    
    # Bug classification settings
    classification_confidence_threshold: float = 0.6
    max_classification_attempts: int = 3
    
    # Context enhancement settings
    max_related_functions: int = 10
    max_dependency_depth: int = 3
    include_test_files: bool = False
    
    # Pattern learning settings
    pattern_similarity_threshold: float = 0.8
    max_stored_patterns: int = 1000
    pattern_cleanup_interval: int = 100


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    patterns_db_path: str = "data/bug_patterns.json"
    feedback_db_path: str = "data/classification_feedback.json"
    cache_dir: str = "cache/advanced_analysis"
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        """Ensure storage directories exist."""
        for path_str in [self.patterns_db_path, self.feedback_db_path]:
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/advanced_analysis.log"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Ensure log directory exists."""
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class IntegrationConfig:
    """Configuration for Enhanced GraphManager integration."""
    
    # Integration mode settings
    enable_graph_context_enhancement: bool = True
    enable_semantic_requirement_mapping: bool = True
    enable_dependency_aware_analysis: bool = True
    enable_violation_guided_analysis: bool = True
    
    # Analysis strategy selection
    use_graph_for_bug_classification: bool = True
    use_graph_for_concept_mapping: bool = True
    use_graph_for_pattern_matching: bool = True
    
    # Performance settings
    max_graph_nodes_for_analysis: int = 1000
    max_dependency_depth_for_context: int = 5
    enable_parallel_graph_analysis: bool = False
    
    # Fallback settings
    fallback_to_basic_analysis: bool = True
    fallback_confidence_threshold: float = 0.3
    
    # Output integration
    merge_analysis_results: bool = True
    preserve_graph_structure: bool = True
    add_analysis_nodes_to_graph: bool = False


@dataclass
class AdvancedAnalysisConfig:
    """Main configuration class for the Advanced Code Analysis system."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # System integration settings
    integrate_with_enhanced_graph_manager: bool = True
    fallback_to_basic_analysis: bool = True
    
    # Performance settings
    parallel_analysis: bool = False
    max_concurrent_requests: int = 3
    
    # Debug settings
    debug_mode: bool = False
    save_intermediate_results: bool = False
    verbose_logging: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdvancedAnalysisConfig':
        """Create configuration from dictionary."""
        llm_config = LLMConfig(**config_dict.get('llm', {}))
        analysis_config = AnalysisConfig(**config_dict.get('analysis', {}))
        storage_config = StorageConfig(**config_dict.get('storage', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        integration_config = IntegrationConfig(**config_dict.get('integration', {}))
        
        # Extract top-level settings
        system_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['llm', 'analysis', 'storage', 'logging', 'integration']}
        
        return cls(
            llm=llm_config,
            analysis=analysis_config,
            storage=storage_config,
            logging=logging_config,
            integration=integration_config,
            **system_settings
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AdvancedAnalysisConfig':
        """Load configuration from JSON or YAML file."""
        import json
        from pathlib import Path
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'llm': {
                'provider': self.llm.provider,
                'model_name': self.llm.model_name,
                'max_completion_tokens': self.llm.max_completion_tokens,
                'temperature': self.llm.temperature,
                'timeout': self.llm.timeout,
                'max_retries': self.llm.max_retries,
                'retry_delay': self.llm.retry_delay,
            },
            'analysis': {
                'max_context_tokens': self.analysis.max_context_tokens,
                'confidence_threshold': self.analysis.confidence_threshold,
                'max_reasoning_rounds': self.analysis.max_reasoning_rounds,
                'enable_pattern_learning': self.analysis.enable_pattern_learning,
                'enable_conflict_detection': self.analysis.enable_conflict_detection,
                'enable_multi_round_reasoning': self.analysis.enable_multi_round_reasoning,
                'classification_confidence_threshold': self.analysis.classification_confidence_threshold,
                'max_classification_attempts': self.analysis.max_classification_attempts,
                'max_related_functions': self.analysis.max_related_functions,
                'max_dependency_depth': self.analysis.max_dependency_depth,
                'include_test_files': self.analysis.include_test_files,
                'pattern_similarity_threshold': self.analysis.pattern_similarity_threshold,
                'max_stored_patterns': self.analysis.max_stored_patterns,
                'pattern_cleanup_interval': self.analysis.pattern_cleanup_interval,
            },
            'storage': {
                'patterns_db_path': self.storage.patterns_db_path,
                'feedback_db_path': self.storage.feedback_db_path,
                'cache_dir': self.storage.cache_dir,
                'enable_caching': self.storage.enable_caching,
                'cache_ttl_hours': self.storage.cache_ttl_hours,
            },
            'logging': {
                'log_level': self.logging.log_level,
                'log_file': self.logging.log_file,
                'enable_console_logging': self.logging.enable_console_logging,
                'enable_file_logging': self.logging.enable_file_logging,
                'log_format': self.logging.log_format,
            },
            'integration': {
                'enable_graph_context_enhancement': self.integration.enable_graph_context_enhancement,
                'enable_semantic_requirement_mapping': self.integration.enable_semantic_requirement_mapping,
                'enable_dependency_aware_analysis': self.integration.enable_dependency_aware_analysis,
                'enable_violation_guided_analysis': self.integration.enable_violation_guided_analysis,
                'use_graph_for_bug_classification': self.integration.use_graph_for_bug_classification,
                'use_graph_for_concept_mapping': self.integration.use_graph_for_concept_mapping,
                'use_graph_for_pattern_matching': self.integration.use_graph_for_pattern_matching,
                'max_graph_nodes_for_analysis': self.integration.max_graph_nodes_for_analysis,
                'max_dependency_depth_for_context': self.integration.max_dependency_depth_for_context,
                'enable_parallel_graph_analysis': self.integration.enable_parallel_graph_analysis,
                'fallback_to_basic_analysis': self.integration.fallback_to_basic_analysis,
                'fallback_confidence_threshold': self.integration.fallback_confidence_threshold,
                'merge_analysis_results': self.integration.merge_analysis_results,
                'preserve_graph_structure': self.integration.preserve_graph_structure,
                'add_analysis_nodes_to_graph': self.integration.add_analysis_nodes_to_graph,
            },
            'integrate_with_enhanced_graph_manager': self.integrate_with_enhanced_graph_manager,
            'fallback_to_basic_analysis': self.fallback_to_basic_analysis,
            'parallel_analysis': self.parallel_analysis,
            'max_concurrent_requests': self.max_concurrent_requests,
            'debug_mode': self.debug_mode,
            'save_intermediate_results': self.save_intermediate_results,
            'verbose_logging': self.verbose_logging,
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate LLM configuration
        if not self.llm.api_key and self.llm.provider in ['openai', 'anthropic']:
            issues.append(f"API key required for {self.llm.provider} provider")
        
        if self.llm.max_completion_tokens <= 0:
            issues.append("max_completion_tokens must be positive")
        
        if not 0.0 <= self.llm.temperature <= 2.0:
            issues.append("temperature must be between 0.0 and 2.0")
        
        # Validate analysis configuration
        if not 0.0 <= self.analysis.confidence_threshold <= 1.0:
            issues.append("confidence_threshold must be between 0.0 and 1.0")
        
        if self.analysis.max_reasoning_rounds <= 0:
            issues.append("max_reasoning_rounds must be positive")
        
        if self.analysis.max_context_tokens <= 0:
            issues.append("max_context_tokens must be positive")
        
        # Validate that context tokens don't exceed LLM max tokens
        if self.analysis.max_context_tokens > self.llm.max_completion_tokens:
            issues.append("max_context_tokens cannot exceed LLM max_completion_tokens")
        
        return issues
