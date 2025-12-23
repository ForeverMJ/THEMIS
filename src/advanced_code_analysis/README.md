# Advanced Code Analysis and Semantic Understanding

This module provides LLM-driven intelligent semantic understanding to enhance the existing Enhanced GraphManager system. It focuses on solving core problems identified in system analysis: inability to understand complex technical requirements, inability to extract key information from problem descriptions, and inability to accurately map technical concepts to code components.

## Architecture Overview

The system is built around four core innovations:

1. **Intelligent Problem Classification** - Classifies bug types and selects optimal analysis strategies
2. **Context Enhancement** - Provides rich code context and domain knowledge to LLMs
3. **Pattern Learning** - Accumulates bug pattern knowledge from successful cases
4. **Multi-Round Reasoning** - Improves analysis accuracy through multi-step verification

## Core Components

### Data Models (`models.py`)
- `BugType`: Classified bug types with confidence and characteristics
- `AnalysisStrategy`: Analysis strategies for different bug types
- `ContextWindow`: Code context information for LLM analysis
- `ReasoningChain`: Multi-step reasoning process with evidence
- `AnalysisResult`: Complete analysis results with confidence scores
- `VerificationResult`: Verification and conflict detection results
- `BugPattern`: Learned patterns from successful bug fixes

### Configuration (`config.py`)
- `LLMConfig`: LLM provider and API configuration
- `AnalysisConfig`: Analysis behavior and thresholds
- `StorageConfig`: Data storage and caching settings
- `AdvancedAnalysisConfig`: Main configuration class

### LLM Interface (`llm_interface.py`)
- `LLMInterface`: Unified interface for different LLM providers
- `OpenAIProvider`: OpenAI API integration
- `AnthropicProvider`: Anthropic API integration
- `MockProvider`: Testing and development provider

## Usage

```python
from advanced_code_analysis import AdvancedAnalysisConfig, LLMInterface
from advanced_code_analysis.models import BugType, AnalysisResult

# Initialize configuration
config = AdvancedAnalysisConfig()

# Create LLM interface
llm = LLMInterface(config.llm)

# Test connection
if await llm.test_connection():
    print("LLM connection successful")
```

## Configuration

The system can be configured through:

1. **Environment Variables**: API keys and basic settings
2. **Configuration Files**: JSON or YAML configuration files
3. **Programmatic Configuration**: Direct instantiation of config classes

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Configuration File Example

```json
{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "max_completion_tokens": 4096,
    "temperature": 0.1
  },
  "analysis": {
    "confidence_threshold": 0.7,
    "max_reasoning_rounds": 5,
    "enable_pattern_learning": true
  }
}
```

## Integration with Enhanced GraphManager

This system is designed to enhance, not replace, the existing Enhanced GraphManager. It provides:

- Semantic understanding of problem descriptions
- Intelligent context collection and enhancement
- Multi-round reasoning and verification
- Pattern-based learning and improvement

The integration maintains backward compatibility while adding advanced LLM-driven capabilities.

## Development Status

This is the initial project structure and core interfaces. The following components will be implemented in subsequent tasks:

- [ ] Bug Classification Engine
- [ ] Semantic Information Extraction Engine  
- [ ] Context Enhancement Engine
- [ ] Multi-Level Concept Mapping Engine
- [ ] Predefined Pattern Matching Engine
- [ ] Multi-Round Reasoning Engine
- [ ] Enhanced AST Analysis Engine
- [ ] Conflict Detection and Handling Engine
- [ ] Result Ranking and Output Engine
- [ ] System Integration and Testing