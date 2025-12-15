"""
LLM interface module for the Advanced Code Analysis system.

This module provides a unified interface for interacting with different
LLM providers (OpenAI, Anthropic, local models, etc.) with retry logic,
error handling, and response parsing.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

from .config import LLMConfig
from .models import PromptTemplate


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    response_time: float
    raw_response: Optional[Dict[str, Any]] = None


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out."""
    pass


class LLMRateLimitError(LLMError):
    """Exception raised when rate limit is exceeded."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate provider-specific configuration."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise LLMError("OpenAI library not installed. Run: pip install openai")
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        client = self._get_client()
        
        # Prepare request parameters
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(**params)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time,
                raw_response=response.model_dump()
            )
        
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Request timed out: {e}")
            elif "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise LLMError(f"OpenAI API error: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate OpenAI-specific configuration."""
        issues = []
        
        if not self.config.api_key:
            issues.append("OpenAI API key is required")
        
        if self.config.model_name not in [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"
        ]:
            issues.append(f"Unknown OpenAI model: {self.config.model_name}")
        
        return issues


class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise LLMError("Anthropic library not installed. Run: pip install anthropic")
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        client = self._get_client()
        
        # Prepare request parameters
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        start_time = time.time()
        
        try:
            response = await client.messages.create(**params)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                model=response.model,
                finish_reason=response.stop_reason,
                response_time=response_time,
                raw_response=response.model_dump()
            )
        
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Request timed out: {e}")
            elif "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise LLMError(f"Anthropic API error: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate Anthropic-specific configuration."""
        issues = []
        
        if not self.config.api_key:
            issues.append("Anthropic API key is required")
        
        if self.config.model_name not in [
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"
        ]:
            issues.append(f"Unknown Anthropic model: {self.config.model_name}")
        
        return issues


class MockProvider(LLMProvider):
    """Mock provider for testing purposes."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.responses = []
        self.call_count = 0
    
    def set_responses(self, responses: List[str]):
        """Set predefined responses for testing."""
        self.responses = responses
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response."""
        if self.call_count < len(self.responses):
            content = self.responses[self.call_count]
        else:
            content = f"Mock response {self.call_count + 1}"
        
        self.call_count += 1
        
        return LLMResponse(
            content=content,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock-model",
            finish_reason="stop",
            response_time=0.1
        )
    
    def validate_config(self) -> List[str]:
        """Mock provider has no validation requirements."""
        return []


class LLMInterface:
    """Main interface for LLM interactions with retry logic and error handling."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = self._create_provider()
        self.logger = logging.getLogger(__name__)
    
    def _create_provider(self) -> LLMProvider:
        """Create appropriate provider based on configuration."""
        if self.config.provider == "openai":
            return OpenAIProvider(self.config)
        elif self.config.provider == "anthropic":
            return AnthropicProvider(self.config)
        elif self.config.provider == "mock":
            return MockProvider(self.config)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def generate(self, prompt: Union[str, PromptTemplate], 
                      template_vars: Optional[Dict[str, Any]] = None,
                      **kwargs) -> LLMResponse:
        """Generate response with retry logic."""
        # Handle prompt template
        if isinstance(prompt, PromptTemplate):
            if template_vars:
                prompt_text = self._render_template(prompt, template_vars)
            else:
                prompt_text = prompt.content
        else:
            prompt_text = prompt
        
        # Retry logic
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(f"LLM request attempt {attempt + 1}")
                response = await self.provider.generate(prompt_text, **kwargs)
                self.logger.debug(f"LLM response received in {response.response_time:.2f}s")
                return response
            
            except (LLMTimeoutError, LLMRateLimitError) as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"LLM request failed (attempt {attempt + 1}), "
                                      f"retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"LLM request failed after {self.config.max_retries + 1} attempts")
                    raise e
            
            except LLMError as e:
                # Don't retry on other LLM errors
                self.logger.error(f"LLM request failed: {e}")
                raise e
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise LLMError("Unknown error in LLM request")
    
    def _render_template(self, template: PromptTemplate, 
                        variables: Dict[str, Any]) -> str:
        """Render prompt template with variables."""
        try:
            return template.content.format(**variables)
        except KeyError as e:
            missing_var = str(e).strip("'\"")
            raise ValueError(f"Missing template variable: {missing_var}")
        except Exception as e:
            raise ValueError(f"Error rendering template: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration for the current provider."""
        return self.provider.validate_config()
    
    async def test_connection(self) -> bool:
        """Test connection to the LLM provider."""
        try:
            response = await self.generate("Hello", max_tokens=10)
            return len(response.content) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False