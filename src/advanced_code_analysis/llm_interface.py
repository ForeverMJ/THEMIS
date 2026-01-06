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
        start_time = time.time()

        # gpt-5 系は responses API を使い、temperature=1固定かつ max_tokens
        if self.config.model_name.startswith("gpt-5"):
            # max_tokens 现在是 max_completion_tokens 的属性别名
            completion_value = kwargs.get("max_tokens", kwargs.get("max_completion_tokens", self.config.max_tokens))
            
            # Responses API 使用正确的参数名称
            params = {
                "model": self.config.model_name,
                "input": prompt,
                "max_output_tokens": completion_value,  # 使用 max_tokens 而不是 max_output_tokens
                "temperature": 1,  # gpt-5 要求 temperature=1
            }
            
            self.logger.debug(f"Calling responses API with model={self.config.model_name}, max_tokens={completion_value}")
            
            try:
                response = await client.responses.create(**params)
                response_time = time.time() - start_time

                # 详细日志
                self.logger.debug(f"Response received: type={type(response)}, has_output={hasattr(response, 'output')}")
                
                # レスポンスからテキストを抽出
                content_text = self._extract_response_text(response)
                
                if not content_text or not content_text.strip():
                    self.logger.warning(f"Empty response from gpt-5 API. Response object: {response}")
                    # 尝试直接访问 response 的属性
                    self.logger.debug(f"Response attributes: {dir(response)}")
                    if hasattr(response, 'output'):
                        self.logger.debug(f"Response.output: {response.output}")

                return LLMResponse(
                    content=content_text,
                    usage={
                        "prompt_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                        "completion_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                    },
                    model=response.model,
                    finish_reason=getattr(response, "finish_reason", "stop"),
                    response_time=response_time,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                self.logger.error(f"OpenAI responses API error: {e}", exc_info=True)
                if "timeout" in str(e).lower():
                    raise LLMTimeoutError(f"Request timed out: {e}")
                elif "rate limit" in str(e).lower():
                    raise LLMRateLimitError(f"Rate limit exceeded: {e}")
                else:
                    raise LLMError(f"OpenAI API error: {e}")

        # 従来の chat/completions
        # Support both max_tokens and max_completion_tokens for compatibility
        # max_tokens 现在是 max_completion_tokens 的属性别名
        completion_value = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or self.config.max_tokens
        requested_temp = kwargs.get("temperature", self.config.temperature)
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": completion_value,
            "temperature": requested_temp,
        }

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

    def _extract_response_text(self, response: Any) -> str:
        """Extract textual content from OpenAI responses output."""
        # responses API: output -> content -> text
        try:
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                self.logger.debug(f"Using response.output_text: {output_text[:100]}...")
                return output_text
        except Exception:
            pass

        try:
            if hasattr(response, "output"):
                output = getattr(response, "output", [])
                self.logger.debug(
                    f"Response has output attribute, type={type(output)}, "
                    f"len={len(output) if hasattr(output, '__len__') else 'N/A'}"
                )

                text_parts = []

                if isinstance(output, dict):
                    output_items = [output]
                elif isinstance(output, (list, tuple)):
                    output_items = list(output)
                elif hasattr(output, "__iter__") and not isinstance(output, (str, bytes)):
                    output_items = list(output)
                else:
                    output_items = [output]

                for i, item in enumerate(output_items):
                    if item is None:
                        continue
                    self.logger.debug(f"Processing output item {i}: type={type(item)}")

                    if hasattr(item, "text"):
                        text_val = getattr(item, "text", None)
                        if text_val:
                            self.logger.debug(f"Found item.text: {text_val[:100]}...")
                            text_parts.append(text_val)
                            continue
                    if hasattr(item, "output_text"):
                        text_val = getattr(item, "output_text", None)
                        if text_val:
                            self.logger.debug(f"Found item.output_text: {text_val[:100]}...")
                            text_parts.append(text_val)
                            continue

                    if hasattr(item, "content"):
                        content_list = getattr(item, "content", [])
                        self.logger.debug(
                            f"Item {i} has content: type={type(content_list)}, "
                            f"len={len(content_list) if hasattr(content_list, '__len__') else 'N/A'}"
                        )

                        if isinstance(content_list, dict):
                            content_iter = [content_list]
                        elif isinstance(content_list, (list, tuple)):
                            content_iter = list(content_list)
                        elif hasattr(content_list, "__iter__") and not isinstance(content_list, (str, bytes)):
                            content_iter = list(content_list)
                        else:
                            content_iter = [content_list]

                        for j, content in enumerate(content_iter):
                            if content is None:
                                continue
                            self.logger.debug(f"Processing content {j}: type={type(content)}")

                            if hasattr(content, "text"):
                                text_val = getattr(content, "text", None)
                                if text_val:
                                    self.logger.debug(f"Found text: {text_val[:100]}...")
                                    text_parts.append(text_val)
                                    continue
                            if hasattr(content, "output_text"):
                                text_val = getattr(content, "output_text", None)
                                if text_val:
                                    self.logger.debug(f"Found output_text: {text_val[:100]}...")
                                    text_parts.append(text_val)
                                    continue

                            if isinstance(content, dict):
                                text_val = content.get("text") or content.get("output_text")
                                if text_val:
                                    self.logger.debug(f"Found text in dict: {text_val[:100]}...")
                                    text_parts.append(text_val)
                                    continue

                    if isinstance(item, dict):
                        if "content" in item:
                            content_list = item["content"]
                            if isinstance(content_list, list):
                                for content in content_list:
                                    if isinstance(content, dict):
                                        text_val = content.get("text") or content.get("output_text")
                                        if text_val:
                                            text_parts.append(text_val)

                if text_parts:
                    result = "\n".join(text_parts)
                    self.logger.debug(
                        f"Extracted {len(text_parts)} text parts, total length={len(result)}"
                    )
                    return result
                else:
                    self.logger.warning("No text parts found in output")
        except Exception as e:
            self.logger.error(f"Error extracting response text: {e}", exc_info=True)

        # Fallback attributes
        for attr in ("output_text", "content", "message", "text"):
            try:
                val = getattr(response, attr, None)
                if isinstance(val, str) and val.strip():
                    self.logger.debug(f"Using fallback attribute '{attr}': {val[:100]}...")
                    return val
            except Exception:
                continue

        # Final fallback - convert to string
        self.logger.warning("All extraction methods failed, converting response to string")
        return str(response)

    def validate_config(self) -> List[str]:
        """Validate OpenAI-specific configuration."""
        issues = []
        
        if not self.config.api_key:
            issues.append("OpenAI API key is required")
        
        if self.config.model_name not in [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-5-mini", "gpt-5-mini"
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
        # Anthropic は max_completion_tokens が有効なので従来のまま。ただし gpt-5 系は対象外。
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": kwargs.get("max_completion_tokens", self.config.max_completion_tokens),
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

        # Usage tracking (best-effort; depends on provider returning usage)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.successful_requests: int = 0
    
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

                self._record_usage(response)
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

    def _record_usage(self, response: LLMResponse) -> None:
        """Record token usage from a successful response (best-effort)."""
        try:
            usage = response.usage or {}
            self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.total_tokens += int(usage.get("total_tokens", 0) or 0)
            self.successful_requests += 1
        except Exception:
            # Usage tracking must never break the request flow.
            pass

    def get_usage_stats(self) -> Dict[str, int]:
        """Get cumulative token usage stats for this interface instance."""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "successful_requests": self.successful_requests,
        }

    def reset_usage_stats(self) -> None:
        """Reset cumulative token usage stats."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.successful_requests = 0
    
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
            # gpt-5 系に配慮し completion キーを指定
            completion_kwargs = {"max_completion_tokens": 10} if self.config.model_name.startswith("gpt-5") else {"max_completion_tokens": 10}
            response = await self.generate("Hello", **completion_kwargs)
            return len(response.content) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
