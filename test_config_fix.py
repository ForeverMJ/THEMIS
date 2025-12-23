#!/usr/bin/env python3
"""
测试 LLMConfig 的 max_tokens 属性修复
"""

from src.advanced_code_analysis.config import LLMConfig

def test_max_tokens_property():
    """测试 max_tokens 属性是否正常工作"""
    print("Testing LLMConfig.max_tokens property...")
    
    config = LLMConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        max_completion_tokens=2000
    )
    
    print(f"  max_completion_tokens: {config.max_completion_tokens}")
    print(f"  max_tokens (property): {config.max_tokens}")
    
    assert config.max_tokens == config.max_completion_tokens, "max_tokens should equal max_completion_tokens"
    assert config.max_tokens == 2000, "max_tokens should be 2000"
    
    print("  ✓ max_tokens property works correctly")
    
    # Test that it's read-only (property)
    try:
        config.max_tokens = 3000
        print("  ✗ max_tokens should be read-only!")
    except AttributeError:
        print("  ✓ max_tokens is read-only (as expected)")
    
    return True


def test_default_model():
    """测试默认模型是否改为 gpt-4o-mini"""
    print("\nTesting default model...")
    
    config = LLMConfig()
    
    print(f"  Default model: {config.model_name}")
    assert config.model_name == "gpt-4o-mini", "Default model should be gpt-4o-mini"
    print("  ✓ Default model is gpt-4o-mini")
    
    return True


def test_env_override():
    """测试环境变量覆盖"""
    print("\nTesting environment variable override...")
    
    import os
    
    # 保存原始值
    original_model = os.getenv("LLM_MODEL")
    original_provider = os.getenv("LLM_PROVIDER")
    
    try:
        # 设置测试值
        os.environ["LLM_MODEL"] = "gpt-4o"
        os.environ["LLM_PROVIDER"] = "openai"
        
        config = LLMConfig()
        
        print(f"  Model from env: {config.model_name}")
        print(f"  Provider from env: {config.provider}")
        
        assert config.model_name == "gpt-4o", "Model should be overridden by env"
        assert config.provider == "openai", "Provider should be overridden by env"
        
        print("  ✓ Environment variable override works")
        
        return True
        
    finally:
        # 恢复原始值
        if original_model:
            os.environ["LLM_MODEL"] = original_model
        elif "LLM_MODEL" in os.environ:
            del os.environ["LLM_MODEL"]
        
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


def main():
    print("="*60)
    print("LLMConfig Fix Verification")
    print("="*60)
    
    try:
        test_max_tokens_property()
        test_default_model()
        test_env_override()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
