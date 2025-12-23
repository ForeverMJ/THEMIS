#!/usr/bin/env python3
"""
测试环境变量覆盖逻辑

验证 .env 中的 LLM_MODEL 和 LLM_PROVIDER 是否正确覆盖默认值
"""

import os
from dotenv import load_dotenv
from src.advanced_code_analysis.config import LLMConfig

def test_default_values():
    """测试默认值（没有环境变量）"""
    print("="*70)
    print("测试 1: 默认值（清除环境变量）")
    print("="*70)
    
    # 保存并清除环境变量
    original_model = os.environ.pop("LLM_MODEL", None)
    original_provider = os.environ.pop("LLM_PROVIDER", None)
    
    try:
        config = LLMConfig()
        
        print(f"  默认 model_name: {config.model_name}")
        print(f"  默认 provider: {config.provider}")
        
        assert config.model_name == "gpt-4o-mini", f"Expected 'gpt-4o-mini', got '{config.model_name}'"
        assert config.provider == "openai", f"Expected 'openai', got '{config.provider}'"
        
        print("  ✓ 默认值正确")
        return True
        
    finally:
        # 恢复环境变量
        if original_model:
            os.environ["LLM_MODEL"] = original_model
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider


def test_env_override():
    """测试环境变量覆盖"""
    print("\n" + "="*70)
    print("测试 2: 环境变量覆盖")
    print("="*70)
    
    # 设置测试环境变量
    os.environ["LLM_MODEL"] = "gpt-4o"
    os.environ["LLM_PROVIDER"] = "openai"
    
    config = LLMConfig()
    
    print(f"  环境变量 LLM_MODEL: {os.getenv('LLM_MODEL')}")
    print(f"  环境变量 LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"  配置 model_name: {config.model_name}")
    print(f"  配置 provider: {config.provider}")
    
    assert config.model_name == "gpt-4o", f"Expected 'gpt-4o', got '{config.model_name}'"
    assert config.provider == "openai", f"Expected 'openai', got '{config.provider}'"
    
    print("  ✓ 环境变量覆盖成功")
    return True


def test_dotenv_loading():
    """测试从 .env 文件加载"""
    print("\n" + "="*70)
    print("测试 3: 从 .env 文件加载")
    print("="*70)
    
    # 清除环境变量
    os.environ.pop("LLM_MODEL", None)
    os.environ.pop("LLM_PROVIDER", None)
    
    # 加载 .env 文件
    load_dotenv(override=True)
    
    print(f"  .env 加载后 LLM_MODEL: {os.getenv('LLM_MODEL')}")
    print(f"  .env 加载后 LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    
    config = LLMConfig()
    
    print(f"  配置 model_name: {config.model_name}")
    print(f"  配置 provider: {config.provider}")
    
    # .env 文件中的值应该被使用
    env_model = os.getenv('LLM_MODEL')
    if env_model:
        assert config.model_name == env_model, f"Expected '{env_model}', got '{config.model_name}'"
        print(f"  ✓ 使用了 .env 中的模型: {env_model}")
    else:
        assert config.model_name == "gpt-4o-mini", "Should use default when .env is empty"
        print(f"  ✓ .env 中没有 LLM_MODEL，使用默认值")
    
    return True


def test_explicit_config_priority():
    """测试显式配置的优先级"""
    print("\n" + "="*70)
    print("测试 4: 显式配置 vs 环境变量")
    print("="*70)
    
    # 设置环境变量
    os.environ["LLM_MODEL"] = "gpt-4o"
    
    # 显式指定配置（应该被环境变量覆盖）
    config = LLMConfig(model_name="gpt-3.5-turbo")
    
    print(f"  显式指定: gpt-3.5-turbo")
    print(f"  环境变量: {os.getenv('LLM_MODEL')}")
    print(f"  最终配置: {config.model_name}")
    
    # 当前实现：环境变量优先级更高
    assert config.model_name == "gpt-4o", f"Environment variable should override, got '{config.model_name}'"
    
    print("  ✓ 环境变量优先级高于显式配置")
    print("  注意: 如果需要显式配置优先，需要修改 __post_init__ 逻辑")
    
    return True


def main():
    print("环境变量覆盖逻辑测试")
    print("="*70)
    
    try:
        test_default_values()
        test_env_override()
        test_dotenv_loading()
        test_explicit_config_priority()
        
        print("\n" + "="*70)
        print("✓ 所有测试通过!")
        print("="*70)
        
        print("\n总结:")
        print("  1. 默认值: gpt-4o-mini")
        print("  2. 环境变量会覆盖默认值")
        print("  3. .env 文件通过 load_dotenv() 设置环境变量")
        print("  4. 当前实现: 环境变量 > 显式配置 > 默认值")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
