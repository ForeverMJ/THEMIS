#!/usr/bin/env python3
"""
测试模型切换功能

使用方法:
    python test_model_switch.py
"""

import os
from src.model_switcher import ModelSwitcher
from src.advanced_code_analysis.config import LLMConfig


def test_preset_models():
    """测试预设模型"""
    print("\n" + "="*70)
    print("测试 1: 预设模型切换")
    print("="*70)
    
    switcher = ModelSwitcher()
    test_models = ["gpt-4o", "gpt-3.5-turbo", "claude-3.5-sonnet"]
    
    for model_name in test_models:
        try:
            config = switcher.get_model_config(model_name=model_name)
            print(f"\n✓ {model_name:20} → {config.provider:12} / {config.model_name}")
        except Exception as e:
            print(f"\n✗ {model_name:20} → 错误: {e}")


def test_config_files():
    """测试配置文件"""
    print("\n" + "="*70)
    print("测试 2: 配置文件加载")
    print("="*70)
    
    switcher = ModelSwitcher()
    config_files = list(switcher.config_dir.glob("*.json"))
    
    if not config_files:
        print("\n⚠️  未找到配置文件")
        return
    
    for config_file in config_files[:3]:  # 只测试前3个
        try:
            config = switcher.get_model_config(config_file=str(config_file))
            print(f"\n✓ {config_file.name:30} → {config.provider} / {config.model_name}")
        except Exception as e:
            print(f"\n✗ {config_file.name:30} → 错误: {e}")


def test_env_variables():
    """测试环境变量"""
    print("\n" + "="*70)
    print("测试 3: 环境变量配置")
    print("="*70)
    
    # 保存原始环境变量
    original_model = os.getenv("LLM_MODEL")
    original_provider = os.getenv("LLM_PROVIDER")
    
    try:
        # 测试设置环境变量
        os.environ["LLM_MODEL"] = "gpt-4o"
        os.environ["LLM_PROVIDER"] = "openai"
        
        config = LLMConfig()  # 应该从环境变量读取
        
        print(f"\n环境变量:")
        print(f"  LLM_MODEL = {os.getenv('LLM_MODEL')}")
        print(f"  LLM_PROVIDER = {os.getenv('LLM_PROVIDER')}")
        print(f"\n加载的配置:")
        print(f"  provider = {config.provider}")
        print(f"  model_name = {config.model_name}")
        
        if config.model_name == "gpt-4o" and config.provider == "openai":
            print("\n✓ 环境变量配置正常工作")
        else:
            print("\n✗ 环境变量配置未生效")
    
    finally:
        # 恢复原始环境变量
        if original_model:
            os.environ["LLM_MODEL"] = original_model
        elif "LLM_MODEL" in os.environ:
            del os.environ["LLM_MODEL"]
        
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]


def test_full_config():
    """测试完整配置"""
    print("\n" + "="*70)
    print("测试 4: 完整配置生成")
    print("="*70)
    
    switcher = ModelSwitcher()
    
    try:
        full_config = switcher.get_full_config(model_name="gpt-4o")
        
        print(f"\n✓ 完整配置生成成功")
        print(f"  LLM Provider: {full_config.llm.provider}")
        print(f"  LLM Model: {full_config.llm.model_name}")
        print(f"  Max Tokens: {full_config.llm.max_completion_tokens}")
        print(f"  Temperature: {full_config.llm.temperature}")
        print(f"  Confidence Threshold: {full_config.analysis.confidence_threshold}")
        
    except Exception as e:
        print(f"\n✗ 完整配置生成失败: {e}")


def test_api_keys():
    """测试API密钥配置"""
    print("\n" + "="*70)
    print("测试 5: API密钥检查")
    print("="*70)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"\nOpenAI API Key: {'✓ 已配置' if openai_key else '✗ 未配置'}")
    if openai_key:
        print(f"  预览: {openai_key[:10]}...")
    
    print(f"\nAnthropic API Key: {'✓ 已配置' if anthropic_key else '✗ 未配置'}")
    if anthropic_key:
        print(f"  预览: {anthropic_key[:10]}...")
    
    if not openai_key and not anthropic_key:
        print("\n⚠️  警告: 未找到任何API密钥")
        print("   请在 .env 文件中配置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY")


def main():
    print("\n" + "="*70)
    print("模型切换功能测试")
    print("="*70)
    
    try:
        test_preset_models()
        test_config_files()
        test_env_variables()
        test_full_config()
        test_api_keys()
        
        print("\n" + "="*70)
        print("✅ 所有测试完成")
        print("="*70)
        print("\n使用 'python switch_model.py --list' 查看所有可用模型")
        print("使用 'python switch_model.py <model_name>' 切换模型\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
