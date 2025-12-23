#!/usr/bin/env python3
"""
快速模型切换脚本

使用方法:
    python switch_model.py gpt-4o
    python switch_model.py claude-3.5-sonnet
    python switch_model.py --list
"""

import sys
import os
from pathlib import Path
from src.model_switcher import ModelSwitcher, PRESET_MODELS


def update_env_file(model_name: str, provider: str):
    """更新 .env 文件中的模型配置"""
    env_path = Path(".env")
    
    if not env_path.exists():
        # 如果 .env 不存在，从 .env.example 复制
        example_path = Path(".env.example")
        if example_path.exists():
            with open(example_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    # 读取现有内容
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 更新或添加模型配置
    model_found = False
    provider_found = False
    new_lines = []
    
    for line in lines:
        if line.startswith('LLM_MODEL='):
            new_lines.append(f'LLM_MODEL={model_name}\n')
            model_found = True
        elif line.startswith('LLM_PROVIDER='):
            new_lines.append(f'LLM_PROVIDER={provider}\n')
            provider_found = True
        else:
            new_lines.append(line)
    
    # 如果没找到，添加到文件末尾
    if not model_found:
        new_lines.append(f'\nLLM_MODEL={model_name}\n')
    if not provider_found:
        new_lines.append(f'LLM_PROVIDER={provider}\n')
    
    # 写回文件
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    return env_path


def main():
    if len(sys.argv) < 2:
        print("使用方法: python switch_model.py <model_name>")
        print("或者: python switch_model.py --list")
        return 1
    
    switcher = ModelSwitcher()
    
    # 列出可用模型
    if sys.argv[1] in ['--list', '-l']:
        print("\n" + "="*70)
        print("可用的预设模型:")
        print("="*70)
        
        # 按提供商分组
        openai_models = []
        anthropic_models = []
        
        for name, config in PRESET_MODELS.items():
            if config['provider'] == 'openai':
                openai_models.append((name, config))
            elif config['provider'] == 'anthropic':
                anthropic_models.append((name, config))
        
        print("\n🤖 OpenAI 模型:")
        print("-" * 70)
        for name, config in openai_models:
            print(f"  • {name:20} → {config['model_name']}")
        
        print("\n🧠 Anthropic 模型:")
        print("-" * 70)
        for name, config in anthropic_models:
            print(f"  • {name:20} → {config['model_name']}")
        
        print("\n" + "="*70)
        print("使用方法: python switch_model.py <model_name>")
        print("例如: python switch_model.py gpt-4o")
        print("="*70 + "\n")
        return 0
    
    model_name = sys.argv[1]
    
    # 检查模型是否存在
    if model_name not in PRESET_MODELS:
        print(f"❌ 错误: 未知的模型 '{model_name}'")
        print("\n可用的模型:")
        for name in PRESET_MODELS.keys():
            print(f"  • {name}")
        print("\n使用 'python switch_model.py --list' 查看详细信息")
        return 1
    
    # 获取模型配置
    model_config = PRESET_MODELS[model_name]
    provider = model_config['provider']
    
    try:
        # 更新 .env 文件
        env_path = update_env_file(model_name, provider)
        
        # 验证配置
        llm_config = switcher.get_model_config(model_name=model_name)
        
        print("\n" + "="*70)
        print("✅ 模型切换成功!")
        print("="*70)
        print(f"\n当前配置:")
        print(f"  • 模型名称: {model_name}")
        print(f"  • 提供商: {provider}")
        print(f"  • 完整模型名: {llm_config.model_name}")
        print(f"  • 最大令牌数: {llm_config.max_completion_tokens}")
        print(f"  • 配置文件: {env_path}")
        
        # 检查API密钥
        api_key_var = f"{provider.upper()}_API_KEY"
        if not llm_config.api_key:
            print(f"\n⚠️  警告: 未找到 {api_key_var}")
            print(f"   请在 .env 文件中设置: {api_key_var}=your_key_here")
        else:
            key_preview = llm_config.api_key[:10] + "..." if len(llm_config.api_key) > 10 else "***"
            print(f"\n✓ API密钥已配置: {key_preview}")
        
        print("\n" + "="*70)
        print("现在可以运行你的脚本，它将使用新的模型配置")
        print("例如: python run_experiment_enhanced.py")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
