"""
模型切换工具 - 便捷地在不同LLM模型之间切换

使用方法:
1. 通过配置文件: python src/model_switcher.py --config configs/models/openai_gpt4.json
2. 通过模型名称: python src/model_switcher.py --model gpt-4o
3. 通过环境变量: export LLM_MODEL=gpt-4o && python your_script.py
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from src.advanced_code_analysis.config import LLMConfig, AdvancedAnalysisConfig


# 预定义的模型配置
PRESET_MODELS = {
    "gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "max_completion_tokens": 4096,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "max_completion_tokens": 4096,
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "max_completion_tokens": 4096,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "max_completion_tokens": 4096,
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022",
        "max_completion_tokens": 4096,
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229",
        "max_completion_tokens": 4096,
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "max_completion_tokens": 4096,
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "max_completion_tokens": 4096,
    },
}


class ModelSwitcher:
    """模型切换器 - 管理和切换不同的LLM模型配置"""
    
    def __init__(self):
        self.config_dir = Path("configs/models")
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self, 
                        model_name: Optional[str] = None,
                        config_file: Optional[str] = None,
                        provider: Optional[str] = None) -> LLMConfig:
        """
        获取模型配置
        
        优先级: config_file > model_name > 环境变量 > 默认值
        """
        # 1. 从配置文件加载
        if config_file:
            return self._load_from_file(config_file)
        
        # 2. 从环境变量获取
        env_model = os.getenv("LLM_MODEL")
        env_provider = os.getenv("LLM_PROVIDER")
        
        # 3. 使用传入的参数或环境变量
        model_name = model_name or env_model or "gpt-5-mini"
        provider = provider or env_provider
        
        # 4. 从预设配置获取
        if model_name in PRESET_MODELS:
            preset = PRESET_MODELS[model_name]
            return LLMConfig(
                provider=preset["provider"],
                model_name=preset["model_name"],
                max_completion_tokens=preset["max_completion_tokens"],
            )
        
        # 5. 自定义配置
        if provider:
            return LLMConfig(
                provider=provider,
                model_name=model_name,
            )
        
        # 6. 默认为OpenAI
        return LLMConfig(
            provider="openai",
            model_name=model_name,
        )
    
    def _load_from_file(self, config_file: str) -> LLMConfig:
        """从JSON配置文件加载"""
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        llm_config = config_dict.get('llm', {})
        return LLMConfig(**llm_config)
    
    def list_available_models(self) -> Dict[str, Any]:
        """列出所有可用的预设模型"""
        return PRESET_MODELS
    
    def list_config_files(self) -> list:
        """列出所有配置文件"""
        return list(self.config_dir.glob("*.json"))
    
    def create_config_file(self, name: str, llm_config: LLMConfig):
        """创建新的配置文件"""
        config_path = self.config_dir / f"{name}.json"
        config_dict = {
            "llm": {
                "provider": llm_config.provider,
                "model_name": llm_config.model_name,
                "max_completion_tokens": llm_config.max_completion_tokens,
                "temperature": llm_config.temperature,
                "timeout": llm_config.timeout,
                "max_retries": llm_config.max_retries,
                "retry_delay": llm_config.retry_delay,
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return config_path
    
    def get_full_config(self, 
                       model_name: Optional[str] = None,
                       config_file: Optional[str] = None) -> AdvancedAnalysisConfig:
        """获取完整的分析配置（包含LLM配置）"""
        llm_config = self.get_model_config(model_name=model_name, config_file=config_file)
        
        # 创建完整配置
        full_config = AdvancedAnalysisConfig()
        full_config.llm = llm_config
        
        return full_config


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM模型切换工具")
    parser.add_argument("--model", "-m", help="模型名称 (例如: gpt-4o, claude-3.5-sonnet)")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--provider", "-p", help="提供商 (openai, anthropic)")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用模型")
    parser.add_argument("--list-configs", action="store_true", help="列出所有配置文件")
    
    args = parser.parse_args()
    
    switcher = ModelSwitcher()
    
    if args.list:
        print("\n可用的预设模型:")
        print("-" * 60)
        for name, config in switcher.list_available_models().items():
            print(f"  {name:20} - {config['provider']:12} - {config['model_name']}")
        return
    
    if args.list_configs:
        print("\n可用的配置文件:")
        print("-" * 60)
        for config_file in switcher.list_config_files():
            print(f"  {config_file}")
        return
    
    # 获取配置
    try:
        llm_config = switcher.get_model_config(
            model_name=args.model,
            config_file=args.config,
            provider=args.provider
        )
        
        print("\n当前模型配置:")
        print("-" * 60)
        print(f"  提供商: {llm_config.provider}")
        print(f"  模型: {llm_config.model_name}")
        print(f"  最大令牌数: {llm_config.max_completion_tokens}")
        print(f"  温度: {llm_config.temperature}")
        print(f"  超时: {llm_config.timeout}秒")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
