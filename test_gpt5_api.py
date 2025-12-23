#!/usr/bin/env python3
"""
测试 GPT-5-mini Responses API 调用

验证 API 参数和响应提取是否正确
"""

import asyncio
import logging
from src.advanced_code_analysis.config import LLMConfig
from src.advanced_code_analysis.llm_interface import LLMInterface

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_gpt5_simple():
    """Test simple GPT-5 call"""
    print("="*80)
    print("Test 1: Simple text generation")
    print("="*80)
    
    config = LLMConfig(
        provider="openai",
        model_name="gpt-5-mini",
        max_completion_tokens=100,
        temperature=0.1
    )
    
    llm = LLMInterface(config)
    
    try:
        response = await llm.generate("Say hello in one sentence.", max_completion_tokens=50)
        
        print(f"\nSuccess:")
        print(f"   Content: {response.content}")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {response.usage}")
        print(f"   Time: {response.response_time:.2f}s")
        
        if not response.content or not response.content.strip():
            print(f"\nWarning: Empty response!")
            print(f"   Raw response: {response.raw_response}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def test_gpt5_json():
    """Test GPT-5 JSON generation"""
    print("\n" + "="*80)
    print("Test 2: JSON generation")
    print("="*80)
    
    config = LLMConfig(
        provider="openai",
        model_name="gpt-5-mini",
        max_completion_tokens=200,
        temperature=0.1
    )
    
    llm = LLMInterface(config)
    
    prompt = """
Generate a simple JSON object with two fields:
{
    "name": "test",
    "value": 123
}

Return ONLY the JSON, nothing else.
"""
    
    try:
        response = await llm.generate(prompt, max_completion_tokens=100)
        
        print(f"\nSuccess:")
        print(f"   Content: {response.content}")
        print(f"   Length: {len(response.content) if response.content else 0}")
        
        if response.content:
            # Try to parse JSON
            import json
            try:
                parsed = json.loads(response.content.strip())
                print(f"   JSON parsed successfully: {parsed}")
            except json.JSONDecodeError as e:
                print(f"   JSON parse failed: {e}")
        else:
            print(f"\nWarning: Empty response!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def test_gpt5_with_fallback():
    """Test GPT-5 with fallback"""
    print("\n" + "="*80)
    print("Test 3: Fallback to GPT-4o")
    print("="*80)
    
    # 先尝试 GPT-5
    config_gpt5 = LLMConfig(
        provider="openai",
        model_name="gpt-5-mini",
        max_completion_tokens=100
    )
    
    llm_gpt5 = LLMInterface(config_gpt5)
    
    try:
        response = await llm_gpt5.generate("Hello", max_completion_tokens=20)
        if response.content and response.content.strip():
            print(f"GPT-5 works: {response.content[:50]}...")
        else:
            print(f"GPT-5 returned empty response, trying fallback...")
            raise ValueError("Empty response")
    except Exception as e:
        print(f"GPT-5 failed: {e}")
        print(f"   Trying fallback to GPT-4o...")
        
        # Fallback to GPT-4o
        config_gpt4 = LLMConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            max_completion_tokens=100
        )
        
        llm_gpt4 = LLMInterface(config_gpt4)
        
        try:
            response = await llm_gpt4.generate("Hello", max_completion_tokens=20)
            print(f"GPT-4o works: {response.content[:50]}...")
        except Exception as e2:
            print(f"GPT-4o also failed: {e2}")


async def main():
    """运行所有测试"""
    print("GPT-5-mini API Test")
    print("="*80)
    
    await test_gpt5_simple()
    await test_gpt5_json()
    await test_gpt5_with_fallback()
    
    print("\n" + "="*80)
    print("All tests completed")
    print("="*80)
    
    print("\nSuggestions:")
    print("   If GPT-5 returns empty response:")
    print("   1. Check if API key has GPT-5 access")
    print("   2. Switch to GPT-4o: python switch_model.py gpt-4o")
    print("   3. Check detailed logs for errors")


if __name__ == "__main__":
    asyncio.run(main())
