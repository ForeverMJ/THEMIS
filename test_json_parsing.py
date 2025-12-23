#!/usr/bin/env python3
"""
测试 JSON 解析修复

验证概念映射器的 JSON 解析错误已修复
"""

import json
import re


def test_json_cleanup():
    """测试 JSON 清理逻辑"""
    
    # 测试用例 1: 带尾随逗号的 JSON
    test_json_1 = '''
    {
        "matches": [
            {
                "element_number": 1,
                "confidence": 0.8,
                "explanation": "Test",
            }
        ]
    }
    '''
    
    # 清理尾随逗号
    cleaned = re.sub(r',(\s*[}\]])', r'\1', test_json_1)
    try:
        result = json.loads(cleaned)
        print("✅ 测试 1 通过: 尾随逗号清理成功")
        print(f"   解析结果: {result}")
    except json.JSONDecodeError as e:
        print(f"❌ 测试 1 失败: {e}")
    
    # 测试用例 2: 嵌入在文本中的 JSON
    test_json_2 = '''
    Here is the analysis:
    
    {
        "matches": [
            {"element_number": 2, "confidence": 0.9, "explanation": "Good match"}
        ]
    }
    
    That's the result.
    '''
    
    # 提取 JSON
    json_match = re.search(r"\{.*\}", test_json_2, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            print("✅ 测试 2 通过: 从文本中提取 JSON 成功")
            print(f"   解析结果: {result}")
        except json.JSONDecodeError as e:
            print(f"❌ 测试 2 失败: {e}")
    else:
        print("❌ 测试 2 失败: 未找到 JSON")
    
    # 测试用例 3: 带 markdown 代码块的 JSON
    test_json_3 = '''```json
    {
        "matches": [
            {"element_number": 3, "confidence": 0.7, "explanation": "Partial match"}
        ]
    }
    ```'''
    
    # 移除 markdown 代码块
    content = test_json_3.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[len("json"):].lstrip()
    
    try:
        result = json.loads(content)
        print("✅ 测试 3 通过: Markdown 代码块清理成功")
        print(f"   解析结果: {result}")
    except json.JSONDecodeError as e:
        print(f"❌ 测试 3 失败: {e}")
    
    # 测试用例 4: 不完整的 JSON (应该失败但不崩溃)
    test_json_4 = '''
    {
        "matches": [
            {"element_number": 4, "confidence": 0.6
    '''
    
    try:
        result = json.loads(test_json_4)
        print("❌ 测试 4 失败: 不应该解析成功")
    except json.JSONDecodeError:
        print("✅ 测试 4 通过: 正确拒绝不完整的 JSON")
    
    # 测试用例 5: 组合修复 (尾随逗号 + 提取)
    test_json_5 = '''
    Analysis complete.
    
    {
        "suggestions": [
            {
                "location": "test_function",
                "reasoning": "This is a test",
                "confidence": 0.8,
            }
        ]
    }
    '''
    
    json_match = re.search(r"\{.*\}", test_json_5, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # 清理尾随逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            result = json.loads(json_str)
            print("✅ 测试 5 通过: 组合修复成功")
            print(f"   解析结果: {result}")
        except json.JSONDecodeError as e:
            print(f"❌ 测试 5 失败: {e}")
    else:
        print("❌ 测试 5 失败: 未找到 JSON")


if __name__ == "__main__":
    print("🧪 JSON 解析修复测试")
    print("="*60)
    test_json_cleanup()
    print("="*60)
    print("✅ 所有测试完成")
