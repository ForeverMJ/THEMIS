"""
Demo mode for Advanced Code Analysis system.

This script demonstrates the system capabilities without requiring API keys
by using mock responses and showing the analysis pipeline structure.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List

# Import system components
try:
    from src.advanced_code_analysis.advanced_code_analyzer import AdvancedCodeAnalyzer
    from src.advanced_code_analysis.config import AdvancedAnalysisConfig
    from src.advanced_code_analysis.models import (
        BugType, AnalysisResult, AnalysisStrategy, ContextWindow
    )
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    print("❌ Advanced analysis system components not available")


def load_text(path: Path) -> str:
    """Load text file with UTF-8 encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def create_mock_analysis_result(issue_text: str, source_code: str) -> AnalysisResult:
    """Create a mock analysis result for demonstration."""
    
    # Analyze the issue text to determine likely bug type
    bug_type = "logic_error"
    if "api" in issue_text.lower() or "function" in issue_text.lower():
        bug_type = "api_issue"
    elif "performance" in issue_text.lower() or "slow" in issue_text.lower():
        bug_type = "performance"
    elif "boundary" in issue_text.lower() or "edge" in issue_text.lower():
        bug_type = "boundary_condition"
    
    # Create mock findings based on common patterns
    findings = []
    recommendations = []
    
    if "dot" in issue_text.lower() and "blueprint" in issue_text.lower():
        findings = [
            "检测到Blueprint名称验证问题",
            "发现字符串处理逻辑可能存在缺陷",
            "缺少对特殊字符的适当验证"
        ]
        recommendations = [
            "添加Blueprint名称验证函数，检查是否包含点号",
            "实现适当的错误处理机制",
            "添加单元测试覆盖边界情况"
        ]
    elif "password" in issue_text.lower():
        findings = [
            "密码验证函数存在异常处理问题",
            "错误处理逻辑不完整",
            "缺少适当的日志记录"
        ]
        recommendations = [
            "修改validate_password函数的异常处理",
            "添加try-catch块处理密码比较失败",
            "实现安全的错误消息返回机制"
        ]
    else:
        findings = [
            f"识别为{bug_type}类型的问题",
            "代码结构分析完成",
            "发现潜在的改进点"
        ]
        recommendations = [
            "建议进行详细的代码审查",
            "考虑添加更多的错误处理",
            "增加相关的测试用例"
        ]
    
    return AnalysisResult(
        bug_location="待进一步分析确定",
        root_cause=f"基于{bug_type}模式的初步分析",
        fix_suggestion="请参考推荐建议进行修复",
        confidence=0.75,
        reasoning_chain=[
            "分析问题描述和代码结构",
            f"识别为{bug_type}类型问题",
            "生成相应的修复建议",
            "评估分析置信度"
        ],
        supporting_evidence=[
            "问题描述关键词匹配",
            "代码模式识别",
            "历史案例对比"
        ]
    )


def demonstrate_bug_classification(issue_text: str):
    """Demonstrate bug classification capabilities."""
    
    print("🔍 Bug Classification Demo")
    print("-" * 40)
    
    # Simulate classification logic
    keywords = {
        "logic_error": ["logic", "condition", "if", "else", "loop"],
        "api_issue": ["api", "function", "method", "call", "parameter"],
        "performance": ["slow", "performance", "optimization", "speed"],
        "boundary_condition": ["boundary", "edge", "limit", "range"]
    }
    
    scores = {}
    for bug_type, words in keywords.items():
        score = sum(1 for word in words if word in issue_text.lower())
        scores[bug_type] = score
    
    best_match = max(scores, key=scores.get)
    confidence = scores[best_match] / len(keywords[best_match])
    
    print(f"📊 Classification Results:")
    print(f"   Primary Type: {best_match}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   All Scores: {scores}")
    
    return best_match, confidence


def demonstrate_context_enhancement(source_code: str):
    """Demonstrate context enhancement capabilities."""
    
    print("\n📚 Context Enhancement Demo")
    print("-" * 40)
    
    # Analyze code structure
    lines = source_code.split('\n')
    functions = [line.strip() for line in lines if line.strip().startswith('def ')]
    classes = [line.strip() for line in lines if line.strip().startswith('class ')]
    imports = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
    
    print(f"📊 Code Structure Analysis:")
    print(f"   Total lines: {len(lines)}")
    print(f"   Functions: {len(functions)}")
    print(f"   Classes: {len(classes)}")
    print(f"   Imports: {len(imports)}")
    
    if functions:
        print(f"   Function examples: {functions[:3]}")
    if classes:
        print(f"   Class examples: {classes[:2]}")
    if imports:
        print(f"   Import examples: {imports[:3]}")
    
    # Estimate context complexity
    complexity_score = len(functions) * 2 + len(classes) * 3 + len(imports)
    print(f"   Complexity Score: {complexity_score}")
    
    return {
        'total_lines': len(lines),
        'functions': len(functions),
        'classes': len(classes),
        'imports': len(imports),
        'complexity_score': complexity_score
    }


def demonstrate_pattern_matching(issue_text: str, source_code: str):
    """Demonstrate pattern matching capabilities."""
    
    print("\n🎯 Pattern Matching Demo")
    print("-" * 40)
    
    # Common bug patterns
    patterns = {
        "assignment_error": ["=", "==", "assign"],
        "null_pointer": ["None", "null", "NoneType"],
        "type_error": ["type", "TypeError", "isinstance"],
        "index_error": ["index", "IndexError", "list"],
        "validation_error": ["validate", "check", "verify"]
    }
    
    detected_patterns = []
    for pattern_name, keywords in patterns.items():
        if any(keyword in issue_text.lower() or keyword in source_code.lower() 
               for keyword in keywords):
            detected_patterns.append(pattern_name)
    
    print(f"🔍 Detected Patterns:")
    if detected_patterns:
        for pattern in detected_patterns:
            print(f"   ✓ {pattern}")
    else:
        print("   No specific patterns detected")
    
    # Suggest analysis strategies
    print(f"\n💡 Suggested Analysis Strategies:")
    if "validation_error" in detected_patterns:
        print("   • Focus on input validation logic")
        print("   • Check error handling mechanisms")
    if "type_error" in detected_patterns:
        print("   • Analyze type conversions")
        print("   • Review function signatures")
    if not detected_patterns:
        print("   • General code review approach")
        print("   • Structural analysis recommended")
    
    return detected_patterns


def demonstrate_multi_round_reasoning(issue_text: str):
    """Demonstrate multi-round reasoning capabilities."""
    
    print("\n🧠 Multi-Round Reasoning Demo")
    print("-" * 40)
    
    reasoning_steps = [
        "初始问题分析：理解用户描述的问题",
        "代码结构分析：识别相关的代码组件",
        "模式匹配：查找已知的bug模式",
        "上下文收集：收集相关的代码上下文",
        "假设生成：基于分析生成可能的原因",
        "验证假设：检查假设的合理性",
        "生成建议：提供具体的修复建议"
    ]
    
    print("🔄 Reasoning Process:")
    for i, step in enumerate(reasoning_steps, 1):
        print(f"   {i}. {step}")
    
    # Simulate confidence evolution
    confidence_evolution = [0.3, 0.5, 0.6, 0.7, 0.8, 0.75, 0.85]
    
    print(f"\n📈 Confidence Evolution:")
    for i, conf in enumerate(confidence_evolution, 1):
        print(f"   Round {i}: {conf:.2f}")
    
    final_confidence = confidence_evolution[-1]
    print(f"\n🎯 Final Confidence: {final_confidence:.2f}")
    
    return reasoning_steps, final_confidence


async def run_demo():
    """Run the complete demo."""
    
    print("🚀 Advanced Code Analysis System - Demo Mode")
    print("=" * 60)
    print("📝 Note: This demo uses mock responses to show system capabilities")
    print("   For real analysis, configure API keys and use run_quick_test.py")
    print()
    
    # Load experiment data
    base = Path(__file__).parent
    req_path = base / "experiment_data" / "issue.txt"
    code_path = base / "experiment_data" / "source_code.py"
    
    try:
        issue_text = load_text(req_path)
        source_code = load_text(code_path)
    except FileNotFoundError as e:
        print(f"❌ Could not load experiment data: {e}")
        print("Using sample data for demo...")
        issue_text = "There's a validation error in the user input function. When users provide invalid data, the system crashes instead of showing an error message."
        source_code = """
def validate_user_input(data):
    if data is None:
        return False
    # Missing validation logic here
    return process_data(data)

def process_data(data):
    return data.upper()  # This will crash if data is not a string
"""
    
    print(f"📋 Demo Data:")
    print(f"   Issue: {issue_text[:100]}...")
    print(f"   Code length: {len(source_code)} characters")
    
    # Demonstrate each component
    bug_type, classification_confidence = demonstrate_bug_classification(issue_text)
    context_info = demonstrate_context_enhancement(source_code)
    detected_patterns = demonstrate_pattern_matching(issue_text, source_code)
    reasoning_steps, final_confidence = demonstrate_multi_round_reasoning(issue_text)
    
    # Generate mock analysis result
    print("\n📊 Complete Analysis Result")
    print("=" * 60)
    
    mock_result = create_mock_analysis_result(issue_text, source_code)
    
    print(f"🎯 Analysis Summary:")
    print(f"   Bug Type: {bug_type}")
    print(f"   Classification Confidence: {classification_confidence:.2f}")
    print(f"   Final Confidence: {final_confidence:.2f}")
    print(f"   Processing Time: 2.34s (simulated)")
    
    print(f"\n🔍 Key Findings:")
    for i, finding in enumerate(mock_result.supporting_evidence, 1):
        print(f"   {i}. {finding}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(mock_result.reasoning_chain, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n🧠 System Capabilities Demonstrated:")
    print(f"   ✓ Intelligent bug classification")
    print(f"   ✓ Context-aware code analysis")
    print(f"   ✓ Pattern-based problem detection")
    print(f"   ✓ Multi-round reasoning process")
    print(f"   ✓ Confidence scoring and validation")
    print(f"   ✓ Structured recommendation generation")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Configure API keys in .env file")
    print(f"   2. Run 'python run_quick_test.py' for real analysis")
    print(f"   3. Try 'python run_experiment_advanced.py' for comprehensive testing")
    print(f"   4. Use the system in your own code analysis workflows")


async def main():
    """Main demo function."""
    await run_demo()


if __name__ == "__main__":
    asyncio.run(main())