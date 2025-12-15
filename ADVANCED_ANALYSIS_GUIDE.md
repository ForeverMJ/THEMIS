# 高级代码分析系统使用指南

## 概述

高级代码分析系统是一个基于LLM的智能代码分析工具，专门设计用于处理复杂的软件工程问题。该系统通过以下核心创新提升代码分析能力：

- 🧠 **智能问题分类** - 根据bug类型选择最适合的分析策略
- 📚 **上下文增强** - 为LLM提供丰富的代码上下文和领域知识
- 🎯 **模式学习** - 从成功案例中积累bug模式知识
- 🔄 **多轮推理** - 通过多步骤验证提高分析准确性

## 快速开始

### 1. 演示模式（无需API密钥）

最快的体验方式，展示系统功能：

```bash
python run_demo_mode.py
```

这将展示系统的各个组件如何工作，包括：
- Bug分类演示
- 上下文增强分析
- 模式匹配检测
- 多轮推理过程

### 2. 配置API密钥

要使用真实的LLM分析功能，需要配置API密钥：

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，添加你的API密钥
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. 快速测试

验证系统是否正常工作：

```bash
# 基本测试
python run_quick_test.py

# 测试所有策略
python run_quick_test.py --all-strategies
```

### 4. 完整实验

运行完整的分析实验：

```bash
python run_experiment_advanced.py
```

## 分析策略

系统提供四种分析策略：

### AUTO_SELECT（推荐）
- 自动根据问题类型选择最佳策略
- 适合大多数使用场景
- 平衡准确性和性能

### ADVANCED_ONLY
- 纯LLM驱动的语义分析
- 适合复杂逻辑错误
- 提供详细推理链

### GRAPH_ONLY
- 基于代码结构的分析
- 适合架构问题
- 快速且可靠

### INTEGRATED
- 结合LLM和图分析
- 最全面的分析结果
- 需要更多计算资源

## 使用示例

### 基本API使用

```python
import asyncio
from src.enhanced_graph_adapter import (
    EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions
)

async def analyze_code():
    # 初始化分析器
    adapter = EnhancedGraphAdapter()
    
    # 配置分析选项
    options = AnalysisOptions(
        strategy=AnalysisStrategy.AUTO_SELECT,
        confidence_threshold=0.6,
        debug_mode=True
    )
    
    # 执行分析
    result = await adapter.analyze(
        issue_text="函数中存在逻辑错误，需要修复",
        target_files=["my_code.py"],
        options=options
    )
    
    # 处理结果
    if result.success:
        print(f"置信度: {result.confidence_score:.2f}")
        print("发现的问题:")
        for finding in result.primary_findings:
            print(f"  - {finding}")
        print("修复建议:")
        for rec in result.recommendations:
            print(f"  - {rec}")
    else:
        print(f"分析失败: {result.error_message}")

# 运行分析
asyncio.run(analyze_code())
```

### 批量分析

```python
async def batch_analysis():
    adapter = EnhancedGraphAdapter()
    
    # 定义测试案例
    test_cases = [
        {
            "name": "逻辑错误案例",
            "issue": "if条件判断错误导致程序行为异常",
            "files": ["logic_error.py"]
        },
        {
            "name": "API使用错误",
            "issue": "函数调用参数不正确",
            "files": ["api_error.py"]
        }
    ]
    
    results = []
    for case in test_cases:
        result = await adapter.analyze(
            issue_text=case["issue"],
            target_files=case["files"],
            options=AnalysisOptions(strategy=AnalysisStrategy.AUTO_SELECT)
        )
        results.append((case["name"], result))
    
    # 分析结果
    for name, result in results:
        status = "✅" if result.success else "❌"
        confidence = result.confidence_score if result.success else 0
        print(f"{status} {name}: 置信度 {confidence:.2f}")

asyncio.run(batch_analysis())
```

## 结果解读

### 分析结果结构

```python
class UnifiedAnalysisResult:
    strategy_used: AnalysisStrategy      # 使用的分析策略
    success: bool                        # 分析是否成功
    confidence_score: float              # 置信度评分 (0-1)
    processing_time: float               # 处理时间（秒）
    primary_findings: List[str]          # 主要发现
    recommendations: List[str]           # 修复建议
    error_message: Optional[str]         # 错误信息
    
    # 高级分析特有
    bug_classification: Optional[str]    # Bug分类
    reasoning_chain: List[str]          # 推理链
    
    # 图分析特有
    graph_statistics: Optional[Dict]     # 图统计信息
    violation_report: Optional[Dict]     # 违规报告
```

### 置信度评分指南

- **0.8-1.0**: 高置信度，建议直接采用分析结果
- **0.6-0.8**: 中等置信度，建议结合人工审查
- **0.4-0.6**: 低置信度，需要进一步分析
- **0.0-0.4**: 很低置信度，可能需要更多上下文

## 性能优化

### 配置优化

```python
# 高性能配置
options = AnalysisOptions(
    strategy=AnalysisStrategy.GRAPH_ONLY,  # 更快的策略
    max_context_tokens=4000,               # 减少token使用
    confidence_threshold=0.5,              # 降低阈值
    max_concurrent_requests=1              # 减少并发
)

# 高准确性配置
options = AnalysisOptions(
    strategy=AnalysisStrategy.INTEGRATED,  # 最全面的分析
    max_context_tokens=8000,               # 更多上下文
    confidence_threshold=0.7,              # 提高阈值
    debug_mode=True                        # 详细日志
)
```

### 缓存使用

```python
# 启用缓存以避免重复分析
adapter = EnhancedGraphAdapter()

# 第一次分析
result1 = await adapter.analyze(issue_text, files, options)

# 相同输入的第二次分析将使用缓存
result2 = await adapter.analyze(issue_text, files, options)

# 清除缓存
adapter.clear_caches()
```

## 故障排除

### 常见问题

1. **API密钥错误**
   ```
   Error: The api_key client option must be set
   ```
   解决：检查 `.env` 文件中的 `OPENAI_API_KEY` 设置

2. **分析失败**
   ```
   Analysis failed: Rate limit exceeded
   ```
   解决：降低 `max_concurrent_requests` 或等待API限制重置

3. **置信度过低**
   ```
   Low confidence score: 0.3
   ```
   解决：提供更详细的问题描述或使用INTEGRATED策略

### 调试技巧

1. **启用调试模式**
   ```python
   options = AnalysisOptions(debug_mode=True)
   ```

2. **检查系统状态**
   ```python
   adapter = EnhancedGraphAdapter()
   status = adapter.get_system_status()
   print(f"可用系统: {status['systems_initialized']}")
   print(f"可用策略: {status['available_strategies']}")
   ```

3. **验证系统配置**
   ```python
   validation_results = await adapter.validate_systems()
   for system, issues in validation_results.items():
       if issues:
           print(f"{system} 问题: {issues}")
   ```

## 扩展和定制

### 自定义分析策略

系统支持添加自定义分析策略：

```python
# 实现自定义策略
class CustomAnalysisStrategy:
    def analyze(self, issue_text, code_files):
        # 自定义分析逻辑
        pass

# 注册策略
adapter.register_strategy("custom", CustomAnalysisStrategy())
```

### 添加新的Bug模式

```python
# 定义新的bug模式
new_pattern = {
    "name": "custom_pattern",
    "description": "自定义bug模式",
    "regex_patterns": [r"pattern1", r"pattern2"],
    "fix_templates": ["修复模板1", "修复模板2"]
}

# 添加到系统
adapter.add_bug_pattern(new_pattern)
```

## 最佳实践

1. **选择合适的策略**
   - 逻辑错误：ADVANCED_ONLY 或 AUTO_SELECT
   - 架构问题：GRAPH_ONLY
   - 复杂问题：INTEGRATED

2. **优化问题描述**
   - 提供具体的错误现象
   - 包含相关的上下文信息
   - 说明期望的行为

3. **合理设置参数**
   - 根据代码复杂度调整 `max_context_tokens`
   - 根据准确性要求设置 `confidence_threshold`
   - 在性能和准确性之间找到平衡

4. **结果验证**
   - 检查置信度评分
   - 审查推理链的逻辑
   - 结合人工判断做最终决策

## 集成到现有工作流

### CI/CD集成

```yaml
# GitHub Actions 示例
- name: Advanced Code Analysis
  run: |
    python run_experiment_advanced.py > analysis_report.txt
    # 处理分析结果
```

### IDE插件集成

```python
# VS Code 插件示例
async def analyze_current_file():
    current_file = get_current_file()
    issue_description = get_user_input("描述问题:")
    
    adapter = EnhancedGraphAdapter()
    result = await adapter.analyze(
        issue_text=issue_description,
        target_files=[current_file]
    )
    
    show_analysis_results(result)
```

## 总结

高级代码分析系统提供了强大的LLM驱动代码分析能力，通过智能分类、上下文增强、模式学习和多轮推理，显著提升了代码问题诊断的准确性和效率。

关键优势：
- 🎯 高准确性的问题定位
- 🚀 快速的分析响应
- 🔍 详细的推理过程
- 📈 持续的学习改进
- 🛠️ 灵活的配置选项

立即开始使用：
```bash
python run_demo_mode.py  # 体验功能
python run_quick_test.py # 快速测试
```