# 高级代码分析系统实验使用指南

本文档介绍如何使用新的高级代码分析系统进行实验和测试。

## 前置要求

### API配置

在运行实验之前，需要配置LLM API密钥：

1. **复制环境变量模板**
   ```bash
   cp .env.example .env
   ```

2. **配置OpenAI API密钥**
   ```bash
   # 编辑 .env 文件
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **或者设置环境变量**
   ```bash
   export OPENAI_API_KEY=your_actual_api_key_here
   ```

### 依赖安装

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 可用的实验脚本

### 1. `run_quick_test.py` - 快速测试脚本

最简单的测试方式，用于验证系统是否正常工作。

```bash
# 基本测试
python run_quick_test.py

# 测试所有可用策略
python run_quick_test.py --all-strategies
```

**功能：**
- 快速验证系统初始化
- 测试基本分析功能
- 显示系统状态和可用策略
- 比较不同分析策略的性能

### 2. `run_experiment_advanced.py` - 完整实验脚本

全面的实验脚本，提供详细的分析比较。

```bash
python run_experiment_advanced.py
```

**功能：**
- 运行多种分析策略（AUTO_SELECT, ADVANCED_ONLY, INTEGRATED）
- 与传统Enhanced GraphManager进行对比
- 处理多个实验案例
- 提供详细的性能指标和结果分析
- 显示置信度评分和推理链

### 3. `run_experiment_enhanced.py` - Enhanced GraphManager实验

专门测试Enhanced GraphManager功能的脚本。

```bash
python run_experiment_enhanced.py
```

### 4. `demo_integration.py` - 系统集成演示

展示系统集成和配置管理功能。

```bash
python demo_integration.py
```

## 实验数据结构

系统支持以下实验数据结构：

```
experiment_data/
├── issue.txt              # 默认问题描述
├── source_code.py         # 默认源代码
├── Answer.txt             # 期望答案（可选）
├── case1/                 # 实验案例1
│   ├── issue.txt
│   ├── source_code.py
│   └── Answer.txt
├── case2/                 # 实验案例2
│   ├── issue.txt
│   └── source_code.py
└── case3/                 # 实验案例3
    ├── issue.txt
    ├── source_code.py
    └── Answer.txt
```

## 分析策略说明

### AUTO_SELECT（自动选择）
- 系统根据问题类型自动选择最适合的分析策略
- 推荐用于一般用途

### ADVANCED_ONLY（仅高级分析）
- 使用LLM驱动的语义理解系统
- 适合复杂的逻辑错误和语义问题
- 提供详细的推理链和置信度评分

### GRAPH_ONLY（仅图分析）
- 使用Enhanced GraphManager进行结构化分析
- 适合架构问题和需求合规性检查

### INTEGRATED（集成分析）
- 结合LLM分析和图分析的优势
- 提供最全面的分析结果
- 需要两个系统都可用

## 输出结果解读

### 高级分析结果

```
📋 Advanced Analysis Results:
   Strategy Used: auto_select
   Success: ✅
   Processing Time: 2.45s
   Confidence Score: 0.85

🔍 Primary Findings (3):
   1. 发现赋值错误：变量x被赋值为常量而非变量
   2. 函数调用参数类型不匹配
   3. 缺少错误处理机制

💡 Recommendations (2):
   1. 修改第15行的赋值语句
   2. 添加try-catch错误处理

🧠 Advanced LLM Analysis Available:
   Bug Type: logic_error
   Reasoning Steps: 4
      1. 识别问题类型为逻辑错误
      2. 分析代码上下文
      3. 定位具体错误位置
      4. 生成修复建议
```

### 传统分析结果

```
📋 Traditional Analysis Results:
   Processing Time: 1.23s
   ✅ No conflicts detected

📊 Graph Statistics:
   Total nodes: 45
   Total edges: 67
   Node types: {'function': 12, 'class': 3, 'variable': 30}

⚠️  Violation Analysis:
   Total violations: 2
   Satisfies requirements: 8
```

## 性能指标

系统提供以下性能指标：

- **处理时间**：分析完成所需时间
- **置信度评分**：分析结果的可信度（0-1）
- **成功率**：分析成功完成的比例
- **内存使用**：分析过程中的内存消耗
- **Token使用**：LLM API调用的token消耗

## 配置选项

### 分析选项（AnalysisOptions）

```python
options = AnalysisOptions(
    strategy=AnalysisStrategy.AUTO_SELECT,  # 分析策略
    confidence_threshold=0.6,               # 置信度阈值
    include_requirements=True,              # 包含需求分析
    debug_mode=True,                        # 调试模式
    max_context_tokens=8000,               # 最大上下文token数
    max_concurrent_requests=3               # 最大并发请求数
)
```

### 系统配置

可以通过配置文件自定义系统行为：

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "max_tokens": 4000,
    "temperature": 0.1
  },
  "analysis": {
    "confidence_threshold": 0.6,
    "max_context_tokens": 8000,
    "debug_mode": false
  },
  "integration": {
    "enable_graph_context_enhancement": true,
    "fallback_confidence_threshold": 0.4
  }
}
```

## 故障排除

### 常见问题

1. **系统初始化失败**
   ```
   ❌ Advanced analysis system not available
   ```
   - 检查依赖是否正确安装
   - 确认LLM API配置正确

2. **分析失败**
   ```
   ❌ Analysis failed: API rate limit exceeded
   ```
   - 检查API配额和速率限制
   - 尝试降低并发请求数

3. **置信度过低**
   ```
   ⚠️  Low confidence score: 0.3
   ```
   - 尝试提供更详细的问题描述
   - 增加代码上下文信息
   - 使用INTEGRATED策略

### 调试技巧

1. **启用调试模式**
   ```python
   options = AnalysisOptions(debug_mode=True)
   ```

2. **查看详细日志**
   ```bash
   export LOG_LEVEL=DEBUG
   python run_quick_test.py
   ```

3. **检查系统状态**
   ```python
   adapter = EnhancedGraphAdapter()
   status = adapter.get_system_status()
   print(status)
   ```

## 最佳实践

1. **选择合适的策略**
   - 逻辑错误：使用ADVANCED_ONLY或AUTO_SELECT
   - 架构问题：使用GRAPH_ONLY
   - 复杂问题：使用INTEGRATED

2. **优化性能**
   - 合理设置max_context_tokens
   - 使用缓存避免重复分析
   - 并行处理多个文件

3. **提高准确性**
   - 提供清晰的问题描述
   - 包含相关的代码上下文
   - 设置合适的置信度阈值

## 示例用法

### 基本使用

```python
import asyncio
from src.enhanced_graph_adapter import EnhancedGraphAdapter, AnalysisStrategy, AnalysisOptions

async def analyze_code():
    adapter = EnhancedGraphAdapter()
    
    options = AnalysisOptions(
        strategy=AnalysisStrategy.AUTO_SELECT,
        confidence_threshold=0.6
    )
    
    result = await adapter.analyze(
        issue_text="函数中存在逻辑错误",
        target_files=["my_code.py"],
        options=options
    )
    
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence_score}")
    for finding in result.primary_findings:
        print(f"Finding: {finding}")

asyncio.run(analyze_code())
```

### 批量分析

```python
async def batch_analyze():
    adapter = EnhancedGraphAdapter()
    
    test_cases = [
        ("case1", "逻辑错误问题", ["file1.py"]),
        ("case2", "API使用错误", ["file2.py"]),
        ("case3", "性能问题", ["file3.py"])
    ]
    
    results = []
    for case_name, issue, files in test_cases:
        result = await adapter.analyze(
            issue_text=issue,
            target_files=files,
            options=AnalysisOptions(strategy=AnalysisStrategy.AUTO_SELECT)
        )
        results.append((case_name, result))
    
    # 分析结果
    for case_name, result in results:
        print(f"{case_name}: {result.confidence_score:.2f}")

asyncio.run(batch_analyze())
```

## 扩展和定制

系统支持以下扩展方式：

1. **自定义分析策略**
2. **添加新的bug模式**
3. **集成其他LLM提供商**
4. **自定义输出格式**

详细的扩展指南请参考开发文档。