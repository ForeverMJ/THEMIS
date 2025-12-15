# 开始使用高级代码分析系统

## 🎯 选择适合你的使用方式

### 1. 🚀 快速体验（推荐新用户）

无需任何配置，立即体验系统功能：

```bash
python run_demo_mode.py
```

这将展示：
- 智能bug分类演示
- 代码上下文分析
- 模式匹配检测
- 多轮推理过程
- 完整的分析流程

### 2. 🔧 配置API密钥进行真实分析

如果你想使用真实的LLM分析功能：

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑 .env 文件，添加你的OpenAI API密钥
# OPENAI_API_KEY=your_actual_api_key_here

# 3. 运行快速测试
python run_quick_test.py
```

### 3. 📊 运行完整实验

体验系统的完整分析能力：

```bash
python run_experiment_advanced.py
```

这将：
- 测试多种分析策略
- 与传统方法进行对比
- 处理多个实验案例
- 提供详细的性能指标

## 🎮 交互式使用

### 基本API使用

```python
import asyncio
from src.enhanced_graph_adapter import EnhancedGraphAdapter, AnalysisOptions, AnalysisStrategy

async def my_analysis():
    # 初始化系统
    adapter = EnhancedGraphAdapter()
    
    # 分析代码问题
    result = await adapter.analyze(
        issue_text="我的代码中有一个逻辑错误",
        target_files=["my_file.py"],
        options=AnalysisOptions(
            strategy=AnalysisStrategy.AUTO_SELECT,
            confidence_threshold=0.6
        )
    )
    
    # 查看结果
    if result.success:
        print(f"置信度: {result.confidence_score:.2f}")
        for finding in result.primary_findings:
            print(f"发现: {finding}")
        for rec in result.recommendations:
            print(f"建议: {rec}")

asyncio.run(my_analysis())
```

## 📁 实验数据结构

系统可以处理以下格式的实验数据：

```
experiment_data/
├── issue.txt              # 问题描述
├── source_code.py         # 源代码
├── Answer.txt             # 期望答案（可选）
├── case1/                 # 具体案例
│   ├── issue.txt
│   ├── source_code.py
│   └── Answer.txt
└── case2/
    ├── issue.txt
    └── source_code.py
```

## 🔍 分析策略选择指南

| 问题类型 | 推荐策略 | 说明 |
|---------|---------|------|
| 逻辑错误 | `AUTO_SELECT` 或 `ADVANCED_ONLY` | LLM擅长理解复杂逻辑 |
| API使用错误 | `ADVANCED_ONLY` | 需要语义理解 |
| 架构问题 | `GRAPH_ONLY` | 结构分析更有效 |
| 需求合规性 | `GRAPH_ONLY` | 专门的需求映射 |
| 复杂综合问题 | `INTEGRATED` | 结合两种方法的优势 |
| 不确定 | `AUTO_SELECT` | 让系统自动选择 |

## 📊 结果解读

### 置信度评分
- **0.8-1.0**: 🟢 高置信度 - 可以直接采用
- **0.6-0.8**: 🟡 中等置信度 - 建议人工审查
- **0.4-0.6**: 🟠 低置信度 - 需要更多信息
- **0.0-0.4**: 🔴 很低置信度 - 可能需要重新分析

### 典型输出示例

```
📋 Analysis Results:
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
```

## 🛠️ 故障排除

### 常见问题

1. **系统初始化失败**
   ```
   ❌ Advanced analysis system not available
   ```
   **解决**: 检查依赖安装 `pip install -r requirements.txt`

2. **API密钥错误**
   ```
   Error: The api_key client option must be set
   ```
   **解决**: 检查 `.env` 文件中的 `OPENAI_API_KEY` 设置

3. **分析置信度过低**
   ```
   Low confidence score: 0.3
   ```
   **解决**: 
   - 提供更详细的问题描述
   - 尝试使用 `INTEGRATED` 策略
   - 增加代码上下文信息

### 调试技巧

```python
# 启用调试模式
options = AnalysisOptions(debug_mode=True)

# 检查系统状态
adapter = EnhancedGraphAdapter()
status = adapter.get_system_status()
print(f"可用系统: {status['systems_initialized']}")
```

## 🎯 使用场景

### 1. 代码审查
```bash
# 分析特定文件的问题
python -c "
import asyncio
from src.enhanced_graph_adapter import *

async def review():
    adapter = EnhancedGraphAdapter()
    result = await adapter.analyze(
        issue_text='请审查这个文件中的潜在问题',
        target_files=['your_file.py']
    )
    print('审查结果:', result.primary_findings)

asyncio.run(review())
"
```

### 2. Bug诊断
```bash
# 使用高级分析诊断复杂bug
python run_quick_test.py
```

### 3. 批量分析
```bash
# 分析多个实验案例
python run_experiment_advanced.py
```

### 4. 集成到工作流
```python
# 在你的代码中集成分析功能
from src.enhanced_graph_adapter import EnhancedGraphAdapter

def analyze_code_issue(issue_description, code_files):
    adapter = EnhancedGraphAdapter()
    # 异步分析逻辑
    pass
```

## 📚 进一步学习

- 📖 [完整使用指南](ADVANCED_ANALYSIS_GUIDE.md)
- 🧪 [实验详细说明](EXPERIMENT_USAGE.md)
- 🔧 [系统集成文档](INTEGRATION_SUMMARY.md)
- 📋 [需求和设计文档](.kiro/specs/advanced-code-analysis/)

## 💡 最佳实践

1. **从演示模式开始** - 先运行 `run_demo_mode.py` 了解系统能力
2. **逐步配置** - 先用基本配置，再根据需要调整
3. **选择合适策略** - 根据问题类型选择最适合的分析策略
4. **关注置信度** - 低置信度结果需要人工验证
5. **提供详细描述** - 越详细的问题描述，分析结果越准确

## 🚀 开始你的第一次分析

```bash
# 1. 体验系统功能
python run_demo_mode.py

# 2. 如果满意，配置API密钥
cp .env.example .env
# 编辑 .env 文件

# 3. 运行真实分析
python run_quick_test.py

# 4. 探索更多功能
python run_experiment_advanced.py
```

现在你已经准备好使用这个强大的代码分析系统了！🎉