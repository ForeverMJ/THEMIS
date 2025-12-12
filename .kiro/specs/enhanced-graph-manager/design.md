# 增强型GraphManager设计文档

## 概述

增强型GraphManager是一个复杂的代码分析系统，它通过四个核心功能模块来实现精确的代码理解和需求验证：

1. **结构提取 (Structural Extraction)** - 使用Python AST进行严格的代码结构分析
2. **语义注入 (Semantic Injection)** - 将需求分解并映射到相关代码组件
3. **依赖跟踪 (Dependency Tracing)** - 显式跟踪定义-使用关系链
4. **违规标记 (Violation Flagging)** - 自动检测潜在的需求违规

该系统的核心优势在于能够提供比纯LLM方法更精确的代码分析，特别是在调试复杂代码问题时能够准确追踪变量和函数的依赖关系。

## 架构

系统采用分层架构设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphManager API Layer                   │
├─────────────────────────────────────────────────────────────┤
│  Violation Flagging  │  Semantic Injection  │  Query Engine │
├─────────────────────────────────────────────────────────────┤
│           Dependency Tracing Engine                         │
├─────────────────────────────────────────────────────────────┤
│           Structural Extraction Engine                      │
├─────────────────────────────────────────────────────────────┤
│                NetworkX Knowledge Graph                     │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

- **StructuralExtractor**: 基于AST的代码结构分析器
- **SemanticInjector**: LLM驱动的需求分析和映射器
- **DependencyTracer**: 定义-使用关系跟踪器
- **ViolationFlagger**: 需求违规检测器
- **GraphQueryEngine**: 图查询和分析引擎

## 组件和接口

### StructuralExtractor

负责从Python代码中提取精确的结构信息：

```python
class StructuralExtractor:
    def extract_functions(self, ast_tree: ast.AST) -> List[FunctionNode]
    def extract_classes(self, ast_tree: ast.AST) -> List[ClassNode]
    def extract_variables(self, ast_tree: ast.AST) -> List[VariableNode]
    def extract_call_edges(self, ast_tree: ast.AST) -> List[CallEdge]
    def extract_instantiation_edges(self, ast_tree: ast.AST) -> List[InstantiationEdge]
```

### SemanticInjector

处理需求分解和代码映射：

```python
class SemanticInjector:
    def decompose_requirements(self, issue_text: str) -> List[RequirementNode]
    def map_requirements_to_code(self, requirements: List[RequirementNode], 
                                code_nodes: List[CodeNode]) -> List[MappingEdge]
    def analyze_requirement_relevance(self, requirement: RequirementNode, 
                                    code_node: CodeNode) -> float
```

### DependencyTracer

跟踪代码中的依赖关系：

```python
class DependencyTracer:
    def trace_variable_dependencies(self, ast_tree: ast.AST) -> List[DependencyEdge]
    def find_definition_usage_chains(self, variable_name: str) -> List[DefUseChain]
    def resolve_self_references(self, class_context: ClassNode) -> List[SelfReferenceEdge]
    def build_transitive_dependencies(self) -> Dict[str, Set[str]]
```

### ViolationFlagger

检测需求违规：

```python
class ViolationFlagger:
    def analyze_requirement_satisfaction(self, requirement: RequirementNode, 
                                       code_nodes: List[CodeNode]) -> ViolationReport
    def flag_potential_violations(self, graph: nx.DiGraph) -> List[ViolationFlag]
    def prioritize_violations(self, violations: List[ViolationFlag]) -> List[ViolationFlag]
```

## 数据模型

### 节点类型

```python
@dataclass
class FunctionNode:
    name: str
    args: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    line_number: int

@dataclass
class ClassNode:
    name: str
    bases: List[str]
    methods: List[str]
    docstring: Optional[str]
    line_number: int

@dataclass
class VariableNode:
    name: str
    var_type: Optional[str]
    defined_in: str
    line_number: int

@dataclass
class RequirementNode:
    id: str
    text: str
    priority: int
    testable: bool
```

### 边类型

```python
@dataclass
class CallEdge:
    caller: str
    callee: str
    line_number: int

@dataclass
class DependencyEdge:
    source: str
    target: str
    dependency_type: str  # DEPENDS_ON, USES_VAR, DEFINED_IN
    context: str

@dataclass
class ViolationEdge:
    requirement: str
    code_node: str
    status: str  # SATISFIES, VIOLATES
    reason: str
    confidence: float
```

## 正确性属性

*属性是应该在系统的所有有效执行中保持为真的特征或行为——本质上是关于系统应该做什么的正式陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 属性 1: 函数提取完整性
*对于任何* 包含函数定义的Python代码，结构提取应该为每个函数创建一个FunctionNode，包含正确的名称、参数和返回类型信息
**验证: 需求 1.1**

### 属性 2: 类提取完整性  
*对于任何* 包含类定义的Python代码，结构提取应该为每个类创建一个ClassNode，包含正确的继承关系
**验证: 需求 1.2**

### 属性 3: 成员变量提取完整性
*对于任何* 包含self.xxx定义的类，结构提取应该为每个成员变量创建一个VariableNode
**验证: 需求 1.3**

### 属性 4: 调用关系正确性
*对于任何* 包含函数调用的代码，系统应该在调用者和被调用者之间创建正确的CALLS边
**验证: 需求 1.4**

### 属性 5: 实例化关系正确性
*对于任何* 包含类实例化的代码，系统应该创建正确的INSTANTIATES边
**验证: 需求 1.5**

### 属性 6: 变量定义关系正确性
*对于任何* 变量定义，系统应该创建正确的DEFINED_IN边链接变量到其定义位置
**验证: 需求 1.6**

### 属性 7: 需求分解原子性
*对于任何* 复杂的问题文本，分解后的每个RequirementNode应该表示一个原子的、可测试的需求
**验证: 需求 2.1, 2.2**

### 属性 8: 需求映射相关性
*对于任何* 需求和代码组件，映射算法应该正确识别相关性并建立适当的连接
**验证: 需求 2.3, 2.4**

### 属性 9: 需求层次追溯性
*对于任何* 嵌套或复合需求，分解过程应该保持父需求和子组件之间的可追溯性
**验证: 需求 2.5**

### 属性 10: 依赖关系传递性
*对于任何* 变量依赖链A->B->C，系统应该正确维护传递性关系并支持双向遍历
**验证: 需求 3.1, 3.2, 3.3, 3.5**

### 属性 11: Self引用解析正确性
*对于任何* self.method()调用，系统应该正确解析并链接到对应的类成员定义
**验证: 需求 3.4**

### 属性 12: 违规检测准确性
*对于任何* 需求-代码对，违规检测应该准确分配SATISFIES或VIOLATES状态并提供有意义的原因
**验证: 需求 4.1, 4.2, 4.3**

### 属性 13: 违规优先级合理性
*对于任何* 多个违规的集合，优先级排序应该基于严重性和影响进行合理排序
**验证: 需求 4.4**

### 属性 14: 动态更新一致性
*对于任何* 代码更新，系统应该正确重新评估受影响需求的违规状态
**验证: 需求 4.5**

### 属性 15: API返回值正确性
*对于任何* API调用，返回的数据结构应该符合预期的类型和格式规范
**验证: 需求 5.1, 5.2, 5.3, 5.4**

### 属性 16: 序列化往返一致性
*对于任何* 图对象，序列化后再反序列化应该产生等价的图结构
**验证: 需求 5.5**

## 错误处理

### 解析错误处理
- 语法错误的Python代码应该产生有意义的错误消息
- 不完整的代码片段应该尽可能提取可用信息
- AST解析失败应该回退到基于文本的分析

### 依赖解析错误
- 无法解析的引用应该标记为未解析依赖
- 循环依赖应该被检测并适当处理
- 外部模块引用应该标记为外部依赖

### LLM集成错误
- LLM API失败应该有重试机制
- 无效的LLM响应应该回退到基于规则的分析
- 网络超时应该有适当的错误处理

## 测试策略

### 单元测试方法
系统将使用单元测试来验证特定的功能组件：
- AST解析器的正确性测试
- 图构建算法的边界情况测试
- API接口的输入验证测试
- 错误处理路径的测试

### 基于属性的测试方法
系统将使用Hypothesis库进行基于属性的测试，每个属性测试将运行至少100次迭代：
- 生成随机Python代码片段测试结构提取
- 生成随机需求文本测试语义注入
- 生成随机依赖关系测试跟踪算法
- 生成随机图结构测试序列化往返

每个基于属性的测试必须使用以下格式标记对应的设计文档属性：
'**Feature: enhanced-graph-manager, Property {number}: {property_text}**'

单元测试和属性测试是互补的：单元测试捕获具体的错误，属性测试验证通用的正确性。两种测试方法都必须包含在测试策略中，以提供全面的覆盖。