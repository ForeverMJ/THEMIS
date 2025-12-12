# 增强型GraphManager需求文档

## 介绍

增强型GraphManager是一个复杂的代码分析系统，它结合了结构化代码提取和语义需求分析来创建综合知识图谱。该系统能够进行精确的依赖关系跟踪、需求验证和违规检测，特别专注于需要理解代码关系的调试场景。

## 术语表

- **GraphManager**: 构建和管理结合代码结构与需求的知识图谱的主系统
- **结构提取**: 基于AST的解析，提取代码的事实结构而不进行解释
- **语义注入**: 使用LLM分析将需求映射到相关代码组件的过程
- **依赖跟踪**: 显式跟踪代码中定义-使用关系
- **违规标记**: 在正式判断之前自动检测潜在需求违规
- **FunctionNode**: 表示函数的图节点，包含名称、参数和返回类型信息
- **ClassNode**: 表示类的图节点，包含继承关系
- **VariableNode**: 表示类成员变量的图节点，特别是self.xxx定义
- **RequirementNode**: 表示从问题描述中分解出的需求的图节点
- **CALLS**: 表示函数A调用函数B的边类型
- **INSTANTIATES**: 表示类A创建类B实例的边类型
- **DEFINED_IN**: 表示变量A在方法/函数B中定义的边类型
- **DEPENDS_ON**: 表示使用依赖于定义的边类型
- **USES_VAR**: 表示代码使用特定变量的边类型
- **SATISFIES**: 表示需求被代码满足的边属性
- **VIOLATES**: 表示需求被代码违反的边属性

## 需求

### 需求 1

**用户故事:** 作为调试复杂代码问题的开发者，我希望系统能够从代码中提取精确的结构信息，以便我能理解代码组件之间的事实关系。

#### 验收标准

1. WHEN 解析Python代码 THEN 系统 SHALL 提取包含函数名、参数列表和返回类型注解的FunctionNode条目
2. WHEN 解析Python代码 THEN 系统 SHALL 提取包含类名和继承关系的ClassNode条目
3. WHEN 解析类定义 THEN 系统 SHALL 为所有self.xxx成员变量定义提取VariableNode条目
4. WHEN 分析函数调用 THEN 系统 SHALL 在调用函数和被调用函数之间创建CALLS边
5. WHEN 分析类实例化 THEN 系统 SHALL 在实例化类和被实例化类之间创建INSTANTIATES边
6. WHEN 分析变量定义 THEN 系统 SHALL 创建DEFINED_IN边将变量链接到其定义位置

### 需求 2

**用户故事:** 作为处理复杂需求的开发者，我希望系统能够将问题描述分解为原子需求组件，以便我能将特定需求映射到相关代码段。

#### 验收标准

1. WHEN 处理问题文本 THEN 系统 SHALL 将其分解为表示原子需求的单独RequirementNode条目
2. WHEN 创建RequirementNode条目 THEN 系统 SHALL 确保每个节点表示单一的、可测试的需求
3. WHEN 分析需求 THEN 系统 SHALL 识别可能与每个需求相关的代码组件
4. WHEN 将需求映射到代码 THEN 系统 SHALL 将RequirementNode条目定位在图中相关代码节点附近
5. WHEN 处理嵌套或复合需求 THEN 系统 SHALL 维护父需求和分解组件之间的可追溯性

### 需求 3

**用户故事:** 作为分析代码依赖关系的开发者，我希望系统能够显式跟踪定义-使用链，以便我能追踪变量和函数在整个代码库中的连接方式。

#### 验收标准

1. WHEN 分析变量使用 THEN 系统 SHALL 创建DEPENDS_ON边将使用位置链接到定义位置
2. WHEN 分析变量引用 THEN 系统 SHALL 创建USES_VAR边指示哪些代码段使用特定变量
3. WHEN 跟踪依赖关系 THEN 系统 SHALL 维护定义和使用之间的双向遍历能力
4. WHEN 分析self上的方法调用 THEN 系统 SHALL 正确链接到类成员定义
5. WHEN 处理复杂依赖链 THEN 系统 SHALL 为多步依赖保持传递性关系

### 需求 4

**用户故事:** 作为验证代码正确性的开发者，我希望系统能够自动标记潜在的需求违规，以便我能在正式代码审查之前识别问题。

#### 验收标准

1. WHEN 分析需求-代码关系 THEN 系统 SHALL 为需求-代码边分配SATISFIES或VIOLATES状态
2. WHEN 检测潜在违规 THEN 系统 SHALL 在边属性中记录违规的具体原因
3. WHEN 标记违规 THEN 系统 SHALL 提供关于需要纠正什么的可操作信息
4. WHEN 处理多个需求 THEN 系统 SHALL 按严重性和影响优先排序违规
5. WHEN 更新代码 THEN 系统 SHALL 重新评估受影响需求的违规状态

### 需求 5

**用户故事:** 作为以编程方式使用GraphManager的开发者，我希望有全面的图构建和分析API，以便我能将系统集成到自动化工作流中。

#### 验收标准

1. WHEN 调用结构提取API THEN 系统 SHALL 返回具有正确类型节点和边的NetworkX DiGraph对象
2. WHEN 调用语义注入API THEN 系统 SHALL 接受需求文本并返回带有需求节点的增强图
3. WHEN 调用依赖跟踪API THEN 系统 SHALL 提供查询定义-使用关系的方法
4. WHEN 调用违规检测API THEN 系统 SHALL 返回包含原因和位置的结构化违规报告
5. WHEN 与外部系统集成 THEN 系统 SHALL 提供图持久化和交换的序列化方法