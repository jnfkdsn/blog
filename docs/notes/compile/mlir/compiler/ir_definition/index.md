---
order: 20
title: 定义 IR 抽象
updated: 2026-09-14
excludeFromSidebar: true
---

# 定义 IR 抽象

[IR 变换基础](../transforms/)解释怎样处理已有操作。本模块转向定义操作及它们使用的抽象：一项计算表达什么、保存哪些信息、哪些形式合法，以及哪些事实可以交给通用基础设施使用。

## 用区间限制串起五章

先把“将一个有符号整数限制在闭区间内”定义为 `lab.clamp`：输入是 SSA operand，上下界是静态属性，结果是 i32。ODS 生成访问器与部分验证，手写 verifier 补上跨字段关系，工具由此能构造、解析和打印它，专用 Pattern 再把它展开成 arith。

但通用优化并不认识每个新操作。第二章从删除一个未使用的 clamp 开始，观察效果信息怎样影响决定，再用范围接口让同一消费者查询不同操作。第三章把区间做成自己的 Attribute 和 Type，得到带范围保证的结果。第四章把这项计算放入 Region，规定输入绑定、结果传出、隔离与验证顺序。第五章单独追踪文本与对象之间的往返。

```text
约定 clamp 语义
  → 定义 Op 字段、ODS、构造与验证
  → 通过 Trait / Interface 向消费者提供事实
  → 用 Type / Attribute 表达领域契约
  → 定义含 Region 的操作及其传值、验证协议
  → 解析文本、构造状态、验证并打印回文本
  → 后续：把这些操作与类型转换为目标认可的表示
```

这条链的终点是能够解释并修改一个小型 IR 抽象的定义。它还不是完整可执行编译器：ODS 不会自动实现目标代码，接口也不会自动证明语义声明正确。

## 阅读顺序

| 顺序 | 章节 | 沿主例理解的过程 |
|---|---|---|
| 1 | [一个操作是怎样被定义出来的](./op_definition) | 语义 → 字段 → ODS / C++ → verifier → 注册与打印 → 专用 Pass 展开 |
| 2 | [Trait 与 Interface：让通用代码理解操作](./traits_interfaces) | 无用户操作 → 删除判断 → 效果事实 → 自定义范围协议 → 直接与外部模型 |
| 3 | [自定义 Type 与 Attribute：把领域信息放进 IR](./types_attributes) | 操作要求与结果保证 → 参数化定义 → 两层验证 → uniquing → 类型相等与转换需求 |
| 4 | [定义带 Region 的操作：传值、隔离与验证](./regions_assembly) | 输入/参数/yield/结果 → variadic → 分阶段验证与隔离 |
| 5 | [解析与打印：在文本和 IR 对象之间往返](./assembly_format) | 文本字段 → OperationState → 验证 → printer → 往返中的引用与属性 |

五章的核心正文与配套编译器侧示例已经准备完成。学习者已完成第一章阅读，后四章仍按顺序阅读、讨论，再安排相应修改练习；作者编译通过不等于学习者已独立实现。

一次准备整个章节组是合适的：本组概念共用一个案例，提前写完便于检查是否遗漏，也让前后引用稳定。阅读时不必一次消化全部内容。可以先学第 2 章，能解释“同样无人使用，为什么只删除一个”后，再进入类型与区域。

## 把各层责任放回同一操作

假设 `lesson.limit` 输出带范围的值，且位于 `lesson.scope` 内，检查工作可以按下表定位：

| 遇到的情况 | 主要由哪层处理 |
|---|---|
| 范围下界大于上界 | Type / Attribute 的参数验证 |
| 输入不是 i32，或结果不是 RangeType | ODS 生成的字段约束 |
| 结果范围与 bounds 属性不一致 | LimitOp 的手写 verifier |
| 工具想查询结果范围 | StaticBounds Interface 的模型与消费端 |
| 区域入口或出口数量、类型不对应 | ScopeOp 的普通/区域 verifier |
| body 隐式使用外部 SSA 值 | 隔离与 SSA 验证 |
| 文本关键字错误，或打印漏掉字段 | parser / printer 及往返测试 |
| 自定义类型怎样变成 i32，使用者如何衔接 | 后续 Dialect Conversion 与具体转换规则 |

定义与变换由此相接：定义确立有效表示及语义契约，变换在保持这些含义的前提下改变表示。前面学过的 Builder、PatternRewriter、Pass 和测试在这里继续使用。

## 核心完成范围与继续深入的位置

本模块当前主干覆盖 Op 定义、Trait/Op Interface、自定义 Type/Attr、参数与对象验证、Region/单组可变参数、注册、声明式和手写打印解析，并提供成功与失败的可运行证据。以下范围继续保留，不把它们混写成已经实现：

| 进阶范围 | 当前深度 | 后续衔接 |
|---|---|---|
| Type / Attribute Interface、类型推导接口 | 已解释协议位置；本工程实现的是 Op Interface | 遇到领域类型公共查询、形状推导时补实现 |
| 多组 variadic/optional 的分段存储 | 已解释布局歧义及 trait 选择；实现仅含单组 | 结合真实多组参数操作验证 accessor 和更新 |
| 完整 RegionBranch 等区域协议 | 已解释通用消费者为何需要；尚未实现 | 区域分析、控制流转换 |
| 自定义存储分配、可变存储、复杂语法 | 已说明所有权与维护边界 | 用实际参数/语法需求展开深入篇 |
| bytecode、方言版本、兼容迁移 | 未展开 | 工具与序列化专题 |

精确范围继续由[覆盖表 M02—M03、X03](../../coverage)维护。完成本模块的核心阅读后，适合进入 **Dialect Conversion**：以“自定义类型变成目标类型”为具体问题，处理合法性、操作替换及函数/区域边界。

## 配套工程

第一章使用 `aicompiler-labs/llvm-mlir/05-op-definition/`。后四章共用 `06-ir-definition/`，复用前一工程的 Lab dialect，并新增 Lesson dialect；原实验保持独立。

`06-ir-definition/observe.py` 可按 interfaces、types、regions、assembly 分组展示输入、查询结果、清理后的 IR 和打印往返。学习方法仍是先预测一个变化，再观察并解释；`docs/validate_ir_definition.py` 则用于维护回归。完整命令见工程 README。

多个机制组成的正式小 Pass 工程仍放在[贯通教程](../../tutorials/first_pass)，本目录保存定义机制。阶段状态见[学习路径](../../learning_path)。
