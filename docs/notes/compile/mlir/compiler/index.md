---
order: 30
title: 编译器机制
updated: 2026-09-06
excludeFromSidebar: true
---

# 编译器机制

本目录保存构建、分析和变换 MLIR 的机制。当前完成的是机制地图与章节边界，下面列出的实现章节尚未展开；当前阶段先读[核心概念](../core/)与[方言语义](../dialects/)。

## 各机制解决的不同问题

| 机制 | 负责什么 | 必须与什么配合 |
|---|---|---|
| IR API / Builder | 创建、遍历、克隆、替换和删除对象 | 维护 SSA、结构、作用域和生命周期 |
| ODS / 自定义 Type、Attr、Op | 声明抽象的数据结构、约束和语法，生成相关代码 | C++ verifier/解析打印与正反例测试 |
| Trait / Interface | 声明共性约束或提供可查询/调用的通用行为 | 通用变换通过契约理解不同操作 |
| fold / Pattern Rewrite | 表达局部简化或结构改写 | 匹配条件、rewriter 修改协议和 driver |
| Pass / PassManager | 调度变换与分析，定义运行范围和依赖 | 注册、嵌套、线程约束、分析失效 |
| Analysis / DataFlow | 推导支配、别名、常量/形状等事实 | 保守性、收敛、缓存与 invalidation |
| Dialect Conversion | 将操作/类型逐步转成目标认可的形式 | legality、TypeConverter、边界 materialization |
| Bufferization / Deallocation | 把 tensor 值语义落实到存储与生命周期 | 读写冲突、alias、所有权及函数边界 |
| Loop / Structured transforms | 调整计算组织与数据复用 | 依赖、效果、边界、数值与目标约束 |
| LLVM / Target lowering | 降低剩余表示，连接目标代码与运行时 | DataLayout、ABI、translation、JIT/AOT |

同一项变换可以以 Pattern 描述、由 Pass 执行、依赖某个 Analysis，并采用 Conversion 框架管理类型变化。这些概念不在同一个分类维度，不能把它们当作互斥的“几种 Pass”。

## 正文展开方式

先完成每个模块的前置、契约、必要知识项与验证标准，再编写该阶段正文。章节名称预计使用 `ir_api`、`op_definition`、`traits_interfaces`、`rewriting`、`passes`、`analyses`、`dialect_conversion`、`bufferization`、`loop_transformations`、`llvm_lowering`；需要时再拆子目录。

实现阶段会对一个小操作或变换给出闭合源码路径：规范/ODS → 实现 → 注册/运行 → 正反例测试。之后再连接多个机制构成转换链。先完整解释一条必要链路，再扩大操作覆盖，避免只有术语介绍却没有可检验的行为。

精确范围与目标深度见[覆盖项 M01—M10、A01—A08](../coverage)，阶段依赖见[学习路径](../learning_path)。正式代码保存在 labs，稳定原理与分析保存在本目录。

分类依据：[官方编译器基础设施文档入口](https://mlir.llvm.org/docs/)。API 与行为继续按本项目固定版本核对。
