---
order: 30
title: 编译器机制
updated: 2026-09-13
excludeFromSidebar: true
---

# 编译器机制

本目录保存构建、分析和变换 MLIR 的机制。基础 IR 阶段已完成，当前进入“使用和实现变换”。

## 当前阅读入口

[Pass 与 pipeline：组织一次 IR 变换](./passes) 已展开：以冗余计算为主例，连接 CSE/canonicalize、运行对象、嵌套调度、日志、分析有效性、失败与源码。

[C++ IR API：从读取对象到安全地修改 IR](./ir_api) 已展开：继续使用 `twice`，解释对象与句柄、use-list、RAUW、创建与插入、删除、构造和克隆；配套 C++ 程序已编译运行，包含支配与隔离失败反例。

当前继续 [PatternRewriter：把一次 IR 修改写成可应用的规则](./rewriting)：沿一次加零消除，从底层修改推导匹配与改写，再连接 driver 调用、实际输出与修改通知。正文重写后的主线与可理解性已获读者肯定。

配套[改写驱动：规则如何协作与收敛](./rewrite_drivers)保留多规则协作、Walk/Greedy 对照、原地更新、benefit、fold/canonicalize 及收敛分析。先理解主章的完整过程，再按需深入；两篇共用已编译验证的 C++ 示例。

现在继续贯通教程[实现并测试一个小 Pass](../tutorials/first_pass)，把规则接入可构建、注册和测试的工程；正文与 `04-small-pass` 实现已验证，学习者独立修改尚待完成。以下机制地图展示长期范围，不表示所有正文已经完成。

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

后续编写参照工作区 `writing_method.md` 中已根据阅读反馈确认的方法。覆盖表用于检查广度，正文沿“已知现象 → 处理过程 → 机制与设计原因 → 实现与边界”展开。一个主要例子持续演进，关键结论展示中间过程；源码解释具体论点，日志和环境细节交给实验目录。先完整写好当前章节，再按依赖扩展后续正文。

章节名称预计使用 `ir_api`、`op_definition`、`traits_interfaces`、`rewriting`、`rewrite_drivers`、`passes`、`analyses`、`dialect_conversion`、`bufferization`、`loop_transformations`、`llvm_lowering`；需要时再拆子目录。

实现阶段会对一个小操作或变换给出闭合源码路径：规范/ODS → 实现 → 注册/运行 → 正反例测试。之后再连接多个机制构成转换链。先完整解释一条必要链路，再扩大操作覆盖，避免只有术语介绍却没有可检验的行为。

精确范围与目标深度见[覆盖项 M01—M10、A01—A08](../coverage)，阶段依赖见[学习路径](../learning_path)。正式代码保存在 labs，稳定原理与分析保存在本目录。

分类依据：[官方编译器基础设施文档入口](https://mlir.llvm.org/docs/)。API 与行为继续按本项目固定版本核对。
