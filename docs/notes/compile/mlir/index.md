---
order: 2
title: MLIR 技术文档
updated: 2026-09-13
excludeFromSidebar: true
tags: [mlir, compiler, learning]
status: draft
---

# MLIR 技术文档

这套文档面向 AI Compiler 的阅读、实现与调试能力：从 IR 的语义与约束出发，逐步进入改写、分析、渐进 lowering、内存与目标后端。知识按所属机制归档；学习顺序由前置依赖决定。整个 workspace 的长期安排仍以 [AI Compiler 学习路线](/notes/compile/ai-compiler/stack_and_roadmap)为准。

**开始阅读用[学习路径](./learning_path)，检查知识范围用[覆盖表](./coverage)，查机制用下方分类。** 覆盖表是计划与缺口清单，不等同于已经写完或已经掌握的目录。

## 分类与职责

| 分类 | 保存的内容 | 当前入口 |
|---|---|---|
| `core/` 核心概念 | 跨方言成立的对象模型、SSA、类型、符号、作用域与效果模型 | [核心概念](./core/) |
| `dialects/` 方言语义 | 各方言的抽象边界、操作契约、执行规则与合法性约束 | [方言语义](./dialects/) |
| `compiler/` 编译器机制 | 构造 IR、定义操作、改写、Pass、分析、转换、bufferization 与 lowering | [机制地图](./compiler/) → [IR 变换基础](./compiler/transforms/) / [定义 IR 抽象](./compiler/ir_definition/) |
| `guides/` 使用指南 | 阅读 IR、检查诊断、查源码、使用工具的可复用方法 | [使用指南](./guides/) |
| `tutorials/` 贯通教程 | 用一段完整程序连接多个机制，解释推理过程 | [贯通教程](./tutorials/) |

分类借鉴官方 [Code Documentation](https://mlir.llvm.org/docs/) 中 Language Reference、Dialects、编译器基础设施、Tools 与 Tutorials 的区分，再按本项目的依赖关系组织。它不是官网目录的逐页翻译。

同一机制只有一个主要定义位置。例如 SSA 在 `core/values_ssa.md` 定义，SCF 章节说明循环怎样使用 SSA，贯通教程再展示它们如何共同表达算法。阅读中的疑问用于改进相应章节，不再默认新增一篇独立问答。

## 当前阶段已展开

阶段 A“认识基础 IR + 最小实验”的基础目标已完成。阶段 B 原理与小 Pass 工程已完成理解，[Pass 与 pipeline](./compiler/transforms/passes)已读完，[C++ IR API](./compiler/transforms/ir_api)的核心问题已讨论并提供观察实验；[PatternRewriter](./compiler/transforms/rewriting)重写后的主线已获肯定，配套[改写驱动](./compiler/transforms/rewrite_drivers)供深入学习。[小 Pass 教程](./tutorials/first_pass)的独立修改尚待验证。现在进入[操作定义](./compiler/ir_definition/op_definition)，配套 ODS/C++、builder、verifier 与打印/展开示例已验证，待阅读。

目前已有正文覆盖：

- 核心概念：对象模型与表示、Value/SSA/支配、Type/Attribute/Properties、符号与作用域、内存效果与推测执行边界。
- 基础方言：builtin、func、arith、cf，以及 SCF 的 if、for、while 和区域传值协议。
- 贯通教程：从一个“带条件的循环累加”算法出发，连接结构、数据流、控制流与 SCF-to-CF；小 Pass 工程连接规则、注册、pipeline 与 lit/FileCheck。
- 使用指南：自定义/通用打印、parser/verifier、IR 变换、源码定位与证据分级。
- 编译器机制：Pass 的运行对象、CSE/canonicalize、pipeline 顺序与嵌套、日志、分析有效性、失败及对应源码路径。
- IR 修改机制：C++ 对象与生命周期、use-list、RAUW、Builder/插入点、删除与遍历、函数构造及 IRMapping 克隆，配套独立 C++ 程序与失败反例。
- 操作定义：以 `lab.clamp` 连接 ODS、生成的 C++ API、静态属性、builder、生成/手写 verifier、注册、打印往返及 Pass 展开。
- Pattern 改写机制：规则与 driver 分工、修改通知、原地更新、worklist、Walk/Greedy 行为对照、benefit、fold/canonicalize 与收敛边界，配套真实 IR 和规则成功轨迹。

Tensor 与 MemRef 在当前阶段先解释值语义、可变存储和别名的区别；完整操作族、布局、DPS、bufferization 在后续阶段展开。当前章节的具体边界都列在[覆盖表](./coverage)中。

## 广度与深度怎样维护

覆盖表用于检查内容是否遗漏；正文按理解过程组织，不机械地逐项列术语。后续章节以一个持续演进的例子连接现象、原理和实现，关键结论展开推导及中间状态，源码围绕具体论点解释。版本和复现日志集中放在文末或实验目录，减少阅读中断。

“深入”随阶段递进：基础阶段认识并解释小程序；阶段 B 建立 Pass 运行模型、C++ 对象修改与实现测试的基础；当前阶段 C 定义操作并逐步进入接口与转换；再往后追踪分析和优化的正确性与性能。扩展专题不自动成为上一阶段新增的完成门槛。

模块 `index.md` 负责定位和导航。机制正文承载完整解释；贯通教程承载跨章节推演。正式实践在阅读和讨论之后安排到 workspace 的 `aicompiler-labs/llvm-mlir/`，生成物进入 `artifacts/`。

## 版本与验证

基础版本为 LLVM `llvmorg-20.1.8`，commit `87f0227cb60147a26a1eeb4fb06e3b505e9c7261`。网页可能对应更新的 MLIR；本系列的语法和行为优先与固定版本源码核对。

示例按“完整合法模块、预期失败模块、局部片段、概念伪代码”区分。验证方法、复现命令和结果范围见[阅读与验证 IR](./guides/inspecting_ir)。验证通过表示相应检查通过；读者掌握程度另记，不根据文档生成状态推断。
