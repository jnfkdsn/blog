---
order: 2
title: MLIR 技术文档
updated: 2026-09-06
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
| `compiler/` 编译器机制 | 构造 IR、定义操作、改写、Pass、分析、转换、bufferization 与 lowering | [机制地图](./compiler/)；正文按阶段展开 |
| `guides/` 使用指南 | 阅读 IR、检查诊断、查源码、使用工具的可复用方法 | [使用指南](./guides/) |
| `tutorials/` 贯通教程 | 用一段完整程序连接多个机制，解释推理过程 | [贯通教程](./tutorials/) |

分类借鉴官方 [Code Documentation](https://mlir.llvm.org/docs/) 中 Language Reference、Dialects、编译器基础设施、Tools 与 Tutorials 的区分，再按本项目的依赖关系组织。它不是官网目录的逐页翻译。

同一机制只有一个主要定义位置。例如 SSA 在 `core/values_ssa.md` 定义，SCF 章节说明循环怎样使用 SSA，贯通教程再展示它们如何共同表达算法。阅读中的疑问用于改进相应章节，不再默认新增一篇独立问答。

## 当前阶段已展开

当前阶段是“读懂并解释 IR”。正文覆盖：

- 核心概念：对象模型与表示、Value/SSA/支配、Type/Attribute/Properties、符号与作用域、内存效果与推测执行边界。
- 基础方言：builtin、func、arith、cf，以及 SCF 的 if、for、while 和区域传值协议。
- 贯通教程：从一个“带条件的循环累加”算法出发，连接结构、数据流、控制流与 SCF-to-CF。
- 使用指南：自定义/通用打印、parser/verifier、IR 变换、源码定位与证据分级。

Tensor 与 MemRef 在当前阶段先解释值语义、可变存储和别名的区别；完整操作族、布局、DPS、bufferization 在后续阶段展开。当前章节的具体边界都列在[覆盖表](./coverage)中。

## 广度与深度怎样维护

每个模块先确定范围与前置知识，再列可检查的知识项。章节至少解释对象或操作契约、语义规则、完整例子、边界/反例、设计原因、源码入口和理解检查。只列术语或 API 名称不算展开。

“深入”也有阶段差异：当前需要能独立推演合法性和执行过程；实现阶段还要追到 ODS、C++ verifier、rewrite 与测试；优化阶段还要给出语义正确性和性能证据。首次读 IR 不以完整阅读 MLIR 源码为完成条件。

模块 `index.md` 负责定位和导航。机制正文承载完整解释；贯通教程承载跨章节推演。正式实践在阅读和讨论之后安排到 workspace 的 `aicompiler-labs/llvm-mlir/`，生成物进入 `artifacts/`。

## 版本与验证

基础版本为 LLVM `llvmorg-20.1.8`，commit `87f0227cb60147a26a1eeb4fb06e3b505e9c7261`。网页可能对应更新的 MLIR；本系列的语法和行为优先与固定版本源码核对。

示例按“完整合法模块、预期失败模块、局部片段、概念伪代码”区分。验证方法、复现命令和结果范围见[阅读与验证 IR](./guides/inspecting_ir)。验证通过表示相应检查通过；读者掌握程度另记，不根据文档生成状态推断。
