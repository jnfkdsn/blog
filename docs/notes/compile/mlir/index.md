---
order: 2
title: MLIR 技术文档
updated: 2026-10-04
excludeFromSidebar: true
tags: [mlir, compiler, learning]
status: draft
---

# MLIR 技术文档

这套文档面向 AI Compiler 的阅读、实现与调试能力：从 IR 的语义与约束出发，逐步进入改写、分析、渐进 lowering、内存与目标后端。知识按所属机制归档；学习顺序由前置依赖决定。整个 workspace 的长期安排仍以 [AI Compiler 学习路线](/notes/compile/ai-compiler/stack_and_roadmap)为准。

**IR 定义之后按[后续阅读顺序](./chapter_plan)逐站推进；阶段标准与进度见[学习路径](./learning_path)，知识范围见[覆盖表](./coverage)，查机制用下方分类。** 覆盖表是计划与缺口清单，不等同于已经写完或已经掌握的目录。

## 分类与职责

| 分类 | 保存的内容 | 当前入口 |
|---|---|---|
| `core/` 核心概念 | 跨方言成立的对象模型、SSA、类型、符号、作用域与效果模型 | [核心概念](./core/) |
| `dialects/` 方言语义 | 各方言的抽象边界、操作契约、执行规则与合法性约束 | [方言语义](./dialects/) |
| `compiler/` 编译器机制 | 构造 IR、定义操作、改写、Pass、分析、转换、bufferization 与 lowering | [机制地图](./compiler/) → [IR 变换基础](./compiler/transforms/) / [定义 IR 抽象](./compiler/ir_definition/) / [IR 转换](./compiler/conversion/) |
| `guides/` 使用指南 | 阅读 IR、检查诊断、查源码、使用工具的可复用方法 | [使用指南](./guides/) |
| `tutorials/` 贯通教程 | 用一段完整程序连接多个机制，解释推理过程 | [贯通教程](./tutorials/) |

分类借鉴官方 [Code Documentation](https://mlir.llvm.org/docs/) 中 Language Reference、Dialects、编译器基础设施、Tools 与 Tutorials 的区分，再按本项目的依赖关系组织。它不是官网目录的逐页翻译。

同一机制只有一个主要定义位置。例如 SSA 在 `core/values_ssa.md` 定义，SCF 章节说明循环怎样使用 SSA，贯通教程再展示它们如何共同表达算法。阅读中的疑问用于改进相应章节，不再默认新增一篇独立问答。

## 内容地图

已有正文从基本IR贯通到算法、内存与目标编译；工具和领域扩展也有对应入口。先按学习路径选择一段连续任务，再用覆盖表查长期范围。

| 模块 | 连续学习过程 |
|---|---|
| [核心](./core/)与[基础方言](./dialects/) | 对象与SSA→类型/符号→条件、循环、区域结果→效果边界 |
| [IR变换](./compiler/transforms/) | Pass调度→C++对象修改→Pattern/driver→小Pass与测试；扩展CFG维护、DRR/PDLL |
| [定义IR](./compiler/ir_definition/) | my.add接入→字段/验证→Trait/Interface→Type/Attr/Region；扩展形状推导、复杂ODS与存储 |
| [Conversion](./compiler/conversion/) | 目标合法性→类型→混合表示→调用/区域边界→1:N展开 |
| [Tensor](./dialects/tensor/)、[Linalg](./dialects/linalg/)与[MemRef](./dialects/memref/) | 张量值→访问/归约→存储、布局和别名 |
| [Bufferization](./compiler/memory/) | 新旧值→复用/复制→所有权/释放→自定义接口 |
| [LLVM](./dialects/llvm/)与[CPU lowering](./compiler/lowering/) | 循环/descriptor→ABI→translation→C调用与JIT/AOT |
| [分析](./compiler/analysis/) | 支配/活跃性→别名/效果→缓存→数据流→优化合法性 |
| [Affine](./dialects/affine/)、[Vector](./dialects/vector/)与[优化](./compiler/optimization/) | 迭代域/依赖→分块融合→向量尾块→Transform→性能证据 |
| [GPU](./dialects/gpu/)与[目标后端](./compiler/targets/) | 并行映射→设备布局/同步→Host/Device→NPU资源与pipe |
| [算法贯通](./tutorials/kernels/) | Softmax基线/按行融合→Matmul调度→Attention与在线状态 |
| [实际编译链](./tutorials/pipelines/) | CPU全链路、Torch前端与Triton/Triton-Ascend/HIVM固定源码路径 |
| [使用指南](./guides/) | 观察/诊断→复现/缩减→回归/数值→C/Python→Bytecode/插件→有限源码阅读 |
| [领域扩展](./dialects/extensions/)与[Complex](./dialects/math/complex) | Quant、Sparse、Shape、IRDL、StableHLO/TOSA及复数表示的有限实例 |

这份内容地图对应当前已约定的教程范围，不要求第一次学习全部读完。扩展篇提供深化入口，具体实现与证据边界仍见各章和覆盖表。

## 广度与深度怎样维护

覆盖表用于检查内容是否遗漏；正文按理解过程组织，不机械地逐项列术语。后续章节以一个持续演进的例子连接现象、原理和实现，关键结论展开推导及中间状态，源码围绕具体论点解释。版本和复现日志集中放在文末或实验目录，减少阅读中断。

“深入”随阶段递进：基础阶段认识并解释小程序；阶段 B 建立 Pass 运行模型、C++ 对象修改与实现测试的基础；当前阶段 C 定义操作并逐步进入接口与转换；再往后追踪分析和优化的正确性与性能。扩展专题不自动成为上一阶段新增的完成门槛。

模块 `index.md` 负责定位和导航。机制正文承载完整解释；贯通教程承载跨章节推演。正式实践在阅读和讨论之后安排到 workspace 的 `aicompiler-labs/llvm-mlir/`，生成物进入 `artifacts/`。

## 版本与验证

基础版本为 LLVM `llvmorg-20.1.8`，commit `87f0227cb60147a26a1eeb4fb06e3b505e9c7261`。网页可能对应更新的 MLIR；本系列的语法和行为优先与固定版本源码核对。

示例按“完整合法模块、预期失败模块、局部片段、概念伪代码”区分。验证方法、复现命令和结果范围见[阅读与验证 IR](./guides/inspecting_ir)。验证通过表示相应检查通过；读者掌握程度另记，不根据文档生成状态推断。

## 从当前阅读进度继续

已理解IR定义后，建议从[Conversion](./compiler/conversion/)的操作与最小类型转换进入[编译流程导读](./tutorials/pipelines/overview)，再按Tensor/Linalg→MemRef/Bufferization→CPU执行链推进。之后选择Softmax或Matmul，把分析、优化和目标映射串入同一个任务。正文已生成不改变个人阅读与实践进度。
