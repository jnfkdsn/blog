---
order: 1
title: 学习路径与阶段标准
updated: 2026-09-24
---

# 学习路径与阶段标准

本文安排 MLIR 内部的阅读依赖，不替代 workspace 的 [AI Compiler 总路线](/notes/compile/ai-compiler/stack_and_roadmap)。知识目录用于查找，学习路径用于决定下一步；目录顺序不必等于所有人的阅读顺序。

## 当前从哪里继续

**阶段 A“认识基础 IR + 最小 IR 实验”已完成；阶段 B 的原理与小 Pass 工程已完成理解，阶段 C 的 IR 定义五章已读完，但各机制的工程联系仍需建立。当前转入 [my.add 增量工程](./tutorials/my_dialect/)，先走通最小定义、生成、注册和读写过程。** 已完成基础正文阅读，并独立编写 `sum_positive.mlir`：结构、类型和两个循环状态的传递经审阅与 verifier 检查通过。计数保留 i32 是当前练习的合法选择。

工作区还提供了验证、SCF→CF 和 C 调用示例，助手已验证五组 CPU 输入。它们帮助连接观察与执行，不把完整 ABI/lowering、全部错误构造或所有基础专题的深入掌握追加为阶段 A 关卡。

已读完[Pass 与 pipeline：组织一次 IR 变换](./compiler/transforms/passes)，并讨论了 [C++ IR API](./compiler/transforms/ir_api) 的核心对象修改问题，重写后的 [PatternRewriter：把一次 IR 修改写成可应用的规则](./compiler/transforms/rewriting)已获得“主线更清楚、更易理解”的反馈，[实现并测试一个小 Pass](./tutorials/first_pass)也已完成理解。[一个操作是怎样被定义出来的](./compiler/ir_definition/op_definition)也已完成学习，IR 定义其余章节也已读完；接下来通过完整的小工程巩固，原章节按需查阅。无需先背完 API；配套程序由编写者完成编译和验证，不据此将学习者的独立 C++ 实现能力标为完成。串联这几章时先读[IR 变换基础总览](./compiler/transforms/)，再按需回查各章。阶段 B 的已学内容如下：

| 顺序 | 章节 | 当前状态 | 本节要建立的能力 |
|---|---|---|---|
| B1 | [Pass 与 pipeline](./compiler/transforms/passes) | 已读完；正文示例已验证 | 理解一次运行、处理顺序、嵌套范围、日志和失败 |
| B2 | [C++ IR API](./compiler/transforms/ir_api) | 核心问题已讨论，按需巩固；观察实验已提供 | 对象生命周期、use-list、遍历、插入、替换/删除、构造与映射克隆 |
| B3 | [PatternRewriter](./compiler/transforms/rewriting)；配套[改写驱动](./compiler/transforms/rewrite_drivers) | 主章写法已获肯定；两篇示例已验证，深入驱动内容按需回查 | 先走通单条规则的匹配、替换、调用与通知，再按需深入规则协作和收敛 |
| B4 | [实现并测试一个小 Pass](./tutorials/first_pass) | 已完成理解；作者工程已验证，学习者独立修改尚无提交证据 | 组合前述机制，完成构建、注册与 lit/FileCheck 正反例测试 |

每一部分继续采用“先读原理、讨论，再做相应实践”的节奏。B1 不要求先能写完整 C++ Pass。

B2 的观察入口为 `aicompiler-labs/llvm-mlir/03-ir-api/`：先预测 RAUW/erase 怎样改变引用，再运行查看中间 IR，最后修改一处输入验证预测。不要求背诵 API；`docs/validate_*.py` 是文档回归工具，通过计数不作为学习完成证据。B4 的工程入口为 `aicompiler-labs/llvm-mlir/04-small-pass/`：先读完整过程，再预测和观察前后 IR，最后独立扩展左侧加零并补测试。

这里的 A—E 是 MLIR 内部的学习阶段，与总路线中的阶段编号不是一一对应。总路线“统一 IR 与 Pass 基础”还包含操作定义、转换、分析与内存等内容，分布在后面的多个模块中；读完 B1 足以继续学习 IR 修改机制，但并不表示总路线的基础范围已经全部完成。

已有传统编译器背景时，CSE、DCE 等熟悉规则可以快速回顾，把精力放在 MLIR 的对象与生命周期、Region 约束、改写协议、legality / TypeConverter、Interface 和内存语义上。常用传统优化及它们依赖的分析可查阅[Dataflow Analysis 与 Pass Pipeline](../traditional/dataflow_pass)；后续的深度通过推导合法性、实现并验证代表性变换建立，不通过增加更多名词或重复简单规则建立。

## 阶段 C：从操作定义继续

[定义 IR 抽象](./compiler/ir_definition/)说明本模块的内部关系，现已备齐本模块五章核心正文。这五章已阅读，但读者反馈概念互相引用、缺少完整工程链。当前不要求重读整组，先跟随 [01：让自己的工具认识 my.add](./tutorials/my_dialect/01_minimal_dialect) 建立统一模型：观察参考工程，再独立完成字段改名任务；后续接入 Pattern/Pass，再逐步增加机制。

| 顺序 | 内容 | 当前状态 | 本次理解目标 |
|---|---|---|---|
| C1 | [一个操作是怎样被定义出来的](./compiler/ir_definition/op_definition) | 已完成阅读；作者 ODS/C++ 工程与正反例已验证 | 语义约定 → ODS → 生成 API → verifier → 注册与打印 → Pass 消费 |
| C2 | [Trait / Interface 与通用消费者](./compiler/ir_definition/traits_interfaces) | 已阅读；工程联系通过增量项目巩固 | 效果与删除判断、范围协议、直接/外部模型及注册 |
| C3 | [自定义 Type / Attribute](./compiler/ir_definition/types_attributes) | 已阅读；工程联系通过增量项目巩固 | 参数定义、两层验证、uniquing、checked 构造与类型相等 |
| C4 | [Region、可变参数与验证](./compiler/ir_definition/regions_assembly) | 已阅读；区域扩展留到后续项目阶段 | 入口/出口对应、隔离与验证顺序 |
| C5 | [解析与打印](./compiler/ir_definition/assembly_format) | 已阅读；本轮观察生成 parser/printer | 文本字段 → OperationState → 验证 → 打印往返 |
| C6 | Dialect Conversion | 后续模块，待编写；先完成 my.add 普通 Pattern 贯通 | legality、类型转换与函数/区域边界衔接 |

C1 的观察入口为 `aicompiler-labs/llvm-mlir/05-op-definition/`。可按需回查自定义/通用打印、builder 构造与验证结果。C2—C5 共用 `aicompiler-labs/llvm-mlir/06-ir-definition/`，观察脚本按章展示查询、清理、类型身份和区域结构。完整 RegionBranch 模型、多组 variadic 实现、自定义高级存储等继续留作进阶范围，见模块总览与覆盖表。

先前已理解 `0+x` 的扩展思路；独立改动和测试的证据后续补交即可，不阻塞当前模块阅读，也不以作者生成工程代替个人实践完成。

## 阶段 A 内容索引（按需回顾）

| 顺序 | 阅读内容 | 前置依赖 | 读完应该能独立说明 |
|---|---|---|---|
| 1 | [IR 对象模型](./core/ir_model) | 无；认识简单函数和加法即可 | 文本与内存表示、四种关系、Region 的语义归属 |
| 2 | [Value、SSA 与支配](./core/values_ssa) | 1 | 值的定义/使用、合流、回边与跨 Region 支配 |
| 3 | [类型与静态信息](./core/types_attributes) | 1—2 | 类型、形状、Attribute、Properties 各自描述什么 |
| 4 | [符号与作用域](./core/symbols_scopes) + [builtin](./dialects/builtin) + [func](./dialects/func) | 1—3 | 函数签名与 Op 签名、SSA 捕获与符号引用的区别 |
| 5 | [arith](./dialects/arith) | 2—3 | 常量、比较、整数/浮点语义、显式转换的边界 |
| 6 | [cf](./dialects/cf) | 2、4—5 | 沿边传值、Block 参数、合流与循环 CFG |
| 7 | [SCF 概览](./dialects/scf/) → [if](./dialects/scf/if) → [for](./dialects/scf/for) → [while](./dialects/scf/while) | 1—6 | Region 进入/退出协议、零次迭代、状态类型对应 |
| 8 | [内存、效果与优化边界](./core/effects) | 2—3、5、7 中的 if/for | SSA 不等于存储不可变、无结果不等于可删除 |
| 9 | [贯通阅读一段 IR](./tutorials/reading_ir) | 1—8 | 从算法逐层解释完整模块与 lowering 后的数据流 |
| 10 | [阅读与验证 IR](./guides/inspecting_ir) | 1—9 | 判断解析、验证、转换、执行、等价性证据的区别 |

`while` 不要求背会全部语法。要求能根据两套参数类型和两个 terminator 还原执行协议，避免把所有结构化控制流都套成 for。

## 一章怎样学

1. 先看本章的具体输入与目标，明确它要解释的过程；前置与范围按学习路径查阅。
2. 沿完整例子逐步推演，在纸上标出对象归属、Value 来源、类型和控制去向，再用新概念解释这些变化。
3. 阅读边界与反例，说明失败的是哪条约束；不要只记报错文本。
4. 用章末检查验证能否脱离原文解释。答案读得懂与独立解释得出是两个状态。
5. 遇到定义冲突或实现疑问，沿文末给出的一个源码入口查证，再回到本章。不沿每个调用无限展开。

正文里的执行表是阅读材料，不要求边读边搭建正式项目。可以先读原理、讨论疑问，再安排正式实验；讲解者会预先验证示例，避免把无效示例留给读者排查。

## 阶段 A 的完成边界

基础目标是认识 Operation/Region/Block、Value/类型与基本控制流，能阅读并解释小程序，并独立完成一个通过 verifier 的最小 IR 实验。当前已经达到这个阶段目标，可以继续学习。

支配错误、区域传值反例、效果分析、源码细节等仍可按后续需要回访。阶段完成不等于每项长期能力都已深入掌握，也不要求在进入 Pass 之前完整实现或解释 LLVM 后端。

## 后续阶段与依赖

| 阶段 | 核心工作 | 开始前需具备 | 完成证据；之后再具体编写实验 |
|---|---|---|---|
| A：阅读 IR（基础目标已完成） | 基础概念与最小 IR | 基本程序阅读能力 | 基础阅读 + 独立编写并解释一个合法小程序 |
| B：使用与实现变换（原理已理解） | pipeline、IR API、fold、PatternRewriter、Pass、lit/FileCheck | A；实现部分按需补 C++ | 一个有正反例测试的小变换，解释匹配条件与保持的不变量 |
| C：定义与转换抽象（当前） | ODS、Type/Attr、Trait/Interface、verifier、Dialect Conversion | A；B 的构造/改写能力 | 小 Dialect 与带类型变化的 conversion；展示不合法输入和转换失败 |
| D：张量与内存 | tensor/memref/linalg、DPS、bufferization、释放、LLVM lowering | B—C 的使用能力 | 一条可运行的 CPU lowering 链，验证形状、别名、生命周期和数值 |
| E：优化与目标后端 | 分析、tiling/fusion/vector、Transform、GPU/NPU 表示与目标项目 | D；所选目标的硬件基础 | 带语义约束、边界测试与性能解释的优化案例 |

阶段并非每个主题只接触一次：例如 B 首先使用 Interface 判断一个 Op 能否改写，C 再实现 Interface，E 再分析它怎样支撑优化。深度递进由[覆盖表](./coverage)的目标深度控制，不能用“学过一次”代替长期能力。

Softmax、归约和 attention 会在 D—E 作为贯通项目。当前的小程序专门训练 IR 语义；进入张量和内存之后再逐步增加计算与性能复杂度。Triton/Triton-Ascend 的具体流水线在对应项目与版本中核对，不假设每个 MLIR 编译器都采用相同 lowering 顺序。
