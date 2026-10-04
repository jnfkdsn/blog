---
order: 1
title: 学习路径与阶段标准
updated: 2026-10-04
---

# 学习路径与阶段标准

本文记录 MLIR 的阶段标准、学习方式与已反馈进度，不替代 workspace 的 [AI Compiler 总路线](/notes/compile/ai-compiler/stack_and_roadmap)。IR 定义之后的逐篇阅读次序统一维护在[后续阅读顺序](./chapter_plan)，可以从第 1 站开始按页推进。

## 当前从哪里继续

Conversion 之后的[编译流程导读](./tutorials/pipelines/overview)、[Tensor](./dialects/tensor/)三篇与 [Linalg](./dialects/linalg/)三篇已提供。若已理解操作与最小类型转换，可从导读继续；Conversion 的实际阅读状态仍待反馈，不因新章生成而记为已学完。

**下一步阅读 [Dialect Conversion：操作转换与目标合法性](./compiler/conversion/operation_conversion)。** 基础 IR、Pass / Pattern 和 IR 定义已经完成当前轮次的阅读理解，可以开始学习怎样把自己定义的操作与类型转换成下游接受的表示。Conversion 的四篇主线、进阶参考与配套工程已提供，阅读安排见下方“下一步：转换基础与编译全链路”；用户阅读与独立实践待后续反馈。

阶段 A 的最小实验 `sum_positive.mlir` 已独立完成；阶段 B 的原理与小 Pass 工程已完成理解；阶段 C 中 IR 定义的重构版已读完，并确认比旧版更易理解。导读和五章机制现统一在[定义 IR 抽象](./compiler/ir_definition/)，自定义例子统一使用 `my` 方言。已有作者工程提供验证依据；个人独立修改与测试仍可在后续任务中积累，不阻塞下一章阅读。

工作区还提供了验证、SCF→CF 和 C 调用示例，助手已验证五组 CPU 输入。它们帮助连接观察与执行，不把完整 ABI/lowering、全部错误构造或所有基础专题的深入掌握追加为阶段 A 关卡。

已读完[Pass 与 pipeline：组织一次 IR 变换](./compiler/transforms/passes)，并讨论了 [C++ IR API](./compiler/transforms/ir_api) 的核心对象修改问题，重写后的 [PatternRewriter：把一次 IR 修改写成可应用的规则](./compiler/transforms/rewriting)已获得“主线更清楚、更易理解”的反馈，[实现并测试一个小 Pass](./tutorials/first_pass)也已完成理解。[一个操作是怎样被定义出来的](./compiler/ir_definition/op_definition)也已完成学习，IR 定义其余章节也已读完；接下来通过转换任务把定义与变换连接起来，原章节按需查阅。无需先背完 API；配套程序由编写者完成编译和验证，不据此将学习者的独立 C++ 实现能力标为完成。串联这几章时先读[IR 变换基础总览](./compiler/transforms/)，再按需回查各章。阶段 B 的已学内容如下：

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

## 为 AI 编译器学习 MLIR 到什么程度

MLIR 学习服务于解释和修改 AI 编译器中的表示与变换。当前建议以“解释一个小组件、修改一处行为、用正反例检查结果”为近期深度，不要求通读 MLIR 实现或记住生成 API。

| 层次 | 应达到的能力 | 进入下一项工作的边界 |
|---|---|---|
| 当前：最小组件 | 说明一个 Op 表达什么、输入输出及合法性；在参考工程上修改定义或一条 Pattern，解释前后 IR 并验证 | 可开始目标项目中一个简单算子的定点阅读，不必实现完整十二项扩展 |
| 随项目推进：编译路径 | 追踪多层 IR 如何降低；理解需要的 Conversion、tensor/memref、layout、bufferization 和内存/同步约束；跑通一条路径并验证结果 | 按所选项目实际采用的机制补齐，不要求所有编译器都经过同一 pipeline |
| 具体任务驱动：框架内部 | 深入某个分析、转换或框架实现，解释正确性、失败与性能 | 在修改该组件或排查问题时进入；不把任意 getter 的底层实现作为通用门槛 |

第一行是开始接触项目的标准，不表示长期基础已全部完成。TVM 的首次完整编译观察可以交错进行，也不以写完所有自定义 Type/Attribute/Interface 为前置。

这种安排与目标项目的关系如下。Triton 官方直接列出其 [MLIR dialects](https://triton-lang.org/main/dialects/dialects.html)，[AscendNPU IR](https://github.com/Ascend/AscendNPU-IR)也明确基于 MLIR，因此 IR、Op 定义、Pass 与转换能力可以直接用于阅读这些工程。扩展后端有时需要新增 Op 或 Dialect，但也可能只是修改现有 Pass；具体取决于要表达的新语义和现有抽象是否足够。

[TVM 架构](https://tvm.apache.org/docs/arch/)使用 Relax 与 TensorIR 等自身的表示。学习 MLIR 能帮助建立“抽象保留什么信息、变换为什么合法、怎样逐步降低”的比较视角；这是学习迁移的判断，不意味着 TVM 的这些 IR 就是 MLIR Dialect，也不意味着它们采用同样的对象模型或 API。

“深度”首先由能否解释语义、约束、变换前提与失败原因衡量。API 的拼写、重载和版本细节可以查阅；Op Interface 所表达的效果、类型或区域协议则需要在消费者依赖它时理解。可以暂缓其模板实现，不能忽略它对正确性的影响。

## 阶段 C：操作定义与语义契约

[定义 IR 抽象](./compiler/ir_definition/)是以下机制的主要学习入口。已经理解 [my.add 导读](./compiler/ir_definition/dialect_basics)后，可以直接从操作定义继续：它用 clamp 展示属性、验证和实际 Pattern 展开。Region 章先使用普通整数，不要求先掌握自定义类型存储；解析打印也可在操作定义后按需阅读。

| 顺序 | 内容 | 内容定位 | 本次理解目标 |
|---|---|---|---|
| C1 | [操作定义：语义、字段与验证](./compiler/ir_definition/op_definition) | clamp 的定义、构造与实际展开 | 语义约定 → 字段 → verifier → builder → Pattern 消费 |
| C2 | [通用语义：Trait 与 Interface](./compiler/ir_definition/traits_interfaces) | 范围查询与效果事实的消费 | 协议 → 实现 → 查询 → 决定，直接与外部接入 |
| C3 | [自定义 Attribute 与 Type](./compiler/ir_definition/types_attributes) | 计算参数与结果保证 | 分层验证、类型身份、checked 构造与转换需求 |
| C4 | [Region 操作](./compiler/ir_definition/regions_assembly) | 普通整数到多结果的区域协议 | 入口/出口对应、隔离、分阶段验证与分析边界 |
| C5 | [解析与打印](./compiler/ir_definition/assembly_format) | 同一个对象的两种文本入口 | 文本字段 → OperationState → 验证 → 打印往返 |
| C6 | [Dialect Conversion](./compiler/conversion/) | 独立机制模块；4 篇主线与 1 篇进阶参考已提供 | legality、类型转换与函数/区域边界衔接 |

C1 的观察入口为 `aicompiler-labs/llvm-mlir/05-op-definition/`。可按需回查自定义/通用打印、builder 构造与验证结果。C2—C5 共用 `aicompiler-labs/llvm-mlir/06-ir-definition/`，观察脚本按章展示查询、清理、类型身份和区域结构。完整 RegionBranch 模型、多组 variadic 实现、自定义高级存储等已有进阶篇与参考工程，见模块总览与覆盖表。

先前已理解 `0+x` 的扩展思路；独立改动和测试的证据后续补交即可，不阻塞当前模块阅读，也不以作者生成工程代替个人实践完成。

## 下一步：转换基础与编译全链路

**后续逐篇阅读以[IR 定义之后的阅读顺序](./chapter_plan)为准。** 该页按依赖安排 11 站：Conversion 基础 → 编译流程导读 → Tensor/Linalg → MemRef/Bufferization → 转换边界与 CPU 执行 → Softmax → 分析 → 结构化优化 → Matmul/Attention → GPU/NPU → 真实项目。这里不再维护第二张可能不同步的章节顺序表。

近期只推进 Conversion 的前两篇，随后进入编译流程导读。混合表示与调用/区域边界在 CPU 执行前回访；1:N、复杂 ODS、接口消费者等放在路线末尾的按需入口。理解 clamp 的字段、Pattern 的引用替换与 verifier 的职责，就足以开始下一章，不要求通读生成代码。

各站给出本轮理解目标和配套实验编号。正文生成与作者验证已经完成；个人阅读、独立修改及项目实践继续依据反馈记录。

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

结构化优化已提供完整六篇：[分块/融合/交换](./compiler/optimization/)之后，读 [Vector 三篇](./dialects/vector/)，再进入[自动向量化](./compiler/optimization/vectorization)、[Transform](./compiler/optimization/transform)与[性能分析](./compiler/optimization/performance)。[Affine](./dialects/affine/)补访问与依赖推理；14/15 实验提供数值、调度诊断和实测。设备映射正文已提供，新增内容不改变已确认的个人阅读进度。

## 主线之后的按需回访

需要接入工具时读[C/Python、Bytecode与插件](./guides/)；需要实现更复杂变换时回访[CFG维护](./compiler/transforms/ir_maintenance)、[1:N](./compiler/conversion/one_to_many)、[复杂ODS](./compiler/ir_definition/complex_ods)与[接口消费者](./compiler/ir_definition/region_interfaces)。遇到新领域表示时从[Quant/Sparse/Shape/IRDL等](./dialects/extensions/)确认用途与边界。源码阅读采用[一个变化的有限链](./guides/source_reading)，不以通读框架为进入项目的条件。

本次写作范围已经有对应正文与作者证据；这不改变上面保留的用户已反馈状态。设备运行、真实项目修改和独立练习仍按学习任务推进。
