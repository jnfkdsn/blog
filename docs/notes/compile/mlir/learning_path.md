---
order: 1
title: 学习路径与阶段标准
updated: 2026-10-17
---

# 学习路径与阶段标准

本页说明不同阶段需要建立的能力和前置依赖。IR 定义之后的逐篇阅读顺序统一在[后续阅读顺序](./chapter_plan)维护；知识归属从[模块首页](./)查找；长期深度用[覆盖表](./coverage)检查。

## 选择学习起点

| 已有基础 | 建议起点 | 本轮主要目标 |
|---|---|---|
| 尚不能解释 MLIR 文本 | 下方基础 IR 阅读表 | 识别对象、值来源、类型和控制流 |
| 能读小程序，希望修改 IR | [IR 变换基础](./compiler/transforms/) | 解释和实现一次局部变换 |
| 已理解 Pass 与 Pattern | [定义 IR 抽象](./compiler/ir_definition/) | 理解操作语义、契约与框架接入 |
| 已理解基本操作定义 | [Conversion 与后续阅读顺序](./chapter_plan) | 连接转换、张量、内存与执行 |
| 已能解释一条编译链 | [优化](./compiler/optimization/)、[目标后端](./compiler/targets/)或具体项目 | 围绕任务深入分析、调度与代码生成 |

个人阅读与实践进度不由本页推断。已掌握前置内容时直接进入所需阶段；遇到具体机制再回访对应章节。

## 阶段标准

| 阶段 | 核心工作 | 前置能力 | 可检查的学习产物 |
|---|---|---|---|
| A：阅读 IR | 对象模型、SSA、类型、基本控制流 | 基本程序阅读能力 | 独立编写并解释一个通过 verifier 的小程序 |
| B：使用与实现变换 | Pipeline、IR API、Pattern、Pass 与测试 | A；实现部分按需补 C++ | 一个有正反例测试的小变换，说明匹配条件与不变量 |
| C：定义与转换抽象 | ODS、Type/Attr、Trait/Interface、verifier、Conversion | A 与 B 的构造和改写能力 | 小 Dialect 与有限转换，展示非法输入和转换失败 |
| D：张量与内存 | Tensor/Linalg、MemRef、Bufferization 与 CPU lowering | 能理解操作及转换的契约 | 一条可调用的 CPU 执行链，解释形状、别名与生命周期 |
| E：优化与目标后端 | 分析、分块、融合、向量化、GPU/NPU 映射 | D 与所选目标的硬件基础 | 带语义约束、边界测试与性能解释的优化案例 |

这些阶段表示能力递进，不是所有项目都必须走完的统一前置列表。第一轮能说明一个小组件、修改一处行为并验证结果，就可以开始具体编译项目的有限阅读；后续随任务补齐相关机制。

## 基础 IR 阅读表


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

## 基础阶段的完成边界

认识 Operation、Region、Block、Value、类型与基本控制流，能解释小程序，并独立完成一个合法 IR 实验，即可进入变换部分。支配错误、区域传值反例、效果分析与源码细节仍可按需要回访；进入 Pass 之前不要求完整解释 LLVM 后端。

## 定义与转换的衔接

先沿 [my.add 导读](./compiler/ir_definition/dialect_basics)建立操作与工具接入的整体认识，再用 [clamp](./compiler/ir_definition/op_definition)理解字段、验证和 Pattern 消费。Traits、Interfaces、Type/Attribute 与 Region 解决更具体的表达需求，按模块顺序逐步展开。

完成基本定义后，从 [Conversion](./compiler/conversion/)开始，并按[后续阅读顺序](./chapter_plan)逐站连接 Tensor/Linalg、MemRef/Bufferization 与 CPU 执行。混合表示和函数边界在需要时回访；1:N、复杂 ODS 与接口消费者属于进一步扩展。

## 与 AI 编译器学习的关系

MLIR 提供理解表示、分析、变换和渐进 lowering 的方法。TVM 使用自己的 Relax 与 TensorIR，具体 API 和执行路径需重新学习；Triton、Triton-Ascend 等项目也有各自的方言、目标约束和工具版本。

已有 IR 与变换基础时，可穿插 [TVM](/notes/compile/tvm/)和 [PyTorch Compiler](/notes/compile/pytorch-compiler/)观察完整链路，不要求先读完全部 MLIR 扩展。需要判断学习顺序时回到[总路线](/notes/roadmap)，需要查长期能力时使用[编译器能力与项目](/notes/compile/capabilities)。

## 按需回访

- 接入工具：[C/Python、Bytecode 与插件](./guides/)。
- 复杂变换：[CFG 维护](./compiler/transforms/ir_maintenance)、[1:N](./compiler/conversion/one_to_many)、[复杂 ODS](./compiler/ir_definition/complex_ods)与[接口消费者](./compiler/ir_definition/region_interfaces)。
- 新领域表示：[Quant、Sparse、Shape、IRDL 等](./dialects/extensions/)。
- 阅读实现：[一个变化的有限源码链](./guides/source_reading)。

API 的拼写与版本细节可以查阅；机制所依赖的语义、约束、变换前提与失败原因，需要通过具体输入推演清楚。阅读、参考示例运行和独立实现分别记录。
