---
order: 1
title: 学习路径与阶段标准
updated: 2026-09-06
---

# 学习路径与阶段标准

本文安排 MLIR 内部的阅读依赖，不替代 workspace 的 [AI Compiler 总路线](/notes/compile/ai-compiler/stack_and_roadmap)。知识目录用于查找，学习路径用于决定下一步；目录顺序不必等于所有人的阅读顺序。

## 当前从哪里继续

之前已经能看懂 IR 导读、包含结构和 for/yield 的说明，可以把相关内容当复习。重写后的章节补上了此前零散出现的约束与边界，建议先检查第 1—2 项，然后顺序阅读；遇到已经能独立解释的内容可以跳读正文，完成章末检查即可。

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

1. 先读章节开头的范围，知道本章解释什么、依赖什么。
2. 阅读定义与完整例子，在纸上标出对象归属、Value 来源、类型和控制去向。
3. 阅读边界与反例，说明失败的是哪条约束；不要只记报错文本。
4. 用章末检查验证能否脱离原文解释。答案读得懂与独立解释得出是两个状态。
5. 遇到定义冲突或实现疑问，沿文末给出的一个源码入口查证，再回到本章。不沿每个调用无限展开。

正文里的执行表是阅读材料，不要求边读边搭建正式项目。可以先读原理、讨论疑问，再安排正式实验；讲解者会预先验证示例，避免把无效示例留给读者排查。

## 当前阶段的完成标准

给一段含函数、分支、循环和内存读写的陌生小程序，能够完成以下检查：

| 检查 | 合格表现 |
|---|---|
| 结构 | 画出 Operation → Region → Block → Operation，不把调用画成包含 |
| 数据流 | 对每个 `%value` 指出定义位置、类型、使用位置，以及跨边/跨区域传值方式 |
| 控制流 | 推演 if/for/while 的正常与边界路径，说明终结操作把值交给谁 |
| 合法性 | 解释一个支配错误、一个类型错误、一个作用域错误 |
| 语义 | 解释一个 IR 合法但改写错误的例子，不能仅依靠 verifier 判断优化正确 |
| 查证 | 从操作名找到 ODS 定义、必要的 C++ 实现和已有测试 |

当前阅读基础来自之前的交流；新章的独立解释与实践均尚未验收。学习记录不能因为正文或测试生成完毕就自动标为“掌握”。

## 后续阶段与依赖

| 阶段 | 核心工作 | 开始前需具备 | 完成证据；之后再具体编写实验 |
|---|---|---|---|
| A：阅读 IR（当前） | 本页 1—10 项 | 基本程序阅读能力 | 上述解释检查 + 最小 IR 实验 |
| B：使用与实现变换 | pipeline、IR API、fold、PatternRewriter、Pass、lit/FileCheck | A，基本 C++ | 一个有正反例测试的小变换，解释匹配条件与保持的不变量 |
| C：定义与转换抽象 | ODS、Type/Attr、Trait/Interface、verifier、Dialect Conversion | A；B 的构造/改写能力 | 小 Dialect 与带类型变化的 conversion；展示不合法输入和转换失败 |
| D：张量与内存 | tensor/memref/linalg、DPS、bufferization、释放、LLVM lowering | B—C 的使用能力 | 一条可运行的 CPU lowering 链，验证形状、别名、生命周期和数值 |
| E：优化与目标后端 | 分析、tiling/fusion/vector、Transform、GPU/NPU 表示与目标项目 | D；所选目标的硬件基础 | 带语义约束、边界测试与性能解释的优化案例 |

阶段并非每个主题只接触一次：例如 B 首先使用 Interface 判断一个 Op 能否改写，C 再实现 Interface，E 再分析它怎样支撑优化。深度递进由[覆盖表](./coverage)的目标深度控制，不能用“学过一次”代替长期能力。

Softmax、归约和 attention 会在 D—E 作为贯通项目。当前的小程序专门训练 IR 语义；进入张量和内存之后再逐步增加计算与性能复杂度。Triton/Triton-Ascend 的具体流水线在对应项目与版本中核对，不假设每个 MLIR 编译器都采用相同 lowering 顺序。
