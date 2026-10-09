---
order: 30
title: IR 转换与目标合法性
updated: 2026-10-02
excludeFromSidebar: true
---

# IR 转换与目标合法性

定义 IR 时，我们决定一种计算保留哪些信息，以及怎样构造和检查它。转换 IR 时，我们决定哪些信息需要落实到下一种表示，以及何时可以将结果交给后续阶段。

本模块接在[定义 IR 抽象](../ir_definition/)之后，与它并列归档。先前的 Pass 和 Pattern 继续负责组织与描述修改；Conversion 补充目标合法性，并在需要时协调类型与数据流边界的变化。

```text
my.clamp：保留区间限制的含义与静态参数
    ↓ 展开计算，并要求不再残留 clamp
arith：用常量与 max/min 表达同一项计算
    ↓ 后续阶段继续选择自己的目标与转换规则
更低层的表示与最终执行
```

## 阅读主线

本模块已有 **4 篇主线 + 1 篇进阶参考**。前两篇建立最小转换；后两篇在相同计算上延伸边界，不必一次读完所有进阶内容。

| 顺序 | 章节 | 本篇完整解释的过程 |
|---|---|---|
| 1 | [操作转换与目标合法性](./operation_conversion) | my.clamp 到 arith；目标、规则、driver 与 partial/full 的完成条件 |
| 2 | [类型转换与函数签名](./type_conversion) | my.limit 的 range 到 i32，连接 TypeConverter、adaptor、使用者与最小返回边界 |
| 3 | [混合表示与边界衔接](./materialization) | 保留旧接口时的 materialization、占位 cast 与消解条件 |
| 4 | [调用与区域边界转换](./boundaries) | caller/callee、Block 参数和 Region 入口/出口的一致转换 |
| 扩展 | [一个值到多个值的转换](./one_to_many) | pair 拆为两个 i32，转换生产者、提取、函数、call/return 与 CFG 边界 |
| 参考 | [复杂转换路径与框架行为](./advanced) | analysis、递归合法性、1:N 表示设计、规则路径与回退；按需阅读 |

类型与边界按依赖分开，第 2 篇先完成一个自包含的函数，第 3 篇只改变“保留旧接口”这个条件，第 4 篇再把数据流扩展到调用和区域。[一个值到多个值的转换](./one_to_many)已有完整 pair 工程，连接函数、调用、Block 参数及 C 数值；作为按需扩展，不是进入张量计算的前置。

07 工程验证第一篇；08 工程复用已有 my 方言，验证后续主线与进阶对照。练习要求预测一个变化、修改有限输入或规则、解释前后结果，不重新讲一遍 Pass 注册和构建。全部后续模块见[章节目录](../../chapter_plan)。

第一章只要求理解 clamp 的语义、字段和普通 Pattern 替换，不要求记住生成代码或实现完整自定义类型存储。它采用已经熟悉的标量计算，让新增问题集中在“允许什么结果、怎样确认完成”。

## 与其他模块的分工

- [IR 定义](../ir_definition/)解释源操作、类型和属性的含义及自身约束。
- [IR 变换基础](../transforms/)解释如何描述局部修改，以及怎样装入 Pass 和 pipeline。
- 本模块解释阶段之间的表示转换、目标要求，以及类型变化时的边界协调。

阅读前两篇后即可进入编译流程导读和 Tensor/Linalg，后续边界与进阶内容按任务回访。当前正文、作者实现证据和仍待深入的项目范围由[覆盖表 M05](../../coverage)维护。框架接口和驱动内部实现按任务查阅，不需要在主线之后全部通读。

[操作转换实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/07-dialect-conversion)与[类型/边界实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/08-type-conversion)展示实际前后 IR 和失败原因；正文中的理解不依赖先搭建实验。
