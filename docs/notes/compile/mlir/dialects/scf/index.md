---
order: 5
title: SCF：结构化控制流
updated: 2026-09-06
excludeFromSidebar: true
---

# SCF：结构化控制流

前置：Operation/Region/Block、SSA、arith、CF。SCF 用拥有 Region 的操作保留条件、循环等结构；进入区域、绑定参数、接收 terminator 传值的规则由每个父操作定义。

## 当前章节

| 章节 | 核心协议 | 需要检查的边界 |
|---|---|---|
| [If：条件区域与结果](./if) | 选一个区域执行，yield 形成 if result | 返回结果时必须覆盖两个分支；捕获/逃逸；select |
| [For：循环状态与结果](./for) | 初始化 → body 参数 → yield → 下一轮或结果 | 零轮、正步长、多状态、类型对应与 CF |
| [While：两区域循环协议](./while) | before → condition → after/结果；after yield 回 before | 两套状态类型、首次判假、do-while、不终止 |

## 统一阅读区域协议

| 操作 | 拥有的 Region | 入口参数来自哪里 | terminator 的意义 |
|---|---|---|---|
| `scf.if` | then/else；无结果时 else 可为空 | 分支没有入口参数；可捕获合法外层值 | yield 提供本次 if 结果 |
| `scf.for` | 一个单 Block body | IV 与本轮携带状态 | yield 提供下一轮携带值，末轮成为结果 |
| `scf.while` | before/after，各一个 Block | before 来自 init/after yield；after 来自 condition | condition 选择继续/退出；yield 返回 before |

这里的“区域传出值”指 terminator 的 operands；Region 自身没有一个独立于父 Op 的 SSA result 列表。理解这一点才能区分内部 `%next` 和外部 `%result`。

结构化不等于“只是漂亮的语法糖”：操作名、边界、Region 结构和接口可以直接给变换提供循环/条件信息。转换到 CF 后，需要从显式控制流与数据流重新分析部分结构。具体编译器选择何时转换，取决于后续变换的需求。

## 后续范围

`execute_region` 在一个结构化位置容纳较一般的区域控制流；`index_switch` 表达多路选择；`parallel`、`forall` 及相应归约/并行插入操作涉及并行语义和映射。它们不等同于换一种拼写的顺序 for，后续需要单独展开执行、归约、同步和资源约束，见[覆盖项 D09](../../coverage)。

依据：[SCFOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)。当前版本为 LLVM 20.1.8；RegionBranchOpInterface 与 LoopLikeOpInterface 在实现阶段继续深入。
