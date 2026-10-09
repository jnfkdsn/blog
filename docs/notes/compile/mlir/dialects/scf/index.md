---
order: 5
title: SCF：结构化控制流
updated: 2026-10-04
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
| [多Block区域与多路选择](./regions_switch) | execute_region的多出口、index_switch选择与CF合流 | Pass支持边界、固定版本selector位宽限制 |
| [并行归约与张量结果](./parallel_results) | parallel贡献/合并、forall共享输出/切片组装 | 初值、空迭代、竞争、串行参考与设备映射 |

## 统一阅读区域协议

| 操作 | 拥有的 Region | 入口参数来自哪里 | terminator 的意义 |
|---|---|---|---|
| `scf.if` | then/else；无结果时 else 可为空 | 分支没有入口参数；可捕获合法外层值 | yield 提供本次 if 结果 |
| `scf.for` | 一个单 Block body | IV 与本轮携带状态 | yield 提供下一轮携带值，末轮成为结果 |
| `scf.while` | before/after，各一个 Block | before 来自 init/after yield；after 来自 condition | condition 选择继续/退出；yield 返回 before |

这里的“区域传出值”指 terminator 的 operands；Region 自身没有一个独立于父 Op 的 SSA result 列表。理解这一点才能区分内部 `%next` 和外部 `%result`。

结构化不等于“只是漂亮的语法糖”：操作名、边界、Region 结构和接口可以直接给变换提供循环/条件信息。转换到 CF 后，需要从显式控制流与数据流重新分析部分结构。具体编译器选择何时转换，取决于后续变换的需求。

## 与分析及目标映射的连接

五篇已覆盖基本区域协议与D09的并行/选择范围。[区域接口](../../compiler/ir_definition/region_interfaces)进一步解释分析怎样消费入口/出口事实；[并行映射](../../compiler/targets/parallel_mapping)连接GPU层次和同步。CPU串行参考与设备并行验证分开记录。

依据：[SCFOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)。当前版本为 LLVM 20.1.8；RegionBranchOpInterface 与 LoopLikeOpInterface 的消费见区域接口章。
