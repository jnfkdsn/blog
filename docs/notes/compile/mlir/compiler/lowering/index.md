---
order: 50
title: CPU Lowering 与执行边界
updated: 2026-10-04
---

# CPU Lowering 与执行边界

从一个矩阵加法出发，先把结构化计算变成循环和访存，再让函数能够按明确 ABI 与 C 交换数据，最终通过 AOT 或 JIT 执行。

## 阅读前置

Tensor/Linalg、MemRef 与 Bufferization；需要时回查 CF 和 Dialect Conversion。

## 阅读顺序

| 顺序 | 章节 | 贯穿过程 |
|---|---|---|
| 1 | [结构化计算到显式循环](./loops) | 由 indexing maps 产生访问，再将循环携带状态改成 CFG 传值。 |
| 2 | [类型降低、ABI 与函数边界](./abi) | 描述符展开、C wrapper、返回结构体及匹配的生命周期。 |
| 3 | [Translation、JIT 与 AOT](./execution) | 从 LLVM IR 到代码生成、符号连接和实际调用，按阶段定位失败。 |

## 完成边界与衔接

三篇主线已有正文。[Linalg 到 CPU 的最小执行链](../../tutorials/pipelines/cpu)将它们接成一个完整案例；[LLVM 方言](../../dialects/llvm/)提供低层操作和类型参考。

示例以 LLVM 20.1.8 和当前 64 位 CPU 为验证目标，目标布局与 ABI 需要随实际项目核对；设备路径归入后续目标模块。整个课程见[章节目录](../../chapter_plan)和[覆盖表](../../coverage)。
