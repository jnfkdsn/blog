---
order: 70
title: Linalg：结构化计算
updated: 2026-10-04
---

# Linalg：结构化计算

从逐元素计算、矩阵乘法和归约的公式还原迭代空间，而不是先记 named op 的列表。

本模块三篇主线已经提供。必要的 IR、坐标推演和结果直接写在正文；可选实验用于实际观察与修改。

## 阅读前置

Tensor 值与形状；标量 Arith 和 Region 传值。

## 阅读顺序

| 顺序 | 章节 | 贯穿案例与需要解释的过程 |
|---|---|---|
| 1 | [迭代空间与访问映射](./iteration_maps) | 逐元素运算：indexing_maps、iterator_types 与区域内标量计算；就地引入 affine map。 |
| 2 | [广播、归约与矩阵乘法](./structured_computations) | 从公式和维度推导 named/generic，解释并行与归约维。 |
| 3 | [Destination-Passing Style 与结果构造](./destination_style) | 沿一个归约追踪 outs、初始值、结果 shape 和新 SSA Value。 |

## 完成边界与衔接

能从公式构造并解释一个小 Linalg 计算；tiling、fusion 归入结构化优化模块。

整个课程的先后关系见[后续章节目录](../../chapter_plan)，当前应该从哪里继续见[学习路径](../../learning_path)，长期深度由[覆盖表](../../coverage)约束。文章已提供不表示学习者已完成阅读或实践；后续按具体案例继续补深度。

接下来进入 [MemRef](../memref/) 与 [Bufferization](../../compiler/memory/)（各三篇主线已提供）；不要求在此阶段先实现完整 tiling 或目标后端。
