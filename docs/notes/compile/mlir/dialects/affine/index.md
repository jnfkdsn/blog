---
order: 90
title: Affine：访问与循环约束
updated: 2026-10-04
---

# Affine：访问与循环约束

循环变换需要回答三个相连的问题：哪些点执行，每个点访问哪里，哪些点必须等待其他点。Affine 把适合这类推理的域与索引关系显式保留，方便计算约束和依赖。

| 顺序 | 章节 | 连续案例 |
|---|---|---|
| 1 | [Affine Map、维度与符号](./maps) | 从图像窗口坐标到源坐标，解释映射、SSA 绑定与作用域 |
| 2 | [Integer Set 与循环域](./domains) | 从下三角条件到余块，解释有效点集合与 min 边界 |
| 3 | [Affine 循环与依赖分析边界](./loops_analysis) | 从跨行递推推导地址相等与依赖距离，观察交换被拒绝及数值反例 |

三篇正文和配套真实查询、CPU 结果已提供。前置是基本循环、Linalg maps 与 MemRef；不要求先学完全部多面体算法。后续[结构化优化](../../compiler/optimization/)把域、访问和依赖用于 tiling、fusion 与 interchange。

Affine 是可分析表示，不是任意循环或地址的万能形式。semi-affine、间接索引和复杂效果是否支持，要按具体分析确认；不支持时保留原顺序或采用其他表示。可观察工程在 [14-structured-optimization](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)。
