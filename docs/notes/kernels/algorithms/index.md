---
order: 30
title: 算子算法
---

# 算子算法

这里按计算本身组织可复用的算法与访存原理。具体 CUDA、Ascend C 或编译器实现从对应模块进入，实际优化测量保存在实践记录中。

- [FlashAttention 原理与实现](./flash_attention)：将 Attention 的计算、在线 Softmax 与分块访存联系起来。
- [Softmax 实践](/posts/softmax)：稳定 Softmax、归约与不同实现的对照。
- [CUDA GEMM 实践](/posts/GEMM)：矩阵分块与数据复用。
- [MLIR 算法贯通](/notes/compile/mlir/tutorials/kernels/)：同一类计算如何经过结构化表示、存储安排和编译变换。
