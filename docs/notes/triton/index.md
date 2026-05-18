---
order: 0
---

# triton学习笔记

Triton 笔记用于对照 CUDA 学习更高层的 GPU 编程抽象：块级张量、mask load/store、`tl.dot`、autotune，以及这些抽象对性能优化的影响。

## 推荐路径

1. [Python 前置知识](/notes/triton/python_basic)
2. [Triton 基础](/notes/triton/triton_basic)
3. [Triton GEMM 优化](/posts/triton_gemm)

## 和 CUDA 的对照

- [CUDA 基础语法](/notes/cuda/cuda_basic_syntax)：对照 thread/block/grid、shared memory、warp 级操作。
- [Softmax 算子实现与优化](/posts/softmax)：对照 CUDA softmax 和 Triton softmax 的表达方式。
- [CUDA 矩阵乘法](/posts/GEMM)：对照手写 CUDA GEMM 和 Triton `tl.dot`。

## 相关地图

- [GPU 编程与算子优化知识地图](/notes/gpu-programming)
