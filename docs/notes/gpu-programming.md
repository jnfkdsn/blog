---
order: 1
title: GPU 编程与算子优化知识地图
updated: 2026-05-18
tags: [cuda, triton, gpu, kernel, optimization, learning-map]
status: draft
---

# GPU 编程与算子优化知识地图

这条线负责把 C++、CUDA、Triton、性能分析和算子优化串起来。目标不是记 API，而是形成一套看到算子后能判断并行方式、访存模式、瓶颈来源和优化方向的思维框架。

## 学习主线

1. 先补 C++ / 构建系统基础，能读懂 kernel 工程代码。
2. 再理解 CUDA 线程层次、内存层次、同步和规约。
3. 用 Roofline、Occupancy、Nsight Compute 判断瓶颈。
4. 通过 Reduce、Softmax、GEMM 这类算子训练优化直觉。
5. 用 Triton 对照 CUDA，理解更高层抽象保留了什么、隐藏了什么。

## 分析方法

- [C++ 前置知识](/notes/cuda/cpp)：读懂 kernel 工程代码需要的语言基础。
- [CMake 构建实践](/notes/cuda/cmake)：理解 CUDA / C++ 项目的构建方式。
- [CUDA 基础语法](/notes/cuda/cuda_basic_syntax)：线程层次、同步、shared memory、warp 操作。
- [Roofline 分析](/notes/cuda/roofline)：判断 memory-bound / compute-bound。
- [Triton 基础](/notes/triton/triton_basic)：块级编程模型、`tl.load`、`tl.store`、`tl.dot`、autotune。

## 典型算子

- [Reduce 优化实践](/posts/reduce)：规约、warp divergence、warp shuffle、多 block reduce。
- [Softmax 算子实现与优化](/posts/softmax)：数值稳定、访存次数、online softmax、向量化加载。
- [Transpose 访存优化](/posts/transpose)：coalescing、shared memory、bank conflict。
- [CUDA 矩阵乘法](/posts/GEMM)：tiling、thread tile、向量化加载、双缓冲。
- [Triton GEMM 优化](/posts/triton_gemm)：super-grouping、Tensor Core、FP8。

## 和推理系统的连接

- [Flash Attention 原理与实现](/notes/infer/flash_attention)：把 softmax、矩阵乘和 IO-aware 思路放进 attention。
- [LLM 架构基础](/notes/infer/llm_architechture)：理解 RMSNorm、RoPE、KV cache 等算子出现的位置。
- [量化笔记](/notes/infer/quantization)：把 GEMM、反量化和内存带宽问题连起来。

## 复盘问题

- 一个 kernel 是 memory-bound 还是 compute-bound？
- 当前优化是在减少访存、提高并行度，还是提高计算单元利用率？
- CUDA 版本里哪些细节被 Triton 抽象掉了？这些抽象会不会影响性能上限？

## 实践模板

```md
## 问题定义
## Baseline 实现
## 性能现象
## Bottleneck 分析
## 优化版本
## Benchmark 对比
## 关键结论
## 还没搞懂的
## 相关笔记
```
