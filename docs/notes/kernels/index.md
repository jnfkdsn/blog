---
order: 30
title: 算子编程与优化
updated: 2026-10-17
tags: [cuda, triton, gpu, kernel, optimization, learning-map]
status: draft
---

# 算子编程与优化

这一部分从具体算子出发，解释计算如何分工、数据如何搬运、同步如何保证正确，以及怎样定位性能瓶颈。CUDA 与 Ascend C 分别展开目标编程模型；Triton 的语言与编译器统一在编译器模块学习。

## 模块入口

| 模块 | 内容 |
|---|---|
| [CUDA](./cuda/) | GPU 线程、存储、同步、Tensor Core 与异步流水 |
| [Ascend C](./ascend-c/) | AI Core 架构、Tensor 与队列、Vector 算子、Tiling 与验证 |
| [算子算法](./algorithms/) | Attention 等计算的算法组织与访存原理 |
| [Triton / Triton-Ascend](/notes/compile/triton/) | 块级程序如何表达并进入目标编译链 |

## CUDA 阅读与实践路线

1. 先补 C++ / 构建系统基础，能读懂 kernel 工程代码。
2. 再理解 CUDA 线程层次、内存层次、同步和规约。
3. 用 Roofline、Occupancy、Nsight Compute 判断瓶颈。
4. 通过 Reduce、Softmax、GEMM 这类算子训练优化直觉。
5. 学习低精度、Tensor Core 和异步流水，把普通 CUDA Core GEMM 推进到 MMA mainloop。
6. 用 Triton、CUTLASS 对照 CUDA，并把 `matmul/tl.dot` 连接到编译器 lowering。
7. 把优化方法迁移到 RMSNorm、RoPE、SwiGLU、Quantized GEMM 和 Attention。

NPU 方向从 [Ascend C 阅读顺序](./ascend-c/)进入，分别理解核间分工、片上存储与搬运流水，再选择一个 Vector 算子验证。两类硬件共享正确性与性能分析方法，但执行与同步模型需分别学习。

## 分析方法

- [C++ 前置知识](/notes/kernels/cuda/cpp)：读懂 kernel 工程代码需要的语言基础。
- [CMake 构建实践](/notes/kernels/cuda/cmake)：理解 CUDA / C++ 项目的构建方式。
- [CUDA 基础语法](/notes/kernels/cuda/cuda_basic_syntax)：线程层次、同步、shared memory、warp 操作。
- [Roofline 分析](/notes/kernels/cuda/roofline)：判断 memory-bound / compute-bound。
- [低精度数值与混合精度计算](/notes/kernels/cuda/low_precision)：区分存储、乘法输入、累加器和输出精度。
- [Tensor Core 编程](/notes/kernels/cuda/tensor_core)：WMMA、MMA tile、warp fragment 和 epilogue。
- [Ampere 异步拷贝与软件流水线](/notes/kernels/cuda/async_pipeline)：`cp.async`、stage 和 copy-compute overlap。
- [Triton 基础](/notes/compile/triton/triton_basic)：块级编程模型、`tl.load`、`tl.store`、`tl.dot`、autotune。

## 典型算子

- [Reduce 优化实践](/posts/reduce)：规约、warp divergence、warp shuffle、多 block reduce。
- [Softmax 算子实现与优化](/posts/softmax)：数值稳定、访存次数、online softmax、向量化加载。
- [Transpose 访存优化](/posts/transpose)：coalescing、shared memory、bank conflict。
- [CUDA 矩阵乘法](/posts/GEMM)：tiling、thread tile、向量化加载、双缓冲。
- [Tensor Core GEMM 实践](/posts/tensor_core_gemm)：WMMA、shared-memory 复用、异步流水和 CUTLASS/Triton 对照。
- [Triton GEMM 优化](/posts/triton_gemm)：super-grouping、Tensor Core、FP8。

## CUDA 路线中的知识连接

```text
CUDA execution / memory hierarchy
  -> Reduce / Softmax / Transpose
  -> FP32 CUDA Core GEMM
  -> low precision + Tensor Core MMA
  -> cp.async software pipeline
  -> CUTLASS / Triton
  -> compiler lowering
  -> LLM fused kernels and inference runtime
```

每一层都要同时保留三类证据：

- correctness：reference、误差、边界 shape。
- performance：duration、吞吐、stall、资源占用。
- mechanism：数据流、指令路径、为什么这一优化在当前 shape 生效。

## 和推理系统的连接

- [Flash Attention 原理与实现](/notes/kernels/algorithms/flash_attention)：把 softmax、矩阵乘和 IO-aware 思路放进 attention。
- [LLM 架构基础](/notes/inference/model_architecture)：理解 RMSNorm、RoPE、KV cache 等算子出现的位置。
- [量化笔记](/notes/inference/quantization/)：把 GEMM、反量化和内存带宽问题连起来。

## 复盘问题

- 一个 kernel 是 memory-bound 还是 compute-bound？
- 当前优化是在减少访存、提高并行度，还是提高计算单元利用率？
- Tensor Core 是否真的被使用？证据来自源代码、PTX/SASS 还是 profiler？
- pipeline 增加 stage 后，隐藏的延迟是否大于额外 shared memory/寄存器成本？
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
