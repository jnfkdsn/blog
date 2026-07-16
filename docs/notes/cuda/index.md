---
order: 0
---

# CUDA 学习笔记

记录 CUDA / AI Infra 学习过程中的长期知识。这里主要沉淀基础概念、工程化方法和性能分析工具，具体算子实现放在 [实践记录](/posts/) 中。

## 推荐路径

1. [C++ 前置知识](/notes/cuda/cpp)
2. [CMake 构建实践](/notes/cuda/cmake)
3. [CUDA 基础语法](/notes/cuda/cuda_basic_syntax)
4. [Roofline 分析](/notes/cuda/roofline)
5. [低精度数值与混合精度计算](/notes/cuda/low_precision)
6. [Tensor Core 编程](/notes/cuda/tensor_core)
7. [Ampere 异步拷贝与软件流水线](/notes/cuda/async_pipeline)

前四篇建立 CUDA/SIMT 与性能分析基础；后三篇把普通 CUDA Core GEMM 推进到低精度、warp-level MMA 和异步数据流水。

## 相关地图

- [GPU 编程与算子优化知识地图](/notes/gpu-programming)

## 相关实践

- [Reduce 优化实践](/posts/reduce)
- [Softmax 算子实现与优化](/posts/softmax)
- [CUDA 矩阵乘法](/posts/GEMM)
- [Tensor Core GEMM 实践](/posts/tensor_core_gemm)

## 后续进阶路线

完成 Tensor Core GEMM 闭环后，再按当前 AI Infra 主线扩展：

1. RMSNorm、RoPE、SwiGLU：练习 reduction、layout 和 epilogue fusion。
2. Quantized GEMM：连接低精度格式、scale、反量化和 Tensor Core。
3. Paged Attention / KV Cache：连接 kernel、显存布局和推理系统。
4. Stream、Event、CUDA Graph：连接 kernel 与推理 runtime。
5. NCCL / Tensor Parallel：在单 GPU 性能模型清楚后再进入多 GPU。
