---
order: 0
---

# 推理系统笔记

这里沉淀 LLM 推理相关的长期知识：模型结构、Attention、KV cache、量化、推理引擎源码阅读，以及它们和底层算子优化之间的关系。

## 推荐路径

1. [LLM 架构基础](/notes/infer/llm_architechture)
2. [Flash Attention 原理与实现](/notes/infer/flash_attention)
3. [量化笔记](/notes/infer/quantization)
4. [nano-vLLM 源码阅读](/notes/infer/nanovllm)

## 和算子优化的连接

- [Softmax 算子实现与优化](/posts/softmax)：Flash Attention 中 online softmax 的基础。
- [CUDA 矩阵乘法](/posts/GEMM)：prefill 和 decode 中 GEMM 性能分析的基础。
- [Triton GEMM 优化](/posts/triton_gemm)：理解 Triton 在推理算子中的使用方式。
- [Roofline 分析](/notes/cuda/roofline)：判断推理 kernel 的瓶颈来源。

## 相关地图

- [LLM 推理系统知识地图](/notes/llm-inference)
- [GPU 编程与算子优化知识地图](/notes/gpu-programming)
