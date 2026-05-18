---
order: 3
title: LLM 推理系统知识地图
updated: 2026-05-18
tags: [llm, inference, attention, quantization]
status: draft
---

# LLM 推理系统知识地图

这条线负责把模型结构、推理引擎、算子优化和量化串起来。重点是理解一次 token 生成背后的数据流，以及瓶颈如何在显存、带宽、调度和 kernel 之间转移。

## 学习主线

1. 先理解 Transformer 推理时会调用哪些核心模块。
2. 再看 KV cache、attention、GEMM、normalization 的数据流。
3. 继续学习 Flash Attention、分页 KV cache、continuous batching 等系统优化。
4. 最后用量化、FP8、weight-only 等方法理解低比特推理的性能和精度权衡。

## 核心笔记

- [LLM 架构基础](/notes/infer/llm_architechture)
- [Flash Attention 原理与实现](/notes/infer/flash_attention)
- [量化笔记](/notes/infer/quantization)
- [nano-vLLM 源码阅读](/notes/infer/nanovllm)

## 相关算子

- [Softmax 算子实现与优化](/posts/softmax)
- [CUDA 矩阵乘法](/posts/GEMM)
- [Triton GEMM 优化](/posts/triton_gemm)
- [Roofline 分析](/notes/cuda/roofline)

## 复盘问题

- 推理瓶颈当前在 prefill 还是 decode？
- 主要受限于 GEMM 计算、KV cache 带宽，还是调度开销？
- 量化减少了哪些数据搬运？又引入了哪些反量化开销？
