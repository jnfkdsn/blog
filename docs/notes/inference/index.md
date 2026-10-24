---
order: 40
title: 模型与推理系统
updated: 2026-10-17
---

# 模型与推理系统

这一部分从模型实际执行的计算出发，逐渐连接缓存、请求调度、算子调用与性能。阅读时始终保留一个具体对象：当前处理哪些 token，张量是什么形状，哪些状态需要保存，结果交给谁。

## 已有内容与阅读顺序

| 入口 | 当前内容 | 阅读作用 |
|---|---|---|
| [LLM 架构基础](./model_architecture) | RMSNorm、RoPE、KV cache 等计算片段 | 认识模型中的计算与状态；尚未完整串起 Decoder 的前向过程 |
| [FlashAttention](/notes/kernels/algorithms/flash_attention) | Attention 的分块、在线计算与访存 | 连接模型计算和算子实现 |
| [nano-vLLM 源码笔记](./nanovllm) | 目录、入口和 Engine 片段 | 提供系统阅读入口；尚不是完整的请求生命周期教程 |
| [低比特量化](./quantization/) | 格式、粒度、执行、训练、算法与评测 | 在模型执行基础上理解低精度带来的变化 |

先认识模型计算与 KV cache，再读算子和引擎。量化可以作为一条独立深化支线，不要求完成全部量化算法后才接触推理系统。

## 后续扩展范围

后续需要补齐完整的 Decoder 执行、prefill/decode 对照、批处理与请求调度、分页 KV cache，以及 vLLM 与编译器的衔接。分布式推理、MoE 和通信优化在单设备执行链清楚后展开。本页记录范围，尚未展开的主题不创建空章节。

## 与编译器和算子的连接

- [PyTorch Compiler](/notes/compile/pytorch-compiler/)：从模型程序到计算图、约束与生成代码。
- [算子编程与优化](/notes/kernels/)：从实际形状分析计算、访存、启动与资源开销。
- [Softmax 实践](/posts/softmax)、[CUDA GEMM](/posts/GEMM)与 [Triton GEMM](/posts/triton_gemm)：观察具体实现与测量。

阅读和实验时，可以持续检查三个问题：瓶颈出现在哪个执行阶段；主要时间花在计算、数据读取还是调度；某项优化减少了什么成本，又新增了什么成本。
