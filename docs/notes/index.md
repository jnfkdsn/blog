---
order: 0
---

# 学习笔记

这里是知识库入口。内容按“知识地图 → 领域笔记 → 实践记录”的方式组织：地图负责串联路线，领域笔记沉淀概念，实践记录保留实现和性能复盘。

## 知识地图

- [GPU 编程与算子优化知识地图](/notes/gpu-programming)：C++、CUDA、Triton、性能分析和典型算子优化的主线。
- [LLM 推理系统知识地图](/notes/llm-inference)：模型结构、KV cache、量化、推理引擎和底层 kernel 的连接。

## 领域笔记

- [CUDA 学习笔记](/notes/cuda/)：线程模型、内存层次、同步、构建系统和性能分析。
- [Triton 学习笔记](/notes/triton/)：Triton 编程模型、softmax、GEMM、autotune。
- [编译器学习笔记](/notes/compile/)：传统编译器基础、IR/SSA、优化 pass、后端和 AI Compiler 衔接。
- [Ascend C 算子编程知识库](/notes/CANN/Ascend_C)：AI Core 编程模型、片上流水、tiling、调试和算子优化清单。
- [推理系统笔记](/notes/infer/)：LLM 架构、Flash Attention、量化、nano-vLLM 源码阅读。

## 实践复盘

- [实践记录](/posts/)：把一次次 kernel 实现、性能分析和踩坑记录沉淀下来。
- [项目实战](/projects/)：后续放完整项目路线、设计文档和阶段复盘。
