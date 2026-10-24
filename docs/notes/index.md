---
order: 0
title: 知识库
---

# 知识库

这里围绕同一件事组织知识：一个模型的计算，怎样经过程序表达、编译优化和运行时组织，在硬件上正确而高效地执行。

第一次进入可以先读[学习路线](./roadmap)，了解各领域如何衔接；已有具体问题时，直接进入对应模块。

## 四个领域

| 领域 | 主要内容 | 阅读入口 |
|---|---|---|
| 编程基础 | C++ 对象、所有权、模板、构建与并发 | [C++ 工程基础](./cpp/) |
| 编译器 | IR、分析与变换、图编译、算子编译、代码生成与运行时边界 | [编译器](./compile/) |
| 算子编程与优化 | CUDA、Ascend C、计算与访存、算子算法和性能分析 | [算子编程与优化](./kernels/) |
| 模型与推理系统 | 模型计算、KV cache、量化、请求执行与推理引擎 | [模型与推理系统](./inference/) |

## 常用系列

- [MLIR](./compile/mlir/)：核心 IR、方言语义、编译机制与贯通案例。
- [TVM](./compile/tvm/)与 [PyTorch Compiler](./compile/pytorch-compiler/)：从同一个小模型观察两套独立的编译系统。
- [Triton 与 Triton-Ascend](./compile/triton/)：块级编程模型与目标编译路径。
- [CUDA](./kernels/cuda/)与 [Ascend C](./kernels/ascend-c/)：两类硬件上的算子表达、数据搬运与执行。
- [模型执行基础](./inference/model_architecture)、[量化](./inference/quantization/)与 [nano-vLLM](./inference/nanovllm)：从模型计算走向推理系统。

## 实践与项目

[实践记录](/posts/)保存具体实现、测量和优化复盘；[项目实战](/projects/)保存项目设计与阶段成果。知识库正文解释可复用的原理，必要的代码和结果在文章内给出，配套实验从各章末尾进入。
