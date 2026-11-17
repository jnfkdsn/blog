---
order: 20
title: AI Compiler 基础
updated: 2026-10-30
tags: [ai-compiler, graph-ir, lowering, runtime]
status: draft
---

# AI Compiler 基础

相关入口：[编译器](/notes/compile/) / [传统编译器](/notes/compile/traditional/)

一个张量表达式交给编译器后，为什么还要经历图表示、信息推导、程序变换和实现选择？这个模块用小例子解释这些共同问题，帮助读者在学习不同编译器时辨认：当前处理的是什么、依据哪些信息作决定、结果交给谁。

例如，`y = Relu(x + b)` 规定了计算结果，却没有指定加法与激活是否分开执行、中间结果是否需要存储，以及生成的代码由谁调用。编译器逐步补全这些决定，同时保留原程序要求的行为。下面五篇沿这条关系展开，代码均在文内解释，不要求先运行实验工程。

## 阅读顺序

| 章节 | 要建立的认识 |
|---|---|
| [AI Compiler Pipeline](./pipeline) | 从张量表达式到可调用程序，各阶段解决什么问题 |
| [Graph IR 基础](./graph_ir_basics) | 操作、值和使用关系怎样表示计算，为什么值与存储要分开 |
| [Metadata Inference 基础](./metadata_inference) | 如何从已知输入推导输出信息，未知信息怎样限制优化 |
| [Graph Rewrite 与 Pass 基础](./graph_rewrite_pass) | 如何根据证据修改程序，区分合法性、收益和结构校验 |
| [Lowering 与 Runtime 基础](./lowering_runtime) | 怎样选择实现、安排存储，并把编译产物交给运行时执行 |

这些章节提供通用认识，不规定所有编译器必须采用同一条流水线。某些系统从模型图开始，某些系统直接接收张量块程序；信息推导和改写也可能在不同层次反复进行。

## 与具体系统的分工

这里解释问题与基本推理，系统章节进一步解释实现选择、接口和可修改的位置：

- [TVM](../tvm/)：Relax 与 TensorIR 怎样分工，调度程序怎样改变计算，以及如何选择实现。
- [PyTorch Compiler](../pytorch-compiler/)：怎样捕获 Python 中的计算、管理特化条件，并通过 Inductor 生成代码。
- [MLIR](../mlir/)：怎样构造 IR、定义操作、编写变换和完成多层转换。
- [Triton / Triton-Ascend](../triton/)：怎样把张量块程序转换成目标设备上的执行。
- [NPU IR 与融合](../npu-ir/)：通用的依赖、存储和变换问题，在具体 NPU 后端约束下怎样落实。

第一次接触 AI 编译器，可以顺序阅读这五篇。已经在学习具体系统时，可按遇到的问题回查；这里不构成必须全部读完才能继续系统课程的额外关卡。
