---
order: 50
title: PyTorch Compiler
updated: 2026-11-23
---

# PyTorch Compiler

`torch.compile` 将 PyTorch 程序中的张量计算交给编译器优化，再把编译好的实现用于后续调用。这一组教程以 `relu(x @ weight + bias)` 为主要模型，依次解释图捕获、图变换、代码生成和执行，随后展开部署与训练。

## 编译与执行的完整流程

这套系统首先要从 Python 执行过程中提取张量计算，再将它编译成高效的实现。**Dynamo** 捕获运算及其依赖，得到 FX 图；**AOTAutograd** 处理原地修改和共享存储关系，并在训练时准备前向、反向图；**Inductor** 决定这些计算怎样执行，选择已有库或生成新的计算代码。

编译好的计算代码还需要与用户调用配合：准备参数和存储，执行计算，更新原函数会修改的输入，返回结果。这些工作由生成的调用代码和 AOT 执行包装共同完成。下面以 PyTorch 2.7.1 的默认 Inductor 后端为例，展开一次新编译过程。

```text
调用 Python 模型，传入张量
    ↓
Dynamo 图捕获
    跟踪 Python 执行逻辑，记录张量操作及其依赖
    根据输入的形状、类型等信息推导中间张量的信息
    生成 guards，记录复用编译结果时需要检查的条件
    ↓
FX GraphModule 与示例输入信息
    ↓
Inductor 后端入口 compile_fx
    调用 AOTAutograd，并提供用于编译处理后计算图的回调
    ↓
AOTAutograd 处理计算图
    分析输入修改、共享存储和返回值关系
    将原地更新改写为产生新值的计算，让后续操作读取新值
    将选定的复杂算子分解为较基础的 ATen 运算
    ├─ 推理：得到用于推理的计算图
    └─ 训练：得到前向、反向图，确定反向需要的中间值
    同时记录执行时怎样更新输入、返回共享存储的张量
    ↓ 调用相应的编译回调
Inductor 核心编译部分
    将图中的运算表示为逐元素计算、归约和存储访问等
    根据依赖安排执行顺序，合并适合一起执行的计算
    决定哪些中间结果写入存储，哪些计算调用已有库
    为选定的计算生成内核
    ├─ CPU：常见路径是生成 C++，再编译为本地代码
    └─ GPU：常见路径是生成 Triton，再由其编译器生成设备代码
    ↓
生成调用代码与执行包装
    准备参数和存储，调用编译好的内核或已有库
    按原函数的行为更新输入，返回结果及共享存储的视图
    训练时连接前向执行与后续的反向计算
    ↓
执行本次调用
    将真实数据传入编译好的程序，完成计算并返回结果
    后续调用通过 guards 检查时，直接复用这个程序
```

图中两次出现 Inductor，分别对应后端入口和具体编译工作。入口 `compile_fx` 先把 FX 图交给 AOTAutograd 处理，再通过回调编译处理后的图。这样，输入更新、视图和求导关系先得到整理，后续编译就可以围绕图中的计算依赖安排融合、存储和代码生成。第四章解释前一部分，第五、六章展开后一部分。对应调用可在 [compile_fx 源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/compile_fx.py) 中查看。

这些编译工作通常在首次调用时发生。后续调用先通过 guards 检查已有实现是否适用，适用时便用本次输入执行它。一个 Python 函数发生断图时，会分成多个图段分别编译；一个图段也可以针对不同输入条件保存多个编译版本。第三章解释图段、版本和缓存怎样配合。

第二章还会使用自定义的 `inspect_backend` 观察 Dynamo 交出的图。这个后端打印 FX 图后返回 `gm.forward`，执行时沿用图中已有的 PyTorch 操作。它便于单独研究图捕获；上图中的优化与代码生成则由默认的 Inductor 后端完成。

部署时还可以提前导出模型，通过 AOTInductor 生成可装载的程序，第七章介绍这条流程。第八章则展开训练中的前向、反向图，以及中间值的保存和重算。

## 阅读顺序

| 章节 | 跟踪的变化 |
|---|---|
| [1. 从 PyTorch 函数到编译执行](/notes/compile/pytorch-compiler/01_execution) | 一行模型怎样变成库调用、生成代码和可复用程序 |
| [2. Dynamo 图捕获与 FX 数据流](/notes/compile/pytorch-compiler/02_capture) | Python 求值怎样变成节点、元数据与后端交接 |
| [3. 特化、动态形状与编译缓存](/notes/compile/pytorch-compiler/03_specialization) | batch 变化怎样影响 guards、复用与重新编译 |
| [4. AOT 编译交接、函数式化与算子分解](/notes/compile/pytorch-compiler/04_normalization) | 从原地更新的可观察结果出发，解释规范化及AOT交接 |
| [5. Inductor 的表达式、存储与 Scheduler 融合](/notes/compile/pytorch-compiler/05_inductor_ir) | 表达式融合→物化→实际scheduler依赖与分组 |
| [6. 目标代码、执行包装与编译器修改](/notes/compile/pytorch-compiler/06_codegen) | 从wrapper与内核的完整调用，追到生成代码、问题定位和有限改写 |
| [7. Export 契约与 AOTInductor 部署](/notes/compile/pytorch-compiler/07_export_aot) | 程序与约束怎样变成可加载的目标产物 |
| [8. AOTAutograd 与前向反向图的协作](/notes/compile/pytorch-compiler/08_training) | 损失怎样产生反向图，保存值怎样跨越边界 |
| [9. Inductor 优化与实现](/notes/compile/pytorch-compiler/inductor/) | 按七阶段重新规划机制覆盖；旧阶段教材归档，工具和实验保留 |

第1章用一个具体模型展示编译和调用；第2—4章解释计算图怎样捕获和处理，第5—6章解释图怎样变成可执行代码，并练习修改和排查问题。第7—8章分别展开部署与训练，可按任务选读。

## 阅读目标与共同基础

学完本系列，应能从一段 PyTorch 程序追踪到 FX 图、Inductor 的计算表示和生成代码，解释一次融合或存储安排，并完成一项图改写。

熟悉 PyTorch 张量操作以及 IR、Pass 基本概念的读者可以从第1章开始，通用概念可结合 `ai-basics` 查阅。正文以矩阵乘法加偏置和 ReLU 为主要模型，也用逐元素运算解释融合，用 Softmax 解释归约。

正文包含理解过程所需的代码、图和结果；labs 提供可运行的参考工程。实践时可以先观察图和生成代码，再修改其中一项规则，检查数值结果、输入更新和输出存储关系。

## 版本与实验范围

本系列使用 PyTorch **2.7.1+cpu**，源码 commit `e2d141dbde55c2a4370fac5165b0561b6af4798b`。代码示例、源码位置和生成结果均按该版本组织。

CPU 实验已覆盖实际 Inductor 推理、动态 batch、非连续输入、逐元素双输出的 Scheduler 融合、有限 FX 改写、稳定 Softmax、AOT 包加载，以及前向、反向梯度。观察后端实验用于查看 Dynamo 输出的 FX 图。[Inductor 优化与实现专题](/notes/compile/pytorch-compiler/inductor/)固定使用独立 GPU 环境（PyTorch `2.14.1+cu126`、Triton `3.8.0`、RTX 4060），实测物化阈值、Scheduler 融合、Triton 内核与有效带宽。

| 能力 | 课程提供的基础 | 后续可深入的问题 |
|---|---|---|
| 捕获与形状 | 能解释图、guards、符号维度和断图 | 新语法/自定义操作捕获、复杂数据相关形状 |
| 图变换 | 理解状态、别名、分解和后端契约 | 有限FX优化已提供参考；复杂alias、训练与更多操作另学 |
| 算子实现 | 追踪索引、realization、scheduler前后与实际代码 | Triton kernel、特定硬件调度与编译器优化 |
| 运行与部署 | 理解缓存、wrapper、导出契约与AOT产物 | 服务集成、设备资源与部署兼容性 |
| 训练 | 推导并观察前向/反向和保存值 | 优化器编译、重算策略、分布式训练 |

可复现实验、练习和精确环境见 [PyTorch Compiler labs](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。与其他系统的关系从 [AI Compiler 总入口](/notes/compile/)继续。
