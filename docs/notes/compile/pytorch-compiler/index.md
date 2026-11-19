---
order: 50
title: PyTorch Compiler
updated: 2026-10-30
---

# PyTorch Compiler

这一组教程沿 `relu(x @ weight + bias)` 解释 PyTorch 程序怎样进入编译器，怎样选择并生成实现，最后怎样作为推理或训练程序执行。以推理为主，训练篇补上前向/反向图的关系；既关心图捕获、形状和部署，也深入算子索引与生成代码。

阅读时把整条路径保留在脑中，再按章节展开其中一段：

```text
Python 模型与输入条件
  → Dynamo 捕获 FX 图，并保护捕获假设
  → 整理算子、可变状态与调用协议
  → Inductor 表达计算、选择实现、组织融合和调度
  → 库调用 + 生成内核 + 执行包装
  → 检查适用条件、执行并返回结果
```

这是一条典型的 `torch.compile` / Inductor 主线。Export/AOT 部署从显式导出契约进入；训练需要额外的前向/反向图组织。它们在相关章节分别连接，不画成所有程序都必须依次通过的单一流程。

## 阅读顺序

| 章节 | 跟踪的变化 |
|---|---|
| [1. 从 PyTorch 函数到编译执行](/notes/compile/pytorch-compiler/01_execution) | 一行模型怎样变成库调用、生成代码和可复用程序 |
| [2. Dynamo 图捕获与 FX 数据流](/notes/compile/pytorch-compiler/02_capture) | Python 求值怎样变成节点、元数据与后端交接 |
| [3. 特化、动态形状与编译缓存](/notes/compile/pytorch-compiler/03_specialization) | batch 变化怎样影响 guards、复用与重新编译 |
| [4. AOT 编译交接、函数式化与算子分解](/notes/compile/pytorch-compiler/04_normalization) | Dynamo后的AOT分派、mutation/alias契约与分解 |
| [5. Inductor 的表达式、存储与 Scheduler 融合](/notes/compile/pytorch-compiler/05_inductor_ir) | 表达式融合→物化→实际scheduler依赖与分组 |
| [6. 目标代码、执行包装与编译器修改](/notes/compile/pytorch-compiler/06_codegen) | CPU代码与Triton映射；诊断和真实FX后端改写 |
| [7. Export 契约与 AOTInductor 部署](/notes/compile/pytorch-compiler/07_export_aot) | 程序与约束怎样变成可加载的目标产物 |
| [8. AOTAutograd 与前向反向图的协作](/notes/compile/pytorch-compiler/08_training) | 损失怎样产生反向图，保存值怎样跨越边界 |

初读先完成第1章；接着第2—4章建立图层，第5—6章深入后端，第6章同时完成诊断与有限修改，第7—8章是部署与训练专题，可按任务选读。若已经熟悉某个机制，可以带着章末问题跳读，仍建议观察一份完整输出。

## 阅读目标与共同基础

本系列要求能够解释设计选择、沿实际表示追踪一次优化，并在参考工程上修改有限行为。每章先建立具体任务，再解释决定使用哪些事实、改变了什么、怎样验证；源码只选能确认该决定的入口，不以接口数量衡量深度。

`ai-basics` 可按需查阅，不是必须先读完的前置课程。已学过IR、SSA、Pass的读者直接沿本系列前进；影响当前结论的通用原理会在正文中补齐。主要模型仍是矩阵乘法加偏置和ReLU；逐元素链用于隔离融合，Softmax用于观察归约，换例时说明要隔离的问题。

正文独立提供关键代码、IR、状态变化和结果。labs中的参考观察用于复现和查证，练习只选择少量有边界的修改；运行参考脚本不等于独立实现，也不要求每章新增练习。

## 本轮范围与证据

本系列使用固定 PyTorch **2.7.1+cpu**，源码 commit `e2d141dbde55c2a4370fac5165b0561b6af4798b`。它是可复现的教学基线，不声称是最新版本。精确 API 以固定源码和版本文档为准；新版本应重新核验接口与生成结果。

CPU 实验已覆盖实际 Inductor 推理、动态 batch、非连续输入、逐元素双输出的scheduler融合、有限FX改写、稳定 Softmax、AOT 包加载，以及前向/反向梯度。观察后端返回 FX Python 的部分单独标注，不把它当成机器码编译证据。GPU/Triton 路径提供有明确标注的代码推演和独立观察入口；固定CPU环境检查为SKIPPED，本机未执行设备程序。

| 能力 | 课程提供的基础 | 后续可深入的问题 |
|---|---|---|
| 捕获与形状 | 能解释图、guards、符号维度和断图 | 新语法/自定义操作捕获、复杂数据相关形状 |
| 图变换 | 理解状态、别名、分解和后端契约 | 有限FX优化已提供参考；复杂alias、训练与更多操作另学 |
| 算子实现 | 追踪索引、realization、scheduler前后与实际代码 | Triton kernel、特定硬件调度与编译器优化 |
| 运行与部署 | 理解缓存、wrapper、导出契约与AOT产物 | 服务集成、设备资源与部署兼容性 |
| 训练 | 推导并观察前向/反向和保存值 | 优化器编译、重算策略、分布式训练 |

这些内容提供后续开发的基础，不以读完或运行通过代替独立实现能力。当前阶段的练习是修改有限行为、预测图和代码变化，再验证数值与边界。

可复现实验、练习和精确环境见 [PyTorch Compiler labs](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。与其他系统的关系从 [AI Compiler 总入口](/notes/compile/)继续。
