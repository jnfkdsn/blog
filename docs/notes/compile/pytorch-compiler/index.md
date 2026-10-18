---
order: 50
title: PyTorch Compiler
updated: 2026-10-17
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
| [4. 图规范化、原地修改与算子分解](/notes/compile/pytorch-compiler/04_normalization) | 隐含状态怎样变成明确的数据关系 |
| [5. Inductor 的计算、索引与存储表示](/notes/compile/pytorch-compiler/05_inductor_ir) | 张量计算怎样展开成索引表达式和 buffer |
| [6. 融合、调度与生成程序的执行](/notes/compile/pytorch-compiler/06_codegen) | 依赖和代价怎样决定调用、循环和存储 |
| [7. Export 契约与 AOTInductor 部署](/notes/compile/pytorch-compiler/07_export_aot) | 程序与约束怎样变成可加载的目标产物 |
| [8. AOTAutograd 与前向反向图的协作](/notes/compile/pytorch-compiler/08_training) | 损失怎样产生反向图，保存值怎样跨越边界 |

初读先完成第1章；接着第2—4章建立图层，第5—6章深入后端，第7—8章扩展部署与训练。若已经熟悉某个机制，可以带着章末问题跳读，仍建议观察一份完整输出。

## 本轮范围与证据

本系列使用固定 PyTorch **2.7.1+cpu**，源码 commit `e2d141dbde55c2a4370fac5165b0561b6af4798b`。它是可复现的教学基线，不声称是最新版本。精确 API 以固定源码和版本文档为准；新版本应重新核验接口与生成结果。

CPU 实验已覆盖实际 Inductor 推理、动态 batch、非连续输入、稳定 Softmax、AOT 包加载，以及前向/反向梯度。观察后端返回 FX Python 的部分单独标注，不把它当成机器码编译证据。GPU/Triton 路径在本系列说明职责和连接，本机未执行设备程序。

| 能力 | 课程提供的基础 | 后续可深入的问题 |
|---|---|---|
| 捕获与形状 | 能解释图、guards、符号维度和断图 | 新语法/自定义操作捕获、复杂数据相关形状 |
| 图变换 | 理解状态、别名、分解和后端契约 | 实现真实图优化并验证全部语义条件 |
| 算子实现 | 理解索引、布局、realization、融合与代码生成 | Triton kernel、特定硬件调度与编译器优化 |
| 运行与部署 | 理解缓存、wrapper、导出契约与AOT产物 | 服务集成、设备资源与部署兼容性 |
| 训练 | 推导并观察前向/反向和保存值 | 优化器编译、重算策略、分布式训练 |

这些内容提供后续开发的基础，不以读完或运行通过代替独立实现能力。当前阶段的练习是修改有限行为、预测图和代码变化，再验证数值与边界。

可复现实验、练习和精确环境见 [PyTorch Compiler labs](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。与其他系统的关系从 [AI Compiler 总入口](/notes/compile/)继续。
