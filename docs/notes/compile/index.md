---
order: 20
title: 编译器
updated: 2026-10-17
---

# 编译器

编译器把程序中的计算与约束转换成适合目标执行的表示。本部分先建立 IR、分析和变换的共同基础，再进入具体系统，观察每种系统保留了什么信息、选择了什么实现，以及产物怎样被调用。

[技术全景](./stack)负责解释系统定位；[知识库学习路线](/notes/roadmap)连接编译器、算子和推理系统；[长期能力与项目](./capabilities)提供深化方向和实践标准。

## 基础与框架

| 模块 | 主要内容 | 适合怎样使用 |
|---|---|---|
| [传统编译器](./traditional/) | 前端、IR/SSA/CFG、分析、循环优化、代码生成 | 建立基础，或遇到具体机制时回顾 |
| [AI 编译基础](./ai-basics/) | 计算图、shape/layout、融合、lowering 与运行时 | 将传统概念连接到张量程序 |
| [MLIR](./mlir/) | 多层 IR、操作定义、Pass、转换、内存与目标后端 | 学习可复用的编译器构造机制 |

## 具体编译系统

| 系列 | 贯穿主线 |
|---|---|
| [TVM](./tvm/) | 模型导入 → Relax → TensorIR → 调度与实现选择 → 运行 |
| [PyTorch Compiler](./pytorch-compiler/) | Python 捕获 → 图与约束 → Inductor → 代码生成、部署与训练 |
| [Triton / Triton-Ascend](./triton/) | 张量块程序 → IR 与布局 → 目标编译 → 加载和启动 |
| [NPU IR 与融合](./npu-ir/) | 图语义与元信息 → 融合合法性与收益 → NPU 表示和执行约束 |

已有 MLIR 基础时，可以先分别阅读 TVM、PyTorch Compiler 的第一章，再选择一套走通。两套都观察过以后，用[系统对照](./compiler_comparison)比较图层、算子实现和运行时边界，随后进入 Triton 或具体图优化任务。它们是不同系统，不是顺次串联的一条工具链。

## 跨领域连接

- [算子编程与优化](/notes/kernels/)提供算法、访存和硬件执行直觉，帮助判断变换是否值得。
- [模型与推理系统](/notes/inference/)提供实际形状、数据布局、调用频率和运行时约束。
- [C++ 工程基础](/notes/cpp/)用于理解对象生命周期、生成代码、注册与构建，不要求先通读全部内容。

系列的版本与验证范围见各自首页；[TVM / PyTorch 覆盖表](./compiler_courses_coverage)和 [MLIR 覆盖表](./mlir/coverage)用于查内容范围，不替代阅读路线。
