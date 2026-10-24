---
order: 20
title: Ascend C 算子编程
updated: 2026-10-17
---

# Ascend C 算子编程

Ascend C 把 AI Core 的计算、片上存储、数据搬运与流水协作暴露给算子开发者。这组笔记沿“数据放在哪里 → 由谁计算 → 怎样搬运和同步 → 怎样验证”组织，先建立执行模型，再进入一个完整 Vector 算子。

## 阅读顺序

| 章节 | 主要内容 |
|---|---|
| [AI Core 架构与执行模型](./architecture) | Scalar、Vector、Cube、存储层次与核间分工 |
| [Tensor、存储与流水资源](./storage) | GlobalTensor、LocalTensor、TPipe、TQue、TBuf 与 TPosition |
| [Vector 算子与 Host Tiling](./vector_kernel) | CopyIn/Compute/CopyOut、分块参数与工程流程 |
| [算子分类与实现模式](./operator_patterns) | Elementwise、Reduce、Matmul 与融合计算 |
| [性能分析、正确性与实践](./performance) | 资源与流水检查、数值验证、profiling 和练习顺序 |

这些章节按主题整理自原 Ascend C 笔记，保留其代码和解释。原示例版本为 CANN 社区版 8.3.RC1.alpha001；实际工程以本机 CANN、芯片和 API 约束为准，目录整理不代表示例已在新的设备或版本上验证。

## 相关入口

[CUDA](../cuda/)提供另一种硬件编程视角；[Triton-Ascend 编译路径](/notes/compile/triton/ascend_compiler_path)解释块级程序如何进入目标编译链；[NPU IR](/notes/compile/npu-ir/)关注表示、融合与 lowering。它们分别解释程序表达、编译处理和目标执行的不同部分。
