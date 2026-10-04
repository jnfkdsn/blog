---
order: 50
title: 贯通教程
updated: 2026-10-04
excludeFromSidebar: true
---

# 贯通教程

教程用完整案例连接多个知识模块，展示阅读和推理过程。通用定义仍以对应机制章节为准，教程不承担整个 MLIR 的完备目录职责。

阶段 A：[从算法读懂 MLIR](./reading_ir)，用动态长度数组的正数累加串起包含结构、SSA、分支/循环、内存效果、算法不变量和实际 SCF-to-CF 输出。

阶段 B：[实现并测试一个小 Pass](./first_pass)，把 Pattern、函数 Pass、注册、MlirOptMain、pipeline 与 lit/FileCheck 连成完整工程。配套代码已验证，阅读后的左侧加零扩展由学习者独立完成。

自定义操作的接入导读与机制章节已统一收录于[定义 IR 抽象](../compiler/ir_definition/)。需要从最小例子开始时，阅读该模块的 [my.add 导读](../compiler/ir_definition/dialect_basics)。本目录继续收录跨模块贯通案例，不再单独维护 `my_dialect` 系列。

前置分别为相应阶段的概念与机制章节，阅读顺序见[学习路径](../learning_path)。[编译流程导读](./pipelines/overview)已提供，沿矩阵加法连接 Tensor/Linalg、buffer、循环与执行；[Linalg 到 CPU](./pipelines/cpu)进一步连接完整 C 调用、动态布局和数值检查，[真实前端](./pipelines/frontend)已有固定源码篇。[归约和算子贯通](./kernels/)已有 Softmax 两篇，Matmul/Attention三篇也已提供；正式实践在 labs 中记录环境、命令、判据与结果。

[归约与 Softmax 两篇](./kernels/)已提供：先建立稳定算法的 Tensor/Linalg 与 CPU 数值基线，再比较四阶段和按行融合的存储。Matmul/Attention连接调度、存储与在线状态递推。
