---
order: 50
title: 贯通教程
updated: 2026-09-24
excludeFromSidebar: true
---

# 贯通教程

教程用完整案例连接多个知识模块，展示阅读和推理过程。通用定义仍以对应机制章节为准，教程不承担整个 MLIR 的完备目录职责。

阶段 A：[从算法读懂 MLIR](./reading_ir)，用动态长度数组的正数累加串起包含结构、SSA、分支/循环、内存效果、算法不变量和实际 SCF-to-CF 输出。

阶段 B：[实现并测试一个小 Pass](./first_pass)，把 Pattern、函数 Pass、注册、MlirOptMain、pipeline 与 lit/FileCheck 连成完整工程。配套代码已验证，阅读后的左侧加零扩展由学习者独立完成。

定义机制导读：[自定义 Dialect 入门](./my_dialect/)，用 my.add 解释定义 → 生成 C++ → 编译注册 → 解析、验证与打印。后续直接进入[定义 IR 抽象](../compiler/ir_definition/)，其中用 clamp 展示更丰富的字段、验证与实际改写，不另建一套重复机制的系列。

前置分别为相应阶段的概念与机制章节，阅读顺序见[学习路径](../learning_path)。后续贯通案例可以连接自定义 Dialect、张量到内存、Softmax 等项目；正式实践在 labs 中记录环境、命令、判据与结果。
