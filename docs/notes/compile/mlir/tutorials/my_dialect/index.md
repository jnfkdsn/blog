---
order: 30
title: 从 my.add 开始开发 Dialect
updated: 2026-09-24
excludeFromSidebar: true
---

# 从 my.add 开始开发 Dialect

这组教程用一个逐步扩展的工程，把已经读过的 IR 定义机制连成可观察的过程。现有 `compiler/ir_definition/` 继续作为机制参考；这里每次只增加一项工程能力。

当前从 [01：让自己的工具认识 my.add](./01_minimal_dialect) 开始。它解释两条链：构建时，ODS 怎样生成并接入 C++；运行时，工具怎样找到操作定义并构造、验证和打印 IR。

| 阶段 | 产物 | 当前状态 |
|---|---|---|
| 01：最小工程 | 能读写 my.add 的工具；字段改名观察任务 | 已提供参考实现与任务，学习者任务待完成 |
| 02：接回变换 | my.add → arith.addi 的 Pattern/Pass | 后续按进度编写 |
| 后续增量 | builder、验证、Trait/Interface，再到 Type/Attr/Region | 随具体需求展开，不一次性交付全部机制 |

[Lab 入口](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)保存工程、命令和任务；blog 解释机制与实际观察。参考版本按 `stages/` 留存，学习者在 `work/` 修改，生成物留在 workspace 的 artifacts。

每步采用“读本步讲解 → 运行观察 → 完成小修改 → 解释结果”的顺序。必要的外壳由作者提供，学习者逐步接手核心逻辑。读完文档、作者工程通过和独立完成任务分别记录。
