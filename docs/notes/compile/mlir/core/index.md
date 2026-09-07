---
order: 10
title: 核心概念
updated: 2026-09-06
excludeFromSidebar: true
---

# 核心概念

本目录定义跨方言共用的对象、引用与语义边界。它回答“任何一个 MLIR 方言都要建立在哪些机制上”，操作的特殊规则在 `dialects/` 查询。

| 章节 | 内容 | 前置 |
|---|---|---|
| [IR 对象模型与表示](./ir_model) | 文本/内存/序列化、递归容器、Region 类别、Op 包装与源码 | 简单程序阅读 |
| [Value、SSA 与支配](./values_ssa) | 定义/使用、Block 参数、合流/回边、层次支配 | 对象模型 |
| [类型、Attribute 与 Properties](./types_attributes) | 值与静态信息、形状、类型转换、位置 | Value/SSA |
| [符号、作用域与隔离](./symbols_scopes) | SymbolTable、可见性、SSA 捕获与符号引用 | 前三章 |
| [内存、效果与优化边界](./effects) | Tensor/MemRef、别名、效果、UB、推测执行 | 前四章及 arith/SCF 基础 |

建议先读前四章，进入基础方言和 SCF，再回读效果模型。`core/` 的归档位置不代表它的全部章节必须连续读完。完整次序见[学习路径](../learning_path)，目标与缺口见[覆盖表 C01—C11](../coverage)。
