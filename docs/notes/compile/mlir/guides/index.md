---
order: 40
title: 使用指南
updated: 2026-09-06
excludeFromSidebar: true
---

# 使用指南

本目录按可复用的工具任务组织内容：怎样观察、验证、查证和定位问题。机制本身的原理链接回 `core/`、`dialects/` 或 `compiler/`，不在每篇指南中重写一套定义。

当前：[阅读、验证 IR 与定位源码](./inspecting_ir)，覆盖 generic print、diagnostics、pipeline、固定版本查证和文档示例复现。

后续单独展开调试与最小复现、lit/FileCheck、数据/性能验证、bindings 与工具集成，见[覆盖表 X01—X07](../coverage)。当一篇指南需要反复解释新的机制时，应先把该机制归档，再引用它。
