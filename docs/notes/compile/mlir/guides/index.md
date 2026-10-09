---
order: 40
title: 使用指南
updated: 2026-10-04
excludeFromSidebar: true
---

# 使用指南

本目录围绕可复用的工具任务组织：观察编译过程，保留可解释的失败，检验修改，以及把 MLIR 接入自己的工具。关键输入、操作和结果都在正文，labs 提供完整复现。

| 阅读入口 | 要完成的过程 |
|---|---|
| [阅读、验证 IR 与定位源码](./inspecting_ir) | 文本与通用格式、诊断、pipeline 和固定版本查证 |
| [失败 Pipeline 与最小复现](./debugging) | 失败边界→输入和配置重放→保留同一原因的缩减 |
| [回归测试与证据层次](./regression_tests) | 把成功、未匹配和拒绝行为写成可区分的检查 |
| [数值、边界与性能证据](./numerics_performance) | 语义契约→参考和误差→输入边界→计时与归因 |
| [C API 与所有权](./c_api) | 解析→折叠42→构造7→转移拥有关系→释放 |
| [Python 绑定](./python_bindings) | 同一修改过程→插入点→包装与底层生命周期 |
| [Bytecode与方言版本](./bytecode) | 旧bias→版本记录→当前offset→相同计算与拒绝边界 |
| [插件与动态扩展](./plugins) | 将方言和Pass能力加载进宿主，区分IRDL约束定义 |
| [有限源码阅读](./source_reading) | 从x+0变化追ODS、生成类、fold、driver、Conversion和测试 |

[章节目录](../chapter_plan)与[覆盖表](../coverage)记录各项范围。机制的完整解释归入 `core/`、`dialects/` 或 `compiler/`，工具指南通过具体任务连接它们。
