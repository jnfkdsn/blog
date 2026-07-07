---
order: 0
title: AI Compiler 基础
updated: 2026-07-05
tags: [ai-compiler, graph-ir, lowering, runtime]
status: draft
---

# AI Compiler 基础

相关入口：[AI Compiler](/notes/compile/ai-compiler/) / [传统编译器](/notes/compile/traditional/)

AI Compiler 基础层负责建立进入 NPU IR 融合前需要的概念：

```text
capture/import
  -> graph IR
  -> metadata inference
  -> graph rewrite pass
  -> tensor IR / lowering
  -> codegen / runtime
```

这些内容不绑定具体系统。PyTorch Inductor、MLIR、TVM、XLA、NPU 厂商编译器都会以不同形式实现这些阶段。

## 笔记

- [AI Compiler Pipeline](/notes/compile/ai-compiler/basics/pipeline)：整体编译链路和每一层的输入输出。
- [Graph IR 基础](/notes/compile/ai-compiler/basics/graph_ir_basics)：op、tensor value、use-def、metadata、side effect。
- [Metadata Inference 基础](/notes/compile/ai-compiler/basics/metadata_inference)：shape、dtype、layout、alias 信息如何支撑优化。
- [Graph Rewrite 与 Pass 基础](/notes/compile/ai-compiler/basics/graph_rewrite_pass)：decomposition、canonicalization、DCE、CSE、fusion pass。
- [Lowering 与 Runtime 基础](/notes/compile/ai-compiler/basics/lowering_runtime)：从 graph/tensor IR 到 kernel/library/runtime。

## 与 NPU IR 融合的关系

| 基础概念 | NPU IR 融合里的用途 |
|---|---|
| Graph IR | 找 producer-consumer 子图，改写 fused op |
| Metadata | 判断 shape/dtype/layout 是否可融合 |
| Rewrite Pass | 实现 pattern matching、rewrite、DCE、verify |
| Tensor IR / Lowering | 判断 fused op 是否能生成 NPU kernel |
| Runtime | 评估 memory planning、workspace、kernel launch、cache |
