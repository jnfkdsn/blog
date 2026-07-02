---
order: 0
title: AI Compiler
updated: 2026-07-03
tags: [ai-compiler, torch-compile, inductor, mlir, tvm]
status: draft
---

# AI Compiler

相关入口：[编译器学习笔记](/notes/compile/) / [传统编译器](/notes/compile/traditional/)

AI Compiler 后续单独放在这个目录下，避免和传统编译器基础混在同一层。

## 计划中的专题

- PyTorch 2.x compile pipeline：TorchDynamo、FX、AOTAutograd、Inductor。
- Graph IR：FX Graph、node metadata、op schema、decomposition。
- Tensor IR：loop nest、shape、stride、layout、buffer。
- Fusion：producer-consumer fusion、reduction fusion、layout-aware fusion。
- Lowering：ATen -> loop IR -> Triton/C++/CUDA/Ascend C。
- Runtime：guard、JIT cache、memory planning、kernel launch、stream/event。
- MLIR / TVM：dialect、pass pipeline、schedule、codegen。

## 和传统编译器的对应

| 传统编译器 | AI Compiler |
|---|---|
| typed AST | graph metadata、shape/dtype/layout 信息 |
| IR / SSA | FX IR、Inductor IR、MLIR SSA value |
| dataflow analysis | shape propagation、layout propagation、alias/side-effect analysis |
| optimization pass | graph rewrite、fusion、decomposition、DCE、CSE |
| lowering/codegen | Triton/CUDA/C++/Ascend C kernel 生成 |
| runtime | guard、JIT cache、memory planner、graph executor |
