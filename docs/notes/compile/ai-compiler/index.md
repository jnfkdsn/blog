---
order: 0
title: AI Compiler
updated: 2026-07-03
tags: [ai-compiler, torch-compile, inductor, mlir, tvm]
status: draft
---

# AI Compiler

相关入口：[编译器学习笔记](/notes/compile/) / [传统编译器](/notes/compile/traditional/)

AI Compiler 处理的是 tensor program 的捕获、表示、分析、优化、lowering 和运行时执行。和传统编译器相比，它的输入更多是 graph/tensor IR，优化目标也从标量指令效率扩展到 shape、layout、memory traffic、kernel launch、device runtime 和硬件层次结构。

```text
model / python function / graph
  -> capture / import
  -> graph IR
  -> metadata inference
  -> graph rewrite / fusion
  -> tensor IR / loop IR
  -> target lowering
  -> codegen / library call
  -> runtime execution
```

## 总体大纲

| 模块 | 核心问题 | 当前状态 |
|---|---|---|
| Capture / Import | Python、框架 graph、ONNX/HLO 如何进入编译器 | 基础展开 |
| Graph IR | op、tensor value、use-def、metadata、side effect 如何表示 | 基础展开，NPU IR 深入 |
| Metadata Inference | shape、dtype、layout、format、alias 如何推导 | 基础展开，NPU IR 深入 |
| Graph Rewrite | decomposition、canonicalization、DCE、CSE、layout rewrite | 基础展开，结合 fusion 深入 |
| Fusion | 哪些 op 能融合，融合后是否值得，如何改写 IR | 重点展开 |
| Tensor IR / Schedule | loop nest、tiling、vectorization、buffer、memory scope | 基础展开，NPU 约束深入 |
| Lowering | graph/tensor IR 如何下沉到 NPU kernel/library | 基础展开，NPU 约束深入 |
| Runtime | memory planning、workspace、stream/event、cache、guard | 基础展开，融合相关深入 |
| 系统专题 | PyTorch Inductor、MLIR、TVM、XLA | 暂不生成详细内容 |

## AI Compiler 基础

- [AI Compiler 基础](/notes/compile/ai-compiler/basics/)：基础专题入口。
- [AI Compiler Pipeline](/notes/compile/ai-compiler/basics/pipeline)：从模型/graph 到 runtime execution 的整体链路。
- [Graph IR 基础](/notes/compile/ai-compiler/basics/graph_ir_basics)：op、tensor value、metadata、side effect、use-def。
- [Metadata Inference 基础](/notes/compile/ai-compiler/basics/metadata_inference)：shape、dtype、layout、alias 信息如何在 graph 上传播。
- [Graph Rewrite 与 Pass 基础](/notes/compile/ai-compiler/basics/graph_rewrite_pass)：canonicalization、decomposition、DCE、CSE、fusion pass 的位置。
- [Lowering 与 Runtime 基础](/notes/compile/ai-compiler/basics/lowering_runtime)：Graph IR 到 kernel/library/runtime 的基本层次。

## NPU IR 融合主线

- [NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)：NPU IR 融合专题入口。
- [NPU IR 表示](/notes/compile/ai-compiler/npu-ir/graph_ir)：op、tensor value、use-def、side effect。
- [Shape、Dtype、Layout Metadata](/notes/compile/ai-compiler/npu-ir/metadata_shape_layout)：融合依赖的 tensor 元信息。
- [算子融合总览](/notes/compile/ai-compiler/npu-ir/fusion_overview)：融合类型、收益来源、风险。
- [Fusion Legality](/notes/compile/ai-compiler/npu-ir/fusion_legality)：判断能不能融合。
- [Fusion Profitability](/notes/compile/ai-compiler/npu-ir/fusion_profitability)：判断值不值得融合。
- [Fusion Pass 实现](/notes/compile/ai-compiler/npu-ir/fusion_pass_impl)：pattern、rewrite、use-def 更新、测试。
- [Memory、Buffer 与 NPU 存储层次](/notes/compile/ai-compiler/npu-ir/memory_buffer)：GM/UB/L1/L0、buffer lifetime、workspace。
- [NPU Lowering 约束](/notes/compile/ai-compiler/npu-ir/lowering_constraints)：融合结果下沉到 NPU kernel 时的限制。

## 和传统编译器的对应

| 传统编译器 | AI Compiler |
|---|---|
| typed AST | graph metadata、shape/dtype/layout 信息 |
| IR / SSA | FX IR、Inductor IR、MLIR SSA value |
| dataflow analysis | shape propagation、layout propagation、alias/side-effect analysis |
| optimization pass | graph rewrite、fusion、decomposition、DCE、CSE |
| lowering/codegen | Triton/CUDA/C++/Ascend C kernel 生成 |
| runtime | guard、JIT cache、memory planner、graph executor |
