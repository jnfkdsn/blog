---
order: 0
title: 编译器学习笔记
updated: 2026-07-01
tags: [compiler, ai-compiler, ir, optimization]
status: draft
---

# 编译器学习笔记

这组笔记负责传统编译器基础和 AI Compiler 的衔接。传统编译器部分先覆盖前端、中端、后端、runtime 的骨架；AI Compiler 部分后续再单独展开 PyTorch compile、FX/Inductor、MLIR/TVM、Triton lowering、dynamic shape、layout 和 fusion。

## 传统编译器

- [编译器基础知识地图](/notes/compile/compiler_basic)：整体 pipeline 和学习顺序。
- [前端：Lexer、Parser、AST、语义分析](/notes/compile/frontend_ast_sema)：源码如何变成 typed AST。
- [IR、CFG、SSA](/notes/compile/ir_ssa_cfg)：中端表示、控制流、use-def、phi、dominance。
- [Dataflow Analysis 与 Pass Pipeline](/notes/compile/dataflow_pass)：分析、优化 pass、pass manager、测试方式。
- [Lowering、Codegen、Runtime](/notes/compile/lowering_codegen_runtime)：后端下沉、寄存器分配、调用约定、解释器和 JIT。
- [循环优化](/notes/compile/loop_optimization)：LICM、unroll、tiling、interchange、dependence analysis、vectorization。

## 和 AI Compiler 的连接

| 传统编译器基础 | AI Compiler 继续学习 |
|---|---|
| AST / typed AST | FX Graph、op schema、metadata |
| IR / SSA / CFG | FX IR、Inductor IR、MLIR dialect、StableHLO |
| dataflow analysis | shape propagation、layout propagation、alias/side-effect analysis |
| pass pipeline | graph rewrite、fusion、decomposition、constant folding、DCE |
| lowering / codegen | Triton lowering、LLVM lowering、CUDA/Ascend C kernel 生成 |
| runtime | memory planner、JIT cache、stream、graph executor、shape guard |

## 实践路线

1. 用一个 toy language 实现前端：整数、变量、表达式、if/while。
2. Lower 到三地址 IR，打印 basic block。
3. 加 SSA 或 pseudo-SSA，维护 use-def。
4. 实现 constant folding、DCE、CSE、CFG simplification。
5. 实现一个 IR interpreter。
6. 再进入 PyTorch 2.x compile pipeline，观察真实 AI Compiler 如何组织 IR 和 pass。
