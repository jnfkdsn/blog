---
order: 0
title: 编译器学习笔记
updated: 2026-07-01
tags: [compiler, ai-compiler, ir, optimization]
status: draft
---

# 编译器学习笔记

这组笔记分成两条线：传统编译器基础和 AI Compiler。传统编译器负责建立 IR、CFG、SSA、pass、lowering、runtime 等底层概念；AI Compiler 负责把这些概念映射到 tensor graph、shape/layout、fusion、kernel lowering 和 runtime execution。

## 目录

- [传统编译器](/notes/compile/traditional/)：前端、中端、后端、runtime 的基础机制。
- [AI Compiler](/notes/compile/ai-compiler/)：PyTorch compile、FX/Inductor、MLIR/TVM、Triton lowering、dynamic shape、layout、fusion。

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
