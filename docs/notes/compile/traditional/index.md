---
order: 0
title: 传统编译器
updated: 2026-07-03
tags: [compiler, ir, ssa, optimization]
status: draft
---

# 传统编译器

相关入口：[编译器学习笔记](/notes/compile/)

传统编译器部分覆盖从源码到可执行代码的基本链路：

```text
source
  -> frontend
  -> IR / CFG / SSA
  -> analysis / optimization pass
  -> lowering / codegen
  -> runtime
```

## 笔记

- [编译器基础](/notes/compile/traditional/compiler_basic)：整体 pipeline、各阶段输入输出、和 AI Compiler 的对应关系。
- [前端：Lexer、Parser、AST、语义分析](/notes/compile/traditional/frontend_ast_sema)：源码如何变成 typed AST。
- [IR、CFG、SSA](/notes/compile/traditional/ir_ssa_cfg)：中端表示、控制流、use-def、phi、dominance、mem2reg。
- [Dataflow Analysis 与 Pass Pipeline](/notes/compile/traditional/dataflow_pass)：分析、优化 pass、pass manager、测试方式。
- [Lowering、Codegen、Runtime](/notes/compile/traditional/lowering_codegen_runtime)：后端下沉、寄存器分配、调用约定、解释器和 JIT。
- [循环优化](/notes/compile/traditional/loop_optimization)：LICM、unroll、tiling、interchange、dependence analysis、vectorization。
