---
order: 1
title: 编译器基础
updated: 2026-07-01
tags: [compiler, ir, ssa, cfg, optimization]
status: draft
---

# 编译器基础

经典编译器可以粗略分为四段：

```text
source code
  -> frontend: lexer / parser / AST / semantic analysis
  -> middle-end: IR / CFG / SSA / analysis / optimization pass
  -> backend: lowering / instruction selection / register allocation / codegen
  -> runtime: memory / call convention / stack / heap / execution support
```

AI Compiler 的输入不一定是传统源码，也可能是 Python bytecode、FX Graph、ONNX、StableHLO、Torch IR、TensorIR 等。但它仍然绕不开几个核心问题：

- 程序如何表示：AST、Graph IR、SSA IR、Tensor IR。
- 程序是否合法：type、shape、dtype、layout、side effect。
- 程序如何改写：constant folding、DCE、CSE、fusion、layout rewrite。
- 程序如何下沉：high-level op -> tensor loop -> vectorized code -> kernel。
- 程序如何执行：runtime、memory planning、cache、JIT、library call。

## 笔记结构

- [前端：Lexer、Parser、AST、语义分析](/notes/compile/traditional/frontend_ast_sema)：从源码字符串到 typed AST，重点是 token、表达式优先级、符号表和类型检查。
- [IR、CFG、SSA](/notes/compile/traditional/ir_ssa_cfg)：从 AST lowering 到控制流图和 SSA，重点是 basic block、phi、dominance、use-def。
- [Dataflow Analysis 与 Pass Pipeline](/notes/compile/traditional/dataflow_pass)：分析如何驱动优化，重点是 liveness、constant propagation、DCE、CSE、pass manager。
- [Lowering、Codegen、Runtime](/notes/compile/traditional/lowering_codegen_runtime)：从中端 IR 到机器相关实现，重点是指令选择、寄存器分配、调用约定、解释器/JIT。
- [循环优化](/notes/compile/traditional/loop_optimization)：循环识别、LICM、unroll、tiling、interchange、dependence analysis、vectorization。

## 一个程序穿过编译器

源程序：

```c
int f(int x) {
  int y = 1 + 2 * x;
  if (y > 10) {
    return y;
  }
  return y + 1;
}
```

词法分析后得到 token：

```text
INT IDENT(f) LPAREN INT IDENT(x) RPAREN LBRACE
INT IDENT(y) ASSIGN INT_LITERAL(1) PLUS INT_LITERAL(2) STAR IDENT(x) SEMI
IF LPAREN IDENT(y) GT INT_LITERAL(10) RPAREN ...
```

语法分析得到 AST：

```text
Function f(x: int) -> int
  VarDecl y
    Binary(+)
      Int(1)
      Binary(*)
        Int(2)
        Name(x)
  If
    cond: Binary(>, Name(y), Int(10))
    then: Return(Name(y))
  Return(Binary(+, Name(y), Int(1)))
```

语义分析给 AST 补类型和符号绑定：

```text
Name(x) -> parameter symbol x: int
Name(y) -> local symbol y: int
Binary(*): int, Binary(+): int, Binary(>): bool
return expression type == function return type
```

Lowering 到三地址 IR：

```text
entry:
  %0 = const 1
  %1 = const 2
  %2 = mul %1, %x
  %y = add %0, %2
  %3 = gt %y, 10
  br %3, then0, else0

then0:
  ret %y

else0:
  %4 = add %y, 1
  ret %4
```

优化 pass 可以继续改写：

```text
constant folding:
  如果表达式两边都是常量，直接算出结果

DCE:
  删除没有 use 且无副作用的指令

CFG simplification:
  删除不可达 basic block，合并空跳转块
```

后端再把 IR 下沉到目标机器或虚拟机指令：

```text
load x
mul 2, x
add 1
cmp 10
branch ...
```

## 传统编译器和 AI Compiler 的对应关系

| 传统编译器概念 | AI Compiler 里的对应物 |
|---|---|
| 源码 | Python bytecode、FX Graph、ONNX、StableHLO、TorchScript |
| AST | GraphModule、op graph、高层 IR |
| 语义分析 | dtype/shape/layout 推导，op schema 校验，dynamic shape guard |
| IR | FX IR、AOTAutograd graph、Inductor IR、Triton IR、MLIR dialect |
| CFG/SSA | 控制流 IR、SSA value、block argument、use-def |
| pass | graph rewrite、fusion、CSE、DCE、layout propagation、decomposition |
| lowering | aten op -> loop IR -> Triton/C++/CUDA/kernel library |
| backend | Triton、LLVM、CUDAGraph、cuBLAS/cuDNN、Ascend C/TBE |
| runtime | memory planner、graph executor、JIT cache、stream/event、shape guard |

## 学习顺序

第一阶段：能解释一条编译流水线。

- token、AST、typed AST、IR、CFG、SSA、pass、codegen 都知道输入输出。
- 能把一个小程序手工翻译成 AST 和三地址 IR。

第二阶段：能手推中端机制。

- 给定 CFG，能找 basic block、前驱后继、回边。
- 给定 SSA，能解释 phi 和 use-def。
- 给定 dataflow 方程，能迭代到 fixed point。
- 给定 pass 前后 IR，能说明 legality 和收益。

第三阶段：做一个 toy compiler/interpreter。

- 支持整数、变量、if、while、函数调用中的一部分。
- 前端生成 AST，sema 生成 typed AST。
- lowering 到三地址 IR，跑几个基础 pass。
- 用解释器或简单 codegen 执行 IR。

第四阶段：进入 AI Compiler。

- 学 PyTorch 2.x compile pipeline：TorchDynamo、FX、AOTAutograd、Inductor。
- 学 Tensor IR：shape、stride、layout、loop nest、scheduler、fusion。
- 学 kernel lowering：Triton/CUDA/LLVM/Ascend C 后端。
