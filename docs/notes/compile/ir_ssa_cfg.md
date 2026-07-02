---
order: 3
title: IR、CFG、SSA
updated: 2026-07-01
tags: [compiler, ir, cfg, ssa, dominator]
status: draft
---

# IR、CFG、SSA

相关入口：[编译器学习笔记](/notes/compile/) / [编译器基础知识地图](/notes/compile/compiler_basic)

IR 是编译器中端处理程序的主要形式。AST 接近源语言，IR 更接近“规则化的程序操作序列”。

```text
AST
  -> lowering
  -> IR
  -> CFG
  -> SSA / use-def
  -> analysis and optimization
```

## AST 为什么不够用

AST 保留源语言结构：

```text
IfStmt
WhileStmt
BinaryExpr
CallExpr
ReturnStmt
```

这对报错和语义检查很友好，但对优化不够规则。比如：

```c
return (1 + 2) * x;
```

AST：

```text
Return
  Binary(*)
    Binary(+)
      Int(1)
      Int(2)
    Name(x)
```

中端更想看到显式中间结果：

```text
%0 = const 1
%1 = const 2
%2 = add %0, %1
%3 = load x
%4 = mul %2, %3
ret %4
```

这种形式更容易做 constant folding、DCE、CSE 和 dataflow analysis。

## 三地址码

三地址码的特点是每条指令只做一件小事：

```text
dst = op arg0, arg1
dst = load addr
store value, addr
br cond, true_block, false_block
jump block
ret value
```

例如：

```c
int y = (a + b) * (a + b);
```

Lowering 后：

```text
%0 = add %a, %b
%1 = add %a, %b
%2 = mul %0, %1
```

CSE 可以把重复的 `%1 = add %a, %b` 消掉：

```text
%0 = add %a, %b
%2 = mul %0, %0
```

## Basic Block 和 CFG

Basic block 是一段顺序执行的指令：

- 只有一个入口。
- 只有一个出口。
- 中间没有跳转目标，也没有跳转指令。

例子：

```c
if (x > 0) {
  y = x;
} else {
  y = -x;
}
return y;
```

CFG：

```text
entry:
  %0 = gt %x, 0
  br %0, then0, else0

then0:
  %y1 = copy %x
  jump merge0

else0:
  %1 = neg %x
  %y2 = copy %1
  jump merge0

merge0:
  %y3 = phi [then0: %y1], [else0: %y2]
  ret %y3
```

CFG 记录 block 之间的前驱和后继：

```text
pred(merge0) = {then0, else0}
succ(entry) = {then0, else0}
```

优化和分析经常在 CFG 上运行，而不是在原始 AST 上运行。

## SSA

SSA 是 Static Single Assignment：每个 SSA value 只被定义一次。

非 SSA：

```text
x = 1
x = x + 2
return x
```

SSA：

```text
x0 = const 1
x1 = add x0, 2
ret x1
```

SSA 的好处：

- use-def 关系明确，每个 use 都指向唯一 def。
- constant propagation 更容易追踪值来源。
- DCE 更容易判断某个定义是否无用。
- CSE 更容易判断表达式是否等价。
- 寄存器分配前可以从 SSA value 得到 live range。

## Phi

分支合流时，一个变量可能来自不同控制流路径。

源程序：

```c
if (c) {
  x = 1;
} else {
  x = 2;
}
return x;
```

SSA：

```text
entry:
  br %c, then0, else0

then0:
  x1 = const 1
  jump merge0

else0:
  x2 = const 2
  jump merge0

merge0:
  x3 = phi [then0: x1], [else0: x2]
  ret x3
```

`phi` 不是运行时函数调用。它表示：如果控制流从 `then0` 来，`x3` 取 `x1`；如果从 `else0` 来，`x3` 取 `x2`。

工程实现里 phi 通常放在 block 开头：

```text
block merge0:
  phis:
    x3 = phi (then0 -> x1, else0 -> x2)
  insts:
    ret x3
```

## Dominator

Block A dominate Block B：从 entry 到 B 的所有路径都必须经过 A。

```text
entry
  |
  v
header
 /    \
body  exit
 \    /
  v  v
merge
```

`entry` dominate 所有 block。`header` dominate `body`，因为到 `body` 必须经过 `header`。

Dominator 的用途：

- 判断循环回边。
- 构造 SSA。
- 做 LICM。
- 做 code motion。
- 分析控制依赖。

循环回边通常定义为：

```text
B -> H，并且 H dominates B
```

`H` 是循环 header，`B` 是 loop latch。

## SSA Construction 的核心直觉

完整 SSA construction 涉及 dominance frontier。先记两个动作：

1. 给变量的每次赋值重命名。
2. 在控制流合流、且变量可能来自多个定义的位置插入 phi。

例子：

```c
x = 0;
if (c) {
  x = 1;
}
return x;
```

CFG：

```text
entry -> then0 -> merge0
      -> merge0
```

SSA：

```text
entry:
  x0 = const 0
  br %c, then0, merge0

then0:
  x1 = const 1
  jump merge0

merge0:
  x2 = phi [entry: x0], [then0: x1]
  ret x2
```

这里 `merge0` 需要 phi，因为它有两个前驱，并且 `x` 在其中一条路径上被重新定义。

## Memory 和 SSA

SSA 对普通 value 很清楚，但 memory 不简单。

```c
a[i] = 1;
x = a[j];
```

如果不知道 `i` 和 `j` 是否相等，就不能判断 `x` 一定是 1。这就是 alias analysis 的问题。

常见处理方式：

- 简单编译器：把 load/store 当作有副作用，少做激进优化。
- LLVM：用 memory dependence、alias analysis、MemorySSA 等机制分析内存。
- AI Compiler：tensor 通常是较大粒度的 value，但 view、in-place、alias、mutation 仍然复杂。

## IR 数据结构感

一个 toy IR 可以这样设计：

```text
Module
  functions: list[Function]

Function
  blocks: list[BasicBlock]
  entry: BasicBlock

BasicBlock
  name: str
  phis: list[Phi]
  insts: list[Instruction]
  terminator: Branch | Jump | Return
  preds: list[BasicBlock]
  succs: list[BasicBlock]

Instruction
  opcode: add / mul / load / store / call ...
  operands: list[Value]
  result: Value?

Value
  name: str
  type: Type
  def: Instruction | Phi | Argument | Constant
  uses: list[Use]
```

维护 use-def 时，每次替换 operand 都要更新 uses：

```text
replace_all_uses(old, new):
  for use in old.uses:
    use.user.operands[use.index] = new
    new.uses.append(use)
  old.uses.clear()
```

很多 pass 的实现都依赖这个小机制。
