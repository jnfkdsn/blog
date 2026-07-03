---
order: 4
title: Dataflow Analysis 与 Pass Pipeline
updated: 2026-07-01
tags: [compiler, dataflow, optimization, pass]
status: draft
---

# Dataflow Analysis 与 Pass Pipeline

相关入口：[传统编译器](/notes/compile/traditional/) / [IR、CFG、SSA](/notes/compile/traditional/ir_ssa_cfg)

Dataflow analysis 回答的是：在程序某个点上，编译器能确定什么信息。Optimization pass 利用这些信息做语义保持的 IR 改写。

```text 
IR + CFG
  -> analysis: liveness / reaching definition / constant propagation
  -> transform pass: DCE / CSE / LICM / CFG simplification
  -> updated IR
```

## Dataflow 的组成

一个 dataflow analysis 通常包含：

```text
方向：forward 或 backward
状态：每个 program point 上保存什么信息
transfer function：一个 block 如何改变信息
meet/join：多个前驱或后继的信息如何合并
初值：entry/exit 和其他 block 的初始状态
fixed point：迭代到信息不再变化
```

Forward analysis 从前往后传播，例如 reaching definitions、constant propagation。

Backward analysis 从后往前传播，例如 liveness。

## Liveness

Liveness 回答：某个变量在程序点之后还会不会被使用。它是 backward analysis，因为一个值是否 live 取决于未来路径是否还会使用它。

Block 级公式：

```text
live_in[B]  = use[B] union (live_out[B] - def[B])
live_out[B] = union live_in[S] for S in succ[B]
```

```text
B1:
  a = const 1
  br cond, B2, B3

B2:
  b = add a, 1
  ret b

B3:
  ret 0
```

`a` 在 `B1` 出口 live，因为后继 `B2` 会使用它；即使 `B3` 不使用，合并后仍然 live。

Liveness 用途：

- DCE：定义结果之后没有 use，且指令无副作用，可以删。
- Register allocation：live range 决定哪些 value 同时需要寄存器。
- Spill 决策：live range 长、冲突多的值更容易被 spill。

## Constant Propagation

Constant propagation 回答：某个 SSA value 是否一定是常量。

常见状态：

```text
UNDEF: 还没有定义或不可达
CONST(c): 一定是常量 c
TOP: 不是确定常量
```

合并规则：

```text
meet(UNDEF, CONST(1)) = CONST(1)
meet(CONST(1), CONST(1)) = CONST(1)
meet(CONST(1), CONST(2)) = TOP
meet(TOP, anything) = TOP
```

```text
entry:
  br %cond, then0, else0

then0:
  x1 = const 1
  jump merge0

else0:
  x2 = const 1
  jump merge0

merge0:
  x3 = phi [then0: x1], [else0: x2]
  y = add x3, 2
```

`x3` 合并两个 `CONST(1)`，仍然是 `CONST(1)`，所以 `y` 可以变成 `CONST(3)`。

如果 else 分支是 `x2 = const 2`，`x3` 会变成 `TOP`，`y` 不能折叠成单个常量。这个例子对应的是“不同前驱的信息如何合并”。

## Fixed Point 迭代

循环会导致信息需要反复传播。

```text
entry:
  i0 = const 0
  jump header

header:
  i1 = phi [entry: i0], [body: i2]
  c = lt i1, 10
  br c, body, exit

body:
  i2 = add i1, 1
  jump header
```

第一次看到 `header` 时，`i1` 可能是 `CONST(0)`。处理回边后，`i2` 又依赖 `i1`，多轮迭代后会发现 `i1` 不是单一常量，状态变成 `TOP`。

分析框架通常使用 worklist：

```text
worklist = all blocks
while worklist not empty:
  B = pop(worklist)
  old = out[B]
  in[B] = meet(out[pred] for pred in preds[B])
  out[B] = transfer(B, in[B])
  if out[B] changed:
    push successors of B
```

## Transform Pass

Optimization pass 是对 IR 的语义保持变换。

常见 pass：

- Constant folding：编译期计算常量表达式。
- Constant propagation：把已知常量传播到 use 点。
- DCE：删除结果无 use 且无副作用的指令。
- CSE：消除重复计算。
- CFG simplification：删除不可达 block、合并空 block。
- Inlining：把函数调用展开到调用点。
- LICM：把循环不变计算移到循环外。

## DCE

DCE 的核心判断：

```text
instruction result has no uses
and instruction has no side effect
```

可删除：

```text
%0 = add %x, %y
ret %x
```

不可删除：

```text
store %x, %ptr
call may_have_side_effect()
ret %x
```

工程实现里通常先标记有用指令：

```text
worklist = side_effecting_inst + terminators + returned_values
while worklist not empty:
  inst = pop()
  mark inst live
  for operand in inst.operands:
    mark operand.def live
delete unmarked pure instructions
```

这种方式叫 mark-sweep DCE。

## CSE

CSE 消除公共子表达式。基本块内 CSE 可以用 expression table：

```text
table: (opcode, operands, type) -> value
for inst in block:
  key = canonical_key(inst)
  if key in table and inst is pure:
    replace_all_uses(inst.result, table[key])
    delete inst
  else:
    table[key] = inst.result
```

例子：

```text
%0 = add %a, %b
%1 = add %a, %b
%2 = mul %0, %1
```

变成：

```text
%0 = add %a, %b
%2 = mul %0, %0
```

注意点：

- `add a,b` 和 `add b,a` 是否相同取决于 op 是否 commutative。
- `load ptr` 不能随便 CSE，因为中间可能有 store 改写内存。
- 浮点表达式也要小心，NaN、舍入、fast-math flag 会影响 legality。

## CFG Simplification

常见变换：

```text
br true, B1, B2  -> jump B1
empty block with single successor -> redirect predecessors
unreachable block -> delete
block with single pred/succ -> merge
```

```text
B0:
  br true, B1, B2

B1:
  jump B3

B2:
  jump B3
```

简化后：

```text
B0:
  jump B1

B1:
  jump B3
```

删除 block 时要同步维护：

- predecessor/successor 列表。
- phi incoming edge。
- block 中 value 的 uses。

CFG pass 的 bug 很容易表现为 phi incoming 数量不匹配。

## Pass Pipeline

Pass 顺序会影响效果。

```text
constant folding
  -> constant propagation
  -> CFG simplification
  -> DCE
  -> CSE
  -> DCE
```

原因：

- constant propagation 可能让分支条件变成常量。
- CFG simplification 可能产生新的 dead code。
- CSE 替换 use 后可能让旧指令变 dead。
- DCE 经常需要在多个 pass 后重复跑。

Pass manager 需要处理：

- pass 运行顺序。
- analysis 结果缓存和失效。
- before/after dump。
- pass 级别测试。
- debug 开关和统计信息。

简单 pass manager：

```text
for pass in pipeline:
  changed = pass.run(module)
  if changed:
    invalidate analyses affected by pass
```

## Pass 测试

Pass 测试比端到端测试更细。工程里常用两类断言：

- 文本 IR FileCheck：检查关键指令是否存在或消失。
- 语义测试：优化前后执行结果一致。

AI Compiler 中也一样：graph rewrite/fusion pass 需要同时验证 graph 结构和数值正确性。
