---
order: 4
title: Dataflow Analysis 与 Pass Pipeline
updated: 2026-09-15
tags: [compiler, dataflow, optimization, pass]
status: draft
---

# Dataflow Analysis 与 Pass Pipeline

相关入口：[传统编译器](/notes/compile/traditional/) / [IR、CFG、SSA](/notes/compile/traditional/ir_ssa_cfg)

Dataflow analysis 回答的是：在程序某个点上，编译器能确定什么信息。Optimization pass 利用这些信息做语义保持的 IR 改写。

本文先建立数据流分析的基本模型，再从局部表达式扩展到跨基本块、内存、循环与跨过程优化。阅读时关注“需要证明什么，证明之后能改什么”。常用 Pass 的名字用于定位实现，不能代替对合法性的解释。

文中的代码是解释算法的伪 IR / 伪代码，不是可以直接交给 `opt` 或 `mlir-opt` 的完整输入。除非单独说明，整数例子采用固定宽度、按位宽回绕的运算，不附带溢出标志；内存例子采用有效地址上的普通非 volatile、非 atomic 访问。

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

- 死赋值消除：目标变量之后不再 live 时，可以考虑删除赋值；仍要确认右侧计算可删除。SSA 上的简单 DCE 往往直接使用 use-list，不必先运行完整 liveness。
- Register allocation：live range 决定哪些 value 同时需要寄存器。
- Spill 决策：live range 长、冲突多的值更容易被 spill。

## Constant Propagation

Constant propagation 回答：某个 SSA value 是否一定是常量。

常见状态：

```text
BOTTOM: 分析尚未获得可执行定义的信息
CONST(c): 一定是常量 c
TOP: 无法在这个分析域中确定为单个常量
```

合并规则：

```text
join(BOTTOM, CONST(1)) = CONST(1)
join(CONST(1), CONST(1)) = CONST(1)
join(CONST(1), CONST(2)) = TOP
join(TOP, anything) = TOP
```

这里约定 `BOTTOM ≤ CONST(c) ≤ TOP`，合并取上界，因此写作 `join`；不同教材采用的偏序方向可能相反。`BOTTOM` 是分析状态，不是程序读取未初始化变量的语义，也不是 LLVM IR 的 `undef` 或 `poison`。SCCP 还会单独维护可执行边的信息。

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
  in[B] = join(out[pred] for pred in preds[B])
  out[B] = transfer(B, in[B])
  if out[B] changed:
    push successors of B
```

这是前向分析的示意，采用前述偏序约定。标准单调框架依靠单调传递函数和有限高度的状态域等条件保证收敛；更复杂的无限高度域还可能需要 widening 等策略。遇到循环不意味着无条件反复执行任意改写就会收敛。

## Transform Pass

Optimization pass 是对 IR 的语义保持变换。

这里采用常用的教学表述。落实到 LLVM 时，合法性还要遵守 undefined behavior、poison、浮点和内存模型的具体规则，不能只按实数代数或理想化顺序程序判断。

常见 pass：

- Constant folding：编译期计算常量表达式。
- Constant propagation：把已知常量传播到 use 点。
- DCE：删除结果无 use 且无副作用的指令。
- CSE：消除重复计算。
- CFG simplification：删除不可达 block、合并空 block。
- Inlining：把函数调用展开到调用点。
- LICM：把循环不变计算移到循环外。

这组操作只覆盖了一部分中端工作。完整一些的常用范围如下；一行表示一类职责，不意味着在所有编译器中都对应一个独立 Pass。

| 处理对象 | 常见优化或规范化 | 关键问题 |
|---|---|---|
| 局部表达式 | folding、copy propagation、InstCombine、reassociation | 能否得到更简单或更统一的表达式？ |
| SSA 与控制流 | SCCP、DCE/ADCE、CSE/GVN、PRE、SimplifyCFG、jump threading | 哪些值相等、哪些路径可执行、哪些计算影响结果？ |
| 局部存储 | mem2reg、SROA | 哪些内存对象可以改用 SSA 值表达？ |
| 内存访问 | load forwarding、DSE、memcpy optimization | 哪次写会被哪次读观察到，哪些访问可以省去？ |
| 循环 | LoopSimplify、LCSSA、LICM、indvars、strength reduction、rotate、unswitch、unroll、fusion、distribution、interchange、tiling | 依赖允许怎样改变迭代与访存顺序？ |
| 向量化 | loop vectorization、SLP vectorization | 哪些标量计算可以打包，目标机器上是否划算？ |
| 函数与模块 | inlining、IPSCCP、function specialization、attribute inference、dead argument elimination、GlobalDCE、GlobalOpt、tail recursion elimination | 跨调用边界有哪些可利用的信息？ |

下面先保留 DCE、CSE 和 CFG 的基本推演，再逐步扩展这些机制。循环的详细变换放在[循环优化](./loop_optimization)；mem2reg 的 SSA 构造放在[IR、CFG、SSA](./ir_ssa_cfg#mem2reg)。

## DCE

DCE 的核心判断：

```text
instruction results have no uses
and instruction is safe to erase under the IR semantics
```

对普通整数加法，第二项很容易满足；对调用、同步、可能不返回的计算等，则需要具体语义与属性支持。“没有内存写”还不足以推出“可以删除”。

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

一个保守的 mark-sweep 版本可以先标记有用指令：

```text
worklist = instructions that must be preserved, including terminators
while worklist not empty:
  inst = pop()
  if inst is already marked live:
    continue
  mark inst live
  for operand in inst.operands:
    if operand has a defining instruction:
      push operand.def
delete unmarked instructions that are safe to erase
```

这种方式叫 mark-sweep DCE。这里把所有 terminator 当作根，保留控制流；它不是完整的 ADCE 算法。另一类常见实现从可删除且无 use 的指令开始，删除后递归检查其 operands。后文的 ADCE 会进一步处理没有可观察作用的依赖环和控制流。

## CSE

CSE 消除公共子表达式。基本块内 CSE 可以用 expression table：

```text
table: (opcode, operands, type, semantic attributes/flags) -> value
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
- expression key 必须包含影响语义的属性和标志；对带 Region 的操作，还需要相应的结构等价判断，不能只比较上述简单键。
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

## 从局部规则到路径推理

### InstCombine、拷贝传播与重结合

Constant folding 要求表达式的输入已知为常量；许多简化不需要这个条件：

```text
%a = add %x, 0       -> %a 的 uses 改用 %x
%b = mul %a, 1       -> %b 的 uses 改用 %a
%c = copy %b         -> %c 的 uses 改用 %b
```

InstCombine 一类优化把代数规则、位运算规则、比较和类型转换的简化组织成局部改写，产生更简单或更规范的表达式。拷贝传播消除的是不改变值语义的复制；在 SSA IR 中，一些源码赋值本来就只需引用同一个 Value，未必真的产生 `copy` 指令。类型转换不能因为外形像赋值就直接消除。

Reassociation 改变可重结合表达式的分组，使常量或公共部分相邻。例如对上述整数语义，`(x + 3) + 5` 可以重组为 `x + (3 + 5)`，继而折叠成 `x + 8`。这两个职责可以由不同 Pass 配合，也可能部分重叠。

困难在规则的前提：严格浮点加法通常不满足结合律；带 `nsw` / `nuw` 的整数运算涉及 poison，重写时不能随意照搬标志。认识规则之后，还需要能给出“哪些语义条件改变时，规则就不能直接使用”。

LLVM 的 InstCombine 与 MLIR 的 canonicalize 都承担局部简化与规范化工作，但操作集合、规则来源和驱动协议不同，不能把它们视作同一个实现。

### SCCP：常量与可执行边共同收敛

前面的常量传播在合流处合并输入；如果某条输入边根本不会执行，它提供的值就不应该降低分析精度。

```text
entry:
  %flag = const true
  br %flag, left, right
left:
  %a = const 7
  jump merge
right:
  %b = const 99
  jump merge
merge:
  %v = phi [left: %a], [right: %b]
  return %v
```

如果机械地合并两个前驱，`7` 与 `99` 会得到 `TOP`。SCCP（Sparse Conditional Constant Propagation）同时求解两类事实：

1. `%flag` 为真，所以只有 `entry → left` 可执行。
2. `left → merge` 可执行，`right → merge` 尚无可执行证据，合并只得到 `CONST(7)`。
3. `%v` 为常量，又可能让后续分支确定，进一步改变可执行边集合。

“Sparse” 主要指沿 SSA 的 def-use 关系传播值信息；“Conditional” 指将控制流可达性纳入求解。随着新边被发现可执行，已有常量结论仍可能上升为 `TOP`，所以必须迭代收敛，不能把“尚未发现可达”过早当成最终不可达。

这个简单例子也能通过多轮普通常量传播和 CFG 清理处理。SCCP 的意义是把值与控制流放在同一个求解过程中，而不是声称只有它能得到这个结果。

### GVN 与 PRE：相等还不等于可以直接替换

CSE 的基本块表达式表可以向更大范围扩展。GVN（Global Value Numbering）为计算建立值等价关系，利用这些关系寻找冗余表达式和可消除的 load。这里的 global 通常指跨基本块的函数内范围，不等于跨整个程序。

跨块替换必须回答两个问题：候选值与当前值是否等价？候选值在这里是否可用，即是否满足支配等要求？

```text
entry:
  br %cond, left, right
left:
  %a = add %x, %y
  jump merge
right:
  jump merge
merge:
  %b = add %x, %y
  return %b
```

假设 `%x`、`%y` 是函数参数。`%a` 与 `%b` 表达同样的计算，但 `%a` 不支配 `merge`；走右侧路径时没有 `%a`，因此不能直接替换 `%b`。

PRE（Partial Redundancy Elimination，部分冗余消除）可以在缺少计算的右侧路径补出 `%r = add %x, %y`，并在 `merge` 用 `phi [left: %a], [right: %r]` 取代 `%b`。左侧路径少算一次，右侧路径的计算次数不变。

这个例子中加法安全且右侧直接进入合流点。如果插入点还通往不需要该值的路径，或者表达式会访问内存、可能产生陷阱，就必须重新检查执行条件、代码移动安全性和收益。GVN 与 PRE 是相关但不同的概念，具体 GVN 实现可能包含部分 PRE 能力；“GVN 可以寻找相等值”不意味着它会自动实现所有合法的 PRE。

## 从 SSA 值进入内存优化

### SROA 与 mem2reg：先改变表示，再暴露优化

考虑一个地址不逃逸、字段访问可以独立分析的局部对象：

```text
%pair = alloca {i32, i32}
store %x, pair.field0
store 0,  pair.field1
%v = load pair.field0
return %v
```

如果一直把 `%pair` 看作整体内存对象，后续优化需要追踪读写关系。SROA（Scalar Replacement of Aggregates）按访问情况拆分可分离的存储片段，并尽可能将它们提升为 SSA 值。本例最终可以直接 `return %x`，第二个字段的存储也随局部对象一起消失。

mem2reg 则主要把满足提升条件的局部栈槽的 load/store 改成 SSA 定义与 phi，具体推演见[已有章节](./ir_ssa_cfg#mem2reg)。LLVM 的 SROA 自身也会执行提升，不必机械理解成“先拆成字段，再一定单独跑 mem2reg”。

这种变换的价值是让后续常量传播、DCE、GVN 更容易看到数据关系。地址逃逸、复杂重叠访问、volatile 等因素可能阻止部分或全部提升。“提升到寄存器”在这里指 SSA 表示，并不保证后端不会 spill。

### Load forwarding 与 DSE：追踪哪次写能被观察到

先看一次可以转发的读取：

```text
store %x, %p
%v = load %p
return %v
```

如果写和读访问相同字节范围、类型解释兼容，且中间没有改变该内存的操作，可以用 `%x` 替换 `%v`。这叫 store-to-load forwarding；它描述一种变换，可能由 GVN、EarlyCSE 等实现承载。

加入另一条写之后，地址关系就变得关键：

```text
store %x, %p
store %y, %q
%v = load %p
```

只有证明 `%q` 的写不会覆盖 `%p` 的读取范围，或有其他足够的值信息，才能继续使用 `%x`。指针的 SSA 名字不同不能证明它们不重叠。

DSE（Dead Store Elimination）处理的是另一端：一条写是否永远不会被观察到？

```text
store 1, %p
store 2, %p
return
```

在约定的普通访问下，第二条写完整覆盖第一条写，中间没有读，第一条写可以删除。DCE 不能仅凭“store 没有 SSA 结果”得出这个结论；DSE 需要分析它产生的内存状态是否还有观察者。

一旦两次写之间有 `load %p`、可能读取该地址的调用，或者第二条写只覆盖部分字节，判断就会变化。若第二条写位于条件分支中，还需要考虑未执行覆盖写的路径；volatile、atomic 和异常退出也不能套用这个简单例子。

Alias Analysis 回答地址范围是否可能重叠，以及调用可能读写什么；MemorySSA 用 def/use/phi 结构组织内存依赖，方便搜索潜在的覆盖写。**MemorySSA 不会自动证明两个地址不别名，也不会把所有 load 变成不可变 SSA 常量。** LLVM 20.1.8 的 DSE 实现还显式检查中间读取、执行关系、完整覆盖和屏障条件，见文末源码入口。

### Memcpy optimization：减少搬运要有存储证据

对于先把数据复制到临时缓冲区，再复制到最终目的地的代码，可能直接复制到目的地；对于刚用固定字节初始化的源区域，后续 copy 也可能改成相应的初始化。

合法性取决于源和目的范围、重叠、大小、生命周期以及临时缓冲区是否还有其他用途。`memcpy` 与允许重叠的 `memmove` 语义不同，不能仅凭“看起来都是拷贝”互换。

这与 AI Compiler 中减少临时 buffer / copy 有共同的内存推理基础。但 tensor bufferization 还涉及值语义、in-place 决策、别名和所有权，不能直接等同于 LLVM 的 memcpy optimization。

## 从删除指令到改写控制流

### ADCE：连成环的无用计算也可能是死代码

简单 use-list DCE 很擅长删除没有使用者的叶子指令，再向前递归。但一组计算可以互相使用，却完全不影响返回值或副作用，例如循环中一个最终未被使用的附加累加器：其 phi 使用回边的加法结果，加法又使用 phi，两者都不满足“没有 use”。

ADCE（Aggressive DCE）从可观察行为出发，保留支持这些行为的必要计算，并处理控制依赖。如果一个分支决定是否执行有用的 store，分支条件即使不直接提供 store 的值，也必须保留。这就是仅沿数据 operands 标记不足以实现完整 ADCE 的原因。

在此基础上可以删除无用的计算环，以及不再影响行为的控制流。是否允许删除可能无限执行的循环，还取决于终止性和进度语义；“循环没有内存写”不是充分证明。ADCE 也不是把 DCE 多运行几次就能完全替代的。

### Jump threading：同一个条件在不同入口上可有不同答案

```text
left:                         right:
  jump merge                    jump merge

merge:
  %v = phi [left: 0], [right: %x]
  %c = eq %v, 0
  br %c, yes, no
```

沿 `left → merge` 进入时，后面的分支一定去 `yes`；从 `right` 进入时则未必。Jump threading 利用这种前驱相关的信息，让左侧路径直接去 `yes`，或复制必要的中间计算后再连接过去。

它不要求 `%c` 在所有入口上都是常量。如果 `merge` 中还有必须执行的计算或副作用，不能直接跳过；改写还需维护 phi 与支配关系。复制代码也有体积成本，因此合法不意味着一定值得做。

## 循环与向量化在 Pass 地图中的位置

[循环优化](./loop_optimization)已经介绍 LICM、展开、分块、交换、依赖分析与循环向量化。这里补齐相关常见职责，详细算法后续在对应章节展开：

| 机制 | 作用与前提 |
|---|---|
| LoopSimplify / LCSSA | 建立便于分析和改写的循环结构；LCSSA 通过出口 phi 表达被循环外使用的循环内定义。它们主要降低后续 Pass 的实现复杂度，未必直接加速程序 |
| Induction variable simplification / ScalarEvolution | 前者规范化归纳变量与相关条件；后者是分析，描述迭代中的数值演化，并在可证明时推导范围或迭代次数 |
| Loop strength reduction | 例如将每轮的 `base + i * stride` 地址计算改成递增地址；要考虑溢出、目标寻址模式和寄存器压力，不能认定乘法换加法总是更快 |
| Loop rotation | 调整 header、条件检查和循环体的布局，为其他优化创造合适形式；必须保留零次迭代等行为 |
| Loop unswitching | 把循环不变条件提到循环外，产生按条件选择的循环版本；收益是减少循环内分支，代价是代码增长，条件的提前求值也需合法 |
| Loop fusion / distribution | 合并循环以利用局部性，或拆分循环以隔离依赖、促进向量化；两者方向相反，都需要依赖与收益分析 |
| Loop deletion / idiom recognition | 前者删除可证明无用且允许删除的循环；后者把识别出的循环模式转换为如 memset/memcpy 的操作或调用 |
| Loop vectorization / SLP | 前者将多个循环迭代组合执行；SLP 将可组合的标量指令打包，例如一段代码中多个相邻元素的同类运算，并不要求以循环为输入 |

向量化还需要处理尾部、mask、对齐、别名和成本模型。遇到浮点归约时，改变分组顺序涉及数值语义。把循环表示成 MLIR `vector` 操作，或映射到 GPU/NPU 并行执行，还需要相应目标的映射和 lowering，不能只凭“做了向量化”推断最终硬件行为。

## 跨过程与模块优化

函数边界会隐藏参数值、内存效果和被调用目标。跨过程优化把信息沿调用关系传播，或改变这个边界。

| 机制 | 主要作用 | 关键限制 |
|---|---|---|
| Inlining | 将函数体带到调用点，让调用参数与函数内部计算相遇，暴露常量传播和循环优化机会 | 代码体积、递归、调用语义与内联成本 |
| IPSCCP | 跨调用传播常量参数、返回值等信息 | 需要处理外部调用者、间接调用和递归；不能把一个调用点的常量推广到所有调用 |
| Function specialization | 为特定参数或调用目标生成函数版本，再分别优化 | 版本数量、代码体积和调用点重定向的正确性 |
| Function attribute inference | 推导函数的读写、参数捕获等性质，为其他优化提供条件 | 必须覆盖相关控制流和被调用函数；不能因为没有显式 store 就认定整个函数无效果 |
| Dead argument elimination / argument promotion | 删除内部调用链中的无用参数，或在合法时用值参数替代某些间接传参 | 签名、ABI、地址逃逸和所有相关调用点能否一起修改 |
| GlobalDCE / GlobalOpt | 删除不可达的内部函数和全局对象，或简化全局变量的使用与初始化 | 可见性、符号逃逸、动态链接等可观察行为 |
| Tail recursion elimination | 将满足条件的尾递归改成循环，减少重复建立调用帧 | 递归位置、栈对象生命周期等；一般尾调用能否省去调用帧还取决于目标 ABI |

例如内部函数 `f(x, mode)` 根据 `mode` 选择不同算法。如果所有可分析调用都传入 `mode=0`，IPSCCP 可能传播这个事实并帮助删除另一条分支；如果调用者既有 `0` 又有 `1`，通用函数内的 `mode` 就不能直接固定。

此时可以选择内联，让每个调用点分别简化；也可以生成专门版本，保留通用版本服务其他调用。三者可能配合，但解决问题的方式不同。函数属性推导则可能保留调用边界，仅告诉调用者“该调用不会改写所关心的内存”，使调用者中的 load 优化成为可能。

LTO（Link-Time Optimization）扩大跨模块优化时可见的范围，使这些优化获得更多信息；它不是一个替代所有跨过程分析的单独算法。

## 分析、合法性和收益怎样连接

认识 Pass 清单之后，需要继续建立它们依赖的证据链：

| 分析或辅助信息 | 主要消费者 | 它帮助回答的问题 |
|---|---|---|
| Dominance | GVN、代码移动、SSA 更新 | 新值能否在每条相关路径上到达使用点？ |
| Post-dominance / control dependence | ADCE、控制流改写 | 哪个判断决定可观察行为是否执行？ |
| Alias / ModRef / MemorySSA | DSE、load forwarding、LICM | 哪些访问可能观察或改变同一片内存？ |
| LoopInfo / ScalarEvolution / dependence | 循环变换、向量化 | 循环结构、迭代演化和跨迭代依赖是什么？ |
| Call graph / 函数摘要 | IPSCCP、属性推导、跨过程 DCE | 调用边界之外有哪些可能行为？ |
| Target cost / profile 信息 | 内联、向量化、复制代码、布局 | 已经合法的变换在目标与负载上是否值得做？ |

这些不是所有编译器都必须具有的同名独立分析 Pass；也不是每个优化都必须使用表中全部设施。特别是 MemorySSA 与 Alias Analysis 提供互补信息，成本模型也不能代替合法性证明。

对 AI Compiler，最需要迁移的是这一推理方式：融合为什么没有破坏依赖？一次 buffer 写是否覆盖仍要读取的旧值？layout conversion 能否消除？在语义允许之后，局部存储、寄存器压力和并行度又怎样影响收益？只记住 DCE、GVN 或 fusion 的名称，还不足以回答这些问题。

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

加入上述优化后，可以用一条概念链理解它们如何相互创造机会：

```text
SROA / mem2reg                    暴露 SSA 数据关系
  -> InstCombine / SCCP          暴露常量与不可执行路径
  -> SimplifyCFG / DCE           清理简化后的结构
  -> LoopSimplify / LCSSA        建立循环变换所需形式
  -> LICM / 循环变换 / 向量化     改变计算与访存组织
  -> GVN / DSE / cleanup         清理新产生的冗余
```

这是职责示意，不是 LLVM `-O2` 的准确顺序，也不是推荐直接复制的命令。真实 pipeline 会安排内联与跨过程优化，反复做局部清理，并根据优化等级、目标和代码形态决定具体 Pass。有的规范化增加 phi 或分支，是为了降低后续变换的复杂度，不能只按当前指令数量判断好坏。

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

这里的 `changed` 只是教学抽象。实际管理器根据 preservation 契约及分析之间的依赖决定失效；修改了 IR 的 Pass 也可能维护并保留部分分析。MLIR 中的运行对象、嵌套 pipeline、缓存失效和失败行为见 [Pass 与 pipeline](../mlir/compiler/transforms/passes)。

## Pass 测试

Pass 测试比端到端测试更细。工程里常用两类断言：

- 文本 IR FileCheck：检查关键指令是否存在或消失。
- 语义测试：优化前后执行结果一致。

AI Compiler 中也一样：graph rewrite/fusion pass 需要同时验证 graph 结构和数值正确性。

还应该包含“不能优化”的反例：例如有可能别名的写阻止 load forwarding，仍可观察的旧值阻止 DSE，未授权的浮点重结合不能执行。结构测试验证变换是否发生，执行对比提供具体输入上的行为证据；两者都不自动构成对所有程序输入的等价性证明。

## 阅读深度与后续衔接

这篇负责常用中端优化的机制地图。初读能说明每类优化的输入、作用、主要前提，以及一项阻止优化的条件即可；实现阶段再选择代表性算法深入工作表、分析更新和测试。无需在开始 MLIR C++ IR API 前逐个实现这里列出的 Pass。

已有传统编译器基础时，后续重点是理解这些概念如何落到 MLIR 的多层表示、Region、Interface、改写协议与转换框架。当前 Pass 章建立运行和组织模型；IR API 与 PatternRewriter 负责实际修改；Dialect Conversion、bufferization 和目标优化继续补上表示转换、内存与性能推理。它们共同支撑全景路线中的开发能力。

## 资料与源码入口

下面的源码链接固定为工作区使用的 `llvmorg-20.1.8`。本文新增例子是机制推演，没有声称它们已作为 LLVM / MLIR 输入运行验证。Pass 的具体覆盖范围和默认顺序仍需核对所用版本。

- [LLVM Passes 总览](https://llvm.org/docs/Passes.html)：查询分析与变换的职责；其中部分命名或表述有历史背景，命令注册以对应版本为准。
- [LLVM 20.1.8 PassRegistry.def](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Passes/PassRegistry.def)：核对新 Pass Manager 的名字与调度层级。
- [SCCP.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Transforms/Scalar/SCCP.cpp) / [SCCPSolver.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Transforms/Utils/SCCPSolver.cpp)：观察可执行块、值状态与求解器如何连接。
- [GVN.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Transforms/Scalar/GVN.cpp)：值编号、冗余 load 与 PRE 的具体处理。
- [SROA.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Transforms/Scalar/SROA.cpp)：按访问拆分 alloca 并执行提升。
- [DeadStoreElimination.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Transforms/Scalar/DeadStoreElimination.cpp)：文件开头概述 MemorySSA 遍历、观察者、覆盖范围和执行条件检查。
- [ADCE.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/lib/Transforms/Scalar/ADCE.cpp)：从活跃根传播到数据与控制依赖。
- [MemorySSA](https://llvm.org/docs/MemorySSA.html)：理解内存依赖表示与 Alias Analysis 的关系。
- [LLVM 自动向量化](https://llvm.org/docs/Vectorizers.html)：区分 loop vectorizer 与 SLP，以及收益、运行时检查和尾部处理。
