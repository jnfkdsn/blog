---
order: 1
title: 支配、活跃性与操作移动
updated: 2026-10-04
---

# 支配、活跃性与操作移动

优化经常要把一次计算放到别的位置：提前计算，避免分支内重复执行；推迟计算，缩短中间值的存活范围；用已经算出的值替代后面的相同计算。修改操作链表很容易，困难在于新位置还能不能取得输入，以及修改后每个使用者还能不能取得结果。

本章从一个带分支的整数计算出发，沿“输入在哪里产生 → 使用在哪条路径发生 → 结果怎样离开分支”判断可用性。随后再看一个值最后在哪里被使用。前者由支配关系回答，后者由活跃性分析回答；它们会成为后续优化的依据。

## 1. 分支计算与可用的值

下面的函数先计算 `base=x+1`，再根据 `%flag` 返回 `base+1` 或 `base-1`：

<!-- analysis-example: dominance -->
```text
module {
  func.func @choose(%x: i32, %flag: i1) -> i32 {
    %one = arith.constant 1 : i32
    %base = arith.addi %x, %one : i32
    %r = scf.if %flag -> i32 {
      %local = arith.addi %base, %one : i32
      scf.yield %local : i32
    } else {
      %other = arith.subi %base, %one : i32
      scf.yield %other : i32
    }
    return %r : i32
  }
}
```

当 `x=5` 时，进入任意分支前，`base` 都已经是 6。true 分支算出 7，false 分支算出 5；`scf.if` 把所选分支 yield 的值作为 `%r` 交给 return。

这里有三个不同的结果：`%local` 是 true 区域内部的计算结果，`%other` 是 false 区域内部的计算结果，`%r` 是父操作 `scf.if` 的结果。即使某次执行中 `%r` 接到了 `%local` 的数值，它们仍是不同的 SSA Value，具有不同的定义位置和作用域。

这也解释了为什么不能把最后一行直接改为 `return %local`：false 路径根本没有产生它，而且它没有通过区域结果离开自己的作用域。正确的合流已经由 `scf.yield → scf.if result` 表达。

## 2. 支配关系与所有到达路径

在服从 SSA 支配规则的控制流区域中，如果到达 B 的每条可执行控制流路径都先经过 A，就说 A 支配 B。用一个定义替换其他值时，需要保证它在所有相关使用位置可用，不能只观察某一次执行轨迹。

最直观的反例是下面的控制流示意：

```text
entry
  ├─ true  → then:  v = x+1 ─┐
  └─ false → else:          ─┴─→ join: use(v)
```

从 else 到 join 的路径没有经过定义 v 的位置，所以 then 不支配 join。修复方式是在两条进入 join 的边上都提供一个值，并让 join 通过 Block 参数接收它；这与 SCF 父结果承担的合流职责相呼应。[SSA 章节](../../core/values_ssa)给出了沿边传参的完整形式。

在同一个普通 SSA Block 中，顺序也参与判断：定义在前，使用在后。跨 Block 时不能只比文本行号；跨 Region 时还要遵守区域嵌套、捕获和隔离约束。

回到主例：

| 查询 | 结论 | 原因 |
|---|---|---|
| `%base` 在 `%local` 的加法处可用吗 | 是 | 外层定义先于 if，true 区域允许捕获它 |
| `%local` 在函数 return 处可用吗 | 否 | 它属于 true 区域，false 路径也没有产生它 |
| `%r` 在函数 return 处可用吗 | 是 | return 位于产生 `%r` 的 if 之后 |

这三个结论足以指出一次替换的结构条件，却还不足以许可任何移动。

从另一个方向看，如果从 A 出发走向函数退出的路径都经过 B，就说 B 后支配 A。在没有提前返回的菱形 CFG 中，join 后支配 then 和 else；then 却不后支配 entry，因为 entry 可以进入 else。配套实际查询得到 `join postdominates then: 1`、`then postdominates entry: 0`。增加一个绕过 join 的 return，就会改变前一个结论。

后支配适合推理控制依赖和公共后继位置，但不能单独证明一段程序必然终止；带无限循环的图还需按分析对退出与不可达节点的约定解释。这里的简单菱形没有这些额外分支。

## 3. 操作移动的两端约束

假设希望把 true 分支中的 `%local = base+1` 提到 if 前面。需要分别检查输入端和输出端：

```text
原位置：base → if { local = base+1; yield local }
新位置：base → local = base+1 → if { yield local }
```

输入端要求 `%base` 与 `%one` 都支配新位置；因此可以插在 `%base` 后面，却不能插在 `%base` 前面。输出端要求新结果仍能供原来的 yield 使用。这个例子的外层定义可以被 true 区域捕获，所以成立。

还改变了一件事：false 路径原本不会执行这条加法，现在也会执行。对这里没有 overflow flag 的普通 `i32 addi`，结果按位宽取模，操作没有需要保留的内存效果，可以安全地额外计算。若换成可能除零的除法，即使输入都可用，也必须继续证明额外执行是允许的。这个问题将在[优化合法性](./legality)中展开。

相反，若把 `%base` 移入 true 分支，false 分支对它的使用就失去了来源。若把整个 if 移到 `%base` 前面，则两个区域捕获的 `%base` 都变成了尚不可用的值。检查移动时，既要看候选操作显式 operands，也要看它的嵌套区域捕获了哪些外部值。

## 4. 层次支配与区域语义

MLIR 的 IR 是嵌套结构，不能把所有 Region 摊平为一张普通 CFG。比如 `func.func` 隔离来自外层的 SSA 捕获，函数接收外部数据应通过参数；普通 `scf.if` 可以使用其外层已经可用的 SSA 值。

`DominanceInfo` 处理区域感知的支配查询，但使用者仍需理解目标操作的区域契约。把一个操作移到另一层，可能同时改变捕获、执行次数、区域退出规则和内存效果的相对顺序。

还要区分 SSACFG Region 和 Graph Region。上述“同一 Block 中必须先定义后使用”的推演针对有 SSA 支配要求的区域。Graph Region 可以采用不同的顺序语义；不能把在函数体中成立的文本先后判断无条件套到所有区域。需要泛化时，可以查询 `hasSSADominance(region)`，再按所属方言的规则处理。

## 5. 活跃性与最后一次使用

支配回答“此处有没有这个值”。活跃性回答另一个方向的问题：从某个位置往后，还有没有路径会使用这个值。

主例中，`%base` 必须活跃到 if 内部，因为选中的分支还要用它计算。但 if 完成以后，函数只返回 `%r`，不再使用 `%base`；所以 `%base` 在 if 之后已经不活跃。`%r` 恰好相反：刚离开 if 时，return 还等着它。

按固定版本的 `Liveness` 实际查询：

```text
base dead after if: 1
if result dead after if: 0
```

`1` 表示查询条件成立。这里的“dead after”指该位置之后不再需要这个 SSA Value，不表示可以删掉它此前的定义：if 内的两个分支仍有使用。删除定义需要考虑它的全部使用，而不是某一个位置之后的使用。

跨 Block 的活跃性通常可以由后向方程理解：

```text
LiveOut(B) = 合并后继在入口需要的值，并处理沿边传参的对应
LiveIn(B)  = UseBeforeDef(B) ∪ (LiveOut(B) − Def(B))
```

如果 B 内要使用某个在 B 外产生的值，它必须在入口可用；如果后继还需要某个值，并且 B 没有在内部产生它，也必须把需求传回入口。循环会把需求沿回边传回，直到集合不再变化。MLIR 的实现还要处理嵌套区域与 BlockArgument，这也是直接写一段扁平集合公式不足以替代实际分析的原因。

## 6. SSA 活跃性与 Buffer 生命周期

对于整数 `%base`，可以把活跃性直观理解成“后面还需不需要这个数”。MemRef 的 SSA 值描述一片存储，事情多了一层：没有人再使用某一个描述符，不等于没有人再访问它背后的分配。

```text
%view = memref.subview %buffer[...]
... 最后一次直接使用 %buffer ...
... 通过 %view 继续读写同一片分配 ...
```

即使 `%buffer` 这个 Value 已经不活跃，`%view` 仍可能指向同一分配。释放、复用存储要结合别名、逃逸和所有权，不能只用 `Liveness::isDeadAfter` 决定。上一模块的[所有权与释放](../memory/ownership)处理的是这项额外责任。

同样，MLIR SSA 活跃性并不直接等于机器寄存器的最终活跃区间。后续 lowering 可能展开、合并或删除操作，目标指令选择和寄存器分配会在更低层重新处理这些关系。

## 7. 查询代码与分析范围

取得主例的操作和值之后，真正的查询只需要以下几行。这里是从观察程序抽出的调用片段，`base`、`local`、`branch`、`ret` 都已从 IR 中取得：

```cpp
DominanceInfo dominance(func);
Liveness liveness(func);

bool inputAvailable = dominance.dominates(base, local);
bool resultAvailable = dominance.dominates(branch.getResult(0), ret);
bool noLaterUse = liveness.isDeadAfter(base, branch);
```

分析对象读取的是当前 IR 的结构。若之后改了 CFG 或使用关系，旧结论是否有效就必须重新判断；[缓存与失效](./caching)会解释 PassManager 如何协助维护。

函数内的这些关系还没有回答“谁调用谁”。跨函数传播事实时，会使用另一个图：`CallGraph` 通过 callable/call 接口建立调用关系。例如 `entry → helper → external` 中，前两个函数有函数体，external 只有声明。实际调用图能连接 entry 与 helper，对外部函数体则落入未知 callee 边界。知道符号名并不等于知道函数内部行为。

递归调用形成的强连通分量还可能需要跨函数不动点；如果当前任务仅在一个函数内部移动操作，就没必要先建立完整的程序级分析。图的范围应由要证明的结论决定。

## 阅读自查

1. 把 `%local` 提到 if 前面时，为什么检查它支配 yield 仍然不够？
2. `%base` 在 if 后已经 dead，为什么不能据此删除产生它的加法？
3. 一个 MemRef 的最后一次直接使用结束后，哪种情况仍然阻止释放其分配？

下一篇[别名、效果与内存依赖](./alias_effects)把判断对象从“值在哪里可用”推进到“同一地址的内容是否发生变化”。

## 实现依据与实践

示例与查询基于 LLVM/MLIR 20.1.8。接口对应 [Dominance.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/Dominance.h)、[Liveness.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Analysis/Liveness.h)和 [CallGraph.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Analysis/CallGraph.h)。配套[分析实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/12-analysis)保留完整 C++、实际输出与一个支配失败输入，不需要运行实验也可以沿正文推导上述判断。
