---
order: 4
title: 数据流事实的传播与收敛
updated: 2026-10-04
---

# 数据流事实的传播与收敛

编译器看到 `3+4`，容易确定结果为 7。但如果这个结果先进入分支，再通过 yield 返回，还能知道分支结果是什么吗？如果两条路径都返回 7，运行时条件虽然未知，结果仍然可以确定。反过来，只要一条可能执行的路径返回 8，就不能把结果统一替换为 7。

这类结论来自沿 IR 关系传播信息。本章先手工推导一个函数中的常量事实，再解释 MLIR 如何把同一过程交给数据流求解器。格、合流与不动点分别对应“怎样表示信息”“怎样合并路径”和“何时已经算够”，不会凭空引入另一套计算。

## 1. 从常量计算到分支结果

主例仍使用 SCF，但这次关注的不是值在哪里可用，而是值可能等于什么：

<!-- analysis-example: constants -->
```text
module {
  func.func @propagate(%flag: i1, %x: i32) -> i32 {
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %sum = arith.addi %c3, %c4 : i32
    %r = scf.if %flag -> i32 {
      scf.yield %sum : i32
    } else {
      %c7 = arith.constant 7 : i32
      scf.yield %c7 : i32
    }
    %out = arith.addi %r, %x : i32
    return %out : i32
  }
}
```

函数的 `%flag` 与 `%x` 是运行时输入。分析不能任意指定它们，但可以顺着已经确定的信息计算：

| 位置 | 已知事实 | 产生的新结论 |
|---|---|---|
| 两个 constant | 值分别为 3、4 | `%c3=3`，`%c4=4` |
| addi | 两个输入已知 | `%sum=7` |
| true 出口 | yield sum | 这条路径给 if 结果 7 |
| false 出口 | yield c7 | 这条路径也给 if 结果 7 |
| if 结果 | 任意可达分支都给 7 | `%r=7` |
| 最后 addi | r=7，但 x 任意 | out 不是可确定的单一常量 |

固定版本的求解器实际返回：

<!-- analysis-output: constants -->
```text
sum: 7 : i32
if result: 7 : i32
out: unknown
```

unknown 表示当前分析没有得到一个适用于全部允许执行的常量，不表示程序结果随机或没有定义。函数仍然计算确定的 `x+7`。

## 2. 常量事实的三类状态

为了逐步得出上表中的结论，需要区分“尚未处理到”与“已经无法用一个常量概括”。如果把这两者都写成 unknown，就难以正确合并还没计算完的路径。

常量分析可以用三类状态：

```text
                 unknown（任何可能值）
                  /      |      \
               const 3  const 7  const 8 ...
                  \      |      /
                  uninitialized
```

最下面表示还没有事实；中间每个状态表示确定等于某一个常量；最上面表示不再限定为某一个常量。这是本章常量域的格结构。图中的上下表示信息合并后可以走向的状态，不是数值大小关系。

当不同路径进入同一个结果，需要 join：

| 左边事实 | 右边事实 | join 结果 |
|---|---|---|
| uninitialized | 7 | 7 |
| 7 | 7 | 7 |
| 7 | 8 | unknown |
| 7 | unknown | unknown |

例如 true 出口已经求得 7、false 出口还没访问时，暂时得到 7。后来 false 也给 7，结论不变；若后来给 8，就上升到 unknown。求解器继续处理依赖此结果的操作，避免消费者永久停留在先前不完整的结论。

## 3. Transfer Function 与操作语义

路径合流使用 join，一条操作内部则需要根据输入事实推导输出事实。这通常称为 transfer function，传递函数。

对本例的 `addi`：两个输入分别为常量 3 和 4，按 `i32` 加法的语义得到常量 7。若一个输入为 unknown，输出通常也只能是 unknown。某些操作还有更强的特殊规则，例如与零的某些运算可以在另一输入未知时仍得到确定结果；是否采用这些规则，取决于操作的语义与实现。

MLIR 的 SparseConstantPropagation 会利用操作提供的 folding 能力来计算常量事实。分析因而不需要亲自实现每一个方言的完整解释器；操作把自己的局部折叠语义提供出来，求解框架负责传播与依赖。

这里仍在计算分析状态。直接运行 solver 并查询结果，不会自动把主例的 `scf.if` 从 IR 删除。要把事实落实为常量替换和无用代码清理，还需要消费者 Pass。

## 4. 控制流、可达性与区域边界

把 else 的常量从 7 改为 8，且 `%flag` 仍未知，两条路径都可能执行，if 结果变成 unknown。

再把条件改成恒 true。语义上 false 分支不会执行，它的 8 就不应参与结果合流；if 结果又可以是 7。这说明传播的不仅是数值事实，还包括哪些边和区域可达。

配套求解使用两项协作分析：`DeadCodeAnalysis` 提供可执行路径信息，`SparseConstantPropagation` 提供常量事实。前者的名字容易让人误会，它在这个上下文中参与的是可达性求解，并不等于自己已经把所有不可达 IR 擦掉。

对于 `cf.br`，传播需要知道传给后继 BlockArgument 的 operand；对于 `scf.if`，传播需要知道区域入口、出口和父结果的对应。框架通过控制流与区域分支接口理解这些边界。自定义带 Region 的操作如果没有提供足够协议，通用求解就不能凭外形猜它像 if、loop 还是函数。

这把[区域接口](../ir_definition/regions_assembly)与分析联系起来：接口不仅用于 verifier，它还告诉外部消费者“值从哪里进入，沿哪些执行边流向哪里”。

## 5. 工作列表与不动点

线性地扫描一次文本，对前面这个小函数可能够用。但循环中的事实会沿回边返回；一个值的新结论也可能影响早先已经访问过的操作。求解器需要重复处理受影响的部分。

可以把工作列表理解为“还有哪些依赖旧事实的判断需要重算”：

```text
某个输入状态变化
    ↓
把依赖它的操作或程序位置加入工作列表
    ↓
重新执行局部传递/合流
    ↓
输出状态若变化，继续通知依赖者
    ↓
没有状态再变化，达到不动点
```

对常量域，每个状态只能沿 `uninitialized → 某常量 → unknown` 这个方向变化。即使整数常量有很多种，单个状态也不会在 7、8、9 之间无穷往返；7 与 8 的冲突直接合并为 unknown。有限 IR 与这种有限高度的单调变化共同支撑收敛。

例如循环状态初始为 0、每轮加 1，在循环头需要合并初值 0 与回边可能的 1。常量域无法用一个常量同时表示它们，会上升到 unknown，而不是尝试把循环执行次数一个个枚举出来。

一般数据流分析未必都具有这么短的上升链。区间等更丰富的域，可能需要额外的收敛策略、保守退化或求解限制。不能把“使用 DataFlowSolver”当成任意自定义更新规则都能终止的保证。

## 6. 稀疏、稠密与整数范围

主例把事实附着在 SSA Value 上，并沿 use-def 和控制流参数传递。这是稀疏数据流的直观形式：某个结果变化时，优先唤醒依赖这个值的地方。

稠密数据流则通常在程序点维护状态，例如“执行到这条操作之前，哪些内存位置已经被初始化”。一次写入会更新整个抽象内存状态，下一程序点接收它。二者区别在于事实存放的位置与传播关系，不是“精确”和“不精确”的同义词；分析方向也可以根据任务选择前向或后向。

常量域只保留单点值。对有界循环 `%i=0..3`，虽然无法把 `%i` 当成一个常量，却可以保留范围：

```text
scf.for %i = %c0 to %c4 step %c1 {
  %next = arith.addi %i, %c1 : index
  ...
}
```

对完整实验中的这一循环，加载 `IntegerRangeAnalysis` 后，实际得到有符号范围：

<!-- analysis-output: ranges -->
```text
i: 0..3
next: 1..4
```

这样的事实可用于证明比较恒真、缩小边界检查或判断索引范围。但位宽、signed/unsigned 解释、溢出和动态边界都会影响推导。固定版本通过 `InferIntRangeInterface` 和循环边界信息传播整数范围；它也依赖可达性分析，不是只加载一个名字就能推断全部事实。

## 7. 从求解结果到 IR 改写

对主例，完整的分析配置与查询核心如下。这里的 `module` 和 `value` 已经从解析后的 IR 取得：

```cpp
DataFlowSolver solver;
solver.load<dataflow::DeadCodeAnalysis>();
solver.load<dataflow::SparseConstantPropagation>();
if (failed(solver.initializeAndRun(module)))
  return failure();

auto *state = solver.lookupState<
    dataflow::Lattice<dataflow::ConstantValue>>(value);
if (state && !state->getValue().isUninitialized()) {
  Attribute constant = state->getValue().getConstantValue();
  // constant 非空才表示已知常量；空 Attribute 表示 unknown。
}
```

现在运行实际变换 `sccp`，再以 `canonicalize` 清理主例，可以得到：

<!-- analysis-output: constants-optimized -->
```text
module {
  func.func @propagate(%arg0: i1, %arg1: i32) -> i32 {
    %c7_i32 = arith.constant 7 : i32
    %0 = arith.addi %arg1, %c7_i32 : i32
    return %0 : i32
  }
}
```

分支已经不再影响结果，两个输入常量及中间加法也被替代，最后保留 `x+7`。这份输出是组合 pipeline 的结果；不能把所有删除都归因于 solver 的一次查询。

到这里形成了完整过程：表示事实 → 沿操作与控制边传播 → 收敛 → 由变换消费事实 → 清理 IR。后续写自定义分析时，首先应确定需要支持哪种判断、怎样保守表示不知道，再选择求解框架和接口。

## 阅读自查

1. else 返回 8、flag 未知时，为什么 `%r` 不能保持 7？若 flag 恒 true 呢？
2. uninitialized 与 unknown 为什么必须区分？
3. solver 能查到 `%r=7`，为什么原来的 if 还可能留在 IR 中？
4. 一个循环变量不是常量，是否意味着分析对它完全没有可用信息？

## 实现依据与实践

配套[分析实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/12-analysis)包含常量合流、改变分支、可达性与范围观察。API 按 LLVM/MLIR 20.1.8 的 [ConstantPropagationAnalysis.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h)、[SparseAnalysis.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h)与 [IntegerRangeAnalysis.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h)核验。阅读其他版本的旧教程时，应按实际源码确认求解类与状态类型，不直接混用历史 API。
