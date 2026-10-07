---
order: 5
title: CSE、DCE 与 LICM 的合法性
updated: 2026-10-04
---

# CSE、DCE 与 LICM 的合法性

前面几篇分别建立了值的可用性、内存依赖、分析有效期和传播机制。现在把它们放回优化任务：看到重复计算，能否合并？看到无人使用的结果，能否删除？看到每轮计算相同输入的表达式，能否提到循环外？

三个问题看起来都在“少做计算”，改变的行为却不同。合并用一个已有结果替代另一个；删除去掉一次执行；外提可能让原本不执行的路径也执行。需要证明什么，取决于实际改动了什么。

## 1. 重复整数计算的合并

先看这个完整函数：

```text
func.func @pure(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %a = arith.addi %x, %c1 : i32
  %b = arith.addi %x, %c1 : i32
  %dead = arith.muli %x, %x : i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}
```

`%a` 和 `%b` 的操作名、类型、operands 及相关语义属性相同，都是这里的普通 `i32 addi`。两个输入已经由 SSA 固定，没有隐藏的可变内存内容参与结果。并且 `%a` 的定义位于 `%b` 及其使用者之前，替换后仍满足支配。

于是可以把 `%r` 的第二个 operand 从 `%b` 改成 `%a`，再删除产生 `%b` 的加法。CSE（公共子表达式消除）把“同一个结果已经算过”变成引用复用：

```text
原来：a = x+1；b = x+1；r = a+b
之后：a = x+1；           r = a+a
```

检查相同不能只比打印出的操作名字。不同类型、不同溢出或浮点属性、不同 operands，都会影响等价性；控制流上已经有一个相同表达式，也不表示其结果在当前路径可用。

## 2. 无用结果与可删除效果

主例中的 `%dead` 没有使用者，因此计算出的整数不会影响返回值。该乘法也没有必须保留的副作用，可以删除。这是 DCE 的基本情形。

换成下面的操作，判断就不同：

```text
memref.store %v, %p[%i] : memref<?xi32>
```

store 没有 SSA 结果，但它的写入可能被后续代码观察。`use_empty()` 在这里不能作为删除依据。函数调用也类似：返回值无人使用，不等于函数内部没有外部可观察行为。

LLVM 20.1.8 的通用 CSE 实现同时处理一部分 trivially dead 操作。因此对第一节的函数仅运行 `cse`，实际就得到：

```text
func.func @pure(%arg0: i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1_i32 : i32
  %1 = arith.addi %0, %0 : i32
  return %1 : i32
}
```

重复加法和无用乘法都消失了，但它们消失的理由分别是复用已有结果与删除无效计算。观察最终 IR 时需要把理由分开，不能把所有变化都归为一条用户 Pattern。

更深入地看，`isOpTriviallyDead` 结合结果使用与效果信息决定是否可删；在标准语义允许的情况下，无用普通读取也可以被删除。操作可能触发 UB，并不自动意味着必须保留它来“报错”：UB 不是需要保留的正常结果。若一种检查必须有可观察的失败行为，就应该用能表达这种行为的操作契约。相反，提前执行一个原先不会执行的危险操作，可能把原本有定义的程序变成 UB；这正是外提需要单独证明的事情。

## 3. Load 合并与中间写入

对连续的两条相同普通 load：

```text
%a = memref.load %p[%c0] : memref<1xi32>
%b = memref.load %p[%c0] : memref<1xi32>
%r = arith.addi %a, %b : i32
```

本章约定单线程、有效存储、无其他写入或特殊读语义。两次读取之间内容没有改变，当前 CSE 能将它们合并为一次 load。

在中间加入 `memref.store %v, %q[%c0]` 后，情况回到[别名章节](./alias_effects)的主例。p/q 可能相同，第一次读到 3，写入后第二次读到 9；合并会把正确的 12 改成 6。实际 CSE 保留两次 load 和中间 store。

这是两层证据相遇的位置：相同 load 提供表达式候选，内存效果与位置关系决定能否复用。当前通用 CSE 的只读操作处理比“任意别名分析加路径推理”更有限，它在同 Block 的支持范围内检查中间效果；即使某个特殊 store 可被其他分析证明无关，也未必自动优化。

因此，“没有优化”不能直接推出“语义上不能优化”。定位原因时，要看候选是否匹配、所需事实是否取得、实现是否使用了该事实。

## 4. 循环不变量的外提

现在考虑函数计算 `n` 次 `x+1` 的累加：

```text
func.func @hoist(%n: index, %x: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1 : i32
  %zero = arith.constant 0 : i32
  %result = scf.for %i = %c0 to %n step %c1
      iter_args(%acc = %zero) -> i32 {
    %scale = arith.addi %x, %one : i32
    %next = arith.addi %acc, %scale : i32
    scf.yield %next : i32
  }
  return %result : i32
}
```

每轮 `%scale` 的两个输入都来自循环外，而且没有随迭代改变；`%next` 却依赖当前 `%acc`，它是上一轮的结果，必须留在循环内。

把 `%scale` 提前后，循环体变成：

```text
%scale = arith.addi %x, %one : i32
%result = scf.for %i = %c0 to %n step %c1
    iter_args(%acc = %zero) -> i32 {
  %next = arith.addi %acc, %scale : i32
  scf.yield %next : i32
}
```

当 `x=2,n=4`，原来每轮算 3，共算四次；现在先算一次 3，再累计四次，结果都为 12。LICM（循环不变量外提）希望减少的是循环内重复工作，不是改变累加状态的递推关系。

LLVM 20.1.8 的通用 LICM 借助 `LoopLikeOpInterface` 取得循环边界、判断值是否定义在外，并通过该接口移动操作。候选还要满足无内存效果与可推测执行的检查。这里“是否不变”与“是否允许提前执行”是分开的。

## 5. 零次循环与推测执行

令 `n=0`。原程序一次循环都不进入，`%scale` 完全不计算，直接返回初始累加值 0。外提后会额外计算一次 `x+1`，但对这个普通整数加法，额外计算没有破坏有定义的行为。

现在只把循环内的计算改为：

```text
%q = arith.divsi %x, %d : i32
%next = arith.addi %acc, %q : i32
```

`%x` 和 `%d` 仍然定义在循环外，所以每轮 operands 相同。但当 `n=0,d=0` 时，原程序不会执行除法，返回 0 是有定义的；外提后会执行除零，改变了原程序的语义边界。

这给出了一个只靠“不随迭代变化”检查无法发现的错误。整数有符号除法还存在最小负数除以 -1 等需要考虑的情形；对未知输入，不能把它当成无条件可推测执行。当前实际 LICM 将这个 `divsi` 留在循环内。

相似地，循环内 load 即使地址由外部 SSA 值计算，也可能读取被循环内其他写入改变的内容；即使内容不变，零次循环时提前 load 还可能引入原本没有发生的无效访问。通用 LICM 的保守条件避免了直接跨过这些证明责任。更强的优化可以在额外前提、guard 或专门内存分析下进行，但那是另一套已证明条件。

## 6. 执行顺序与数值契约

合法性还取决于运算所承诺的数值语义。`(a+b)+c` 与 `a+(b+c)` 在无限精度数学里相等，浮点舍入下却可能不同。合并完全相同的浮点表达式，与改变归约的加法次序，不是同一种变换。

在并行归约、向量化和 tile 内外部分和合并时，需要明确哪些重排被允许，是否有相关 fast-math 契约，以及误差判据是什么。不能因为当前整数主例输出相同，就把结论直接推广到 Softmax、Matmul 或 Attention。

终止行为也属于执行语义。一项循环或调用即使没有写内存，也可能不返回；把它无条件提到原本不会执行的路径，会改变整个程序是否终止。效果与推测执行协议正是为了让消费者可以询问这些不同维度的保证。官方[效果与推测执行说明](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)区分了这些责任。

## 7. 从合法性判断到验证证据

本章采用两类验证，分别回答不同问题。

首先检查变换后的结构：`pure` 只剩两条加法；`load_pair` 只剩一次 load；`load_store` 仍有两次 load 和一次 store；`hoist` 的 invariant 加法位于循环之前；未知除数的 `divsi` 仍在循环内。

然后把优化前后的两个模块分别降低到 LLVM、编译为本机对象，并用同一份 C 调用程序执行。实际核对包括：

| 调用条件 | 优化前后期望结果 |
|---|---|
| pure(-3)、pure(0)、pure(5) | -4、2、12 |
| p=q，初始 3，写 9 | 返回 12，内存最终为 9 |
| p/q 不同分配，p 初始 3，向 q 写 9 | 返回 6，q 最终为 9 |
| x=2，循环 0/1/4 次 | 0/3/12 |
| 除法循环 0 次，除数 0 | 0；没有实际执行除法 |
| 12/3 累加 1/4 次；-12/3 累加 3 次 | 4/16；-12 |

这里没有执行“强行外提除零”的错误程序来期待崩溃，因为 UB 不保证产生某种特定观察。反例已经由有定义路径被改成无定义路径的推演成立；运行验证的是合法变换保留这条路径。

这些检查也不是所有输入的形式化证明。语义论证说明变换依据，结构测试阻止实现漏掉条件，数值测试发现选定边界上的错误，三者各有职责。

## 阅读自查

1. CSE 为什么同时需要等价性与支配关系？
2. 无人使用的普通读取可以删除，为什么仍不意味着可以任意提前执行它？
3. `%d` 定义在循环外，为什么不能单凭这一点外提 `%x/%d`？
4. 对本章输出做 `verify` 和做 C 数值对照，各自能发现什么问题？

接下来学习结构化优化时，可以把这里的检查扩大到整个迭代域：一条标量计算的输入、效果与执行条件，变成一个 tile 或一组线程的依赖、边界与同步条件。机制增加了，证明对象仍然是具体行为变化。

## 实现依据与实践

行为按 LLVM/MLIR 20.1.8 的 [CSE.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/CSE.cpp)、[SideEffectInterfaces.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Interfaces/SideEffectInterfaces.cpp)和 [LoopInvariantCodeMotionUtils.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp)核验。[分析实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/12-analysis)保存完整输入、查询程序、观察入口和数值驱动；练习要求修改一个影响合法性的条件并解释变化，无需先实现完整通用优化器。
