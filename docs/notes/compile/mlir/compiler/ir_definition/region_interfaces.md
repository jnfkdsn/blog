---
order: 70
title: 区域、循环与调用的接口协议
updated: 2026-10-04
---

# 区域、循环与调用的接口协议

[区域操作定义](./regions_assembly)已经约定了 `my.scope` 的语义：进入一次，把输入交给 Block 参数，最后把 yield 的值交给父结果。解析器能建立这个容器，verifier 能检查类型对应，但通用常量传播还不知道“外面的 5 就是里面的 x”。

本章把这条传值关系提供给分析器，观察它怎样推出整个 scope 的结果为 7。随后把同一思路推广到分支、循环和调用，解释这些接口分别让消费者看到了哪一部分程序。

## 1. 容器里的一次计算

为集中观察协议，本章工程把 scope 收窄为一个 i32 输入和结果；执行一次、显式传值、禁止外部 SSA 捕获的语义与前章相同。

```mlir
func.func @scope_constant() -> i32 {
  %c5 = arith.constant 5 : i32
  %r = "my.scope"(%c5) ({
    ^bb0(%x: i32):
      %c2 = arith.constant 2 : i32
      %sum = arith.addi %x, %c2 : i32
      my.yield %sum : i32
  }) : (i32) -> i32
  return %r : i32
}
```

读者根据定义可以推演：x=5，sum=7，r=7。通用分析器却不能对所有带 Region 的操作都照做：另一个操作可能不执行区域、执行多次，甚至只把区域当作待执行的程序保存。

因此“拥有一个 Region”没有提供足够的执行事实。我们需要显式说明入口、出口以及相应值的位置关系。

## 2. RegionBranch 描述控制转移

`RegionBranchOpInterface` 将父操作和它拥有的区域看作几个控制转移点。对 scope，可能路径只有：

```text
父操作入口 → body → 父操作的结果
```

查询起点是 parent 时，操作回答 body；查询起点是 body 时，回答 parent 的结果。核心实现为：

```cpp
void ScopeOp::getSuccessorRegions(
    RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &successors) {
  if (point.isParent())
    successors.emplace_back(&getBody(),
        getBody().empty() ? Block::BlockArgListType{}
                         : getBody().front().getArguments());
  else
    successors.emplace_back(getOperation()->getResults());
}
```

空 body 的分支是为了让查询在错误 IR 上也不直接访问不存在的 Block；操作 verifier 仍拒绝空 body。接口实现不应通过崩溃来报告用户 IR 错误。

这里的 parent 不是“重新执行 scope”。从区域出发返回 parent 的 successor，表示离开区域并形成父操作结果。同一个词在两个查询位置承担入口与出口角色，需要结合转移方向阅读。

## 3. 控制边上的值对应

仅知道会进入 body 还不能推出 x=5。分析器还要知道 scope 的哪个输入被传到哪些入口参数。

本例 `getEntrySuccessorOperands` 返回 scope 的输入序列，对应 body successor 中列出的一个 Block 参数。于是这条边建立了：

```text
%c5 → %x
```

出口则由 `my.yield` 的终结操作协议描述。它声明 `ReturnLike`，将其 operand 序列作为退出区域时传出的值；父操作的 successor inputs 指向结果 r。于是建立第二条边：

```text
%sum → %r
```

入口和出口是两次独立映射，不能根据名字相似或恰好只有一个值就省略。多结果情况下尤其需要保持顺序；若第三个返回值的类型或数量不匹配，接口/操作验证应拒绝对应 IR。

我们的自定义检查仍负责“恰好一个 i32 入口参数、终结操作必须为 my.yield”；接口的控制边验证与这些结构条件协作。已有普通 verifier 不会自动替你生成正确的控制流语义。

## 4. 常量传播穿过边界

消费者现在能完成连续推演：

| 分析位置 | 得到的事实 | 信息来源 |
|---|---|---|
| scope 外部 | c5=5 | arith.constant |
| body 入口 | x=5 | RegionBranch 的入口映射 |
| body 内部 | sum=7 | arith.addi 的常量计算 |
| scope 结果 | r=7 | yield 与父结果的映射 |

工程执行 SCCP，再进行 canonicalize 后，实际得到：

```mlir
func.func @scope_constant() -> i32 {
  %c7_i32 = arith.constant 7 : i32
  return %c7_i32 : i32
}
```

这是编译器侧优化输出，不是执行自定义 scope 的机器码。最终清理还依赖效果信息：本例 scope 的递归效果来自内部纯计算。若 body 含有可观察写入，即使返回值已知，也不能仅因结果被常量替换就删除那些写入。

这里不需要为 SCCP 新增一个“认识 my.scope 名字”的分支。通用分析消费控制流接口，具体操作负责准确描述自己的语义。

## 5. 分支与循环的不同路径

同样的协议可以描述 SCF，但答案与 scope 不同。

对于未知条件的 `scf.if`，入口可能去 then 或 else；每个区域结束后都返回父结果。若两边分别产生 7 和 9，分析不能把结果定为单个常量。若条件已经证明为 true，额外的入口可达性查询可以排除 else，精度便提高。

对于携带一个状态的 `scf.for`，固定版本的基本 successor 查询给出：

```text
parent → body 或 parent
body   → body 或 parent
```

第一行包含零次执行，第二行包含回边与退出。循环入口值与上次 yield 共同影响 iter_arg，分析可能需要迭代到稳定状态。不能把循环照搬成“入口只传一次、出口只返回一次”的 scope。

诱导变量也不是普通 init operand 的直接转发：它由循环边界与步长产生。接口中的 successor inputs 可以只列相应的 loop-carried 参数；其余参数如何处理，需要结合循环协议或专门分析，不能把“没有映射”当成值不存在。

本章探针真实打印 `scf.for from parent: region0[1] parent[1]`，方括号表示本例沿边关联的值数，不是循环迭代次数。

## 6. LoopLike 与 RegionKind 的职责

知道回边存在，不等于已经有一个方便的循环变换 API。`LoopLikeOpInterface` 另外提供循环区域、相关边界或状态访问等能力，供循环消费者查询。例如 LICM 需要知道循环区域，并结合效果、依赖和可推测执行条件判断移动是否合法。

另一层区别是区域自身采用什么语义。`RegionKindInterface` 区分 SSACFG 与 Graph：前者具有基本块控制流和相应 SSA 支配要求；后者不采用相同的顺序控制流规则。探针观察到函数体与本例 scope 的 body 需要 SSA dominance，而 `builtin.module` 的区域是 Graph。

Graph 不意味着内部任意子操作都取消支配规则：module 里包含的 func 仍有自己的 SSACFG 区域。也不能把 Graph 理解成“运行时自动并行执行所有节点”。执行含义由所属操作/方言规定。

因此三个问题需要分开回答：

- RegionKind：这个区域内部采用哪类结构语义？
- RegionBranch：控制怎样进入、离开或在所拥有的区域间流动？
- LoopLike：这个操作怎样作为循环被查询和变换？

不是所有区域操作都应实现全部三个接口。

## 7. 从区域边界到函数调用

函数调用还多了一步：找到被调用者。工程的另一个例子是 `entry` 调用 `choose(true,3)`，被调用函数在 true 分支返回 7，然后 entry 把该值作为循环初始状态。

`FunctionOpInterface` 提供函数签名、参数/结果等结构能力；`CallableOpInterface` 描述可被调用的实体及其 body；`CallOpInterface` 描述调用目标和实参，并可通过符号表解析直接目标。三者不是同一个接口的不同名字，也不要求每个 callable 都恰好是 func.func。

本例探针解析到实际 callee。加入 inliner、SCCP 和清理后，调用与分支消失，循环的初始值成为 7。n=0 时返回 7；n 次循环每次再加 3，则按 i32 算术得到相应结果。这段数值关系是语义推演，实验主要核对变换后的 IR。

内联不只需要调用目标可解析，还要允许跨方言边界搬入 body、重连参数和 return。`DialectInlinerInterface` 提供这类约定。工具必须注册 Func 的 inliner extension；仅加载 Func dialect 不代表所有扩展已经就绪。对外部声明、不能解析的间接调用或不允许内联的边界，必须保留相应限制。

## 8. 克隆与移动的额外约束

复制整个 Region 时，入口 BlockArgument、内部结果和分支目标都要映射到副本。`IRMapping` 可以记录这些对应，但它不会自动证明新位置的外部捕获依然受支配，也不会将符号引用转换成 SSA 引用。

本例使用隔离，将 SSA 输入集中到操作边界，降低了映射遗漏的风险。内联还需要把 caller 的实参与被复制 body 的入口参数关联，把 return 值接到原 call 的 users，并处理必要的 CFG 连接。控制流接口提供语义依据，具体重写仍须维护结构和作用域。

若自定义操作已有完整 Region verifier，下一步应先明确目标消费者需要什么：常量传播关心可达边与传值，循环移动关心循环范围与效果，内联关心跨边界是否合法。按这个问题选择接口，比同时添加大量接口再猜它们为何有效更容易验证。

## 检查与依据

可做一个有限修改：把 scope 的输入从常量 5 改成函数参数，预测 SCCP 能否继续把返回值改成 7。另一个思考题是将内部加法换成带副作用的操作；结果常量化和区域可删除还是否等价？

固定依据为 [ControlFlowInterfaces](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/ControlFlowInterfaces.td)、[稀疏数据流消费者](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Analysis/DataFlow/SparseAnalysis.cpp)、[SCF 实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/SCF/IR/SCF.cpp)、[调用接口](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/CallInterfaces.td)与 [RegionKind 定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/RegionKindInterface.td)。

[19 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/19-extension-interfaces)提供自定义 scope 的实际接口与常量传播、标准分支/循环查询、函数内联及错误边界检查。它是对前章执行协议的接续，不是要求学习者从头实现一个完整控制流框架。
