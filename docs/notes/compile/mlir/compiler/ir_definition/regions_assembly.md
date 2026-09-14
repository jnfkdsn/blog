---
order: 4
title: 定义带 Region 的操作：传值、隔离与验证
updated: 2026-09-14
---

# 定义带 Region 的操作：传值、隔离与验证

到这里已经能定义一项标量计算、给它自己的类型和属性，并让通用代码查询它的能力。但真实方言经常定义含有一段计算的操作：函数、循环、分支、设备执行区域都有 Region。

本章把前一章的范围限制放进一个 `lesson.scope`，逐步规定输入如何进入、结果如何传出，以及 verifier 应怎样检查。**Region 只提供容器；完整操作定义必须补上执行与传值协议。**

## 1. 把计算放进区域，先写出四组值的对应

我们为 `lesson.scope` 规定一个很小的语义：进入一次区域，执行其中唯一 Block 一次，然后把 `lesson.yield` 的值作为父操作结果传出。输入按位置绑定到 Block 参数，结果按位置对应 yield 的 operand。区域不能隐式捕获外面的 SSA 值。

下面先看 `scope.mlir` 的第一个函数片段，完整文件还包含多结果与空结果变体：

```text
func.func @clip_inside(%x: i32) -> !lesson.range<-4, 7> {
  %r = "lesson.scope"(%x) ({
  ^bb0(%local: i32):
    %clipped = lesson.limit %local bounds(#lesson.bounds<-4, 7>) : !lesson.range<-4, 7>
    lesson.yield %clipped : !lesson.range<-4, 7>
  }) : (i32) -> !lesson.range<-4, 7>
  return %r : !lesson.range<-4, 7>
}
```

通用格式中的括号要按结构读：`(%x)` 是操作的 operand 列表；`({ ... })` 包含它拥有的 Region；末尾函数式类型写操作的输入与结果类型，不是区域本身的函数声明。

假设输入是 12，按我们规定的语义，值的流动是：

| 位置 | 此次对应值 | 关系 |
|---|---|---|
| scope 的 operand `%x` | 12 | 外部计算交给操作的输入 |
| 入口参数 `%local` | 12 | 进入区域时按位置绑定 |
| yield 的 operand `%clipped` | 7 | 区域内 clamp 的计算结果 |
| scope 的 result `%r` | 7 | 退出时按位置接收 yield 传出的值 |

这张表是语义推演；本工程尚未提供 scope 的执行器或 lowering。parser 不会执行该表，verifier 也不会计算输入 12 的结果。

`%x` 与 `%local` 是不同的 Value：前者定义在函数入口，后者定义在 scope 的入口。`%clipped` 与 `%r` 同样不同，分别是内部计算和外部父操作的结果。传值关系来自操作的协议，并不是四个名字自动互为别名。

## 2. 为什么不能从“含一个 Region”推出这些规则

`scf.for` 会重复执行 body，`scf.if` 选择区域，函数定义的 Region 并不在定义出现时直接执行。同样一个容器结构可以承载不同的执行协议。

本例至少需要明确这些约定：

- body 必须存在且只有一个 Block；我们不允许声明式空 Region。
- scope operand 与入口 Block 参数的数量、顺序和类型一致。
- 最后一项操作必须是 `lesson.yield`。
- yield operand 与父操作结果的数量、顺序和类型一致。
- 区域中的 SSA 外部依赖通过 scope operand 显式传入。

注意我们没有要求“所有输入类型等于所有结果类型”。主例输入 i32，输出 RangeType；区域本来就能执行改变表示的计算。把 `SameOperandsAndResultType` 加到这个操作上，会错误排除合法主例。

从契约再写 ODS，才能知道哪些约束可以复用，哪些必须自己实现：

<!-- source-example: scope-ops -->
```text
def ScopeOp : Op<Lesson_Dialect, "scope", [SingleBlock, IsolatedFromAbove, RecursiveMemoryEffects]> {
 let arguments = (ins Variadic<AnyType>:$inputs);
 let results = (outs Variadic<AnyType>:$outputs);
 let regions = (region AnyRegion:$body);
 let hasVerifier = 1;
 let hasRegionVerifier = 1;
}
def YieldOp : Op<Lesson_Dialect, "yield", [Pure, Terminator, HasParent<"ScopeOp">]> {
 let arguments = (ins Variadic<AnyType>:$values);
 let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}
```

yield 的 `assemblyFormat` 使用 optional group：非空 values 是锚点，有值时打印值与类型，空列表则省略整个分组，得到裸 `lesson.yield`。

这里 `SingleBlock` 提供单 Block 结构约束，`IsolatedFromAbove` 限制跨区域捕获；`Terminator` 标明 yield 在 Block 中的结构角色，`HasParent` 要求它出现在 ScopeOp 中。具体的入口/出口类型对应，仍需要手写检查。

`RecursiveMemoryEffects` 告诉效果处理代码继续考察内部操作。不能因为外层只是容器，就宣称内部任意计算都无效果。yield 自身声明 Pure，表示它不引入额外内存效果等；`Terminator` 仍要求维护其结构职责，不能看到其结果无人使用就单独删除这个出口。

## 3. 可变数量意味着一组值，不是一个特殊 Value

ODS 中 `Variadic<AnyType>:$inputs` 表示名为 inputs 的一组 operand，运行时长度可为零或更多。生成访问器返回一段范围，不是一个能容纳列表的特殊 SSA Value。outputs 与 yield values 也各自是一组。

因此，交换两个输入可以写成：

```text
%r:2 = "lesson.scope"(%x, %y) ({
^bb0(%a: i32, %b: i64):
  lesson.yield %b, %a : i64, i32
}) : (i32, i64) -> (i64, i32)
```

`%r:2` 为两个结果提供打印名称；它们仍是两个独立的 OpResult。yield 的第 0 项交给 `%r#0`，第 1 项交给 `%r#1`。这里故意使用不同类型，让错误地调换位置更容易暴露。

当三组列表都为空，区域依然可以有一个 Block 和一个不传值的 yield。**空参数、空结果与空 Region 是不同情况。** 完整合法文件包含这三个变体：

<!-- irdef-example: scope-input -->
```text
module {
  func.func @clip_inside(%x: i32) -> !lesson.range<-4, 7> {
    %r = "lesson.scope"(%x) ({
    ^bb0(%local: i32):
      %clipped = lesson.limit %local bounds(#lesson.bounds<-4, 7>) : !lesson.range<-4, 7>
      lesson.yield %clipped : !lesson.range<-4, 7>
    }) : (i32) -> !lesson.range<-4, 7>
    return %r : !lesson.range<-4, 7>
  }
  func.func @swap(%x: i32, %y: i64) -> (i64, i32) {
    %r:2 = "lesson.scope"(%x, %y) ({
    ^bb0(%a: i32, %b: i64):
      lesson.yield %b, %a : i64, i32
    }) : (i32, i64) -> (i64, i32)
    return %r#0, %r#1 : i64, i32
  }
  func.func @empty() {
    "lesson.scope"() ({
      lesson.yield
    }) : () -> ()
    return
  }
}
```

本章输入侧只有一组 variadic，结果侧也只有一组，各自的分组没有歧义。如果操作输入有两组可独立变化的列表，例如 `sources` 与 `destinations`，单凭总 operand 数量无法恢复分界：三个值可能按 1+2 分，也可能按 2+1 分。

此时通常需要 `AttrSizedOperandSegments` 记录各组长度，或在语义确实要求各组等长时使用 `SameVariadicOperandSize`。示意布局：

```text
operand 列表：[src0, dst0, dst1]
分段长度：    [1,    2]
```

optional operand 也存在这个问题，因为它的数量可能是 0 或 1；optional attribute 不占 SSA operand 槽位，是另一类字段。固定参数、多组可变参数混合时，分段信息要按 ODS 声明顺序维护。`AttrSizedResultSegments` 对应结果分组；嵌套列表还需内部各段长度，不能用一个总数替代。

读取多组可变参数操作时，应连同生成 accessor 和分段字段一起读，才能确定一段实际 operand 列表怎样对应各组参数。

## 4. 先确认外壳，再检查区域内的出口

考虑损坏的输入：scope 没有 Block；入口参数数量错误；yield 给出错误类型；甚至区域中某项操作自身都不合法。父操作 verifier 若一开始就强行读取“最后一个操作的正确 yield 参数”，可能在发出有意义的诊断之前就访问非法结构。

MLIR 因而区分普通操作验证和依赖内部操作的区域验证。沿本例需要的关系阅读顺序：

```text
结构与 ODS 字段检查
  → 不依赖内部操作合法性的 trait/interface 检查与 verify()
  → 验证区域内的操作
  → 依赖内部操作的检查与 verifyRegions()
```

这不是建议手动按顺序调用两个成员函数。应从工具或完整 `mlir::verify` 入口进入整个验证过程，框架负责相应步骤。结构性 trait 会先检查必要结构，ODS 再检查字段和类型种类；完整实现还处理 SSA、支配等约束。

我们的代码把两件事放在对应阶段：

<!-- source-example: scope-verify -->
```cpp
LogicalResult ScopeOp::verify() {
 if (getBody().empty())
   return emitOpError("requires a nonempty body");
 Block &entry = getBody().front();
 if (entry.getArgumentTypes() != getInputs().getTypes())
   return emitOpError("entry argument types must match input types");
 return success();
}
LogicalResult ScopeOp::verifyRegions() {
 Block &entry = getBody().front();
 if (entry.empty())
   return emitOpError("requires a lesson.yield terminator");
 auto yield = dyn_cast<YieldOp>(entry.back());
 if (!yield)
   return emitOpError("requires a lesson.yield terminator");
 if (yield.getValues().getTypes() != getOutputs().getTypes())
   return emitOpError("yield types must match result types");
 return success();
}
```

`verify()` 先拒绝空 body，再读取入口参数。它只比较输入与 Block 参数，不依赖内部计算。

`verifyRegions()` 才看最后一个操作是不是合法的 yield，并比较它传出的类型与父操作结果。它依赖前一阶段已经保证 body 非空，也受益于内部操作的验证结果。例如错误地把 `lesson.yield` 放到其他父操作中，会被其 `HasParent` 契约拒绝。

比较类型序列同时约束数量和逐项类型。若只比较“第一个元素类型”，零结果、多结果和遗漏值等情况就可能漏检。

一个紧邻的反例是：父操作宣称返回 i64，区域却 yield 入口的 i32。两项类型各自都合法，入口也绑定正确，失败发生在出口对应关系：

<!-- irdef-invalid: scope-yield-mismatch | yield types must match result types -->
```text
module {
  func.func @bad(%x: i32) -> i64 {
    %r = "lesson.scope"(%x) ({
    ^bb0(%local: i32):
      lesson.yield %local : i32
    }) : (i32) -> i64
    return %r : i64
  }
}
```

## 5. 隔离是显式依赖的契约

回到主例，把 body 中 `lesson.limit` 的输入从 `%local` 改为外部 `%x`，数值推演似乎完全相同。然而它违反了本例的 `IsolatedFromAbove`：body 直接引用了定义在 scope 外面的 SSA 值。

显式传入的写法把区域依赖集中在父操作的 operand 列表，方便克隆、搬移和独立处理这片区域时寻找输入。隔离不自动证明移动安全，但让依赖结构更明确。作用域规则必须由方言设计决定：许多 SCF 区域允许捕获，不能因此认为捕获本身不合法；本例是明确选择禁止它。

同理，如果希望 scope 能包含任意函数调用，需要进一步考虑符号可见性、调用效果和相关接口。`IsolatedFromAbove` 针对 SSA 捕获，不等于区域无法通过符号引用外部函数。

## 6. IR 合法以后，通用分析还缺什么

目前已有结构验证、类型对应和递归效果信息，但通用数据流分析不会阅读本章的中文语义，就自动知道 body 执行一次、输入怎样绑定、yield 怎样映射到父结果。

这些区域控制与数据流关系通常通过 `RegionBranchOpInterface` 等协议提供给消费者。它与本章接口的关联是：

- `SingleBlock` 回答结构限制。
- `RecursiveMemoryEffects` 帮助消费者汇总内部效果。
- 区域分支接口向相关分析提供可能的进入/退出及值传递关系。
- lowering 负责把这一执行协议变成目标支持的表示。

本工程只提供了前两类信息。后续要让通用分析穿过这个区域，还需要提供区域分支协议；结构验证通过本身不会建立这些控制流知识。

## 阅读后的一个小推演

先解释 `@swap` 中四组值的对应，再把两个 yield operand 的顺序调回去、保持父结果类型不变。应由哪一层拒绝？然后考虑把 `@empty` 的 yield 删除：空结果并不允许缺失所需 terminator。

现在应能从 scope 的四组值还原进入与退出过程，并判断结构、类型和捕获错误分别由哪层处理。接着读[解析与打印](./assembly_format)，专门追踪文本怎样构造成对象、对象又怎样写回文本。

## 配套观察与依据

观察入口为 `aicompiler-labs/llvm-mlir/06-ir-definition/README.md`，维护验证为 `docs/validate_ir_definition.py`。生成物保存在 artifacts，不要求先做完正式实验才能读原理。

固定版本 LLVM `llvmorg-20.1.8`：

- [Operations.md](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)：variadic/optional、验证顺序与 assembly format。
- [OpDefinition.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/OpDefinition.h)：单 Block、隔离、terminator 与分段 trait。
- [Verifier.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/IR/Verifier.cpp)：完整 IR 验证入口与区域递归。
- [ControlFlowInterfaces.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/ControlFlowInterfaces.td)：区域控制流接口的下一步查阅入口。

完整项目见 `06-ir-definition/Lesson.td`、`Lesson.cpp`。工程已验证单组 variadic、零/多结果和区域正反例；多组分段为原理推演，尚未提供 scope lowering 与 RegionBranch 模型。高级范围继续登记在覆盖表，当前证据不包含目标执行。
