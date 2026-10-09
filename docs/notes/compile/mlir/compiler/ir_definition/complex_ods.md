---
order: 80
title: 可变参数分组与操作属性存储
updated: 2026-10-04
---

# 可变参数分组与操作属性存储

前面的操作只有固定数量的输入，或者一组可变输入。现在希望表达一个简单的加权求和：输入序列和权重序列等长，还可以提供一个初始值。计算本身很熟悉，新的问题是：当 Operation 把所有 operands 存成一条序列时，怎样知道各组从哪里开始、在哪里结束？

本章先让分组信息支持一次实际展开，再讨论修改参数时怎样保持分组，以及这些静态字段在 Properties 中怎样保存。

## 1. 两组输入与一个可选初值

定义 i32 算术语义：

```text
result = seed + Σ inputs[i] × weights[i]
```

没有 seed 时从零开始；没有输入项时直接返回 seed 或零。算术遵循无额外溢出标志的 i32 加乘语义，不能额外假定无限精度整数。

一个具体实例为：

```text
%r = my.weighted_sum inputs(%c2, %c3) weights(%c4, %c5) seed %c9
```

这些 operands 分别来自常量 2、3、4、5、9，所以结果为 `9+2×4+3×5=32`。不带类型尾缀是本例声明式格式的选择：每个 operand/result 都约束为 i32，生成 parser 可以据此解析类型。

在内存中的通用 Operation 层，operand 顺序为：

```text
[c2, c3, c4, c5, c9]
 └inputs┘ └weights┘ seed
```

仅知道总数为 5，不能区分 [2,2,1] 与 [1,3,1] 等分组。字段名不会自动成为每个 Value 的标签，需要另外保存组长度。

## 2. 分段长度怎样恢复访问范围

ODS 中声明三组，并使用 `AttrSizedOperandSegments`：

```tablegen
def WeightedSumOp : Op<My_Dialect, "weighted_sum",
    [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<I32>:$inputs,
                       Variadic<I32>:$weights,
                       Optional<I32>:$seed);
  let results = (outs I32:$result);
  let hasVerifier = 1;
  // assemblyFormat 在工程中定义。
}
```

本例的 `operandSegmentSizes=[2,2,1]` 与声明顺序对应。inputs 从位置 0 开始取 2 项；weights 从前面组长度之和 2 开始取 2 项；seed 从 4 开始取 1 项。

生成的 `getInputs()`、`getWeights()` 与 `getSeed()` 依据这份元数据取得视图。Optional 的长度只能为 0 或 1；它是有特定数量约束的一组，不是用空 Value 填充一个永远存在的位置。

参数全空时仍有三项长度 `[0,0,0]`。组的数量由操作定义决定，组内元素数量才随实例变化。

## 3. 结构约束与计算约束

正确的分段首先需要长度非负、数量与组数一致、长度和等于 operand 总数，并满足 Optional 限制。这些是“能否正确解释这串字段”的约束。

加权求和还有一项自己的计算约束：inputs 与 weights 必须等长。分段 `[1,3,1]` 即使总数等于 5，也不适合配对相乘。因此还要手写：

```cpp
LogicalResult WeightedSumOp::verify() {
  if (getInputs().size() != getWeights().size())
    return emitOpError("requires equal input and weight counts");
  return success();
}
```

不能依靠后面的 `zip` 悄悄只处理较短的一组。那会让结构上可读取的错误程序产生不符合操作定义的结果。

同样，`SameVariadicOperandSize` 与显式分段解决的是不同场景：前者要求相关可变组采用相同长度推断；本例还包含可选 seed，选择显式分段更直接。多组可变结果有对应的 `AttrSizedResultSegments`；结果分段与 operand 分段各自描述自己的列表。

## 4. 消费者按语义组展开

展开 Pass 不读取“总 operand 的前一半”，而是通过已验证的访问器取得两组。以 seed 为初始值，依次生成乘法和加法：

```cpp
Value acc = op.getSeed();
if (!acc)
  acc = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
for (auto pair : llvm::zip(op.getInputs(), op.getWeights())) {
  Value product = rewriter.create<arith::MulIOp>(
      loc, std::get<0>(pair), std::get<1>(pair));
  acc = rewriter.create<arith::AddIOp>(loc, acc, product);
}
rewriter.replaceOp(op, acc);
```

代码省略了 Pass 外壳与插入点设置。实际运行展开和 canonicalize 后，完整函数直接返回 `arith.constant 32 : i32`；空输入且无 seed 的函数返回零。

这让分组机制的作用变得具体：它保存了消费者必须遵守的配对关系。ODS 没有替我们实现加权求和，但让结构信息可靠地到达实现者。

## 5. 修改参数与同步元数据

现在追加已有的第一对输入和权重，即再增加 `2×4`，期望结果变成 40。新的底层 operands 为：

```text
[2,3,2, 4,5,4, 9]
分段长度：[3,3,1]
```

如果只插入 Value 而保留旧长度，访问器将读错组边界。生成的 mutable range 可以同时维护相应分段：

```cpp
Value input = op.getInputs().front();
Value weight = op.getWeights().front();
op.getInputsMutable().append(input);
op.getWeightsMutable().append(weight);
```

工程先保存 Value，避免修改一个组后继续使用可能过期的 range。两次修改完成后，检查器看到 [3,3,1]，展开结果确为 40。中间暂时不等长的状态只存在于本次成对修改内部，不能在它尚未修复时当作合法操作交给消费者。

这段代码处于普通 Pass 的直接改写中。若置于 Pattern/driver 管理的改写过程，还应使用相应 rewriter 的通知协议，把原地修改包在允许的修改入口内；不能因访问器会更新长度就忽略 driver 的约定。

## 6. 多层分组与空子组

批量处理时，可能希望一次返回若干组各自的和：`[[2,3],[4],[]] → [5,4,0]`。这是“一组里面还有多组”，不是前面的多个命名字段。

ODS 可以使用：

```tablegen
let arguments = (ins
  VariadicOfVariadic<I32, "groupSizes">:$groups,
  DenseI32ArrayAttr:$groupSizes);
let results = (outs Variadic<I32>:$results);
```

底层 operands 仍然平铺为 [2,3,4]，`groupSizes=[2,1,0]` 恢复三个子组。`getGroups()` 返回可逐组访问的范围，消费者分别从零累加，再替换成三个结果。

必须保留最后那个零：`[2,1]` 只有两个组，`[2,1,0]` 有三个组，只是最后一组为空。两者总 operand 数相同，结果数量却不同。本例 verifier 额外要求“每组恰有一个结果”。

实际 generic IR 中，分组信息明确出现：

```text
%r:3 = "my.group_sum"(%c2, %c3, %c4)
  <{groupSizes = array<i32: 2, 1, 0>}>
  : (i32, i32, i32) -> (i32, i32, i32)
```

工程核对了分组、展开后的三个常量和空组。真实项目也可以为这种嵌套结构设计可读的 assemblyFormat；此处保留 generic 形式，以便直接看到存储字段。

## 7. Properties 与可丢弃属性

上例 `<{...}>` 并不是普通的 `{...}` 属性字典拼写差异。本工程为 My dialect 启用 `usePropertiesForAttributes`，将操作固有字段保存在操作的 Properties 中。

对 WeightedSum，生成存储中有一个长度为 3 的 C++ 整数数组；为了打印/序列化，可以把它转换成 `operandSegmentSizes = array<i32: ...>` 的属性表示。探针实际观察到：

```text
properties={operandSegmentSizes = array<i32: 2, 2, 1>}
discardable={my.note = "example"}
```

前者影响操作字段的解释，不能丢失；后者是本例附加的说明信息，删除它不改变加权求和。Properties 是操作持有的结构化存储，Attribute 则是由 Context 管理的值对象；Properties 的成员本身也可以是 Attribute 句柄，如 GroupSum 的 DenseI32ArrayAttr。

从旧代码阅读 `getAttr`/字典接口时，要确认它访问的是兼容视图、固有属性还是纯 discardable 字典，不能假设所有语义字段都在同一个物理字典内。通常优先使用生成访问器，修改后用 generic print 和 verifier 检查完整信息。

## 检查与依据

练习可从删除 seed 开始：先预测分段长度与结果，再执行观察。然后把一组长度改错，区分“分段总数不成立”和“输入权重不等长”两类诊断。不要通过修改生成的 .inc 修复实例错误。

本章固定 LLVM 20.1.8。依据为 [ODS 规范](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)、[OperandRange 实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/IR/OperationSupport.cpp)及 [Operation 的属性/Properties 接口](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/Operation.h)。[在线 ODS 文档](https://mlir.llvm.org/docs/DefiningDialects/Operations/)适合查分类，代码以固定版本核对。

[20 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/20-ods-storage)展示分组、修改、实际展开和错误诊断。下一篇[参数存储与类型/属性接口](./storage_interfaces)转向 Type/Attribute 的共享身份和数据所有权。
