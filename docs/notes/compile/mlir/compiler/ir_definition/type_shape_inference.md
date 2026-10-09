---
order: 60
title: 结果类型推导与运行时形状
updated: 2026-10-04
---

# 结果类型推导与运行时形状

假设编译器只需要知道 `my.add_one` 结果有多长，并不需要计算结果的元素。若尺寸与输入相同，先执行整个加一再取得尺寸就多做了工作。本章从这次尺寸查询出发，解释“推导结果 Type”与“构造运行时尺寸值”为什么是两种能力，以及通用变换怎样使用它们。

## 1. 一个动态维度的两个问题

沿用[自定义 Bufferization](../memory/custom_operations)中的逐元素操作。输入和 init 都是一维 f32 Tensor，运行时长度必须相同：

```mlir
%r = my.add_one %input outs(%init) : tensor<?xf32>
%c0 = arith.constant 0 : index
%n = tensor.dim %r, %c0 : tensor<?xf32>
```

编译器在构造 r 时已经知道它的类型是 `tensor<?xf32>`：rank 为 1、元素为 f32，长度不是编译期常量。但执行时 `%n` 仍要成为具体整数，例如 17。

于是有两个不同的答案：

| 问题 | 答案的形式 | 本例结果 |
|---|---|---|
| r 应具有哪种 Type | 编译器中的 Type 对象 | `tensor<?xf32>` |
| r 的第 0 维在运行时是多少 | 目标程序中的 SSA 值或静态常量 | `tensor.dim %input, 0` |

类型里的问号是未知尺寸标记，不是引用某个 SSA Value。不能通过“把 ? 换成 %n”构造普通 RankedTensorType 来解决第二个问题。

## 2. 结果类型从已知输入推导

本例明确要求 operands 与 result 类型相同。ODS 声明保留这一约束，并接入结果类型推导：

```tablegen
[Pure, SameOperandsAndResultType,
 DeclareOpInterfaceMethods<InferTypeOpInterface>,
 DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]
```

其余 operand/result 定义仍约束 ranked f32 Tensor，手写 verifier 再要求 rank=1。在固定版本中，TableGen 能利用相同类型约束生成 `inferReturnTypes`，不需要再手写一份同名实现。

因此编译器在已有两个 operands 时，可以选择不显式传入结果类型的 builder：

```cpp
my::AddOneOp::build(builder, state, input, init);
```

生成的构造代码调用类型推导，把得到的 Type 放入 OperationState 的结果类型列表。随后创建的 Operation 仍具有明确的结果类型；“推导”没有取消 IR 的类型系统。

作者程序用静态和动态输入分别观察到 `tensor<4xf32>` 与 `tensor<?xf32>`。这个过程只发生在编译器里，没有执行加一，也没有读目标数组的内容。

## 3. 推导与验证的分工

推导回答“根据当前信息应该得到什么类型”；验证回答“这份操作实例是否满足全部约定”。生成的 builder 通常不是完整的错误输入防火墙。

例如两个输入分别为 `tensor<4xf32>` 和 `tensor<5xf32>`，违反同类型约束；两个输入都是 `tensor<2x2xf32>`，则违反本例的 rank=1 限制。操作仍需要由 verifier 拒绝，不能因为某个推导分支选出了一个 Type 就接受所有 operands。

动态情况还更进一步：两个 `tensor<?xf32>` 的 Type 可以相同，运行时长度却分别为 4 与 5。静态 verifier 无法仅凭这两个 Type 发现不一致。项目应通过调用契约、已有形状证明或运行时检查保证前提；本例采用调用契约，不把类型相等写成动态长度相等。

这也是 AI 编译器中形状工作的一条边界：静态类型事实、符号关系证明和执行时条件相关，却不能互相替代。

## 4. 把结果形状具体化

消费者希望把 `%n = dim(%r,0)` 改写为 input 的尺寸，需要操作提供“怎样计算结果各维”的规则。`ReifyRankedShapedTypeOpInterface` 的实现给出这种规则。

本例只有一个结果、一维形状，所以返回的外层列表含一个结果形状，内层列表含一个尺寸项。实现的核心为：

```cpp
auto type = getInput().getType();
if (type.isDynamicDim(0))
  shapes.push_back({builder.create<tensor::DimOp>(
      getLoc(), getInput(), 0).getResult()});
else
  shapes.push_back({builder.getIndexAttr(type.getDimSize(0))});
```

静态长度 4 可以直接表示为 Attribute；动态长度需要创建 `tensor.dim`，得到目标程序中的 index Value。接口用 `OpFoldResult` 容纳这两种形式，调用者再决定何时把静态常量物化为操作。

这里的“物化”只指建立计算尺寸的 IR，不是运行这段 IR。Tensor 真正有 17 项时，新的 dim 在执行中才产生 17。

## 5. 通用 Pass 怎样使用形状规则

`resolve-ranked-shaped-type-result-dims` 看到 dim 的输入由某个操作产生，便尝试通过形状具体化接口取得相应结果的各维。它按结果编号与维度编号选中替代项，再替换原来的 dim。

本例的实际变化是：

```mlir
// 之前
%r = my.add_one %input outs(%init) : tensor<?xf32>
%n = tensor.dim %r, %c0 : tensor<?xf32>

// 形状规则提供的替代
%n = tensor.dim %input, %c0 : tensor<?xf32>
```

如果 r 只用于尺寸查询，替换后 add_one 没有剩余 use。因为这个 Tensor 操作已声明无副作用，随后的清理可以删除整项元素计算。不能把这次删除全归给类型推导：真正的链是“具体化尺寸 → 替换查询 → 死操作清理”。

静态实例的尺寸最终变成 `arith.constant 4 : index`。动态实例仍保留 input 的 dim，因为编译器并不知道实际长度。两者都不再依赖计算出的 r。

## 6. 更一般的形状关系

加一的结果形状与输入相同，是最简单的关系。切片结果的尺寸来自 sizes 参数；拼接结果某一维可能是各输入对应维的和；reshape 还需考虑元素总数和合法拆分。

这些场景仍然沿同一过程展开：先说明形状关系，再决定哪些部分能成为静态类型事实，哪些部分必须生成运行时 IR，最后明确合法性由谁检查。不能把生成 `arith.addi` 的形状表达式本身当成越界、溢出或约束成立的证明。

`InferShapedTypeOpInterface` 还提供 shaped type components 与形状相关推导能力，适合需要部分形状信息的消费者。它不是本例必须额外实现的前置，也不是与 Reify 完全同义的别名。选接口应先看目标 Pass 实际查询什么，而不是把名字带 Infer 的接口全部加上。

## 7. 类型细化的传播边界

有时分析进一步证明 `%r` 长度为 17。把某个 Value 的 Type 改成 `tensor<17xf32>`，会影响使用它的操作、函数签名和区域参数约束。推导出更具体信息不意味着可以任意原地改类型。

消费者可能保留动态接口，在内部利用静态事实；也可能引入允许的 `tensor.cast`；跨越自定义类型或 ABI 时还需使用[类型转换](../conversion/type_conversion)中的全边界处理。形状推导提供事实，传播策略负责维持整份 IR 的一致性。

本章实验并不执行一个全程序 shape refinement 系统。它明确完成两项能力：生成 builder 推导 Type，以及一个真实 Pass 把结果 dim 改成输入 dim，并删除无用计算。

## 检查与依据

试着预测三个变化：把 4 改成动态长度后哪些常量消失；额外读取 r 的元素后 add_one 是否还能删除；把 init 的动态长度改成与 input 不同后，静态类型检查为什么不足以保证正确。

固定实现见 [InferType 接口定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/InferTypeOpInterface.td)与 [ResolveShapedTypeResultDims 消费者](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/MemRef/Transforms/ResolveShapedTypeResultDims.cpp)。[官方形状推导概述](https://mlir.llvm.org/docs/ShapeInference/)可帮助定位设计范围；具体 API 以固定版本为准。

[19 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/19-extension-interfaces)包含动态/静态输入、生成 builder 的观察与实际尺寸改写。下一篇[区域与调用接口](./region_interfaces)把“操作提供事实，通用消费者作决定”推广到控制流。
