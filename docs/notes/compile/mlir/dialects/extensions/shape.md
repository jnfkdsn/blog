---
order: 30
title: Shape 计算、约束与 Witness
updated: 2026-10-04
---

# Shape 计算、约束与 Witness

Tensor类型中的问号说明某个尺寸在编译时未知，却没有告诉程序怎样计算输出形状，也没有保证两个动态输入一定能广播。Shape方言把尺寸计算与条件表示为IR，使编译器能够折叠静态事实，并把剩余条件转换为运行时检查。

本章只追踪广播这一项任务：输入形状[2,1]与[3]得到[2,3]；若尺寸来自运行时，先建立可广播条件，再让依赖它的计算继续。

## 1. Shape 本身也能成为 SSA 值

```text
%a = shape.const_shape [2, 1] : tensor<2xindex>
%b = shape.const_shape [3] : tensor<1xindex>
%r = shape.broadcast %a, %b
    : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
```

这里`tensor<2xindex>`是保存两个尺寸的extent tensor，不是形状为[2,1]的数据张量。广播从末维对齐：1与3可以扩展成3，前面的2保留，因此r保存[2,3]。

固定版本canonicalize将其折叠为一条shape.const_shape [2,3]。这个过程没有分配一个2×3的数据结果，只计算了后续数据操作需要的尺寸。

Shape还提供!shape.shape和!shape.size等类型，具有与普通index/extent tensor不同的表示能力，包括错误状态的建模。不能把它们只视为不同拼写；选择哪一种会影响后续可用的转换路径。

## 2. 未知尺寸需要条件

对于`tensor<?x?xf32>`与`tensor<?xf32>`，可先取得形状：

```text
%as = shape.shape_of %a : tensor<?x?xf32> -> tensor<2xindex>
%bs = shape.shape_of %b : tensor<?xf32> -> tensor<1xindex>
%w = shape.cstr_broadcastable %as, %bs
    : tensor<2xindex>, tensor<1xindex>
```

结果w为!shape.witness，表示依赖代码需要满足的形状约束。它不是普通布尔值，不能直接当作scf.if的i1条件。cstr操作表达断言式要求，相关计算必须遵守它所建立的执行顺序和事实。

当输入换回[2,1]与[3]，同一约束可折叠为shape.const_witness true，因为编译器已经确认条件。

## 3. 依赖事实的区域

依赖w的计算放入shape.assuming：

```text
%r = shape.assuming %w -> (tensor<?xindex>) {
  %s = shape.broadcast %as, %bs
      : tensor<2xindex>, tensor<1xindex> -> tensor<?xindex>
  shape.assuming_yield %s : tensor<?xindex>
}
```

区域表达“在这些约束已得到满足的前提下继续计算”。它不是一个错误时自动跳过的if分支；完整lowering应把约束兑现成检查、静态证明或项目明确保证，再消去这种中间表示。

这也连接前面的Region协议：yield把内部shape值交给父操作结果；外部不能直接引用内部s。

## 4. 从约束到明确检查

实际运行convert-shape-constraints并canonicalize后，动态主例出现：

```text
%ok = shape.is_broadcastable %as, %bs
    : tensor<2xindex>, tensor<1xindex>
cf.assert %ok, "required broadcastable shapes"
%r = shape.broadcast %as, %bs
    : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
```

现在is_broadcastable产生i1，cf.assert承担明确的断言，assuming可被去掉。输出中仍有shape操作；这一步只是把条件执行顺序落到更普通的控制/效果表示，不是最终机器执行。

不能为了让pipeline通过，随手将所有witness替换为true。那会删除未被证明的运行时前提，使不兼容尺寸进入后续计算。

## 5. Verifier 与运行时合法性

把静态输入改成[2,2]与[3]，两者不能广播。但固定实验的普通解析与canonicalize并不会把所有此类矛盾都变成verifier错误；相关约束会保留。IR结构可验证，与运行时条件成立是两件事。

因此“mlir-opt退出成功”不能证明任意动态形状都合法。应检查所需约束是否被证明、是否变为运行时断言，以及目标执行策略如何报告不满足条件。

[类型/形状推导](../../compiler/ir_definition/type_shape_inference)中的InferType与Reify提供另一类能力：操作向消费者解释结果类型或尺寸。它们不要求所有项目都显式使用Shape方言。Shape提供可组合的形状计算/约束表示，具体编译器可以选择自己的表示和消解路径。

## 依据与实践

固定来源：[Shape设计](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/ShapeDialect.md)、[ShapeOps](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Shape/IR/ShapeOps.td)、[约束转换](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/ShapeToStandard/ConvertShapeConstraints.cpp)。

[25 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/25-domain-extensions)核对[2,3]折叠、true witness、动态cf.assert和不兼容条件保留；未执行动态失败输入。有限练习：给定[M,1]和[N]，列出可静态确认的事实，再与[M,K]和[N]比较。
