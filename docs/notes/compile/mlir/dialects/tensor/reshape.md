---
order: 3
title: 形状变换与元素对应
updated: 2026-10-03
---

# 形状变换与元素对应

同一组六个元素可以排成 2×3，也可以排成 3×2。两者元素数相同，但每个二维坐标对应哪个元素，需要有明确规则。只写“修改 shape”不足以说明计算含义。

本章沿一个矩阵先展平再拆维，解释 reshape 如何建立新坐标，并与转置、切片和类型信息调整区分。

## 1. 元素顺序与新坐标

仍使用：

```text
A = [[1, 2, 3],
     [4, 5, 6]]
```

按该稠密张量的逻辑维度顺序，将相邻的两个维度合并，得到 `[1,2,3,4,5,6]`。再把这个长度 6 的维度拆为 3×2，得到：

```text
R = [[1, 2],
     [3, 4],
     [5, 6]]
```

为了追踪某个元素，引入逻辑线性编号 p。原坐标 `(i,j)` 对应 `p=i*3+j`；新坐标 `(u,v)` 对应 `p=u*2+v`。例如原 A[0,2]=3 的 p 为 2，进入新 R[1,0]。

这里先解释逻辑元素对应，没有指定底层物理地址。是否能够用相同存储和新视图实现，要结合后续布局与 bufferization 决定。

## 2. Collapse 与 Expand 的完整过程

上述两步可以写为：

<!-- tensor-example: reshape -->
```text
module {
  func.func @reshape(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %flat = tensor.collapse_shape %a [[0, 1]]
      : tensor<2x3xf32> into tensor<6xf32>
    %r = tensor.expand_shape %flat [[0, 1]] output_shape [3, 2]
      : tensor<6xf32> into tensor<3x2xf32>
    return %r : tensor<3x2xf32>
  }
}
```

`collapse_shape` 将两个相邻维度合成一个，`expand_shape` 再把一个维度拆成两个。

`[[0,1]]` 是 reassociation，即维度分组。它指的是维度编号，不是元素索引，也不是新 shape `[0,1]`。在 collapse 中，高 rank 输入的第 0、1 维合为一个；在 expand 中，高 rank 输出的第 0、1 维对应低 rank 输入的那一维。

对本例，必须同时满足 `2*3=6` 和 `3*2=6`。维度变了，元素总数和上述顺序保持一致。

## 3. 相邻维度分组

如果输入是 `tensor<2x3x4xf32>`，可以有：

```text
[[0,1],[2]]：把前两维合并，得到 6×4
[[0],[1,2]]：把后两维合并，得到 2×12
```

分组按顺序覆盖相邻维度。不能用 `[[0,2],[1]]` 把不相邻的第 0、2 维当成普通连续分组合并；这会跨过第 1 维，改变当前机制所描述的关系。

有时算法先要交换维度，再合并它们。这时应分别表达重排与 reshape，让后续编译器清楚哪些步骤可能涉及不同的数据组织，而不是把所有变化藏进一个形状标签。

## 4. 动态尺寸与输出形状

如果原来的行数未知，仍可以先展平，再按原尺寸恢复：

<!-- tensor-example: dynamic-reshape -->
```text
module {
  func.func @flatten_restore(%a: tensor<?x3xf32>) -> tensor<?x3xf32> {
    %c0 = arith.constant 0 : index
    %rows = tensor.dim %a, %c0 : tensor<?x3xf32>
    %flat = tensor.collapse_shape %a [[0, 1]]
      : tensor<?x3xf32> into tensor<?xf32>
    %r = tensor.expand_shape %flat [[0, 1]] output_shape [%rows, 3]
      : tensor<?xf32> into tensor<?x3xf32>
    return %r : tensor<?x3xf32>
  }
}
```

当 `%rows=5` 时，flat 的实际长度为 15；expand 的 `output_shape [%rows,3]` 指定恢复为 5×3。

之所以取原来的 rows，而不是任意输入，是为了让乘积关系有依据。对于另一段接收扁平张量及外部 `%rows` 的代码，需要另外保证 `flat_length = rows*3`。类型里的 `?` 不会自动建立这个等式，也不会在不满足时补元素或截断。

动态 `output_shape` 提供真正的 SSA 尺寸；静态 3 直接写在列表和结果类型中。固定版本的语法将这两类信息放在同一组输出尺寸里。

## 5. tensor.reshape 的形状操作数

除了相邻维度分组，还可以显式提供一个描述目标 shape 的一维张量：

<!-- tensor-example: reshape-shape-operand -->
```text
module {
  func.func @reshape_shape(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %shape = arith.constant dense<[3, 2]> : tensor<2xi64>
    %r = tensor.reshape %a(%shape)
      : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
    return %r : tensor<3x2xf32>
  }
}
```

这里 `%shape` 的两个元素是 3 和 2；其自身类型 `tensor<2xi64>` 中的 2 表示“目标有两个维度”。不要将 shape 张量的长度与被 reshape 数据的元素数混淆。

这种接口便于接收来自程序计算的 shape。reassociation 则直接保留维度之间的分组关系，适合编译器围绕合并与拆分进行推理。两种写法都要求元素类型与元素总数等约束成立。

本例用常量 shape 使关系清楚。若 shape 值在运行时产生，需要由程序确保尺寸和总元素数一致；通过静态验证并不意味着所有可能的 shape 输入都正确。

## 6. Reshape、转置与 Cast 的区别

对同一个 2×3 的 A：

| 操作 | 结果或含义 |
|---|---|
| reshape 到 3×2 | `[[1,2],[3,4],[5,6]]` |
| 转置到 3×2 | `[[1,4],[2,5],[3,6]]` |
| cast 到 `tensor<?x3xf32>` | 元素和实际 2×3 形状不变，只调整类型保留的信息 |
| 取一行 slice | 选择原张量的一部分元素 |

reshape 与转置碰巧有相同结果形状，却有不同坐标对应；同 shape 不等于同值。cast 则不能把两个不同的已知静态 shape 当成等价类型随意转换。

在 Tensor 层先分清这些语义，再到存储层判断实现成本。一个 reshape 在某种连续布局上可能只是视图变化，在更复杂来源上则要考虑布局或额外处理；不能仅凭操作名承诺零拷贝。

## 理解检查与衔接

原 A[1,1]=5 在 3×2 的 reshape 结果中位于哪里？为什么转置后的坐标不同？若动态 flat 长度为 10，是否可以直接 expand 为 `%rows×3` 并令 `%rows=3`？

到这里，张量值、形状、局部窗口和重新分组已经有了明确含义。下一步进入 [Linalg 的迭代空间与访问映射](../linalg/iteration_maps)，把这些值组织成完整计算。

## 实现与查阅

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)包含静态、动态和 shape 操作数三种输入。固定版本定义见 [TensorOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td) 中 CollapseShapeOp、ExpandShapeOp、ReshapeOp；静态不兼容分组由实际工具验证。不要将结构检查当作所有运行时尺寸已经被检查。
