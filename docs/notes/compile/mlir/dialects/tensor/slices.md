---
order: 2
title: 切片与张量更新
updated: 2026-10-03
---

# 切片与张量更新

图像算法经常先取出一个窗口，处理后再放回原图。Tensor IR 可以直接保留这一层含义，而不用立即把窗口展开成一组地址。

本章从一个 4×4 矩阵的 2×2 窗口开始。主线是：确定窗口坐标 → 取出窗口值 → 修改局部值 → 产生更新后的完整矩阵。

## 1. 原矩阵与窗口坐标

输入设为：

```text
A = [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11],
     [12, 13, 14, 15]]
```

从原矩阵的 [1,1] 开始取 2×2 区域，结果应为：

```text
ROI = [[5,  6],
       [9, 10]]
```

窗口自己的 [0,0] 对应 A[1,1]。因此，必须区分原矩阵坐标与局部坐标；后面的三组参数正是描述这个对应关系。

## 2. Offset、Size 与 Stride

取出窗口的核心操作是：

```text
%roi = tensor.extract_slice %a[1, 1][2, 2][1, 1]
  : tensor<4x4xi32> to tensor<2x2xi32>
```

三组方括号依次是 offsets、sizes、strides：

| 参数 | 本例 | 含义 |
|---|---|---|
| offsets | `[1,1]` | 局部坐标零点位于原矩阵哪里 |
| sizes | `[2,2]` | 每一维取多少个元素 |
| strides | `[1,1]` | 局部索引增加 1 时，原索引增加多少 |

对于每个维度，索引对应为：

```text
source_index = offset + local_index * stride
```

所以窗口 [1,0] 映射为原矩阵 [1+1,1+0]=[2,1]，取到 9。这里的 size 是元素个数，不是结束位置；不能按 Python 中 `start:stop` 的 stop 来读第二组参数。

对非空窗口，一个维度最后访问的位置是 `offset+(size-1)*stride`。这个位置及其他被访问位置必须落在源范围内。动态窗口需要程序维护这些前提，不会因为写了 extract_slice 就自动裁剪边界。

## 3. 修改窗口并插回矩阵

把 ROI[0,0] 改为 99，再产生更新后的完整矩阵：

<!-- tensor-example: patch -->
```text
module {
  func.func @patch(%a: tensor<4x4xi32>) -> (tensor<2x2xi32>, tensor<4x4xi32>) {
    %roi = tensor.extract_slice %a[1, 1][2, 2][1, 1]
      : tensor<4x4xi32> to tensor<2x2xi32>
    %c0 = arith.constant 0 : index
    %v = arith.constant 99 : i32
    %changed = tensor.insert %v into %roi[%c0, %c0] : tensor<2x2xi32>
    %b = tensor.insert_slice %changed into %a[1, 1][2, 2][1, 1]
      : tensor<2x2xi32> into tensor<4x4xi32>
    return %roi, %b : tensor<2x2xi32>, tensor<4x4xi32>
  }
}
```

可以逐个 Value 追踪：

| 值 | 元素内容 |
|---|---|
| `%roi` | `[[5,6],[9,10]]` |
| `%changed` | `[[99,6],[9,10]]` |
| `%b` | 原矩阵窗口被 changed 替换，窗口之外保持 A 的内容 |

最终 B 为：

```text
[[ 0,  1,  2,  3],
 [ 4, 99,  6,  7],
 [ 8,  9, 10, 11],
 [12, 13, 14, 15]]
```

`tensor.insert_slice` 将 changed 的局部坐标映射回目标矩阵的窗口。其结果是新的完整张量，类型与目标 A 一致。原 A 与原 ROI 仍然保持各自的值；返回 `%roi` 时，其中的第一个元素仍是 5。

这里并不是先“拿到指向 A 的可写窗口”再通过它修改 A。物理上能否使用视图、是否复用存储，由后续表示和分析决定。Tensor 层先规定程序最终能够观察到什么。

## 4. 步长切片与下采样

若每隔一个位置取一个元素：

<!-- tensor-example: strided-slice -->
```text
module {
  func.func @subsample(%a: tensor<4x4xi32>) -> tensor<2x2xi32> {
    %r = tensor.extract_slice %a[0, 0][2, 2][2, 2]
      : tensor<4x4xi32> to tensor<2x2xi32>
    return %r : tensor<2x2xi32>
  }
}
```

局部 [0,0]、[0,1]、[1,0]、[1,1] 分别访问原 [0,0]、[0,2]、[2,0]、[2,2]，因此结果为：

```text
[[0,  2],
 [8, 10]]
```

stride 改变了对应的元素集合，并不只是改变打印格式。如果把一个同样形状的 patch 用这些参数插回去，只有上述四个离散位置被替换，其间其他元素仍保留目标值。

这也说明 `tensor<2x2xi32>` 只描述结果形状和元素类型；它无法单独告诉你这个值来自哪个窗口。来源关系需要从产生它的操作中读取。

## 5. 动态窗口与形状关系

窗口的位置和大小也可以来自函数输入：

<!-- tensor-example: dynamic-slice -->
```text
module {
  func.func @window(%a: tensor<?x?xf32>, %row: index, %col: index,
                    %height: index, %width: index) -> tensor<?x?xf32> {
    %r = tensor.extract_slice %a[%row, %col][%height, %width][1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }
}
```

此时 `%height`、`%width` 决定结果实例的尺寸，结果类型用 `?` 保留动态性。offsets 的 `%row/%col` 不决定结果大小，而是决定窗口在源里的位置。

对于正大小、步长为 1 的窗口，需要满足：

```text
0 <= row, 0 <= col
row + height <= source_rows
col + width  <= source_cols
```

这些是调用或显式检查应维护的条件。例子中的函数只表达窗口计算，没有替调用方插入检查。处理零大小、负输入或边界溢出时，应按具体操作约束和整数计算方式另行设计，不能把正常窗口公式当成完整的安全检查实现。

## 6. 降秩切片与单位维度

如果只取第 2 行，先得到的窗口形状是 1×4。由于第一维只有一个位置，可以让结果直接表示成长度为 4 的向量：

<!-- tensor-example: rank-reduce -->
```text
module {
  func.func @row(%a: tensor<4x4xi32>) -> tensor<4xi32> {
    %r = tensor.extract_slice %a[2, 0][1, 4][1, 1]
      : tensor<4x4xi32> to tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
```

offset、size、stride 的数量仍对应源的两个维度。结果类型省略了 size 为 1 的维度，因此局部向量 `%r[j]` 对应 A[2,j]。

这种 rank reduction 只去掉允许省略的单位维度。它不能把 2×2 窗口随意变成长度为 4 的向量；后者需要重新组合坐标关系，属于下一章的 reshape。

有多个单位维度时，可能存在不同的合法降秩方式。读代码应结合显式结果类型确认保留了哪些维度，不要只根据 rank 变小就猜索引对应。

## 理解检查与衔接

把 offsets 改为 `[0,1]`，sizes 和 strides 保持原样，ROI 应取哪些数？如果只修改 `%changed` 而不执行 insert_slice，为什么完整矩阵不会自动出现新值？为什么 size 为 `[2,2]` 的窗口不能靠降秩直接得到 `tensor<4xi32>`？

继续阅读[形状变换与元素对应](./reshape)，区分“选择一部分元素”和“重新组织全部元素的索引”。

## 实现与查阅

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)保留静态、步长、动态与降秩输入，验证局部和完整结果。固定版本的定义见 [TensorOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td) 中 ExtractSliceOp 与 InsertSliceOp；形状推导和验证见 [TensorOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp)。
