---
order: 1
title: Tensor 值语义与形状
updated: 2026-10-03
---

# Tensor 值语义与形状

图像处理中常说“把某个像素改成 99”。如果用普通数组实现，这通常意味着写内存；在 tensor IR 中，则先表达“得到一个仅该位置不同的新张量”。这个区别决定了旧结果能否继续使用，也为后续存储复用提供了需要维护的语义。

本章从一次元素更新开始，再将矩阵尺寸推广到运行时决定，最后说明怎样构造新的张量。

## 1. 一次更新产生两个可区分的值

设 A 是：

```text
A = [[1, 2, 3],
     [4, 5, 6]]
```

把 [0,1] 更新为 99，得到 B：

```text
B = [[1, 99, 3],
     [4,  5, 6]]
```

这个操作之后，A[0,1] 仍是 2，B[0,1] 是 99。旧值和新值可以同时被程序使用。完整 IR 把这种区别直接写出来：

<!-- tensor-example: element-update -->
```text
module {
  func.func @update(%a: tensor<2x3xf32>, %value: f32) -> (f32, f32, tensor<2x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b = tensor.insert %value into %a[%c0, %c1] : tensor<2x3xf32>
    %old = tensor.extract %a[%c0, %c1] : tensor<2x3xf32>
    %new = tensor.extract %b[%c0, %c1] : tensor<2x3xf32>
    return %old, %new, %b : f32, f32, tensor<2x3xf32>
  }
}
```

`tensor.insert` 的结果是 `%b`，它没有把 `%a` 这个 SSA Value 改成另一份内容。后面两条 extract 分别读取两个值，所以对于上述输入，返回的前两个标量应为 2 和 99。

这就是值语义对实现提出的要求：无论底层怎样安排内存，观察旧 A 的程序都不能突然看到 B 的新元素。某些情况下可以复用存储，另一些情况下需要保留数据；需要维护的结果已经由这里的 IR 确定。

## 2. Tensor 类型描述哪些信息

`tensor<2x3xf32>` 中，2 和 3 是两个维度的长度，f32 是元素类型。它的 rank 为 2，元素数为 6。对这里的普通稠密张量，可以用两个索引选中一个标量。

类型本身没有写出六个元素的数值。数值来自函数输入、常量或产生张量的操作。`tensor<2x3xf32>` 也没有指定一个可供 C++ 随意读写的裸指针；地址、stride 和分配需要在存储表示中进一步解释。

Tensor 类型属于 builtin 类型系统，Tensor dialect 提供 extract、insert、slice 等操作。Linalg 操作同样可以产生和接收这种类型；看到 tensor 类型，不意味着整段计算都必须由 `tensor.*` 完成。

### 2.1 静态尺寸与动态尺寸

| 类型 | 编译时已知的信息 |
|---|---|
| `tensor<2x3xf32>` | rank 为 2，两个维度分别为 2、3 |
| `tensor<?x3xf32>` | rank 为 2，第二维为 3，第一维从实例取得 |
| `tensor<?x?xf32>` | rank 为 2，两个维度都从实例取得 |
| `tensor<*xf32>` | 类型中没有固定 rank |

`?` 不是负数，也不是程序运行时永远未知。某次调用传入 5×3 的张量时，它的第一维就是 5；只是编译时类型没有把 5 固定下来。

两个参数都写 `tensor<?x3xf32>`，并不保证它们运行时第一维相等。这一点在逐元素加法中会成为需要维护的形状前提。

## 3. tensor.dim 与运行时形状

将同一矩阵接口推广为行数可变：

<!-- tensor-example: dynamic-shape -->
```text
module {
  func.func @shape(%a: tensor<?x3xf32>) -> (index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = tensor.dim %a, %c0 : tensor<?x3xf32>
    %cols = tensor.dim %a, %c1 : tensor<?x3xf32>
    return %rows, %cols : index, index
  }
}
```

`tensor.dim %a, %c0` 返回第一维的长度。若输入实例是 5×3，则 `%rows=5`；第二条 dim 返回 3。

两者都是 index 类型的 SSA Value，所以可以参与循环边界、尺寸计算或构造新张量。区别在于编译器可以从静态类型直接推得第二维为 3，将那次 dim 折叠为常量；第一维通常需要保留动态查询。

把形状从类型取成 Value，是后续动态代码的重要连接：类型描述哪些尺寸已知，Value 则携带本次执行所需的尺寸。它不是让一个类型对象在运行时发生改变。

索引还必须指向存在的维度。对 rank 2 的张量查询第 2 维越出了合法范围，不能把 dim 当成会自动返回 0 的安全查询。

## 4. 新张量的形状与元素内容

### 4.1 tensor.empty 只提供结果形状

如果已知 `%rows`，可以写：

```text
%empty = tensor.empty(%rows) : tensor<?x3xf32>
```

动态 operand 按类型中动态维度出现的顺序提供，因此这里只传第一维。第二维已经由类型里的 3 给出。

`empty` 的内容是未指定的，名字也不表示“全零”。它适合供后续操作完整定义结果元素，例如矩阵逐元素加法。若后续要从已有内容累加，就必须先提供有意义的初始值；Linalg 的归约章节会具体展示。

### 4.2 tensor.generate 定义每个坐标的值

如果希望结果满足 `T[i,j] = 10*i+j`，可以直接用坐标构造：

<!-- tensor-example: generate -->
```text
module {
  func.func @coordinates(%rows: index) -> tensor<?x3xindex> {
    %c10 = arith.constant 10 : index
    %t = tensor.generate %rows {
    ^bb0(%i: index, %j: index):
      %base = arith.muli %i, %c10 : index
      %v = arith.addi %base, %j : index
      tensor.yield %v : index
    } : tensor<?x3xindex>
    return %t : tensor<?x3xindex>
  }
}
```

输入 `%rows=2` 时，结果为：

```text
[[ 0,  1,  2],
 [10, 11, 12]]
```

Region 的参数是当前元素的坐标，不是从一个输入张量取出的元素。对于 `(1,2)`，计算 `1*10+2`，yield 12，成为 T[1,2]。

body 的各次调用没有规定顺序。这个构造描述各点如何得到值，不适合依赖“上一个点已经执行完”来累积可变状态。对本例每个坐标独立计算，因而无需这样的顺序。

## 5. tensor.cast 与形状信息的调整

有时同一个值需要进入一个更通用的接口，例如从固定两行进入动态行数函数：

<!-- tensor-example: shape-cast -->
```text
module {
  func.func @forget_rows(%a: tensor<2x3xf32>) -> tensor<?x3xf32> {
    %r = tensor.cast %a : tensor<2x3xf32> to tensor<?x3xf32>
    return %r : tensor<?x3xf32>
  }
}
```

这个 cast 保留元素和实际形状，只让结果类型不再固定行数。它不把 2×3 重排成别的矩阵，也不改变 f32 元素。

相反，从 `tensor<?x3xf32>` 向 `tensor<2x3xf32>` 细化时，实际第一维必须确实为 2。cast 本身不是用户输入检查逻辑；不能对任意行数执行细化，并期待它自动裁剪或补齐。已知静态维度互相矛盾的 cast 会被 verifier 拒绝，动态实例是否满足前提还需要程序保证。

## 6. 验证约束与运行时前提

对于本章的元素访问，需要同时考虑：

- 索引数量与 rank 对应，插入标量的类型与元素类型一致。
- 实际索引落在本次张量的范围内。
- 对动态尺寸进行细化或组合时，相关等式成立。

类型和 operand 结构中的矛盾可以在验证时发现。依赖函数输入的边界则不会因为 parser/verifier 通过就自动成立；需要调用约定、已有证明或显式检查。

这与传统数组程序的边界问题相通，只是 MLIR 把一部分信息提前放进了类型和操作约束中。

## 理解检查与衔接

如果 `%b` 产生后再读取 `%a[0,1]`，为什么结果不能变成 99？如果两个输入都是 `tensor<?x3xf32>`，为什么仍可能无法逐元素相加？如果将 generate 的行数改为 0，结果应有多少个元素？

下一章把单点更新推广到[切片与张量更新](./slices)，追踪一个区域与原矩阵的坐标关系。

## 实现与查阅

本章的完整 IR 和关键数值对照收录于[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)。固定版本的定义见 [TensorOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td) 中 InsertOp、DimOp、EmptyOp、GenerateOp 和 CastOp；验证与折叠实现见 [TensorOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp)。先围绕上述前提查证，不需要通读文件。
