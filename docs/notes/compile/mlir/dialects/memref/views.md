---
order: 3
title: Subview 与别名关系
updated: 2026-10-04
---

# Subview 与别名关系

图像算法经常围绕 ROI 或 tile 进行计算。切出一个区域后，理想情况是直接操作已有存储中的那一块，而不为每个窗口复制数据。

`memref.subview` 表达的正是这种关系：**产生一个新的访问视图，底层存储仍然来自原视图。** 新的 SSA 名字不意味着新的数组。这种共享使局部更新高效，也要求编译器正确处理别名与生命周期。

## 1. 从窗口坐标映射回原坐标

沿用上一章的 `4×4` 数组，下面截取从 `[1,1]` 开始的 `2×2` 窗口：

```text
%w = memref.subview %m[1, 1][2, 2][1, 1]
  : memref<4x4xi32> to memref<2x2xi32, strided<[4, 1], offset: 5>>
```

紧跟 `%m` 的三组参数依次是：在源视图中的起点、结果的各维大小、在源坐标中采样的步幅。对这个例子：

```text
window[i,j] → source[1+i, 1+j]
```

最后一组 `[1,1]` 容易和结果类型中的 `[4,1]` 混淆。前者回答“窗口索引加一，源坐标加多少”；后者回答“窗口索引加一，线性存储偏移加多少”。两者单位不同，要通过源视图的布局联系起来。

## 2. 连续切片与跨步切片

在不降 rank 的情况下，设源视图 offset 为 `o`，各维步长为 `s[d]`，subview 起点为 `a[d]`，采样步幅为 `k[d]`，则结果布局为：

```text
newOffset    = o + Σ a[d] × s[d]
newStride[d] = s[d] × k[d]
newSize[d]   = 指定的结果大小
```

这不是 subview 独有的神秘算法，只是把“结果坐标 → 源坐标 → 存储偏移”两次代入合成一次。

因此，从 `4×4` 原数组每隔一个元素采样，得到：

```text
%step = memref.subview %m[0, 0][2, 2][2, 2]
  : memref<4x4xi32> to memref<2x2xi32, strided<[8, 2]>>
```

结果四个位置对应原数组 `[0,0]`、`[0,2]`、`[2,0]`、`[2,2]`。若原内容为 `0…15`，窗口内容就是 `[[0,2],[8,10]]`。它仍然没有复制数据。

subview 参数必须描述合法范围。动态尺寸与起点可能使合法性依赖运行时值，不能把“工具成功解析了 subview”视为所有调用都不会越界。

## 3. 嵌套视图的偏移组合

视图还可以继续切片。在 offset 为 5 的窗口上，从窗口 `[1,0]` 再取一行，新的 offset 为 `5 + 1×4 + 0×1 = 9`。

下面的完整函数同时观察嵌套窗口与跨步窗口：

<!-- memory-example: nested-view -->
```text
module {
  func.func @nested(%m: memref<4x4xi32>) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %w = memref.subview %m[1, 1][2, 2][1, 1]
      : memref<4x4xi32> to memref<2x2xi32, strided<[4, 1], offset: 5>>
    %s = memref.subview %w[1, 0][1, 2][1, 1]
      : memref<2x2xi32, strided<[4, 1], offset: 5>>
        to memref<1x2xi32, strided<[4, 1], offset: 9>>
    %step = memref.subview %m[0, 0][2, 2][2, 2]
      : memref<4x4xi32> to memref<2x2xi32, strided<[8, 2]>>
    %c1 = arith.constant 1 : index
    %a = memref.load %s[%c0, %c0] : memref<1x2xi32, strided<[4, 1], offset: 9>>
    %b = memref.load %step[%c1, %c1] : memref<2x2xi32, strided<[8, 2]>>
    return %a, %b : i32, i32
  }
}
```

对 `0…15` 的原数组，它返回 `(9,10)`。第一个数来自原数组 `[2,1]`，第二个来自 `[2,2]`。这两次访问虽经过不同的视图链，最终都可以还原为原数组的坐标。

这里不应把 `memref.reinterpret_cast` 当成另一种 subview 来套用。Subview 的位置与步幅相对输入视图组合；reinterpret_cast 则以底层存储为基础指定新的元数据，不会按上述方式自动叠加输入视图的 offset。需要窗口相对切片时，使用 subview 的语义更直接。

## 4. 共享分配与访问重叠

回到上一章的写入：

```text
memref.store %v, %w[%c0, %c1] : memref<2x2xi32, strided<[4,1], offset: 5>>
%r = memref.load %m[%c1, %c2] : memref<4x4xi32>
```

两次访问都落到线性偏移 6，因此 load 读到刚刚写入的值。`%w` 和 `%m` 是不同的 SSA Value，却存在别名关系。

考虑另外两个窗口：原数组第一行和最后一行。它们访问的元素没有重叠，但底层分配相同。这引出两个不同层次的问题：

| 编译器正在做的决定 | 需要区分什么 |
|---|---|
| 能否交换两次访存、并行更新两个窗口 | 实际访问的位置是否可能重叠，读写方向是什么 |
| 某个窗口使用结束后能否释放 | 底层分配是否仍被其他视图使用 |

第一个问题有时需要精确到索引；第二个问题必须追踪整次分配。即使窗口之间不重叠，释放共享分配仍会同时使它们失效。

类型相同也不足以断言别名：两个函数参数可以拥有完全相同的布局类型，却来自两次独立分配。类型描述访问形式，具体 Value 的来源及调用约定才决定是否共享存储。

## 5. 视图、复制与所有权

如果需要独立存储，就必须把这个需求表达出来。例如下面是局部片段，`%w` 已经是前面的窗口：

```text
%packed = memref.alloc() : memref<2x2xi32>
memref.copy %w, %packed
  : memref<2x2xi32, strided<[4,1], offset: 5>> to memref<2x2xi32>
```

复制按对应逻辑坐标搬运元素。新 `%packed` 的行步长为 2；修改它不会修改 `%w`。当这份新分配不再使用时，由适当的拥有者释放它。

与之相对，不能因为 `%w` 是新产生的 Value 就对 `%w` 单独 `dealloc`。Subview 没有创建新的分配；`memref.dealloc` 也不应作用于这种别名视图。手写程序需要在所有视图访问结束后，释放原始分配。自动释放机制则会追踪并使用适当的底层分配信息。

这也解释了与 `tensor.extract_slice` 的区别。Tensor slice 先表达一个不可变的张量值；MemRef subview 明确共享可修改存储。是否能把前者实现成后者，要看后续写入会不会破坏仍然需要的旧张量值。[Bufferization](../../compiler/memory/bufferization)正是衔接这两种语义的下一步。

## 阅读自查

1. `memref.subview %m[1,0][2,2][1,2]` 作用于连续 `4×4` 数组时，offset 与 strides 分别是多少？
2. 两个不重叠窗口为什么仍然不能各自释放一次底层存储？
3. 修改 subview 和修改通过 `memref.copy` 得到的新数组，分别会影响哪些观察者？

## 资料与实践

LLVM 20.1.8 下的 subview、reinterpret_cast、copy 和 dealloc 契约见[操作定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)，布局推导可从[MemRefOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp)中的 `SubViewOp::inferResultType` 继续查阅。

可选实践：[视图、别名与数值观察](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/10-memory-bufferization)。
