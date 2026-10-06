---
order: 2
title: 布局、偏移与步长
updated: 2026-10-04
---

# 布局、偏移与步长

对一个 `4×4` 图像块，只取中间的 `2×2` 区域时，逻辑形状变小了，但原数组并没有重新排列。这个小窗口的下一行仍然要跨过原数组的一整行。

这就是 MemRef 需要同时保存形状和布局的原因：**形状说明哪些索引有效，布局说明这些索引落到哪里。** 本章沿同一个窗口计算位置，建立 offset、size 和 stride 的含义。

## 1. 连续矩阵的地址计算

设原数组按行连续存放，元素内容等于它在线性数组里的序号：

```text
       列0  列1  列2  列3
行0     0    1    2    3
行1     4    5    6    7
行2     8    9   10   11
行3    12   13   14   15
```

对 `memref<4x4xi32>`，默认布局的行步长为 4，列步长为 1。访问 `[i,j]`，相对数据基址的元素偏移是：

```text
elementOffset(i, j) = 0 + i × 4 + j × 1
```

因此 `[1,2]` 对应第 6 个元素。如果这个目标上的 `i32` 占 4 字节，字节地址再增加 `6×4=24` 字节。MemRef 的 strided 布局中，offset 和 stride 以**元素**为单位，不能直接当成字节数。

这里的计算只是在解释地址对应。元素中恰好存了 6，是我们选择的测试数据；把它改写为 99 后，地址仍然是第 6 个元素的位置。

## 2. 子窗口的形状与布局

现在取从原数组 `[1,1]` 开始的 `2×2` 窗口：

```text
窗口逻辑坐标       原数组坐标      线性元素偏移
[0,0]             [1,1]          5
[0,1]             [1,2]          6
[1,0]             [2,1]          9
[1,1]             [2,2]         10
```

窗口第一项从偏移 5 开始；窗口向下一行仍要跨过 4 个元素。因此窗口的类型是：

```text
memref<2x2xi32, strided<[4, 1], offset: 5>>
```

对应公式变成：

```text
elementOffset(i, j) = 5 + i × 4 + j
```

`2×2` 是逻辑范围，`[4,1]` 是两个维度各前进一步时增加的元素偏移。两者不必吻合为紧密存放的数组。若把类型错误地写成默认的 `memref<2x2xi32>`，默认行步长会变成 2，第二行就会被解释为从另一处开始。

也由此可以区分“取一个视图”和“打包一份副本”：视图保留原存储的间隔；复制到新建的连续 `2×2` buffer 后，新 buffer 才具有 `[2,1]` 的步长。

## 3. 视图元数据的实际观察

下面这个完整函数创建窗口，向窗口 `[0,1]` 写入 99，然后从原数组 `[1,2]` 读取。它还返回窗口的 offset 和两个 stride：

<!-- memory-example: view -->
```text
module {
  func.func @window(%m: memref<4x4xi32>) -> (i32, index, index, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %v = arith.constant 99 : i32
    %w = memref.subview %m[1, 1][2, 2][1, 1]
      : memref<4x4xi32> to memref<2x2xi32, strided<[4, 1], offset: 5>>
    memref.store %v, %w[%c0, %c1]
      : memref<2x2xi32, strided<[4, 1], offset: 5>>
    %r = memref.load %m[%c1, %c2] : memref<4x4xi32>
    %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %w
      : memref<2x2xi32, strided<[4, 1], offset: 5>> -> memref<i32>, index, index, index, index, index
    return %r, %offset, %strides#0, %strides#1 : i32, index, index, index
  }
}
```

传入合法、可写的 `4×4` 存储后，返回值为：

```text
读取结果 = 99
offset   = 5
stride0  = 4
stride1  = 1
```

`extract_strided_metadata` 把访问需要的元数据显式取出来。`%sizes#0`、`%sizes#1` 对应两个维度大小，`%strides#0`、`%strides#1` 对应两个步长；这里的 `#0/#1` 是多结果操作的结果编号，不是在访问数组元素。

`%base` 提供底层存储的基视图。取得它没有分配或复制数组，也没有授予一份新的释放责任。示例关注索引映射，所以只返回几个标量。

## 4. 静态信息与动态信息

窗口的位置和尺寸可能运行时才知道，类型也可以写成：

```text
memref<?x?xi32, strided<[?, ?], offset: ?>>
```

这些 `?` 表示相应数字不固定在类型里；本次执行仍有具体 offset、sizes、strides。编译器可以把它们作为运行时元数据用于地址计算。

已知布局转为更一般的动态布局，并不要求重新排列元素：

<!-- memory-example: cast -->
```text
module {
  func.func @cast_view(%m: memref<4x4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %w = memref.subview %m[1, 1][2, 2][1, 1]
      : memref<4x4xi32> to memref<2x2xi32, strided<[4, 1], offset: 5>>
    %d = memref.cast %w : memref<2x2xi32, strided<[4, 1], offset: 5>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
    %r = memref.load %d[%c0, %c1] : memref<?x?xi32, strided<[?, ?], offset: ?>>
    return %r : i32
  }
}
```

`memref.cast` 之后，`%d` 的类型保留了更少的静态信息，但它仍指向同一个窗口。实际 offset 仍为 5，步长仍为 `[4,1]`。输入采用前面的 `0…15` 数据时，函数返回 6。

反方向从动态信息收紧成静态信息时，语义上是在断言相应大小和布局满足静态约束。是否生成显式运行时检查取决于所用的验证与 lowering 流程；不能把 cast 当作自动修正不匹配尺寸、复制或重新布局的通用操作。若确实要把 `[4,1]` 的窗口打包成 `[2,1]`，应取得目标存储并执行复制。

## 5. 布局信息与低层表示

对于这里讨论的 ranked strided memref，常见 CPU lowering 会通过描述符传递分配基址、用于访问的对齐基址、offset、sizes 和 strides。随后把这些信息组合成具体地址。

分配基址用于找到需要释放的那次分配，访问基址与 offset/strides 用于找到某个元素。存在对齐调整时，两种指针职责尤其需要分清。当前先掌握这种信息分工，完整字段类型、函数 ABI 和 LLVM lowering 留到[CPU Lowering](../../compiler/lowering/)展开。

上面的线性公式针对 strided 布局，并不概括所有可能的 affine 布局。内存空间则是另一条信息：它可以区分目标上的不同存储区域，但不会单独执行设备间拷贝。具体空间解释和合法布局由后端契约决定。

现在可以沿坐标找到元素了。下一章[Subview 与别名](./views)继续讨论：两个独立 SSA Value 指向同一片存储后，一次写入会影响谁，哪一个 Value 可以被用于释放。

## 阅读自查

1. 上述窗口 `[1,1]` 的元素偏移是多少？为什么不是 3？
2. 将窗口 cast 成全动态布局，实际元素位置是否变化？
3. 两个类型都是 `memref<2x2xi32, strided<[4,1], offset: 5>>` 的参数，是否一定指向同一块内存？

## 资料与实践

示例和元数据按 LLVM 20.1.8 验证。具体契约见[MemRef 类型定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/Builtin.md)、[MemRef 操作定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)。描述符的后续表示见[LLVM lowering 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/TargetLLVMIR.md)。

可选实践：[窗口与元数据实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/10-memory-bufferization)。
