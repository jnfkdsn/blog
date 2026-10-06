---
order: 1
title: 存储对象与索引访问
updated: 2026-10-04
---

# 存储对象与索引访问

矩阵加法的 Tensor IR 描述了输入矩阵和结果矩阵，却没有规定结果放在哪片内存里。计算真正执行时，每次读取和写入最终都要落到存储上。MemRef 就在这一步进入：它让 IR 能够表达“通过这个视图，访问这片存储的某个元素”。

理解 MemRef 可以沿一条很短的过程展开：**取得存储 → 按索引读写 → 在所有访问结束后释放。** 本章先把这条过程讲清楚；下一章再展开索引如何变成地址。

## 1. 从张量值到存储视图

先比较两段核心语句。假定 `%t` 是一个张量，`%m` 是可写的存储视图，索引都合法：

```text
%new = tensor.insert %v into %t[%i, %j] : tensor<2x3xi32>
memref.store %v, %m[%i, %j] : memref<2x3xi32>
```

第一句产生新张量 `%new`。原来的 `%t` 仍然代表更新前的值，后续使用 `%t` 必须得到旧内容。

第二句直接修改 `%m` 所指存储中 `[i,j]` 的内容。它没有返回一个新的 memref；后续通过 `%m`，或者通过另一个覆盖同一位置的视图读取，都会看到更新。

这里没有违反 SSA。`%m` 这个 SSA Value 始终标识同一个视图，发生变化的是被引用的内存内容。C++ 中一个不重新赋值的指针仍然可以指向可修改对象，也是这种“引用自身与被引用内容”的区别。

因此，`memref<2x3xi32>` 可以先理解成：一个二维视图，每个逻辑元素是 `i32`，两个维度大小为 2 和 3。视图携带访问存储所需的信息，但**拥有一个视图，并不自动意味着拥有释放底层分配的责任**。函数参数、子视图和新分配都可以具有 memref 类型，来源却不同。

## 2. 分配、写入与读取

下面的完整函数分配一个 `2×3` 数组，将 `[1,2]` 写为 7，再读取并返回它：

<!-- memory-example: storage -->
```text
module {
  func.func @storage() -> i32 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %v = arith.constant 7 : i32
    %m = memref.alloc() : memref<2x3xi32>
    memref.store %v, %m[%c1, %c2] : memref<2x3xi32>
    %r = memref.load %m[%c1, %c2] : memref<2x3xi32>
    memref.dealloc %m : memref<2x3xi32>
    return %r : i32
  }
}
```

执行过程中的关键状态如下：

| 步骤 | 存储状态 | 已知的标量值 |
|---|---|---|
| `memref.alloc` | 获得能容纳 6 个 `i32` 的存储，元素尚未初始化 | `%v = 7` |
| `memref.store` | `[1,2]` 变成 7，其余元素仍未初始化 | 不产生新结果 |
| `memref.load` | 存储不变 | `%r = 7` |
| `memref.dealloc` | 这次分配的生命周期结束 | `%r` 仍然是 7 |
| `return` | 不再访问已经释放的存储 | 返回 7 |

最后两步值得区分：load 已经取得一个独立的标量值，后面释放数组不会使 `%r` 失效。如果返回的是 `%m` 本身，或者释放后再次 load，则访问仍依赖已经失效的存储，含义就完全不同。

`alloc` 负责取得存储，不负责初始化。示例只读取刚刚写入的位置，所以不需要先填充整个数组；如果下一步要对六个元素求和，就必须确保六个元素都已经有定义。验证器接受一段 IR，并不代表它能够证明每次运行都没有未初始化读取。

## 3. 动态尺寸与运行时元数据

图像的宽高可能来自运行时输入。一个维度写成 `?`，表示类型中没有固定这个大小；并不是运行时也不知道大小。

<!-- memory-example: dynamic -->
```text
module {
  func.func @dynamic(%n: index) -> index {
    %m = memref.alloc(%n) : memref<?x3xi32>
    %c0 = arith.constant 0 : index
    %rows = memref.dim %m, %c0 : memref<?x3xi32>
    memref.dealloc %m : memref<?x3xi32>
    return %rows : index
  }
}
```

调用 `@dynamic(5)` 时，`alloc(%n)` 分配 `5×3` 个元素。`memref.dim` 查询第 0 维，得到 5。函数没有读取数组元素，因而不需要初始化内容。

类型和 SSA operand 各自提供一部分信息：

```text
memref<?x3xi32>：二维、列数为 3、元素为 i32
%n：本次执行的行数
二者合起来：本次视图的完整形状
```

动态分配参数只对应动态维度，按维度顺序提供。例如 `memref<?x?xi32>` 需要两个尺寸参数；`memref<4x?xi32>` 只需要第二维的大小。

索引用 `index` 表达。它适合尺寸和地址计算，但类型为 `index` 不会自动保证数值在范围内：对 `2×3` 视图，合法索引要求 `0 ≤ i < 2` 且 `0 ≤ j < 3`。普通 `memref.load/store` 不承诺自动插入越界检查。动态范围需要调用方保证，或在 IR 中明确表达所需检查。

## 4. 分配来源与生命周期

上面的函数自己 `alloc`，也自己 `dealloc`，责任很清楚。实际程序还会通过函数参数借用已有存储：

```text
func.func @read(%m: memref<2x3xi32>, %i: index, %j: index) -> i32 {
  %v = memref.load %m[%i, %j] : memref<2x3xi32>
  return %v : i32
}
```

这个函数只负责读取。传入的存储由谁分配、何时释放，要遵守调用约定。仅凭 `%m` 的类型不能推导“本函数应该释放它”。

另一种局部临时存储可以由 `memref.alloca` 创建。它受最近具有 `AutomaticAllocationScope` 约束的外围操作管理，例如函数或显式 `memref.alloca_scope`；退出该作用域时自动结束生命周期。不能把任意 Block 的结束都理解成 alloca 的释放点，也不应对这种分配再使用 `memref.dealloc`。

选择 alloc 还是 alloca，牵涉分配大小、作用域、目标后端和使用方式。当前先建立一个共同原则：**最后一次访问必须发生在存储生命周期结束之前，而视图能被传递的范围不能超过这段生命周期。** 自动安排释放会在[所有权章节](../../compiler/memory/ownership)展开。

## 5. MemRef 在编译流程中的位置

有了 MemRef，不代表高层计算已经全部展开。Linalg 可以保留矩阵乘法或结构化循环的抽象，同时通过 memref operands 读写存储；之后再把计算降低为循环和标量访存。

```text
Tensor / Linalg：描述张量计算与新结果
          ↓ Bufferization
MemRef / Linalg：选择存储，计算结构仍保留
          ↓ 循环展开
SCF / MemRef / Arith：逐次索引、读写和运算
```

本章解决的是中间这层“存储存在且可被访问”的含义。接下来还需要解释：同为一个 `2×2` 窗口，为什么行步长可能是 2，也可能是 4？这正是[布局、偏移与步长](./layout)的主题。

## 阅读自查

1. 在两次 `memref.load %m[%i]` 中间插入同位置的 store，为什么不能仅凭 operands 相同就合并两次 load？
2. 上例 dealloc 后返回 `%r` 为什么成立，返回 `%m` 为什么不成立？
3. `memref<?x3xi32>` 中的 `?` 分别对编译时类型和运行时视图意味着什么？

## 资料与实践

示例按 LLVM 20.1.8 验证；alloc、load/store、dealloc、alloca 的契约见[固定版本 MemRef 操作定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)，也可查阅[官方 MemRef 文档](https://mlir.llvm.org/docs/Dialects/MemRef/)。

可选实践：[10-memory-bufferization](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/10-memory-bufferization)；其中 `storage.mlir`、`dynamic.mlir` 与本文一致。
