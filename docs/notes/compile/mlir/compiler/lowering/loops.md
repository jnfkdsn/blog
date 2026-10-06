---
order: 1
title: 结构化计算到显式循环
updated: 2026-10-04
---

# 结构化计算到显式循环

Bufferization 已经决定张量结果由哪片存储承载，但 `linalg.generic` 仍然把一次完整计算表达成一个操作。进入 CPU 低层表示之前，还需要把“对全部坐标执行元素计算”展开成真正的迭代和访存。

本章继续使用矩阵加法：`out[i,j] = a[i,j] + b[i,j]`。我们先沿索引映射产生循环，再把循环携带的状态改为控制流传值。这样，lowering 的每一步都对应一项已经明确的计算职责。

## 1. 以输出 Buffer 为边界的加法

下面的函数接收三个二维存储视图，结果写入调用方提供的 `%out`：

<!-- cpu-example: add-into -->
```text
module {
  func.func @add_into(
      %a: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %b: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %out: memref<?x?xf32, strided<[?, ?], offset: ?>>)
      attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : memref<?x?xf32, strided<[?, ?], offset: ?>>,
                   memref<?x?xf32, strided<[?, ?], offset: ?>>)
      outs(%out : memref<?x?xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}
```

三者必须具有相同的运行时形状；索引涉及的存储必须有效，输出可写。当前教学调用还保证输出与输入不重叠，避免额外的跨迭代别名依赖。

类型中的动态 stride/offset 允许它接收连续矩阵，也允许接收其他数组中的窗口。它负责计算，不负责分配和释放参数。这是一种明确的接口选择，便于在后续章节从 C 传入已有数组。

`llvm.emit_c_interface` 标记用于后续函数转换生成 C wrapper，不参与矩阵加法的数值计算；其作用在 [ABI 章节](./abi)解释。

## 2. 从 Indexing Maps 还原访问

三个 indexing maps 都把迭代坐标 `(i,j)` 映射到 operand 坐标 `(i,j)`：

```text
当前迭代 (i,j)
    ├─ A[i,j] → 区域参数 x
    ├─ B[i,j] → 区域参数 y
    └─ Out[i,j] → 对应输出位置

区域计算 x+y → yield → 写到 Out[i,j]
```

两个 iterator 都标为 parallel，表示这个结构化计算没有沿这些维度的归约。它不要求本次 CPU lowering 一定启动多个线程；生成顺序嵌套循环仍是一种合法实现。是否并行执行属于后续映射选择。

区域中的 `%unused` 没有被计算使用，所以这一例不需要读取输出的旧元素。若 body 使用它进行累加，则必须保留读取初值的语义，不能沿用“只读两个输入”的推导。

## 3. 两重循环的实际结果

执行 `convert-linalg-to-loops` 并降低其中的 affine 索引表达后，得到：

<!-- cpu-output: add-loops -->
```text
module {
  func.func @add_into(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %dim_0 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg4] : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %1 = memref.load %arg1[%arg3, %arg4] : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %arg2[%arg3, %arg4] : memref<?x?xf32, strided<[?, ?], offset: ?>>
      }
    }
    return
  }
}
```

工具从输入的两个动态维度取得迭代上界。每一组 `%arg3,%arg4` 对应刚才的 `(i,j)`，两次 load、一次 add 和一次 store 对应区域的元素计算。

对 `2×3` 矩阵，迭代按当前生成顺序经过六个坐标。第一组坐标 `[0,0]` 若读出 1 与 10，写入 11；下一组 `[0,1]` 读出 2 与 20，写入 22。完整结果是 `[[11,22,33],[44,55,66]]`。

动态维度只改变上界的来源。行数为 0 时外层循环不进入，列数为 0 时内层循环不进入；该例没有元素读取或输出写入。调用方仍需提供满足接口表示要求的描述符，而不是任意伪造参数。

## 4. 带状态循环到控制流

加法每次迭代只写独立输出，还看不到循环携带状态。为了说明这部分 lowering，换成一个长度为 4 的整数求和函数：

<!-- cpu-example: sum -->
```text
module {
  func.func @sum(%m: memref<4xi32>) -> i32 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> i32 {
      %v = memref.load %m[%i] : memref<4xi32>
      %next = arith.addi %acc, %v : i32
      scf.yield %next : i32
    }
    return %r : i32
  }
}
```

`%acc` 是本轮读到的累加值，yield 是交给下一轮的状态。将 SCF 降低为 CF，实际得到：

<!-- cpu-output: sum-cf -->
```text
module {
  func.func @sum(%arg0: memref<4xi32>) -> i32 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0, %c0_i32 : index, i32)
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %3 = memref.load %arg0[%0] : memref<4xi32>
    %4 = arith.addi %1, %3 : i32
    %5 = arith.addi %0, %c1 : index
    cf.br ^bb1(%5, %4 : index, i32)
  ^bb3:  // pred: ^bb1
    return %1 : i32
  }
}
```

这里可以按三个位置理解：

1. 入口向 header `^bb1` 传入索引 0 和累加初值 0。
2. header 判断索引是否小于 4；成立时进入 body，否则到退出 Block。
3. body 读取当前元素，更新累加值和索引，再通过回边传给 header。

对 `[3,-2,5,7]`，header 收到的状态依次为 `(0,0)`、`(1,3)`、`(2,1)`、`(3,6)`、`(4,13)`。最后一次判断失败，返回 13。

这里没有一个被反复赋值的 SSA `%acc` 对象。每条边传递的值成为下一次进入 Block 时的参数；SCF 的区域协议被改写成了显式 CFG。

## 5. 计算次序与 Lowering 的边界

把循环表示从 SCF 换成 CF，不意味着获得了交换迭代顺序的许可。整数、浮点、内存效果和跨迭代依赖仍需遵循原来的语义。

特别是浮点归约，如果后来改成树形归约或并行归约，可能改变加法结合顺序，进而改变舍入结果。那需要独立讨论优化条件，不能把它当成“lowering 本来就允许”的附带变化。

同样，`convert-linalg-to-loops` 主要提供一种直接的计算展开。它不会自动证明当前循环顺序最有利于缓存，也不会保证完成 tiling、fusion、SIMD 或线程映射。认识这条直接路径，才能在后面的优化章节清楚地比较改变了什么。

## 6. 向 LLVM 交接

到 CF 阶段，计算与控制流已经显式化，但 MemRef 访问仍依赖形状和 stride，`index` 仍需确定低层类型。接下来需要统一降低相关操作与类型，再清理合法可消解的边界 cast。

```text
Linalg 区域与映射
    ↓
SCF 循环 + MemRef load/store + Arith
    ↓
CF 分支与 Block 参数
    ↓
LLVM 指针运算、load/store、低层函数与跳转
```

完整 pipeline 必须覆盖仍然存在的操作。部分转换成功、临时出现混合方言，并不自动说明已满足 translation 的最终条件。下一章继续解释其中最容易在外部调用时出错的[类型与 ABI 边界](./abi)。

## 阅读自查

1. 本例 Linalg body 为什么不需要读取 `%out` 的旧内容？
2. `parallel` iterator 为什么可以先被实现为顺序循环？
3. 求和的 CF header 参数分别对应 SCF 的哪两种状态？
4. 把 SCF 降成 CF，是否允许顺便改变浮点归约的结合顺序？

## 资料与实践

示例与输出按 LLVM 20.1.8 核验，数值包含连续/非连续矩阵与空迭代域。源码入口见[Linalg Loops](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Linalg/Transforms/Loops.cpp)、[SCFToControlFlow](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp)。

可选实践：[循环、CF 与本机调用](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/11-cpu-lowering)。
