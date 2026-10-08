---
order: 40
title: 向量化与目标降低
updated: 2026-10-04
---

# 向量化与目标降低

[Vector 三篇](../../dialects/vector/)已经解释了向量值、边界访问与收缩。现在把它们接回编译器：给定 `y[i]=2×(x[i]+1)` 的 Linalg 表示，怎样让编译器产生向量计算，又怎样得到可执行代码？

本章采用一条完整路径：先把长度不定的一维工作分成四元素小块，再对每块向量化，最后处理 mask、内存和控制流，进入 LLVM。这样每层 IR 都能对应到同一组输入输出。

## 1. 输入保留了哪些信息

输入的 Linalg generic 如下。`%x`、`%y` 长度相同、连续且不重叠；输出原来的值不参与计算。

<!-- vector-example: elementwise -->
```text
#id = affine_map<(i) -> (i)>
module {
  func.func @elementwise(%x: memref<?xf32>, %y: memref<?xf32>) attributes {llvm.emit_c_interface} {
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<?xf32>) outs(%y : memref<?xf32>) {
    ^bb0(%a: f32, %old: f32):
      %b = arith.addf %a, %one : f32
      %c = arith.mulf %b, %two : f32
      linalg.yield %c : f32
    }
    return
  }
}
```

两个 identity map 表示同一迭代点读 `x[i]`、写 `y[i]`。`parallel` 表示不同点没有通过 Linalg 输出累加产生顺序依赖；结合这里明确的缓冲区不重叠前提，可以把不同点的计算放在一起。

编译器还没有承诺使用多宽的向量。若动态长度是百万，把整个函数直接看成一个百万元素 Vector 既不现实，也没有处理目标资源限制。先分块才能让“工作大小”变成受控选择。

## 2. 先确定每次处理的范围

取 tile size 4，分块后的结构可归纳为：

```text
for i = 0 .. n step 4:
    size = min(4, n - i)
    x_tile = x[i : i + size]
    y_tile = y[i : i + size]
    在这一小块上执行原来的 generic
```

对应 IR 使用 `scf.for`、`affine.min` 和两个 `memref.subview`。`n=10` 时得到大小 4、4、2 的三个视图；没有为最后两项伪造一块实际长度为 4 的缓冲区。

接下来选择宽度 4 的向量。完整块有四条有效 lane，尾块有两条。这里 **tile size 与 vector size 是两项相关但不同的选择**：前者决定工作分区，后者决定局部计算的向量形状。更大的 tile 还可能拆成多个向量；本例特意让两者相同，以清楚展示第一次转换。

## 3. 自动产生局部向量计算

以下是实际生成并经过 canonicalize 的核心结构，统一了 SSA 名字，省略常量声明：

```text
%size = affine.min affine_map<(i)[n] -> (-i+n, 4)>(%i)[%n]
%xs = memref.subview %x[%i] [%size] [1]
  : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
%ys = memref.subview %y[%i] [%size] [1]
  : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
%mask = vector.create_mask %size : vector<4xi1>
%v = vector.mask %mask {
  vector.transfer_read %xs[%c0], %zero {in_bounds = [true]}
    : memref<?xf32, strided<[1], offset: ?>>, vector<4xf32>
} : vector<4xi1> -> vector<4xf32>
%a = arith.addf %v, %ones : vector<4xf32>
%b = arith.mulf %a, %twos : vector<4xf32>
vector.mask %mask {
  vector.transfer_write %b, %ys[%c0] {in_bounds = [true]}
    : vector<4xf32>, memref<?xf32, strided<[1], offset: ?>>
} : vector<4xi1>
```

每个组成部分都有来源：映射变成 transfer 的访问关系，scalar add/mul 变成对应 lane 的 vector add/mul，动态小块长度变成 mask，输出位置变成 transfer_write。`%old` 在算法中未使用，清理后不必读取旧输出。

尾块的 `%size=2`，mask 为 `[true,true,false,false]`。`in_bounds=true` 与 mask 共同保证实际参与访问的 lane 有效；不能去掉 mask 再声称整个四元素访问都在长度 2 的视图内。

## 4. 向量化的适用条件

这个例子顺利，是因为访问关系简单、每个元素的计算支持向量类型，且没有跨迭代依赖。更复杂情况需要先回答额外问题：

| 改变的条件 | 需要重新确定的事情 |
|---|---|
| 输入按行走、输出按列走 | 向量 lane 与两侧内存地址如何对应，是否需要转置或非连续访问 |
| 存在 reduction iterator | 哪些维度进入归约，是否允许改变浮点合并次序 |
| 循环体含不支持向量形式的操作 | 是否有合法拆解，还是需要保留标量部分 |
| 自定义操作未提供所需结构接口 | 向量化器能否取得迭代域、访问及计算语义 |
| 形状是动态的 | 选择的向量尺寸是否覆盖被向量化的小块，如何屏蔽尾部 |

固定版本 `transform.structured.vectorize` 的显式 `vector_sizes` 要求足以覆盖目标小块的迭代尺寸。宽度 4 用于已经分为最多四项的块；它不是自动把任意长度拆成四项的命令。动态情况下某些不正确承诺未必能由静态 verifier 证明，所以要先建立分块的尺寸保证。

这也解释了“向量化器成功”与“所有操作都被向量化”的区别。有些对子操作应用 Patterns 的入口允许没有任何匹配。观察结果时应检查目标 generic 是否消失、出现了什么向量计算，不能只看 Pass 的退出码。

## 5. 把调度选择写成可复现的输入

上述过程由这一小段 Transform 程序驱动：

<!-- vector-example: vectorize -->
```text
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %target tile_sizes [4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled vector_sizes [4] : !transform.any_op
    transform.yield
  }
}
```

此处只需把它理解为“找到 generic → 分块 → 对新 generic 向量化”。它没有把 4 写进原算法，也没有要求重新编译一个 C++ Pass。下一章会解释这些句柄怎样指向真正的 IR，以及修改后为什么必须使用新句柄。

## 6. 从 Vector 到 LLVM

自动产生 Vector 后，还需要继续降低。实验采用的主要阶段为：

```text
lower-vector-mask
    将 vector.mask 的有效性传给具体可屏蔽操作
convert-vector-to-scf
    降低需要循环组织的 transfer 维度
convert-vector-to-llvm
    把支持的向量操作变成 LLVM 操作或 intrinsic
expand-strided-metadata / lower-affine / convert-scf-to-cf
    展开视图的地址信息与结构化控制流
convert-to-llvm / reconcile-unrealized-casts
    完成剩余函数、内存和算术转换
```

这些步骤要按仍存在的操作组合，并不保证任何 Vector 程序都只需这一份固定列表。上一章的 contract、transpose 就先需要相应拆解 Patterns。

本例降低后可以翻译成 LLVM IR，生成目标文件，再由 C 调用。五种实现对长度 0、1、2、3、4、5、7、8、10、17、1025 的结果一致，检查还覆盖输出保护区与输入未被改写。这样的证据支持本例的正确性，不支持“任何自动向量化都会更快”。

## 7. 阅读检查与衔接

1. 若直接对任意动态长度指定 `vector_sizes [4]`，缺少了哪项保证？
2. 分块后 `%size=2`，为何还能计算 `vector<4xf32>`？
3. 最终 IR 有 `vector.mask` 且工具退出成功，是否已经完成了 LLVM translation？

接着读 [Transform Dialect 与显式调度](./transform)，再看[实测性能](./performance)。实现按 LLVM 20.1.8 核验，精确约束见 [Linalg Transform 定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td)、[Linalg 向量化实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Linalg/Transforms/Vectorization.cpp)。完整输入输出和修改任务在 [15 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/15-vector-transform)。
