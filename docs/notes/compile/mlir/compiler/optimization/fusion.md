---
order: 2
title: Fusion 与计算重排
updated: 2026-10-04
---

# Fusion 与计算重排

两次逐元素计算常常可以接在一起：先得到 `p=x+1`，再计算 `y=p*2`。如果先为整个张量生成 p，再遍历 p 生成 y，就需要保存和读取一个中间张量；如果每个元素刚得到 p 就立即计算 y，中间结果可能只需作为局部 SSA 值存在。

这就是本章要完成的变换。之后再改变 consumer 的访问方式，说明融合为何有时带来重复计算，以及“能融合”与“值得融合”为什么不同。

## 1. Producer 与 Consumer

下面的完整 Tensor IR 保留两个阶段。第一个 generic 是 producer，第二个是 consumer；两者以 `%p` 连接：

<!-- structured-example: elementwise -->
```text
module {
  func.func @elementwise(%x: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %x, %c0 : tensor<?x?xi32>
    %n = tensor.dim %x, %c1 : tensor<?x?xi32>
    %init = tensor.empty(%m, %n) : tensor<?x?xi32>
    %p = linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<?x?xi32>) outs(%init : tensor<?x?xi32>) {
    ^bb0(%v: i32, %unused: i32):
      %one = arith.constant 1 : i32
      %r = arith.addi %v, %one : i32
      linalg.yield %r : i32
    } -> tensor<?x?xi32>
    %out_init = tensor.empty(%m, %n) : tensor<?x?xi32>
    %y = linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%p : tensor<?x?xi32>) outs(%out_init : tensor<?x?xi32>) {
    ^bb0(%v: i32, %unused: i32):
      %two = arith.constant 2 : i32
      %r = arith.muli %v, %two : i32
      linalg.yield %r : i32
    } -> tensor<?x?xi32>
    return %y : tensor<?x?xi32>
  }
}
```

每个 map 都是 identity。对任意 `(i,j)`，consumer 只读取 producer 的同一个位置，producer 也只读取 x 的同一个位置。因此可以把这两个标量 body 串起来，而不需要等待 producer 的其他坐标。

## 2. 融合后的数据流

固定版本运行 `linalg-fuse-elementwise-ops` 并清理后，只剩一个 generic，关键 body 为：

```text
^bb0(%x: i32, %unused: i32):
  %p = arith.addi %x, %one : i32
  %y = arith.muli %p, %two : i32
  linalg.yield %y : i32
```

现在一条完整的元素路径是 `X[i,j] → 加一 → 乘二 → Y[i,j]`。中间 p 仍是有意义的计算结果，但不再是必须保存整张 tensor 的跨操作结果。

以 `x=-3` 为例，局部先得 -2，再得 -4，和两阶段一致。这里不依赖重排整数算术，也没有把 `2*(x+1)` 改成另一条可能具有不同溢出属性的表达式；只是把同一次加一的结果直接交给乘法。

Bufferization 之后，原始形式要给中间 P 和输出 Y 安排存储，融合形式只需为最终输出安排相应分配。这个变化来自 IR 数据流被改写，不能仅靠后面多运行一次 deallocation 实现。

## 3. Consumer 需要哪一片 Producer

identity maps 使本例特别直接。一般融合要从 consumer 的需求倒推 producer 的计算范围。

假设 consumer 当前处理输出 tile `[4:6,3:6]`，并且读取 P 的同位置，它需要的 producer slice 就是同一个窗口。可以先在 tile 内计算这片 P，再立即消费，而不先生成整张 P。

若 consumer 读取 P 的转置位置，需求就要按其索引映射换成对应坐标；若 consumer 读取邻域，需求可能比输出 tile 大。融合实现必须维护这些映射，不是把两个循环体放进同一个括号就完成了工作。

对 Tensor 值语义，producer 的结果表达一个不会被其他写入任意改变的值，这为数据流重组提供便利。进入 buffer 形式后，还要检查别名、访存效果和覆盖顺序；未知副作用也可能阻止移动或复制 producer。

## 4. 邻域访问与重复计算

考虑一维变体：

```text
P[i] = f(X[i])
Y[j] = P[j] + P[j+1]
```

若 Y 的一个 tile 包含 j=0..3，需要 P[0..4]；下一块 j=4..7，需要 P[4..8]。两个需求范围都包含 P[4]。若每个 tile 独立在内部重新生成完整所需 P，P[4] 会计算两次。

这种 overlap 类似图像卷积中的 halo：融合减少了大中间数组的生命周期，却可能重复边界工作。若 f 只是一次加法，代价可能很小；若 f 是昂贵的指数或包含复杂归约，就需要认真比较重算成本。

多个 consumer 也有类似问题。把 producer 分别融合进两个 consumer，可能复制 producer 的计算；保留共享中间结果则会占用存储并增加读写。正确方案要根据 producer 的可复制性、访问量、算术量和局部资源共同选择。

## 5. 归约融合的时间边界

[Softmax 按行版本](../../tutorials/kernels/softmax_storage)把 exp 的产生和 sum 累加放在同一遍循环，正是每个元素产生后立即消费的融合。但 normalize 必须等待整个行和完成，因此没有直接和第一个指数一起输出最终结果。

归约结果与点式结果的“何时可用”不同。consumer 如果需要一个完整归约结果，不能只执行 producer 的一个局部迭代就开始消费。若计划使用在线算法或部分归约再合并，还必须重新推导状态更新公式与数值条件。

这也是为什么通用 elementwise fusion、基于 slice 的循环 fusion 和专门算法重组应分别理解。它们可能都减少中间存储，但证明的条件和实际变换并不相同。

## 6. 合法性、收益与实际验证

本例的正确性来自一一对应的索引、可组合的纯整数计算，以及保留同一元素内部的数据依赖。实验比较原始、融合、融合后分块和交换后的函数，包含奇数尺寸、整块、单元素与空域，所有输出按 `2*(x+1)` 核对。

结构检查确认 generic 从两个变成一个，数值检查确认选择的实现保留结果。性能还需要另外测量：小输入可能由调用或分配成本主导；大输入中减少的访存才可能更明显；后端还可能对原始形式做进一步优化。

融合也可能增加单个内核的 live values 和寄存器压力，降低设备上可同时驻留的工作数量。较少中间张量是一项可解释的结构收益，但不是无需测量的加速保证。

## 阅读自查

1. 为什么当前 consumer 不需要等待 producer 的其他坐标？
2. 邻域 consumer 的两个相邻 tile 为什么可能重复计算同一个 producer 点？
3. Softmax 的 exp/sum 可合并，为什么 normalize 仍需要等整行和？

下一篇[循环交换与归约次序](./interchange)继续讨论保持数据关系的同时怎样改变访问顺序。

## 实现依据与实践

当前 elementwise 行为依据 [ElementwiseOpFusion.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp)。Affine fusion 还包含 slice 和代价方面的选择，可查 [Affine Pass 定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Affine/Passes.td)。[14 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)执行的是这里明确列出的 elementwise 变换；邻域与多 consumer 部分为语义和成本推演，未宣称同一个 Pass 自动实现全部变体。
