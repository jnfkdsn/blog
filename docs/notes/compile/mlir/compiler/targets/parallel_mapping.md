---
order: 10
title: 并行工作到执行层次
updated: 2026-10-04
---

# 并行工作到执行层次

[GPU Kernel 章节](../../dialects/gpu/execution)手写了三组、每组四个线程的启动。编译器通常先拿到算法的迭代空间，再决定如何把它分配给设备。理解这个过程，需要把三项决定分开：哪些工作可并行、工作怎样分块、每一层块分给哪一层执行单元。

本章把长度 10 的 `y[i]=2×(x[i]+1)` 从两个抽象并行循环映射成上一章的 kernel。算法保持不变，改变的是迭代点由谁执行。

## 1. 并行性来自计算关系

在输入输出不重叠且长度相同的前提下，迭代点 i 只读 `x[i]`、写 `y[i]`。不同 i 不会修改彼此需要的数据。因此这些点可以重新分配，而不必按 0、1、2 的次序逐个执行。

若算法改成 `y[i]=y[i-1]+x[i]`，这个论证就不成立。把循环名字从 for 改成 parallel，并不能消除真实的数据依赖。分配之前仍然需要前面[依赖与合法性](../analysis/)的知识。

## 2. 用嵌套并行域表达分块

先分为三组，每组四个候选点，尾部由条件保护。下面省略外层常量声明：

```text
scf.parallel (%block) = (%c0) to (%c3) step (%c1) {
  scf.parallel (%thread) = (%c0) to (%c4) step (%c1) {
    %base = arith.muli %block, %c4 : index
    %i = arith.addi %base, %thread : index
    %valid = arith.cmpi ult, %i, %c10 : index
    scf.if %valid {
      %a = memref.load %x[%i] : memref<10xf32>
      %b = arith.addf %a, %one : f32
      %c = arith.mulf %b, %two : f32
      memref.store %c, %y[%i] : memref<10xf32>
    }
    scf.reduce
  }
  scf.reduce
}
```

这时 `%block`、`%thread` 只是为了说明意图取的名字；SCF 本身还没有把它们变成 GPU block/thread ID。`scf.reduce` 在这里终止没有返回归约结果的并行 Region，并不表示程序执行了求和。

这种表示已经决定工作分区。外层三次、内层四次，共十二个候选点；只有 i<10 的十点访问数组。下一步才把两层分给具体硬件层次。

## 3. 映射属性与实际启动

固定版本的 `gpu-map-parallel-loops` 可以给适合的嵌套并行循环附加映射。本例的外层一维循环映射到 block_x，内层映射到 thread_x；实际属性的核心为：

```text
#gpu.loop_dim_map<processor = block_x,
                  map = (d0) -> (d0), bound = (d0) -> (d0)>
#gpu.loop_dim_map<processor = thread_x,
                  map = (d0) -> (d0), bound = (d0) -> (d0)>
```

随后 `convert-parallel-loops-to-gpu` 根据映射建立 `gpu.launch`，把执行实例编号代入原来的 induction variable。一般非零下界、非单位步长还需要计算 `iv=lower+id×step`，组数/线程数来自相应迭代次数。

本例最后得到 grid=(3,1,1)、block=(4,1,1)，kernel 内仍计算 `4b+t` 并保留尾部条件。变换没有重新发明加法算法，只是把抽象迭代编号换成设备编号。

自动附加映射是可用的起点，不是完整的性能搜索器。它不知道仅凭这个小例子就能决定所有矩阵的最佳二维 block、共享存储方案或占用率。

## 4. `scf.forall` 的另一条入口

结构化分块也常使用 `scf.forall`。它支持并行区域和 Tensor 的共享输出组合，例如不同实例写回互不重叠的切片。本例已经使用 Buffer，所以可以把同一结构写成：

```text
scf.forall (%block) in (3) {
  scf.forall (%thread) in (4) {
    // i=4*block+thread；检查 i<10 后执行相同的读、算、写。
  } {mapping = [#gpu.thread<x>]}
} {mapping = [#gpu.block<x>]}
```

映射属性说明预期分工，仍需相应变换把 forall 降低。实际 Transform 调度先生成 launch 并映射外层到 blocks，再把内部 forall 映射到 threads：

```text
%launch = transform.gpu.map_forall_to_blocks %func generate_gpu_launch
  : (!transform.any_op) -> !transform.any_op
%mapped = transform.gpu.map_nested_forall_to_threads %launch
  block_dims = [4, 1, 1] sync_after_distribute = false
  : (!transform.any_op) -> !transform.any_op
```

固定 LLVM 20.1.8 的这两个入口有具体限制：只接受相应 bufferized 形式，支持的维数、静态 trip count 等也受约束。因此不能把“forall 能表达动态 Tensor 并行”直接等同于“这个 GPU 映射入口能接受所有 forall”。不满足条件需要先变换，或选择别的实现。

上面关闭了映射后的 barrier，是因为本例只有互不依赖的逐元素写入，之后没有同 block 的共享数据消费；不是默认建议删同步。若后面读取别的线程产生的 tile，就必须重新检查跨阶段依赖。

## 5. 分工不是归约算法

如果程序要把十个输入相加，不能让十个线程都普通 load/store 同一个累加位置。这样会产生读改写竞争。

需要先选择一种合并方案，例如每线程形成局部和，组内归约，再把不同组的部分和交给下一阶段。SCF 的 parallel reduction、GPU 的 all_reduce、subgroup reduce 或原子操作分别表达不同层次的工作；它们不自动共享同一执行范围。

浮点归约的合并次序也随方案变化。映射的合法性必须同时考虑内存竞争、参与范围和允许的数值语义。前面的逐元素例子只解决了工作分配的第一种简单情形。

## 6. 工作量与资源约束

每个 thread 一项只是可选分工。如果每个 thread 连续处理四项，可以减少线程数，但会改变地址分布、局部寄存器需求和尾部处理；也可以让一个 block 处理二维 tile，再在组内分工。

一般应依次确认：

1. 每个有效迭代点由谁完成，是否重复或遗漏。
2. 不同参与者之间需要交换哪些数据、在哪里同步。
3. 访问是否适合目标的连续事务和共享存储布局。
4. 线程数、共享存储及寄存器需求是否落在资源限制内。
5. 最终代码和设备测量是否支持预期收益。

单独缩短高层 IR、增加并行度或提高某个占用率指标，都不能替代这条判断链。

## 7. 阅读检查与衔接

1. `scf.parallel` 的变量叫 `%thread`，为什么仍不代表它已经是 GPU thread ID？
2. 外层步长从 1 改为 2 时，能否直接用 `block_id` 替代原 IV？
3. forall 映射成功以后删去 barrier，需要检查哪一类消费者？

下一篇讨论[设备布局与数据搬运](./layout_transfer)。当前作者已核验 parallel/forall 两条路径的实际 launch 与 NVVM 结果；并行数据映射可以逐点检查，但设备执行仍需可访问硬件。

实现依据：[SCFToGPU](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToGPU/SCFToGPU.cpp)、[GPU Transform 契约](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td)、[SCF 定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)。
