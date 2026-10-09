---
order: 30
title: Matmul 的分块、累加与复用
updated: 2026-10-04
---

# Matmul 的分块、累加与复用

逐元素计算里，一份输入通常只服务一个输出。矩阵乘法不同：A 的一个元素会参与 C 的一行中多个结果，B 的一个元素会参与 C 的一列中多个结果。分块优化要利用的正是这种重复需求。

本章从 `C=A×B` 的完整归约出发，把它变成一组小矩阵乘法，再沿真实 MLIR 变换检查尾块和累加。最后讨论这些小块怎样进入局部存储与硬件计算单元。这样可以区分“已经改变迭代组织”和“已经实现高效数据复用”。

## 1. 输出元素与归约范围

设 A 为 M×K，B 为 K×N，C 为 M×N：

```text
C[i,j] = sum(k=0..K-1, A[i,k]*B[k,j])
```

i、j 决定输出位置；k 遍历同一输出的所有贡献。不同输出在输入与输出不重叠时可以独立计算，同一输出的贡献则必须完整合并。

取 A=`[[1,2,3],[4,5,6]]`，B=`[[7,8],[9,10],[11,12]]`，得到：

```text
C[0,0] = 1*7 + 2*9 + 3*11 = 58
C = [[58,64],[139,154]]
```

MLIR 的 buffer 形式直接表达这项计算：

```text
%zero = arith.constant 0.0 : f32
linalg.fill ins(%zero : f32) outs(%c : memref<?x?xf32>)
linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
  outs(%c : memref<?x?xf32>)
```

`matmul` 向 outs 的已有内容累加，所以 `C=A×B` 需要先清零。若希望计算 `C=C_initial+A×B`，则保留初值。空 K 时，前者留下零，后者留下原内容；这不是可以忽略的特殊情况。

## 2. 输出块与归约块

选择输出块 BM×BN，每次处理 BK 个归约元素。对一个固定输出块，过程是：

```text
把局部累加器 Ctile 初始化一次
    ↓
取 A 的 BM×BK 块和 B 的 BK×BN 块
    ↓
Ctile += Atile × Btile
    ↓
推进 k，直到覆盖完整 K
    ↓
保存 Ctile
```

三个分块方向承担不同职责：沿 M/N 切分不同输出，沿 K 切分同一输出的贡献。不能在每个 K 块开始时重新清零 Ctile。

例如将上例 K=3 分为前两项与最后一项，C[0,0] 的部分和是 25 与 33，合起来才是 58。若第二块再次初始化，最终只剩 33。每个小矩阵乘法都算对了，整体结果仍然会错，原因是跨块状态丢失。

## 3. 从分块公式到实际 IR

取 BM=2、BN=3、BK=4。给 `linalg.matmul` 应用以下调度，三个大小分别对应其 m、n、k 迭代维：

```text
%tiled, %loops:3 = transform.structured.tile_using_for %mm
  tile_sizes [2, 3, 4]
  : (!transform.any_op) ->
    (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
```

固定版本生成三个外层 `scf.for`，每轮计算当前块的实际大小：

```text
mi = min(2, M-i0)
nj = min(3, N-j0)
kk = min(4, K-k0)

Atile = subview A[i0,k0][mi,kk]
Btile = subview B[k0,j0][kk,nj]
Ctile = subview C[i0,j0][mi,nj]
matmul Atile, Btile outs Ctile
```

这是对生成 IR 的索引整理。实际大小由 `affine.min` 表达，切片由 `memref.subview` 表达。原来的 fill 留在所有分块循环外；每个 k0 对同一个 C 子视图继续累加。

M=5、N=7、K=11 时，最后一组块起点为 (4,6,8)，实际大小为 (1,1,3)。输入块分别为 1×3 与 3×1，输出块为 1×1。三个维度都有尾部，不能只保护输出坐标而忘记 K 的读入范围。

这里的 subview 共享原存储，变换没有自动分配快存储，也没有自动插入 global→shared 的搬运。它首先建立“哪一轮处理哪些数据”的组织，为后续选择复用位置提供边界。

## 4. 数据复用的来源

考虑一个没有尾部的输出块，以及一个 K 块。它完成约 `2*BM*BN*BK` 次浮点运算。A 块有 BM×BK 个元素，每个可服务 BN 个输出列；B 块有 BK×BN 个元素，每个可服务 BM 个输出行。

若这些输入能在某一级快存储保留，理想情况下从更慢一级只需取入 `BM*BK+BK*BN` 个元素。对于 f32，忽略 C 和其他开销时，计算量与这部分搬入字节数之比为：

```text
2*BM*BN*BK / (4*(BM*BK + BK*BN))
= BM*BN / (2*(BM+BN))
```

这个关系解释了增大输出块为什么可能提高复用。但它是一个明确限定的流量模型：真实访问还包含 C、边界、布局转换、缓存命中与重复装载。不能把公式直接当作设备 DRAM 实测。

实际实现需要决定复用发生在哪里。CPU 可以利用缓存、寄存器及局部 packing；GPU 可以分层使用 workgroup 存储与线程寄存器；NPU 则要遵循片上 buffer、搬运和计算单元的格式要求。相同 BM/BN/BK 在不同目标上的成本可能完全不同。

## 5. Tile、寄存器与矩阵指令

让一个执行单元同时保留更多 C 元素，会增加累加器需求。输入块若进入显式局部存储，也占用相应容量。简化估计可以先分别写出：

```text
单份输入块容量：element_bytes * (BM*BK + BK*BN)
局部累加器元素：BM*BN
双缓冲输入容量：约为单份的两倍，另计布局 padding
```

这些量解释了为什么 tile 不能无限增大。更大的复用机会可能被更少的驻留工作、寄存器溢出或额外搬运抵消。累加器最终是否全部落在寄存器，也由目标降低与资源约束决定。

Vector 的 `contract` 或目标矩阵乘指令为小块计算提供进一步表示。要命中特定硬件，还需满足元素类型、精度模式、tile 形状、元素分布与输入布局要求。高层操作名叫 matmul，并不意味着低层必然出现 Tensor Core 或 Cube 指令。

## 6. 循环展开与计算顺序

分块以后，可以把相邻两个 K 块的循环体放进一次循环迭代：

```text
transform.loop.unroll %loops#2 {factor = 2 : i64} : !transform.any_op
```

本例原来每次推进 4 个 k，主循环展开后可以每次推进 8 个 k，并依次执行两个块。不能整除的部分由余数路径处理。K=11 时仍然需要覆盖 [0,4)、[4,8)、[8,11)，不能丢掉最后三项。

这里展开的是 K 块循环，复制了两份带子视图的小 matmul；并没有因此合并成一次硬件矩阵指令。展开可能减少循环开销、暴露更多优化机会，也会增大代码和同时存活的状态。

顺序展开可以保留原有贡献顺序。若进一步采用多个独立累加器、树形归约或不同精度的矩阵指令，浮点舍入路径会变化，需要单独确认数值契约。一次数值测试通过不能代替对所有输入的证明。

## 7. 完整结果与边界检查

配套实验真实运行原始、2×3×4 分块、K 块二倍展开三种版本，经 LLVM lowering 生成 CPU 对象，再从 C 调用。8 组形状覆盖三维余块、整块、单元素、K=0、M/N=0 和多段归约，共 24 组结果对照 double 参考，并检查输出边界哨兵。

这证明这些输入下的变换与执行结果一致，且观察到了实际子视图和余数路径。当前没有给出 Matmul 性能结论，也没有把此标量 CPU 路径当作优化 GEMM 库。

完整工程与有限修改任务在 [17 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/17-matmul-attention)。定义与变换依据见固定版本的 [Linalg 命名操作](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml)和 [SCF Transform 操作](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td)。

## 8. 阅读检查与衔接

1. 沿 N 切两块和沿 K 切两块，为什么对累加器的要求不同？
2. tile_using_for 生成了 subview，能否据此认为输入已经搬到 shared memory？
3. BM/BN 增大带来哪些复用机会，又增加哪些资源需求？
4. K=11、BK=4、展开因子为 2，哪些区间必须恰好处理一次？

接下来把矩阵乘法与 Softmax 连接为 [Attention 的计算与存储](./attention)，进一步分析为什么中间结果会成为性能和容量问题。
