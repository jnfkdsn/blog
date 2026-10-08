---
order: 10
title: 向量值与元素组织
updated: 2026-10-04
---

# 向量值与元素组织

前面的 Linalg 表示告诉编译器“每个元素要算什么”；分块把一次工作的范围缩小到局部。现在还要决定：这个局部里的多个标量，怎样一起参与计算？

沿用逐元素加法。若一个小块包含两行、每行四个数，可以把八次标量加法表达成一次向量加法。这里的“一次”首先是 **IR 里的一次操作**。最终机器可能用两条、更多条指令完成；这正是从 Vector 到目标代码还需要降低的原因。

## 1. 从标量循环到向量值

先看一行的计算：

```text
输入：[0, 1, 2, 3]
加一：[1, 1, 1, 1]
输出：[1, 2, 3, 4]
```

标量形式需要依次取出 `x[i]`，加一，再写回。向量形式先取得四个数构成的值，在各个位置分别做加法，最后把结果写出：

```text
%one = arith.constant 1.0 : f32
%ones = vector.broadcast %one : f32 to vector<4xf32>
%result = arith.addf %input, %ones : vector<4xf32>
```

这里 `%input` 和 `%result` 都各是一个 SSA Value，每个 Value 含四个元素。`arith.addf` 没有变成另一套加法规则：它在对应位置上相加。`broadcast` 则把一个标量复制到所有位置，补齐加法需要的另一组输入。

这些位置通常称为 lane。lane 是向量中的逻辑位置；在这里不能把它直接理解为 GPU 线程、CPU 核或固定硬件执行单元。

## 2. 多维 Vector 保留局部组织

把例子扩展到两行：

```text
tile = [[ 0,  1,  2,  3],
        [10, 11, 12, 13]]     类型 vector<2x4xf32>

tile + 1 = [[ 1,  2,  3,  4],
            [11, 12, 13, 14]]
```

`vector<2x4xf32>` 不只是“八个 float”的另一种拼写。它保留了两维组织，后续可以按行取值、交换维度，或用两个维度参与收缩。比如：

```text
%row = vector.extract %tile[1] : vector<4xf32> from vector<2x4xf32>
%item = vector.extract %tile[1, 2] : f32 from vector<2x4xf32>
```

两项结果分别是 `[10, 11, 12, 13]` 和 `12`。这里是在已有值里选元素，没有读取某个 MemRef 地址；即使最终机器需要寄存器移动或暂存，也不改变这层语义。

把 `[2,4]` 改看成 `[8]` 可以用 `vector.shape_cast`：元素总数保持不变，线性次序也保持不变。它不会得到矩阵转置的次序。转置用 `vector.transpose`，明确交换元素的逻辑坐标。

## 3. 值的转置与内存写入

下面是一段完整计算：读取一个 `2×4` 小块，加一，把结果转置成 `4×2`，再存入另一个缓冲区。

```text
func.func @values(%x: memref<2x4xf32>, %y: memref<4x2xf32>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %tile = vector.transfer_read %x[%c0, %c0], %zero
    {in_bounds = [true, true]} : memref<2x4xf32>, vector<2x4xf32>
  %ones = vector.broadcast %one : f32 to vector<2x4xf32>
  %sum = arith.addf %tile, %ones : vector<2x4xf32>
  %t = vector.transpose %sum, [1, 0]
    : vector<2x4xf32> to vector<4x2xf32>
  vector.transfer_write %t, %y[%c0, %c0]
    {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32>
  return
}
```

`transfer_read` 连接内存与向量值；`transfer_write` 连接向量结果与内存。这里形状和起点保证整块有效，`in_bounds` 明确记下这个事实。关于余块的读写保证，下一篇会单独推演。

上述输入实际得到：

```text
y = [[1, 11],
     [2, 12],
     [3, 13],
     [4, 14]]
```

转置改变了 `%t` 的元素排列，写入才改变 `%y` 的内容。它没有把 `%x` 的 MemRef 描述符改成一个转置视图，也没有在原地重排 `%x`。因此需要区分三件事：向量值如何排列、缓冲区的地址如何计算、最终使用什么搬运指令。三者有关联，但不由一个 `transpose` 全部决定。

## 4. Tensor、MemRef 与 Vector 的分工

现在把这一小块放回前面的矩阵计算：

| 表示 | 在本例里提供什么 | 尚未决定什么 |
|---|---|---|
| Tensor | 完整矩阵的值语义、形状与数据流 | 哪块实际存储承载结果 |
| MemRef | 已有缓冲区的形状、步长、偏移与地址空间 | 其中一段怎样组成局部向量运算 |
| Vector | 一组元素共同参与计算时的局部形状和操作 | 如何匹配目标寄存器、指令、线程分工 |

这不是必须串行经过的三站。Linalg 可以在 Tensor 上先向量化，也可以在 Bufferization 后向量化；Vector 的 transfer 也有 Tensor 形式。选择时机取决于流水线想保留哪些信息。

本例先使用 MemRef，是为了直接看见“从哪里读、计算了什么、往哪里写”，不是因为 Vector 要求所有输入都先变成缓冲区。

## 5. 从虚拟向量到目标表示

如果目标支持四个 `f32` 的向量运算，`2×4` 的加法可以分成两组。但“形状刚好合适”仍不等于编译器必然给出最理想的指令；转置、边界、地址连续性都会影响降低。

固定版本的 LLVM 类型转换给了一个清楚的观察点：

```text
vector<2x4xf32>
    ↓ LLVMTypeConverter
!llvm.array<2 x vector<4xf32>>
```

两行成为 LLVM aggregate 的两个成员，每个成员是一维向量。这只是降低中的一种表示：aggregate 并不自动意味着堆分配；后续可以拆成寄存器值，也可能因资源压力发生 spill。

因此 `vector<1024xf32>` 在语法上成立，也不意味着存在一个装得下 1024 个数的硬件寄存器。较大的虚拟向量需要进一步拆分。相反，scalable vector 的某些维度由运行时 `vscale` 决定，类型中的数字只给出倍数；它适合特定目标，不能把固定宽度实例的拆分方法直接套过去。

学到这里，阅读 Vector IR 时首先应问：这一组元素对应原计算的哪部分，哪些操作逐元素执行，哪些操作重排或合并元素？寄存器宽度与线程分配是后续要做的选择。

## 6. 阅读检查与衔接

1. 上例执行完后，`%x` 的内容会改变吗？哪条操作真正写内存？
2. 把 `vector<2x4xf32>` shape cast 为 `vector<8xf32>`，为何不会得到 `[0,10,1,11,...]`？
3. 一个八元素加法用了两条机器指令，是否违反“一条 Vector 加法”的语义？

接着看 [Transfer、Mask 与边界](./transfer_mask)。自动产生这些向量操作的过程在[向量化与目标降低](../../compiler/optimization/vectorization)中展开。

本文按 LLVM 20.1.8 核验完整读取、转置和写入，实际检查八个位置及类型降低。精确契约见 [VectorOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td) 与 [Vector 设计说明](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/Vector.md)；可选复现与练习在 [15 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/15-vector-transform)。
