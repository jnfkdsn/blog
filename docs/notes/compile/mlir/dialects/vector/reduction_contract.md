---
order: 30
title: 归约、收缩与目标形态
updated: 2026-10-04
---

# 归约、收缩与目标形态

逐元素计算里，每条 lane 产生自己的结果。Softmax 求和、矩阵乘法累加则要求把若干元素合成一个结果。Vector 既需要表达“一组元素一起算”，也需要表达“哪些维度最后消失”。

本章先把四个数求和，再将这个过程放入一个 `2×3` 乘 `3×2` 的矩阵小块。它们分别展示 reduction 和 contraction；降低会继续决定如何拆解，而数值约束限制了可以怎样拆。

## 1. 从四个数到一个结果

输入 `%v=[1,2,3,4]`，初始累加值为零：

```text
%zero = arith.constant 0.0 : f32
%sum = vector.reduction <add>, %v, %zero : vector<4xf32> into f32
```

结果为标量 `10`。前面的向量加法是四对数各自相加；这条归约让四个位置共同贡献给一个标量。`<add>` 指定合并方式，`%zero` 指定开始累加时已有的值。

这对分块归约很有用：先得到一个块的部分和，再与其他块的部分和合并。但对浮点数，分块本身就可能改变合并次序。因此不能只凭实数代数中的结合律判断重排合法。

## 2. 浮点次序的可观察差异

考虑 `f32` 输入：

```text
[1e20, 1, -1e20, 3]
```

从零开始顺序相加：`1e20+1` 舍入回约 `1e20`，再加 `-1e20` 得零，最后结果为 `3`。若先合并 `1e20` 与 `-1e20`，再合并 `1` 与 `3`，结果则为 `4`。

本章实际 pipeline 使用默认 `reassociate-fp-reductions=false`，没有设置 fastmath，降低得到：

```text
%r = "llvm.intr.vector.reduce.fadd"(%zero, %v)
  <{fastmathFlags = #llvm.fastmath<none>}>
  : (f32, vector<4xf32>) -> f32
```

对应的 LLVM intrinsic 在没有 `reassoc` 时按规定顺序执行加法，这组输入本机得到 `3`。允许 reassociation 后，后端可以采用别的归约树；允许不等于保证一定选择某棵树或一定得到 `4`。

这里核验的是具体 lowering 路径。为模型选择更宽松的精度契约，需要结合容差、误差积累和极端输入决定。不能把 `reassoc` 当作不影响结果的纯性能开关，也不能从四个数推断完整 Softmax 的误差。

对于多维值，`vector.multi_reduction` 可以直接指定哪些维度消失，例如把 `2×4` 的第二维求和得到长度 2 的行和。输出形状保留未被归约的维度；之后可能进一步转成一维 reduction 或其他形式。

## 3. 矩阵小块的收缩

现在把归约放到矩阵乘法内部。输入与初始结果为：

```text
A = [[1,2,3],        B = [[ 7, 8],       C = [[1,1],
     [4,5,6]]             [ 9,10],            [1,1]]
                          [11,12]]
```

每个结果位置都计算 `C[i,j] + Σk A[i,k]B[k,j]`。输出需要保留 `i,j`，而 `k` 被累加掉。Vector 把这组关系放进 `contract`：

```text
%r = vector.contract {
  indexing_maps = [affine_map<(i,j,k) -> (i,k)>,
                   affine_map<(i,j,k) -> (k,j)>,
                   affine_map<(i,j,k) -> (i,j)>],
  iterator_types = ["parallel", "parallel", "reduction"]}
  %av, %bv, %cv : vector<2x3xf32>, vector<3x2xf32> into vector<2x2xf32>
```

`%av`、`%bv`、`%cv` 是已经读入的向量值。映射解释同一迭代点 `(i,j,k)` 分别取哪两个输入元素，以及更新哪个累加位置。它和 Linalg 的索引映射相似，但此刻操作对象已经是局部向量，而不是整个矩阵的 Tensor 或 MemRef。

以 `(i,j)=(0,0)` 为例：

```text
1 + 1×7 + 2×9 + 3×11 = 59
```

完整结果为 `[[59,65],[140,155]]`。第三个操作数不是空白输出位置；它的已有值参与累加。若把全 1 的 `%cv` 换成全 0，四个结果都会少 1。

## 4. 从收缩到具体计算组织

`contract` 还没有规定必须用一种机器指令。沿着 `k` 展开，一种自然组织是每次取 A 的一列与 B 的一行，形成外积并累加：

```text
k=0：[[1],[4]] × [[7,8]]   加到 C
k=1：[[2],[5]] × [[9,10]]  加到上一步结果
k=2：[[3],[6]] × [[11,12]] 加到上一步结果
```

本章选用固定版本的 `lower_contraction` 默认 OuterProduct 策略。实际中间 IR 进入更细的向量乘加；继续 lowering 后出现 `llvm.intr.fmuladd`。这些输入都是小整数，结果恰好可精确表示，因此适合检查索引与累加是否正确；它们没有检验一般浮点输入下融合乘加与分离乘加的全部差异。

不同目标可能更适合 dot product，或特定尺寸、精度、布局的矩阵指令。选择某种目标形式，需要同时满足输入类型、tile 形状、数据分布和累加语义。仅仅出现 `vector.contract`，不会自动承诺使用 GPU Tensor Core 或 NPU Cube 指令。

## 5. 降低的各层职责

本章的计算可以分成以下过程：

```text
Linalg / 算法小块
    ↓ 形成局部向量计算
Vector contract / reduction / transfer
    ↓ 选择拆分与访问方式
较细的 Vector 运算 + 循环 / mask
    ↓ 类型和操作转换
LLVM 向量、aggregate、intrinsic
    ↓ 指令选择与寄存器分配
机器代码
```

将 transfer 的外层向量维度改写成循环，不等于把所有 Vector 都变成标量。将一个 `contract` 拆成乘加，也不等于已经做完目标指令选择。每一步解决的约束不同，完成整个过程可能需要多个 Pass 和 Pattern 组合。

这也是前面学习 Pattern、Conversion 和类型转换的实际用途：拆解规则负责把一种表示变成另一种，Conversion 检查目标是否已经支持剩下的操作；性能仍由整个计算组织与目标共同决定。

## 6. 阅读检查与衔接

1. `vector.reduction` 与两个向量的逐元素 `addf`，输出形状为什么不同？
2. 矩阵例子的 `%cv` 若换成零，哪个公式发生变化？
3. 看到 `llvm.intr.fmuladd`，能否直接断言生成了一条硬件 FMA，或证明任意输入与原标量程序逐位一致？

接着用[自动向量化](../../compiler/optimization/vectorization)把 Linalg 主例走到这些形式，再用[性能分析](../../compiler/optimization/performance)检查成本。

版本为 LLVM 20.1.8。核验包括四项归约、次序敏感输入、收缩四个结果及真实 lowering。依据为 [VectorOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td)、[VectorToLLVM](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp)、[LLVM vector reduction 契约](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/docs/LangRef.rst)。完整观察见 [15 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/15-vector-transform)。
