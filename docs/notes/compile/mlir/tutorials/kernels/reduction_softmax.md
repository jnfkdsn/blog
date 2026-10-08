---
order: 1
title: 归约与 Softmax 的结构化表示
updated: 2026-10-04
---

# 归约与 Softmax 的结构化表示

前面的矩阵加法中，每个输出只依赖同位置的两个输入。Softmax 多了一种联系：一个元素的最终结果依赖整行的最大值和指数之和。编译器必须同时表达逐元素计算、跨元素归约和行结果的广播。

本章把熟悉的稳定 Softmax 完整写成 Tensor、Linalg、Arith 和 Math 共同组成的 IR，再沿已有 CPU 编译路径核对数值。先建立能够解释和验证的算法基线，存储与融合随后单独展开。

## 1. 稳定 Softmax 的计算过程

对一行长度为 N 的输入，定义：

$$
m=\max_j x_j,\qquad e_j=\exp(x_j-m),\qquad
s=\sum_j e_j,\qquad y_j=e_j/s.
$$

在精确实数数学中，分子分母同时乘以 `exp(-m)` 不改变比例。在浮点计算中，这一步让指数的输入不大于零，避免直接计算巨大正指数。

以 `[1,2,3]` 为例，四步结果是：

| 阶段 | 数据 |
|---|---|
| 最大值 | 3 |
| 移位与指数 | `[exp(-2),exp(-1),1]` |
| 指数之和 | 约 1.5032147 |
| 归一化 | 约 `[0.09003057,0.24472847,0.66524096]` |

对 `[1001,1002,1003]`，移位后的输入相同，也应得到相同的理想结果。本章以有限 f32 输入、沿最后一维逐行归一化为基本契约；具体误差由 f32 运算和数学实现决定。

## 2. 二维计算与两个归约

现在输入为 `tensor<MxNxf32>`，每行独立：

```text
X[M,N]
  ├─ 沿 j 求最大值 → Max[M]
  └─ 与 Max 按行广播相减，再 exp → E[M,N]
                                        ├─ 沿 j 求和 → Sum[M]
                                        └─ 除以按行广播的 Sum → Y[M,N]
```

对 `Max[i]`，不同 j 的输入都汇到同一个结果位置。Linalg 的输入 map 为 `(i,j)->(i,j)`，输出 map 为 `(i,j)->(i)`；i 标为 parallel，j 标为 reduction。这个投影已经说明了归约发生在哪里。

计算 `E[i,j]` 时，输入 X 的 map 不变，而 Max 的 map 为 `(i,j)->(i)`。于是 j 从 0 变化到 N-1 时，同一行最大值被重复取用，形成广播。这里两个 iterator 都为 parallel，因为各个 E 元素分别产生。

Sum 使用与 Max 相同的投影，只把标量更新从 maximum 改成 add。最后 Y 的计算又恢复为两个 parallel iterator，并按行读取 Sum。

这条 IR 不需要把“Linalg 转成 Tensor”。Linalg 描述计算关系，Tensor 提供值、形状与结果载体，Math 描述指数，Arith 描述局部数值运算。它们可以共同存在于同一个编译阶段。

## 3. 归约初值与结果语义

最大值归约从负无穷开始：读取第一项 v 后得到 `max(v,-∞)=v`，再与后续元素比较。如果初值随意设成 0，当一行全部为负数时，所谓最大值就不再是输入最大值。

求和归约从 0 开始：每个指数值加到当前累积值上。它的 outs 对应已有累积状态，而不是一片可忽略内容的“空结果空间”。因此必须先 fill，再进行归约。

两次逐元素阶段则不同：exp 的输出只取决于 X 和 Max，normalize 只取决于 E 和 Sum，它们不读取 destination 的原内容。`tensor.empty` 可作为这些完整写入结果的初始载体，但不能直接用未初始化的 empty 作为求和初值。

这正是 [DPS](../../dialects/linalg/destination_style) 在算法里的实际用途：所有阶段都能有 outs，但 body 是否读取旧值，决定了必须保留什么初始内容。

## 4. 完整 Tensor IR

下面是可解析的完整模块。动态维度由 `tensor.dim` 取得，两个一维结果都长度 M，两个逐元素结果都形状 M×N。

<!-- softmax-example: softmax -->
```text
module {
  func.func @softmax(%x: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n = tensor.dim %x, %c1 : tensor<?x?xf32>
    %neg_inf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    %row_empty = tensor.empty(%m) : tensor<?xf32>
    %max_init = linalg.fill ins(%neg_inf : f32) outs(%row_empty : tensor<?xf32>) -> tensor<?xf32>
    %max = linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%x : tensor<?x?xf32>) outs(%max_init : tensor<?xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %r = arith.maximumf %v, %acc : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>
    %matrix_empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %exp = linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i)>, affine_map<(i,j) -> (i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %max : tensor<?x?xf32>, tensor<?xf32>) outs(%matrix_empty : tensor<?x?xf32>) {
    ^bb0(%v: f32, %row_max: f32, %unused: f32):
      %shift = arith.subf %v, %row_max : f32
      %e = math.exp %shift : f32
      linalg.yield %e : f32
    } -> tensor<?x?xf32>
    %sum_empty = tensor.empty(%m) : tensor<?xf32>
    %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<?xf32>) -> tensor<?xf32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%exp : tensor<?x?xf32>) outs(%sum_init : tensor<?xf32>) {
    ^bb0(%e: f32, %acc: f32):
      %r = arith.addf %acc, %e : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>
    %out_empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i)>, affine_map<(i,j) -> (i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%exp, %sum : tensor<?x?xf32>, tensor<?xf32>) outs(%out_empty : tensor<?x?xf32>) {
    ^bb0(%e: f32, %row_sum: f32, %unused: f32):
      %p = arith.divf %e, %row_sum : f32
      linalg.yield %p : f32
    } -> tensor<?x?xf32>
    return %out : tensor<?x?xf32>
  }
}
```

阅读这段 IR 时，可以沿 `%max → %exp → %sum → %out` 四个结果追踪，不必逐行记住构造语法。每个 generic 的 maps 负责把当前 `(i,j)` 变成 operand 坐标，body 负责本次元素或累积更新，yield 把结果交回相应输出位置。

其中 `%neg_inf` 使用 IEEE f32 的负无穷位模式；它是最大值归约的初值，输入契约仍为有限元素。`llvm.emit_c_interface` 只为后续生成 C 调用包装，不参与算法计算。

## 5. 从张量计算到 CPU 结果

沿已有机制可以走通：

```text
上述 Tensor/Linalg IR
    ↓ One-Shot Bufferization
输入视图、行统计 buffer、指数 buffer、输出 buffer
    ↓ 所有权与释放
保留返回结果，释放内部临时分配
    ↓ Linalg → loops；Math → libm
显式循环、访存与 expf 调用
    ↓ SCF/MemRef/Func 等 → LLVM → LLVM IR
CPU 对象文件与 C 调用
```

这里的数学路径明确选择 libm，使用单精度 `expf`。循环按所选顺序执行 f32 累加，没有额外启用浮点 reassociation。最终 C 包装返回一个结果描述符；输入为借用，返回分配由调用方释放。参数、描述符与返回空间的区别见 [ABI 章节](../../compiler/lowering/abi)。

对两行相同的 `[1,2,3]`，实际每行得到：

```text
[0.0900305733, 0.244728476, 0.665240943]
```

这与前面手工过程一致，但不会与无限精度十进制逐位相同。相对于以实际 f32 输入提升成 double 后计算的参考，该例最大绝对误差约 `1.23e-8`。

编译成功说明转换接通，数值对照说明选定输入上的计算正确。二者分别验证工程链和算法行为。

## 6. 形状与特殊值边界

当 N=1，每行只有一个有限数，减去自身得 0，指数为 1，结果应为 1。全等输入的结果应接近 `1/N`。很大的整体平移不应造成直接正指数溢出；十分悬殊的元素可能让较小权重下溢为 0，这需要与参考和误差目标一起判断。

M=0 表示没有行，输出仍是形状 `0×N` 的空张量。N=0 时，数学上不存在通常意义的归一化概率行。本实验另外定义了一个工程边界：返回相同空形状，不生成任何输出元素，也不要求行和为 1。最后的逐元素除法没有迭代，所以不会实际计算 `0/0`。这个约定不应冒充所有框架的空轴行为。

NaN、正无穷和全负无穷行不属于本章有限输入基线。例如全负无穷行会发生 `-∞-(-∞)`，产生 NaN；正无穷减自身也存在同类问题。若 Attention 使用负无穷 mask，必须另行定义“整行被屏蔽”时希望输出什么，并在实现中处理，不能认为减最大值自动覆盖全部特殊值。

## 7. 数值验证的判据

参考实现对实际存入的 f32 输入使用 double 求最大值、指数、累加和除法。这样比较的是编译实现相对于更高精度计算的误差，不会把输入量化误差误记到 lowering 上。

当前实验使用：

$$
|y-y_{ref}|\le 2\times10^{-6}+2\times10^{-5}|y_{ref}|.
$$

同时检查有效输出有限且非负，非空行的和与 1 的误差不超过 `5e-5`；检查输入没有被修改，输出窗口外的哨兵没有被写坏，并检查返回形状。空行扩展不执行行和判据。

测试覆盖 2×3、整体正/负大平移、均匀行、指数下溢、单列、宽度 17 和 513，以及两种空形状。宽度 513 的例子最大绝对误差约 `6.36e-9`，最大行和误差约 `8.66e-7`。这些是当前宿主、当前 pipeline 和所选数据的实测，不是对任意长度、硬件或输入的误差上界。

后续改变归约树、向量宽度或数学实现后，应重新核对同一套边界，并根据实际任务制定误差标准。只测试 `[1,2,3]` 无法检验整个实现。

## 阅读自查

1. 两个归约为什么都使用 `(i,j)->(i)`，而指数阶段仍是两个 parallel iterator？
2. 哪两个 outs 必须初始化，哪两个阶段不依赖 destination 旧值？
3. 全负无穷的屏蔽行为什么不满足本章稳定算法的输入契约？
4. IR verifier、编译链接、double 参考和哨兵检查分别验证什么？

下一篇[Softmax 的存储与融合](./softmax_storage)沿相同计算检查中间值必须保存多久，并构造按行完成计算的实现。

## 实现依据与实践

完整 IR、编译链和 C 驱动保存在 [13-softmax 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/13-softmax)。结构语义对应 [LinalgOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td)与 [MathOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Math/IR/MathOps.td)；数值实现沿实际选用的 MathToLibm 路径核验。本文建立 CPU 正确性基线，没有测量加速比。
