---
order: 2
title: 广播、归约与矩阵乘法
updated: 2026-10-03
---

# 广播、归约与矩阵乘法

上一章中，每个迭代点 `(i,j)` 都访问三个张量的 `[i,j]`。改变这些访问关系，就能得到更常用的计算：让一个输入忽略 i，是沿行复用的 bias；让输出忽略 j，则多个迭代点共同形成一个归约结果。

本章沿这两个变化，从逐元素加法走到矩阵乘法。每一步先看公式和数值，再放回 Linalg 的 maps、iterator 和 body。

## 1. 广播与重复读取

现在 B 不再是 2×3 的矩阵，而是三个列偏置：

```text
A = [[1,2,3],    bias = [10,20,30]
     [4,5,6]]

C[i,j] = A[i,j] + bias[j]
C = [[11,22,33],
     [14,25,36]]
```

计算仍遍历 `(i,j)`，bias 的坐标只需要 j。两个不同的点 `(0,2)`、`(1,2)` 都读取 bias[2]，但写入不同输出位置。

<!-- tensor-example: broadcast -->
```text
module {
  func.func @bias(%a: tensor<2x3xf32>, %bias: tensor<3xf32>) -> tensor<2x3xf32> {
    %empty = tensor.empty() : tensor<2x3xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %bias : tensor<2x3xf32>, tensor<3xf32>)
      outs(%empty : tensor<2x3xf32>) {
    ^bb0(%x: f32, %b: f32, %unused: f32):
      %v = arith.addf %x, %b : f32
      linalg.yield %v : f32
    } -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
```

与上一章相比，主要变化集中在 bias 的类型和第二条 map：从二维 `[i,j]` 变成一维 `[j]`。IR 不需要先复制 bias 成 2×3 的张量，访问关系已经表达了重复使用。

这里的广播不是“不同 shape 都会自动套用 NumPy 规则”。操作明确提供映射；若输入有一个大小为 1 的维度，需要根据实际形状和操作约束设计相应访问关系。写 generic 时，广播语义由这些规则共同给出。

## 2. 归约与重复更新同一个结果

再改变目标：计算每一行的和。

```text
S[i] = sum_j A[i,j]
S = [6,15]
```

现在不同 j 对应同一个输出 S[i]。访问关系变成：

```text
输入： (i,j) → (i,j)
输出： (i,j) → (i)
```

输出映射丢掉 j，但计算仍需遍历 j。因此，j 是归约维度：多个迭代点为同一个结果累积贡献。仅仅删掉输出映射的一维还不够，需要 body 描述如何将新输入与累积值结合，并提供初始值。

<!-- tensor-example: row-sum -->
```text
module {
  func.func @row_sum(%a: tensor<2x3xf32>) -> tensor<2xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2xf32>) -> tensor<2xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a : tensor<2x3xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %next = arith.addf %acc, %x : f32
      linalg.yield %next : f32
    } -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
}
```

`linalg.fill` 先得到全零的初始张量。Region 的 `%x` 是 A[i,j]，`%acc` 是当前输出元素的累积内容，yield `%acc+%x` 更新这一结果。

按 j 从小到大理解第 0 行的一种顺序展开：

| j | 读到的 x | 进入本次计算的 acc | 提交的新值 |
|---|---:|---:|---:|
| 0 | 1 | 0 | 1 |
| 1 | 2 | 1 | 3 |
| 2 | 3 | 3 | 6 |

这里 `%acc` 是每个迭代点上的标量输入，不是手写 SCF 中的整个循环携带张量。Linalg 将输出元素与归约域的关系保留在操作内部，循环 lowering 再落实读取和更新。

## 3. 初始值与数值语义

将初始值从 0 改为 10，按相同计算规则得到 `[16,25]`。因此，初始化是算法的一部分，不能因为“只想求和”就把 `%init` 随意换成 `tensor.empty`。

如果归约维度长度为零，没有任何输入贡献，结果应保留相应初始值。这也说明初始化不仅仅是第一次迭代的实现技巧。

浮点加法还涉及求值顺序。上表使用小整数，所有值在 f32 中都能精确表示；一般浮点输入不能据此推出任意重排都得到相同结果。`reduction` 标记表达归约结构，单看它无法得出具体变换的数值保证。选择并行树归约、向量化或 FMA 时，还需核对具体操作、变换规则、fast-math 等约定及允许的误差。本章用顺序循环解释数据贡献，不将这组小整数结果推广为任意归约调度都逐位一致。

## 4. 矩阵乘法的三个迭代维度

矩阵乘法把上述归约再推广一步：

```text
C[i,j] = sum_k A[i,k] * B[k,j]
```

输入 A 为 2×3、B 为 3×2，输出 C 为 2×2。遍历变量是 `(i,j,k)`：i、j 定位一个输出，k 枚举这个输出的贡献。

| 对象 | 访问映射 | 维度含义 |
|---|---|---|
| A | `(i,j,k) → (i,k)` | 输出行与归约位置 |
| B | `(i,j,k) → (k,j)` | 归约位置与输出列 |
| C | `(i,j,k) → (i,j)` | 输出位置 |

对应 iterator 为 parallel、parallel、reduction。两个输入都是 rank 2，操作却有三个迭代维度；这再次说明循环空间与单个张量 shape 是相关但不同的概念。

取：

```text
A = [[1,2,3],    B = [[1,2],
     [4,5,6]]        [3,4],
                     [5,6]]
```

C[0,1] 的三个贡献为 `1*2`、`2*4`、`3*6`，从 0 累加得到 28。完整结果为 `[[22,28],[49,64]]`。

## 5. Named Op 与 Generic 的共同结构

常见矩阵乘法已有 named op：

<!-- tensor-example: matmul -->
```text
module {
  func.func @matmul(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>)
                       outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %r : tensor<2x2xf32>
  }
}
```

这里 `linalg.matmul` 自带标准矩阵乘法的结构。用 generic 表达同一核心计算时，采用上一节的三条 maps、三种 iterator，并在 body 中写：

```text
^bb0(%a: f32, %b: f32, %acc: f32):
  %product = arith.mulf %a, %b : f32
  %next = arith.addf %acc, %product : f32
  linalg.yield %next : f32
```

named 形式让语义更集中，也方便识别常见运算；generic 形式把访问和标量计算显式写出。二者都属于结构化计算。可以用 generalization 将本例 named 形式展开，再对照上述关系，而不应把 named op 当作一个完全没有内部计算的函数调用。

还要注意其累加语义：matmul 向 destination 的初始内容加入乘积和。这里先 fill 0，才对应普通 `A×B`；若初始内容是已有 C，则表达 `C+A×B`。

## 6. 从访问关系定位错误

对这组例子，可以按固定顺序推导：先找所有迭代维度，再逐个应用 operand 的 map，最后看每次 yield 进入哪个输出位置。

例如把 bias 的 map 误写成 `(i,j)->(i)`，表达的会是按行选择偏置，而非按列；即便某些 shape 恰好允许它通过验证，算法也可能变了。又如求行和时忽略 `%acc`，只 yield `%x`，就没有实现求和。

这类问题说明 verifier 与数值测试承担不同工作：验证结构不能代替确认公式与映射一致。最有效的小测试应让各行、各列数值不同，避免全 1 输入掩盖坐标写错。

## 理解检查与衔接

若要求列和，输入 map、输出 map 和 iterator 分别怎样改变？把 bias 数据长度与 A 的列数设为不同值，为什么同一个投影不能继续表达原算法？matmul 的初始矩阵全部为 1 时，结果与这里相差多少？

下一章集中解释[Destination-Passing Style 与结果构造](./destination_style)，将 empty、fill、outs、累积值和结果 Value 放进同一个模型。

## 实现与查阅

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)包含广播、行归约与 matmul 的实际输入，并对小矩阵核对结果。固定版本的 named/generic 定义见 [LinalgStructuredOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td) 与 [LinalgStructuredOps.yaml](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml)。本文没有进行硬件并行归约的性能或误差实验。
