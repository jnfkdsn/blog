---
order: 1
title: 迭代空间与访问映射
updated: 2026-10-03
---

# 迭代空间与访问映射

Tensor 操作已经能表达值、形状和切片，但仅凭这些信息，还不知道一次矩阵加法要访问哪些坐标、在每个位置做什么。Linalg 将这些关系放在同一个结构化操作中。

从最熟悉的计算开始：`C[i,j] = A[i,j] + B[i,j]`。本章会把它拆成计算域、访问规则和标量计算，再在 `linalg.generic` 中重新对应起来。

## 1. 从算法到迭代点

对 2×3 的输入，等价的计算过程是：

```text
for i in [0, 2):
  for j in [0, 3):
    C[i,j] = A[i,j] + B[i,j]
```

这里有三类不同信息：

1. 要处理的迭代点是 `(i,j)`，范围为 `0≤i<2, 0≤j<3`。
2. 在每个点，A、B 和 C 都使用坐标 `[i,j]`。
3. 取出两个标量后，对它们执行浮点加法。

第一类决定“哪些点参与”，第二类决定“每个点访问哪里”，第三类决定“读出值之后如何计算”。Linalg 保留三者之间的联系，后续可以据此重新组织计算。

## 2. 一条完整的 Generic 操作

<!-- tensor-example: add -->
```text
module {
  func.func @add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %empty = tensor.empty() : tensor<2x3xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
      outs(%empty : tensor<2x3xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
```

先忽略属性的具体语法，沿整体数据流读：A 和 B 作为 `ins`，empty 作为 `outs`，Region 计算一个输出元素，最终 `%r` 是整个 2×3 张量结果。

这里 `%empty` 只提供尚未指定内容的目标张量。Region 没有使用 `%unused`，而是为每个输出点给出 `%x+%y`。因此结果不依赖 empty 的元素初始值。结果与 destination 的详细关系在本模块第三章展开。

## 3. 从迭代坐标到张量坐标

### 3.1 Indexing Map 的输入与输出

本例的一个访问映射为：

```text
affine_map<(i, j) -> (i, j)>
```

左侧是当前操作的迭代坐标，右侧是某个 operand 的元素坐标。它不把一个张量转换成另一个张量，而是给出“位于这个迭代点时，应该访问该张量哪里”的规则。

`indexing_maps` 按 inputs 然后 outputs 的顺序排列，所以三个相同映射分别属于 A、B、empty 对应的结果位置。

例如在 `(1,2)`：

| 对象 | 应用映射 | 当前元素 |
|---|---|---|
| A | `(1,2) → (1,2)` | A[1,2] |
| B | `(1,2) → (1,2)` | B[1,2] |
| 输出 | `(1,2) → (1,2)` | 结果 C[1,2] |

后面广播会让某个映射只返回 `(j)`；归约会让输出映射只返回 `(i)`。理解映射的方向，才能解释“迭代点很多，但某个 operand 被重复访问”的情况。

### 3.2 迭代范围从哪里来

这里没有显式填写 `i<2` 和 `j<3`，因为各 operand 的 shape 与映射共同提供了这些范围。对于当前的恒等映射，两个维度分别对应大小 2 和 3。

更一般的 Linalg 需要从所有访问关系中得到一致、可支持的迭代域。不能随意写任意 affine map，并假定框架总能推出合法循环。本章与下一章采用恒等映射和投影，先让范围的来源可直接检查。

迭代维数也不等于每个张量的 rank。广播的 bias 可以是 rank 1，整个操作仍有两个迭代维度；矩阵乘法的两个输入都是 rank 2，计算却有三个迭代维度。

## 4. Region 参数与一次标量计算

对于当前操作，Region 参数按 operand 顺序对应每个点上的元素：

```text
%x       ← A[i,j]
%y       ← B[i,j]
%unused  ← destination 在 [i,j] 的元素
```

它们是 f32 标量，不是完整张量，也不是坐标 i 和 j。这与前面 `tensor.generate` 的 Region 参数有明显区别：generate 接收坐标，Linalg 的这个 body 接收由 maps 选出的元素。

设 A[1,2]=6、B[1,2]=60，Region 执行：

```text
%x=6, %y=60
%sum = 6 + 60 = 66
linalg.yield 66
```

yield 的第一个值对应第一个输出位置，即 C[1,2]。它不是立即从外层函数返回；一次 Linalg 操作将所有迭代点的结果组织为最终张量，再交给函数的 return。

若需要在 body 中使用迭代坐标，可使用相应的 `linalg.index` 操作。不能把 `%x` 当成循环索引，或仅凭 Block 参数位置猜测其含义。

## 5. Parallel 迭代与执行方式

两个 iterator 都标为 `parallel`。在这个逐元素例子中，不同迭代点计算不同输出位置，body 又只做当前两个输入值的加法，没有跨点累积依赖。

这种计算关系允许后续选择并行组织，但它没有创建线程、指定 GPU block，也没有保证生成向量指令。普通 CPU lowering 完全可以生成两层顺序循环，仍符合此处语义。

Linalg 的标记是操作作者对计算结构的描述，不是编译器自动替任意 body 证明所有优化合法。若 body 引入不受这些坐标关系描述的效果，就不能沿用纯逐元素计算的推理随意移动或并行化它。

## 6. 动态 Shape 与维度一致性

把行数改成动态时，仍然必须保证 A 和 B 的实际行数相等。本例用一个显式检查将前提写清楚：

<!-- tensor-example: add-dynamic -->
```text
module {
  func.func @add_dynamic(%a: tensor<?x3xf32>, %b: tensor<?x3xf32>) -> tensor<?x3xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %a, %c0 : tensor<?x3xf32>
    %n = tensor.dim %b, %c0 : tensor<?x3xf32>
    %same = arith.cmpi eq, %m, %n : index
    cf.assert %same, "row counts must match"
    %empty = tensor.empty(%m) : tensor<?x3xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<?x3xf32>, tensor<?x3xf32>)
      outs(%empty : tensor<?x3xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<?x3xf32>
    return %r : tensor<?x3xf32>
  }
}
```

两次 dim 分别取得实际行数，比较后由 `cf.assert` 检查。随后 `tensor.empty(%m)` 提供相同的结果行数，第二维 3 仍由类型指定。

这里的 assert 是示例主动加入的程序行为，不是 `linalg.generic` 自动生成的检查。实际编译器也可能利用更早的 shape 证明、框架约定或外部输入检查维持同一前提。

`tensor<?x3xf32>` 的问号没有把两个输入的第一维绑在一起。静态类型相同和运行时 shape 相同，是两个需要区分的判断。

## 7. 结构化信息与循环展开

对本例执行 bufferization 和循环 lowering 后，核心循环体会读取 A[i,j]、B[i,j]，相加后写入输出[i,j]。[编译流程导读](../../tutorials/pipelines/overview)给出了实际完整输出。

在 Linalg 中，三条关系集中保存在 maps、iterators 和 body 里；展开成循环后，关系分散到了循环边界、load/store 下标和标量操作中。较早保留结构，有利于针对计算域做 tile、fusion 等变换。是否值得保留到哪一步，要由具体 pipeline 和目标决定。

## 理解检查与衔接

如果把 B 的映射改为 `(i,j)->(j)`，B 的类型需要怎样改变，每一行会读到哪些值？如果 Region 直接 yield `%unused`，结果还会是 A+B 吗？为什么标记 parallel 不保证实验工具生成多线程程序？

下一章继续改变访问关系，进入[广播、归约与矩阵乘法](./structured_computations)。

## 实现与查阅

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)提供静态、动态加法与实际循环输出。固定版本的 GenericOp 定义见 [LinalgStructuredOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td)，循环展开可从 [Loops.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Linalg/Transforms/Loops.cpp) 中输入加载、body 处理与输出存储的关系进入。
