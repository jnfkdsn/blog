---
order: 3
title: Destination-Passing Style 与结果构造
updated: 2026-10-04
---

# Destination-Passing Style 与结果构造

前两章里，Linalg 一方面通过 `outs` 接收一个张量，另一方面又产生一个新的结果 `%r`。这很容易让人疑惑：输出既然已经传进去了，为什么还需要返回值？它究竟是在修改旧张量，还是在创建新张量？

理解这件事，可以先把两个问题分开：**如何描述结果的形状和初始内容，以及运行时用哪片存储实现结果。** Destination-Passing Style，简称 DPS，先在 IR 中建立前一种关系，并为后续存储选择提供线索。

## 1. 逐元素计算中的 Destination

回到矩阵加法，核心结构如下：

```text
%empty = tensor.empty() : tensor<2x3xf32>
%r = linalg.generic ...
    ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
    outs(%empty : tensor<2x3xf32>) {
  ^bb0(%x: f32, %y: f32, %unused: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
} -> tensor<2x3xf32>
```

这是[上一章完整示例](./iteration_maps)的结构摘录。`%empty` 提供一个与结果对应的 destination；`%r` 是执行整个张量计算后得到的新 SSA Value。

对每个 `[i,j]`，body 计算 `%x+%y`。它不使用 destination 原来的元素，因此 `%empty` 的未指定内容不会参与结果计算。这个例子只需要正确形状，不需要先填零。

如果先 fill 0 再做同一项逐元素加法，数值仍可相同，但那份零内容本来没有被 body 使用。编译器能否清理相关工作属于优化问题；算法层面无需为这个结果提供累加初值。

## 2. 归约中的初始内容

行和则不同：body 使用 `%acc`，它来自当前输出元素的累积内容。

```text
%zero = arith.constant 0.0 : f32
%empty = tensor.empty() : tensor<2xf32>
%init = linalg.fill ins(%zero : f32)
    outs(%empty : tensor<2xf32>) -> tensor<2xf32>
```

fill 把每个位置定义为 0，得到 `%init`。归约以它为 destination，于是每一行从 0 开始累加。

两种计算的区别可以归纳为：

| 计算 | 是否读取 destination 的旧元素 | 本例所需准备 |
|---|---|---|
| 逐元素 A+B | 否，body 忽略对应参数 | 只提供形状，empty 足够 |
| 行和 | 是，`next=acc+x` | fill 0，得到确定的初始值 |
| `C+A×B` | 是，累加到 C | 使用给定 C 的内容 |

“destination”这个角色本身没有承诺所有旧元素都会被读取。实际怎样使用，要继续看操作语义与 body。相反，读取未指定内容的计算不能仅靠验证通过就被认为实现了正确算法。

## 3. Destination 与结果的对应关系

在本系列使用的 tensor 形式 DPS 操作中，一个 tensor destination 与一个结果建立对应关系，类型相同，动态维度的实际大小也必须一致。这种对应称为 tied relationship：它告诉通用机制，某个结果是围绕哪个 destination 构造的。

```text
输入 A、B ─────→ 元素计算 ─────→ 新结果 R
                       ↑
         destination：形状，以及需要时的初始内容
```

MLIR 的 `DestinationStyleOpInterface` 将这类关系作为可查询协议提供给变换。消费者可以询问哪些 operands 是 inputs、哪些是 inits，以及某个 init 对应哪个结果；随后再结合具体操作的读写行为做决定。

这正好连接前面 IR 定义模块学习的 Interface：接口并不是凭空规定每个算法都要填零，而是把跨操作共通的结构关系暴露出来。是否读取、是否完整覆盖以及能否复用，还需要其他语义信息和分析。

DPS 也不限于 Linalg。前面 `tensor.insert` 和 `tensor.insert_slice` 都围绕已有 destination 产生新结果：更新的部分来自新输入，未更新部分继承 destination。

## 4. 新结果与旧值可以同时存在

为了看清 tensor 语义，直接把 A 作为加法的 destination，并同时返回旧 A 和新结果：

<!-- tensor-example: destination-old-use -->
```text
module {
  func.func @add_preserve(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>)
      -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
      outs(%a : tensor<2x3xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<2x3xf32>
    return %a, %r : tensor<2x3xf32>, tensor<2x3xf32>
  }
}
```

对于 A=`[[1,2,3],[4,5,6]]`，B=`[[10,20,30],[40,50,60]]`，两个返回值分别是原 A 和 `[[11,22,33],[44,55,66]]`。

`outs(%a)` 没有把 SSA Value `%a` 的含义改成 A+B。IR 明确要求旧值仍可被观察。如果某种实现直接覆盖存放 A 的唯一一份数据，且没有采取其他保留措施，后面的第一个返回值就会错误。

因此，tied relationship 不能直接解释成“这两个 Value 一定共享一块可写内存”。它首先是 IR 的结果构造关系。

## 5. 从结果构造走向存储复用

现在再考虑底层实现。有两种情况：

```text
情况一：之后只需要 R，不再观察旧 A
    → 在满足其他约束时，可以考虑让 R 复用 A 的存储

情况二：之后还需要旧 A 和 R
    → 必须保证旧值仍可观察；简单覆盖同一份内容不成立
```

这里只是在推导合法性条件，不预告某个版本分析器必然选择哪种优化。输入是否可写、是否与其他值共享存储、哪些路径还会读取旧内容等，也会影响决定。

DPS 给出一个直接的复用候选：结果已经关联到 destination，不需要先从任意操作中猜测候选。但最后是否原地执行，要由 Bufferization 将这些候选与读取、写入和别名关系一起分析。

对于切片更新尤其直观：将一个 patch 插回原张量时，destination 本来就描述了未更新区域。若原值不再需要，复用整个底层 buffer 可能省去复制；若旧值还要使用，就必须维护它的内容。

## 6. Tensor 形式与 Buffer 形式

[编译流程导读](../../tutorials/pipelines/overview)中的逐元素加法在 bufferization 后仍是 `linalg.generic`，但 `outs` 接收 MemRef，操作不再产生 tensor 结果。

这时 destination 已是存储对象。Linalg 在执行时向该存储写入结果，其他 MemRef 视图若与它别名，就可能观察到写入。于是“结果如何产生”的值语义，落实成了“哪些地址被写”的存储语义。

两个层次应这样连接：

```text
tensor 层：构造新值，旧值的可观察含义不变
    ↓ 选择能够保持该语义的存储方案
buffer 层：显式读写、别名和生命周期
```

直接把 tensor IR 中的 outs 当成可变数组参数，会跳过中间的合法性问题；完全忽略 outs，又会错过编译器选择存储的重要信息。

## 理解检查与下一步

为什么矩阵加法可以直接使用 empty，行归约却需要初始化？如果将归约的 fill 常量改为 10，算法如何变化？为什么“旧 A 不再使用”仍不是证明任意原地写入合法的全部条件？

下一模块进入 [MemRef](../memref/) 和 [Bufferization](../../compiler/memory/)：先认识存储、布局、视图和别名，再追踪上述复用候选如何变成具体决策。两处各三篇主线已提供，可以沿这条顺序继续阅读。

## 实现与查阅

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)保留 old-use 输入，并同时检查旧值与新值，避免仅验证 R 正确却遗漏 A 被破坏。

固定版本的共通契约见 [DestinationStyleOpInterface.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/DestinationStyleOpInterface.td)；关于复用候选与冲突的后续机制见 [Bufferization](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Bufferization.md)。这里建立需要维护的语义，不提前要求理解完整分析实现。
