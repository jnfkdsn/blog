---
order: 3
title: Affine 循环与依赖分析边界
updated: 2026-10-04
---

# Affine 循环与依赖分析边界

知道循环遍历哪些点、每个点访问哪些元素以后，就可以继续问：一个迭代是否要等待另一个迭代的写入？如果没有这种跨迭代依赖，就有机会交换、分块或并行执行；如果存在，则必须保留相应先后关系。

本章用一段跨行递推把约束推理具体化。它也说明 Affine 的价值：把循环域和访问关系放在便于分析的形式里，而不是给普通循环换一个名字。

## 1. 一段跨行递推

下面的更新读取左上一行、右边一列的元素，再加 1 写到当前位置：

<!-- structured-example: wave -->
```text
module {
  func.func @wave(%a: memref<5x7xi32>) attributes {llvm.emit_c_interface} {
    %one = arith.constant 1 : i32
    affine.for %i = 1 to 5 {
      affine.for %j = 0 to 6 {
        %v = affine.load %a[%i - 1, %j + 1] : memref<5x7xi32>
        %r = arith.addi %v, %one : i32
        affine.store %r, %a[%i, %j] : memref<5x7xi32>
      }
    }
    return
  }
}
```

外层 i 从 1 到 4，内层 j 从 0 到 5。数组最后一列和第 0 行保持初值，所有访问都在 5×7 范围内。我们讨论原地更新，所以后面的迭代可能读取前面刚产生的结果。

对于 `(i,j)=(2,0)`，读取的是 `(1,1)`。由于原顺序先完成整行 1，再进入行 2，读取能看到行 1 的新值。这项先后关系是计算含义的一部分。

## 2. 从地址相等推导依赖

设产生某个写入的迭代为 `(is,js)`，消费该值的迭代为 `(it,jt)`。两次访问同一个元素需要：

```text
写地址 (is,js)
读地址 (it-1,jt+1)

is = it-1
js = jt+1
```

还要同时满足两个迭代都在各自的循环域内，以及写发生在读之前。由等式可得从生产者到消费者的距离：

```text
(it-is, jt-js) = (1,-1)
```

第一维增加 1，第二维减少 1。在先 i 后 j 的词典序中，第一项为正，生产者在消费者之前；若交换两层循环，先看 j，第一项就变成 -1，原本的消费者可能先执行。

这比“两个操作都访问 a”更精确。一般 AliasAnalysis 能提示同一来源，Affine 访问分析进一步结合索引等式、迭代域和先后条件，判断哪些迭代之间存在依赖。

## 3. 循环携带依赖与并行范围

这条依赖跨越 i，阻止当前外层循环简单并行执行。对固定的一行 i，不同 j 都读取已经完成的上一行，写当前行不同位置；在本例范围内，内层 j 可以并行。

实际调用固定版本的分析工具，得到：

<!-- structured-output: wave-analysis -->
```text
identity legal: 1
swap legal: 0
outer parallel: 0
inner parallel: 1
```

identity 是保持 `[i,j]` 顺序，swap 是尝试 `[j,i]` 顺序。`isLoopParallel` 和 `isValidLoopInterchangePermutation` 从 Affine 结构中计算所需判断；这些查询没有真的移动循环，也没有启动线程。

在合适的 C++ 上下文中，关键调用是：

```cpp
SmallVector<unsigned> permutation{1, 0};
bool legal = affine::isValidLoopInterchangePermutation(loops, permutation);
bool innerParallel = affine::isLoopParallel(loops[1]);
```

这里 loops 是按外到内收集的完美嵌套循环。实际工程还要确保调用符合 API 的支持范围和 IR 契约，不能把任意操作列表当成循环嵌套输入。

## 4. 约束求解与保守失败

一般的访问依赖检查会组合源域、目标域、地址相等关系和顺序条件。若能证明约束没有整数解，就排除这类依赖；若有解，则存在需要考虑的关系。若输入包含当前分析不支持的情形，就必须保守处理。

`checkMemrefAccessDependence` 的结果明确区分 `HasDependence`、`NoDependence` 与 `Failure`。Failure 表示检查未完成，不能按“没有发现问题”解释为可以优化。

例如访问通过内存加载得到的下标、复杂的未知副作用、分析不支持的步长或 semi-affine 形式，都可能超出某个具体消费者的能力。不同分析支持范围也不同：一种操作可被 lower，不意味着任意高层依赖检查都能处理它。

这项区分使通用框架可以稳健地扩展：已经证明的情形实施优化，信息不足的情形保持原程序，或转交另一条能提供证明的路径。

## 5. Affine 与 SCF 的选择

Affine 强调可表达、可分析的边界和访问关系。SCF 则可以接收更一般的 SSA 上界、条件和运算结果。例如 gather 的地址来自 `indices[i]`，可以正常写成 load 加 SCF 循环，但不能仅通过改名把它变成普通仿射映射。

MLIR 可以在一个函数里混合不同表示：外层规则矩形区域用 Affine，内部需要一般控制流的部分用其他方言。随后 `lower-affine` 把 affine.for/if/访问等转换为更通用的循环、算术和 MemRef 操作。

降低会保留已表达计算，但也可能使整体访问关系不再以一个 map 直接出现。若还计划进行某类 Affine 分析，通常应在它需要的信息结构仍存在时安排相关变换。具体 pipeline 按任务选择，而不是要求所有计算都尽早转成 Affine 或尽早离开 Affine。

## 6. 数值反例与分析结论

把数组初始内容设为 `a[i,j]=100*i+j`。原顺序中：

```text
a[1,1] = a[0,2]+1 = 3
a[2,0] = a[1,1]+1 = 4
```

若手工交换循环为先 j 后 i，处理 j=0、i=2 时，位置 `[1,1]` 尚未更新，仍为 101，于是写入 102。配套本机程序实际观察到原结果 4、错误交换结果 102。

这两份 IR 的结构与类型都合法，也没有越界；错误来自对依赖顺序的破坏。分析拒绝交换的意义因此能直接从数组内容看见，不只是一个 API 返回了 false。

## 阅读自查

1. 距离 `(1,-1)` 为什么允许当前次序，却阻止交换两层循环？
2. 外层不能简单并行，为什么内层仍可能并行？
3. 分析返回 Failure 与 NoDependence 应如何分别处理？

下一步可进入[分块与边界](../../compiler/optimization/tiling)，把域与访问关系用于构造合法调度。

## 实现依据与实践

依赖模型与接口依据 [AffineAnalysis.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h)、[AffineAnalysis.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Affine/Analysis/AffineAnalysis.cpp)及 [LoopUtils.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp)。[14 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)包含真实分析工具与保持有定义行为的错误交换对照。
