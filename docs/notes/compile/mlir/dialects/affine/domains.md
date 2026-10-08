---
order: 2
title: Integer Set 与循环域
updated: 2026-10-04
---

# Integer Set 与循环域

索引映射告诉我们当前迭代访问哪里，但还没说明哪些迭代应该发生。矩形输出需要所有行列组合；下三角计算只需要 `j≤i`；分块后的最后一块只包含落在原形状内的元素。

这些情况可以统一理解为整数坐标上的约束。本章从一个 4×4 下三角填充开始，再把约束用于解释余块边界。

## 1. 矩形域与三角区域

完整 4×4 迭代域为：

$$
0\le i<4,\qquad 0\le j<4.
$$

在其中选择 `i-j≥0` 的点，就得到包含对角线的下三角。若满足条件写 1，否则写 0，结果应为：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

例如 `(2,1)` 满足 `2-1≥0`，`(1,2)` 不满足。这里的对象是一组有效点，而不是把一个坐标转换成另一个坐标。

## 2. Integer Set 与条件执行

Affine 的 Integer Set 用一组同时成立的整数约束描述这种集合。完整实现如下：

<!-- structured-example: triangle -->
```text
#lower = affine_set<(i,j) : (i-j >= 0)>
module {
  func.func @triangle(%out: memref<4x4xi32>) attributes {llvm.emit_c_interface} {
    %one = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        affine.if #lower(%i, %j) {
          affine.store %one, %out[%i, %j] : memref<4x4xi32>
        } else {
          affine.store %zero, %out[%i, %j] : memref<4x4xi32>
        }
      }
    }
    return
  }
}
```

`#lower` 只保存 `i-j≥0`。外面两个循环提供 `0≤i,j<4`，实际 then 区域的执行域是这几组约束的交集。Integer Set 不会自动从类型中补出所有循环边界，也不代表为它单独分配了一个布尔张量。

`affine.if` 把当前 `%i,%j` 绑定到集合的变量，按是否满足约束选择区域。集合中以逗号分隔的多个约束采用“同时成立”；任意析取关系不能直接当成同一串 conjunction，可能需要拆分区域或采用更一般的表示。

与 map 对照：map 产生索引结果，set 描述允许的点，affine.if 使用集合控制实际执行。

## 3. 半开区间与整数不等式

编程循环通常使用半开区间 `[0,N)`，约束系统常把它写成：

```text
i >= 0
N - 1 - i >= 0
```

因为 i 是整数，`i<N` 等价于 `i≤N-1`。这里的减一来自离散整数域，不是随意的边界修补。N=0 时这两条约束没有共同解，表示空迭代域。

带步长的循环还需要刻画哪些点实际被访问。例如从 0 开始步长 2 的循环，并不访问区间内所有整数点，还要保持偶数步进关系。支持何种步长与整除关系由具体分析能力决定，不能只丢掉 step 再进行依赖证明。

## 4. 分块后的合法点

把 5×7 的矩形按 2×3 分块。外层块起点分别是 `bi∈{0,2,4}`、`bj∈{0,3,6}`。若始终在块内执行 2×3 个点，最后一行或最后一列会越界。

一种等价的有效域写法为：

```text
bi <= i < min(bi+2,5)
bj <= j < min(bj+3,7)
```

当 `(bi,bj)=(4,6)`，只允许 `(i,j)=(4,6)`。这是一个 1×1 的余块，其余候选点不在原迭代域中。

Affine 的循环上界可以取一个 map 的多个结果的最小值。例如生成的两个 map 是：

```text
#rows = affine_map<(bi) -> (bi+2,5)>
#cols = affine_map<(bj) -> (bj+3,7)>
```

循环用 `to min #rows(%bi)` 时，表示同时满足两个上界。Tensor tiling 也可以先算 `min(2,5-bi)` 这样的实际切片长度，再用 extract_slice 处理有效块。它们表达的是同一组边界，IR 的承载方式不同。

## 5. 约束怎样支持优化

如果想证明每个有效元素恰好处理一次，可以把每个块对应的点集合列出，检查不同块不重叠，且它们的并集等于原域。对上面的 5×7 例子，行方向长度为 `[2,2,1]`，列方向为 `[3,3,1]`，所有块的元素数之和为 `(2+2+1)*(3+3+1)=35`。

元素数相同本身仍不够：重复一些点、漏掉另一些点，也可能总数为 35。还需要起点、步进与边界保证无重叠覆盖。并且若点之间存在依赖，即使集合完全相同，执行顺序变化仍可能改变结果。

所以循环域提供了优化证明的一部分：哪些迭代存在、边界在哪里、两个候选访问是否可能落在同一处。下一篇继续把域与访问关系组合起来，分析依赖和变换边界。

## 阅读自查

1. `#lower` 只有一条不等式，完整 then 区域为什么仍只有有限个点？
2. N=0 时，如何从整数约束看到循环域为空？
3. 9 个块合计 35 个元素，为什么还需要证明它们不重叠且没有遗漏？

## 实现依据与实践

集合与循环边界依据固定版本的 [Affine.md](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/Affine.md)及 [AffineOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Affine/IR/AffineOps.td)。[结构化优化实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)保留下三角输出与实际 2×3 tiling IR，正文中的集合推演与真实 lowering 对照。
