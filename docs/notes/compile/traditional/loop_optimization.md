---
order: 6
title: 循环优化
updated: 2026-07-01
tags: [compiler, loop, licm, vectorization]
status: draft
---

# 循环优化

相关入口：[传统编译器](/notes/compile/traditional/) / [Dataflow Analysis 与 Pass Pipeline](/notes/compile/traditional/dataflow_pass)

循环优化是传统编译器和 AI Compiler 都绕不开的主题。传统编译器里它决定 CPU cache、vectorization、分支和寄存器压力；AI Compiler 里它对应 tensor loop、tiling、fusion、memory hierarchy、kernel schedule。

## 循环识别

循环通常从 CFG 中识别。一个边 `B -> H` 如果满足 `H dominates B`，就是回边。`H` 是 loop header，`B` 是 loop latch。

```text
entry
  |
  v
header <----- latch
  |             ^
  v             |
body ----------+
  |
  v
exit
```

循环结构：

- preheader：进入循环前的单一 block，LICM 常把指令移动到这里。
- header：循环入口，通常包含条件判断或 phi。
- body：循环主体。
- latch：跳回 header 的 block。
- exit：离开循环后的 block。

标准化循环结构可以简化后续优化：

```text
loop-simplify:
  create preheader
  make single backedge latch
  make dedicated exits
```

## LICM

LICM 是 Loop Invariant Code Motion，把循环中每次迭代结果相同的计算移到循环外。

源程序：

```c
for (int i = 0; i < n; ++i) {
  y[i] = a * b + x[i];
}
```

`a * b` 与 `i` 无关，可以移动到循环外：

```c
tmp = a * b;
for (int i = 0; i < n; ++i) {
  y[i] = tmp + x[i];
}
```

Legality 条件：

- 指令 operands 在循环外定义，或本身也是 loop invariant。
- 指令没有副作用。
- 移动后不会改变异常/陷阱行为，或语言/IR 允许这种移动。
- 对 load 来说，需要证明循环内没有可能改写同一地址的 store。

IR 中：

```text
loop:
  %0 = mul %a, %b
  %1 = load x[i]
  %2 = add %0, %1
```

`mul` 容易 hoist；`load` 需要 alias analysis。

## Loop Unroll

Loop unroll 减少循环控制开销，增加指令级并行，也可能增加代码体积。

原始：

```c
for (int i = 0; i < n; ++i) {
  y[i] = x[i] + 1;
}
```

unroll by 4：

```c
for (int i = 0; i + 3 < n; i += 4) {
  y[i]     = x[i]     + 1;
  y[i + 1] = x[i + 1] + 1;
  y[i + 2] = x[i + 2] + 1;
  y[i + 3] = x[i + 3] + 1;
}
for (; i < n; ++i) {
  y[i] = x[i] + 1;
}
```

收益：

- 减少 branch 和 induction variable 更新次数。
- 暴露更多独立指令，利于调度。
- 给 vectorization 创造连续操作。

代价：

- 代码膨胀。
- 寄存器压力上升。
- 尾部处理更复杂。

## Loop Tiling

Tiling 把大循环拆成小块，提高 cache/local memory 复用。

矩阵乘：

```c
for i in 0..M:
  for j in 0..N:
    for k in 0..K:
      C[i][j] += A[i][k] * B[k][j]
```

Tiling 后：

```c
for ii in 0..M step BM:
  for jj in 0..N step BN:
    for kk in 0..K step BK:
      for i in ii..ii+BM:
        for j in jj..jj+BN:
          for k in kk..kk+BK:
            C[i][j] += A[i][k] * B[k][j]
```

优化点：

- `A/B/C` tile 能放进 cache、shared memory、UB、L1/L0。
- 减少慢速内存重复访问。
- 让内层循环更适合 vectorization 或 tensor core/cube。

AI kernel 优化里的 block tile、thread tile、warp tile、本质上也是 loop tiling 和数据复用设计。

## Loop Interchange

Loop interchange 改变循环顺序，改善访存连续性。

```c
for j in 0..N:
  for i in 0..M:
    sum += A[i][j];
```

如果 `A` 是 row-major，`A[i][j]` 随 `i` 变化会跨行访问。交换后：

```c
for i in 0..M:
  for j in 0..N:
    sum += A[i][j];
```

内层 `j` 连续，cache locality 更好。

合法性取决于依赖关系。不能随便交换存在 loop-carried dependence 的循环。

## Dependence Analysis

循环变换要保证依赖不被破坏。

```c
for (int i = 1; i < n; ++i) {
  a[i] = a[i - 1] + 1;
}
```

这里 `a[i]` 依赖上一轮的 `a[i-1]`，不能简单并行化。

依赖类型：

- true dependence / RAW：先写后读。
- anti dependence / WAR：先读后写。
- output dependence / WAW：两次写同一位置。

对于 tensor compiler，依赖分析对应：

- 哪些 axis 可以 parallel。
- 哪些 axis 可以 vectorize。
- reduce axis 和 spatial axis 是否能交换。
- fusion 是否会改变读写顺序。

## Vectorization

Vectorization 把标量循环改成向量指令。

标量：

```c
for (int i = 0; i < n; ++i) {
  y[i] = x[i] + 1;
}
```

向量化：

```text
for i in 0..n step 8:
  vx = vector_load x[i:i+8]
  vy = vector_add vx, 1
  vector_store y[i:i+8], vy
```

条件：

- 内存访问连续或可 gather/scatter。
- 循环迭代之间无阻碍向量化的依赖。
- 数据对齐或有处理非对齐的路径。
- 尾部元素有 mask 或 remainder loop。

AI Compiler 中，Triton 的 block-level program、Ascend C 的 Vector API、LLVM vectorizer 都和这个主题有关。
