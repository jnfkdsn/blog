---
order: 1
title: NPU IR 表示
updated: 2026-07-05
tags: [ai-compiler, npu, graph-ir, use-def]
status: draft
---

# NPU IR 表示

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

NPU IR 融合首先依赖 IR 表示。融合 pass 必须知道每个 op 的输入输出、tensor value 的 producer/consumer、op 是否有副作用、metadata 是否完整，以及 graph 改写后如何保持一致。

## 基本对象

一个 graph IR 通常包含：

```text
Graph
  nodes: list[OpNode]
  inputs: list[TensorValue]
  outputs: list[TensorValue]

OpNode
  op_type: Add / MatMul / Relu / Reshape / Cast / ...
  inputs: list[TensorValue]
  outputs: list[TensorValue]
  attrs: map
  side_effect: bool
  metadata: shape / dtype / layout / device / format

TensorValue
  producer: OpNode | GraphInput
  users: list[Use]
  metadata: shape / dtype / layout / alias / memory
```

融合关心的不是语法，而是 graph 上的依赖：

```text
producer output
  -> consumer input
```

例如：

```text
x -> Add -> y -> Relu -> z
```

其中 `Add` 是 `y` 的 producer，`Relu` 是 `y` 的 user。把 `Add + Relu` 融合，本质上是替换 `y` 的 producer-consumer 子图。

## Use-Def 和 Def-Use

融合 pass 最常用的两个方向：

```text
def-use: 从 producer value 找到所有 users
use-def: 从 consumer input 找到 producer
```

用途：

- 找 producer-consumer pattern。
- 判断中间 value 是否 single-use。
- 重定向 fused op 的输出 use。
- 删除融合后不再使用的 old op。
- 保证 graph 拓扑顺序和依赖关系一致。

单 consumer 融合：

```text
A -> op1 -> T -> op2 -> B
```

如果 `T` 只有 `op2` 一个 user，融合比较直接：

```text
A -> fused_op1_op2 -> B
```

多 consumer 情况：

```text
A -> op1 -> T -> op2 -> B
             \-> op3 -> C
```

如果把 `op1` 融进 `op2`，`op3` 仍然需要 `T`。这时有几种选择：

- 不融合。
- 只融合 `op2` 后半段，让 `op1` 保留。
- 复制 `op1` 的计算，分别服务 `op2/op3`。
- 做 multi-output fusion，让 fused region 同时产生 `B/C` 或保留 `T`。

多 consumer 是 fusion profitability 的重要分叉点，因为复制计算可能比省掉内存更贵。

## Op 分类

融合判断通常先按 op 类型分类：

| 类型 | 例子 | 融合特点 |
|---|---|---|
| Elementwise | Add、Mul、Relu、Sigmoid、Cast | 最容易融合，shape 一致或可 broadcast |
| Broadcast | Add with scalar、Expand | 需要检查 broadcast axis 和 layout |
| Reduction | ReduceSum、ReduceMax、Softmax 中的 sum/max | 融合复杂，涉及 reduce axis 和并行策略 |
| Matmul/Conv | MatMul、Conv2D、BatchMatMul | 通常以主算子为 anchor，融合 bias/activation/cast |
| Layout/Shape | Reshape、Transpose、Permute、Squeeze | 可能是 view，也可能是 real copy |
| Memory/Stateful | Inplace、Assign、Random、IO | 融合和重排受 side effect 限制 |

NPU IR 融合常见 anchor：

- elementwise chain：`Add -> Relu -> Cast`
- matmul epilogue：`MatMul -> BiasAdd -> Activation`
- normalization 子图：`Reduce -> Elementwise -> Reduce -> Elementwise`
- layout transform 附近：`Transpose -> Op -> Transpose`

## Graph Invariants

改写 IR 后要保持几个不变量：

- 每个 tensor value 有明确 producer。
- 每个 use 都能找到合法 def。
- graph output 不能指向已删除 value。
- side-effect op 的相对顺序保持语义。
- metadata 与新 graph 一致。
- 拓扑顺序满足 producer 在 consumer 前。
- fused op 的 attrs 足以让后续 lowering 还原计算语义。

常见 bug：

- 删除 old producer 后，某个 user 仍然指向 old value。
- fused op 输出 metadata 没更新，后续 shape/layout pass 崩掉。
- DCE 删除了 graph output 依赖的 node。
- 把有副作用 op 移动到错误位置。
- 多输出 op 只替换了其中一个 output 的 uses。

## Fused Op 和 Fused Region

融合结果通常有两种表示。

第一种是 fused op：

```text
FusedMatMulBiasRelu
  inputs: A, B, bias
  outputs: Y
  attrs: matmul attrs + bias axis + activation type
```

优点是后端匹配简单，缺点是 fused op 类型会膨胀。

第二种是 fused region：

```text
FusionRegion
  inputs: A, B, bias
  body:
    t0 = MatMul(A, B)
    t1 = Add(t0, bias)
    y = Relu(t1)
  outputs: y
```

优点是表达灵活，缺点是后端要能 lower region 内部 IR。

NPU 后端常见做法是两者结合：简单高频模式变成 specialized fused op；复杂 elementwise chain 用 fused region 或 kernel template 表示。
