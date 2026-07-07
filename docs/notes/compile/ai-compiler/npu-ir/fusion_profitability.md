---
order: 5
title: Fusion Profitability
updated: 2026-07-05
tags: [ai-compiler, npu, fusion, cost-model]
status: draft
---

# Fusion Profitability

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

Fusion profitability 判断“值不值得融合”。一个融合在语义上合法，也可能因为资源压力、重复计算、layout 代价或后端限制而不划算。

## 成本模型的输入

常见输入：

```text
graph pattern
tensor shape / dtype / layout
producer output size
consumer count
estimated GM read/write
kernel launch count
NPU memory hierarchy
tiling constraints
backend support
```

输出：

```text
fuse / do not fuse / defer to later pass
```

## Memory Traffic

融合最常见收益来自减少 GM 读写。

不融合：

```text
producer writes T to GM
consumer reads T from GM
```

融合：

```text
T stays in register / UB / local buffer
```

粗略收益：

```text
saved_bytes = size(T) * (write_once + read_once)
```

例如 `T` 是 `fp16 [1024, 4096]`：

```text
size(T) = 1024 * 4096 * 2B = 8MB
saved traffic ≈ 16MB
```

如果这个 chain 有多个 elementwise op，中间 tensor 越多，融合收益越明显。

## Kernel Launch

小算子场景里，kernel launch overhead 可能比计算本身更显著。

```text
Add -> Relu -> Cast
```

不融合需要 3 次 kernel launch；融合后 1 次。对小 shape、batch size 小、推理服务中频繁调用的子图，launch 减少是重要收益。

但对于大 MatMul/Conv，主要耗时在计算，launch overhead 占比小。此时融合收益更多来自 epilogue 避免输出 tensor 多次 GM 往返。

## 重复计算

多 consumer 融合可能引入重复计算。

```text
t = Exp(x)
y1 = Add(t, a)
y2 = Mul(t, b)
```

如果分别融合：

```text
FusedExpAdd(x, a)
FusedExpMul(x, b)
```

`Exp(x)` 被计算两次。是否值得取决于：

- `Exp` 计算成本。
- `t` 的 GM write/read 成本。
- `t` 是否很大。
- 两个 consumer 是否能做 multi-output fusion。

计算贵的 producer 通常不适合复制；纯简单 elementwise producer 在 memory-bound 场景中有时可以复制。

## Buffer 压力

融合后中间值留在片上，不代表免费。

资源压力包括：

- register 数量。
- UB buffer 大小。
- L1/L0 tile 空间。
- temporary workspace。
- queue / event / pipeline 资源。

过度融合可能导致：

- tile size 变小。
- occupancy 下降。
- spill 到更慢内存。
- double buffer 放不下。
- pipeline stage 不平衡。

NPU 上要特别关注 UB/L1/L0 容量。elementwise chain 通常可 streaming，不需要保存全部中间 tensor；reduction 或多输出 fusion 可能需要更多临时 buffer。

## Layout 和 Copy

融合如果消掉 layout transform，收益很大；如果引入额外 layout transform，可能亏。

收益模式：

```text
TransData -> Elementwise
```

如果 elementwise 可直接处理 TransData 前的 format，则可以删除 transform。

亏损模式：

```text
OpA(format A) -> OpB(format B)
```

强行融合后 fused op 内部仍然要做 A->B copy，甚至让后端失去原本高效 kernel。

Profitability 要估算：

- layout transform 是否真实 copy。
- copy 数据量。
- format 对 vector/cube 指令效率的影响。
- alignment 是否变差。

## 主算子保护

MatMul/Conv 常常有高度优化的 library/kernel。融合不能破坏主算子的最优实现。

通常适合融合进主算子 epilogue：

- bias add
- activation
- simple cast
- scale / dequant

需要谨慎：

- complex reduction
- large transpose
- consumer 会改变 MatMul tiling 的复杂逻辑
- fusion 后无法使用高性能 Cube/MatMul kernel

判断原则：

```text
主算子性能损失 < epilogue 融合收益
```

如果融合导致 MatMul 从 library call 退化为普通 generated kernel，多数情况下不划算。

## Dynamic Shape 和 Cache

dynamic shape 下，融合可能增加 guard 和编译版本数量。

```text
shape [N, C]
N dynamic
C dynamic
```

如果 fused schedule 依赖具体 `N/C`，可能生成多个 specialized kernel。收益需要和 compile/cache 成本一起看：

- guard 数量是否增加。
- cache key 是否变细。
- 编译时间是否上升。
- runtime fallback 是否变多。

## 简化 Cost Model

一个实用的早期 cost model 可以先用规则驱动：

```text
if illegal:
  reject
if elementwise chain and output single-use:
  fuse
if matmul/conv epilogue in supported set:
  fuse
if producer is expensive and multi-use:
  reject copy-fusion
if fused temporary exceeds UB/register budget:
  reject
if fusion introduces extra layout copy:
  reject unless copy can be folded
```

再逐步加入数值估计：

```text
score =
  saved_global_memory_bytes * memory_weight
  + saved_launch_count * launch_weight
  - duplicated_compute_cost
  - extra_layout_copy_cost
  - resource_pressure_penalty
```

早期工程里，不需要一开始追求复杂模型。先把明显收益和明显风险的规则做准，比写一个不可靠的复杂模型更重要。
