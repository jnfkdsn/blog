---
order: 7
title: Memory、Buffer 与 NPU 存储层次
updated: 2026-07-05
tags: [ai-compiler, npu, memory, buffer, fusion]
status: draft
---

# Memory、Buffer 与 NPU 存储层次

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

NPU IR 融合的主要收益通常来自 memory：减少 GM 中间 tensor 写回、减少重复读取、让 producer-consumer 在片上存储中传递。理解 buffer lifetime 和 NPU 存储层次，是判断融合是否值得的基础。

## 存储层次

抽象层次：

```text
GM / HBM
  -> UB / L1
  -> L0A / L0B / L0C
  -> register / vector lane / cube internal
```

不同 NPU 名称会不同，但核心问题类似：

- GM 容量大，带宽相对片上存储低。
- UB/L1 容量小，带宽高，适合 tile 和中间值。
- L0 更靠近矩阵计算单元，主要服务 Cube/MatMul 类计算。
- DMA/MTE 负责不同层级之间搬运。

融合的目标之一：

```text
中间 tensor 不落 GM
```

但中间值必须放到某个地方：register、UB、L1、workspace，或者被 streaming 消费。

## Tensor Materialization

materialization 表示中间结果被真实写入某个 buffer。

不融合：

```text
t = Add(x, y)     # materialize t in GM
z = Relu(t)       # read t from GM
```

融合：

```text
z = FusedAddRelu(x, y)
```

`t` 可能只是 kernel 内部的临时值，不再成为 graph-level tensor buffer。

materialization 不是一定坏。以下情况可能必须 materialize：

- 中间 tensor 是 graph output。
- 中间 tensor 有多个 consumer。
- 后续 op 需要不同 schedule 或不同 device。
- 中间值太大，片上存储放不下。
- consumer 需要随机访问 producer 全量输出。

## Buffer Lifetime

buffer lifetime 是一个 tensor buffer 从分配到最后一次使用的区间。

融合会改变 lifetime：

```text
before:
  t buffer live from producer end to consumer start/end

after:
  t removed or becomes kernel-local temporary
```

收益：

- graph-level memory planner 可以少分配一个 buffer。
- peak memory 可能下降。
- GM traffic 下降。

风险：

- fused kernel 内部 temporary 增加。
- multi-output fusion 可能延长某些值的 lifetime。
- 为了保留另一个 consumer，producer 仍需 materialize。

## Workspace

workspace 是运行时临时内存。融合可能减少 workspace，也可能增加 workspace。

减少：

- 删除中间 tensor buffer。
- 删除 layout transform buffer。

增加：

- fused reduction 需要 partial buffer。
- fused op 需要额外 tile staging。
- layout conversion 被移到 fused kernel 内部。

对 NPU 后端，workspace 还要考虑：

- 是否在 GM 分配。
- 是否可复用。
- 是否影响 stream 并发。
- 是否需要特殊对齐。

## UB / L1 压力

融合后中间结果如果进入 UB/L1，需要检查容量。

粗略公式：

```text
tile_memory =
  sum(input tiles)
  + sum(output tiles)
  + sum(intermediate tiles)
  + temporary buffers
```

如果 `tile_memory` 超过 UB/L1，后端只能：

- 缩小 tile。
- 增加分块次数。
- 把部分中间值 spill 到 GM。
- 放弃融合。

过度融合的常见问题就是中间临时值太多，导致 tile 变小，吞吐下降。

## Streaming Fusion

Elementwise chain 通常可以 streaming：

```text
load x tile
load y tile
t = add(x, y)
z = relu(t)
store z tile
```

这里不需要保存完整 `t`，只需要 tile 内临时值。Streaming fusion 对 UB 压力较小，通常收益稳定。

Reduction fusion 不一定能 streaming：

```text
ReduceSum -> Div
```

`Div` 需要 reduce 结果，可能要先完成某个轴的归约，再广播回去。Softmax 这类模式需要多阶段处理，融合后 schedule 更复杂。

## Memory Planning 和 Fusion 的关系

Memory planner 负责复用 graph-level buffer。Fusion pass 会改变 planner 的输入：

- 删除中间 tensor。
- 新增 fused op output。
- 修改 lifetime。
- 可能新增 workspace。

理想 pipeline：

```text
fusion
  -> DCE
  -> metadata inference
  -> buffer lifetime analysis
  -> memory planning
  -> lowering
```

如果先做 memory planning 再 fusion，fusion 后需要重新分析 lifetime。

## NPU 视角的检查项

- 中间 tensor 是否从 graph-level buffer 删除。
- 节省的 GM traffic 有多大。
- fused temporary 是否放得进 UB/L1/L0。
- 是否需要额外 workspace。
- layout transform buffer 是否被删除或内联。
- 多 consumer 是否导致 producer 仍需 materialize。
- fused kernel 是否破坏 double buffer / DMA pipeline。
- output 是否仍满足 alignment 和 format 要求。
