---
order: 2
title: Shape、Dtype、Layout Metadata
updated: 2026-07-05
tags: [ai-compiler, npu, shape, layout, metadata]
status: draft
---

# Shape、Dtype、Layout Metadata

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

融合 pass 不能只看 op type。tensor 的 shape、dtype、layout、format、alias、memory scope 决定融合是否合法，也决定融合后能否高效 lower 到 NPU。

## Metadata 内容

一个 tensor value 常见 metadata：

```text
shape: [N, C, H, W] / symbolic shape / dynamic dim
dtype: fp16 / fp32 / bf16 / int8 / ...
layout: logical dimension order
format: ND / NCHW / NHWC / NZ / fractal / vendor format
stride: 每个维度的步长
contiguous: 是否连续
device: NPU / CPU
alias: 是否和其他 value 共享 buffer
memory_scope: GM / UB / L1 / L0 / workspace
```

在 NPU 场景里，`layout` 和 `format` 要区分：

- `layout` 更偏逻辑维度顺序，例如 NCHW、NHWC。
- `format` 更偏物理存储格式，例如 ND、NZ、fractal、5HD 或厂商内部格式。

融合 pass 需要知道两类信息：

```text
语义信息：shape / dtype / broadcast / reduce axis
后端信息：format / stride / alignment / memory scope
```

## Shape Inference

shape inference 负责为每个 op 输出推导 shape。

Elementwise：

```text
Add([N, C], [N, C]) -> [N, C]
Add([N, C], [C])    -> [N, C]    # broadcast
```

MatMul：

```text
MatMul([M, K], [K, N]) -> [M, N]
BatchMatMul([B, M, K], [B, K, N]) -> [B, M, N]
```

Reduce：

```text
ReduceSum([N, C, H, W], axis=C, keepdim=false) -> [N, H, W]
ReduceSum([N, C, H, W], axis=C, keepdim=true)  -> [N, 1, H, W]
```

融合依赖 shape inference 的位置：

- 判断 producer 输出 shape 是否能作为 consumer 输入。
- 判断 broadcast 语义是否保持。
- 判断 fused op 输出 shape。
- 判断 dynamic shape 下是否需要 guard。
- 判断 tiling 和 buffer 大小。

## Dtype Inference

dtype 决定计算精度、cast 插入和后端指令选择。

常见问题：

- `fp16 + fp16 -> fp16` 还是 accumulate 到 `fp32`。
- `matmul fp16` 输出是否是 `fp16`，中间 accumulator 是否 `fp32`。
- `int8` 算子是否带 scale、zero point、dequant。
- `Cast` 能不能被融合进 producer 或 consumer。

融合时要检查：

- fused op 内部 dtype 转换是否等价。
- cast 位置变化是否改变舍入行为。
- reduction accumulation dtype 是否保持。
- NPU 后端是否支持该 dtype 组合。

典型模式：

```text
MatMul(fp16, fp16) -> fp16
  -> Cast(fp32)
  -> Add(fp32)
```

如果后端 MatMul 支持 fp32 accumulator 或 epilogue cast，可能融合；否则移动 cast 会改变数值语义。

## Layout / Format Propagation

layout propagation 记录每个 tensor 当前使用的逻辑布局和物理格式。

示例：

```text
Conv(NCHW) -> Relu -> Add
```

如果 `Relu/Add` 都支持 NCHW，则可以保持布局不变。

如果中间出现 format transform：

```text
OpA(ND) -> TransData(NZ) -> OpB(NZ)
```

融合要判断：

- `TransData` 是真实 copy 还是 metadata-only view。
- producer 能不能直接生成 consumer 需要的 format。
- consumer 能不能接受 producer 当前 format。
- format transform 能不能被前后 op 吸收。

NPU 上 layout/format 常常比普通 CPU/GPU graph 更重要，因为硬件矩阵单元、vector 单元、DMA 搬运对 format 和 alignment 有明确要求。

## Broadcast Metadata

Broadcast 不是简单的 shape 相等。需要记录哪些 axis 被扩展。

```text
x: [N, C, H, W]
bias: [C]
y = x + bias
```

这里 bias 通常沿 `N/H/W` broadcast。融合进 MatMul/Conv epilogue 时，要确认 bias axis 和主算子输出 channel 维一致。

检查项：

- broadcast 后 shape 是否等于 consumer 期望。
- broadcast axis 在当前 layout 下是否仍然对应同一语义维度。
- broadcast value 是否可以放入 scalar/register/UB。
- dynamic shape 下 broadcast 维度是否可证明为 1 或相等。

## Alias 和 View

一些 op 不产生新 buffer，而是 view：

```text
Reshape
Squeeze
Unsqueeze
View
Slice
Transpose  # 有时是 view，有时需要 copy
```

融合时要区分：

- metadata-only view：只改变 shape/stride，不写新内存。
- materialized copy：产生新 buffer。
- in-place view：多个 tensor value 共享底层 storage。

alias 信息影响：

- 是否可以删除中间 value。
- 是否可以重排 op。
- 是否可以复用 buffer。
- 是否存在写后读、读后写冲突。

## Metadata 更新

融合 pass 改写 IR 后必须更新 metadata。

```text
old:
  t0 = Add(x, y)      # shape [N,C], dtype fp16
  t1 = Relu(t0)       # shape [N,C], dtype fp16

new:
  t1 = FusedAddRelu(x, y)
```

`FusedAddRelu` 的输出 metadata 应等于旧 `Relu` 输出 metadata，而不是简单复制 `Add` 输出。多数 elementwise chain 两者相同，但 reduction、cast、layout transform 附近不能这么假设。

最低限度的 verifier：

- 所有 op 输入 dtype/shape 满足 op schema。
- 所有 graph output metadata 存在。
- fused op output metadata 与替换前 output 等价。
- layout/format 不丢失。
- alias 信息不凭空消失。
