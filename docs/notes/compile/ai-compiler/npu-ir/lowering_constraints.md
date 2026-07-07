---
order: 8
title: NPU Lowering 约束
updated: 2026-07-05
tags: [ai-compiler, npu, lowering, codegen]
status: draft
---

# NPU Lowering 约束

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

融合 pass 不能只在 graph 层判断。融合后的 IR 必须能 lower 到 NPU 后端，否则 graph 层融合只是把问题推迟。

## Lowering 需要的信息

fused op / fused region 至少要提供：

```text
inputs / outputs
op sequence or fused op type
shape / dtype / layout / format
broadcast / reduce axis
accumulation dtype
memory scope requirements
attrs for original ops
backend capability key
```

如果这些信息缺失，后端无法生成正确 kernel 或选择 library call。

## Library Call Boundary

某些高性能 op 依赖厂商 library 或专用 kernel：

- MatMul
- Conv
- BatchMatMul
- LayerNorm / Softmax 的高性能模板

融合不能破坏这些边界，除非后端有对应 fused epilogue 或 fused template。

安全模式：

```text
MatMul -> Bias -> Activation
```

如果 NPU backend 支持 MatMul epilogue，这种融合通常可 lower。

危险模式：

```text
MatMul -> Transpose -> Reduce -> Elementwise
```

如果后端没有 fused lowering，融合后可能失去 library call，性能下降或无法 codegen。

## Tiling 约束

融合后要重新考虑 tiling：

```text
tile shape
input tile size
output tile size
intermediate tile size
UB/L1/L0 capacity
alignment
```

Elementwise chain 的 tiling 通常与输出 shape 对齐。

MatMul epilogue 的 tiling 通常由 MatMul 决定，后处理跟着 output tile 走。

Reduction fusion 的 tiling 由 reduce axis、并行策略和 partial buffer 决定。

如果融合后需要同时满足多个 op 的不同 tiling，可能会失败。

## Memory Scope

Lowering 要决定中间值在哪个 memory scope：

```text
register
UB
L1
L0
GM workspace
```

例子：

- Elementwise 中间值可以是 register/UB tile。
- MatMul accumulator 在 L0C 或等价内部 buffer。
- Bias/activation 可以在 output tile 上处理。
- Reduction partial 可能需要 UB 或 GM workspace。

融合 pass 如果能提前标注 memory intent，可以帮助后端判断是否可实现。

## DMA / Pipeline

NPU kernel 常涉及搬运与计算流水：

```text
CopyIn
  -> Compute
  -> CopyOut
```

融合后可能改变流水：

- 输入更多，CopyIn 压力增加。
- 中间值减少，CopyOut 压力下降。
- 计算阶段更长，可能更好隐藏搬运。
- temporary 更多，double buffer 可能放不下。

Lowering 要检查 fused schedule 是否还能形成稳定 pipeline。

## Format 和 Alignment

后端通常对 format/alignment 有约束：

- 地址对齐。
- tile 大小对齐。
- channel 维对齐。
- fractal/NZ/ND 格式转换。
- vector instruction 的 block size。
- cube instruction 的 M/N/K tile 限制。

融合 pass 的 metadata 必须把这些信息传给 lowering。缺失 alignment 或 format 信息时，后端可能需要插入额外 copy 或 fallback。

## Dynamic Shape

Dynamic shape 下，lowering 需要处理：

- runtime shape guard。
- symbolic tile size。
- fallback kernel。
- compile cache key。
- shape-dependent workspace。

融合可能增加 dynamic shape 的复杂度。例如 elementwise chain 通常容易处理；reduction + broadcast fusion 需要证明 reduce 后 shape 和 broadcast shape 在 runtime 仍然匹配。

## Lowering Failure 的处理

工程中需要明确 fallback：

```text
fusion candidate
  -> graph-level legal
  -> profitability accepted
  -> lowering failed
```

处理方式：

- 在 fusion 前加入 backend capability check，避免生成无法 lower 的 fused op。
- lowering 失败时回退到未融合 graph。
- 保留 debug dump，记录失败 pattern、shape、dtype、layout。
- 对高频失败 pattern 补充 legality/profitability 规则。

NPU IR 融合 pass 最好把 backend capability 作为输入，而不是事后才发现无法 codegen。

## 检查项

- fused op 是否有 lowering。
- dtype/layout/format 组合是否支持。
- tiling 是否能同时满足所有 fused ops。
- UB/L1/L0 是否放得下 temporary。
- DMA pipeline 是否仍然合理。
- 是否破坏高性能 library call。
- dynamic shape guard 是否可表达。
- fallback 是否保留。
