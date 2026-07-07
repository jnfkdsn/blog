---
order: 3
title: Metadata Inference 基础
updated: 2026-07-05
tags: [ai-compiler, metadata, shape, layout]
status: draft
---

# Metadata Inference 基础

相关入口：[AI Compiler 基础](/notes/compile/ai-compiler/basics/)

Metadata inference 为 graph 上的 tensor value 补充 shape、dtype、layout、alias、device 等信息。没有 metadata，graph rewrite 和 lowering 很难安全进行。

## Metadata 类型

常见字段：

```text
shape
dtype
layout
stride
device
requires_grad
alias / view
memory_format
symbolic shape constraint
```

AI Compiler 里的“语义正确”和“后端可执行”都依赖这些信息。

## Shape Inference

shape inference 根据 op schema 推导输出 shape。

```text
Add([N, C], [C]) -> [N, C]
MatMul([M, K], [K, N]) -> [M, N]
ReduceSum([N, C], axis=1, keepdim=true) -> [N, 1]
```

它回答：

- 输出 tensor 有多大。
- broadcast 是否合法。
- reduction 后维度如何变化。
- reshape/view 是否保持元素数量。
- dynamic shape 下需要哪些约束。

## Dtype Inference

dtype inference 决定输出类型和内部计算类型。

```text
Add(fp16, fp16) -> fp16
Cast(fp16 -> fp32) -> fp32
MatMul(fp16, fp16) -> fp16 output, fp32 accumulator? 
```

融合时 dtype 很关键，因为移动 cast 或改变 accumulation dtype 会改变数值语义。

## Layout Inference

layout / stride / memory_format 决定 tensor 如何在内存中排列。

```text
NCHW
NHWC
contiguous
channels-last
vendor format
```

Graph rewrite 需要知道：

- op 是否支持当前 layout。
- layout transform 是否是 view 还是 copy。
- producer 能否直接生成 consumer 需要的 layout。
- fused op 的输出 layout 是什么。

## Alias / View Inference

一些 op 不分配新 buffer：

```text
view
reshape
squeeze
slice
transpose
```

它们可能和输入共享 storage。alias 信息影响：

- 能否删除中间 value。
- in-place op 是否安全。
- buffer 是否可复用。
- pass 是否能重排 op。

## Metadata Inference 的位置

典型 pipeline：

```text
graph import
  -> metadata inference
  -> graph rewrite
  -> metadata update / re-inference
  -> lowering
```

每次 graph rewrite 后，metadata 可能失效。工程上通常有两种策略：

- 局部更新：rewrite pass 为新 op 计算 metadata。
- 全图重推：简单但成本更高。

NPU IR 融合通常需要局部更新，因为 fused op 的 metadata 是后续 lowering 的输入。
