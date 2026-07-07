---
order: 4
title: Fusion Legality
updated: 2026-07-05
tags: [ai-compiler, npu, fusion, legality]
status: draft
---

# Fusion Legality

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

Fusion legality 判断“能不能融合”。它只回答语义是否安全，不回答性能是否值得。一个融合机会必须先通过 legality，再进入 profitability 判断。

## 基本条件

```text
candidate subgraph
  -> dependency check
  -> side-effect check
  -> metadata check
  -> alias/in-place check
  -> backend support check
```

输出是：

```text
legal / illegal / need more analysis
```

`need more analysis` 很常见，例如 dynamic shape、unknown alias、unknown layout。工程上通常保守处理：无法证明安全，就不融合。

## Producer-Consumer 依赖

最简单的合法融合：

```text
producer -> tensor -> consumer
```

条件：

- consumer 使用 producer 的输出。
- producer 输出没有其他必须保留 materialized tensor 的 use。
- producer 和 consumer 之间没有不可重排的 op。
- fused op 能产生和 consumer 原输出等价的结果。

多 consumer 情况：

```text
producer -> t -> consumer1
             -> consumer2
```

合法策略取决于融合形式：

- 如果删除 producer，会破坏 consumer2，不合法。
- 如果保留 producer，同时复制 producer 到 consumer1 fused region，语义可能合法，但 profitability 需要评估重复计算。
- 如果 fused region 支持 multi-output，可同时覆盖多个 consumer。

## Side Effect

有副作用的 op 不能随意移动、删除或跨越。

常见 side effect：

- 写内存或 in-place 修改。
- 随机数。
- IO。
- device synchronization。
- shape/runtime state 更新。
- 会抛异常且异常顺序需要保留。

例子：

```text
x -> OpA -> t
stateful_op()
t -> OpB -> y
```

如果融合 `OpA + OpB` 会让 `OpB` 跨过 `stateful_op`，需要证明这个重排不改变语义。无法证明时不融合。

## Shape / Broadcast

Elementwise 融合要求 fused region 内 shape 关系可推导。

```text
x: [N, C]
b: [C]
t = Add(x, b)
y = Relu(t)
```

合法性检查：

- `Add` 的 broadcast 规则合法。
- `Relu` 输出 shape 等于 `Add` 输出 shape。
- fused op 输出 shape 等于原 `Relu` 输出。
- dynamic shape 下需要 guard 或 symbolic equality。

Broadcast 还要结合 layout。`[C]` 在 NCHW 和 NHWC 下对应的物理访问方式可能不同，后端 epilogue 是否支持要单独检查。

## Dtype 和数值语义

Dtype 变化可能影响融合合法性。

```text
t0 = MatMul(fp16, fp16)
t1 = Cast(t0, fp32)
t2 = Add(t1, bias_fp32)
```

如果融合后把 `Add` 提前到 fp16 上做，就改变了数值语义。合法融合必须保持：

- cast 位置等价。
- accumulation dtype 等价。
- rounding 行为等价或 IR 允许 fast-math。
- quant/dequant scale 和 zero point 不丢失。

## Layout / Format

layout/format 不兼容是 NPU 融合常见阻碍。

```text
producer output: ND
consumer expected: NZ
```

合法融合需要满足至少一个条件：

- producer 能直接生成 consumer format。
- consumer 能接受 producer format。
- format transform 是 metadata-only view。
- fused op 内部能显式包含 format transform。

如果融合后需要额外插入真实 copy，要进入 profitability 判断；如果后端无法表达，则 legality 失败。

## Alias 和 In-place

alias 表示多个 tensor value 可能共享底层 buffer。

```text
view = Reshape(x)
InplaceAdd(view, y)
z = Use(x)
```

这里 `view` 和 `x` 共享 storage。融合或重排 `InplaceAdd` 可能改变 `z` 看到的数据。

检查项：

- op 是否 in-place。
- output 是否 alias input。
- view op 是否 metadata-only。
- fused region 是否改变读写顺序。
- 删除中间 tensor 是否影响外部 alias。

## Reduction 语义

Reduction 融合要检查 reduce axis 和数值稳定。

```text
ReduceSum -> Add
```

通常比 elementwise 更复杂：

- reduce axis 是否保持。
- keepdim 是否保持。
- 后续 broadcast 是否匹配 reduce 后 shape。
- 浮点加法重排是否允许。
- 是否需要多阶段 partial reduce。
- dynamic shape 下 reduce size 是否影响 schedule。

Softmax 类模式还要保证 `max/sub/exp/sum/div` 的顺序不破坏数值稳定。

## Backend Support

Graph 层语义合法，不代表后端能实现。

后端支持检查：

- fused op 类型是否有 lowering。
- dtype/layout/format 组合是否支持。
- fused region 内 op 是否都能 lower 到同一 kernel。
- UB/L1/L0 memory scope 是否能表达中间数据。
- 是否存在无法融合的 library call boundary。

例如 `MatMul + Bias + Relu` 在很多后端是合法且支持的；`MatMul + arbitrary Reduce + Transpose` 可能语义合法，但后端没有对应 lowering。

## Legality Checklist

- producer-consumer 依赖明确。
- graph output 不依赖被删除的中间 value。
- 多 consumer 有明确处理策略。
- side-effect op 不被错误重排。
- shape/broadcast 可证明。
- dtype/cast/quant 语义保持。
- layout/format 可表达。
- alias/in-place 安全。
- reduction axis 和数值语义保持。
- 后端存在 fused op 或 fused region lowering。
