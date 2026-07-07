---
order: 3
title: 算子融合总览
updated: 2026-07-05
tags: [ai-compiler, npu, fusion]
status: draft
---

# 算子融合总览

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

算子融合把多个 graph op 合并成一个 fused op 或 fused region。主要目标是减少中间 tensor materialization、减少 GM 读写、减少 kernel launch，并为后端生成更适合 NPU 的计算单元。

## 融合收益来源

不融合：

```text
op1:
  read x from GM
  compute t
  write t to GM

op2:
  read t from GM
  compute y
  write y to GM
```

融合：

```text
fused_op:
  read x from GM
  compute t in register / UB / local buffer
  compute y
  write y to GM
```

收益来源：

- 减少中间 tensor 的 GM write/read。
- 减少 kernel launch。
- 减少 layout transform 或 copy。
- 让 producer-consumer 在片上存储中传递。
- 把 bias、activation、cast 放入主算子 epilogue。
- 给 tiling/schedule 提供更大的优化空间。

## 常见融合类型

### Elementwise Chain

```text
Add -> Relu -> Cast -> Mul
```

特点：

- 语义简单。
- shape 通常相同或可 broadcast。
- memory-bound 场景收益明显。
- 后端常生成一个 fused elementwise kernel。

### MatMul / Conv Epilogue

```text
MatMul -> BiasAdd -> Relu
Conv -> BatchNorm folded -> Activation
```

特点：

- MatMul/Conv 是 anchor。
- 后处理通常在 output tile 上完成。
- 能避免大输出 tensor 多次 GM 往返。
- 需要后端支持 epilogue 或 fused region lowering。

### Reduction 相关融合

```text
ReduceMax -> Sub -> Exp -> ReduceSum -> Div
```

典型例子是 Softmax。特点：

- 涉及 reduce axis。
- 通常需要多阶段计算。
- 融合收益大，但 schedule 复杂。
- 需要考虑数值稳定和中间结果复用。

### Layout / Copy Folding

```text
Transpose -> Op
Op -> Transpose
Cast -> Op
```

目标是让 producer 直接生成 consumer 所需格式，或者让 consumer 接受 producer 当前格式。真实收益取决于 format transform 是否可以消掉。

## 融合的两个判断

融合 pass 通常分成两类判断：

```text
legality: 能不能融合
profitability: 值不值得融合
```

Legality 关注语义正确：

- 数据依赖是否保持。
- side effect 是否不被重排。
- shape/dtype/layout 是否兼容。
- alias/in-place 是否安全。
- 后端能否表达 fused op。

Profitability 关注收益：

- 减少多少 GM 读写。
- 减少多少 kernel launch。
- 是否增加重复计算。
- UB/register/local buffer 是否放得下。
- 是否破坏更高收益的 library call。
- 是否让 tiling 和 lowering 变复杂。

## 融合不是越多越好

过度融合可能带来反效果：

- fused region 太大，register/UB 压力增加。
- 原本可用高性能 library 的 MatMul/Conv 被破坏。
- 多 consumer 下复制 producer，导致重复计算。
- reduction + elementwise 过度融合，让 schedule 复杂且 occupancy 下降。
- layout 不兼容，融合后反而插入更多 format transform。
- dynamic shape 下需要更多 guard，compile cache 命中变差。

NPU IR 融合应同时看 graph 层和后端层：

```text
graph 层：语义、依赖、metadata
backend 层：tiling、memory hierarchy、format、kernel/library support
runtime 层：launch、workspace、cache、guard
```
