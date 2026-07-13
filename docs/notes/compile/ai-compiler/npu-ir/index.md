---
order: 0
title: NPU IR 融合知识库
updated: 2026-07-05
tags: [ai-compiler, npu, ir, fusion, graph]
status: draft
---

# NPU IR 融合知识库

相关入口：[AI Compiler](/notes/compile/ai-compiler/) / [AI Compiler 基础](/notes/compile/ai-compiler/basics/) / [传统编译器](/notes/compile/traditional/)

NPU IR 算子融合的核心任务：

```text
识别可融合子图
  -> 判断语义是否合法
  -> 估算融合收益和硬件代价
  -> 改写 IR
  -> 维护 metadata / use-def / pass invariants
  -> 交给后续 lowering 和 runtime
```

融合不是简单地把几个 op 合成一个 op。它同时涉及 graph 语义、tensor metadata、memory traffic、layout、buffer lifetime、NPU 片上存储和后端 codegen 能力。

进入这部分前，先具备以下基础概念：

- Graph IR 如何表示 op、tensor value 和 use-def。
- shape、dtype、layout metadata 如何在 graph 上推导。
- graph rewrite pass 如何匹配 pattern 并改写 IR。
- lowering 如何把 graph/tensor IR 下沉到 kernel 或 library call。
- runtime 如何处理 memory、stream、cache 和 guard。

## 主线结构

| 笔记 | 内容 |
|---|---|
| [NPU IR 表示](/notes/compile/ai-compiler/npu-ir/graph_ir) | op、tensor value、use-def、side effect、graph invariants |
| [HFusion IR](/notes/compile/ai-compiler/npu-ir/hfusion) | HFusion structured op、ins/outs、常见 op、转换关系 |
| [Shape、Dtype、Layout Metadata](/notes/compile/ai-compiler/npu-ir/metadata_shape_layout) | shape/dtype/layout/format/alias 信息如何支撑融合 |
| [算子融合总览](/notes/compile/ai-compiler/npu-ir/fusion_overview) | 融合类型、收益来源、失败原因 |
| [Fusion Legality](/notes/compile/ai-compiler/npu-ir/fusion_legality) | 能不能融合：语义、依赖、alias、layout、后端支持 |
| [Fusion Profitability](/notes/compile/ai-compiler/npu-ir/fusion_profitability) | 值不值得融合：memory traffic、launch、tiling、buffer 压力 |
| [Fusion Pass 实现](/notes/compile/ai-compiler/npu-ir/fusion_pass_impl) | pattern matching、rewrite、metadata 更新、测试 |
| [Memory、Buffer 与 NPU 存储层次](/notes/compile/ai-compiler/npu-ir/memory_buffer) | GM/UB/L1/L0、lifetime、workspace、memory planning |
| [NPU Lowering 约束](/notes/compile/ai-compiler/npu-ir/lowering_constraints) | 融合结果下沉到 NPU kernel 的约束 |

## 一条融合 pass 的基本链路

```text
Graph IR
  -> collect candidate patterns
  -> check legality
  -> estimate profitability
  -> create fused op / fused region
  -> redirect uses
  -> delete dead nodes
  -> infer metadata
  -> verify graph
```

每一步都要维护 IR 的不变量：

- value 的 producer 唯一。
- use-def 链正确。
- graph 拓扑顺序正确。
- side-effect op 的相对顺序不被破坏。
- tensor metadata 与新 op 输出一致。
- 后续 lowering 能识别 fused op 或 fused region。

## 对传统编译器基础的依赖

| 传统基础 | 在 NPU IR 融合里的作用 |
|---|---|
| IR / SSA value | 表示 tensor value 的定义和使用 |
| use-def / def-use | 找 producer-consumer、替换 use、删除 dead node |
| dataflow analysis | 判断值的传播、活跃范围、依赖关系 |
| side effect / alias | 判断是否能重排、删除、融合 |
| pass pipeline | 管理 fusion pass 前后的 analysis 和 IR 校验 |
| loop tiling / vectorization | 融合后 lower 到 NPU kernel 时需要考虑 |
| memory optimization | 融合的主要收益来自减少 GM 读写和中间 tensor materialization |
