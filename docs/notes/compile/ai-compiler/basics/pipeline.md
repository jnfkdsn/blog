---
order: 1
title: AI Compiler Pipeline
updated: 2026-07-05
tags: [ai-compiler, pipeline]
status: draft
---

# AI Compiler Pipeline

相关入口：[AI Compiler 基础](/notes/compile/ai-compiler/basics/)

AI Compiler 的目标是把 tensor program 变成更适合目标硬件执行的 graph、kernel 或 library 调用。它处理的不是传统源码字符串，而是模型、Python 函数、框架 graph 或标准化 IR。

## 总体链路

```text
model / function / graph
  -> capture / import
  -> graph IR
  -> metadata inference
  -> graph rewrite
  -> fusion
  -> tensor IR / schedule
  -> target lowering
  -> codegen / library call
  -> runtime execution
```

每一层的输入输出：

| 阶段 | 输入 | 输出 | 核心问题 |
|---|---|---|---|
| Capture / Import | Python、framework graph、ONNX/HLO | graph IR | 如何稳定表示原程序 |
| Metadata Inference | graph IR | 带 shape/dtype/layout 的 graph | 每个 tensor 是什么 |
| Graph Rewrite | graph IR + metadata | 改写后的 graph | 语义保持变换 |
| Fusion | producer-consumer 子图 | fused op/region | 减少 memory 和 launch |
| Tensor IR / Schedule | graph op 或 fused region | loop/buffer/schedule | 如何组织计算 |
| Lowering | tensor IR / high-level op | target IR / kernel / library call | 如何映射到硬件 |
| Runtime | compiled artifact | 执行结果 | memory、cache、stream、guard |

## Capture / Import

不同系统入口不同：

- PyTorch 2.x：从 Python frame / bytecode 捕获 FX Graph。
- ONNX Runtime：导入 ONNX graph。
- XLA：接收 HLO/StableHLO。
- TVM：Relay / Relax / TensorIR。
- MLIR 系统：导入或构造某个 dialect。

这一阶段的关键产物是 graph：计算从动态图或框架表示变成编译器可分析、可改写的 IR。

## Graph 级优化

Graph 层保留 op 语义，适合做：

- decomposition：把复杂 op 拆成更基础的 op。
- canonicalization：把等价写法归一。
- constant folding：计算常量子图。
- DCE：删除无用 op。
- CSE：合并重复计算。
- layout rewrite：调整或消除 layout transform。
- fusion：合并 producer-consumer 子图。

Graph 层优化的优势是语义信息还在，知道一个 op 是 `MatMul`、`Softmax`、`LayerNorm`。下沉到 loop 或 kernel 后，这些高层语义可能会丢失。

## Tensor IR / Schedule

Tensor IR 描述的是 op 内部的循环、buffer 和访问模式。

```text
for i in M:
  for j in N:
    C[i, j] = A[i, j] + B[i, j]
```

这一层适合处理：

- tiling。
- vectorization。
- memory scope。
- parallel axis。
- reduction schedule。
- buffer reuse。

NPU IR 融合如果要影响后端性能，最终会连接到 tensor IR / schedule：融合后的子图是否能被表示成合理 loop 和 buffer 计划。

## Lowering 和 Runtime

Lowering 把 graph/tensor IR 映射到硬件：

```text
graph op / fused region
  -> tensor loop / schedule
  -> target kernel or library call
  -> runtime launch
```

Runtime 负责：

- memory planning。
- workspace 分配。
- compiled artifact cache。
- shape guard。
- stream/event。
- kernel launch。
- fallback。

AI Compiler 的性能不只来自 codegen。过多 kernel launch、过细 cache key、workspace 反复分配、同步过多，都可能吞掉 graph 优化收益。
