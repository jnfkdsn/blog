---
order: 5
title: Lowering 与 Runtime 基础
updated: 2026-07-05
tags: [ai-compiler, lowering, runtime]
status: draft
---

# Lowering 与 Runtime 基础

相关入口：[AI Compiler 基础](/notes/compile/ai-compiler/basics/)

Lowering 把 graph/tensor IR 下沉到目标后端。Runtime 负责执行编译产物。Graph 层优化如果不能被 lowering 和 runtime 接住，就不能变成真实性能收益。

## Lowering 层次

```text
Graph IR
  -> Tensor IR / Loop IR
  -> Target IR
  -> kernel source / binary / library call
```

Graph IR：

```text
y = Relu(Add(x, b))
```

Tensor/Loop IR：

```text
for i in range(numel):
  y[i] = max(x[i] + b[i], 0)
```

Target lowering：

```text
Triton kernel
CUDA kernel
C++ kernel
NPU kernel
vendor library call
```

## Library Call 和 Generated Kernel

AI Compiler 后端通常有两种执行方式：

- 调用已有高性能库。
- 生成自定义 kernel。

Library call 适合：

- MatMul。
- Conv。
- BatchMatMul。
- 高度优化的 Softmax/LayerNorm。

Generated kernel 适合：

- elementwise chain。
- simple reduction。
- fused epilogue。
- shape/layout 特化场景。

Fusion pass 要考虑：融合后是否还能使用高性能 library。不能为了少一个中间 tensor 而破坏大算子的主性能路径。

## Runtime 负责什么

Runtime 不是单纯“调用 kernel”。常见职责：

- 分配 output tensor。
- 分配 workspace。
- 做 memory planning。
- 管理 compiled artifact cache。
- 检查 shape/dtype/layout guard。
- 选择 kernel 或 fallback。
- 管理 stream/event。
- 调用 kernel 或 library。

执行链路：

```text
check guards
  -> allocate outputs/workspace
  -> get compiled kernel from cache
  -> launch kernel / library call
  -> return tensor handles
```

## Guard 和 Cache

JIT 型 AI Compiler 需要判断已编译代码能否复用。

cache key 可能包含：

- graph structure。
- shape / symbolic shape。
- dtype。
- layout。
- device。
- compile options。
- target capability。

guard 用于运行时检查：

```text
input shape == expected
input dtype == expected
input stride/layout satisfies constraint
device matches
```

dynamic shape 下，guard 和 cache key 的设计会影响 compile 次数和运行时稳定性。

## Memory Planning

Memory planning 决定中间 tensor 和 workspace 如何分配/复用。

优化目标：

- 降低 peak memory。
- 复用生命周期不重叠的 buffer。
- 减少分配次数。
- 保证 alignment。
- 避免不必要的 host-device copy。

Fusion 会改变 memory planning：

- 删除中间 tensor。
- 改变 buffer lifetime。
- 可能新增 fused kernel workspace。
- 改变 output layout。

因此 fusion 后通常要重新做 lifetime analysis 或 memory planning。

## 对 NPU IR 的意义

NPU IR 融合的 graph 结果最终要满足 lowering/runtime：

- fused op 有后端实现。
- shape/dtype/layout 能确定。
- workspace 可分配。
- kernel launch 参数可生成。
- memory scope 能映射到 GM/UB/L1/L0。
- runtime 能缓存和复用编译产物。

这也是为什么 NPU IR fusion 不能只看 graph pattern，还要看后端 capability 和 runtime 成本。
