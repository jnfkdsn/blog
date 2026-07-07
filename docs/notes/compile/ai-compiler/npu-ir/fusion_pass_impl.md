---
order: 6
title: Fusion Pass 实现
updated: 2026-07-05
tags: [ai-compiler, npu, fusion, pass]
status: draft
---

# Fusion Pass 实现

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

Fusion pass 的工程核心不是“识别到一个模式”这么简单，而是完整维护 graph invariants：pattern 匹配、合法性检查、收益判断、IR 改写、metadata 更新、dead node 删除、pass 后验证。

## Pass 输入输出

输入：

```text
Graph IR
op registry / schema
shape dtype layout metadata
target capability
cost model config
```

输出：

```text
updated Graph IR
new fused op / fused region
updated metadata
invalidated analyses
fusion statistics
```

## 基本流程

```text
for node in topological_order(graph):
  pattern = match_candidate(node)
  if not pattern:
    continue
  if not check_legality(pattern):
    continue
  if not check_profitability(pattern):
    continue
  rewrite_graph(pattern)
  update_metadata()
  run_local_dce()
verify_graph()
```

一些实现会先收集所有 candidate，再统一选择；另一些实现会边遍历边改写。边遍历边改写简单，但要小心 iterator 失效和拓扑顺序变化。

## Pattern Matching

Pattern 可以从 consumer 往 producer 找，也可以从 producer 往 consumer 找。

consumer-driven：

```text
Relu
  input producer is Add
```

适合 epilogue 和 elementwise chain，因为 consumer 类型常是融合尾部。

producer-driven：

```text
MatMul
  users include BiasAdd
  BiasAdd user is Relu
```

适合以 MatMul/Conv 为 anchor 的融合。

常见 pattern 表达：

```text
MatMul -> BiasAdd -> Relu
Add -> Relu
Cast -> Add
ReduceMax -> Sub -> Exp -> ReduceSum -> Div
```

Pattern match 只负责找形状，不负责最终决定。合法性和收益必须独立检查。

## Legality Hook

每类 fusion pattern 可以有自己的 legality hook：

```text
check_matmul_epilogue(pattern):
  check single output
  check bias broadcast axis
  check activation supported
  check dtype/cast semantics
  check layout support
  check backend capability
```

通用 legality：

- use-def 完整。
- side-effect 安全。
- alias/in-place 安全。
- graph output 不被破坏。
- fused output metadata 等价。

## Rewrite

以 `MatMul -> BiasAdd -> Relu` 为例：

```text
old:
  t0 = MatMul(A, B)
  t1 = BiasAdd(t0, bias)
  y  = Relu(t1)

new:
  y = FusedMatMulBiasRelu(A, B, bias)
```

rewrite 步骤：

1. 创建 fused op。
2. 设置 fused op inputs。
3. 设置 attrs：原 MatMul attrs、bias axis、activation kind。
4. 创建 fused output value。
5. 将旧 `Relu` output 的 uses 重定向到 fused output。
6. 标记旧节点可删除。
7. 运行 local DCE 删除无 use 节点。

注意：如果 `t0/t1` 是 graph output 或有其他 user，不能简单删除旧节点。

## Metadata 更新

fused output metadata 通常应该继承被替换子图的最终输出，而不是中间 producer。

```text
fused.output.metadata = old_final_output.metadata
```

但 attrs 和内部 metadata 仍需要记录：

- matmul input format。
- bias shape 和 broadcast axis。
- activation type。
- accumulation dtype。
- output dtype/format。

后续 lowering 依赖这些信息生成正确 kernel。

## DCE 和 CSE 配合

Fusion pass 后通常要接 DCE：

```text
old nodes become unused
  -> delete dead nodes
```

有时还需要 CSE：

```text
fusion creates duplicated constants / casts
  -> CSE removes duplicates
```

Pass pipeline 常见顺序：

```text
canonicalization
  -> shape/layout inference
  -> fusion
  -> DCE
  -> metadata verify
  -> lowering
```

## 验证

Fusion pass 测试至少覆盖两类。

Graph 结构测试：

- 期望 fused op 出现。
- 旧 op 被删除。
- graph output 指向新 value。
- 多 consumer 时旧 producer 是否保留。
- metadata 是否存在。

数值正确性测试：

- 原 graph 和 fused graph 输出一致。
- 覆盖 broadcast、dynamic shape、dtype/cast、layout。
- 对浮点设置合理误差。

Debug dump：

```text
before fusion
after pattern match
after rewrite
after dce
after metadata inference
```

融合 pass 的 bug 通常不是“没匹配到”，而是 rewrite 后某个 use、metadata 或 side-effect 顺序被破坏。
