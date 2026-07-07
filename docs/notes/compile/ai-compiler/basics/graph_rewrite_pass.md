---
order: 4
title: Graph Rewrite 与 Pass 基础
updated: 2026-07-05
tags: [ai-compiler, graph-rewrite, pass]
status: draft
---

# Graph Rewrite 与 Pass 基础

相关入口：[AI Compiler 基础](/notes/compile/ai-compiler/basics/)

Graph rewrite pass 对 graph IR 做语义保持变换。算子融合是 graph rewrite 的一种，但在 fusion 之前通常还有 decomposition、canonicalization、constant folding、DCE、CSE 等 pass。

## Pass 的基本结构

```text
input graph
  -> analyze
  -> match pattern
  -> check legality
  -> rewrite graph
  -> update metadata
  -> verify graph
```

Pass 的输出仍然是 graph IR，只是结构更适合后续优化或 lowering。

## 常见 Graph Rewrite

Decomposition：

```text
HighLevelOp -> smaller primitive ops
```

Canonicalization：

```text
统一等价写法，减少 pattern 数量
```

Constant folding：

```text
常量输入的子图直接计算
```

DCE：

```text
删除无 user 且无 side effect 的 op
```

CSE：

```text
合并等价 op，重定向 users
```

Fusion：

```text
合并 producer-consumer 子图
```

## Pattern Matching

Pattern 描述想匹配的 graph 形状。

```text
MatMul -> Add -> Relu
Add -> Relu
Transpose -> Op -> Transpose
```

Pattern matching 只回答“形状像不像”，不回答“能不能改”。能不能改要靠 legality。

## Rewrite

Rewrite 是修改 graph：

- 创建新 node。
- 创建新 output value。
- 把旧 value 的 users 指向新 value。
- 删除 dead node。
- 更新 graph output。
- 更新 metadata。

核心风险：

- use-def 断裂。
- graph output 指向已删除 value。
- side-effect 顺序被破坏。
- metadata 丢失。

## Analysis Invalidation

Pass 改写 graph 后，一些 analysis 结果会失效。

可能失效：

- shape/layout metadata。
- liveness / use count。
- alias 信息。
- topological order。
- cost model cache。

Pass manager 要知道哪些结果需要重新计算。简单实现可以在每次 rewrite 后重新 run verifier 和 metadata inference；大型系统会做更细粒度的 invalidation。

## Pass 测试

Graph rewrite pass 测试通常包含：

- graph 结构测试：某些 op 是否出现或消失。
- metadata 测试：shape/dtype/layout 是否保持。
- 数值测试：rewrite 前后输出是否一致。
- negative case：不应该 rewrite 的图是否保持不变。

对 fusion pass，negative case 很重要，因为很多 bug 来自“融合了不该融合的子图”。
