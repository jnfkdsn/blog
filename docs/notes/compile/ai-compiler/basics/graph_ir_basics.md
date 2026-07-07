---
order: 2
title: Graph IR 基础
updated: 2026-07-05
tags: [ai-compiler, graph-ir, ir]
status: draft
---

# Graph IR 基础

相关入口：[AI Compiler 基础](/notes/compile/ai-compiler/basics/)

Graph IR 用节点和边表示 tensor program。节点通常是 op，边通常是 tensor value 的定义和使用关系。

## 基本结构

```text
Graph
  inputs
  nodes
  outputs

Node
  op_type
  inputs
  outputs
  attrs
  metadata

Value
  producer
  users
  metadata
```

例子：

```text
x, w, b
  -> MatMul(x, w) -> t0
  -> Add(t0, b)   -> t1
  -> Relu(t1)     -> y
```

Graph IR 需要表达：

- op 类型。
- tensor value 的 producer。
- tensor value 的 users。
- op attrs，例如 transpose、axis、epsilon。
- tensor metadata，例如 shape、dtype、layout。
- side effect，例如 in-place、random、IO、sync。

## Graph IR 和传统 IR 的差异

传统三地址 IR 更像标量指令序列：

```text
%0 = add %a, %b
%1 = mul %0, %c
```

Graph IR 更像 tensor op DAG：

```text
t0 = MatMul(A, B)
t1 = Add(t0, bias)
y  = Relu(t1)
```

差异：

- value 通常是 tensor，不是标量寄存器。
- op 粒度更大。
- shape/dtype/layout 是一等信息。
- control flow 可能被弱化或独立表示。
- side effect 和 alias 主要围绕 tensor buffer。

## Use-Def

use-def 是 graph rewrite 的基础。

```text
def: 哪个 node 产生了这个 value
use: 哪些 node 使用了这个 value
```

融合、DCE、CSE 都需要 use-def：

- Fusion：找 producer-consumer。
- DCE：删除没有 users 且不是 graph output 的 node。
- CSE：把等价 node 的 users 重定向到已有 value。
- Layout rewrite：替换 transform op 的 uses。

## Op Schema

Op schema 描述 op 的合法输入输出。

```text
Add:
  inputs: Tensor, Tensor
  attrs: broadcast rule
  output: Tensor

ReduceSum:
  inputs: Tensor
  attrs: axis, keepdim
  output: Tensor
```

schema 用途：

- 校验 graph。
- 做 shape/dtype inference。
- 判断 decomposition。
- 判断 fusion legality。
- 指导 lowering。

没有 schema 的 graph rewrite 很容易变成字符串匹配，后续维护困难。

## Side Effect

大多数 tensor op 是 pure：相同输入得到相同输出，不修改外部状态。

非 pure op：

- in-place update。
- random。
- IO。
- state update。
- device synchronization。

Graph rewrite 不能随意重排 side-effect op。融合 pass 如果跨越 side-effect op，需要证明重排不改变语义；多数情况下直接禁止。

## Graph Verifier

Graph IR 需要 verifier 检查基本不变量：

- 所有 input value 有定义。
- 所有 graph output 有定义。
- producer 在 consumer 前，或 graph 允许无序并单独拓扑排序。
- op 输入数量和 dtype/shape 满足 schema。
- metadata 不缺失。
- side-effect op 顺序被表示。
- graph 没有悬空 use。

Verifier 是 graph rewrite pass 的安全网。
