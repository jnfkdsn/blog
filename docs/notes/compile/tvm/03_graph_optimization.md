---
order: 3
title: 操作实现转换与图融合
updated: 2026-10-28
---

# 操作实现转换与图融合

Relax 已经记录 `matmul → add → relu`，也知道每个结果的形状。现在要把模型交给 CPU：矩阵乘法的循环从哪里来？三个操作为什么被选在一组？放到一组以后，中间张量又在哪里？

本章沿同一个模型回答这三个问题：先生成实现，再根据依赖选择分组，最后连接组内函数与 buffer。主线是“编译器取得什么信息，依据什么作决定，决定后程序如何变化”。

## 1. 操作语义与具体实现

输入仍为 `x[2,3]`、`weight[3,2]`、`bias[2]`，都是 float32：

```python
mm = R.matmul(x, weight)
added = R.add(mm, bias)
result = R.nn.relu(added)
```

`mm` 的形状为 `[2,2]`。矩阵乘法语义已经确定，但这里没有循环，也没有选择库函数。本章先采用生成 TensorIR 的路径；外部实现选择在第六章展开。

`LegalizeOps` 遍历 Relax 调用，查询操作已注册的转换规则。矩阵乘法规则的结构摘录如下，计算定义和检查暂时省略：

```python
@register_legalize("relax.matmul")
def _matmul(bb, call):
    # 此处定义 te_matmul，描述输出元素怎样由输入计算。
    # 此处还检查生成实现需要的 dtype 等信息。
    ...
    return bb.call_te(
        te_matmul, call.args[0], call.args[1], primfunc_name_hint="matmul"
    )
```

`call` 是正在处理的调用对象，两个参数是 IR 中的 x 和 weight。规则执行时不计算输入矩阵的数值，而是生成以后执行矩阵乘法的程序。

`te_matmul` 使用张量表达式描述计算。对本例，其核心意思为：

```text
输出范围：0 <= i < 2，0 <= j < 2
归约范围：0 <= k < 3
输出元素：sum_k(x[i,k] * weight[k,j])
```

`bb.call_te` 将这种描述转换成 TensorIR PrimFunc，加入当前模块，并返回相应的 Relax 调用。因此一次转换同时提供了具体实现和引用它的调用。TE 的通用构造接口可以按需查阅，当前需要理解它承担“描述元素计算以生成实现”的作用。

## 2. 转换后的模块与调用关系

矩阵乘法调用变为：

```python
mm = R.call_tir(
    Module.matmul, (x, weight),
    out_sinfo=R.Tensor((2, 2), "float32"),
)
```

`Module.matmul` 指向新增的 TensorIR 函数，`(x, weight)` 是输入，`out_sinfo` 描述结果结构。输出存储的分配还会在后续阶段落实。

加法和 ReLU 也取得实现后，模块关系是：

```text
main（Relax）：
    mm     = call_tir matmul(x, weight)
    added  = call_tir add(mm, bias)
    result = call_tir relu(added)

matmul（TensorIR）：遍历输出位置，沿内积维累加
add   （TensorIR）：读取 mm[i,j]、bias[j]，写出 added[i,j]
relu  （TensorIR）：读取 added[i,j]，写出 max(added[i,j], 0)
```

main 仍组织张量数据流，各个 PrimFunc 描述具体计算。得到正确的实现还没有回答怎样分块、缓存或向量化；这些决定需要继续在具体实现上进行。

## 3. 融合候选的计算模式

将相关计算放进一个低层函数，能为后续调整计算位置提供空间。但操作可能有其他消费者，实现也可能不适合某种组合，不能仅凭代码行相邻就分组。

`AnnotateTIROpPattern` 分析 TensorIR，为实现附上模式分类。本例实际输出为：

| 实现 | `op_pattern` | 供分组使用的性质 |
|---|---:|---|
| matmul | 4 | 可与输出侧逐元素处理组合的一类计算，`kOutEWiseFusable` |
| add | 0 | `kElemWise`；输出与矩阵输入位置对应，bias 被广播读取 |
| relu | 0 | `kElemWise`；每个输出消费对应位置的输入 |

数字是此版本的分类编码，无需背诵。虽然 add 的 bias 有广播，本例实际分类仍是 0，不能仅根据算子名字猜类别。

Relax 原来已经知道操作语义、形状和类型；现在又取得了实现层的访问和计算模式。后续策略使用这些模式筛选组合，而不是在这里运行所有候选来测量谁最快。

## 4. 使用路径与分组边界

主例只有一条链：

```text
matmul → add → relu → 区域输出
```

算法沿使用关系寻找下游候选，并检查连接路径上的模式与已有分组。本例符合规则，最终将三个计算合为一组。

更多操作可能形成分支。下面是解释算法的示意图，scale 表示逐元素缩放：

```text
             ┌→ relu ───┐
matmul ──────┤          ├→ add → 输出
             └→ scale ──┘
```

只沿 relu 路径考虑融合，会漏掉 scale 仍需要 matmul 结果这件事。算法先寻找从当前节点出发的所有使用路径必经的下游节点；这里两条路径在 add 汇合。这个关系称为后支配关系，它帮助确定要检查的候选范围。

找到汇合点以后，还要检查两条路径上的计算模式和边界。若其中有不支持通用融合的 opaque 调用，就不能因为路径最终汇合而直接合并。该图说明候选选择过程，不承诺任意同形图都得到相同分组。

固定实现的主要过程是：

```text
建立生产—使用图，标记区域外使用
    ↓
计算后支配关系，寻找候选下游节点
    ↓
CheckPath：检查路径与已有组是否符合模式规则
    ↓
检查分组规模等限制
    ↓
CommitFuse：将通过检查的节点归入同一组
```

实现用并查集维护组，并分阶段应用不同模式的规则。理解本例可以先不展开树构造和并查集的内部代码，但不能省略“为什么检查两条路径”这个决策依据。

若模型还要返回 mm，它就必须离开当前区域，不能继续视作只有组内消费者的临时值。具体分组需要重新计算。数学上能够设计多输出融合，不保证当前通用算法一定采用它；合法性、策略选择和实际收益是不同问题。

## 5. Relax 分组与 TensorIR 函数组合

`FuseOps` 将选定调用提取到带 Primitive 属性的 Relax 函数。主例此时的实际结构为：

```text
main：调用 group(x, weight, bias)

group（Relax，实际名称为 fused_matmul_add_relu）：
    call_tir matmul
    call_tir add
    call_tir relu

模块中仍有三个独立的 TensorIR PrimFunc
```

分组函数表达共同处理的边界。随后 `FuseTIR` 把组内实现组合起来：将 matmul 输出接到 add 输入，将 add 输出接到 relu 输入，同时整合参数与局部存储。

组合后的数值意图如下，是省略轴声明的伪代码：

```text
fused_matmul_add_relu(x, weight, bias, out)：
    准备 M[2,2] 和 A[2,2]
    第一组循环：M[i,j] = sum_k(x[i,k] * weight[k,j])
    第二组循环：A[i,j] = M[i,j] + bias[j]
    第三组循环：out[i,j] = max(A[i,j], 0)
```

main 现在通过一次 call_tir 调用这个 PrimFunc。实际生成 IR 仍有两个 `T.alloc_buffer` 和三组循环。它们进入了共同的优化范围，但中间矩阵仍然被写入和读取。

## 6. 循环调整与中间存储

进一步考虑每算完一个输出元素，就完成它的后处理：

```text
对每个 i,j：
    acc = 0
    对每个 k：acc += x[i,k] * weight[k,j]
    out[i,j] = max(acc + bias[j], 0)
```

这种组织不要求完整的 M、A 中间矩阵。它省去了显式中间矩阵及相应读写，局部值最终是否全部留在寄存器，还取决于后端和资源条件。这是进一步的候选方案，不是上节输出已经完成的变换。

后处理必须等待对应归约完成。若两项乘积为 -2、3，bias 为零，正确计算得到 `max(-2+3,0)=1`。若每次累加后都做 ReLU，第一次将 -2 改成 0，第二次得到 3，结果就改变了。

因此观察优化时要分别回答：调用是否合并、循环如何组织、中间数据是否仍然读写。复用一个 buffer 只说明空间可重用，不自动消除该 buffer 上的 store/load。

## 7. 实现入口与推演

本例经历“生成实现 → 决定分组 → 连接实现”。下一章展开 PrimFunc 内部的循环、计算块与归约约束，为进一步调整执行组织做准备。

1. LegalizeOps 是否需要 x 的真实元素？它生成了哪两项互相引用的内容？
2. 菱形图中，为什么只检查 relu 分支不足以决定 matmul 的分组？若 scale 是 opaque 调用，会影响哪一步？
3. 合成一个 PrimFunc 后仍有 M、A 两块存储，能否说中间访存已经消除？改变什么计算组织才有机会省掉它们？

规则与调用链可查 [matmul legalization](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/transform/legalize_ops/linear_algebra.py)、[BlockBuilder.call_te](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/block_builder.py)、[LegalizeOps](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/legalize_ops.cc)。分组对应 [FuseOps](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/fuse_ops.cc) 与 [GraphPartitioner](https://github.com/apache/tvm/blob/v0.20.0/src/relax/analysis/graph_partitioner.cc)，组合对应 [FuseTIR](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/fuse_tir.cc)。[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)保存主例的实际阶段输出；分支图是本文算法推演，未冒充已运行的分支实验。
