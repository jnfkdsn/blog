---
order: 8
title: Relax 改写、调度修改与问题定位
updated: 2026-10-30
---

# Relax 改写、调度修改与问题定位

前面的章节解释了 TVM 已有机制。本章把其中一个决定交给我们自己实现：识别连续两次 ReLU，删除外层重复计算。任务很小，却会经过实际图匹配、替换、Pass接入、lowering和CPU执行，适合建立“怎样修改编译器”的完整经验。

先实现规则，再验证共享使用者和不匹配情况，最后把同样的调查方法用于主模型和调度。它不是把激活换成另一种功能；优化前后的程序应保持约定的结果。

## 1. 从重复计算确定改写目标

先看纯张量数据流：

```python
with R.dataflow():
    inner = R.nn.relu(x)
    result = R.nn.relu(inner)
    R.output(inner, result)
return inner, result
```

inner 的元素已经经过同一种 ReLU，再应用一次不会增加新的截断。期望输出变成：

```python
with R.dataflow():
    inner = R.nn.relu(x)
    R.output(inner)
return inner, inner
```

保留 inner 的两个逻辑使用，只消除外层计算。这里讨论 Relax 的纯张量值语义；如果导入源语言还要求两个返回对象可被独立原地修改，导入/调用边界必须另行保存这种可观察契约，不能直接把本规则当成任意 Python Tensor 对象的替换规则。

输入 `[-2,0,3,8]` 的两项结果都应为 `[0,0,3,8]`。但 `relu(negative(x))` 不能套这条规则：x=-2 时应该返回2，改成 relu(x) 会得到0。

## 2. DPL 描述的是匹配关系

Relax 的 Dataflow Pattern Language 可以表达这个局部结构：

```python
from tvm.relax.dpl import wildcard, is_op, rewrite_call

x = wildcard()
inner = is_op("relax.nn.relu")(x)
outer = is_op("relax.nn.relu")(inner)
```

x 接受任意输入表达式；inner 要求一个参数为 x 的 ReLU；outer 再要求消费 inner 的 ReLU。实际 Relax 往往先把 inner 绑定到变量，outer 使用该变量。匹配机制沿绑定关系认识这条数据流，不要求两个调用直接嵌在同一个文本表达式里。

模式表达候选形状，语义理由来自第一节。同名操作的契约是本规则成立的依据；不能只把匹配到的两个任意“激活操作”都视为重复。

## 3. 替换回调与 Pass 接入

下面是完整的规则实现，依赖已经导入的 tvm.relax：

```python
@relax.transform.function_pass(opt_level=0)
class RemoveRepeatedRelu:
    def transform_function(self, func, mod, ctx):
        x = wildcard()
        inner = is_op("relax.nn.relu")(x)
        outer = is_op("relax.nn.relu")(inner)

        def replace(call, matched):
            return matched[inner]

        return rewrite_call(outer, replace, func)
```

`matched` 连接模式对象与匹配到的表达式。回调返回的是已经匹配的内层结果，不是原始 x；后者可能仍有负数。rewrite_call 负责将替换应用到函数并维护绑定，function_pass 将此函数级变换接入模块Pass系统。

与手工编辑IR不同，回调不是自己删除图上所有看似相关的节点。inner 仍是输出或其他操作的输入时，它必须留下。实际改写结果只有一次 ReLU，函数返回 `(inner, inner)`，负例函数保持结构相同。

这与 MLIR 的 pattern/rewriter 有相近问题域，但接口协议不同。此处回调返回替换表达式，由Relax改写设施整理函数；不要把MLIR的replaceOp调用方式直接移植过来。

## 4. 让规则真正进入编译流程

规则定义本身不执行优化。把它应用到模块，再交给后续流程：

```python
rewritten = RemoveRepeatedRelu()(mod)
optimized = relax.get_pipeline("zero")(rewritten)
executable = tvm.compile(optimized, target="llvm")
vm = relax.VirtualMachine(executable, tvm.cpu())
```

应在 high-level ReLU 仍存在时执行该规则。如果先做LegalizeOps，候选可能已经是call_tir；原模式不再匹配，不代表回调写错，而是插入阶段不对。

验证时分别回答三件事：

| 证据 | 回答的问题 |
|---|---|
| 改写后的Relax只剩一次ReLU | 我们的规则是否真的生效 |
| 不匹配函数保持结构相同 | 是否扩大了规则适用范围 |
| 改写前后都编译运行、比较全部输出 | 图改写与后续实现是否一起保持结果 |

CPU参考使用混合正负、全负、全正及包含NaN/无穷/负零的输入，比较前后数值（NaN按对应位置比较）。这不是对所有浮点位模式、布局和源语言别名行为的形式化证明；支持范围应与规则和测试一起陈述。

## 5. 从参考实现到自己的修改

这一章提供完整参考规则，是为了让扩展链可见。独立练习不要求从空文件重建工程：可以在现有测试入口下增加一个有明确语义的模式，先写预期变换和拒绝条件，再实现回调。

例如尝试折叠两次相同、互为逆的转置，先限制秩和axes均已知，检查两次置换的复合确实为恒等，保留不满足条件的图。不能仅凭“有两次transpose”就删除；三维的两次相同置换未必回到原布局。需要另外考虑所选IR层的值语义与下游布局实现，不将练习结果冒充所有框架的view优化。

调度方向则继续使用已有独立骨架：为32×32 Matmul实现参数化列分块，比较tile4/8/7，并解释有效范围、初始化和k累加顺序。选做缓存扩展时，先预测producer的范围和次数，再检查实际IR；不以新增接口数量作为完成标准。

这两个任务分别练习图层语义改写与计算组织修改，不要求每学一节都写一次实验。

## 6. 发现问题时寻找第一个偏差

回到 `relu(x @ weight + bias)`。如果主例第二列由7.5变为8.5，可能是某个访问把 bias[j] 写成了 bias[i]。第一行第二列本来取-0.5，却误取+0.5。

先比较导入的Relax，再比较Legalize后的PrimFunc，最后检查调度和目标代码。若Relax仍表达正确广播，而某份TensorIR第一次出现bias[i]，就把调查收敛到产生这份访问的变换；无需同时怀疑VM和全部后端。

另一个现象是函数名含fused，但仍有两个alloc_buffer。一次Relax调用只能证明函数组合；TensorIR循环与buffer才能说明内部数据流；目标代码与测量再回答物理实现和实际成本。这些证据回答不同层次的问题。

对自己的改写也一样：规则没有触发，先查模式和插入阶段；触发但结果错，先查替换与使用者；只有编译后的结果错，再追后续阶段。保留最小复现输入，每次只改变一个条件。

## 7. 设计取舍与阶段结果

TVM提供两条本章已经实际走通的扩展方式：用Relax规则改写计算关系，用Schedule程序改写张量实现。前者依赖操作语义、绑定和结构信息，后者依赖计算块、读写和循环约束。编译器基础设施承担通用的遍历、维护与后端连接，我们需要对新决策的语义和适用条件负责。

本轮目标是能够解释并修改一个有限行为。通读所有源码、实现新后端和证明最优性能都不是它的前置。后续学习PyTorch时可以用相同重复ReLU问题比较FX修改与调用边界；进入Triton时再把重点转向kernel IR、布局和设备lowering。

1. 回调为什么返回matched[inner]而不是matched[x]？inner还有另一个使用者时为什么不能直接删除？
2. Legalize之后原规则不匹配，应先检查什么？只比较最终数值能否证明这条规则执行过？
3. 修改shape为其他值、增加另一种激活或改变源语言别名约定，哪些前提需要重新确认？

参考工程与前后IR来自 [design_probe](https://github.com/jnfkdsn/aicompiler/blob/main/tvm-e2e/design_probe.py)。精确协议可查 [DPL rewrite_call](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/dpl/rewrite.py) 与 [function_pass](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/transform/transform.py)；独立调度练习仍在 [TVM labs](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e/exercises)。
