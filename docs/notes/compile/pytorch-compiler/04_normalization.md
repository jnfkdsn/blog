---
order: 4
title: 图规范化、原地修改与算子分解
updated: 2026-10-17
---

# 图规范化、原地修改与算子分解

捕获图以后，计算已经从 Python 控制逻辑中分离出来，但后端面对的表示仍然可能很复杂。同一种计算可以通过许多 API 写出；张量还可能被原地修改，多个变量可能引用同一块存储。如果直接对节点做重排和融合，就容易遗漏这些关系。

图规范化的目的，是将这些行为整理成后续优化可以一致处理的形式，同时保留程序的可观察语义。本章先用一次原地更新建立这个过程，再讨论算子分解如何缩小后端需要处理的语义集合。

## 1. 原地修改中的隐藏数据依赖

考虑给特征加 1，再做 ReLU：

```python
def local_update(x):
    y = x.clone()
    y.add_(1)
    return y.relu()
```

输入为 `[-2,0,3]` 时，输出应为 `[0,1,4]`，外部传进来的 x 保持不变。

若只把三次调用顺序排列出来，很容易看到 `clone → add_ → relu`。困难在于 `add_` 的结果不仅通过返回值传递，它还改变了 y 指向的存储。随后对 y 的读取必须看到更新后的内容。

这个关系在加入别名以后更明显：

```python
y = x.clone()
view = y.view(-1)
y.add_(1)
return view
```

`view` 虽然在 `add_` 之前创建，返回时也应观察到加 1 的结果。若编译器只追“变量 view 是在哪一行创建的”，却没有处理它与 y 的存储关系，就可能返回旧值。

因此后端需要的不是更漂亮的 API 名字，而是一份足以揭示读写先后、别名和输出关系的表示。

## 2. Functionalization 与显式值变化

对于第一个局部更新例子，可以把原地更新改写为返回新值的计算：

```text
原来：y = clone(x); mutate_add(y, 1); result = relu(y)
改写：y0 = clone(x); y1 = add(y0, 1); result = relu(y1)
```

改写后，`y0` 与 `y1` 代表不同的逻辑值，ReLU 明确消费更新后的 `y1`。这种把可变张量行为整理成函数式数据流的过程称为 **functionalization**。

配套实验用 `make_fx` 观察独立的 `torch.func.functionalize`。原图核心是：

```python
clone = torch.ops.aten.clone.default(x)
add_ = torch.ops.aten.add_.Tensor(clone, 1)
relu = torch.ops.aten.relu.default(add_)
```

处理后的核心变为：

```python
clone = torch.ops.aten.clone.default(x)
add = torch.ops.aten.add.Tensor(clone, 1)
relu = torch.ops.aten.relu.default(add)
```

这里发生的实质变化是：加法用返回的新逻辑值表达更新，后续使用也随之转向新值。数值验证得到 `[0,1,4]`。

这并不要求最终机器程序为每个逻辑值分配独立存储。后端明确了依赖和别名约束以后，仍可以重新做合法的存储复用。**先把语义关系表达清楚，再决定物理存储怎样复用**，比从含糊的状态修改中猜测依赖可靠。

## 3. 输入修改与调用边界补偿

如果去掉 clone，情况就不同了：

```python
def input_update(x):
    x.add_(1)
    return x * 2
```

调用者既能看到返回值，也能在调用后读取被修改的 x。若只生成一个不修改输入的新张量，虽然返回结果相同，程序整体语义仍然改变了。

独立 functionalization 观察中的核心处理是：

```python
updated = torch.ops.aten.add.Tensor(x, 1)
result = torch.ops.aten.mul.Tensor(updated, 2)
torch.ops.aten.copy_.default(x, updated)
return result
```

内部计算使用清楚的值关系，边界处的 copy 则把应当被外部观察到的更新落实回原输入。本实验既比较返回值，也比较调用后的 x，防止只测返回结果漏掉副作用。

在实际 AOT 编译处理中，图与 wrapper 可以分担输入更新、别名恢复等职责，不必恰好以这份独立实验的代码组织。重要的是契约：哪些输出是用户结果，哪些用于恢复输入修改，哪些与输入共享存储，都必须传递给消费者。

这也解释了为什么不能简单删除所有 `copy_`：它有时是可消除的内部搬运，有时承担程序接口的可观察更新。判断依据是数据关系和语义，不能只根据操作名称。

## 4. 算子分解与后端语义集合

另一个复杂性来自算子种类。Python 接口、复合数学函数和历史重载可能很多，如果后端为每种写法分别实现 lowering，会重复处理相同计算。

可以将某些复合算子展开成一组已有基础操作，这称为 **decomposition**。例如本系列后面会观察 Softmax。稳定的数值形式可以推导为：

```text
m = max_j(x[j])
e[j] = exp(x[j] - m)
y[j] = e[j] / sum_k(e[k])
```

展开后得到 max reduction、减法、exp、sum reduction、除法；后端可以复用已有的归约和逐元素实现，也可以看到它们之间的融合机会。

但“数学上相等”还不够。若写成 `exp(x)/sum(exp(x))`，输入 `[1000,1001,999]` 就可能溢出。对于低精度输入，还要决定累加和中间计算的 dtype。真实分解必须遵守所支持的数值语义，不能只套代数公式。

同时，保留某个高层算子也可能更有利：专用库或后端模式可能正好支持它。分解表是后端选择的一部分，并不存在所有系统都必须先展开成同一套最小操作的规则。

## 5. 规范化、图优化与实现选择

这几种变换经常相邻出现，但解决的问题不同。沿着主例可以区分：

```text
用户写法：x @ weight + bias，再 ReLU
    ↓ 整理调用和语义
ATen 图：mm → add → relu
    ↓ 合适的图变换与实现选择
可能采用：addmm 库调用 → 生成的 ReLU
```

将不同表面写法整理到一致的操作语义，帮助后端少处理重复情况；把 `mm + add` 合成 `addmm` 则利用了两项计算间的关系。最后选择库调用还是生成代码，又取决于具体目标和实现代价。

这些工作不是严格“一次性做完就再不回来”的流水线。某次分解可能暴露新的融合机会，某次实现选择又可能要求布局调整。理解它们时，应关注每次变换新增了什么信息、改变了什么表示，以及消费者接下来怎样使用结果。

## 6. AOT 处理与推理路径

在 `torch.compile(..., backend="inductor")` 的固定版本实现中，`compile_fx` 会组织 AOTAutograd 相关处理，并设置前向、反向或推理编译入口。虽然名字中带有 Autograd，它涉及的图整理、函数式化和调用契约也能服务于推理路径。

因此推理不是简单地“Dynamo 图原样交给一个 C++ 生成器”。中间需要把复杂调用、状态和别名关系整理到后端能处理的表示。

训练还会构造前向/反向关系以及保存中间值的协议，放到[训练编译](/notes/compile/pytorch-compiler/08_training)中展开。本章先明确：**AOTAutograd 中的 AOT 与将模型导出成部署包的 AOT，不是同一项任务。** 前者可以发生在用户第一次调用 `torch.compile` 时。

## 7. 推演与实践入口

1. 局部 clone 后的 `add_` 改成普通 add，为什么可以保留返回语义？
2. 对输入的原地更新，为什么只比较返回 Tensor 不足以验证转换？
3. 将一个操作分解成更多图节点，为什么反而可能更便于优化？
4. 已经函数式化的图，最终还能不能使用原地写入的机器实现？需要什么依据？

实验见 [labs 的 normalize](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。语义契约见 [functionalize 契约与实现](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/eager_transforms.py)；编译交接见 [compile_fx](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/compile_fx.py) 和 [AOT runtime wrappers](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/_aot_autograd/runtime_wrappers.py)。独立实验用于解释机制，不声称覆盖所有 view、mutation 和别名组合。
