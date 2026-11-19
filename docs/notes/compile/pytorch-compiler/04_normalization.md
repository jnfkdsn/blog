---
order: 4
title: AOT 编译交接、函数式化与算子分解
updated: 2026-10-30
---

# AOT 编译交接、函数式化与算子分解

Dynamo交出了FX图，但它仍可能包含原地修改、别名和复杂算子。Inductor若直接把每个节点展开成循环，就要同时处理这些状态语义和所有表面写法。PyTorch在这两个阶段之间安排AOT相关处理，把后端需要计算的图与调用者应看到的行为接起来。

本章先定位这项交接，再沿一次输入更新观察：哪些状态被变成显式数据流，哪些工作需要在边界恢复，最后哪些算子被交给Inductor。训练的梯度推导与保存值细节留到第八章。

## 1. Dynamo 后端入口与 AOT 分派

固定2.7.1的主要交接可以整理为：

```text
Dynamo：FX GraphModule + 示例输入/元数据
    ↓ 调用Inductor后端入口 compile_fx
AOT相关处理：分析mutation/alias、整理图与调用协议
    ├─ 推理：将推理图交给配置的编译回调
    └─ 需要autograd：组织前向/反向图、保存值和编译回调
    ↓
Inductor核心：lowering → 调度 → 生成实现
    ↓
包装编译结果，恢复用户调用约定
```

compile_fx是整条后端的入口，不等于所有输入已经成为Inductor内部IR。源码中它配置前向、推理和反向编译入口，再调用AOT相关设施。`aot_autograd.py`根据需求选择`aot_dispatch_base`或`aot_dispatch_autograd`等处理路径。这里用“AOT分派”描述这项工作，不依赖较新资料中的名称，把它误当成本版本额外串联的一层。

AOTAutograd中的AOT指相关图工作在对应执行前组织；它可以发生于第一次torch.compile调用内部。推理也需要处理mutation和别名，不因没有backward就能跳过这些契约。Export/AOTInductor的离线部署是第七章的另一种入口。

接下来用可直接观察的functionalization说明其中一项机制。该独立观察不是整个AOT管线的逐pass记录；实际实现可以将补偿工作分配到图和wrapper的不同位置。

## 2. 原地修改中的隐藏数据依赖

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

把第二个例子的存储状态实际展开，能看见变量创建顺序不足以解释结果：

| 步骤 | y 的内容 | view 读到的内容 |
|---|---|---|
| clone 后 | `[-2,0,3]` | 尚未创建 |
| view 创建后 | `[-2,0,3]` | `[-2,0,3]` |
| y.add_(1) 后 | `[-1,1,4]` | `[-1,1,4]` |

view 与 y 指向同一底层存储。转换若只把 add_ 改成返回新张量，却仍让 view 引用旧逻辑值，就会错误返回 `[-2,0,3]`。完整的函数式化需要跟踪别名，并将后续读取连接到更新后的状态；这也是它超过“替换一个操作名字”的工作。

因此后端需要一份足以揭示读写先后、别名和输出关系的表示。

## 3. Functionalization 与显式值变化

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

## 4. 输入修改与调用边界补偿

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

代入输入 `x=[-2,0,3]`：updated 是 `[-1,1,4]`，返回值是 `[-2,2,8]`，调用结束后的原输入也必须变成 `[-1,1,4]`。若删除最后的 copy，返回值仍然对，但调用方重新读取 x 时得到旧值，程序行为已经不同。

内部计算使用清楚的值关系，边界处的 copy 则把应当被外部观察到的更新落实回原输入。本实验既比较返回值，也比较调用后的 x，防止只测返回结果漏掉副作用。

在实际 AOT 编译处理中，图与 wrapper 可以分担输入更新、别名恢复等职责，不必恰好以这份独立实验的代码组织。重要的是契约：哪些输出是用户结果，哪些用于恢复输入修改，哪些与输入共享存储，都必须传递给消费者。

这也解释了为什么不能简单删除所有 `copy_`：它有时是可消除的内部搬运，有时承担程序接口的可观察更新。判断依据是数据关系和语义，不能只根据操作名称。

## 5. 算子分解与后端语义集合

另一个复杂性来自算子种类。Python 接口、复合数学函数和历史重载可能很多，如果后端为每种写法分别实现 lowering，会重复处理相同计算。

可以将某些复合算子展开成一组已有基础操作，这称为 **decomposition**。例如本系列后面会观察 Softmax。稳定的数值形式可以推导为：

```text
m = max_j(x[j])
e[j] = exp(x[j] - m)
y[j] = e[j] / sum_k(e[k])
```

展开后得到 max reduction、减法、exp、sum reduction、除法；后端可以复用已有的归约和逐元素实现，也可以看到它们之间的融合机会。

以 `[1000,1001,999]` 为例，先减最大值后得到 `[-1,0,-2]`，指数约为 `[0.3679,1,0.1353]`，行和约为1.5032，输出约为 `[0.2447,0.6652,0.0900]`。这些中间值既说明分解为何能执行，也给后端的归约和逐元素阶段一个可追踪的数据依赖。

这里的分解只生成更具体的计算图，尚未决定是否保存整行指数。后续可以融合或重算部分结果，但必须保留“分母是完整行和”等语义关系。

但“数学上相等”还不够。若写成 `exp(x)/sum(exp(x))`，输入 `[1000,1001,999]` 就可能溢出。对于低精度输入，还要决定累加和中间计算的 dtype。真实分解必须遵守所支持的数值语义，不能只套代数公式。

同时，保留某个高层算子也可能更有利：专用库或后端模式可能正好支持它。分解表是后端选择的一部分，并不存在所有系统都必须先展开成同一套最小操作的规则。

## 6. 规范化、图优化与实现选择

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

## 7. 规范化的收益与代价

函数式化让隐式更新成为显式依赖，分解让后端复用较小的操作集合。这两项选择降低后端需要同时处理的复杂性，却也可能引入临时结果、边界copy或丢失高层模式。后续Inductor必须重新考虑物化、融合和存储复用，而不是把函数式图中的每个值都分配一块独立内存。

例如本章的输入更新必须保留，但中间updated是否单独分配，需要结合别名和使用关系决定。Softmax必须保持稳定数值语义，但指数结果可以物化，也可能按具体实现重算。图规范化提供清楚的计算契约，没有直接选定唯一的高性能实现。

要调查一次规范化，先确认Dynamo交出了什么图，再看AOT交给后端的图及返回协议。出现copy时要先问它保护哪个可观察状态，而不是直接删除。只比较输出数值、遗漏输入修改和别名，是这类优化最常见的验证缺口。

## 8. 推演与实践入口

1. view 在 add_ 之前创建，为什么仍要看到更新值？仅将 add_ 改名为 add、保持旧引用是否足够？
2. 输入 [-2,0,3] 先原地加1再乘2，为什么同时比较返回值和调用后的输入才能验证转换？
3. 推理为何也可能经过AOT相关处理？分解和函数式化让后端更容易处理什么，又将哪些选择留给Inductor？

实验见 [labs 的 normalize](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。语义契约见 [functionalize 契约与实现](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/eager_transforms.py)；编译交接见 [compile_fx](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/compile_fx.py) 和 [AOT runtime wrappers](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/_aot_autograd/runtime_wrappers.py)。独立实验用于解释机制，不声称覆盖所有 view、mutation 和别名组合。
