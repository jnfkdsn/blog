---
order: 8
title: AOTAutograd 与前向反向图的协作
updated: 2026-10-17
---

# AOTAutograd 与前向反向图的协作

推理只要求从输入得到输出。训练还要根据损失计算参数梯度，因此前向执行结束时，一些原本可以释放的值可能必须留下来。编译器需要处理的不再只是一个前向图，还包括反向图，以及两者之间的数据协议。

本章在原来的矩阵模型上增加平方均值损失。先手工推导反向需要哪些量，再观察 AOTAutograd 实际交付的两个图，就能理解它为什么这样划分输入和输出。

## 1. 从推理模型到训练目标

令：

```text
Z = X @ W + b
Y = relu(Z)
L = sum(i,j, Y[i,j]^2) / (B*N)
```

代码只增加一个损失：

```python
def loss(x, weight, bias):
    y = torch.relu(x @ weight + bias)
    return y.square().mean()
```

这里 x 为 `[B,K]`，weight 为 `[K,N]`，bias 为 `[N]`，L 是标量。训练需要得到与 x、weight、bias 对应形状的梯度。

假定反向从 `dL/dL=1` 开始，对 Y 有：

```text
G_Y = 2 * Y / (B*N)
```

再经过 ReLU，非正输入处采用 PyTorch ReLU 在该点的反向约定，得到：

```text
G_Z = G_Y * indicator(Z > 0)   # 逐元素相乘
```

继续传播到矩阵乘法与广播加法：

```text
G_X = G_Z @ transpose(W)
G_W = transpose(X) @ G_Z
G_b = sum_i(G_Z[i,:])
```

这几步已经暴露了反向的输入需求：它需要 X、W，以及足以恢复激活值和 ReLU 掩码的信息。对于本例有限普通输入，可以使用 Y 本身判断正值区域，并计算 `2Y/(BN)`。

## 2. 前向结果与保存值

用户调用 `loss` 只需要返回标量 L，但编译后的前向内部可能返回更多内容。固定实验捕获到的前向图末尾是：

```python
mm = torch.ops.aten.mm.default(x, weight)
add = torch.ops.aten.add.Tensor(mm, bias)
relu = torch.ops.aten.relu.default(add)
square = torch.ops.aten.pow.Tensor_Scalar(relu, 2)
mean = torch.ops.aten.mean.default(square)
return (mean, x, weight, relu)
```

这是实测输出的参数重命名版。第一项是用户结果，后三项供反向使用。它们通过编译框架的包装保存，用户不会突然从 `loss(...)` 得到一个四元组。

从公式能解释为何保存这些量：X 用于计算 weight 梯度，weight 用于计算输入梯度，relu 用于损失导数和激活反向。bias 的具体数值没有直接出现在这些梯度公式里，所以这份前向不必为了反向单独保存 bias。

保存值可能是原输入引用，也可能是新分配中间量。它们的生命周期延长到反向消费完为止。这会影响峰值内存，也限制了前向阶段哪些存储可以提前释放或覆盖。

## 3. 反向图的输入输出协议

对应的反向图接收保存值以及输出的上游梯度。本例返回标量损失，调用 `.backward()` 时上游梯度默认为 1。更一般的输出可能接收一个梯度 Tensor。

将实际反向图中的 detach、形状包装等辅助节点省略，可整理为：

```text
输入：X、W、Y，以及上游标量梯度 g
GY = expand(g, [B,N]) / (B*N) * (2*Y)
GZ = ReLUBackward(GY, Y)
Gb = sum(GZ, dim=0)
GW = transpose(X) @ GZ
GX = GZ @ transpose(W)
返回：GX、GW、Gb
```

这里每个箭头都来自前面的链式法则。反向不是“把前向代码倒着执行”，而是构造新的计算，其中也有矩阵乘法、归约、广播和逐元素操作。

这些新操作同样需要后端融合、布局和代码生成。因此编译器既可以优化前向，也可以优化反向；前向/反向之间还需要维持保存值与梯度的契约。

## 4. AOTAutograd 的位置

AOTAutograd 对待编译函数进行自动微分相关追踪与图组织，处理函数式化、前向/反向关系和分区，再把相应图交给编译回调。

配套实验为了观察边界，分别安装前向和反向 compiler，将收到的 GraphModule 保存后返回可执行的 FX 包装。这个实验展示了两份图和调用协议，随后另用真正的 Inductor 编译同一个 loss，并执行 `.backward()`。

这两份证据回答不同问题：观察回调让图的接口可见，Inductor 运行验证实际生成实现的梯度。不能仅打印出反向图就宣称目标代码正确。

本机固定输入 `[4,7] @ [7,3]` 的实验得到损失约 `0.247959`，输入、weight、bias 三项梯度均通过与 eager 的数值比较。当前样本记录的最大绝对差为零，但这不是所有算子、精度和调度下逐位相同的保证；验证使用明确的浮点容差。

## 5. 保存与重计算的取舍

保存 Y 可以直接供反向使用，但会延长其存活期。另一种选择是在反向重新计算部分前向值，以换取更少的保存内存。

例如，若某个中间值只是廉价逐元素操作的结果，重算可能划算；若需要重新执行大型矩阵乘法，节省的内存是否值得计算成本就要重新评估。

前向/反向分区因此不只是画一条“用户输出之前/之后”的线。分区要选择哪些值跨边界保存，以及哪些计算放在反向重新执行。不同策略会影响：

- 前向末尾保留哪些数据，以及峰值内存；
- 反向多执行多少计算与访存；
- 融合机会、内核边界和数值行为。

对随机操作、可变状态、别名或副作用，重算还需要额外语义保证。不能把任意有相同数学表达式的节点都复制到反向。

本例展示的是默认捕获结果和对应梯度；没有用该小模型测量不同分区策略的峰值内存，也不据此宣称某种重算策略最优。

## 6. 训练循环中的其他边界

一次实际训练通常还包含清空梯度、参数更新、随机数、混合精度和分布式通信。它们并不会因为前向成功 `compile` 就自动成为同一份已验证图的一部分。

应先明确编译范围：只编译模型前向并让框架组织反向，还是编译包含优化器更新的步骤？哪些参数在调用间发生修改，哪些状态影响捕获条件？

这一轮先验证标量损失与反向梯度。若之后研究训练编译，再继续加入优化器状态、动态 loss scaling、checkpoint、梯度累积和通信，逐项解释状态及执行边界。分布式训练不能仅从单机反向图推导其完整性能。

## 7. 推演与实践入口

1. 用户只需要 loss，为什么前向内部仍然返回 X、W、Y？
2. bias 梯度为什么沿 batch 维求和，而不是做一次矩阵乘法？
3. 如果 weight 不需要梯度，哪些反向计算或保存需求可能改变？
4. 为什么“反向图已生成”和“反向目标代码已正确执行”需要分别验证？

实验见 [labs 的 training](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。有限源码可从 [aot_autograd.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/aot_autograd.py) 的编译边界进入，按需读 [分区实现](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/partitioners.py) 与 [runtime wrappers](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_functorch/_aot_autograd/runtime_wrappers.py)。
