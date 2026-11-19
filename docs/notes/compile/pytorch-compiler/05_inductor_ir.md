---
order: 5
title: Inductor 的表达式、存储与 Scheduler 融合
updated: 2026-10-30
---

# Inductor 的表达式、存储与 Scheduler 融合

规范化的图已经能说明“做矩阵乘法，再加 bias，再 ReLU”。但 CPU 和 GPU 还需要更具体的答案：循环遍历哪些元素，每个元素从哪里加载，结果写到哪里，以及哪些计算可以共用一次遍历。

Inductor 在这一阶段把张量操作转换成便于索引分析、融合和代码生成的内部表示。本章先追踪一个输出元素怎样成为计算表达式，再用一次真实的融合前后输出，解释为什么有些操作在进入scheduler前已经组合，有些则保留为独立节点后再融合。

## 1. 从张量运算到元素计算

先假设矩阵乘法已产生 `M[B,N]`，需要继续计算：

```text
Y[i,j] = relu(M[i,j] + bias[j])
```

从图看，是 add 和 relu 两个节点。从输出元素看，只需完成一次加载 M、一次加载 bias、一次加法和一次激活。计算 `(i,j)` 时，其他位置的结果与它独立。

可以用下面的索引函数表达这段计算。它是概念伪代码，不是要求背诵的 Inductor 构造 API：

```python
def output_element(i, j):
    m = load(M, i * N + j)
    b = load(bias, j)
    return relu(m + b)
```

这个索引函数在编译器中描述“将来怎样计算一个元素”，不是已经对某个实际指针执行 load。生成器以后根据它构造循环体；目标程序运行时才真正读取 M 与 bias。

代入主例 `M=[[0,8],[1,-3]]`、`bias=[0.5,-0.5]`：位置 `(1,1)` 读取 M 的相对偏移3和 bias 的偏移1，得到 -3 与 -0.5，计算 ReLU(-3.5)=0。一个抽象的索引表达式因此同时联系了输出坐标、读取位置和计算结果。

循环范围为 `0≤i<B, 0≤j<N`。计算函数描述一个位置做什么，范围描述要生成哪些位置。两者结合，就能形成 CPU 循环或 GPU 并行程序。

这比“add 输出一个 Tensor，relu 再输入一个 Tensor”更接近融合需要的信息：relu 可以直接消费加法表达式，不必首先把全部 add 结果写回内存。

## 2. 广播如何进入索引

bias 的形状只有 `[N]`，却参与每一行计算。广播并不要求构造 B 份 bias。它只改变输入索引与输出索引的关系：

| 输出位置 | M 的元素 | bias 的元素 |
|---|---|---|
| `(0,0)` | `M[0,0]` | `bias[0]` |
| `(0,1)` | `M[0,1]` | `bias[1]` |
| `(1,0)` | `M[1,0]` | `bias[0]` |

如果把二维循环展平成 `p∈[0,BN)`，连续 M 的偏移是 p，bias 的偏移则是 `p % N`。在 N 固定时，索引可进一步简化；N 为符号时，仍可保留相应整数表达式。

广播关系因此影响内存访问分析，也影响融合与向量化。它不是一项在前端确认 shape 后就可以丢弃的信息。

## 3. 逻辑位置与物理布局

刚才 `i*N+j` 假定 M 连续存储。更一般的地址关系是：

```text
offset(i,j) = storage_offset + i * stride[0] + j * stride[1]
```

`s0,s1` 是以元素计的 stride。shape 表示有哪些逻辑元素，stride 表示这些元素如何位于存储中。

配套实验使用同样的 x 数值构造两种 `[5,7]` 张量：

```python
contiguous_x = x
strided_x = x.t().contiguous().t()
```

两者 shape 和元素值相同，stride 分别为 `[7,1]` 与 `[1,5]`。例如逻辑位置 `(1,2)`，前者的相对偏移为 9，后者为 11。

后端必须使用正确索引，或者显式变换布局后再调用要求特定布局的实现。不能仅凭 `[5,7]` 就假定每一行在内存中连续。两个布局在本机 Inductor 编译后都通过了与 eager 的数值比较。

布局也影响代价：内层循环访问相邻地址时，更容易形成向量化和合并访问；跨较大步长读取时，可能需要不同循环顺序、复制或专用实现。布局选择因此与算子实现、融合和存储规划相互影响。

## 4. 延迟计算与 Realization

如果每遇到一个 pointwise 节点就分配输出内存，前面的 add 和 relu 已经被迫分开。另一种组织方式是先保存“如何计算这个值”的表达式，等到必须有独立存储时才把它落实为 buffer。

例如：

```text
add 的结果：表达式 M(i,j) + bias(j)
relu 的结果：表达式 relu(M(i,j) + bias(j))
```

进一步把两个元素计算函数连起来，表达式融合的意图是：

```text
原来：add_element(i,j) = load(M,i,j) + load(bias,j)
      relu_element(i,j) = relu(add_element(i,j))

代入以后：relu_element(i,j) = relu(load(M,i,j) + load(bias,j))
```

relu 使用的是“怎样算出加法结果”的表达式，而不是必须先读取一个保存全部加法结果的数组。这里移除了对中间数组的需求，但没有移除加法本身。

当 relu 可以直接使用加法表达式时，加法结果无需单独落地。这种按需形成有名字存储对象的过程，在 Inductor 中称为 **realization**。

何时需要落地？常见原因是后续外部库要求一个实际张量输入，某个中间值需要复用以避免重复计算，或者边界/别名约束要求明确存储。是否落地以及何时落地，不能只看“这个节点有没有返回值”。

在固定版本实现中，逐元素计算由 `Pointwise` 一类对象表达；归约有 `Reduction` 等表示。`StorageBox.realize()` 可以把延迟计算包装为 `ComputedBuffer`，注册到图中的 buffer 和操作集合。`TensorBox`、`StorageBox` 参与维护逻辑张量与存储/计算对象之间的关系。

可以把 `StorageBox.realize()` 的作用按编译时对象变化理解：原来盒子里保存 Pointwise 等计算描述；realize 为它建立有名字和布局信息的 ComputedBuffer，并登记到当前图；后续调度与代码生成就能把它作为独立结果处理。

这一步是在编译器内确定存储/调度对象，不是调用模型的那一刻已经 malloc 出实际张量。真正的分配、写入和复用，还要由生成程序与运行时落实。之后的调度也仍可能组合计算或重用存储，所以“存在一个有名字的 IR buffer”不等于“最终必定独占一块物理分配”。

如果中间结果是昂贵归约，两个消费者各内联一遍可能把归约重复两次；先产生一个可共享结果可以避免重复，却要付出存储与读取成本。只有跟踪消费者和计算代价，才能判断何时物化合适。

## 5. 一个逐元素链为何已经只有一个节点

为了隔离外部matmul的影响，使用一个小变体：

```python
def elementwise(x, y, z):
    return torch.relu(x + y) * z
```

本机对长度33的连续float32向量编译，实际 `ir_pre_fusion.txt` 中就只有一个 `SchedulerNode(ComputedBuffer)`，循环体整理为：

```text
p = 当前输出索引
value = relu(load(x,p) + load(y,p)) * load(z,p)
store(buf0,p,value)
```

三个张量操作没有先强制成为三个独立buffer。前面解释的延迟计算允许把加法、ReLU和乘法组合到一份元素表达式中，之后才将最终结果物化。因此这里的一个节点不能作为“scheduler刚刚把三个节点融合”的证据；要对照它真正收到的输入。

这项设计减少不必要的中间物化，也降低后续调度节点数量。若消费者要求实际张量、某个值需要对外返回或存在其他约束，计算就可能必须先建立独立buffer，交给下一层处理。

## 6. 从两份 ComputedBuffer 到一个融合组

只改变一个条件：同时返回ReLU的中间结果。

```python
def observable(x, y, z):
    a = torch.relu(x + y)
    return a, a * z
```

a现在是用户输出，必须保存其可观察值。实际融合前表示有两个节点，下面保留主要字段：

```text
op0: SchedulerNode(ComputedBuffer)
  reads:  x[p], y[p]
  writes: buf0[p] = relu(x[p]+y[p])
  users:  op1, OUTPUT

op1: SchedulerNode(ComputedBuffer)
  reads:  buf0[p], z[p]
  writes: buf1[p] = buf0[p]*z[p]
  users:  OUTPUT
```

scheduler中的 `MemoryDep` 记录buffer、索引和迭代范围，两个节点都遍历33个元素。op1读取的buf0[p]由op0在同一p产生；两者设备和当前CPU后端的循环组织兼容，合并不会引入依赖环。固定实现从候选组织进入 `can_fuse`，结合后端检查、评分及配置相关的收益判断，再构造融合节点。

实际 `ir_post_fusion.txt` 出现：

```text
op0_op1: FusedSchedulerNode(SchedulerNode, SchedulerNode)
  reads: x[p], y[p], z[p]
  writes: buf0[p], buf1[p]
```

注意两个输出buffer都保留了。生成代码却可以在同一次遍历中完成：

```text
a = relu(x[p] + y[p])
store(buf0,p,a)
store(buf1,p,a*z[p])
```

a仍必须写出供调用者使用，乘法可以直接消费局部计算值，省掉重新从buf0加载。这个案例说明：输出有其他使用者不必然禁止融合，但它限制能否删除存储或覆盖旧值。实际writer也不能让buf1覆盖唯一的buf0，因为调用方需要两份不同结果。

如果访问变成跨索引读取，或依赖路径要求某个外部阶段先完成，就需要重新检查，不能只看两个节点都是逐元素。合法性和收益也分开：组合后寄存器压力、计算重复或目标限制都可能影响选择。本例没有对每个候选进行计时搜索，也不把CPU策略推广为所有GPU策略。

这条链可以与TVM比较：TVM的通用图分组使用实现模式和图路径；这里的Inductor scheduler消费已物化计算的访存依赖及后端条件。它们解决相关问题，但并不共享同一个分组算法。

## 7. 外部实现与内部计算共存

回到第一章的实际 CPU 结果，Inductor 选择了 `addmm` 库调用，所以 bias 加法已经包含在库实现中。此时内部 pointwise 表示只需处理 ReLU。

实际调度前输出的关键部分为：

```text
op0: ExternKernelSchedulerNode
  kernel = extern_kernels.addmm
  output = buf0, size=[2,2], stride=[2,1]

op1: SchedulerNode(ComputedBuffer)
  read  = buf0[p], 0 <= p < 4
  value = relu(read)
  write = buf1[p]
```

这是原始调度表示的整理摘录。外部实现和可生成的计算共同构成后端工作集合，scheduler 需要维护它们之间的依赖。

“Extern” 在这里表示通过已有调用接口执行；不能据此断言它一定来自某一个具体 BLAS 库。更底层的实际库选择还与 PyTorch 构建、dtype、设备和形状相关。

图层已经将 `mm+add` 组织为 addmm，并不意味着 ReLU 也能穿过这个外部调用边界合并进去。若库不暴露适用的 epilogue 接口，后端就需要独立 ReLU，或比较另一种整体实现。

## 8. 归约维度与并行约束

逐元素计算中，一个输出位置独立于其他位置。Softmax 则不同：

```text
m[i] = max_k(x[i,k])
y[i,j] = exp(x[i,j] - m[i]) / sum_k(exp(x[i,k] - m[i]))
```

输出维度里有行 i 和列 j，而 max/sum 还需要遍历整行的归约维 k。生成代码时不能让每个位置只读自己就完成结果。

因此后端要同时表达输出范围与归约范围，并决定归约怎样分块、由哪些执行单元协作、是否需要中间存储。归约与 pointwise 的融合可以减少搬运，但可能引入重复计算、寄存器压力或同步成本。

例如同一行有三列，如果每个输出独立重算整行的 max 和 sum，结果仍可能正确，但同一归约会被多次执行。另一个方案是先算行统计量，再让三列共享；它需要保存统计量并安排生产—消费顺序。后端比较的正是这些不同的计算与存储组织，而不只是一个 softmax 节点的名称。

这也是为什么 Inductor 需要比 FX 更接近循环和索引的表示：仅仅知道“存在一个 softmax 节点”，不足以比较这些实现方式。

## 9. 推演与实践入口

1. 三个FX逐元素操作在ir_pre_fusion里只有一个节点，为什么不能直接说是scheduler把三者融合了？
2. observable变体融合后为什么还要两个输出buffer？减少了一次读取与删除一个输出存储有什么不同？
3. MemoryDep中的索引和范围怎样帮助判断候选？若消费者改读相邻位置，为什么需要重新检查？

实验见 [labs 的 inductor](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。有限源码范围为 [lowering.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/lowering.py) 中具体操作的转换、[ir.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/ir.py) 的计算与存储表示，以及 [graph.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/graph.py) 向 scheduler 的交接。

逐元素两种变体的实际记录来自 [design_probe](https://github.com/jnfkdsn/aicompiler/blob/main/pytorch-compiler/design_probe.py)。调度入口可查 [scheduler.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/scheduler.py) 的 fuse_nodes_once、can_fuse 与后端交接。第六章继续追同一输出到目标代码、wrapper与执行，并实现一次有限FX改写。
