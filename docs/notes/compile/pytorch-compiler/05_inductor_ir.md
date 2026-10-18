---
order: 5
title: Inductor 的计算、索引与存储表示
updated: 2026-10-17
---

# Inductor 的计算、索引与存储表示

规范化的图已经能说明“做矩阵乘法，再加 bias，再 ReLU”。但 CPU 和 GPU 还需要更具体的答案：循环遍历哪些元素，每个元素从哪里加载，结果写到哪里，以及哪些计算可以共用一次遍历。

Inductor 在这一阶段把张量操作转换成便于索引分析、融合和代码生成的内部表示。本章不按类名逐一介绍，而是追踪一个输出元素怎样从图上的值，变成计算表达式，再变成存储与调度对象。

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

当 relu 可以直接使用加法表达式时，加法结果无需单独落地。这种按需形成有名字存储对象的过程，在 Inductor 中称为 **realization**。

何时需要落地？常见原因是后续外部库要求一个实际张量输入，某个中间值需要复用以避免重复计算，或者边界/别名约束要求明确存储。是否落地以及何时落地，不能只看“这个节点有没有返回值”。

在固定版本实现中，逐元素计算由 `Pointwise` 一类对象表达；归约有 `Reduction` 等表示。`StorageBox.realize()` 可以把延迟计算包装为 `ComputedBuffer`，注册到图中的 buffer 和操作集合。`TensorBox`、`StorageBox` 参与维护逻辑张量与存储/计算对象之间的关系。

这些名字对应刚才的需求：表达式先说明怎么算，buffer 进一步说明有一个要物化和调度的结果。不是每个程序都会按同一种固定对象链经历全部步骤。

## 5. 外部实现与内部计算共存

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

## 6. 归约维度与并行约束

逐元素计算中，一个输出位置独立于其他位置。Softmax 则不同：

```text
m[i] = max_k(x[i,k])
y[i,j] = exp(x[i,j] - m[i]) / sum_k(exp(x[i,k] - m[i]))
```

输出维度里有行 i 和列 j，而 max/sum 还需要遍历整行的归约维 k。生成代码时不能让每个位置只读自己就完成结果。

因此后端要同时表达输出范围与归约范围，并决定归约怎样分块、由哪些执行单元协作、是否需要中间存储。归约与 pointwise 的融合可以减少搬运，但可能引入重复计算、寄存器压力或同步成本。

这也是为什么 Inductor 需要比 FX 更接近循环和索引的表示：仅仅知道“存在一个 softmax 节点”，不足以比较这些实现方式。

## 7. 推演与实践入口

1. bias 的广播如何体现在索引函数中，为什么不必复制成完整矩阵？
2. `[5,7]`、stride `[1,5]` 的逻辑 `(2,3)` 位于哪个相对元素偏移？
3. 一个中间表达式被两处使用时，直接内联与物化分别可能付出什么代价？
4. 为什么一个合法的张量图，还不能直接决定 GPU 上的线程分工？

实验见 [labs 的 inductor](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。有限源码范围为 [lowering.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/lowering.py) 中具体操作的转换、[ir.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/ir.py) 的计算与存储表示，以及 [graph.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/graph.py) 向 scheduler 的交接。
