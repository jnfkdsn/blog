---
order: 2
title: 模型导入、Relax 数据流与形状信息
updated: 2026-11-24
---

# 模型导入、Relax 数据流与形状信息

第一章直接写出了模型的 Relax 表示。现在假设手里只有一个已有的 PyTorch 模型 `relu(x @ weight + bias)`：TVM 首先要拿到一份自己能够检查、改写的程序，后续才能为它选择实现。

导入后，模型关系可以写成：

```text
x、weight → matmul → mm
mm、bias  → add    → added
added     → relu   → y → 返回
```

但仅有这些连线还不够。例如 x 是 `[B,3]`、weight 是 `[3,2]`，才能推出 mm 是 `[B,2]`；weight 是每次调用传入的参数，还是模型中固定的常量，也会影响编译时能提前完成哪些工作。**本章要说明的，是 TVM 怎样把计算关系及其成立条件一起保存在 Relax 中。**

下面先观察源图怎样成为 Relax 绑定，再沿着同一份结果解释形状信息、数据流边界和参数。最后改变 batch，检查哪些信息可以保留为符号。这里还没有为矩阵乘法选择循环；我们正在为下一章的实现转换准备一份信息足够的模型表示。

## 1. 捕获源图与翻译 IR

本例先通过 PyTorch FX 取得简单的图，再由 TVM 翻译：

```python
import torch
from tvm.relax.frontend.torch import from_fx

class Model(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.relu(x @ weight + bias)

fx_model = torch.fx.symbolic_trace(Model())
mod = from_fx(fx_model, [
    ((2, 3), "float32"),
    ((3, 2), "float32"),
    ((2,), "float32"),
])
```

导入器沿 FX 图的节点和引用关系翻译计算。对本例，可以按下面的步骤追踪，变量名作示意：

| 读到的源节点 | 在 Relax 中建立的结果 | 后续保存的对应关系 |
|---|---|---|
| 输入 x、weight、bias | 三个有 shape/dtype 的参数 | 源输入节点 → Relax 参数 |
| matmul(x, weight) | `mm = R.matmul(x, weight)` | 源 matmul 节点 → mm |
| add(matmul, bias) | `added = R.add(mm, bias)` | 源 add 节点 → added |
| relu(add) | `result = R.nn.relu(added)` | 源 relu 节点 → result |
| 输出 relu | 返回 result | 函数输出连接完成 |

例如翻译 add 时，不能把参数中的 matmul 当成已经算出的矩阵。导入器查出它先前对应的 Relax 值 mm，再把 mm 作为新加法的输入。这样源图的依赖才能完整保留下来。

两步职责不同。FX tracer 将这个简单 Python 模型变成图；`from_fx` 读取已有图，根据节点含义生成 Relax。不能把“TVM 能导入 FX”理解为“TVM 自动支持任意 Python 语义”。源图能否取得、导入器是否认识某个节点，都有各自边界。

本例使用普通 symbolic tracing，不等同于 PyTorch Compiler 中 Dynamo 的全部捕获能力。针对 ExportedProgram，固定 TVM 版本还有单独的导入入口；本实验不把不同入口的支持范围混在一起。

## 2. 节点到 Relax 绑定

实际导入结果的核心如下，参数重命名为正文名称：

```python
with R.dataflow():
    mm: R.Tensor((2, 2), "float32") = R.matmul(x, weight, out_dtype="float32")
    added: R.Tensor((2, 2), "float32") = R.add(mm, bias)
    result: R.Tensor((2, 2), "float32") = R.nn.relu(added)
    R.output(result)
return result
```

每个绑定把一个计算结果关联到变量，后续表达式引用这些变量。编译器可以沿 mm→added→result 追踪数据流，分析某个中间值被谁使用，决定怎样分组或改写。

Relax 函数和后续 TensorIR `PrimFunc` 可以同时放在 IRModule 中。全局函数引用将不同层次连接起来：模型层调用一个张量程序，不必把所有循环直接嵌入每个模型节点。

这也让导入和优化可以分开演进。导入器首先负责正确表达源计算；后续 pass 再根据目标和实现策略修改它。导入时就强行固定所有循环选择，会过早丢失优化空间。

把两层函数放在同一模块中，也让模型优化能够检查底层实现。例如第三章会查看 TensorIR 的访问模式，再据此决定 Relax 中哪些调用可以融合。修改实现时，必须同步检查调用参数、结果类型和形状，否则模型入口可能调用一份不再符合要求的函数。

## 3. StructInfo 与后续决策

上面的绑定给出了“谁依赖谁”。后续转换还要知道结果是什么形状和类型，才能为它生成合法的计算与输出空间。

Relax 的 **StructInfo** 描述表达式的结构信息，例如某个值是 Tensor、dtype 是 float32、秩为 2、形状是 `(2,2)`。它可以比一般类型信息更细，并允许某些信息暂时未知或使用符号表达。

对本例，推理过程是：

```text
x[2,3] 与 weight[3,2]
  → 内积维3匹配
  → matmul结果[2,2]
  → bias[2]可沿首维广播
  → add与relu结果仍为[2,2]
```

跟踪一个输出位置 `(i,j)`，这些结构信息会真正进入决定：内积维 3 让生成实现知道要累加三项；bias 的末维长度 2 让访问表达式使用 `bias[j]`；输出 `[2,2]` 和 float32 让后续知道结果需要四个元素、共 16 字节的元素空间。运行时还要核对实际输入是否符合这份程序的约定。

例如把 bias 改为 `[3]`，它就无法与 `[2,2]` 按原规则广播。编译器需要拒绝这种组合，而不能仍按 `bias[j]` 生成原来的计算。StructInfo 提供的 shape 和 dtype，正是检查广播、推导结果形状和安排输出空间的依据。

即使完整 shape 未知，已知的 rank、dtype 和部分维度也能支持分析。每项优化根据自己所需的条件使用这些信息。

## 4. Dataflow 区域的意义

模型内部还有一个问题：哪些中间值只在这一段计算里使用，哪些值必须保留给外面的代码？

`R.dataflow()` 标出适合以纯数据依赖分析和变换的区域，内部变量通常作为区域局部数据流值，`R.output` 指定哪些值可以流向外部。

本例中 mm 和 added 只服务于最终 result，因此只将 result 输出。若模型需要同时返回激活前后的结果，就要让相应值都越过区域边界，后续融合和存储复用也必须保留两者。

可以把区域出口和函数出口分别写清楚：

```python
with R.dataflow():
    mm = R.matmul(x, weight)
    added = R.add(mm, bias)
    result = R.nn.relu(added)
    R.output(added, result)
return added, result
```

主例现在多返回一个结果：`R.output` 将值传出 dataflow 区域，`return` 再交给调用方。假如 added 的一个元素为 -3.5，两个返回结果就分别需要保留 -3.5 和 0，因此存储安排必须保留这两种值。

如果这段计算只通过输入输出传递结果，编译器就可以沿数据依赖判断哪些操作可以移动、合并。若一个调用还会修改其他存储或产生外部效果，就必须一并保留这些行为，不能随意把它标为纯操作来参与融合。

与 MLIR 的 Region 类似，这里也需要关注值的作用域和出口，但不应直接套用 MLIR 的所有结构规则。应以 Relax 自己的数据流和函数语义为准。

这里要区分变量的可见范围与存储的使用时间。DataflowVar 能否在区域外引用，由 IR 的结构规则决定；内存能否释放或复用，则要看之后是否还有代码通过其他引用读取它。因此，离开 `R.dataflow()` 不会直接执行 free，`R.output` 也不会直接分配或复制数据。

## 5. 权重作为输入或常量

数据流区域说明了计算内部的关系，但函数边界还有一个选择：权重是运行时输入，还是编译器已经知道的值？

这里 weight、bias 显式作为输入，相同结构的不同权重可复用同一程序。若要在编译时利用其数值，需要先将它们绑定为常量。

若源模型将它们保存为模块参数，导入过程可以按选项把参数保留为输入，或将数据绑定为常量。两者影响后续工作：

- 绑定常量使常量计算、预处理和特定优化有更多机会，但产物与这些参数值发生关联。
- 保留输入更便于更换权重或把参数存储与程序分开管理，但编译时不能假设其元素值已知。

固定版本的 FX 导入器支持 `keep_params_as_input`。导入后可以查看函数签名和参数元数据，确认哪些模型参数作为运行输入、哪些已绑定为常量。

仅固定 weight，仍然不能提前算出依赖 x 的矩阵乘法。例如已知一列权重为 `[1,2]`，输出仍是 `x[0]*1 + x[1]*2`；乘数已知，输入数值未知。编译器可能预处理权重或简化特定运算，但不能把整个结果变成常量。

常量折叠处理已经确定的数值。主例的三个 Tensor 都是运行输入，其计算保留到执行阶段。

## 6. 符号 batch 与动态执行

常量绑定解决了哪些数据在编译时已知。现在保留同一模型，只让输入 batch 在调用时才确定，观察形状关系怎样继续传递。

将输入 shape 从 `(2,3)` 改成 `("batch",3)`，输出则为 `("batch",2)`。权重内积维仍然固定，编译器可以表达：

```text
输入行数 B 来自运行时
输出行数也是 B
每行输出2个元素，内积遍历3个元素
```

在对应的手写 Relax 实验中，一次编译后用 batch 1、2、5 执行，结果均与 NumPy 参考一致。生成程序需要读取维度、计算输出需求，并在数值函数中使用相应动态边界。

B=5 时，程序准备 `[5,2]` 输出，i 遍历五行，j 遍历两列，k 累加三项。动态 B 决定本次行数，同时保持输出与输入行数相同。

这里使用手写的符号 Relax，前面的 FX 导入示例仍使用静态形状。若从其他模型导入动态形状，需要另行确认对应导入器是否支持其中的符号维度、数据相关形状和控制流。

后续变换必须继续保留这个 B。例如输入有 5 行，若错误地按固定 2 行准备输出和生成循环，就会漏算三行，甚至使读写越界。形状信息会直接影响运行行为。

## 7. 推演与实践入口

1. 导入 add 节点时，它的第一个参数引用源 matmul 节点。导入器怎样找到对应的 Relax 值？如果尚不支持这个源操作，问题属于图捕获还是图翻译？
2. x 未知而 weight 固定时，为什么仍然算不出整个模型？若额外返回 added，为什么不能直接将其唯一存储覆盖为 ReLU 结果？
3. 对 `x[B,3] @ weight[3,2]`，B=5 时结果形状是什么？i、j、k 的哪些边界由运行输入决定？

实验见 [TVM labs 的 import_model](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)。对应固定源码为 [FX importer](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/frontend/torch/fx_translator.py)、[StructInfo 定义](https://github.com/apache/tvm/blob/v0.20.0/include/tvm/relax/struct_info.h) 和 [BlockBuilder](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/block_builder.py)。
