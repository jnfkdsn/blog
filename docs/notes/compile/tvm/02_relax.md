---
order: 2
title: 模型导入、Relax 数据流与形状信息
updated: 2026-10-28
---

# 模型导入、Relax 数据流与形状信息

第一章直接写出了模型的 Relax 表示。实际使用时，模型通常已经存在于 PyTorch 等框架中。导入的任务，是把已取得的源程序语义映射到 TVM 表示，并保留后续编译需要的形状、类型、参数和调用关系。

本章仍然使用矩阵乘法、bias 和 ReLU，先观察一个实际导入结果，再解释里面的信息怎样支持后续转换。

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

对导入器而言，FX 图不是一张图片，而是节点及引用关系。翻译本例时，可以按下面的过程理解其工作，变量名仅作示意：

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

## 3. StructInfo 与后续决策

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

把 bias 改为 `[3]`，它与 `[2,2]` 的末维无法按原规则广播。错误发生在计算结构的约定上，不能继续使用原来的循环然后指望结果正确。StructInfo 的价值正是将这些决定所依赖的事实显式提供给消费者。

因此 shape/dtype 不是为了让打印结果好看而增加的注释。若把 bias 换成不兼容的长度，广播关系就改变或不成立，后面不能照原实现继续执行。

未知 shape 也不等于不存在任何信息。已知 rank、dtype 或部分维度仍然有价值。优化能用多少信息，取决于它需要证明什么条件。

## 4. Dataflow 区域的意义

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

这是在主例上增加一个可观察结果。`R.output` 让值离开 dataflow 区域，`return` 把值交给函数调用方；两者都不是“立即复制一块矩阵”的命令。若 added 的某个元素为 -3.5，调用方现在必须分别得到 -3.5 和 0。覆盖唯一的 added 存储再把它当成两个结果返回，就会改变行为。

纯数据流假设给优化提供空间：在保持依赖和可观察结果的前提下，可以重组内部计算。具有副作用的调用和更一般控制逻辑则需要遵守相应区域/调用规则，不能为了方便融合随意声明为纯操作。

与 MLIR 的 Region 类似，这里也需要关注值的作用域和出口，但不应直接套用 MLIR 的所有结构规则。应以 Relax 自己的数据流和函数语义为准。

## 5. 权重作为输入或常量

当前模型把 weight、bias 显式传入，因此同一编译程序可以接受相同结构但不同数值的权重。编译器不能仅因它们名为 weight 就把数值当成常量。

若源模型将它们保存为模块参数，导入过程可以按选项把参数保留为输入，或将数据绑定为常量。两者影响后续工作：

- 绑定常量使常量计算、预处理和特定优化有更多机会，但产物与这些参数值发生关联。
- 保留输入更便于更换权重或把参数存储与程序分开管理，但编译时不能假设其元素值已知。

固定版本的 FX 导入器提供 `keep_params_as_input` 相关能力。使用时要查看导入后的函数签名与参数元数据，而不是仅看原 PyTorch 模型有几个显式 `forward` 参数。

仅固定 weight，仍然不能提前算出依赖 x 的矩阵乘法。例如已知一列权重为 `[1,2]`，输出仍是 `x[0]*1 + x[1]*2`；乘数已知，输入数值未知。编译器可能预处理权重或简化特定运算，但不能把整个结果变成常量。

常量折叠也只适用于已经确定的常量计算。本系列主例的三个 Tensor 都是输入，所以相应 pass 并不会凭空计算出全部网络输出。

## 6. 符号 batch 与动态执行

将输入 shape 从 `(2,3)` 改成 `("batch",3)`，输出则为 `("batch",2)`。权重内积维仍然固定，编译器可以表达：

```text
输入行数 B 来自运行时
输出行数也是 B
每行输出2个元素，内积遍历3个元素
```

在对应的手写 Relax 实验中，一次编译后用 batch 1、2、5 执行，结果均与 NumPy 参考一致。生成程序需要读取维度、计算输出需求，并在数值函数中使用相应动态边界。

当 B=5 时，生成的执行过程要准备 `[5,2]` 输出。伪代码中的 i 遍历五行，j 仍遍历两列，k 仍累加三项。B 未知只意味着部分边界延迟到运行时，并不删除输出与输入具有相同行数这一关系。

这个实验验证的是符号 Relax 的编译与运行，没有宣称前面的静态 FX 导入调用已覆盖所有动态形状模型。每条导入路径对符号、数据相关 shape 和控制流的支持，都需要单独确认。

有了符号 batch，接下来的变换仍然需要保留维度之间的关系。将某个结果误写成固定2行，不仅是少了一条注释，而会使存储和循环契约发生错误。

## 7. 推演与实践入口

1. 导入 add 节点时，它的第一个参数引用源 matmul 节点。导入器怎样找到对应的 Relax 值？如果尚不支持这个源操作，问题属于图捕获还是图翻译？
2. x 未知而 weight 固定时，为什么仍然算不出整个模型？若额外返回 added，为什么不能直接将其唯一存储覆盖为 ReLU 结果？
3. 对 `x[B,3] @ weight[3,2]`，B=5 时结果形状是什么？i、j、k 的哪些边界由运行输入决定？

实验见 [TVM labs 的 import_model](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)。对应固定源码为 [FX importer](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/frontend/torch/fx_translator.py)、[StructInfo 定义](https://github.com/apache/tvm/blob/v0.20.0/include/tvm/relax/struct_info.h) 和 [BlockBuilder](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/block_builder.py)。
