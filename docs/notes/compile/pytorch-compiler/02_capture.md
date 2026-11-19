---
order: 2
title: Dynamo 图捕获与 FX 数据流
updated: 2026-10-30
---

# Dynamo 图捕获与 FX 数据流

上一章已经看到：`torch.compile` 能为 `relu(x @ weight + bias)` 选择一个执行方案。要作出选择，编译器首先必须取得相互连接的计算。然而 Python 函数保存的是一段可执行逻辑，里面还可能有循环、条件、对象访问和日志调用，并没有现成的完整张量图。

图捕获的工作，就是从这段逻辑中整理出可以交给后端处理的计算，同时维持剩余 Python 行为的语义。本章沿着“Python 表达式怎样成为图节点，再怎样成为可调用对象”展开。

## 1. Python 执行顺序与张量数据流

仍然使用同一个函数：

```python
def model(x, weight, bias):
    return torch.relu(x @ weight + bias)
```

若按 Python 求值顺序看，需要取得 `x`、`weight`，执行 `@`，再取得 `bias` 做加法，最后查找并调用 `torch.relu`。其中“查找函数名”和“矩阵相乘”属于不同层次：前者是 Python 程序行为，后者是希望优化的张量计算。

Dynamo 在 CPython 的 frame evaluation 边界接入，通过字节码分析跟踪这些行为。frame 可以先理解为某次函数执行所需的局部变量与执行位置；这里无需先掌握 Python 解释器全部实现。

以 `x @ weight + bias` 为例，捕获期间需要同时维护两类信息：一个值在图中由哪个节点代表，以及该值的形状、类型等元数据。可以按下面的顺序理解，表格是处理意图，不是逐条字节码的调试日志：

| 当前行为 | 图中的变化 | 可推导的信息 |
|---|---|---|
| 取得 x、weight | 建立输入引用 | `[2,3]` 与 `[3,2]` |
| 处理矩阵乘法 | 新建 matmul 节点，参数引用两个输入 | 结果 `[2,2]` |
| 处理加法 | 新建 add 节点，参数引用 matmul 和 bias | 结果仍为 `[2,2]` |
| 处理 ReLU | 新建 relu 节点，引用 add | 结果 shape 不变 |
| 处理返回 | 输出引用 relu | 后端需要交付哪个值 |

在处理 add 时，捕获器已经有代表 matmul 结果的代理，不需要先求出矩阵元素。因而“前一个结果成为后一个输入”可以通过节点引用表达出来。

当它遇到可识别的张量操作时，会记录一个图节点。遇到常量、对象成员和可推导的 Python 行为时，则可能在捕获期间处理，并把依赖的假设记下来。例如某个模块的 `training` 属性决定一条分支，捕获结果必须受对应条件保护。

因此图捕获既不是扫描函数源码中的字符串，也不是把某一次计算得到的全部数值存下来。它要保留的是“如何由输入计算输出”，以及这份描述在什么条件下成立。

## 2. FX 节点与 GraphModule

本例的图包含三个输入、三个计算节点和一个输出节点。按 FX 的节点类别整理如下：

| 节点 | 类别 | 保存的信息 |
|---|---|---|
| x、weight、bias | `placeholder` | 从外部调用取得的参数 |
| matmul | `call_function` | 调用矩阵乘法，参数引用 x 和 weight |
| add | `call_function` | 调用加法，参数引用 matmul 和 bias |
| relu | `call_function` | 调用激活，参数引用 add |
| output | `output` | 返回 relu 的结果 |

这里的 `matmul` 参数引用的是**节点**，不是字符串形式的变量名。由这种引用关系，后端可以知道矩阵乘法的结果只有一个使用者，或者某个中间值同时流向两个分支。这与已学过的 use-def 关系有相通之处。

FX 的 `GraphModule` 把图及其所需模块属性组织在一起，并能生成 Python `forward` 方法。下面是本例捕获结果的整理版，只将临时变量重命名：

```python
def forward(self, x, weight, bias):
    matmul = x @ weight
    add = matmul + bias
    relu = torch.relu(add)
    return (relu,)
```

打印结果像 Python，不意味着编译器重新分析这段打印文本。实际交接的是图对象。生成的 `forward` 提供了一个直接执行图的方式，也使图容易观察。

tuple 返回是本例后端图的调用协议。外层编译包装会恢复原函数的返回结构，用户调用 `compiled(...)` 仍然得到原来的 Tensor。内部协议和用户函数接口可以不同，但必须有明确的恢复过程。

## 3. 形状信息与没有真实数值的计算

只知道存在矩阵乘法还不够。后端还需要知道输入秩、维度关系、dtype、设备和 stride 等信息。例如 x 为 `[2,3]`、weight 为 `[3,2]`，便能推出结果为 `[2,2]`；若内积维不匹配，计算本身就不成立。

在捕获和后续追踪中，PyTorch 可以使用携带张量元数据的 **FakeTensor**。它支持推导操作对 shape、stride、dtype 等的影响，而不以运行真实大矩阵为前提。捕获器使用的代理将这类元数据计算与图节点记录联系起来。

对于本例：

```text
Fake x：形状 [2,3]，float32，逻辑设备 CPU
Fake weight：形状 [3,2]，float32
         ↓ 推导 matmul 的元数据
Fake matmul：形状 [2,2]，float32
```

这一步没有得到 `[[0,8],[1,-3]]` 的元素值。元素值要到真正执行程序时才计算。也因此，一个依赖 `x.sum()` 实际数值的 Python `if`，不能仅凭这些元数据决定走哪一支。

FakeTensor 比“每个元素都填零的临时 Tensor”更准确：它不是用假数值近似模型结果，而是专门支持元数据和符号关系的推导。对于无法仅凭元数据确定的事情，系统需要其他机制或暴露边界。

## 4. Backend 的输入输出契约

可以安装一个很小的后端，让交接过程直接可见：

```python
graphs = []

def inspect_backend(gm, example_inputs):
    graphs.append(gm)
    print(gm.code)
    return gm.forward

compiled = torch.compile(model, backend=inspect_backend, fullgraph=True)
y1 = compiled(x, weight, bias)
y2 = compiled(x, weight, bias)
```

这里 `gm` 是捕获的 GraphModule，`example_inputs` 提供后端处理所需的输入信息。后端返回一个可调用对象，Dynamo 之后通过它执行相应图段。

这个观察后端直接返回 `gm.forward`，所以图仍然通过 Python 调用现有 PyTorch 操作执行。它证明了“图已捕获、调用已交接”，没有证明已经生成更快的机器码。默认 Inductor 后端会在同一交接位置继续编译，返回另一种执行实现。

从用户第一次调用开始，这个观察后端的交接可以完整展开：

```text
第一次 compiled(x,w,b)
  Dynamo 构造 gm，并准备适用条件
  调用 inspect_backend(gm, example_inputs)
    打印图，返回 gm.forward
  保存后端返回的 callable
  调用这个 callable，实际计算 y

第二次 compiled(x2,w2,b2)，相关条件仍满足
  复用已有 callable
  callable 读取新的元素并计算 y2
```

这里 backend 是“为图准备执行方式”的函数，它返回的 callable 才负责每次执行图。两个函数可能都是 Python 对象，但调用时机和职责不同。换成 Inductor 时，后端返回的执行入口连接的是编译产物；当前观察后端返回 FX forward，所以仍调用现有 PyTorch 操作。

固定实验中，连续两次输入相同条件的调用只触发一次 `inspect_backend`，两次结果均为：

```text
[[0.5, 7.5], [1.5, 0.0]]
```

后端被调用的时机是准备图实现；返回的 callable 被调用的时机是执行图。区分这两类调用，才能正确理解后面记录的“编译次数”。

## 5. Python 分支与捕获边界

现在给原模型增加一个根据输入形状选择的分支：

```python
def shape_branch(x, weight, bias):
    y = x @ weight + bias
    if x.shape[0] > 4:
        return torch.relu(y)
    return y
```

如果本次捕获已知 batch 为 2，便能确定走最后一行。对这份特化实现而言，图里可以不包含通用的 Python `if`，但它不能不加条件地用于 batch 为 5 的调用。对应条件必须进入复用判断。

再把条件改为：

```python
if x.sum() > 0:
    return torch.relu(y)
```

这时，同样的 shape 可以对应不同分支。靠 shape 保护原先选定的分支就不够了。常规捕获可能在这个位置断图，由 Python 决定分支，再继续捕获后面的计算；如果要求 `fullgraph=True`，不支持的捕获会直接报错。需要图内控制流时，应考虑显式控制流操作及其约束，不能认为任意 Python 分支都会自动转成图内条件节点。

若第一次输入 x 的元素和为正，第二次元素和为负，而 shape 完全相同，两个调用应走不同分支。FakeTensor 的元数据相同，无法替代这项实际数值判断。普通 Python 分支和显式图内条件的区别，正在于分支由哪一层决定并执行。

断图后的执行仍需恢复局部值和继续位置。例如已经算出的 `y` 要交给后续 Python 或图段，不能因为分图就重新定义程序含义。下一章会把这件事与 guards 和重编译分别解释。

可以把捕获的产物看成两部分：一份FX计算图，以及调用编译图、维持剩余Python行为所需的执行逻辑。Dynamo会改写字节码并在需要时建立继续执行的位置；FX并不是整个Python函数的全部语义副本。

以断图场景为例，原字节码表达“算y、执行Python行为、再消费y”；图1接管y的计算，保留的执行逻辑调用图1、传递y并继续Python行为，后续再交给图2。这里是概念执行顺序，不是某个Python版本的实际反汇编清单。字节码细节随解释器变化，稳定的设计问题是怎样接回局部状态和控制位置。

这种设计允许部分编译，但也让图边界影响优化：图1与图2之间存在可观察Python行为时，后端不能仅凭张量公式相邻就把它们跨边界融合。要修改这种行为，应先定位捕获或残留执行逻辑，而不是从Inductor内核里寻找缺失的Python语句。

## 6. 有限源码阅读与学习边界

对图捕获有了上述模型以后，源码只需先追一条边界：

```text
编译入口
  → Dynamo 分析当前 Python frame
  → 构造 FX 图和相关假设
  → 调用 backend(gm, inputs)
  → 保存返回的 callable，并在满足条件时调用它
```

对应实现可从 [eval_frame.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_dynamo/eval_frame.py) 进入，在 [output_graph.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_dynamo/output_graph.py) 查后端编译的交接，再按需要读 [symbolic_convert.py](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_dynamo/symbolic_convert.py)。当要修改某种 Python 语法的捕获行为时，才继续深入它对应的变量追踪和字节码处理。

这一阶段更有价值的问题是“哪一种事实变成了图节点，哪一种事实变成了保护条件”，而不是背下所有内部类名。

## 7. 推演与实践入口

1. inspect_backend 只调用一次但模型调用两次，第二次实际由哪个 callable 计算结果？为何这还不能证明生成了机器码？
2. 处理 add 节点时，为什么不需要先算出 matmul 的元素？FakeTensor 能提供什么、不能提供什么？
3. 两次输入 shape 相同但元素和异号，根据 x.sum() 的 Python 分支为什么不能仅靠 shape 选择同一条路径？

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)的 `capture` 观察后端入口和复用；`shapes` 展开下一章的条件变化。元数据机制可核对 [FakeTensor 官方说明](https://docs.pytorch.org/docs/2.7/torch.compiler_fake_tensor.html)，FX 结构可核对 [FX 文档](https://docs.pytorch.org/docs/2.7/fx.html)。
