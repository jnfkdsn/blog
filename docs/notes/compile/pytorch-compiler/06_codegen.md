---
order: 6
title: 目标代码、执行包装与编译器修改
updated: 2026-10-30
---

# 目标代码、执行包装与编译器修改

第五章得到一个融合组，含义是两个计算可以共同生成实现。这个组还没有可调用地址，也没有为本次输入分配输出。接下来要把循环体变成目标代码，编译装载，再让wrapper按协议调用。

本章先沿逐元素双输出例子走通CPU执行，再解释GPU/Triton的对应边界。最后在实际backend入口加入一个很小的FX改写，通过前后图与数值验证，完成从“观察生成代码”到“修改一次编译决定”的连接。

## 1. 调度结果怎样进入 CPU 代码

仍然计算：

```python
def observable(x, y, z):
    a = torch.relu(x + y)
    return a, a * z
```

固定长度33、连续float32的CPU生成物中，两个scheduler节点已经组成一个融合组。C++生成器将其索引与计算转换成循环。下列片段来自实际生成结果，参数和临时名为便于阅读做了重命名，展示完整向量段：

```cpp
auto vx = at::vec::Vectorized<float>::loadu(x + p, 16);
auto vy = at::vec::Vectorized<float>::loadu(y + p, 16);
auto vz = at::vec::Vectorized<float>::loadu(z + p, 16);
auto sum = vx + vy;
auto a = at::vec::clamp_min(sum, decltype(sum)(0));
auto result = a * vz;
a.store(out_a + p);
result.store(out_result + p);
```

此机器的输出按16个元素一组推进。p=0和16处理完整段，p=32只剩一个元素，生成代码使用有效长度1的加载和存储。16是本次目标与实现的观察，不是Inductor固定要求的向量宽度。

这里的a同时写入第一个输出、参与乘法，没有先写出再从out_a加载。两个输出仍各有存储，因为调用者需要两份结果。这样就能将第五章的融合决策连接到具体访存，而不是只看函数名里有没有fused。

## 2. 编译、装载与调用

实际output_code.py使用 `async_compile.cpp_pybinding` 提交C++实现，等待编译结果可用，再在wrapper中调用。核心结构为：

```text
生成C++源和接口描述
    ↓ 本机C++编译、链接；缓存命中时可复用产物
得到并装载可调用实现
    ↓
wrapper(x,y,z)：
    检查必要的shape/stride约定
    分配out_a和out_result
    调用内核(x,y,z,out_a,out_result)
    返回两个Tensor
```

生成源码是在构造程序，编译把它变成目标代码，装载使当前进程能够取得执行入口，调用才读取本批次的33个元素。后续条件仍满足时，可以直接复用执行入口，不必每次编译C++。

本例wrapper分配两个长度33的输出。若只返回最终乘法结果，第五章的单输出变体可以只有一个输出分配；若某个旧值已经不再使用，其他案例还可能复用buffer。复用需要使用关系和别名依据，并非两个Tensor形状相同就能覆盖。

## 3. GPU 路径与 Triton 的职责

相同逐元素计算在GPU上需要划分工作、限制越界并安排设备调用。下面是对应计算的完整Triton核心示意，只针对连续float32与有限值输入帮助阅读；它不是本机Inductor生成物，也没有设备执行证据：

```python
import triton
import triton.language as tl

@triton.jit
def pointwise(X, Y, Z, OUT_A, OUT_R, N: tl.constexpr,
              BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < N
    x = tl.load(X + p, valid, other=0.0)
    y = tl.load(Y + p, valid, other=0.0)
    z = tl.load(Z + p, valid, other=0.0)
    a = tl.maximum(x + y, 0.0)
    tl.store(OUT_A + p, a, valid)
    tl.store(OUT_R + p, a * z, valid)
```

N=33、BLOCK=32时，两个program分别覆盖0..31与32..63，第二个只有p=32有效。program是Triton工作划分的单位，不等于“一项Python运算对应一个线程”；warp、lane和布局等具体映射还要由Triton及编译配置落实。边界mask必须保护访存，不能只在最后屏蔽结果。

实际Inductor生成程序还带有启发式、配置和元信息，不必长得与这份教学代码完全相同。主要分工是：Inductor从模型图选择实现、形成融合组并生成Triton程序；Triton继续进行自身IR转换与目标代码生成；运行时通过设备入口和stream等协议发起执行。外部库调用也可以与生成的Triton内核共存。

本系列固定环境是2.7.1+cpu。配套GPU入口在有适用CUDA版PyTorch和设备时编译真实torch.compile变体、保存生成代码并校验结果；当前没有据此宣称GPU正确性或加速。这里也不能把其他版本的调度类名、输出格式直接套到本版本。

## 4. 库调用和归约为何形成不同实现

主模型 `relu(x @ weight + bias)` 的CPU结果采用addmm外部调用和独立ReLU内核。外部实现已经有自己的内部循环，scheduler不能像组合两个Pointwise表达式一样任意把ReLU塞进去。可以利用库的epilogue能力，或选择另一个整体实现，但需要相应实现与选择策略。

Softmax则在内部产生归约依赖。此前固定 `[2,3]` CPU观察中，虽然函数名为 `cpp_fused__softmax_0`，wrapper仍准备temporary与output两块矩阵。整理后的计算为：

```text
循环组一：求行最大值，将exp(x-row_max)写到temporary
循环组二：读取本行temporary求和，将归一化结果写到output
```

小归约被展开成元素表达式，函数融合没有消除全部临时存储。对于更大归约，还要比较分块、共享统计量、重算及资源压力。一个函数里有几个循环、读写哪些buffer，比名称更能说明优化效果。

这些例子体现后端的取舍：逐元素表达式容易就地组合，归约需要保持跨元素依赖，已有库要求遵守调用边界。正确性、兼容性和实际成本共同决定最终方案。

## 5. 一条可操作的诊断链

定位时先确定第一个异常出现在哪份表示，而不是同时阅读全部源码：

| 证据 | 重点检查 |
|---|---|
| Dynamo交给backend的FX图 | 捕获了什么，是否断图，参数与返回协议 |
| AOT交给后端的ATen图 | mutation、分解、保存值和别名契约 |
| ir_pre_fusion.txt | 哪些表达式已经组合，哪些结果被物化 |
| ir_post_fusion.txt | scheduler实际合了哪些节点，保留哪些输出 |
| output_code.py及目标代码 | 实际调用、索引、分配和中间读写 |
| 数值与计时 | 是否保持行为，时间消耗在哪个边界 |

不同debug配置还会提供 `fx_graph_runnable.py` 等文件，名称和可用性随版本而变。不能把任意名为fx_graph的文件都当成未经规范化的Dynamo原图；先确认它由哪个编译边界保存。

例如双输出变体在pre阶段有两节点、post阶段有一组、代码仍分配两输出，这是符合契约的结果。若后一个输出覆盖了前一个，应调查存储和别名；若候选根本没合，才查看scheduler的合法性与收益判断。

性能也要划清范围：首次调用可能含捕获和编译，稳态主要执行生成程序；若总时间90%在addmm，只优化剩余ReLU的收益上限很有限。GPU计时还要处理异步执行。生成内核数、分配数和端到端时间是三项不同证据。

## 6. 在 Backend 中加入一次实际 FX 改写

现在实现一个有限推理优化：将 `torch.relu(torch.relu(x))` 的外层ReLU删除。选择显式torch.relu调用，避免在一个练习中同时处理所有方法名和ATen重载。

先考虑反例：

```python
def shared(x):
    a = torch.relu(x)
    return a, torch.relu(a)
```

两项结果的数值相同，但源程序返回两个分别产生的Tensor。若直接返回 `(a,a)`，调用者修改第一项就会改变第二项。数值比较可能发现不了这个错误。因此本规则保守地要求inner只有外层ReLU一个使用者，拒绝上述共享情况。

完整的有限改写函数如下：

```python
def remove_repeated_relu(gm):
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.relu or node.kwargs:
            continue
        if len(node.args) != 1:
            continue
        inner = node.args[0]
        if not isinstance(inner, torch.fx.Node):
            continue
        if (inner.op != "call_function" or inner.target is not torch.relu
                or inner.kwargs or len(inner.args) != 1 or len(inner.users) != 1):
            continue
        node.replace_all_uses_with(inner)
        gm.graph.erase_node(node)
        changed += 1
    gm.graph.lint()
    gm.recompile()
    return changed
```

替换的是外层节点的使用引用，删除的是外层节点；inner保留。lint检查图结构，recompile更新GraphModule的Python执行代码，两者都不等于数值正确性证明，也不等于目标代码生成。

把这项修改接入真实Inductor：

```python
from torch._inductor.compile_fx import compile_fx

def backend(gm, example_inputs):
    remove_repeated_relu(gm)
    return compile_fx(gm, example_inputs)

compiled = torch.compile(repeated, backend=backend, fullgraph=True)
```

其中repeated是本节的单输出重复ReLU函数。compile_fx继续组织第四章的AOT处理和后端编译，返回实际执行入口。该内部API固定于2.7.1，不作为跨版本稳定扩展接口承诺。

## 7. 修改的验证与边界

实际CPU参考验证三个函数：重复ReLU改写一次；共享inner的双输出函数不改写；`relu(-x)`不改写。每个函数用混合正负、全负、全正三组输入与eager比较。双输出还在返回后修改第一项，确认第二项没有被连带修改。

我们同时保存backend入口的before/after FX，避免“结果正确，但自定义规则一次也没触发”。这项实验验证有限模式及调用链，没有覆盖训练、全部浮点特殊值、Tensor子类或所有原地语义，也没有证明性能优于Inductor已有优化。

独立修改时，可以先扩展一个明确的模式或拒绝条件，再写对应反例。遇到长源码，先追`backend → compile_fx → 某次图交接`，或`scheduler候选 → can_fuse → 后端codegen`中的一条。需要解释哪项决定，就读到能确认该决定的位置。

## 8. 推演与后续阅读

1. 双输出变体为什么融合成一组后仍要两块输出存储？C++中的局部a省掉了哪次读取？
2. 为什么图里有三个逐元素操作，不一定意味着scheduler融合前三个节点？如何通过三份输出区分？
3. 删除重复ReLU为何检查inner的使用者？lint、数值比较和调用后修改输出分别保护什么？

本章完成普通推理编译主线及一次有限扩展。下一章Export/AOT是部署专题，第八章研究训练协议；可以根据任务选择顺序，不把它们作为开始Triton学习的新门槛。

[CPU参考与改写](https://github.com/jnfkdsn/aicompiler/blob/main/pytorch-compiler/design_probe.py)保存实际图、调度表示和代码。[GPU观察入口](https://github.com/jnfkdsn/aicompiler/blob/main/pytorch-compiler/gpu_probe.py)用于适用设备环境；核心源码为 [C++ codegen](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/codegen/cpp.py)、[Triton codegen](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/codegen/triton.py)、[wrapper](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/codegen/wrapper.py) 和 [compile_fx](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/compile_fx.py)。
