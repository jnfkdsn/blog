---
order: 3
title: 操作实现转换与图融合
updated: 2026-10-17
---

# 操作实现转换与图融合

Relax 已经表达出 `matmul → add → relu`，但这三项高层操作还没有决定怎样执行。编译器需要给它们找到实现，并判断能否把相邻计算放到共同的优化范围中。

本章沿固定版本的实际变换输出，观察三次结构变化：高层操作变成低层函数调用，相关调用被分组，最后组合成一个 TensorIR 函数。把这三件事分清，才能理解“融合后还需要调度”的原因。

## 1. 高层语义与 Legalization

原始函数的一部分是：

```python
mm = R.matmul(x, weight)
added = R.add(mm, bias)
result = R.nn.relu(added)
```

`R.matmul` 告诉编译器做矩阵乘法，没有给出一个可以直接生成 CPU 指令的完整循环体。这里需要一条已注册的规则，将操作映射到 TensorIR 实现及相应调用。

这个过程由本例的 `LegalizeOps` 完成。转换后的入口核心是：

```python
mm = R.call_tir(Module.matmul, (x, weight),
                out_sinfo=R.Tensor((2, 2), "float32"))
added = R.call_tir(Module.add, (mm, bias),
                   out_sinfo=R.Tensor((2, 2), "float32"))
result = R.call_tir(Module.relu, (added,),
                    out_sinfo=R.Tensor((2, 2), "float32"))
```

IRModule 中同时新增三个 `PrimFunc`，分别描述矩阵乘法、广播加法和激活。模型的值关系仍然存在，只是每次计算现在有了具体实现目标。

这里的 legalization 是把某种表示映射到后续可处理的实现，不是宣称已选到最快算法。默认生成的循环可能很朴素，需要继续安排执行顺序和存储。

## 2. 实现结构怎样帮助图层判断

原本图层只知道一个操作的名字；得到 TensorIR 实现后，还可以分析计算和访问模式。例如某个函数逐元素处理输出，另一个函数包含归约，其结果再被逐元素消费者使用。

本例中的 `AnnotateTIROpPattern` 为后续分组提供这类模式信息。消费者据此判断哪些组合符合预设的融合策略。

这条关系值得保留：

```text
低层实现的访问和计算结构
  → 提取可供判断的模式信息
  → 图层据此决定怎样组织相邻调用
```

所以模型优化与张量程序优化并不总是严格单向隔离。某个低层实现提供的信息，反过来可能影响高层分组。

模式分类也不是完整的性能证明。它帮助筛选候选组合；最终效果还取决于目标、数据大小、存储与调度。

## 3. 常量折叠与实现选择的前提

这条 pipeline 中还有 `FoldConstant`。如果一段计算完全依赖已知常量，并满足对应求值条件，编译时就可以算出结果，减少运行时工作。

当前主例的 x、weight、bias 都是运行输入，所以这一步不会把整个函数算成固定的 2×2 常量。即使某个输入在 Python 变量名中叫 weight，也不是它已经被绑定成常量的证据。

进一步的模型流程还可能有推理算子分解、无用绑定消除、布局变换等工作。它们应由实际需求引入：复合操作需要后端可处理形式，未使用结果无需保留，库实现需要特定布局。不是每次编译都按同一清单机械执行全部 pass。

若某个目标有优质的库实现，可能在通用 legalization 之前或合适的位置选择外部调用。过早展开成失去特征的细小操作，反而可能错过高层库模式。具体先后需要查看所选目标 pipeline。

## 4. FuseOps 的分组结果

在本例中，`FuseOps` 将三项相关调用提取到一个带 `Primitive` 属性的 Relax 函数中。此时模块结构是：

```text
main：调用 fused_matmul_add_relu

fused_matmul_add_relu（Relax Function）：
    call_tir matmul
    call_tir add
    call_tir relu

matmul / add / relu：三个 TensorIR PrimFunc
```

这一步明确了一个组合边界：希望将这组调用作为整体继续处理。函数名带 fused，但此时它还是一个组织调用的 Relax 函数，低层仍然有三个独立 PrimFunc。

分组要保留输入输出和所有外部使用。如果模型同时返回 mm 和最终 result，mm 就成为组外也需要的值；分组可能改变，或需要把它保留为额外输出。编译器不能因希望融合而删除可观察结果。

本例只有一条顺序数据流和一个最终输出，因此边界相对简单。更复杂模型中的多个消费者、外部调用和效果都可能限制组合。

## 5. FuseTIR 的函数组合

随后 `FuseTIR` 把分组函数中的 TensorIR 调用组合为一个新的 `PrimFunc`。实际模块最终只保留主入口和融合后的低层函数：

```text
main（Relax）
    → call_tir fused_matmul_add_relu

fused_matmul_add_relu（TensorIR）
    → matmul block
    → add block
    → relu block
```

实现层要把前一次调用的输出 buffer 接成下一次调用的输入，并合并函数参数和局部存储。它维护的是函数边界与 buffer 数据流关系。

本例的实际输出仍含有：

```python
matmul_intermediate = T.alloc_buffer((2, 2))
T_add_intermediate = T.alloc_buffer((2, 2))
```

后面依次执行三组循环。可见“合成一个 PrimFunc”首先让编译器获得共同的变换范围，没有自动把所有计算变成一段没有中间内存的完美内核。

这与 PyTorch Softmax 例子中的观察相通：调用数减少、循环合并、临时存储消除，是可以分别发生的变化。需要看具体输出，不能只根据名字判断。

## 6. 继续调度与正确性边界

有了一个包含三组计算的函数，后续可以尝试把消费者移到生产者的局部循环附近，让中间结果更快被消费，或者将某个逐元素阶段内联。

但矩阵乘法结果只有在对应 k 归约完成后才是最终值。不能为了融合把 ReLU 放进每次累加后执行，否则负的部分和可能提前被截断，改变结果。例如累加项 `[-2,3]` 应得到 1；若每次累加后截断，再加 3，就会得到 3。

这个反例说明：融合不仅是在文本上把几段循环放在一起，还要保持何时产生完整结果、何时消费结果的语义。调度系统中的依赖和归约约束正是为这类问题服务。

最终，本例完成了函数组合并在 CPU 上得到与原始模型一致的 `[[0.5,7.5],[1.5,0]]`。性能与进一步访存优化留到具体调度验证，而不由数值正确直接推出。

## 7. 推演与实践入口

1. LegalizeOps 后，为什么 main 仍是 Relax 函数？
2. FuseOps 和 FuseTIR 分别改变了哪个层次的结构？
3. 为什么“一个融合函数”仍可能拥有多个中间 buffer？
4. 将激活移入矩阵归约内部，为什么通常改变语义？

实验见 [TVM labs 的 graph](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)，每个阶段保存完整 IR。固定实现可核对 [LegalizeOps](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/legalize_ops.cc)、[FuseOps](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/fuse_ops.cc) 和 [FuseTIR](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/fuse_tir.cc)。
