---
order: 4
title: TensorIR 的循环、计算块与存储契约
updated: 2026-10-17
---

# TensorIR 的循环、计算块与存储契约

模型图中的矩阵乘法最终必须计算每一个输出元素。普通循环能描述这件事，但如果希望编译器继续重排、分块和并行化，还需要让它知道哪些轴代表不同输出，哪些轴代表同一个输出的累加，以及初始化应发生在哪个层次。

TensorIR 将这些信息放进循环和计算块的结构中。本章先解释一份最小矩阵程序，为下一章实际改变其执行顺序做准备。

## 1. 输出元素与归约

从：

```text
C[i,j] = sum(k = 0..31, A[i,k] * B[k,j])
```

可以区分三种位置：i 选择输出行，j 选择输出列，k 遍历贡献到同一个输出的输入项。

固定 `(i,j)` 后，k 的不同迭代都更新同一个 `C[i,j]`。如果各迭代直接同时写同一地址，就会产生冲突；而不同 `(i,j)` 在无别名等条件下更新不同输出。

因此编译器不能把三层循环都当作相同的可并行循环。归约关系必须保留。

## 2. 完整 TensorIR 函数

下面是后续调度实验使用的完整函数：

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class Matmul:
    @T.prim_func
    def main(A: T.Buffer((32, 32), "float32"),
             B: T.Buffer((32, 32), "float32"),
             C: T.Buffer((32, 32), "float32")):
        T.func_attr({"tir.noalias": True})
        for i, j, k in T.grid(32, 32, 32):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

A、B 是输入 buffer，C 是由调用方提供的输出 buffer。函数通过写 C 交付结果，而不是按普通 Python 函数返回一个新数组。

编译器读取 TVMScript 后得到 TensorIR 对象。`T.grid`、`T.block` 等语法描述 IR 结构，并不在定义这个类时执行全部矩阵运算。

## 3. 循环变量与 Block 轴

外层 i、j、k 表示当前循环遍历到哪里。block 中的 vi、vj、vk 表示计算本身的坐标，并通过绑定关系映射到外层循环。

`"SSR"` 中前两个 S 表示 spatial 轴：选择不同输出位置；最后一个 R 表示 reduction 轴：对同一输出进行归约。

现在两组变量一一相等，似乎有些重复。分块后它们就有不同作用。例如把 j 分成 `jo,ji`，计算坐标可以变为：

```text
vj = jo * 8 + ji
```

矩阵公式仍然使用 `A[vi,vk] * B[vk,vj]`，只是外层执行组织改变了。把计算语义与循环映射分开，便于分析一次调度是否仍然覆盖同一组输出和归约项。

block 还可以包含读写区域信息，例如读取 A/B 对应元素、写入 C 对应位置。它们支持依赖和合法性分析，不是任意附加、可以随意填写的标签。

## 4. 初始化与更新的语义

`T.init()` 位于归约 block 内，但它的语义不是“每次 k 迭代都将 C 清零”。它描述每个输出归约开始时的初始值；后续更新累计全部归约项。

对于一个输出，完整过程应为：

```text
初始化 C=0
k=0：C += A[i,0]*B[0,j]
k=1：C += A[i,1]*B[1,j]
……
k=31：C += A[i,31]*B[31,j]
```

如果清零被错误放进每次 k 更新前，最后只会留下最后一项。这是理解 reduction block 最直接的正确性检查。

保持 init 与 reduction 的关系，可以让调度器在分块、重排时一起维护它们。需要更明确的执行结构时，`decompose_reduction` 可以把初始化与更新拆成独立 block；下一章会观察拆分后的实际程序。

## 5. Buffer、别名与存储位置

`T.Buffer((32,32), "float32")` 说明元素类型、逻辑形状和相应存储描述。索引最终需要映射到地址；更复杂 buffer 还可以有 stride、偏移和存储作用域等信息。

函数中的 `tir.noalias=True` 是一项供优化使用的前提，不能只当成“推荐开关”。调用方必须满足相应参数不发生不允许的存储别名关系，才可以使用依赖这一前提的优化结果。

例如把 C 与 A 指向重叠存储，某次输出写入可能覆盖后续仍要读取的输入。原来基于独立输入输出的循环重排就未必合法。本实验为 A、B、C 分配独立存储。

`T.alloc_buffer` 描述函数内部所需的局部 buffer，但它最终放在哪一层存储、分配多少空间、能否缩减或复用，还会受到调度和 lowering 影响。GPU 上的 shared/local 等作用域也必须与硬件和执行结构匹配。

## 6. PrimFunc 与模型层调用

单独测试这个函数时，调用方准备 A、B、C，再执行编译函数并读取 C。放进 Relax 模型以后，`R.call_tir` 用张量结果的形式连接上下游，再由后续 pass 将输出分配和低层调用协议落实。

两层因此表达不同视角：

```text
Relax 视角：C = matmul(A, B)，C 是一个张量值
PrimFunc 视角：输入 A、B 与输出存储 C，执行写入
```

这与已学过的张量值语义和 destination-passing 实现关系相通。不能把 `call_tir` 的高层结果写法误认为低层数值函数一定会自行分配并返回一个 Python Tensor。

同一个 PrimFunc 可以经过多种合法 schedule，形成不同执行实现；其对外计算契约需要保持。下一章将实际改变 j 方向的分块和向量化，同时保留同一个矩阵乘法结果。

## 7. 推演与实践入口

1. i、j 与 k 为什么具有不同的并行化条件？
2. j 分成两层后，为什么 block 的 vj 仍然表示同一个输出列坐标？
3. `T.init` 如果每次 k 迭代都执行，会得到什么错误结果？
4. 只检查输出形状，能否发现输入输出错误别名造成的破坏？

完整代码和数值实验见 [TVM labs](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)。固定定义可查 [TIR stmt/block](https://github.com/apache/tvm/blob/v0.20.0/include/tvm/tir/stmt.h)、[Buffer](https://github.com/apache/tvm/blob/v0.20.0/include/tvm/tir/buffer.h) 与 [Schedule API](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/tir/schedule/schedule.py)。
