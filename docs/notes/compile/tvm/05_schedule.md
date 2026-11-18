---
order: 5
title: TensorIR 调度：程序变换、计算位置与存储
updated: 2026-10-30
---

# TensorIR 调度：程序变换、计算位置与存储

第四章的 Matmul 已经是一份可编译的程序。TVM 可以直接编译它，但这不表示它已经具有理想的访问顺序。现在希望同时推进相邻的八个输出列，复用一个 A 元素，并让 B 连续读取。谁作出这个选择，谁负责把选择落实成新程序？

本章采用手工策略：我们选择变换，TVM 修改 TensorIR。主线先走通“原程序 → 调度程序 → 新程序 → 编译”，再增加缓存阶段，理解计算位置和存储怎样一起变化。矩阵分块是观察这项机制的例子，不是所有 TVM 编译必经的默认策略。

## 1. 计算程序与变换程序

第四章的 `Matmul` 描述将来怎样读取 A、B 并写 C。下面这段 Python 则以该 IR 为输入，产生另一份 IR：

```python
sch = tir.Schedule(Matmul)
block = sch.get_block("C")
i, j, k = sch.get_loops(block)
jo, ji = sch.split(j, factors=[None, 8])
sch.reorder(i, jo, k, ji)
sch.vectorize(ji)
sch.decompose_reduction(block, k)
optimized = sch.mod
```

`block`、`i`、`j`、`k` 是调度对象中的引用句柄，既不是矩阵数据，也不是目标程序运行时的整数。`split` 真正执行时，正在运行的是编译器：它创建循环，替换坐标表达式，并维护相关块的绑定。调用完成后，`sch.mod` 已经是变换后的 IRModule。

编译和数值执行发生在后面：

```python
lib = tvm.tir.build(sch.mod, target="llvm")
lib(a_nd, b_nd, c_nd)
```

这里三个 NDArray 分别提供输入 A、B 和输出 C 的存储。如果仍把原来的 `Matmul` 传给 build，这次手工调度就没有应用到那份编译输入上。

这种设计将计算与优化策略分开。同一份正确计算可以尝试几套变换，无需为每个候选重新手写完整循环。代价是调度程序依赖输入 IR 的结构：找不到名为 C 的块、块已被拆分，或循环不满足某个 primitive 的前提，都需要调整策略。

## 2. 从访存需求到分块和重排

原程序按 i、j、k 遍历，每次算完一个 C[i,j]。行主序矩阵的访问为：

```text
固定 i、j，推进 k：
A[i,0]、A[i,1]、A[i,2]……      偏移每次增加1
B[0,j]、B[1,j]、B[2,j]……      偏移每次增加32
```

相邻列又需要同一行 A。我们选择先处理八列，让固定 A[i,k] 对应 B[k,j] 的八个相邻元素。为此拆分列坐标：

```text
j = 8*jo + ji
jo = 0..3，ji = 0..7
```

`split(j, factors=[None, 8])` 推导外层范围为4，将原来的 j 引用改为 `8*jo+ji`。此时顺序还是 i、jo、ji、k；每个输出仍沿 k 算完。再执行 `reorder(i, jo, k, ji)`，才能同时推进八列。

固定 i=0、jo=1、k=2，新内层的工作是：

```text
共同读取 A[0,2]
分别读取 B[2,8] … B[2,15]
分别更新 C[0,8] … C[0,15]
```

每个 C 元素仍按 k=0、1、…、31 接收贡献，只是交错更新了其他列。没有减少乘法数量，改善的是访问组织和潜在复用。TVM 的工作还包括检查重排的结构条件，不能仅靠“公式相同”就对任意含依赖的循环使用同样操作。

## 3. 向量化、初始化与尾部

八个 ji 更新八个输出，所以适合空间向量化：广播一个 A 元素，与 B 的连续八个元素相乘，再更新八个累计值。`vectorize(ji)` 标记这种组织；它不是创建八个 CPU 线程，也不保证最终恰好是一条八元素机器指令。

`decompose_reduction(block, k)` 将归约初始化拆出，放到指定 k 循环之前。上述实际调度输出的核心如下，省略块轴和读写声明：

```python
for i, jo in T.grid(32, 4):
    for ji_init in T.vectorized(8):
        C[i, jo * 8 + ji_init] = T.float32(0)
    for k in range(32):
        for ji in T.vectorized(8):
            C[i, jo * 8 + ji] += A[i, k] * B[k, jo * 8 + ji]
```

每组输出清零一次，再接收32轮贡献。实际脚本先 vectorize 再拆初始化，是为了在该固定版本上满足 primitive 的结构前提。数学上可以设计的终态，不保证任意接口次序都能直接得到。

如果把块宽改为7，最后一组 jo=4 名义上覆盖 j=28..34。默认 split 会为未整除部分生成相应谓词，限制初始化和更新中的有效访问：

```python
T.where(jo * 7 + ji < 32)
```

合法 ji 只有0..3。相比之下，直接把归约 k 标成普通 parallel，会让多个任务读改写相同的 C 元素；本例的调度检查拒绝该操作。归约并行需要另外构造部分和与合并，不能通过修改标签消除依赖。

## 4. CacheRead：创建搬运阶段

循环顺序改变了，数据放在哪里还没有明确改变。现在增加一个单独实验：让 C 的计算通过 A_cache 读取 A。

从原 Matmul 新建 schedule，完成 split 和 reorder 后执行：

```python
a_cache = sch.cache_read(block, 0, "global")
```

这里 `block` 是 C 计算块，索引0选择其读集合中的 A，返回值指向新建的复制块。这个例子中的读集合为 A、B；不能把数字0当成所有程序中固定的 A。

生成结构可以整理为：

```text
先将 A 的所需区域复制到 A_global
C 的计算改读 A_global，仍从 B 读取另一项
```

这不只是附加一个“更快”标签。TVM 建立新 buffer、新 producer，并重定向原计算的读取。当前选择 global 是为了在 CPU 上观察机制；它没有创建 GPU shared memory，也不保证加速。若整块复制之后仍只用一次，新增搬运可能更慢。

## 5. ComputeAt：根据消费者推导生产范围

接着把这个复制块移入 k 循环：

```python
sch.compute_at(a_cache, k)
```

消费者的当前循环是 i、jo、k、ji。固定 i、jo、k 时，八次 ji 都只需要 A[i,k]，所以新 producer 在这个位置需要生成的区域只有一个元素。实际 IR 的核心为：

```text
for i, jo, k:
    A_global[i,k] = A[i,k]
    for ji in 0..7:
        C[i,8*jo+ji] += A_global[i,k] * B[k,8*jo+ji]
```

初始化仍由归约结构表达，上面的伪代码只展示搬运与更新的位置。原始 cache buffer 声明此时仍可能是32×32；生产范围变小、分配位置调整和存储压缩并不是同一个动作，后续 lowering 还要处理存储。

`compute_at` 的关键工作，是移动 producer 后根据消费者需要重新推导其执行范围，而不是单纯给一个已有 for 循环换缩进。它也不限于复制块：普通中间计算可以成为 producer，但移动必须满足依赖、作用域、消费者位置等约束。

位置还有代价。上述方案对每个 jo 重复读取同一 A[i,k]，一行共有4组；如果在更外侧准备整行 A，可能减少重复生产，却扩大同时存活的区域。在更内侧准备单个值，工作集小，但生产次数可能增加。选择必须同时看“每次产生多少”与“总共产生几次”。

## 6. CacheWrite：局部累加与写回

读取缓存解决输入去向；写缓存可以改变累计结果的去向。在另一份完成 split/reorder 的 schedule 中执行：

```python
writeback = sch.cache_write(block, 0, "local")
sch.reverse_compute_at(writeback, jo)
```

`cache_write` 将 C 块的输出改写到 C_local，并创建把结果写回 C 的块；返回句柄指向写回块。`reverse_compute_at` 将这个消费者放到指定循环下，并推导应消费的区域。实际结构整理为：

```text
for i, jo:
    初始化当前八列的 C_local
    for k:
        for ji:
            C_local[i,8*jo+ji] += A[i,k] * B[k,8*jo+ji]
    for ji:
        C[i,8*jo+ji] = C_local[i,8*jo+ji]
```

相比每轮 k 都更新最终 C，现在在局部存储中累计，全部完成后写回。`local` 是存储作用域，最终是否放进寄存器、需要多大实际分配，仍由目标和后续 lowering 决定。不能只见名字就断言“零访存”。

输入缓存与输出缓存一起解释了真正的调度决策：哪些值共同消费输入、在哪个循环层准备它们、累计值存活多久、何时把结果交出去。只列 split、cache、compute_at 的接口职责看不出这些关系。

## 7. 多级分块与硬件映射

更大的 Matmul 可以把 i、j、k 都分成外层块和内层坐标。一个候选的意图是：

```text
一个输出块 C_tile[Bi,Bj]
    沿归约块 ko 推进：
        准备 A_tile[Bi,Bk] 与 B_tile[Bk,Bj]
        用这两块输入更新 C_tile
    将 C_tile 写回
```

例如 Bi=Bj=8、Bk=4 时，一轮需要32个 A 元素和32个 B 元素；A 的每个元素服务8列，B 的每个元素服务8行。相比只分 j，它多利用了行方向的复用，但也增加同时需要保留的输入和64个累计结果。这是设计推演，本章的 CPU 实测仍是前面的有限变体。

GPU 上还要决定输出块属于哪个 thread block，哪些循环绑定到线程，输入 tile 是否进 shared memory，以及合作搬运后的同步。`bind` 表达执行映射，cache scope 表达存储意图，二者需要共同满足硬件和依赖条件。`tensorize` 则可以把满足描述要求的计算块替换成已注册的张量指令实现；匹配形状、布局与约束比改一个名字复杂。

这些能力解释 TensorIR 为何保留计算块及读写信息。当前先理解它们解决的选择问题；GPU/NPU映射和张量指令的性能要在后续真实目标上验证。

## 8. 调度记录、验证与选择

`sch.trace` 记录构成该调度的变换和决定，便于重放和构造其他候选。它使“找到一份快的程序”还能回答“这份程序怎样从原程序得到”。更换输入 IR 后，原 trace 是否仍适用需要重新检查。

本章 CPU 参考已执行 tile=4/8/7、读取缓存移入 k、局部输出缓存等版本，并与矩阵乘法参考比较；另有直接并行归约的拒绝例。缓存变体用于验证结构和正确性，没有据此声称比原版本快。

选择块宽、计算位置与存储 scope 的是策略；实施选择的是调度原语。下一章让规则和搜索系统参与选择。第七章继续观察 sch.mod 怎样经过实际 TIR passes，变成 LLVM IR 和可调用代码。

1. `compute_at(a_cache,k)` 后，为什么每个 i、jo、k 只需复制一个 A 元素？如果 producer 移到不同层，生产范围和总次数会怎样变化？
2. cache_write 返回的块负责哪一步？把写回放在归约未完成的位置会破坏什么？
3. 已经取得 sch.mod，但构建时仍传原 Matmul，哪一套循环会进入编译？正确性通过为什么不能证明缓存方案更快？

固定实现可从 [Schedule API](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/tir/schedule/schedule.py) 的 compute_at/cache_read 等入口，进入 [调度原语实现](https://github.com/apache/tvm/tree/v0.20.0/src/tir/schedule/primitive)。[配套观察](https://github.com/jnfkdsn/aicompiler/blob/main/tvm-e2e/design_probe.py)保存缓存前后 IR 和实际 lowering；现有调度练习继续保留，正文不依赖运行脚本才能理解。
