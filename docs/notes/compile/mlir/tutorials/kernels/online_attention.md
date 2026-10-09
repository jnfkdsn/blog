---
order: 50
title: 在线 Softmax 与分块 Attention
updated: 2026-10-04
---

# 在线 Softmax 与分块 Attention

上一章用完整分数矩阵完成 Attention。现在把 key/value 分成小块，读完一块就更新状态，然后复用同一份临时存储。困难在于后来的块可能产生更大的分数，使前面采用的归一化基准需要改变。

解决这件事以后，分块就不只是把循环边界切小：中间信息的表示也从完整权重变成可以继续合并的状态。本章先推导状态，再看实际 MLIR，最后连接到设备上的 Attention 实现。

## 1. 已处理部分的三个状态

先固定一个 query。设已经处理的有效 key 集合为 J，分数为 s_j，value 向量为 v_j。保存：

```text
m = max(j in J, s_j)
l = sum(j in J, exp(s_j-m))
u = sum(j in J, exp(s_j-m)*v_j)
```

m 是当前指数基准，l 是在此基准下的权重总量，u 是同一基准下尚未归一化的加权向量。若不再有新 key，输出就是 `u/l`。

u 的维度为 P，而不是已经读过的 key 数量。过去每个 key 的单独权重已经被吸收进总量；只要之后能正确改变基准，就不必保存它们。

## 2. 最大值变化后的重新缩放

设新块最大值为 b，更新后的基准 `m'=max(m,b)`。旧元素的权重在新基准下应为：

```text
exp(s_j-m') = exp(s_j-m) * exp(m-m')
```

令 `alpha=exp(m-m')`，则旧 l 和 u 都要乘 alpha。新块中的每个分数直接以 m' 为基准求指数：

```text
p_j = exp(s_j-m')                      j 属于新块
l' = alpha*l + sum(p_j)
u' = alpha*u + sum(p_j*v_j)
```

alpha 同时作用于分子与分母，是同一次坐标基准变化。只缩放 l 或只缩放 u 都会改变最终结果。m' 不小于旧 m，因此 alpha 在有限输入下位于 [0,1]，可能因下溢变成零。

这组等式在实数算术中直接由指数关系得到。浮点实现仍会因乘法、加法和归约顺序不同而产生舍入差异，不能把代数等价理解成逐 bit 相同。

## 3. 两块输入的完整推演

继续使用分数 `[0,0,0,ln(2)]`、value `[10,20,30,40]`，每块最多三个 key。

第一块的最大值为 0，权重均为 1，所以 `m=0, l=3, u=60`。第二块带来更大分数 ln(2)：

| 状态 | 旧值 | 更新动作 | 新值 |
|---|---:|---|---:|
| m | 0 | 与 ln(2) 取最大 | ln(2) |
| alpha | — | exp(0-ln(2)) | 1/2 |
| l | 3 | (1/2)×3 + 1 | 5/2 |
| u | 60 | (1/2)×60 + 1×40 | 70 |

最终 `u/l=70/(5/2)=28`，与完整矩阵基线相同。这个表也说明存储为什么可以复用：第一块的三个分数已经不再需要，只需保留 m/l/u。

若忘记缩放旧 u，却仍正确更新 l，就得到 `(60+40)/(5/2)=40`。这不是较小的数值误差，而是状态契约错误。配套实验会实际构造这个安全的错误版本，让它在此输入上被数值参考拒绝。

## 4. 初始化与空集合

初始状态通常写作 `m=-inf, l=0, u=0`。第一次遇到包含有限分数的非空块时，m' 有限，alpha 为零，旧状态自然不贡献结果。

但如果第一块全无效，就不能计算 `-inf-(-inf)`。本例采用前缀有效长度，只遍历非空有效块：长度为零时完全跳过块循环，输出保持零；其他行每个被处理的块至少含一个有限分数。

任意 mask 的实现需要更明确的“该块是否含有效元素”条件。可以跳过全无效块，或者维护等价的有效状态协议。无论怎样实现，都要同时保护最大值、指数和最终除法，不能只在最终 store 加 mask。

## 5. 从状态递推到 MLIR 循环

固定 query 的 m/l 是两个标量，适合由 `scf.for iter_args` 携带。u 是 P 个元素，教学实现暂存在输出行中，最后再除以 l。核心结构如下，`compute_scores` 等名字是对完整循环的解释性缩写：

```text
out[i,:] = 0
(m,l) = (-inf,0)
for base = 0; base < lengths[i]; base += 3:
    size = min(3, lengths[i]-base)
    scores[0:size] = Q[i,:] · K[base:base+size,:]^T * scale
    new_m = max(m, max(scores))
    alpha = exp(m-new_m)
    weights = exp(scores-new_m)
    new_l = alpha*l + sum(weights)
    out[i,:] = alpha*out[i,:] + weights*V[base:base+size,:]
    (m,l) = (new_m,new_l)
if lengths[i] > 0:
    out[i,:] /= l
```

实际标量状态的更新使用普通 Arith/Math 操作：

```text
%delta = arith.subf %oldmax, %newmax : f32
%alpha = math.exp %delta : f32
%rescaled_sum = arith.mulf %alpha, %oldsum : f32
// 当前块的权重继续累加到 rescaled_sum，得到 newsum。
// 同时对每个输出列缩放旧 u，再加当前块贡献。
scf.yield %newmax, %newsum : f32, f32
```

沿这个结构阅读时，关键不在背接口，而在检查每个旧状态和新贡献是否采用同一个指数基准。输出行在循环期间是未归一化 u，只有函数完成后才是调用者所需的 O。

## 6. 存储变化与执行证据

物化基线分配 `memref<?x?xf32>`，大小为 M×N；在线实现仅分配 `memref<3xf32>`，逐行逐块复用。m/l 是 SSA 状态，u 借用调用者提供的输出行。该 CPU 教学实现仍可能反复读写输出，不能据此认为 u 已自动位于寄存器。

两种实际 MLIR 都经过 lowering、LLVM translation、对象编译和 C 调用。12 组形状/数据覆盖三元素块的余块、完整块、因果前缀、部分/全部空行、大正分数、单 key、空 batch/key/value 维度与较长行，共 24 组结果对照 double 参考。

上面的四 key 例子中，两种 f32 实现均输出 28；用实际 f32 的 ln(2) 输入计算 double 参考得到约 28.000000009。错误的 u 更新则输出 40。分配变化和数值证据分别成立；本实验没有测量吞吐或宣称在线 CPU 实现更快。

## 7. 从一行到设备 Tile

设备实现通常同时处理 BM 个 query 和 BN 个 key。状态随之变成：

```text
m、l：每个 query 各一个，共 BM 个
u：BM×P 的累加块
当前 scores/weights：BM×BN 的临时块
```

Q tile 可在遍历 K/V 块时复用；当前权重与 V 的乘法形成 u 的新贡献。前面的 Matmul 分块、线程分布、局部存储、归约与同步都在这里汇合。BN 增大会改变临时块容量，BM 增大会改变累加器需求，P 较大也会增加资源压力。

在线状态消除了必须物化整个 M×N 权重矩阵的要求，但尚未确定具体硬件布局、矩阵指令、异步搬运、warp/核分工或流水深度。优秀的设备实现需要把这些选择共同设计，不能只把 Python/SCF 循环翻译成一个 kernel 就称为高性能 FlashAttention。

这种按 IO 组织 Attention 的思路可继续阅读 [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)；逐步更新归一化状态的背景见 [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)。本章只实现可验证的前向递推教学路径，完整训练、反向、dropout 和硬件优化留给对应项目。

## 8. 阅读检查与进一步实践

1. 新块的最大值超过旧 m 时，为什么旧 u 和 l 都必须缩放？
2. 没有新块时，m/l/u 各自足以恢复哪些信息，又丢失了哪些信息？
3. 若 mask 不再是有效前缀，哪一步可能出现 `-inf-(-inf)`？
4. 将 query tile 从 1 增为 BM，哪些状态按 BM 增长，哪些与全部 N 无关？

可以在 [17 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/17-matmul-attention)把块大小从 3 改为 4，并预测 N=7 的两块状态；先验证数值，再考虑布局与性能。这个任务足以检查是否掌握递推，不要求先实现完整设备算子。
