---
order: 5
title: 低精度数值与混合精度计算
updated: 2026-07-16
tags: [cuda, fp16, bf16, tf32, mixed-precision, tensor-core]
status: draft
---

# 低精度数值与混合精度计算

相关路线：[CUDA 学习笔记](/notes/cuda/) / [GPU 编程与算子优化知识地图](/notes/gpu-programming) / 后续内容：[Tensor Core 编程](/notes/cuda/tensor_core) / [Tensor Core GEMM 实践](/posts/tensor_core_gemm) / 相关应用：[量化笔记](/notes/infer/quantization)

低精度计算的目标不只是“用更少的 bit 表示数”。在 GPU 上，它同时影响：

- 一个数占用多少存储空间和显存带宽。
- Tensor Core 能选择哪种矩阵乘指令。
- 乘法输入、累加器和输出分别使用什么类型。
- 数值误差、上溢、下溢和舍入如何传播。
- 一个 kernel 最终是 memory-bound 还是 compute-bound。

学习 Tensor Core 前，至少要区分三个概念：

```text
storage type：数据在 global/shared memory 中如何存储
input type：乘法器实际接收的输入格式
accumulator type：部分和用什么格式累加
```

例如常见的混合精度 GEMM 是：

```text
FP16 A × FP16 B -> FP32 accumulator -> FP16 或 FP32 output
```

输入是 FP16，不代表所有中间计算都只能保留 FP16 精度。

## 浮点数表示

一个二进制浮点数可以抽象为：

$$
x = (-1)^s \times 2^e \times (1.f)
$$

其中：

- sign 决定正负。
- exponent 决定可表示的数量级，即动态范围。
- fraction/mantissa 决定同一个数量级内能区分多细的数，即有效精度。

指数位更多，通常表示范围更大；尾数位更多，通常相邻可表示数之间的间隔更小。

## 常见格式

| 格式 | 符号位 | 指数位 | 显式尾数位 | 主要特点 |
|---|---:|---:|---:|---|
| FP32 | 1 | 8 | 23 | 范围和精度都较高，普通 CUDA Core 计算基准 |
| TF32 | 1 | 8 | 10 | FP32 的范围、接近 FP16 的有效精度，主要作为 Tensor Core 计算格式 |
| BF16 | 1 | 8 | 7 | 范围接近 FP32，精度低于 FP16 |
| FP16 | 1 | 5 | 10 | 精度较好但范围较小，最大有限值为 65504 |
| FP8 | 1 | 4/5 | 3/2 | 更低存储与带宽，通常需要 scale；3090 没有原生 FP8 Tensor Core 路径 |
| INT8 | 1 | — | — | 定点整数，需要 scale/zero point 解释真实值 |

### FP16

FP16 的尾数位多于 BF16，因此在数值处于可表示范围内时通常更精细；但只有 5 个指数位，较大的值容易上溢，较小的值容易下溢。

典型用途：

- FP16 权重和激活。
- Tensor Core GEMM 输入。
- FP32 累加后转换为 FP16 输出。

需要警惕：

- 大 reduction 直接用 FP16 累加。
- `exp`、方差、归一化等对范围敏感的运算。
- 先累加再缩放时中间值超过 FP16 范围。

### BF16

BF16 保留与 FP32 相同宽度的指数，因此动态范围接近 FP32，但尾数只有 7 位。它通常比 FP16 更不容易上溢，但单步舍入误差更大。

直觉上：

```text
FP16：范围小一些，细节多一些
BF16：范围大一些，细节少一些
```

BF16 仍然常配合 FP32 accumulator，而不是用 BF16 保存很长的部分和。

### TF32

TF32 主要是 NVIDIA Ampere Tensor Core 的计算格式，不应把它理解成普通的 19-bit 内存类型。应用通常仍以 FP32 存储 A/B；Tensor Core 计算路径读取 FP32 输入的高有效位，以 TF32 精度执行乘法，再使用 FP32 累加。

因此对 cuBLAS 做对比实验时，必须明确：

- FP32 输入是否允许使用 TF32 Tensor Core。
- compute type 是严格 FP32，还是允许 TF32。
- 参考结果用什么精度计算。

否则两个都写着“FP32 GEMM”的实验可能实际走了不同硬件路径。

### FP8 与 INT8

FP8 和 INT8 都需要回答 scale 的粒度：

```text
per-tensor
per-channel
per-token
per-group / per-block
```

区别在于：

- FP8 仍然有指数和尾数，能在格式内部表达一部分动态范围。
- INT8 本身只有整数值，真实值由 `q * scale` 或 `(q - zero_point) * scale` 恢复。
- 低比特 GEMM 的收益取决于硬件是否原生支持，以及量化、反量化能否和 mainloop/epilogue 融合。

RTX 3090 的当前主线应先学 FP16/BF16/TF32 Tensor Core。FP8 可以作为理解量化系统的后续内容，不要把 Hopper/Blackwell 的原生 FP8、block scaling 能力直接套到 Ampere 上。

## 混合精度为什么需要高精度累加

考虑点积：

$$
y = \sum_{k=0}^{K-1} a_k b_k
$$

每一步都包含乘法和累加。即使单个 $a_kb_k$ 误差不大，长 reduction 中误差也会持续传播；浮点加法不满足严格结合律：

$$
(a+b)+c \neq a+(b+c)
$$

因此常见设计是：

```text
低精度输入：降低带宽、提高矩阵乘吞吐
高精度累加：降低长点积的误差和溢出风险
按需要输出：决定下一层的存储与带宽
```

这也是为什么“输入是 FP16”不能直接推出“结果误差一定很大”。误差还取决于：

- accumulator 类型。
- K 的长度和数据分布。
- reduction 顺序。
- 是否存在大幅值 outlier。
- epilogue 中是否继续执行 bias、activation、cast。

## 舍入、上溢和下溢

### 舍入

实数转换到有限格式时，会被映射到附近的可表示数。格式越窄，相邻可表示数之间的间隔越大。

实验中不要只验证 `allclose` 是否通过，还应记录：

```text
max_abs_error = max(|reference - output|)
max_rel_error = max(|reference - output| / max(|reference|, eps))
mean_abs_error
```

相对误差在参考值接近 0 时会被放大，所以要同时看绝对误差。

### 上溢

结果绝对值超过最大有限值时，可能变成 `inf`。FP16 比 BF16 更容易出现这一问题。

### 下溢

绝对值过小时，结果可能进入 subnormal 区域或直接变成 0。对 softmax、归一化和梯度计算，过早变成 0 可能改变最终结果。

## 性能收益从哪里来

低精度可能同时提高三个上限：

1. **显存容量**：同样元素数量占用更少字节。
2. **显存带宽**：一次 memory transaction 能搬更多元素。
3. **计算吞吐**：Tensor Core 对特定低精度类型提供更高的 MMA 吞吐。

但低精度不保证端到端加速：

- kernel 如果由 launch latency 主导，减少字节数帮助有限。
- decode 小 batch GEMM 可能更接近 GEMV，主要受权重读取带宽限制。
- 量化和反量化若没有融合，额外 kernel 和中间 tensor 会抵消收益。
- shape、alignment 或 layout 不满足要求时，library 可能选择非 Tensor Core 路径。

## 一组建议实验

### 实验一：格式转换误差

生成三类 FP32 数据：

- `uniform(-1, 1)`：普通范围。
- 对数均匀分布：覆盖多个数量级。
- 包含少量大 outlier：模拟激活异常值。

分别转换成 FP16、BF16，再转回 FP32，比较绝对误差、相对误差、`inf` 和 0 的数量。

### 实验二：不同累加器的点积

固定 FP16 输入，比较：

```text
FP16 multiply + FP16 accumulate
FP16 multiply + FP32 accumulate
FP64 CPU reference
```

逐步增大 K，观察误差如何随 reduction 长度变化。

### 实验三：GEMM 数值与性能

比较：

```text
严格 FP32 CUDA/cuBLAS
允许 TF32 的 FP32 cuBLAS
FP16 input + FP32 accumulate
BF16 input + FP32 accumulate
```

每种模式同时记录延迟、GFLOPS 和误差，避免把“最快”直接等价为“最好”。

## 学完后的检查问题

- storage type、input type、accumulator type 有什么区别？
- 为什么 BF16 范围接近 FP32，但精度低于 FP16？
- TF32 为什么通常不是用户直接分配的内存数据类型？
- 为什么 FP16 输入经常配 FP32 accumulator？
- 低精度 GEMM 在 decode 小 batch 下为什么仍可能受显存带宽限制？
- 量化的 scale 粒度如何影响精度、metadata 和 kernel 实现？

## 参考资料

- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [CUDA Programming Guide: Floating-Point Computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/floating-point.html)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
