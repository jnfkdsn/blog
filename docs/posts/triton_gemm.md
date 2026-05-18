---
order: 4
title: Triton GEMM 优化
updated: 2026-05-18
tags: [triton, gemm, tensor-core, fp8]
status: draft
---

# Triton GEMM 优化

相关路线：[GPU 编程与算子优化知识地图](/notes/gpu-programming)  
相关基础：[Triton 基础](/notes/triton/triton_basic) / [CUDA 矩阵乘法](/posts/GEMM)

## Naive Triton GEMM


### 完整代码

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    # A 和 B 的 stride（每行有多少个元素）
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 块大小（编译期常量）
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """计算 C = A @ B 的一个 BLOCK_M × BLOCK_N tile"""

    # ======== Step 1: 确定当前 program 负责 C 的哪个 tile ========
    pid_m = tl.program_id(axis=0)  # 行方向的 tile 索引
    pid_n = tl.program_id(axis=1)  # 列方向的 tile 索引

    # ======== Step 2: 计算 A-tile 和 B-tile 的指针偏移 ========
    # A 的行偏移: 当前 tile 在 M 维度的 BLOCK_SIZE_M 行
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # [BLOCK_SIZE_M]
    # B 的列偏移: 当前 tile 在 N 维度的 BLOCK_SIZE_N 列
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # [BLOCK_SIZE_N]
    # K 维度的偏移（初始为 [0, 1, ..., BLOCK_SIZE_K-1]）
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # [BLOCK_SIZE_K]

    # A-tile 的指针矩阵: [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # offs_am[:, None] 是列向量 (BLOCK_M,1), offs_k[None, :] 是行向量 (1,BLOCK_K)
    # 广播得到 (BLOCK_M, BLOCK_K) 的二维偏移
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B-tile 的指针矩阵: [BLOCK_SIZE_K, BLOCK_SIZE_N]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ======== Step 3: 沿 K 维度循环累加 ========
    # 累加器初始化为 0，用 float32 避免精度损失
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # 加载 A-tile [BLOCK_M, BLOCK_K] 和 B-tile [BLOCK_K, BLOCK_N]
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < N) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 核心: tl.dot 做块级矩阵乘，自动映射到 Tensor Core
        accumulator += tl.dot(a, b)

        # 移动 K 维度的指针到下一个 tile
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ======== Step 4: 写出 C-tile ========
    c = accumulator.to(tl.float16)  # 如果输出是 FP16

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton MatMul 的 Python 包装"""
    assert a.shape[1] == b.shape[0], f"维度不匹配: {a.shape} @ {b.shape}"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # grid: C 有多少个 tile
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )
    return c
```

#### 指针算术与广播

```python
# offs_am[:, None] 形状 [BLOCK_M, 1]
# offs_k[None, :]  形状 [1, BLOCK_K]
# 广播后得到 [BLOCK_M, BLOCK_K] 的偏移矩阵
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
```

这和 NumPy 的广播机制完全一样。`a_ptrs` 是一个 `[BLOCK_M, BLOCK_K]` 的指针矩阵，每个元素指向 A 矩阵中对应位置的地址。

#### `tl.dot`：块级矩阵乘

```python
accumulator += tl.dot(a, b)
# a: [BLOCK_M, BLOCK_K]
# b: [BLOCK_K, BLOCK_N]
# tl.dot(a, b): [BLOCK_M, BLOCK_N]
```
`tl.dot` 是 Triton 中**最重要的原语之一**。它做的是块级（tile-level）的矩阵乘法：
- 在 Volta/Ampere/Ada 架构上自动映射到 **Tensor Core**（WMMA / MMA 指令）
- 要求输入维度是 16 的倍数（Tensor Core 的原子操作是 16×16）
- 累加结果通常保持 float32 精度，即使输入是 float16

### naive triton gemm对比CUDA gemm
1. v2分块乘法
```python
for k in range(0, K, BLOCK_SIZE_K):
    a = tl.load(...)
    b = tl.load(...)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```
- 一个program负责一个 BLOCK_SIZE_M × BLOCK_SIZE_N 的输出 tile
- 每次处理的是 `A[BM,BK]` 和 `B[BK,BN]` 子块。

2. v3寄存器
program 内部如何分给线程/warp由编译器决定
结果上通常也会出现“每线程累加多个元素”

3. v4向量化合并访存
```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
```
编译器会基于地址模式自动做：
- 合并访存（coalescing）
- 合适的向量化宽度选择（如 128-bit load/store）

4. 双缓冲
没有显式写双缓冲逻辑

- Triton 后端可能做一定的软件流水化，效果上接近 N-stage buffering





## L2 Cache 优化：Super-Grouping

CUDA/triton默认按行优先顺序调度block/program，
对于横向的四个block：
SM0: Block(bx=0,by=0)  SM1: Block(bx=1,by=0) SM2: Block(bx=2,by=0)  SM3: Block(bx=3,by=0)
在第 t=0 次迭代，4 个 SM 同时加载的 B 片段:
SM0: B[0:8,   0:128]   SM1: B[0:8, 128:256]
SM2: B[0:8, 256:384]   SM3: B[0:8, 384:512]
→ 4 个片段互不重叠，无任何共享
当 N 很大时，处理 `(0,0)` 加载的 B 的第 0 列 tile 在处理 `(0,7)` 时很可能已经被 L2 Cache 驱逐了。而 `(1,0)` 又需要重新从 HBM 加载 B 的第 0-8 列——**B 的同一列被反复从 HBM 加载**，B 在 L2 cache命中率较低，但是A都能命中。

### 解决方案：按小组遍历
将grid分成`GROUP_SIZE_M`行一组，组内先遍历列方向：
```python
GROUP_SUZE_M: tl.constexpr = 8

#计算属于哪个组
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

#组内遍历顺序，将组内一维编号拆成二维坐标
group_inner = pid % num_pid_in_group
pid_m = first_pid_m + (group_inner % group_size_m)
pid_n = group_inner // group_size_m
```

效果
```
对于GROUP_SIZE_M = 4
group_inner=0 -> (m0, n0)
group_inner=1 -> (m1, n0)
group_inner=2 -> (m2, n0)
group_inner=3 -> (m3, n0)
group_inner=4 -> (m0, n1)
group_inner=5 -> (m1, n1)
也就是在同一个n下连续跑多行m
```
1. **1D Grid**：把二维 `(pid_m, pid_n)` 压成一维 `pid`，然后在 kernel 内部用 super-grouping 公式映射回 `(pid_m, pid_n)`。这样可以控制 program 的调度顺序。

## FP16和tensor core

### 浮点数类型

| 格式 | 总位宽 | 符号 / 指数 / 尾数位 | 最大正规数 | 最小正正规数 | 十进制有效位数 | 核心定位 |
|------|--------|-------------------|-----------|-----------|------------|------|
| FP32 | 32 位 | 1/8/23 | ~3.4×10³⁸ | ~1.175×10⁻³⁸ | 6~9 位 | 通用高精度基准 |
| FP16 | 16 位 | 1/5/10 | 65504 | ~6.1×10⁻⁵ | 3~4 位 | 边缘推理、图形渲染 |
| BF16 | 16 位 | 1/8/7 | ~3.4×10³⁸ | ~1.175×10⁻³⁸ | 2~3 位 | 大模型训练 / 推理主力 |
| FP8 E4M3 | 8 位 | 1/4/3 | 448 | 0.015625 | 1~2 位 | 权重 / 激活值计算 |
| FP8 E5M2 | 8 位 | 1/5/2 | 57344 | ~6.1×10⁻⁵ | ~1 位 | 梯度计算 |

BF16：动态范围大（和 FP32 接近），更不容易溢出，训练更稳
FP16：有效精度位更多一点，但动态范围小，训练时更容易 overflow/underflow.

### Triton使用FP16

在 Triton 中使用 Tensor Core 只需要输入是 FP16 且用 `tl.dot`，编译器自动映射：

```python
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# kernel 内部：tl.dot 自动使用 Tensor Core
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # 累加用 FP32
# ...
a_tile = tl.load(a_ptrs, ...)  # 加载的是 fp16
b_tile = tl.load(b_ptrs, ...)  # 加载的是 fp16
accumulator += tl.dot(a_tile, b_tile)  # fp16 × fp16 → fp32 累加
c = accumulator.to(tl.float16)  # 最终结果转回 fp16
```

**混合精度策略**：输入 FP16，累加器 FP32，输出 FP16。
- FP16 减少数据传输量
- FP32 累加避免精度损失（FP16 的有效位数只有 10 bit，大矩阵的累加会丢失精度）

### Tensor Core 对 BLCOK_SIZE的要求
Tensor Core 的最小操作单元是 **16×16×16**：
```
Tensor Core 运算: D[16×16] = A[16×16] × B[16×16] + C[16×16]
```

所以 `BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K` 都必须是 **16 的倍数**。常用配置：
```python
BLOCK_SIZE_M = 128  
BLOCK_SIZE_N = 128  
BLOCK_SIZE_K = 32  
```
BLOCK_SIZE不是16的整倍数会退回CUDA Core

## FP8 GEMM

- 精度更低 → 需要配合 **per-tensor / per-channel scaling**
deepseek v3使用FP8训练
```
标准 FP16 GEMM:
  C = A_fp16 @ B_fp16        (每元素 2 bytes)

DeepSeek FP8 GEMM:
  C = (A_fp8 × scale_A) @ (B_fp8 × scale_B)     (每元素 1 byte)

  其中 scale 是 per-tensor 或 per-block 的缩放因子，
  用于把原始 FP16/FP32 值映射到 FP8 的有限范围内。
```

### triton FP8
```
# FP8 × FP8 → FP32 累加（Tensor Core 原生支持）
accumulator += tl.dot(a, b)
# 应用缩放因子并输出
c = accumulator * scale_a * scale_b
c = c.to(tl.float16)
```

### FP8量化函数

```python
def quantize_to_fp8(x: torch.Tensor):
    """将fp16/fp32tensor 量化为E4M3"""
    # 计算 per-tensor 缩放因子
    amax = x.abs().max().item()
    # FP8 E4M3 的最大值是 240
    scale = 240.0 / amax if amax > 0 else 1.0
    # 量化
    x_scaled = x.float() * scale
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, 1.0 / scale  # 返回量化后的 tensor 和反向缩放因子

# 使用
a_fp16 = torch.randn(M, K, device='cuda', dtype=torch.float16)
b_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)

a_fp8, scale_a = quantize_to_fp8(a_fp16)
b_fp8, scale_b = quantize_to_fp8(b_fp16)

# FP8 GEMM
c = matmul_fp8(a_fp8, b_fp8, scale_a, scale_b)
```
### per tensor和per block scaling
```
Per-tensor scaling（简单但精度较低）:
  整个矩阵共享一个 scale → 异常值会挤压其他值的精度

Per-block scaling（DeepSeek-V3 使用）:
  每 128×128 的 block 有独立的 scale → 更好的精度
  但增加了 scale 的存储和计算开销

Per-channel scaling（AWQ 等量化方法）:
  每个 output channel 一个 scale → 权重量化的常用方案
```
