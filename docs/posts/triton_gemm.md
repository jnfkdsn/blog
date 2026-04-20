---
order: 4
---

# Triton GEMM 优化



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