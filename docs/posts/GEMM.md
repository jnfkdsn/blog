---
order: 3
title: CUDA 矩阵乘法
updated: 2026-05-18
tags: [cuda, gemm, tiling, tensor-core]
status: draft
---

# CUDA 矩阵乘法

相关路线：[GPU 编程与算子优化知识地图](/notes/gpu-programming)  
相关基础：[Roofline 分析](/notes/cuda/roofline) / [Triton GEMM 优化](/posts/triton_gemm)

## Roofline分析

$$C_{M \times N} = A_{M \times K} \times B_{K \times N}$$

$$c_{ij} = \sum_{k=0}^{K-1} a_{ik} \cdot b_{kj}$$

### FLOPs
每个$c_{ij}$需要k次乘法+k-1次加法约2k次浮点数操作：
$$\text{FLOPs} = 2 \cdot M \cdot N \cdot K $$

### 算术强度(AI)
$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2MNK}{(MK + KN + MN) \times 4}$$
对于方阵 $M = N = K = n$：
$$\text{AI} = \frac{2n^3}{3n^2 \times 4} = \frac{n}{6} \quad \text{(float32)}$$

```
算术强度随 n 增长:

  n      FLOPs          Bytes (A+B+C)    AI       区域
  ──────────────────────────────────────────────────
  64     524K           48 KB            10.7     Memory-bound
  256    33.6M          768 KB           42.7     Compute-bound
  1024   2.15G          12 MB            170.7    Compute-bound
  4096   137.4G         192 MB           682.7    Compute-bound
RTX 3090:
  FP32 Peak Compute:       35.6 TFLOPS
  GDDR6X Bandwidth:        936 GB/s
  SMEM:					   17.7 TB/s  
  L2 Cache:                6 MB
Ridge Point = 35.6 TFLOPS / 936 GB/s ≈ 38 FLOPs/Byte
对于较大的n，是严重compute-bound
```


## CUDA V1
每个线程算一个元素
```cpp
__global__ void matmul_v1(const float* A, const float* B, float* C,int M, int N, int K){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row<M&&col<N){
        float sum = 0.0f;
        for(int k=0; k<K;k++){
            sum+=A[row*K+k]*B[k*N+col];
        }
        C[row*M+col] = sum;
    }
}
dim3 block(16,16);
dim3 grid((N+15)/16,(M+15)/16);
matmul_v1<<<grid,block>>>(A,B,C,M,N,K);
```

### 性能分析
CUDA 线程先按 **行优先（row-major）** 线性化，再每 32 个分一个 warp。公式是：
$$\text{linearIdx} = \text{threadIdx.y} \times \text{blockDim.x} + \text{threadIdx.x}$$
```
同一warp内线程读取B时，若blockDim.x<32,warp跨多行访存就不是coalescing，所以blockDim.x一般是32的倍数
V1的HBM访问量(M=N=K=1024)
每个线程各读取一行一列：4KB+4KB
写入一个元素：4B
共1M个线程
总读取量：8GB >> 4MB
AI(HBM_V1) = FLOPs/Bytes = 2N^3/8N^3 = 0.25
严重memory_bound

```

和cublas相比性能差距在10倍以上
```
===== M = N = K = 512 =====
  V1 Naive            :    0.248 ms  |    1082.3 GFLOPS
  cuBLAS SGEMM        :    0.022 ms  |   12025.0 GFLOPS
===== M = N = K = 4096 =====
  V1 Naive            :  128.865 ms  |    1066.5 GFLOPS
  cuBLAS SGEMM        :   11.162 ms  |   12313.6 GFLOPS
```

## CUDA V2 ： block tile
分块乘法:让一个线程块负责一个tile，加载对应的A,B tile到shared memory，然后从shared memory中读取做计算
```
分块矩阵乘法:
  C[BM×BN] = Σ A[BM×BK] × B[BK×BN]   (对 K 维度分块求和)
  
  Step 0: 加载 A₀ 和 B₀ 到 SMEM → 计算局部乘积
  Step 1: 加载 A₁ 和 B₁ 到 SMEM → 累加局部乘积
  Step 2: 加载 A₂ 和 B₂ 到 SMEM → 累加 → 写出最终结果

  每个 A tile 被 block 内的 BN 个列的线程共享
  → 复用次数从 N (全局) 降到了 BN (block 内)
```

流量分析：
$$\text{V1 HBM} = M \cdot N \cdot K \cdot 2 \times 4 \text{ bytes}$$

$$\text{V2 HBM} = \frac{K}{BK} \times (BM \times BK + BK \times BN) \times \frac{M \cdot N}{BM \cdot BN} \times 4 \text{ bytes}$$

对于方阵 $n = M = N = K$，$BM = BN = BK = T$：

$$\text{V2 HBM} = \frac{n}{T} \times 2T^2 \times \frac{n^2}{T^2} \times 4 = \frac{8n^3}{T}$$

$$\text{V1 HBM} = 8n^3$$

$$\text{加速比} = T \quad (\text{例如 } T = 32 \text{ 时，HBM 流量降低 32x})$$


### 性能分析
```
===== M = N = K = 512 =====
  V1 Naive            :    0.134 ms  |    1996.5 GFLOPS
  V2 Shared Tiling    :    0.125 ms  |    2140.8 GFLOPS
  cuBLAS SGEMM        :    0.025 ms  |   10787.8 GFLOPS
===== M = N = K = 1024 =====
  V1 Naive            :    0.992 ms  |    2164.8 GFLOPS
  V2 Shared Tiling    :    0.811 ms  |    2646.7 GFLOPS
  cuBLAS SGEMM        :    0.117 ms  |   18371.9 GFLOPS
===== M = N = K = 4096 =====
  V1 Naive            :   61.314 ms  |    2241.5 GFLOPS
  V2 Shared Tiling    :   45.596 ms  |    3014.3 GFLOPS
  cuBLAS SGEMM        :    6.007 ms  |   22879.3 GFLOPS
```
在N较大时性能提升更大，N较小时HBM读取较少，并且引入了额外的shared memory的读取，提升不明显
```
对于N=1024,B=32

  HBM 读取: 8n³ / T = 8 × 1G / 32 = 256 MB (vs V1 的 8 GB → 32x 改善)
  SMEM 读取: 每个 tile : 2 × BK = 64 次/线程 (As + Bs 各 BK 次) × K/BK = 2K = 2048 次/线程
  瓶颈分析:
    HBM 流量大幅减少
    每个线程只算 1 个 C 元素: 2K FLOPs
	SMEM 读取：2K
	AI(SMEM) = 0.25 FLOPs/smem Bytes
	AI(HBM_V2) = 2N^3/(8N^3/T) = 8FLOPs
	仍是memory-bound
	每次从 SMEM 读一个数，只用于 1 次运算就丢掉了，没有复用。
```

## CUDA V3 ：thread tile
V2 的瓶颈：每个线程只算 1 个 C 元素 → 计算/访存比太低。

**核心思想**：让每个线程算 $TM \times TN$ 个 C 元素。这些元素共享 A 的同一行和 B 的同一列（在 tile 内），从 SMEM 读一次 A 值可以被 TN 次乘法复用。

```
C 矩阵 [M × N]
│
├── Grid 维度 (blockIdx):
│     gridDim.x = N / BN,  gridDim.y = M / BM
│     → 每个 block 负责 C 的一个 BM×BN 的子块
│
├── Block 维度 (blockDim / threadIdx):
│     blockDim.x = BN/TN = 16
│     blockDim.y = BM/TM = 16
│     → 一个 block 里有 16×16 = 256 个线程
│
└── Thread 维度 (寄存器):
      每个线程负责 TM×TN = 8×8 = 64 个 C 元素
```


---
**V3 每个线程在做什么**
Block 负责 C 的一个 128×128 子块。256 个线程把这个子块均匀分成 16×16 = 256 份，每份 8×8。
```
Thread(ty, tx) 负责 C 中的:
  行: [ty*8 .. ty*8+7]
  列: [tx*8 .. tx*8+7]

thread_row = ty * 8
thread_col = tx * 8
```
对 K 维度每次迭代 BK=8 列：
```
for t in 0..K/8:
  1. 所有线程协作把 A[128行×8列] 加载到 As[128][8]
  2. 所有线程协作把 B[8行×128列] 加载到 Bs[8][128]
  3. 每个线程用自己的 8×8 区域做计算:
     for k in 0..7:
       regA[0..7] = As[thread_row + 0..7][k]   ← 从 As 取 8 个行值
       regB[0..7] = Bs[k][thread_col + 0..7]   ← 从 Bs 取 8 个列值
       做外积: regC[m][n] += regA[m] * regB[n]  ← 8×8=64 次 FMA
```


### 资源分析
``` cpp
寄存器使用:
  regC[8][8] = 64 个寄存器
  regA[8]    = 8 个寄存器
  regB[8]    = 8 个寄存器
  其他临时变量 ≈ 16 个寄存器
  总计: 96 个寄存器/线程

  SM 8.6: 65536 个寄存器/SM
  → 最大 active 线程 = 65536 / 96 ≈ 682 → 实际 640 (向下取整到 warp 的倍数)
  → 256 线程/block → 可以有 2 个 active block/SM
  → Occupancy ≈ 640 / 1536 ≈ 42%

  虽然 occupancy 不高，但 register tiling 提供的高计算密度
  弥补了低 occupancy 带来的延迟隐藏损失。

SMEM 使用:
  As[128][8] + Bs[8][128] = 1024 + 1024 = 2048 float = 8 KB
  → SMEM 不是瓶颈
```

### As 读取的coalescing 问题：
```
  内存地址分布 (K=1024, 每格=4 bytes):

  Thread  0-7:  地址 base+0, +4, +8, ..., +28          ← Cache line 1
  Thread  8-15: 地址 base+4096, +4100, ..., +4124       ← Cache line 2 (跨越 K=1024*4=4KB)
  Thread 16-23: 地址 base+8192, ...                     ← Cache line 3
  Thread 24-31: 地址 base+12288, ...                    ← Cache line 4

  每个 cache line 128 bytes = 32 个 float
  但每次只用 8 个 float → 利用率 8/32 = 25%

  可以通过转置 As 为 As[BK][BM]解决？
  
```
```
对于M=N=K=1024,BM=BN=128,BK=8,TM=TN=8

  HBM 读取: n^3/16 = 64MB
  AI = 32 FLOPs/Byte < 38
  仍是memory-bound

```

### As 和Bs Bank conflict

**SMEM Bank Conflict**：V3 的 Bs 存在 4-way bank conflict，As 存在 2-way conflict（详见下方分析）。
Warp 0 = linearIdx 0..31，包含 ty=0,tx=0..15 和 ty=1,tx=0..15：
```
对于Bs
thread_col = tx * 8，只跟 tx 有关，warp 0 的 tx 只有 0..15
读 Bs[k][thread_col + n] 时（比如 n=0）：
  tx=0  → col=0   → bank 0
  tx=1  → col=8   → bank 8
  tx=2  → col=16  → bank 16
  tx=3  → col=24  → bank 24
  tx=4  → col=32  → bank 0  ← 不同地址，同 bank → conflict!
  tx=5  → col=40  → bank 8
  tx=6  → col=48  → bank 16
  tx=7  → col=56  → bank 24
  tx=8  → col=64  → bank 0  ← conflict!
  tx=9  → col=72  → bank 8
  tx=10 → col=80  → bank 16
  tx=11 → col=88  → bank 24
  tx=12 → col=96  → bank 0  ← conflict!
  tx=13 → col=104 → bank 8
  tx=14 → col=112 → bank 16
  tx=15 → col=120 → bank 24
  ty=1 的 16 个线程 → 读同地址 → broadcast，无冲突
bank 0 被 tx=0,4,8,12 四个不同地址访问 → 4-way conflict

对于As
ty=0, tx=0..15 (T0..T15):  thread_row=0 → 全部读 As[m][k]   → 16线程同地址 → broadcast
ty=1, tx=0..15 (T16..T31): thread_row=8 → 全部读 As[8+m][k] → 16线程同地址 → broadcast

As[row][col] 的 bank = (row * 8 + col) % 32

As[m][k]:     bank = (m*8 + k) % 32
As[8+m][k]:   bank = ((8+m)*8 + k) % 32 = (8m + 64 + k) % 32
                                          = (8m + k) % 32   ← 64%32=0，抵消了！

→ 两个地址在同一个 bank → 2-way conflict
```


### 性能分析：
```
===== M = N = K = 512 =====
  V1 Naive            :    0.134 ms  |    2004.9 GFLOPS
  V2 Shared Tiling    :    0.124 ms  |    2160.2 GFLOPS
  V3 Reg Tile         :    0.121 ms  |    2212.2 GFLOPS
  cuBLAS SGEMM        :    0.025 ms  |   10721.6 GFLOPS

===== M = N = K = 1024 =====
  V1 Naive            :    0.992 ms  |    2164.7 GFLOPS
  V2 Shared Tiling    :    0.811 ms  |    2646.9 GFLOPS
  V3 Reg Tile         :    0.294 ms  |    7307.1 GFLOPS
  cuBLAS SGEMM        :    0.117 ms  |   18347.8 GFLOPS

===== M = N = K = 4096 =====
  V1 Naive            :   61.804 ms  |    2223.8 GFLOPS
  V2 Shared Tiling    :   46.093 ms  |    2981.8 GFLOPS
  V3 Reg Tile         :    9.639 ms  |   14258.8 GFLOPS
  cuBLAS SGEMM        :    5.984 ms  |   22967.5 GFLOPS
```
n较小时无法填满SM，是occupancy-bound


## CUDA v4 : 向量化加载
使用float4读取A,B数据，减少load指令数量，
```cpp
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matmul_v4(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int threadRow = ty * TM;
    int threadCol = tx * TN;

    float regC[TM][TN] = {0.0f};
    float regA[TM], regB[TN];

    int numThreads = blockDim.x * blockDim.y;
    int linearIdx = ty * blockDim.x + tx;

    for (int t = 0; t < (K + BK - 1) / BK; t++) {
        // --- 向量化加载 A tile ---
        // A tile: BM × BK = 128 × 8 = 1024 float = 256 个 float4
        // 256 线程，每线程加载 1 个 float4
        {
            int loadIdx = linearIdx;  // 每线程 1 个 float4
            int loadRow = loadIdx / (BK / 4);  // BK/4 = 2 个 float4 每行
            int loadCol = (loadIdx % (BK / 4)) * 4;
            int globalRow = by * BM + loadRow;
            int globalCol = t * BK + loadCol;

            if (globalRow < M && globalCol + 3 < K) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[globalRow * K + globalCol])[0];
                As[loadRow][loadCol]     = tmp.x;
                As[loadRow][loadCol + 1] = tmp.y;
                As[loadRow][loadCol + 2] = tmp.z;
                As[loadRow][loadCol + 3] = tmp.w;
            } else {
                // 边界处理：逐个加载
                for (int i = 0; i < 4; i++) {
                    As[loadRow][loadCol + i] =
                        (globalRow < M && globalCol + i < K)
                        ? A[globalRow * K + globalCol + i] : 0.0f;
                }
            }
        }

        // --- 向量化加载 B tile ---
        // B tile: BK × BN = 8 × 128 = 1024 float = 256 个 float4
        {
            int loadIdx = linearIdx;
            int loadRow = loadIdx / (BN / 4);
            int loadCol = (loadIdx % (BN / 4)) * 4;
            int globalRow = t * BK + loadRow;
            int globalCol = bx * BN + loadCol;

            if (globalRow < K && globalCol + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &B[globalRow * N + globalCol])[0];
                Bs[loadRow][loadCol]     = tmp.x;
                Bs[loadRow][loadCol + 1] = tmp.y;
                Bs[loadRow][loadCol + 2] = tmp.z;
                Bs[loadRow][loadCol + 3] = tmp.w;
            } else {
                for (int i = 0; i < 4; i++) {
                    Bs[loadRow][loadCol + i] =
                        (globalRow < K && globalCol + i < N)
                        ? B[globalRow * N + globalCol + i] : 0.0f;
                }
            }
        }

        __syncthreads();

        // --- 计算（和 V3 相同）---
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) regA[m] = As[threadRow + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) regB[n] = Bs[k][threadCol + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    regC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // --- 向量化写回 C ---
    for (int m = 0; m < TM; m++) {
        int globalRow = by * BM + threadRow + m;
        if (globalRow < M) {
            for (int n = 0; n < TN; n += 4) {
                int globalCol = bx * BN + threadCol + n;
                if (globalCol + 3 < N) {
                    float4 tmp = {regC[m][n], regC[m][n+1],
                                  regC[m][n+2], regC[m][n+3]};
                    reinterpret_cast<float4*>(
                        &C[globalRow * N + globalCol])[0] = tmp;
                } else {
                    for (int i = 0; i < 4 && globalCol + i < N; i++)
                        C[globalRow * N + globalCol + i] = regC[m][n + i];
                }
            }
        }
    }
}
```

> - `float4` 读取要求**16-byte 对齐**。`A[globalRow * K + globalCol]` 必须是 16 的倍数（字节地址）。当 K 不是 4 的倍数时，float4 加载可能跨页 → 要特殊处理边界。


## CUDA V5 双缓冲
V2-V4 的流水线中，**加载**和**计算**是交替进行的：

```
V4 的执行时间线（一个 tile 迭代）:

  ┌────────────────┬─────────────────────┬────────────────┬──────...
  │ Load tile t    │ Compute tile t      │ Load tile t+1  │ Comp...
  │ (HBM → SMEM)  │ (SMEM → Reg → FMA)  │ (HBM → SMEM)  │
  └────────────────┴─────────────────────┴────────────────┴──────...
       idle compute      idle memory          idle compute

  加载和计算互相等待 → pipeline bubble
```

**双缓冲**：用**两份 shared memory**，一份给当前计算，一份给下一次预取。

```
双缓冲流水线:

  SMEM Buffer A (下标 0):  ████ Load t   ↓              ████ Load t+2
  SMEM Buffer B (下标 1):            ████ Load t+1               ████ Load t+3
  Compute:                      ████ Comp t  ████ Comp t+1  ████ Comp t+2
                           ─────────────────────────────────────────────→ time

  加载 tile t+1 和计算 tile t 重叠 → 隐藏内存延迟
```

```cpp
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matmul_v5(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    // 双缓冲: 2 份 SMEM
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int threadRow = ty * TM;
    int threadCol = tx * TN;

    float regC[TM][TN] = {0.0f};
    float regA[TM], regB[TN];

    int numThreads = blockDim.x * blockDim.y;
    int linearIdx = ty * blockDim.x + tx;
    int numTiles = (K + BK - 1) / BK;

    // ===== 预加载第一个 tile (buffer 0) =====
    auto load_tile = [&](int buf, int t) {
        // Load A tile
        {
            int loadIdx = linearIdx;
            int loadRow = loadIdx / (BK / 4);
            int loadCol = (loadIdx % (BK / 4)) * 4;
            int globalRow = by * BM + loadRow;
            int globalCol = t * BK + loadCol;
            if (globalRow < M && globalCol + 3 < K) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[globalRow * K + globalCol])[0];
                As[buf][loadRow][loadCol]     = tmp.x;
                As[buf][loadRow][loadCol + 1] = tmp.y;
                As[buf][loadRow][loadCol + 2] = tmp.z;
                As[buf][loadRow][loadCol + 3] = tmp.w;
            } else {
                for (int i = 0; i < 4; i++)
                    As[buf][loadRow][loadCol + i] =
                        (globalRow < M && globalCol + i < K)
                        ? A[globalRow * K + globalCol + i] : 0.0f;
            }
        }
        // Load B tile
        {
            int loadIdx = linearIdx;
            int loadRow = loadIdx / (BN / 4);
            int loadCol = (loadIdx % (BN / 4)) * 4;
            int globalRow = t * BK + loadRow;
            int globalCol = bx * BN + loadCol;
            if (globalRow < K && globalCol + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &B[globalRow * N + globalCol])[0];
                Bs[buf][loadRow][loadCol]     = tmp.x;
                Bs[buf][loadRow][loadCol + 1] = tmp.y;
                Bs[buf][loadRow][loadCol + 2] = tmp.z;
                Bs[buf][loadRow][loadCol + 3] = tmp.w;
            } else {
                for (int i = 0; i < 4; i++)
                    Bs[buf][loadRow][loadCol + i] =
                        (globalRow < K && globalCol + i < N)
                        ? B[globalRow * N + globalCol + i] : 0.0f;
            }
        }
    };

    load_tile(0, 0);
    __syncthreads();

    // ===== 主循环: 计算当前 tile + 预取下一个 tile =====
    for (int t = 0; t < numTiles; t++) {
        int curBuf = t % 2;
        int nxtBuf = 1 - curBuf;

        // 预取下一个 tile（如果还有）
        if (t + 1 < numTiles) {
            load_tile(nxtBuf, t + 1);
        }

        // 计算当前 tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++)
                regA[m] = As[curBuf][threadRow + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                regB[n] = Bs[curBuf][k][threadCol + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    regC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();  // 确保预取完成后再进入下一轮
    }

    // ===== 写回 C (向量化) =====
    for (int m = 0; m < TM; m++) {
        int globalRow = by * BM + threadRow + m;
        if (globalRow < M) {
            for (int n = 0; n < TN; n += 4) {
                int globalCol = bx * BN + threadCol + n;
                if (globalCol + 3 < N) {
                    float4 tmp = {regC[m][n], regC[m][n+1],
                                  regC[m][n+2], regC[m][n+3]};
                    reinterpret_cast<float4*>(
                        &C[globalRow * N + globalCol])[0] = tmp;
                } else {
                    for (int i = 0; i < 4 && globalCol + i < N; i++)
                        C[globalRow * N + globalCol + i] = regC[m][n + i];
                }
            }
        }
    }
}
```

```
V5 vs V4:
  SMEM 用量: 8 KB × 2 = 16 KB (仍在 sm_86 的 100 KB 限制内)
  加载-计算重叠: 隐藏 ~50-80% 的 HBM 延迟

  进一步优化需要:
    - SMEM bank conflict 消除（swizzle/padding）
    - Warp-level tiling（把 BM×BN tile 再分配给 warp）
    - 使用 cp.async 指令直接从 HBM 到 SMEM（绕过寄存器）
```









11
