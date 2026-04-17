---
order: 3
---

# CUDA 矩阵乘法

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
  HBM Bandwidth:           936 GB/s   
  SMEM:					   17.7 TB/s  
  L2 Cache:                32 MB
Ridge Point = 35.6 TFLOPS / 936 GB/s ≈ 38 FLOPs/Byte
对于较大的n，是严重compute-bound
```

### CPU实现
```cpp
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum = 0.0f;
            for(int k=0;k<K;k++){
                sum+=A[K*i+k]*B[k*N+j];
            }
            C[N*i+j]=sum;
        }
    }
}
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

```cpp
#define BM 32
#define BN 32
#define BK 32

__global__ void matmul_v2(const float* A, const float* B, float* C,int M, int N, int K) {
	// 每个 block 负责 C 的一个 BM × BN tile
	__shared__ float As[BM][BK];
	__shared__ float Bs[BK][BN];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	// 行由 blockIdx.y 决定，列由 blockIdx.x 决定
	int row = by * BM + ty;
	int col = bx * BN + tx;

	float sum = 0.0f;
	for(int t = 0; t < (K + BK - 1) / BK; t++){
		// 加载 A tile: A[row][t*BK + tx]
		int a_col = t * BK + tx;
		if(row < M && a_col < K){
			As[ty][tx] = A[row * K + a_col];
		}
		else{
			As[ty][tx] = 0.0f;
		}
		// 加载 B tile: B[t*BK + ty][col]
		int b_row = t * BK + ty;
		if(b_row < K && col < N){
			Bs[ty][tx] = B[b_row * N + col];
		}
		else{
			Bs[ty][tx] = 0.0f;
		}
		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BK; i++){
			sum += As[ty][i] * Bs[i][tx];
		}
		__syncthreads();
	}
	if(row < M && col < N){
		C[row * N + col] = sum;
	}
}
// 调用
dim3 block(BN, BM);   // 32 × 32 = 1024 threads
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
matmul_v2<<<grid, block>>>(A, B, C, M, N, K);
```
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

```cpp
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__global__ void matmul_v3(const float* A, const float* B, float* C, int M, int N, int K){
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;
	
    int thread_row = ty * TM;
    int thread_col = tx * TN; 

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    int numThreads = blockDim.x * blockDim.y; //256
    int linearIdx = ty * blockDim.x + tx;

    for(int t=0; t<(K+BK-1)/BK;t++){
		for(int i=0; i<BM*BK/num_threads; i++){
			int loadIdx = linearIdx + i * numThreads; //thread0 加载 0，256，512，768
            int loadRow = loadIdx / BK;
            int loadCol = loadIdx % BK;
            int globalRow = by * BM + loadRow;
            int globalCol = t * BK + loadCol;
            As[loadRow][loadCol] = (globalRow < M && globalCol < K)
                                    ? A[globalRow * K + globalCol] : 0.0f;
		}

		for(int i=0; i<BK*BN/num_threads;i++){
			int loadIdx = linearIdx + i * num_threads;
			int loadRow = loadIdx / BN;
			int loadCol = loadIdx % BN;
			int globalRow = loadRow + t * BK;
			int globalCol = loadCol + bx * BN;
			Bs[loadRow][loadCol] = (globalRow < K && globalCol < N)
						? B[globalRow * N + globalCol] : 0.0f;
		}
        // --- 寄存器级别计算 ---
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 加载 A 的 TM 个值到寄存器
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[threadRow + m][k];
            }
            // 加载 B 的 TN 个值到寄存器
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regB[n] = Bs[k][threadCol + n];
            }
            // 外积: TM × TN 次 FMA
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    regC[m][n] += regA[m] * regB[n];
                }
            }
        }
        __syncthreads();
    }

	//写回C
	for(int i=0; i<TM; i++){
		for(int j=0; j<TN; j++){
			int globalRow = thread_row + by * BM + i;
			int globalCol = thread_col + bx * BN + j;
			if(globalRow<M&&globalCol<N){
				C[globalRow*N+globalCol] = regC[i][j];
			}
		}
	}
}
// 调用
dim3 block(BN / TN, BM / TM);  // (16, 16) = 256 threads
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
matmul_v3<<<grid, block>>>(A, B, C, M, N, K);
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
  
  1 个 warp 加载 A tile 触发 4 条 cache line 事务
  可以通过转置加载As解决：
  // 原始: As[BM][BK]，按原始布局存
  // 修改: As[BK][BM]，加载时做转置
  __shared__ float As[BK][BM];  

  // 让 warp 内 32 个线程沿 BM 方向(行方向)连续访问 A，而不是沿 BK 方向
  int loadRow_A = linearIdx / BM;   // 0..7 (BK=8)
  int loadCol_A = linearIdx % BM;   // 0..127 (BM=128)

  // 全局: A[by*BM + loadCol_A][t*BK + loadRow_A]
  // 存入: As[loadRow_A][loadCol_A]  
  globalRow = by * BM + loadCol_A;
  globalCol = t  * BK + loadRow_A;
  As[loadRow_A][loadCol_A] = A[globalRow * K + globalCol];
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


## CUDA v4 : 向量化加载
使用float4读取A,B数据
```cpp

```