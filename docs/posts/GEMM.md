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

## CUDA V2
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

## CUDA V3
