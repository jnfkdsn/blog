---
order: 3
---

# CUDA 基础语法
 
## 1. 3090 参数
| 参数 | 值 |
|------|-----|
| **SM 数量** | 82 |
| **CUDA Cores** | 10496 (128/SM × 82) |
| **FP32 峰值** | 35.6 TFLOPS |
| **显存** | 24 GB GDDR6X |
| **显存带宽** | 936 GB/s |
| **L2 Cache** | 6 MB |
| **Shared Memory / SM** | 最大 100 KB（可配置） |
| **L1 Cache / SM** | shared + L1 共享 128 KB pool |
| **Registers / SM** | 65536 个 (256 KB) |
| **Max Threads / SM** | 1536 |
| **Max Threads / Block** | 1024 |
| **Max Blocks / SM** | 16 |
| **Warp Size** | 32 |
| **Max Warps / SM** | 48 |
| **Tensor Cores** | 328 (4/SM × 82) |
| **基础/Boost 频率** | 1395 / 1695 MHz |
| **TDP** | 350W |


## 2. 函数修饰符
| 修饰符 | 在哪执行 | 谁能调用 | 典型用途 |
|--------|---------|---------|---------|
| `__global__` | GPU | CPU（通过 `<<<>>>` ） | kernel 入口函数 |
| `__device__` | GPU | GPU | kernel 内部的工具函数 |
| `__host__` | CPU | CPU | 普通 C++ 函数(不加修饰符等于host) |
| `__host__ __device__` | 两边 | 两边 | 通用工具函数 |

## 3. 内存管理API
显存分配与释放：
cudaMalloc(&d_data,size);
cudaFree(d_data);

数据传输
```cpp
// cudaMemcpy(目标地址, 源地址, 字节数, 传输方向)
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```
四种传输方向：
| 方向枚举 | 含义 | 场景 |
|---------|------|------|
| `cudaMemcpyHostToDevice` | CPU → GPU | 把输入数据传到 GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU | 把计算结果取回 CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU | GPU 内部数据搬移 |
| `cudaMemcpyHostToHost` | CPU → CPU | 很少用（等于 memcpy） |

初始化显存：
cudaMemset(d_data,0,size); //清0

### Pinned Memory（页锁定内存）
普通的 CPU 内存（`new` / `malloc`）可能被操作系统换到磁盘。`cudaMemcpy` 内部需要先把数据拷到一块 pinned（页锁定）的临时缓冲区，再传给 GPU。
直接分配 pinned memory 可以省掉这次中间拷贝：
```cpp
float* h_data;
cudaMallocHost(&h_data, n * sizeof(float));  // 分配 pinned memory
// 用法和普通指针一样，但 cudaMemcpy 速度更快 (约 2 倍)
cudaFreeHost(h_data);  // 注意用 cudaFreeHost 释放
```
**注意**：pinned memory 占用物理内存且不会被换出，分配过多会影响系统性能。只对需要频繁做 H2D/D2H 传输的缓冲区使用。

## 4. kernel
### 4.1 线程层次结构
```
Grid（网格）
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
├── ...
└── Block N-1
    └── ...
```
- **Grid**：一次 kernel 启动产生的所有线程
- **Block**：一组线程。同一个 Block 内的线程可以通过 shared memory 通信、可以同步
- **Thread**：最小执行单位
kernel_function<<<grid_size, block_size>>>(参数...);
总线程数=grid_size x block_size

- 需要注意的是，线程块中的线程数量是有限制的，因为同一个线程块内的所有线程必须运行在同一个流多处理器（Streaming Multiprocessor, SM）上，并共享该 SM 的有限资源（如寄存器、共享内存等）。
- 当启动一个核函数时，其网格中的线程块会被动态分配到 GPU 上各个可用的 SM 中执行。一旦某个线程块被调度到某个 SM 上，其中的所有线程将始终在该 SM 上并发执行，不会迁移到其他 SM 上,多个线程块可以被分配到同一个SM上                                                                                              


### 4.2 内置变量
| 变量 | 类型 | 含义 |
|------|------|------|
| `threadIdx.x` | `uint3` | 当前线程在 Block 内的索引 |
| `blockIdx.x` | `uint3` | 当前 Block 在 Grid 内的索引 |
| `blockDim.x` | `dim3` | 每个 Block 的线程数 |
| `gridDim.x` | `dim3` | Grid 中的 Block 数 |
**计算全局线程 ID 的公式**：
```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

### 4.3 多维线程配置
Grid Block可以是1D,2D,3D的
```cpp
//2D
dim3 block(16, 16);       // 每个 block 16×16 = 256 个线程
dim3 grid(
    (width + 15) / 16,    // x 方向的 block 数
    (height + 15) / 16    // y 方向的 block 数
);
matrix_kernel<<<grid, block>>>(d_matrix, width, height);
__global__ void matrix_kernel(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = row * width + col;  // 行主序索引
        matrix[idx] *= 2.0f;
    }
}
```

- 每个 Block 最多 **1024** 个线程
- 多维时各维度限制不同：x ≤ 1024, y ≤ 1024, z ≤ 64，且 x × y × z ≤ 1024
- Block size 应该是 **32 的倍数**（warp size），否则有线程浪费

## 5. 同步机制

### 5.1 CPU GPU 同步

```
mykernel<<<grid,block>>>(d_data,N);
cudaDeviceSynchronize();
cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);
```

### 5.2 block内线程同步 `__syncthreads()`
```cpp
__global__ void some_kernel(float* data, int n) {
    __shared__ float sdata[256];  // Block 内共享的内存
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 阶段 1：所有线程把数据加载到 shared memory
    if (i < n) {
        sdata[threadIdx.x] = data[i];
    }
    __syncthreads();
    // 同步确保所有线程都完成了写入
    // 阶段 2：现在可以安全地读取其他线程写入的数据
    if (threadIdx.x > 0 && i < n) {
        float left = sdata[threadIdx.x - 1];  // 读取邻居的值
    }
}
```
- `__syncthreads`只能同步同一个block内的线程

### 5.3 wrap级同步
一个 Warp（32 个线程）是 **SIMT（单指令多线程）** 执行的——同一 Warp 内的线程执行相同的指令。在 Volta 架构（sm_70）之后，Warp 内线程可以独立调度，所以需要显式同步：

```cpp
__syncwarp();  // 同步当前 Warp 内的所有线程
__syncwarp(mask);  // 只同步 mask 指定的线程（mask 是 32 位 bitmask）
```

## 6. 错误处理

使用错误检查宏检查CUDA Runtime API返回的 cudaError_t
```cpp
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
// 使用：
CUDA_CHECK(cudaMalloc(&d_data, size));
```
## 7. Grid-Stride Loop 模式

### 7.1 问题：当数据比线程数多很多
**Grid-Stride Loop** 让有限数量的线程循环处理所有数据：
```cpp
__global__ void my_kernel(float* data, int n) {
    // 每个线程从自己的全局 ID 开始，以 grid 总线程数为步长循环
    int stride = blockDim.x * gridDim.x;  // 总线程数
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += stride) {
        data[i] *= 2.0f;
    }
}
my_kernel<<<128, 256>>>(d_data, N);  // 只有 32768 个线程，但能处理任意大的 N
```
| 优点 | 说明 |
|------|------|
| **灵活** | 不需要根据 N 计算 grid size |
| **可复用** | 同一个 kernel 适用于任何数据规模 |
| **性能好** | grid size 可以调到 SM 数量的倍数，保证 GPU 满载 |
| **可调试** | grid size 设为 1, block size 设为 1，变成串行代码，方便调试 |



## 8. shared memory

GPU内存层级
```
寄存器 (Register)        ← 最快，每个线程私有，~0 cycle 延迟
    ↓
共享内存 (Shared Memory)  ← 同一 Block 内的线程共享，~5 cycle
    ↓
L2 Cache                  ← 所有 SM 共享
    ↓
全局内存 (Global Memory)  ← 最慢，所有线程可访问， (显卡显存容量)
```
- 对全局显存访问的特点：当一个线程束（Warp，通常为 32 个线程）同时发起内存请求时，如果这些线程访问的地址是**连续的**且对齐的，硬件会将这些请求合并（Coalesce）为尽可能少的 DRAM 突发传输，这是构成全局显存的DRAM的特性决定的，一旦选中某一行，硬件会连续输出相邻的一组数据块，如果程序不访问连续地址，多读出的数据会被直接丢弃，导致严重的带宽浪费。



## 9. wrap
warp 是在线程块（block）内部，以 32 个线程为一组进行划分的基本调度单元，是 SM 中最基本的执行单元，一个 SM 可以同时驻留多个 Warp，并且通过 Warp Scheduler 每个时钟周期切换执行不同的 Warp。（SIMT，同一个 Warp 内的 32 个线程在任何时刻执行相同的指令，不同的数据）
### wrap级操作
Warp Shuffle：线程间直接交换数据
同一 Warp 内的线程可以**不经过共享内存**直接读取彼此的寄存器值,不需要__syncthreads()进行同步

常用warp shuffle指令:
```cpp
// 从特定的束内线程获取数值，跨线程束值的广播
T __shfl_sync(unsigned int mask, T var, int srcLane, int width = warpSize);
// 通过线程束上移获取数值
T __shfl_up_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize);
// 通过线程束下移获取数值,在 warp 内实现向下偏移的数据交换,lane ID 为 t 的线程将从 lane ID 为 t + delta 的线程中读取变量 val 的值；若 t + delta >= width，则保留当前线程自身的原始值 val
T __shfl_down_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize);
// 按位异或交换数据，lane ID 为 t 的线程将与 lane ID 为 t XOR laneMask 的线程互换值。XOR 保证交换对称，归约后所有 lane 都持有相同结果
T __shfl_xor_sync(unsigned int mask, T var, unsigned int laneMask, int width = warpSize);
```
例如通过__shfl_down_sync() warp内求和
```cpp
__device__ float warp_reduce_sum_down(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;  // 只有 lane 0 持有正确结果
}
```
而通过__shfl_xor_sync求和则warp内所有线程都持有结果












## 10. 原子操作

## 11 Occupancy：GPU 利用率分析
### 什么是 Occupancy？
**Occupancy** = 实际运行的 Warp 数 / SM 最大可运行的 Warp 数

例如，GPU 每 SM 最多支持 48 个 Warp（1536 个线程），但 kernel 由于资源限制只能在每 SM 上跑 32 个 Warp，那么 Occupancy = 32/48 ≈ 67%。

### 影响 Occupancy 的三个因素
1. **Block Size**：如果 block_size = 64，每个 block 只有 2 个 warp。每 SM 最多 16 个 block → 32 个 warp。而 SM 最多支持 48 个 warp → Occupancy = 67%。
2. **寄存器使用量**：每 SM 有 65536 个寄存器。如果 kernel 每线程用 128 个寄存器，每个 SM 最多 512 个线程 = 16 个 warp → Occupancy = 33%。
3. **Shared Memory 使用量**：每 SM 有 ~100 KB shared memory。如果每 block 用 48 KB，每 SM 只能放 2 个 block → 可能限制 occupancy。