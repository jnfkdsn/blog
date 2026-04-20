### 1.Global Memory 访问优化：Coalescing
GPU全局内存按照128字节（32bytes * 4 扇区）为单位进行事务，当一个wrap的32个线程同时访问连续的内存地址时，把32个请求合并

#### 用shared memory解决非合并访问
```cpp
__global__ void transpose_optimized(const float* input, float* output,
                                     int M, int N) {
    __shared__ float tile[32][32 + 1];  
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    // 步骤 1：从 input 按行读取（coalesced）到 shared memory
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    // 步骤 2：从 shared memory 转置读取到 output（按行写，也是 coalesced）
    int out_x = blockIdx.y * 32 + threadIdx.x;  // 注意 x/y 互换
    int out_y = blockIdx.x * 32 + threadIdx.y;
    if (out_x < M && out_y < N) {
        output[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];  // 转置读取
    }
}
```
1. shared memory是片上SRAM，不存在coalescing，只需要关注bank conflict
2. 同一wrap内，threadidx.x连续变化，对于二维block，CUDA线性化公式是：linear_tid=threadIdx.y × blockDim.x + threadIdx.x
对于dim3 block(32,32),即blockDim.x = 32
```
linear_tid:  threadIdx.y  threadIdx.x
    0            0            0
    1            0            1
    2            0            2
    ...
   31            0           31      ← Warp 0（这 32 个线程 y=0, x=0~31）
   32            1            0
   33            1            1
    ...
   63            1           31      ← Warp 1（这 32 个线程 y=1, x=0~31）
```
每个 warp 刚好是一整行：threadIdx.y 相同，threadIdx.x 从 0 到 31。
如果blockDim.x < 32,每个wrap会跨越两行，coalescing效果变差

### bank conflict
#### Shared Memory 的物理结构
Shared Memory 被分成 **32 个 bank**，每个 bank 宽 4 字节（32 bit）。连续的 4 字节地址被交替分配到不同的 bank：

```
地址:     0    4    8    12   16   ...  124
Bank:   [B0] [B1] [B2] [B3] [B4] ... [B31]
地址:   128  132  136  140  144  ...  252
Bank:   [B0] [B1] [B2] [B3] [B4] ... [B31]
...
```
**bank 分配规则**：`address` 字节的数据在 `Bank = (address / 4) % 32`

**Bank Conflict**: 每个 cycle，每个 bank 只能服务一个访问请求。如果**同一个 Warp 中的多个线程访问同一个 bank 的不同地址**，这些请求必须串行化，发生在同一warp


例如**`float shared[32][32]` 按列访问**：
```cpp
__shared__ float data[32][32];  // 32 行 × 32 列
// 线程 i 访问第 i 行、第 0 列
float val = data[threadIdx.x][0];
// threadIdx.x=0 → data[0][0] → 地址 0     → Bank 0
// threadIdx.x=1 → data[1][0] → 地址 128   → Bank 0  
// threadIdx.x=2 → data[2][0] → 地址 256   → Bank 0
```
解决方案：Padding

```cpp
// 加一列 padding：32 → 33
__shared__ float data[32][33];  // 多加 1 列
// 现在线程 i 访问 data[i][0]：
// threadIdx.x=0 → 地址 0         → Bank 0
// threadIdx.x=1 → 地址 33*4=132  → Bank 1  (132/4 % 32 = 1)
// threadIdx.x=2 → 地址 66*4=264  → Bank 2  (264/4 % 32 = 2)
```

