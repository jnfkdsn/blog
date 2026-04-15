---
order: 2
---

## 问题定义
在softmax中实现了树形归约和warp shuffle，本节目标是系统化这些知识，处理各种reduce算子

对于n个数求和，CPU上需要O(N)顺序遍历，GPU把串行依赖降为log(N)步


## v1
```cpp
__global__ void reduce_v1(const float* input, float* output, int N){
    tid = threadIdx.x;
    gid = blockIdx.x * blockDim.x + threadIdx.x;
    extren __shared__ float sdata[];
    sdata[tid] = (gid<N) ? input[gid] : 0.0f;
    __syncthreads();
    // 交错寻址
    for(int s = 1; s < blockDim.x; s *= 2 ){
        if(tid%(2*s) == 0){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if(tid==0) output[blockIdx.x] = sdata[0];
}
```

### 问题
stride=1时，交错寻址导致同一warp内奇偶线程走不同的分支，串行化导致吞吐量减半。
warp divergence:当一个warp中的线程在执行代码时，由于逻辑分支（如 if-else、switch 或 for 循环次数不同）走向了不同的指令路径，就会发生分化, 硬件先执行 if=true 的线程，其余线程被mask掉,然后执行 if=false 的线程

若使用softmax中顺序寻址的方式
```cpp
for(int i = blockDim.x / 2 ; i>0; i>>=1){
    if(tid<i){
        smem[tid] = smem[tid]+smem[tid+i];
    }
    __syncthreads();
}
```
当i>=32时，整个warp全是true或false，不会存在两个分支，当i<=16时才会在第一个warp出现warp divergence，但是影响较小

## v2:加载时归约
v1除了warp divergence，还存在线程闲置的问题，随着迭代的进行，参与计算的线程主次减半，大量线程处于闲置状态

解决：让每个线程在加载阶段就做第一次加法
```cpp
__global__ void reduce_v2(const float* input, float* output, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x*(blockDim.x * 2)+threadIdx.x;  //每个block处理2*blockDim个元素
    float val = 0.0f;
    if(gid<N) val+=input[gid];
    if (gid + blockDim.x < N) val += input[gid + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

V1: 每个 block 加载 blockDim 个元素，但归约第一步就只用 blockDim/2 个线程
      → 加载阶段线程利用率 100%，归约第一步 50%

V2: 每个 block 加载 2*blockDim 个元素，归约开始时所有线程都有有意义的数据，在等待内存返回数据的同时做一次加法，几乎是零额外开销
      → grid 中 block 数量减半，但每个 block 做的工作翻倍
      → 总线程数减半 → kernel launch 开销减半

- 可以推广到每个thread加载4/8/k个元素，但是若k过高，总block数减少，导致SM不够填满，GPU空闲性能可能下降

## v3 : warp shuffle

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;  
}
__global__ void reduce_v3(const float* input, float* output, int N) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    float val = gid<N?input[gid]:0.0f;
    val = warp_reduce_sum(val);
    __shared__ float smem[32];
    if(lane == 0)
        smem[warp_id] = val;
    __syncthreads();
    //第一个warp最终归约
    int num_warps = N / blcokDim.x;
    val = (tid < num_warps) ? smem[warp_id] : 0.0f;
    if(warp_id==0){
        val = warp_reduce_sum(val);
    }
    if(tid==0) output[blockIdx.x] = val;
}
```
主要改进为消除了同步的开销

## v4 
first add + warp shuffle 消除循环开销
```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) { //强制内联
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__global__ void reduce_v4(const float* input, float* output, int N) {
    int gid = (BLCOK_SIZE*2) * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    float val = 0.0f;
    if(gid<N) val+=input[gid];
    if(gid+BLOCK_SIZE<N) val+=input[gid+BLOCK_SIZE];
    val = warp_reduce_sum(val);

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NUM_WARPS];
    if(lane == 0)
        smem[warp_id] = val;
    __syncthreads();
    //第一个warp最终归约
    val = (tid < NUM_WARPS) ? smem[warp_id] : 0.0f;
    if(warp_id==0){
        val = warp_reduce_sum(val);
    }
    if(tid==0) output[blockIdx.x] = val;
}
```

## v5：多block reduce
### v5.1 多次kernel launch
```cpp
typedef void (*ReduceKernel)(const float*, float*, int);

// 递归调用直到只剩 1 个 block 输出
void launch_multipass(ReduceKernel fn, int bs,
                      const float* d_in, float* d_out, int N,
                      float* d_p1, float* d_p2) {
    int g = (N + bs - 1) / bs;
    fn<<<g, bs>>>(d_in, d_p1, N);
    if (g == 1) { cudaMemcpy(d_out, d_p1, sizeof(float), cudaMemcpyDeviceToDevice); return; }
    int g2 = (g + bs - 1) / bs;
    fn<<<g2, bs>>>(d_p1, d_p2, g);
    if (g2 == 1) { cudaMemcpy(d_out, d_p2, sizeof(float), cudaMemcpyDeviceToDevice); return; }
    fn<<<1, bs>>>(d_p2, d_out, g2);
}
```

### v5.2 原子操作
```cpp
float atomicAdd(float* addr, float val); 
/*代价：
无竞争：不同线程访问不同地址，相当于写入global memory的代价
有竞争：多线程访问同一地址，串行写入
*/
```
原子操作实现多block归约
```cpp
template <int BLOCK_SIZE>
__global__ void reduce_atomic(const float* input, float* output, int N) {
    int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    float val = 0.0f;
    if (gid < N) val += input[gid];
    if (gid + BLOCK_SIZE < N) val += input[gid + BLOCK_SIZE];
    val = warp_reduce_sum(val);
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[NUM_WARPS];
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();
    if (tid < NUM_WARPS) val = warp_sums[tid];
    else val = 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    // 关键区别: 使用 atomicAdd 汇总所有 block 的结果
    if (tid == 0) {
        atomicAdd(output, val);  // ← output 被初始化为 0
    }
}
```
对比：
5.1 无竞争，需要多次的kernel launch开销
5.2 单次kernel launch，grid较大时会有竞争

## CUB库 reduce
```cpp
#include <cub/cub.cuh>
//block-level reduce
__global__ void reduce_cub_block(const float* input, float* output, int N){
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < N) ? input[gid] : 0.0f;
    // 一行代码完成 block 内归约
    float block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0) output[blockIdx.x] = block_sum;
}

//device-level reduce
void reduce_cub_device(const float* input, float* output, int N) {
    // 查询需要的临时存储大小
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes, input, output, N);
    // 分配临时存储
    void* temp_storage;
    cudaMalloc(&temp_storage, temp_bytes);
    // 执行归约
    cub::DeviceReduce::Sum(temp_storage, temp_bytes, input, output, N);
    cudaFree(temp_storage);
}

```

性能比较：
```
N=1048576   N=134217728
v1:0.052    v1:5.937
v2:0.031    v2:3.063
v3:0.029    v3:3.343
v4:0.024    v4:2.208
CUB:0.014   CUB:2.176
```
N较小时，CUB只需一般kernel launch，所以比v4要快，较大时基本持平