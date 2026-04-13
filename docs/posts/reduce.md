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
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    float val = (gid < N) ? input[gid] : 0.0f;
    val = warp_reduce_sum(val);
    __shared__ float warp_sums[32];  
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    int num_warps = blockDim.x / 32;
    val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}
```
主要改进为消除了同步的开销

