---
order: 1
---

# Softmax算子实现与优化
## 计算公式
给定一个向量 $\mathbf{x} = [x_0, x_1, ..., x_{N-1}]$，Softmax 将其转换为概率分布：
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=0}^{N-1} e^{x_j}}$$
直接计算 $e^{x_i}$ 会有数值上溢/下溢
可以使用：
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j} e^{x_j - \max(\mathbf{x})}}$$
- 三步计算流程：
```
输入 x = [x0, x1, ..., x_{N-1}]

Step 1: m = max(x0, x1, ..., x_{N-1})          ← 第一次遍历：归约求最大值
Step 2: s = Σ exp(xi - m)                       ← 第二次遍历：归约求指数和
Step 3: output_i = exp(xi - m) / s              ← 第三次遍历：逐元素归一化
```

## CPU实现
```cpp
void softmax(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        const float* x = input + row * N;
        float* y = output + row * N;
        // max
        float max_val = x[0];
        for (int i = 1; i < N; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        // sum of exp
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(x[i] - max_val);
        }
        // normalize
        for (int i = 0; i < N; i++) {
            y[i] = expf(x[i] - max_val) / sum;
        }
    }
}
```


## CUDA实现一：一个block处理一行
每一行分配一个 Block，Block 内的线程协作完成归约。如果一行有 N 个元素且 N > block_size，每个线程需要处理多个元素。

```cpp
__global__ void softmax_v1(const float* input, float* output, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = input + row * N;  //当前行
    float* y = output + row * N;
    //max
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, x[j]);
    }
    extern __shared__ float smem[];
    smem[tid] = local_max;
    __syncthreads();
    //树形归约求max
    for(int s = blockDim.x / 2; s > 0; s >>= 1 ){
        if(tid<s){
            smem[tid] = fmaxf(smem[tid],smem[tid+s]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    //求和
    float local_sum = 0.0f;
    for(int j = tid; j<N; j+=blockDim.x){
        local_sum+=expf(x[j]-row_max);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for(int i = blockDim.x / 2 ; i>0; i>>=1){
        if(tid<i){
            smem[tid] = smem[tid]+smem[tid+i];
        }
        __syncthreads();
    }
    float row_sum = smem[0];
    // normalize
    for(int j = tid; j<N;j+=blockDim.x){
        y[j] = expf(x[j]-row_max)/row_sum;
    }
}
```

对比CPU的实现：对于[1024,4096]的输入数据
```
int block_size = 256;
int grid_size = M;  // 每行一个 block
size_t smem_size = block_size * sizeof(float);
softmax_v1<<<grid_size, block_size, smem_size>>>(d_input, d_output, M, N);
```
得到结果：
```
CPU:  29.392 ms
GPU:  0.089 ms
Speedup: 330.2x
```
### 性能瓶颈分析
![NCU分析](./images/softmax_v1.png)