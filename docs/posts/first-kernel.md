# 第一个 CUDA Kernel：vector_add

> 日期：2026-04-02

## 目标

在 GPU 上实现两个向量逐元素相加：`c[i] = a[i] + b[i]`

## 我的理解

CUDA 编程的核心思路：把一个大任务拆成几千个小任务，每个 GPU 线程只负责一个元素。

CPU 的写法是一个 for 循环遍历所有元素，GPU 的写法是每个线程用 `blockIdx.x * blockDim.x + threadIdx.x` 算出自己负责的下标，只处理那一个。

## 关键代码

```cuda
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

`if (i < n)` 这个边界检查很重要——线程总数通常不是 n 的整数倍，多出来的线程不能越界访问。

## 踩的坑

1. **忘了 cudaMemcpy 回 Host**：kernel 算完结果在 GPU 上，必须 `cudaMemcpyDeviceToHost` 拷回来才能在 CPU 端读到。一开始打印出来全是 0，排查了半天。

2. **grid/block 大小计算**：`(n + block_size - 1) / block_size` 是向上取整的套路，保证线程数 >= n。

## 下一步

尝试 Softmax kernel——这个涉及 shared memory 和 warp 级操作，复杂度上一个台阶。
