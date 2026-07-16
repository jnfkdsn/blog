---
order: 7
title: Ampere 异步拷贝与软件流水线
updated: 2026-07-16
tags: [cuda, ampere, cp-async, pipeline, double-buffering]
status: draft
---

# Ampere 异步拷贝与软件流水线

相关路线：[CUDA 学习笔记](/notes/cuda/) / [GPU 编程与算子优化知识地图](/notes/gpu-programming) / 前置知识：[CUDA 基础语法](/notes/cuda/cuda_basic_syntax) / [CUDA 矩阵乘法](/posts/GEMM) / 后续实践：[Tensor Core GEMM 实践](/posts/tensor_core_gemm)

高性能 GEMM、Flash Attention 和许多 tiled kernel 都在重复：

```text
把下一块数据从 global memory 搬到 shared memory
使用当前 shared-memory tile 计算
```

如果搬运和计算严格串行，计算单元会等待数据，内存系统也会在计算阶段闲置。软件流水线的目标是让不同 tile 的搬运和计算重叠。

## 三个容易混淆的概念

### 两份 buffer

```cpp
__shared__ half smem_a[2][TILE_SIZE];
```

只有两份存储空间，还不能证明存在并发。它只是让“当前 tile”和“下一 tile”可以同时存在。

### Double buffering

double buffering 描述交替使用两个 buffer 的组织方式：

```text
计算 buffer 0 时准备 buffer 1
计算 buffer 1 时准备 buffer 0
```

如果“准备 buffer 1”仍然用普通同步 load/store，并在计算前完成，那么它只是乒乓存储，还没有真正隐藏搬运延迟。

### Async pipeline

异步流水要求：

1. 发起下一 tile 的拷贝后，线程可以继续执行独立计算。
2. 只在消费该 tile 前等待拷贝完成。
3. buffer 生命周期和同步关系保证生产者不会覆盖消费者仍在使用的数据。

## 同步版本的时间线

```text
Load tile 0 -> Compute tile 0 -> Load tile 1 -> Compute tile 1
```

总时间近似：

$$
T \approx \sum_t (T_{load,t} + T_{compute,t})
$$

双级流水的理想稳态：

```text
prologue: Load tile 0

stage 0:  Compute tile 0  || Load tile 1
stage 1:  Compute tile 1  || Load tile 2
stage 2:  Compute tile 2  || Load tile 3

epilogue: Compute last tile
```

理想情况下稳态每轮更接近：

$$
T_{stage} \approx \max(T_{load}, T_{compute})
$$

实际还会受到依赖、同步、带宽、instruction issue 和资源占用影响。

## Ampere 的异步 global-to-shared copy

Ampere 为 global memory 到 shared memory 的拷贝提供硬件支持。其重要性质包括：

- 拷贝可以相对计算异步进行。
- 数据可以直接进入 shared memory，避免用普通寄存器作为中转。
- 16-byte 拷贝可以选择绕过 L1。
- 通过 barrier、pipeline 或低层 wait group 管理完成事件。

常见接口层次：

```text
高层：cuda::memcpy_async + cuda::pipeline / cuda::barrier
中层：cooperative_groups::memcpy_async
低层：__pipeline_memcpy_async / PTX cp.async
```

在 Ampere 学习路径中，先用 `cuda::pipeline` 理解生产者/消费者语义，再看 PTX `cp.async` 的 commit/wait group。

## Pipeline 的生产者/消费者模型

两级 pipeline 可以抽象为：

```text
producer_acquire(stage)
  -> issue async copy into stage
producer_commit(stage)

consumer_wait(stage)
  -> compute from stage
consumer_release(stage)
```

概念性代码：

```cpp
#include <cooperative_groups.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

template <int STAGES>
__global__ void pipelined_kernel(const half* input, float* output) {
    cg::thread_block block = cg::this_thread_block();

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, STAGES> pipeline_state;
    __shared__ half smem[STAGES][/* tile elements */];

    auto pipe = cuda::make_pipeline(block, &pipeline_state);

    // Prologue：先填充前面的 stage。
    // Mainloop：producer 发起未来 tile，consumer 计算当前 tile。
    // Epilogue：排空已经发起的 stage。
}
```

真实代码还需要明确：

- 每个线程搬多少连续字节。
- global/shared 地址是否满足对齐要求。
- 哪些线程参与 copy，哪些线程参与 compute。
- stage 索引如何循环复用。
- 在最后几个 tile 如何排空 pipeline。
- 边界 tile 的越界元素如何填 0。

## PTX `cp.async` 的分组语义

低层可以把多条 copy 组成 async group：

```text
cp.async ...
cp.async ...
cp.async.commit_group

执行与这些 copy 无依赖的计算

cp.async.wait_group N
```

直觉：

- `commit_group`：把此前发起的 copy 组成一组。
- `wait_group N`：只允许最多 N 个更近的 group 继续未完成，确保要消费的旧 group 已经可见。
- `__syncthreads()`：在 block 内让其他线程安全看到 shared-memory 数据；它和 copy completion 解决的是不同层面的同步问题。

不要把 `cp.async.wait_group` 简化成“等价于 `__syncthreads()`”。前者关注异步 copy group 的完成，后者是 block 线程之间的 barrier 和 memory ordering。

## 两级和多级流水

### 两级

```text
stage 0：正在计算
stage 1：正在加载
```

优点是实现和资源成本较低，适合建立正确性。

### 三级及以上

```text
stage 0：正在计算
stage 1：已经加载完成，等待消费
stage 2：正在加载
```

更多 stage 可以覆盖更长的 memory latency，但会增加：

- shared memory 占用。
- pipeline state 和同步复杂度。
- 可能的寄存器压力。
- 每个 SM 可驻留 block 数下降的风险。

因此 `num_stages` 不是越大越好。应通过实际 shape 和 profiler 决定。

## GEMM 中的两层流水

高性能 Tensor Core GEMM 常同时存在：

### CTA 级流水

```text
global memory -> shared memory
```

用多个 shared-memory stage，让下一 K tile 的 global load 与当前 K tile 的 MMA 重叠。

### Warp 级流水

```text
shared memory -> fragment registers -> MMA
```

当前 fragment 送入 Tensor Core 时，准备下一组 fragment。CUTLASS 文档将其描述为 shared-memory tile 与 warp fragment 两个层次的 double buffering。

只优化 global-to-shared 而忽略 shared-to-register，也可能让 Tensor Core 因等待 fragment 而停顿。

## Shared memory layout 与 bank conflict

异步搬运只解决“何时搬”，不自动解决“怎么放”。Tensor Core mainloop 仍要设计 shared-memory layout：

- global load 是否合并。
- 每个 16-byte copy 是否对齐。
- `ldmatrix` 读取时是否 bank conflict。
- A/B tile 是否需要 padding 或 swizzle。
- stage 之间的地址间隔是否保持对齐。

一个 layout 可能对 global store 很友好，却让后续 `ldmatrix` 产生冲突。因此应从完整数据流分析，而不是单独优化某一步。

## 当前 GEMM V5 应如何理解

[CUDA 矩阵乘法](/posts/GEMM) 的 V5 使用两份 shared memory，但 `load_tile()` 仍是普通 global load + shared store。更准确的定位是：

```text
V5：建立 ping-pong buffer 和 stage 生命周期
V6：用 cp.async / cuda::pipeline 实现真实异步搬运
V7：与 Tensor Core MMA mainloop 合并
```

V5 仍然有学习价值，因为异步化之前必须先保证：

- 两个 buffer 不会互相覆盖。
- 当前 tile 和下一 tile 的索引正确。
- 边界处理正确。
- 同步点不会产生 data race。

但不能仅凭两份 buffer 就声称隐藏了固定比例的 HBM latency。

## 如何验证流水线有效

### 正确性

- 与无流水 baseline 比较输出。
- 覆盖 K 不是 tile/stage 整数倍的情况。
- 使用 Compute Sanitizer 检查 shared-memory race 和越界。

### 性能

比较：

```text
同步单 buffer
同步 ping-pong buffer
cp.async 2 stages
cp.async 3 stages
cp.async 4 stages
```

记录：

- kernel duration。
- global memory throughput。
- Tensor Core / compute pipeline active 情况。
- long scoreboard、barrier、not selected 等 stall。
- registers/thread、shared memory/block、achieved occupancy。

判断逻辑：

- duration 下降且 memory-wait stall 下降，说明延迟覆盖可能有效。
- stage 增加后 occupancy 明显下降、duration 反升，说明资源成本超过收益。
- Tensor Core active 仍低而 memory throughput 已高，可能仍由数据供应限制。
- 两份 buffer 与单 buffer 性能接近，不代表结构无用，可能只是尚未真正异步化。

## Ampere 与 Hopper 的边界

Ampere 主线是 `cp.async` 把一维连续片段从 global memory 搬到 shared memory。Hopper 在此基础上引入 TMA，让少量线程发起更大、更高维的 tensor 搬运。

学习顺序：

```text
Ampere cp.async / pipeline
  -> 理解 stage、barrier、producer-consumer
  -> 再学习 Hopper TMA 和 warp specialization
```

不要在 3090 实验中直接套用需要 compute capability 9.0 的 TMA 代码。

## 学完后的检查问题

- 两份 shared-memory buffer 为什么不等于真正异步？
- `cp.async.wait_group` 与 `__syncthreads()` 分别保证什么？
- 为什么增加 stage 可能降低性能？
- CTA 级和 warp 级流水分别搬运哪一段数据？
- shared-memory layout 为什么必须同时考虑 global copy 和 `ldmatrix`？
- 如何用实验判断流水线是在隐藏延迟，还是只增加了资源占用？

## 参考资料

- [CUDA Programming Guide: Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [NVIDIA Hopper GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
