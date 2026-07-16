---
order: 6
title: Tensor Core GEMM 实践
updated: 2026-07-16
tags: [cuda, tensor-core, wmma, mma, cp-async, gemm]
status: draft
---

# Tensor Core GEMM 实践

相关路线：[GPU 编程与算子优化知识地图](/notes/gpu-programming) / [CUDA 学习笔记](/notes/cuda/) / 前置知识：[低精度数值与混合精度计算](/notes/cuda/low_precision) / [Tensor Core 编程](/notes/cuda/tensor_core) / [Ampere 异步拷贝与软件流水线](/notes/cuda/async_pipeline) / 相关实践：[CUDA 矩阵乘法](/posts/GEMM) / [Triton GEMM 优化](/posts/triton_gemm)

## 实践目标

在 RTX 3090（Ampere SM86）上完成一条可验证的 Tensor Core GEMM 优化链：

```text
V0  reference 与实验框架
V1  WMMA baseline
V2  CTA/warp 分块与 shared-memory 复用
V3  cp.async 多级流水
V4  PTX mma.sync / ldmatrix 对照
V5  CUTLASS、Triton、cuBLAS 横向比较
```

这篇实践不以“手写代码超过 cuBLAS”为完成条件，而以回答以下问题为目标：

- Tensor Core 指令如何由一个 warp 协作执行？
- 大矩阵如何分解成 CTA tile、warp tile 和 MMA tile？
- WMMA baseline 为什么远达不到硬件峰值？
- shared-memory layout、寄存器和流水线如何限制吞吐？
- 同一组 shape 在 prefill 和 decode 场景为什么表现不同？
- Triton、CUTLASS 和 CUDA 分别替开发者处理了哪些细节？

## 问题定义

计算：

$$
C_{M \times N} = A_{M \times K} B_{K \times N}
$$

主要实验路径：

```text
A: FP16 row-major
B: FP16 row-major
accumulator: FP32
C: FP32（先保留高精度，后续再增加 FP16 epilogue）
```

理论浮点操作量：

$$
FLOPs = 2MNK
$$

性能：

$$
GFLOPS = \frac{2MNK}{time_{seconds}} \times 10^{-9}
$$

注意：不同精度、不同 compute mode 的峰值不同，不能用 FP32 CUDA Core 峰值评价 FP16 Tensor Core kernel。

## 实验约定

每次结果必须记录：

```text
GPU 型号与 compute capability
CUDA、driver、编译器版本
nvcc 编译参数与目标架构
M、N、K、layout、leading dimension
A/B/C 数据类型与 accumulator 类型
warmup 次数、正式迭代次数、计时方式
max absolute error、max relative error
kernel duration、GFLOPS
registers/thread、shared memory/block
```

推荐编译参数：

```bash
nvcc -O3 -arch=sm_86 -lineinfo tensor_core_gemm.cu -o tensor_core_gemm
```

调试阶段再增加同步和错误检查；正式 benchmark 使用 CUDA Event 计时，不把 host allocation、H2D/D2H 和第一次 JIT/library initialization 混入 kernel 时间。

## V0：Reference 与正确性保护

至少准备两个 reference：

1. CPU FP64 或小 shape 的朴素实现：作为高精度 oracle。
2. cuBLAS：作为 GPU 性能和工程实现基准。

### 误差指标

```cpp
struct ErrorStats {
    double max_abs;
    double max_rel;
    double mean_abs;
    size_t nan_count;
    size_t inf_count;
};
```

对每个元素：

```cpp
double abs_err = std::abs(reference[i] - output[i]);
double rel_err = abs_err / std::max(std::abs(reference[i]), 1e-12);
```

不能只看 relative error：参考值接近 0 时，相对误差会被放大。不能只看 absolute error：大幅值输出会掩盖相对偏差。

### 输入分布

至少覆盖：

- `uniform(-1, 1)`：常规数据。
- 正值数据：方便检查 layout/转置错误。
- 可预测小矩阵：例如 A/B 全 1，结果应为 K。
- 包含极大/极小值：检查 FP16 范围和 `inf/0`。

### cuBLAS 模式必须显式记录

对 FP32、TF32、FP16/BF16 路径分别设置明确的 data type 和 compute type。不要把 library 默认行为笼统写成“cuBLAS FP32”，因为 Ampere 上 FP32 输入可能允许 Tensor Core TF32 加速。

建议参考组：

```text
cuBLAS strict/pedantic FP32
cuBLAS FP32 input + TF32 Tensor Core
cuBLAS FP16 input + FP32 accumulate
```

## V1：WMMA Baseline

第一版只解决“让 Tensor Core 正确工作”，暂不追求 shared-memory 复用。

### 映射方式

```text
一个 warp -> 一个 16 x 16 输出 tile
K 方向每次处理 16
每个 block 4 个 warp
4 个 warp 沿 M 方向处理相邻 tile
```

### Kernel

下面代码要求 `M/N/K` 都是 16 的倍数；A、B 为 row-major，C 为 row-major：

```cpp
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template <int WARPS_PER_BLOCK>
__global__ void wmma_gemm_v1(const half* A,
                             const half* B,
                             float* C,
                             int M,
                             int N,
                             int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    int warp_id = threadIdx.x / warpSize;
    int warp_m = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    int warp_n = blockIdx.x;

    // 这个条件对同一个 warp 的所有 lane 相同。
    if (warp_m * WMMA_M >= M || warp_n * WMMA_N >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a,
                   WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,
                   WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        const half* a_tile = A + (warp_m * WMMA_M) * K + k;
        const half* b_tile = B + k * N + warp_n * WMMA_N;

        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + (warp_m * WMMA_M) * N + warp_n * WMMA_N;
    wmma::store_matrix_sync(c_tile,
                            c_frag,
                            N,
                            wmma::mem_row_major);
}
```

Launch：

```cpp
constexpr int WARPS = 4;
dim3 block(WARPS * 32);
dim3 grid((N + 15) / 16,
          (M + WARPS * 16 - 1) / (WARPS * 16));

wmma_gemm_v1<WARPS><<<grid, block>>>(A, B, C, M, N, K);
```

### V1 的预期瓶颈

每个 warp 都直接从 global memory 加载自己的 A/B tile。相邻 warp/block 需要相同数据时无法通过 shared memory 复用，因此可能出现：

- global memory traffic 过高。
- Tensor Core 等待 operand。
- K loop 中 load 和 MMA 基本串行。
- tile 太小，指令调度和地址计算开销占比高。

V1 的完成标准不是高性能，而是：

- correctness 通过。
- SASS/Profiler 能确认生成 Tensor Core 指令。
- 能解释 fragment、leading dimension 和 warp-uniform 控制流。

## V2：CTA Tile 与 Shared-memory 复用

V2 将多个 warp 组织成一个 CTA tile，例如：

```text
CTA tile: 128 x 128 x 32
warp tile: 64 x 32 或 32 x 64
MMA tile: 16 x 8 x 16（PTX 层示意）
```

每轮 K mainloop：

```text
1. block 协作加载 A[BM, BK] 到 shared memory
2. block 协作加载 B[BK, BN] 到 shared memory
3. __syncthreads()
4. 每个 warp 从 shared memory 加载自己的 fragments
5. 执行多条 MMA，更新 accumulator
6. __syncthreads()，防止下一轮覆盖当前 tile
```

### 需要设计的参数

| 参数 | 含义 | 主要约束 |
|---|---|---|
| `BM/BN/BK` | CTA tile | shared memory、并行度、边界浪费 |
| `WM/WN` | warp tile | accumulator 寄存器、warp 数量 |
| MMA shape | 指令 tile | 架构和输入类型 |
| warps/block | CTA 内 warp 布局 | occupancy、复用、同步开销 |

### 共享内存不是简单二维数组

V2 需要同时满足：

- global -> shared 写入合并且对齐。
- shared -> fragment 读取适合 WMMA/`ldmatrix`。
- 不产生严重 bank conflict。
- 每个 stage 的地址正确且不覆盖。

第一版可以先用容易验证的 padding layout，再通过 NCU 检查 bank conflict。后续再学习 swizzle。

### V2 验证问题

- 相比 V1，DRAM 读流量是否下降？
- Tensor Core active 时间是否上升？
- shared-memory bank conflict 是否成为新瓶颈？
- accumulator 增大后 registers/thread 和 occupancy 如何变化？

## V3：`cp.async` 多级流水

V3 在 V2 正确的数据布局上异步化 global-to-shared copy。

### 两级流水

```text
prologue: async load K tile 0 -> stage 0

for each K tile:
    async load tile k+1 -> next stage
    wait current stage ready
    MMA current stage
    release current stage

epilogue: drain remaining stage
```

逐步比较：

```text
V2 single buffer synchronous
V2.5 two buffers synchronous
V3 cp.async 2 stages
V3.1 cp.async 3 stages
V3.2 cp.async 4 stages
```

不要一次直接写 4 stages。先让 2 stages 在非整除 K、边界 tile 下通过 correctness 和 Compute Sanitizer，再扩展。

### 资源模型

若每个 stage 存：

$$
BM \times BK + BK \times BN
$$

个 FP16 元素，则 shared memory 近似为：

$$
S = stages \times (BM \times BK + BK \times BN) \times 2\ bytes
$$

stage 增多会直接增加每 block 的 shared memory，可能减少每 SM 驻留 block 数。

### V3 验证问题

- long scoreboard / memory dependency stall 是否下降？
- 增加 stage 后 Tensor Core 是否更连续地 active？
- shared memory 增加导致 occupancy 降低后，净性能是否仍提升？
- prologue/epilogue 占比在小 K 下是否过高？

## V4：PTX `mma.sync` 与 `ldmatrix`

V4 不一定比 WMMA 更快，它的目标是看清编译器隐藏的 lane/register mapping。

学习顺序：

1. 查看 V1/V2 生成的 PTX 和 SASS。
2. 识别 `mma.sync` / HMMA 指令。
3. 理解一个 `m16n8k16` 指令的 A/B/C/D 寄存器集合。
4. 学习 `ldmatrix` 如何由 warp 从 shared memory 装载矩阵 fragment。
5. 再尝试用 inline PTX 实现一个 instruction tile。

必须记录：

- 每个 lane 负责的输入地址。
- A/B fragment 的寄存器数量和打包方式。
- accumulator 中每个 lane 持有哪些输出元素。
- row/col layout 对寄存器解释的影响。

这一阶段适合和 [Lowering、Codegen、Runtime](/notes/compile/traditional/lowering_codegen_runtime) 对照：WMMA C++ 经过编译器后如何变成 PTX，再变成 SASS。

## V5：CUTLASS、Triton、cuBLAS 对照

### CUTLASS

先使用 CUTLASS Profiler 找到 SM86 上匹配数据类型与 layout 的 kernel，再阅读其配置：

```text
ThreadblockShape
WarpShape
InstructionShape
Stages
OperatorClassTensorOp
Epilogue
```

把这些配置映射回自己的 V2/V3 参数，理解 CUTLASS 把哪些策略模板化了。

### Triton

使用相同 M/N/K、输入类型和 accumulator：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(...):
    a = tl.load(...)
    b = tl.load(...)
    acc += tl.dot(a, b)
```

观察：

- `BLOCK_M/N/K`、`num_warps`、`num_stages` 对性能的影响。
- `tl.dot` 是否下沉到预期 MMA 指令。
- 编译器如何选择 shared-memory layout 和 software pipeline。
- 手写 CUDA 中哪些细节在 Triton 里由编译器承担。

### cuBLAS

cuBLAS 是性能上界的重要参考，但比较必须保证：

- 数据类型、compute type 和输出类型一致。
- transpose/layout 语义一致。
- 计时范围一致。
- warmup 和 workspace 策略一致。

目标不是只报告“比 cuBLAS 慢多少”，而是定位差距来自：

```text
tile 选择
data layout
pipeline stages
epilogue
small-shape specialization
kernel launch / workspace policy
```

## Benchmark Shape 设计

### 方阵

用于观察大规模峰值吞吐：

```text
256^3
512^3
1024^3
2048^3
4096^3
```

### LLM Prefill 风格

M 表示一次参与线性层计算的 token 数：

```text
M = 128, 512, 2048
K = 4096
N = 4096 或 11008
```

### LLM Decode 风格

decode 中有效 M 由当前 batch 的 token 数决定：

```text
M = 1, 8, 16, 32, 64, 128
K = 4096
N = 4096 或 11008
```

观察 M 很小时：

- CTA 数量是否足够填满 GPU。
- Tensor Core 峰值是否还有代表性。
- 权重读取是否主导时间。
- batching 增大后 GEMV-like workload 何时逐渐变成更适合 GEMM 的形态。

### 非规则形状

```text
M/N/K 不是 16 的倍数
K 很大、M/N 很小
M 很大、N 较小
```

这组实验用于检查 padding、predicate、边界浪费和 library fallback。

## Profiler 检查表

### 执行

- Tensor Core / HMMA 指令是否出现。
- Tensor Core pipeline active 比例。
- kernel duration、waves per SM。
- eligible/active warps 和 issue efficiency。

### 内存

- DRAM、L2、L1、shared memory throughput。
- global load/store 是否合并。
- shared-memory bank conflict。
- local load/store 是否提示 register spill。

### Stall

- long scoreboard：等待 global/L2 数据。
- short scoreboard：等待 shared-memory/局部依赖。
- barrier：block 同步等待。
- not selected：有可运行 warp，但调度器选择了其他 warp。
- math pipeline throttle：计算管线压力。

不要单独根据一个指标下结论。例如 long scoreboard 高可能是问题，也可能是因为指令数量减少后它在采样中的占比上升；要结合 duration、吞吐和 active cycles。

## 结果记录模板

| Version | M/N/K | Input/Acc/Output | Time | GFLOPS | Max Abs Err | Reg/Thread | SMEM/Block | 主要瓶颈 |
|---|---|---|---:|---:|---:|---:|---:|---|
| cuBLAS | | | | | | | | |
| V1 WMMA | | | | | | | | |
| V2 Shared Tile | | | | | | | | |
| V3 Async Pipeline | | | | | | | | |
| Triton | | | | | | | | |
| CUTLASS | | | | | | | | |

每次优化都补三句话：

```text
问题：上一版主要被什么限制？
方案：这一版只改变了哪个关键变量？
验证：哪些 correctness/performance 证据支持结论？
```

## 完成标准

- [ ] V1 WMMA 在规则 shape 上通过正确性测试。
- [ ] 能从 SASS/Profiler 证明 Tensor Core 指令已生成。
- [ ] V2 能解释 CTA、warp、MMA 三层 tile。
- [ ] V2 的 shared-memory 复用相较 V1 减少 global traffic 或提高性能。
- [ ] V3 在至少两种 stage 配置下完成正确性和性能对比。
- [ ] 能解释为什么更多 stage 不保证更快。
- [ ] 能画出 `global -> shared -> fragment -> MMA -> epilogue` 数据流。
- [ ] 能把 CUTLASS 配置映射回自己的 tile/pipeline 参数。
- [ ] 能解释 Triton `tl.dot` 与 CUDA WMMA/PTX 的抽象差异。
- [ ] 能解释 prefill 与 decode shape 下性能瓶颈为什么不同。

## 还没搞懂的

- Ampere `ldmatrix` 的 lane-address 映射与 shared-memory swizzle。
- WMMA fragment 到 PTX operand register 的具体下沉。
- 不同 CTA/warp tile 对寄存器复用和 occupancy 的定量影响。
- cuBLAS 在 small-M decode shape 下选择了什么 kernel family。
- CUTLASS/CuTe 如何用 Layout Algebra 表达 MMA Atom 和 tiled copy。

## 参考资料

- [CUDA C++ Programming Guide: Warp Matrix Functions](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-matrix-functions)
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CuTe Dense GEMM Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
