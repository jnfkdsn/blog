---
order: 6
title: Tensor Core 编程
updated: 2026-07-16
tags: [cuda, tensor-core, wmma, mma, gemm, ampere]
status: draft
---

# Tensor Core 编程

相关路线：[CUDA 学习笔记](/notes/cuda/) / [GPU 编程与算子优化知识地图](/notes/gpu-programming) / 前置知识：[低精度数值与混合精度计算](/notes/cuda/low_precision) / [CUDA 矩阵乘法](/posts/GEMM) / 动手实践：[Tensor Core GEMM 实践](/posts/tensor_core_gemm)

Tensor Core 是专门执行小矩阵乘加的硬件单元。它解决的不是一般意义上的任意矩阵运算，而是重复执行：

$$
D = A \times B + C
$$

大矩阵 GEMM 会被分块为大量 MMA（Matrix Multiply-Accumulate）指令。高性能实现的核心是让 Tensor Core 持续计算，同时及时把下一批 A/B tile 搬到它能消费的位置。

## RTX 3090 上学什么

RTX 3090 属于 Ampere、compute capability 8.6。当前主线是：

```text
FP16 / BF16 / TF32
  -> WMMA C++ API
  -> PTX mma.sync
  -> shared memory + ldmatrix
  -> cp.async 多级流水
  -> CUTLASS / CuTe / Triton
```

暂时不作为主线：

- `wgmma`：Hopper 的 warp-group MMA 路线。
- TMA：Hopper 引入的 Tensor Memory Accelerator。
- `tcgen05`：Blackwell 的第五代 Tensor Core 指令族。

先把 Ampere 的 warp-level MMA 学透，再看后续架构会更容易理解“执行粒度和数据搬运为什么发生变化”。

## 从标量 FMA 到矩阵 MMA

普通 CUDA Core GEMM 中，每个线程持有若干标量寄存器：

```cpp
acc[m][n] += reg_a[m] * reg_b[n];
```

这里每条 FMA 只更新一个标量结果。Tensor Core MMA 则由一个 warp 协作：

```text
warp 中 32 个 lane 共同持有 A fragment
warp 中 32 个 lane 共同持有 B fragment
warp 中 32 个 lane 共同持有 accumulator fragment
执行一次 mma.sync，更新一个小矩阵 tile
```

fragment 不是某个线程持有的完整矩阵。矩阵元素会按照架构规定分散在 32 个 lane 的寄存器中。

## GEMM 的层次化分块

```text
完整 GEMM: M x N x K
└── CTA / thread block tile
    └── warp tile
        └── MMA instruction tile
```

三个层次分别解决：

| 层次 | 主要任务 | 主要存储 |
|---|---|---|
| CTA tile | 把全局矩阵划分给不同 block | global + shared memory |
| warp tile | 把 CTA 输出 tile 分给不同 warp | shared memory + registers |
| MMA tile | 映射到一条或多条 Tensor Core 指令 | registers |

例如一个 CTA 计算 `128 x 128` 输出 tile，可以由 8 个 warp 分别计算多个 `64 x 32` warp tile；每个 warp tile 再由 `m16n8k16` MMA 指令拼成。具体尺寸不是固定答案，需要在复用、并行度、寄存器压力和 shared memory 使用量之间权衡。

## Ampere 常见 MMA 形状

PTX 层常见形状包括：

| 输入 | accumulator | 示例指令形状 |
|---|---|---|
| FP16 / BF16 | FP32 | `m16n8k16`、`m16n8k8` |
| TF32 | FP32 | `m16n8k4` |
| INT8 | INT32 | `m16n8k32`、`m16n8k16` |

要区分：

- WMMA API 暴露的逻辑 fragment 形状，例如 `16 x 16 x 16`。
- PTX `mma.sync` 的实际指令形状，例如 `m16n8k16`。

一个 WMMA 操作可能由编译器下沉成多条底层 MMA 指令，不能把两层形状直接等同。

## 四层编程接口

### WMMA

`nvcuda::wmma` 是 CUDA C++ 的 warp matrix API。它隐藏 lane 到 fragment register 的具体映射，适合第一次写 Tensor Core kernel。

核心接口：

```cpp
wmma::fragment<...>
wmma::fill_fragment(...)
wmma::load_matrix_sync(...)
wmma::mma_sync(...)
wmma::store_matrix_sync(...)
```

优点：

- 接口比 PTX 直接。
- 编译器负责 fragment 的寄存器布局。
- 能快速建立 warp 协作完成矩阵乘的直觉。

限制：

- 可选 shape 和 layout 相对有限。
- fragment 内部元素到 lane/register 的映射是 opaque 的。
- 很难像 CUTLASS 一样表达完整的多级流水和复杂 epilogue。

### PTX `mma.sync`

PTX 层更接近硬件：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

这类签名编码了：

```text
执行与同步语义
M/N/K 指令形状
A/B layout
accumulator 类型
A/B 输入类型
```

直接使用 PTX 前必须理解每个 lane 提供哪些寄存器，以及 `ldmatrix` 如何把 shared-memory tile 装入这些寄存器。

### CUTLASS / CuTe

CUTLASS 把高性能 GEMM 拆成可组合组件：

```text
problem shape
CTA / warp / instruction tile
global-to-shared copy
MMA mainloop
pipeline schedule
epilogue
```

CuTe 进一步用 Layout 描述数据和线程的多维映射。它适合在已经理解 WMMA、MMA tile 和流水线之后学习，而不是作为 Tensor Core 的第一个入口。

### Triton `tl.dot`

Triton 用块级语义表达矩阵乘：

```python
acc += tl.dot(a, b)
```

编译器根据输入类型、layout、shape 和目标架构决定是否下沉到 Tensor Core。学习 CUDA Tensor Core 后，应尝试查看 Triton 生成的 IR/PTX，回答 `tl.dot` 最终选择了什么 MMA 指令。

## 最小 WMMA 示例

下面只计算一个 `16 x 16 x 16` tile，用于理解 API，不是完整高性能 GEMM：

```cpp
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_tile(const half* A,
                          const half* B,
                          float* C,
                          int lda,
                          int ldb,
                          int ldc) {
    // 该示例只允许一个 warp 参与。
    if (threadIdx.x >= 32) return;

    wmma::fragment<wmma::matrix_a,
                   16, 16, 16,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,
                   16, 16, 16,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,
                   16, 16, 16,
                   float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}
```

这个示例隐含了严格前提：

- A/B/C 指针和 leading dimension 满足 WMMA 对齐要求。
- A 是 row-major，B 是 column-major。
- M/N/K 都正好是 16，没有边界 tile。
- 全 warp 一致执行 `load/mma/store`，不能让部分 lane 跳过。
- C 使用 FP32 accumulator 输出。

实际 GEMM 还需要解决多 tile、多 warp、K 循环、边界和流水线。

## 为什么必须全 warp 一致执行

WMMA/MMA 是 warp collective。参与线程必须在相同控制流路径上执行相同操作，并提供一致的参数。如果只有部分 lane 调用 `mma_sync`，行为未定义，甚至可能挂起。

安全模式：

```cpp
if (warp_has_work) {       // 条件对整个 warp 都相同
    wmma::mma_sync(...);
}
```

危险模式：

```cpp
if (lane_id < 16) {       // warp 内条件不同
    wmma::mma_sync(...);
}
```

边界通常通过：

- 把不完整 tile 先带 predicate 地搬到 shared memory，越界位置填 0。
- 再让整个 warp 对完整的 shared-memory tile 执行 WMMA。

## 数据流比计算指令更重要

一个完整 Tensor Core mainloop 的数据流是：

```text
global memory
  -> shared memory stage
  -> warp fragment registers
  -> mma.sync accumulator registers
  -> epilogue
  -> global memory
```

Tensor Core 提高计算吞吐后，以下问题会更加突出：

- global memory 搬运跟不上 MMA 消费速度。
- shared memory layout 导致 bank conflict。
- `ldmatrix` 访问模式不匹配。
- accumulator 占用大量寄存器，降低 occupancy。
- K mainloop 没有 software pipeline，Tensor Core 等待数据。
- M/N 很小，CTA 数量不足或边界浪费严重。

因此 Tensor Core GEMM 的核心不只是调用 `mma_sync`，而是围绕它设计层次化 tiling 和数据流水线。

## Epilogue

MMA 完成后，accumulator 仍分散在 warp 的寄存器里。Epilogue 负责：

- 应用 `alpha * accumulator + beta * C`。
- 转换成 FP16/BF16/FP32 输出。
- 合并 bias、activation、clamp 或 scale。
- 重新组织数据，实现合并写回。

LLM 中常见可融合形式：

```text
GEMM + Bias
GEMM + Bias + SiLU
GEMM + Scale / Dequantize
```

融合 epilogue 的收益通常来自减少中间 tensor 和 kernel launch，而不是减少 GEMM 的乘加数量。

## 如何确认真的用了 Tensor Core

不要只根据源代码里的 `wmma` 或 `tl.dot` 判断。至少做三层验证：

1. **结果正确**：与 FP32/FP64 reference 比较误差。
2. **指令生成**：用 `cuobjdump --dump-sass` 或 Nsight Compute 查看 HMMA/Tensor Core 指令。
3. **运行时利用率**：查看 Tensor Core pipeline、active cycles、stall 和 memory throughput。

还要检查：

- 编译目标是否为 `sm_86`。
- shape 和 leading dimension 是否满足要求。
- 实际输入类型和 compute type 是否匹配预期。
- library 是否因为对齐或小 shape 退回其他 kernel。

## 和编译器 Lowering 的连接

高层矩阵乘会经历：

```text
matmul / tl.dot
  -> tile M/N/K
  -> choose CTA and warp layouts
  -> stage operands in shared memory
  -> map warp tile to MMA fragments
  -> emit mma.sync
  -> emit SASS HMMA
```

这里的关键不是简单的“一条 op 替换成一条指令”。编译器必须先决定 shape、layout、memory space 和 schedule，才知道一个高层 matmul 能否合法、高效地下沉到目标 MMA 指令。

## 学完后的检查问题

- Tensor Core 与普通 CUDA Core FMA 的执行粒度有什么区别？
- CTA tile、warp tile、MMA tile 分别解决什么问题？
- WMMA `16x16x16` 与 PTX `m16n8k16` 为什么不是同一层概念？
- 为什么 fragment 不能按普通数组理解？
- 为什么 Tensor Core kernel 仍然可能 memory-bound？
- accumulator 为什么会显著增加寄存器压力？
- 什么条件会让 `tl.dot` 或 cuBLAS 没有走预期的 Tensor Core 路径？

## 参考资料

- [CUDA C++ Programming Guide: Warp Matrix Functions](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-matrix-functions)
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [PTX ISA: Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
