---
order: 10
title: AI Core 架构与执行模型
updated: 2026-10-17
---

# AI Core 架构与执行模型

本章属于 [Ascend C 算子编程](./)。示例沿用原笔记的 CANN 社区版 8.3.RC1.alpha001；工程接口与芯片约束需以实际版本为准。

## NPU 硬件架构先导

Ascend C 暴露 AI Core 上计算单元、存储单元和搬运单元之间的数据流。`TQue`、`TPipe`、`DataCopy`、`TPosition` 都对应片上资源管理、数据搬运和流水同步。

### AI Core 组成

AI Core 是昇腾 NPU 上执行算子的计算核心。一个 AI Core 包含三类计算资源：

- `Scalar`：负责地址计算、循环控制、分支判断，并把 Vector、Cube、DMA、同步等指令发射给对应执行单元。复杂分支和大量标量计算会增加 Scalar 开销。
- `Vector`：负责向量运算，典型数据流是 `GM -> UB -> Vector -> UB -> GM`。Elementwise、激活函数、Cast、Reduce、部分数据重排主要走这条线。
- `Cube`：负责矩阵/张量计算，典型数据流是 `GM -> L1 -> L0A/L0B -> Cube -> L0C -> Fixpipe -> GM/L1`。Matmul、Conv、GEMM 融合主要走这条线。

存储抽象：

- `Global Memory`：核外内存，对应 `GlobalTensor<T>`，Host 传入的 `GM_ADDR` 最终会被解释成 GM 上的 tensor。
- `Local Memory`：AI Core 内部存储，对应 `LocalTensor<T>`，实际可能映射到 UB、L1、L0A、L0B、L0C、BiasTable、Fixpipe Buffer 等位置。

数据搬运由 DMA/MTE 单元完成。性能受 `DMA 搬运`、`Vector/Cube 计算`、`结果写回` 的流水并行程度影响。

### 耦合架构和分离架构

不同昇腾产品上的 AI Core 组织方式分为两类：

| 架构 | 特点 | 编程影响 |
|---|---|---|
| 耦合架构 | Cube 和 Vector 同核部署，共享 Scalar，统一加载代码段 | 一个 kernel 内理解为同一个 AI Core 上的矩阵、向量、搬运协同 |
| 分离架构 | AI Core 被拆成 AIC 与 AIV：AIC 偏矩阵计算，AIV 偏向量计算，各有 Scalar，可独立加载代码段 | Matmul + Vector 后处理、量化/反量化、融合算子时要关注 AIC/AIV 数据交互 |

分离架构：

| 核 | 主要单元 | 典型存储 | 典型工作 |
|---|---|---|---|
| `AIV` | Vector、Scalar、MTE2/MTE3 | GM、UB | Elementwise、Reduce、激活、Cast、搬运/格式处理 |
| `AIC` | Cube、Scalar、MTE1/MTE2、Fixpipe | GM、L1、L0A、L0B、L0C、BT/FP Buffer | Matmul、Conv、矩阵块计算、Fixpipe 后处理 |

层次关系：`SPMD` 描述多个 AI Core/AIV/AIC 运行实例如何切分数据；`Vector/Cube` 描述单个运行实例内部如何使用 SIMD/矩阵计算资源；`Pipe/Queue` 描述搬运和计算如何在片上流水并行。

## 总体地图

Ascend C 算子开发可以分成四层：

| 层次 | 关注点 | 典型内容 |
|---|---|---|
| Host 侧工程 | 算子原型、shape 推导、tiling、workspace、kernel launch | `op_host`、`TilingData`、`blockDim`、AscendCL 调用 |
| Kernel 入口 | 多核并行、入参解释、GM 地址切分 | `__global__ __aicore__`、`GM_ADDR`、`GetBlockIdx` |
| 片上流水 | GM/LM 搬运、队列同步、double buffer | `DataCopy`、`TPipe`、`TQue`、`CopyIn/Compute/CopyOut` |
| 计算指令 | Vector、Cube、Scalar 计算 | `Add`、`Mul`、`ReduceSum`、`Matmul`、`Softmax` |

与 CUDA 的概念对应：

| CUDA 直觉 | Ascend C 对应物 | 差异 |
|---|---|---|
| kernel launch | `<<<blockDim, l2ctrl, stream>>>` | `blockDim` 表示启动多少个逻辑核实例；与 CUDA grid/block 二级线程层次不同 |
| `blockIdx.x` | `AscendC::GetBlockIdx()` | 每个 AI Core 执行同一份代码，通过 block index 切分数据 |
| thread/warp | 无完全等价的显式抽象 | Ascend C 常写 tensor 级 Vector/Cube 指令，而非逐 lane/逐 warp 写线程逻辑 |
| global memory | Global Memory / `GlobalTensor` | kernel 侧通常先把 GM 数据搬到 Local Memory 再计算 |
| shared memory / registers | Local Memory / `LocalTensor` | Ascend C 用 `TPosition` 抽象 UB/L1/L0 等片上存储 |
| memcpy async + sync | `DataCopy` + Queue/Pipe/Barrier | 搬运、计算、写回在不同执行单元上流水并行 |

## 执行模型：SPMD、SIMD、SIMT 的边界

`SPMD`、`SIMD`、`SIMT` 属于不同层面的概念。

| 层次 | 更准确的说法 | 在 Ascend C 里怎么看 |
|---|---|---|
| 核间并行 | SPMD 数据并行 | 多个 AI Core/AIV/AIC 运行同一份 kernel，通过 `block_idx` 处理不同数据分片 |
| 核内向量计算 | SIMD / Vector 风格 | 一条 Vector API 对 `LocalTensor` 上的一批元素执行同一类操作 |
| 核内矩阵计算 | Tensor/Cube 计算 | Cube 单元对矩阵 tile 做高吞吐矩阵运算；与 CUDA warp 线程模型不同 |
| 核内流水 | 异步指令流 + Queue/Pipe 同步 | Scalar 发射搬运、计算、同步指令，DMA/Vector/Cube 异步并行执行 |
| CUDA 类比 | SIMT 类比 | CUDA 暴露 thread/warp；Ascend C 更多暴露 tensor、queue、pipe 和矩阵/向量指令 |

执行层次：

```text
Host tiling
  -> 决定 blockDim / 每核数据 / tile 参数
多核 SPMD
  -> 每个 block_idx 处理一个或多个数据分片
核内 SIMD/Cube
  -> Vector 或 Matmul API 对 LocalTensor/tile 计算
片上流水
  -> CopyIn / Compute / CopyOut 通过 Queue/Pipe 串起来
```

说明：官方 Ascend C 文档将跨 AI Core 的编程模型称为 `SPMD`。`SIMT` 可用于和 CUDA 做类比，但不等同于 Ascend C 的显式线程接口。Ascend C 更准确的分层是：核间 `SPMD`，核内 `Vector/SIMD` 与 `Cube` 计算，片上 `Pipe/Queue` 流水。

### 核间：SPMD 数据并行

Ascend C 的核间并行使用 SPMD 思路：多个 AI Core 执行同一份 kernel 代码，每个核通过 `GetBlockIdx()` 获取自己的逻辑 ID，然后处理不同数据分片。

最常见的数据切分方式：

```cpp
uint32_t blockIdx = AscendC::GetBlockIdx();
uint32_t blockNum = AscendC::GetBlockNum();
uint32_t blockLen = (totalLen + blockNum - 1) / blockNum;
uint32_t offset = blockIdx * blockLen;
uint32_t validLen = offset < totalLen ? totalLen - offset : 0;
validLen = validLen > blockLen ? blockLen : validLen;
```

需要注意：

- `blockDim` 结合物理 AIV/AIC 核数、算子类型、数据量和 tiling 设置。
- 每个核处理连续数据片段，便于 GM 搬运合并和尾块处理。
- 多核之间默认没有隐式同步。不同核写同一块 GM 或存在跨核依赖时，需要重新设计数据划分，必要时使用多核同步 API。

### Kernel 入口

Kernel 函数是设备侧入口，通常写法如下：

```cpp
#include "kernel_operator.h"

#define GM_ADDR __gm__ uint8_t*

extern "C" __global__ __aicore__
void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLen)
{
    KernelAdd op;
    op.Init(x, y, z, totalLen);
    op.Process();
}
```

入口约束：

- 入参统一用 `GM_ADDR` 接 GM 指针，进入 `Init` 后再 `reinterpret_cast` 成实际类型。
- kernel 返回值必须是 `void`。
- kernel 内避免复杂对象生命周期管理；核心逻辑放到 `KernelXxx` 类的 `Init/Process/CopyIn/Compute/CopyOut`。
- Host 侧调用是异步的，需要用 stream 同步或事件判断执行完成。

## 官方资料

- [Ascend C 算子开发介绍](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)
- [HelloWorld 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0004.html)
- [基于自定义算子工程的算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html)
- [AI Core 基本架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [SPMD 模型](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0013.html)
- [核函数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0014.html)
- [抽象硬件架构](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
- [编程范式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0016.html)
- [Ascend C API 列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0003.html)
- [DataCopy 普通数据搬运](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0102.html)
- [TQue 简介](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0137.html)
