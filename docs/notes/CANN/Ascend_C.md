---
order: 1
title: Ascend C 算子编程知识库
updated: 2026-06-30
tags: [cann, ascend-c, operator, ai-core, kernel, npu]
status: draft
---

# Ascend C 算子编程知识库

相关路线：[GPU 编程与算子优化知识地图](/notes/gpu-programming)  
官方版本：CANN 社区版 8.3.RC1.alpha001。实际工程以本机 CANN 版本、芯片型号和 API 约束为准。

内容范围：

- NPU/AI Core 计算、存储、搬运模型。
- 耦合架构与 AIC/AIV 分离架构。
- 核间数据并行、核内向量/矩阵计算、片上流水、Host 侧 tiling。
- `GlobalTensor`、`LocalTensor`、`TPipe`、`TQue`、`TBuf`、`DataCopy`、Vector/Cube API。
- 正确性验证、性能定位、优化检查项。

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

## 核心数据结构

### GlobalTensor

`GlobalTensor<T>` 表达 GM 上的一段 typed view。常见用法是在 `Init` 里根据 `blockIdx` 设置当前核负责的起始地址：

```cpp
xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + offset, validLen);
```

使用规则：

- `SetGlobalBuffer` 的长度要和当前核可访问范围一致。
- `GlobalTensor` 上的偏移单位是元素。
- 多输入、多输出算子要保证每个 tensor 使用相同的分片策略，除非算法本身需要不同布局。

### LocalTensor

`LocalTensor<T>` 表达片上 Local Memory 的 tensor view。它通常来自 `TQue::AllocTensor` 或 `TBuf::Get`。

使用规则：

- `LocalTensor` 内容可能是未初始化的，计算前要确保由 `DataCopy`、`Duplicate` 或其他指令写入。
- `LocalTensor` 使用结束后按来源释放或复用。
- `LocalTensor` 的切片可以通过偏移表达，但要自己保证偏移和 buffer 大小合法。

### TPipe / TQue / TBuf

`TPipe` 是片上资源管理入口，负责给 Queue/Buffer 分配内存和事件资源。

`TQue<TPosition, depth>` 是流水 stage 之间传递 `LocalTensor` 的队列。典型动作是：

1. `AllocTensor`：申请一块 local buffer。
2. `DataCopy` 或计算：填充这块 buffer。
3. `EnQue`：把 buffer 交给下一个 stage。
4. `DeQue`：下一个 stage 取出 buffer。
5. `FreeTensor`：使用结束后释放。

`TBuf<TPosition>` 更适合临时变量。它只能管理内存，不做队列入队/出队，也就不承担 stage 间同步。

使用规则：

- 普通 `CopyIn -> Compute -> CopyOut` 流水里，输入和输出用 `TQue`。
- 计算中的临时 workspace 用 `TBuf`。
- 队列 `depth` 不等于 double buffer 个数。多数非原地操作场景 `depth=1` 更常见。
- `pipe.InitBuffer(queue, BUFFER_NUM, bytes)` 里的 `BUFFER_NUM=2` 常用于 double buffer，但也会消耗更多事件和 local memory。

### TPosition

`TPosition` 用逻辑位置隐藏不同芯片的片上物理存储差异。Vector 常见位置：

| 位置 | 含义 |
|---|---|
| `VECIN` | Vector 输入 |
| `VECCALC` | Vector 临时计算 |
| `VECOUT` | Vector 输出 |

Cube 常见位置：

| 位置 | 含义 |
|---|---|
| `A1/B1` | 矩阵计算 L1 输入 |
| `A2/B2` | Cube L0 输入 |
| `C1/C2` | Bias 或中间数据 |
| `CO1/CO2` | Cube 输出中间/最终位置 |

## Vector 算子模板

Elementwise 算子的基本结构：每个核处理一段连续数据，每次处理一个 tile，通过 queue 串联搬入、计算、搬出。

```cpp
#include "kernel_operator.h"

#define GM_ADDR __gm__ uint8_t*

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLen)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockLen = (totalLen + blockNum - 1) / blockNum;
        uint32_t offset = blockIdx * blockLen;
        uint32_t remain = offset < totalLen ? totalLen - offset : 0;
        this->blockLen = remain > blockLen ? blockLen : remain;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + offset, this->blockLen);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y) + offset, this->blockLen);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(z) + offset, this->blockLen);

        pipe.InitBuffer(xQue, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(yQue, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(zQue, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        uint32_t loop = (blockLen + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loop; ++i) {
            uint32_t len = GetTileLen(i);
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline uint32_t GetTileLen(uint32_t tileIdx) const
    {
        uint32_t offset = tileIdx * TILE_LENGTH;
        uint32_t remain = blockLen - offset;
        return remain > TILE_LENGTH ? TILE_LENGTH : remain;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t len)
    {
        AscendC::LocalTensor<half> xLocal = xQue.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = yQue.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[tileIdx * TILE_LENGTH], len);
        AscendC::DataCopy(yLocal, yGm[tileIdx * TILE_LENGTH], len);
        xQue.EnQue(xLocal);
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<half> xLocal = xQue.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = yQue.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = zQue.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        xQue.FreeTensor(xLocal);
        yQue.FreeTensor(yLocal);
        zQue.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t len)
    {
        AscendC::LocalTensor<half> zLocal = zQue.DeQue<half>();
        AscendC::DataCopy(zGm[tileIdx * TILE_LENGTH], zLocal, len);
        zQue.FreeTensor(zLocal);
    }

private:
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> zQue;
    uint32_t blockLen = 0;
};

extern "C" __global__ __aicore__
void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLen)
{
    KernelAdd op;
    op.Init(x, y, z, totalLen);
    op.Process();
}
```

这份模板检查点：

- `offset` 和 `blockLen` 是否覆盖所有元素且无重叠？
- 尾块 `len` 是否小于等于 `TILE_LENGTH`？
- `InitBuffer` 的单位是字节，`DataCopy` 里的连续搬运长度按 API 语义填写，避免把元素数和字节数混用。
- `AllocTensor/FreeTensor` 是否成对？
- `EnQue/DeQue` 顺序是否和流水依赖一致？

## Tiling

Tiling 将运行时 shape、dtype、format、芯片资源和算法参数转换成 kernel 侧参数。

Host 侧 tiling 一般负责：

- 根据输入 shape 计算 `totalLen`、每核数据量、tile 数量、尾块长度。
- 根据芯片 AIV/AIC 核数决定 `blockDim`。
- 根据 UB/L1/L0 容量决定 `TILE_LENGTH`、double buffer 个数、临时 buffer 大小。
- 根据 dtype/format 决定使用 Vector、Cube、高阶 API 或特殊数据搬运参数。
- 需要动态 shape 时，把这些结果写入 `TilingData`，kernel 侧读取后执行。

Tiling 参数检查：

1. 算子是 memory-bound 还是 compute-bound？
2. 单核一次 tile 需要多少输入、输出、临时 local memory？
3. double buffer 后 local memory 是否仍能放下？
4. 每个 tile 的搬运字节数是否满足对齐要求？
5. 尾块是用单独分支、mask，还是 DataCopy 参数处理？
6. `blockDim` 是按 AIV 核数、AIC 核数，还是 AIV/AIC 组合数设置？

## 工程化开发流程

有两种常见入口：

| 方式 | 适用场景 | 特点 |
|---|---|---|
| Kernel 直调 | 独立验证 kernel、写 sample、做性能实验 | Host 侧手写 AscendCL 初始化和 kernel launch |
| 自定义算子工程 | 接入框架、动态 shape、正式算子包 | 需要原型注册、shape 推导、tiling、编译部署 |

自定义算子工程通常包含：

| 目录 | 内容 |
|---|---|
| `op_kernel` | Ascend C kernel 实现 |
| `op_host` | 原型注册、shape 推导、tiling、信息库 |
| `framework` | 框架适配层 |
| `CMakeLists.txt` / `build.sh` | 编译入口 |

工程化算子流程：

1. 写清楚算子数学定义、输入输出 shape、dtype、format。
2. 实现最小 kernel 直调版本，验证核心数据流。
3. 把静态常量改成 tiling 参数，支持动态 shape。
4. 用 CPU golden 或已有框架算子做正确性比对。
5. 接入 `op_host` 的 shape/tiling/proto 注册。
6. 执行 profiling，定位 GM 搬运、Vector/Cube 利用率、Scalar 控制、同步等待等瓶颈。
7. 逐步加入 double buffer、多核均衡、格式转换、融合计算。

## 算子分类方法

### Elementwise

特点：

- 输入输出等长或可 broadcast。
- 通常 memory-bound。
- 优化方向：减少 GM 访问、提高搬运对齐、形成 CopyIn/Compute/CopyOut 流水。

模板：

```text
GM x/y -> VECIN
Vector Add/Mul/Relu/Cast -> VECOUT
VECOUT -> GM z
```

### Reduce

特点：

- 需要跨元素聚合，通常分为单核内 reduce 和多核间二次 reduce。
- 关注 reduce 轴、输出 shape、临时 workspace、原子/同步或二阶段 kernel。

常见策略：

- 小 reduce 轴：单核处理一行或一个分组。
- 大 reduce 轴：每核处理一段，先写 partial，再用第二个 kernel 汇总。
- Softmax 可拆成 `max -> exp/sum -> normalize`，重点是避免重复 GM 读写和保证数值稳定。

### Matmul / Cube

特点：

- compute-bound 潜力高，但数据格式、tiling 和 Cube pipeline 更复杂。
- Matmul 高阶 API 可作为 baseline；底层 L0 搬运适合后续深入优化。

典型流程：

```text
SetTensorA / SetTensorB / SetBias
Iterate
GetTensorC
End
```

关注点：

- `A1/B1/A2/B2/CO1/CO2` 的逻辑位置。
- ND/NZ 等格式对搬运和 Cube 指令的影响。
- AIC/AIV 分离架构下，Cube 输出接 Vector 后处理时的数据流。

### 融合算子

融合用于减少 GM 往返。例如 `Matmul + Bias + Relu`：

```text
GM A/B -> Cube -> local C
local C -> Vector activation
Vector output -> GM
```

融合条件：

- 中间结果是否很大，写回 GM 成本是否高？
- 融合后 local memory 是否够？
- Cube 与 Vector 的 pipeline 是否能平衡？
- 精度转换、量化、layout transform 是否能随路完成？

## 性能优化清单

### 1. 多核切分

- `blockDim` 通常按目标计算单元物理核数设置，再根据数据量和波次调整。
- 每核数据量太小会导致调度和流水开销占比过高。
- 尾核数据量差异太大时会造成长尾，必要时调整切分方式。

### 2. 搬运效率

- GM 访问连续。
- tile 大小按 `32B`、datablock、cacheline、dtype size 对齐。
- 避免同一份 GM 数据在多个 stage 反复搬入。
- 不连续搬运要明确 `DataCopyParams` 的 repeat、blockLen、stride 单位。

### 3. 流水并行

- `CopyIn/Compute/CopyOut` 三段耗时保持平衡。
- double buffer 可以隐藏搬运延迟，但会增加 local memory 和事件消耗。
- `TQue` 数量、buffer 数量受硬件事件资源限制；过多 queue 会增加资源压力。

### 4. Scalar 开销

- 减少复杂 `if/else`、除法、取模和深层循环控制。
- Host tiling 侧可计算的参数放入 tiling data，避免 kernel 内重复计算。
- 尾块处理集中，主循环保持简单。

### 5. Vector 指令

- 使用 Ascend C 提供的 Vector API，比如 `Add`、`Mul`、`Cast`、`Duplicate`、`ReduceSum`。
- 注意 API 的 dtype、mask、repeat、对齐和通路约束。
- 原地操作能减少 buffer 和 copy，但要确认输入输出别名不会破坏依赖。

### 6. Cube 指令

- GEMM 类算子先用 Matmul 高阶 API 建 baseline。
- 根据 M/N/K、dtype、format、芯片 AIC 数调 tiling。
- 融合 Vector 后处理时，检查 Cube 输出到 Vector 输入的路径是否多绕 GM。

### 7. 同步与依赖

- Queue 解决单核内 stage 间通信和同步。
- 多核间默认无依赖；存在依赖时使用多核同步 API，并评估开销。
- 不同核写同一输出区域存在数据竞争风险，除非使用明确的归约协议。

## 正确性验证

验证顺序：

1. 最小 shape：比如 1 个 tile、1 个核、无尾块。
2. 尾块 shape：`totalLen` 不是 `blockDim * TILE_LENGTH` 的整数倍。
3. 多核 shape：核数大于 1，且每核数据量不同。
4. dtype 覆盖：`half`、`float`、`bfloat16_t`、量化类型。
5. format 覆盖：ND/NZ/5HD 等实际支持格式。
6. 随机数据 + 边界值：0、负数、大数、NaN/Inf，如果算子语义涉及。
7. 与 CPU golden 或框架内置算子对比，分别记录绝对误差和相对误差。

常见错误：

- `DataCopy` 长度单位写错。
- `InitBuffer` 按元素数传了，实际需要字节数。
- 尾块读写越界。
- `AllocTensor` 后忘记 `FreeTensor`。
- `EnQue/DeQue` 顺序不匹配，导致等待或结果错乱。
- `blockDim` 超过硬件/场景合理范围。
- kernel 侧对 GM 指针类型转换错误。

## 调试和 Profiling

调试时先分层定位：

| 现象 | 检查项 |
|---|---|
| 编译失败 | CANN 环境变量、soc_version、头文件、API 支持型号 |
| 运行挂住 | Queue 顺序、Alloc/Free 数量、越界访问、多核同步 |
| 结果全错 | GM 地址 offset、dtype cast、DataCopy 方向 |
| 少量元素错 | 尾块、对齐、mask、stride、format |
| 性能差 | GM 搬运次数、tile 大小、double buffer、blockDim、Scalar 分支 |

Profiling 指标：

- GM 读写带宽是否接近瓶颈。
- Vector/Cube 利用率是否偏低。
- MTE 搬运和计算是否重叠。
- Scalar 指令占比是否异常。
- 每个核工作量是否均衡。

## 实践模板

记录模板：

```md
## 算子定义
- 输入/输出：
- dtype/format：
- 数学公式：
- 与框架内置算子的差异：

## Baseline
- 多核切分：
- tile 策略：
- 片上数据流：
- Host tiling 参数：

## 正确性
- golden：
- shape 覆盖：
- 误差阈值：
- 已知边界：

## 性能
- 测试芯片/CANN版本：
- blockDim：
- tile length：
- profiling 结论：
- bottleneck：

## 优化记录
- v1：
- v2：
- v3：

## 还没搞懂的
```

## 实践顺序

1. 跑通 HelloWorld 和 Vector Add，熟悉 kernel 入口、Host 调用和工程构建。
2. 写 Elementwise 三件套：Add、Mul、Relu/Cast，掌握 `DataCopy + TQue`。
3. 写 ReduceSum/ReduceMax，理解单核归约、多核 partial、二阶段汇总。
4. 写 Softmax，练习多次遍历、数值稳定和 GM 访问优化。
5. 学 Matmul 高阶 API，理解 Cube 数据流和 tiling。
6. 做一个融合算子，比如 `Matmul + Bias + Activation`。
7. 系统补 Profiling，形成“指标 -> 瓶颈 -> 改动”的闭环。

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
