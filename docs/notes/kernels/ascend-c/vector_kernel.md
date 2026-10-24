---
order: 30
title: Vector 算子与 Host Tiling
updated: 2026-10-17
---

# Vector 算子与 Host Tiling

本章属于 [Ascend C 算子编程](./)。示例沿用原笔记的 CANN 社区版 8.3.RC1.alpha001；工程接口与芯片约束需以实际版本为准。

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
