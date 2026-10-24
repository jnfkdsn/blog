---
order: 40
title: 算子分类与实现模式
updated: 2026-10-17
---

# 算子分类与实现模式

本章属于 [Ascend C 算子编程](./)。示例沿用原笔记的 CANN 社区版 8.3.RC1.alpha001；工程接口与芯片约束需以实际版本为准。

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
