---
order: 50
title: 性能分析、正确性与实践
updated: 2026-10-17
---

# 性能分析、正确性与实践

本章属于 [Ascend C 算子编程](./)。示例沿用原笔记的 CANN 社区版 8.3.RC1.alpha001；工程接口与芯片约束需以实际版本为准。

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
