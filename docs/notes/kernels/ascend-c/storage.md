---
order: 20
title: Tensor、存储与流水资源
updated: 2026-10-17
---

# Tensor、存储与流水资源

本章属于 [Ascend C 算子编程](./)。示例沿用原笔记的 CANN 社区版 8.3.RC1.alpha001；工程接口与芯片约束需以实际版本为准。

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
