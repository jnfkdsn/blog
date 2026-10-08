---
order: 100
title: Vector：向量计算与传输
updated: 2026-10-04
---

# Vector：向量计算与传输

分块确定局部工作范围后，Vector 表达一组元素怎样共同计算、怎样连接存储，以及哪些元素需要合并。先看局部值，再看尾块，最后进入归约和矩阵小块。

| 顺序 | 章节 | 阅读时追踪什么 |
|---|---|---|
| 1 | [向量值与元素组织](./values) | 2×4 小块读取、加一、转置、写出；区分值、布局与机器表示 |
| 2 | [Transfer、Mask 与边界](./transfer_mask) | 长度 10、宽度 4 的最后两项；padding、mask、有效地址与标量余块 |
| 3 | [归约、收缩与目标形态](./reduction_contract) | 四项归约和 2×3×2 矩阵累加；次序、映射与目标拆解 |

前置为 Linalg、MemRef 和分块；先不要求熟悉硬件指令。之后进入[自动向量化](../../compiler/optimization/vectorization)、[Transform 调度](../../compiler/optimization/transform)与[性能分析](../../compiler/optimization/performance)，连接“表示什么”和“怎样产生、怎样评估”。

本模块提供固定宽度 CPU 示例和限定的 scalable vector 定位，不把逻辑 lane 直接当 GPU 线程或物理寄存器。设备分布和目标映射在 [GPU/NPU 后端](../../compiler/targets/)继续。
