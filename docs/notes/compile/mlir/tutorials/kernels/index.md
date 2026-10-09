---
order: 40
title: 归约与算子贯通
updated: 2026-10-04
---

# 归约与算子贯通

算法贯通把前面的表示、分析、内存和目标机制放回一项完整计算。先建立可以核对的结果，再讨论存储和调度，最后进入目标优化。编译成功、数值满足条件和性能改善分别需要证据。

## 已有主线

| 顺序 | 章节 | 完成的理解过程 |
|---|---|---|
| 1 | [归约与 Softmax 的结构化表示](./reduction_softmax) | 从稳定算法推导 max/exp/sum/normalize，解释归约初值与广播；完整 IR 连到 CPU 数值 |
| 2 | [Softmax 的存储与融合](./softmax_storage) | 从四个中间分配进入按行处理，融合 exp/sum 并使用输出暂存；解释访问变化与数值前提 |
| 3 | [Matmul 的分块、累加与复用](./matmul) | 三维分块、跨 K 累加、尾块与展开；区分子视图和真正的数据搬运 |
| 4 | [Attention 的计算与中间存储](./attention) | QK、归一化与 WV，追踪中间矩阵及 mask |
| 5 | [在线 Softmax 与分块 Attention](./online_attention) | 推导 m/l/u 状态与重新缩放，用 CPU 执行核验存储变化和数值 |

前置为 Tensor/Linalg、标准 Bufferization 和 CPU 执行；指数与索引可回查 [Math/Index](../../dialects/math/)。[13-softmax 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/13-softmax)提供实际阶段、两实现对照和有限练习，正文独立完成核心解释。

## 算法与设备边界

| 章节 | 依赖与核心过程 | 状态 |
|---|---|---|
| Matmul 的分块与复用 | 在 tiling、布局与 Vector 基础上推导局部累加与数据复用 | 已有三种调度的 CPU 数值证据 |
| Attention 的分块计算 | 连接归约、在线状态与设备存储；说明 FlashAttention 的计算重组 | 已有两种存储组织与错误递推反例；高性能设备实现属于独立项目 |

当前 Softmax 提供 CPU 正确性与存储基线，尚未宣称计时加速或设备性能。进一步的形状、精度、Mask 和后端选择沿具体项目扩展，长期范围见[覆盖表](../../coverage)。
