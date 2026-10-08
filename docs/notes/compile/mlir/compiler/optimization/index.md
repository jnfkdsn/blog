---
order: 70
title: 结构化优化与调度
updated: 2026-10-04
---

# 结构化优化与调度

结构化优化改变工作怎样组织：把完整迭代域划成 tile，让 producer/consumer 在局部相遇，调整遍历顺序，再选择向量或设备执行。合法性来自域、访问、依赖和数值契约，收益则需要结合目标与测量。

## 已有阅读主线

| 顺序 | 章节 | 解释的具体过程 |
|---|---|---|
| 1 | [Tiling 与边界处理](./tiling) | 5×7 划为 2×3 块，追踪最后 1×1 余块与 Tensor 切片/插回 |
| 2 | [Fusion 与计算重排](./fusion) | 两个逐元素 generic 合成一条数据流，再解释邻域、重复计算与归约边界 |
| 3 | [循环交换、并行化与归约次序](./interchange) | 从改变访问顺序进入依赖反例，区分布局、线程更新和浮点归约 |
| 4 | [向量化与目标降低](./vectorization) | 同一 generic 分块为 4，生成 mask/transfer，继续降低到 LLVM |
| 5 | [Transform Dialect 与显式调度](./transform) | 定位对象、消费句柄、处理失败与维持映射有效性 |
| 6 | [调度选择与性能解释](./performance) | 固定 x86-64 对比三种实现，连接汇编、计时与有限结论 |

已理解 Linalg、内存表示与基本分析即可进入。[Affine 三篇](../../dialects/affine/)按需要补映射、域和依赖推理；[Softmax 存储](../../tutorials/kernels/softmax_storage)提供另一项已完成的融合案例。

Vector 的值语义、边界与收缩在 [Vector 三篇](../../dialects/vector/)解释，可在第 4 篇前阅读。这里侧重编译器如何产生和调度这些表示，避免重复定义操作。

可选 [14 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)打印分块/融合/交换、依赖与数值差异；[15 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/15-vector-transform)连接向量化、Transform 协议与计时，各保留两个有限任务。性能结论限定在实际配置，不由 tile 数或向量宽度推断普遍加速。
