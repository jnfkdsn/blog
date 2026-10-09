---
order: 80
title: GPU / NPU 映射与目标后端
updated: 2026-10-04
---

# GPU / NPU 映射与目标后端

沿计算、搬运、同步与启动几条关系理解设备执行，同时保留 GPU 和 NPU 两条路线。

算法已经分块后，还要决定每块由谁执行、数据怎样到达计算单元、消费者何时可以读取，以及最终怎样被 Host 启动。本模块沿这条过程展开五篇。

## 阅读前置

循环与内存、基本 lowering；设备硬件概念随选定目标补充。

## 阅读顺序

| 顺序 | 章节 | 贯穿案例与需要解释的过程 |
|---|---|---|
| 1 | [并行工作到执行层次](./parallel_mapping) | scf.parallel/forall 到 launch 的真实映射，先证明有效工作不丢失。 |
| 2 | [设备布局与数据搬运](./layout_transfer) | 转置的地址矛盾、线程分布与共享存储 padding。 |
| 3 | [同步、异步依赖与流水](./pipelining) | 从两个 slot 推导数据就绪与复用依赖，连接三类异步场景。 |
| 4 | [Host / Device 编译与启动](./host_device) | 从捕获参数到 NVVM/设备对象，再到 runtime 调用与启动 ABI。 |
| 5 | [NPU 表示与执行约束](./npu_constraints) | AscendNPU-IR VecAdd 的 GM/UB、执行 pipe、自动同步和内存规划。 |

## 完成边界与衔接

先读 [GPU 三篇](../../dialects/gpu/)理解 launch 与存储；本模块再说明表示怎样进入目标。NPU 篇固定项目源码与 LLVM 子模块版本，保留硬件代际边界。

配套 [16 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/16-gpu-execution)提供实际转换、依赖模型与有限修改任务。编译证据、源码推演与硬件运行分别标注；目前没有 GPU/NPU 性能结论。
