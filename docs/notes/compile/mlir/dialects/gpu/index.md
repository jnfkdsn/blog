---
order: 110
title: GPU：并行执行与设备操作
updated: 2026-10-04
---

# GPU：并行执行与设备操作

先把十个逐元素工作分给执行实例，再让四个线程通过共享存储交换数据，最后把整个计算接到 Host 提交与等待。三篇分别解释“谁来算、在哪里读写、何时可以消费”。

## 阅读前置

循环、内存表示；执行层次与目标映射可交错学习。

## 阅读顺序

| 顺序 | 章节 | 贯穿案例与需要解释的过程 |
|---|---|---|
| 1 | [Kernel 与执行层次](./execution) | 从十二个候选位置到十个有效点，解释 launch、block/thread 与边界。 |
| 2 | [地址空间与设备存储](./memory_spaces) | 四线程轮转中，global、workgroup、private 分别承载什么。 |
| 3 | [同步与异步操作](./synchronization) | 邻居读取的 barrier，以及复制/启动/等待之间的 token 依赖。 |

## 完成边界与衔接

能解释一个简单 GPU IR 的执行与访问；[目标后端模块](../../compiler/targets/)继续实际映射、布局、流水和 Host/Device 编译，并对照 NPU。

正文按 LLVM 20.1.8 核验；实际转换、逻辑模型与设备运行分开记录。完整代码和观察见 [16 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/16-gpu-execution)。当前环境未完成 GPU 执行或性能测量。
