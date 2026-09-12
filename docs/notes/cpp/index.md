---
order: 0
title: C++ 工程基础
updated: 2026-09-09
---

# C++ 工程基础

面向 CUDA 算子、推理 runtime 和 LLVM/MLIR 开发。目标是能读懂接口、判断所有权、写出正确的并发代码，并定位性能和构建问题。以 C++17 为示例基线；C++20 特性单独标注，实际采用版本以项目工具链为准。

## 学习顺序

| 顺序 | 主题 | 学完后应该能做什么 |
| --- | --- | --- |
| 1 | [类型、引用与类](/notes/cpp/basics) | 解释参数传递、const、初始化和 lambda 捕获 |
| 2 | [对象生命周期与内存布局](/notes/cpp/object_lifetime) | 找到悬空引用、越界、对齐和大小计算问题 |
| 3 | [所有权、RAII 与移动语义](/notes/cpp/ownership) | 设计资源包装和所有权明确的接口 |
| 4 | [STL、算法与非拥有视图](/notes/cpp/stl) | 按访问模式选容器，判断失效和分配成本 |
| 5 | [模板与编译期编程](/notes/cpp/templates) | 阅读模板接口，区分编译期配置和运行时参数 |
| 6 | [编译、链接与调试](/notes/cpp/build_debug) | 构建多文件项目并定位内存错误 |
| 7 | [多线程与任务队列](/notes/cpp/concurrency) | 用锁和条件变量实现可关闭的有界队列 |
| 8 | [C++ 并发内存模型](/notes/cpp/memory_model) | 用 happens-before 解释共享数据访问的正确性 |

前六篇与 CUDA/MLIR 实践交替进行，不必学完全部并发内容才开始写 kernel。多线程先学锁和条件变量，再学原子操作；复杂无锁数据结构留到有具体需求时。

## 三条应用分支

- **CUDA / CUTLASS：**基础、生命周期、模板 → [CUDA C++ 导读](/notes/cuda/cpp) → [CUDA 基础语法](/notes/cuda/cuda_basic_syntax) → [异步流水线](/notes/cuda/async_pipeline)。关注 host/device 边界、布局和异步资源使用。
- **推理 runtime：**所有权、STL、构建 → 多线程、内存模型 → [推理系统笔记](/notes/infer/)。关注任务取消、背压、队列关闭和 buffer 复用。
- **LLVM / MLIR：**类与模板、视图生命周期、编译链接 → [编译器学习笔记](/notes/compile/)。随后学习 LLVM 的 `StringRef`、`ArrayRef`、`SmallVector`、`DenseMap` 和 `isa/cast/dyn_cast`；专用 API 留在编译器目录。

## 阶段验收

1. 实现只能移动的 buffer，验证移动构造、移动赋值、自移动和空对象状态。
2. 用容器保存任务，解释每次插入是否会使已有指针或迭代器失效。
3. 实现有界阻塞队列，验证关闭时唤醒等待者、排空旧任务、多生产者/消费者不丢任务。
4. 用 CMake 构建多文件程序，分别用 ASan/UBSan 和 TSan 检查适用的 CPU 路径。
5. 封装一个 CUDA 算子，说明 launch 返回与执行完成的区别，以及谁持有输入输出直到使用结束。

## 内容边界

这里的“对象与存储”讨论生命周期和布局；“并发内存模型”讨论线程间顺序。CUDA 的 global/shared memory、stream、event 和同步 scope 在 [CUDA 目录](/notes/cuda/) 展开。
