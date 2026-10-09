---
order: 50
title: StableHLO、TOSA 与后端可优化表示
updated: 2026-10-04
---

# StableHLO、TOSA 与后端可优化表示

一个框架已经确定要进行张量加法、卷积或归约，但还没有决定怎样分块、如何存储以及由哪些线程执行。此时需要一种能在工具之间传递计算含义的表示；后端则还需要适合循环与数据复用分析的结构。

StableHLO和TOSA都服务于张量计算表达，但不是必须串联的两个编译阶段。本章用一项2×3浮点加法说明：同一个计算在交换表示中保留什么，进入Linalg后又增加了哪些可供优化的结构。

## 1. 高层张量加法

输入A、B均为`tensor<2x3xf32>`，结果C的每个位置为A[i,j]+B[i,j]。TOSA形式为：

```text
%r = tosa.add %a, %b
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
```

这里已约定逐元素加法，却没有明确循环顺序、目标内存或线程块。TOSA提供张量级操作及相关数值要求，使前端与后端围绕明确的操作契约衔接。它并不是某一种NPU指令集的另一种拼写。[固定TOSA设计](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/TOSA.md)解释其表达范围和数值目标。

## 2. 转换为访问关系与标量区域

固定工程对函数运行tosa-to-linalg，实际得到tensor.empty和linalg.generic：

```text
indexing_maps = [(i,j)->(i,j), (i,j)->(i,j), (i,j)->(i,j)]
iterator_types = ["parallel", "parallel"]
```

区域读取两个输入元素，生成arith.addf，并yield结果。这里的empty只提供目的张量，输出初值没有被读取。

计算仍然是同一项加法，但映射和迭代类型现在成为显式结构，后续可以应用已经学习的tiling、fusion、bufferization与vectorization。转换没有自动选择最终设备调度，也没有让输出从tensor必然变成memref。

当输入涉及广播、量化或更复杂操作时，转换必须额外保持相应约束；不能用这个同形状浮点例子证明所有TOSA模型都能走同一短pipeline。

## 3. StableHLO 的项目边界

StableHLO是OpenXLA维护的机器学习操作集合及兼容性相关基础设施，用于框架与编译器之间传递计算。它有自己的语义规范、发布与序列化兼容策略；不应把“Stable”理解为所有优化策略或任意工具组合都稳定。[项目说明](https://openxla.org/stablehlo)

对应加法可在StableHLO中表达为张量级add，但本工作区的固定LLVM工具没有因此自动具备StableHLO注册和转换。该项目位于独立仓库，需要使用匹配的项目工具链；本篇没有把generic unknown-op解析当成真实StableHLO验证。

具体项目可能从框架进入StableHLO，再通过项目提供的变换走向后端；也可能直接进入Linalg或其他IR。[Torch-MLIR源码篇](../../tutorials/pipelines/frontend)展示的是固定版本的一条实际选择，不是一条要求所有前端照搬的标准路线。

## 4. 选择共同边界时要核对什么

接入一个新项目，先明确双方支持的操作、类型、动态形状和数值契约，再确认版本及序列化范围。然后用一个有限模型观察它经过哪些实际阶段，哪些操作仍未合法化。

交换表示负责保留计算含义，Linalg等结构化表示提供优化入口，GPU/HIVM等目标相关表示继续表达执行和资源约束。它们可以组合，但组合依据是消费者需要的信息与实际转换支持，而不是方言名字的固定高低排序。

本篇TOSA加法已完成解析与真实Linalg转换；StableHLO部分是项目定位与官方规范入口，没有构建或执行该项目。若后续研究任务需要它，应固定该项目与LLVM依赖，再选一项操作完成定义、转换、边界及执行验证。

## 依据与实践

TOSA转换入口：[TosaToLinalg](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/TosaToLinalg/TosaToLinalg.cpp)。StableHLO的操作行为以[官方规范](https://openxla.org/stablehlo/spec)及对应版本为准。

[25 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/25-domain-extensions)保留本篇完整TOSA输入和Linalg输出。有限练习：解释输出generic的三个identity map分别属于哪个operand，再连接已学的CPU贯通链。
