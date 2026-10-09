---
order: 110
title: 沿 HIVM 向量加法追踪定义与消费
updated: 2026-10-04
---

# 沿 HIVM 向量加法追踪定义与消费

学习过 MLIR 的定义、Pass 和 Conversion 后，可以用一个真实操作检查这些机制怎样合作。这里选择 AscendNPU-IR 的 `hivm.hir.vadd`，沿声明、接口、改写、下游调用和集成测试追一条有限路径。

[NPU 表示与执行约束](/notes/compile/mlir/compiler/targets/npu_constraints)已经解释 GM→UB→向量加法→GM 的整体计算。本篇接着回答：这项块级语义由什么定义，编译器怎样决定它可以进入哪种实现，而不是只罗列相关文件。

## 1. 块级加法保留的契约

集成例子把两份 16 元素 i16 输入搬入 UB，然后计算：

```text
hivm.hir.vadd ins(%a_ub, %b_ub :
  memref<16xi16, #hivm.address_space<ub>>,
  memref<16xi16, #hivm.address_space<ub>>)
  outs(%c_ub : memref<16xi16, #hivm.address_space<ub>>)
```

这条操作同时留下计算种类、输入输出、元素类型和存储位置。即使最后会变成库调用或低层指令，前面的内存与同步分析仍可利用这些信息。

`HIVMVectorOps.td` 中的 `VAddOp` 继承二元 elementwise 基类，并附加元素类型、向量化、计算单元和接口等约束。读者不必一次展开所有 Trait：当前要检查的是两个源值怎样对应加法输入、dst 怎样表达写入位置、哪些 dtype/形式被该路径支持。

## 2. ODS 生成与手写实现的接合

构建配置把 `HIVMVectorOps.td` 交给 `mlir_tablegen`，分别生成 `HIVMVectorOps.h.inc` 与 `HIVMVectorOps.cpp.inc`。前者被公开头文件接入，后者在操作实现与 Dialect 注册所需位置按生成宏接入。

这与教学 `my.add` 是同一种工程机制，但生成接口只解决结构与入口，不会自动写出向量加法的所有目标实现。后续的向量化、额外 buffer、库函数约定和标量展开仍由对应手写实现或共享模板提供。

实际阅读时可以先把生成步骤理解为“使 VAddOp 的类型、字段和声明可供 C++ 使用”。等到某个访问器或验证行为影响当前结果，再查它对应的生成代码。本次只核对生成配置和 include 接合，未构建该 fork 的 `.inc`，因此不展示伪造的生成输出。

## 3. 接口如何影响实现选择

库调用路径需要回答：当前操作支持的最大 rank 是多少？库函数叫什么？需要哪些额外缓冲区或参数？这些问题由 `OpWithLibraryFunction`、`ExtraBufferOpInterface` 等协议提供信息。

例如，固定版本给 VAddOp 的库路径记录了最大 rank=3。当一个 buffer 操作高于这个 rank，转换不能原封不动把所有维度交给三维库例程；需要把外层维度展开为循环，再将局部子视图交给可接受的实现。

这解释了接口的实际价值。通用消费者不必为每种向量操作复制一份 rank 判断和参数组织，但各操作仍必须提供准确答案。接口名本身不是正确性保证：错误的额外 buffer 大小、rank 或参数约定会在更低层变成错误调用。

## 4. 向量操作到库调用的规则

在 memory-based 的 `HIVMToStandard.cpp` 中，`VectorOpToLibraryCallPattern<VAddOp>` 是 VAdd 的消费者之一。核心处理为：

```text
确认当前是 buffer 语义
    ↓
从操作接口取得库函数名、最大 rank 和调用参数
    ↓
rank 足够低：直接生成调用
rank 过高：构造外层循环与低秩子视图，再生成调用
    ↓
替换/删除原来的 HIVM 操作
```

其前置 `hasPureBufferSemantics()` 说明此路径假定 Tensor 已经完成相应存储转换。它不是碰到 Tensor 就自动替用户做 Bufferization。

Pass 将 VAdd 等操作声明为非法，注册相关模式，再用 Conversion 组织合法化。源码文件名含 Standard，也不表示项目必须存在一个叫 Standard 的最终方言；该路径生成的可以是 func、MemRef、Arith、SCF 等共同组成的表示。

## 5. 用上游测试检查具体边界

`libcall-noinline-membase.mlir` 明确指定 Ascend910B1，输入是一维 16 元素 f16 向量加法。FileCheck 期待产生 `vadd_1d_half` 的私有声明，并分别检查内联策略；结合前面的规则，可追踪调用怎样引用这个声明。

```text
Ascend910B1 一维 f16 的 hivm.hir.vadd
          ↓ 该测试配置的转换
func.call @vadd_1d_half(...)
```

这是固定上游测试的期望形态，不是本机实测输出，也不是说前面的 i16 集成例子调用同一个 half 例程。类型、rank、广播形式与目标配置都可能改变所选名称和参数。

阅读测试时应同时看输入、RUN 配置和 CHECK。只从测试文件搜到一个函数名，不能证明任意 VAdd 都走这条分支。仓库还包含 RegBase/SIMT 相关实现，Pass 入口会按架构选择不同路径。

## 6. 从库调用继续到设备产物

HIVM pipeline 在内存规划、必要展开和同步处理之后接入相应低层转换。产生库调用以后还需解决实现链接、MemRef/指针 ABI、其他剩余操作和目标代码生成；“已经没有 vadd”不等于设备程序已经可以启动。

这里应给阅读链设置一个可解释的边界：本篇追清 VAdd 的结构、实现接口和一种明确消费者，再用集成测试连接运行时。若任务变成优化特定 dtype 的例程或生成新目标指令，再沿所选库函数实现继续深入，不必从一开始遍历全部库和后端。

## 7. VecAdd 集成测试的运行时闭环

上游集成输入直接从 HIVM 开始，经 `bishengir-compile` 生成 kernel.o。C++ 驱动读取产物，注册设备 binary 和名为 add 的函数，分配输入输出并复制数据，最后启动并等待。

关键次序是：

```text
读取 kernel.o → 注册 binary / function
    → 分配设备缓冲区 → 输入复制
    → rtKernelLaunch → stream synchronize
    → 输出复制 → 比较期望值 → 释放资源
```

输入为 0—15 与十六个 1，期望输出 1—16。调用者按此样例的 ABI 准备三个设备指针；不要拿标准 MLIR CPU 实验的完整 MemRef descriptor 直接替换它。类型降低与 Host 参数打包必须属于同一约定。

这条运行链把前面的 ODS 和 Pass 放回了实际用途：定义并保留块级语义，使分析和目标转换能组织正确实现；最终由运行时把输入交给设备。它并不要求把所有框架 API 都记住。

## 8. 固定版本与阅读产物

本篇核对提交 `37c6ebc33d1789500014e6a7b2b62164b3cb566b`，LLVM gitlink 为 `06913395ef8aca66a9814723f4ccc616570b0fed`。当前没有构建该 fork、调用 CANN 或执行 NPU；上述期望来自上游源码与测试，不计入本机设备数值证据。

对应源码位置：

- [操作定义与生成配置](https://github.com/Ascend/AscendNPU-IR/tree/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/include/bishengir/Dialect/HIVM/IR)：跟 VAddOp 和 HIVMVectorOps 的 TableGen 目标。
- [库接口实现](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp)：确认消费者查询的信息。
- [HIVMToStandard](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/lib/Conversion/HIVMToStandard/HIVMToStandard.cpp)：跟模板规则、目标合法性和架构分支。
- [转换测试](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/test/Conversion/HIVMToStandard/HIVMToStandard/libcall-noinline-membase.mlir)与[VecAdd 集成工程](https://github.com/Ascend/AscendNPU-IR/tree/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/test/Integration/HIVM/VecAdd)：分别观察局部输出契约与完整调用链。

可以把阅读结果写成三条可核对的结论：当前输入走哪条 rank/dtype 分支；哪个接口答案决定调用；哪项前置未满足会使这条路径不成立。后续为 NPU 增加操作或改写时，就能据此列出需要同时维护的定义、消费者和测试。
