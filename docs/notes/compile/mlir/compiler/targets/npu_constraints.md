---
order: 50
title: NPU 表示与执行约束
updated: 2026-10-04
---

# NPU 表示与执行约束

GPU 章节把工作分给 block/thread，再让线程通过存储和同步协作。学习 NPU 时可以沿用“计算、搬运、依赖、资源”这些问题，但具体答案必须来自所选硬件和编译器。

这里选 AscendNPU-IR 上游的 VecAdd 集成例子：从片外输入读两个 16 元素数组，在片上缓冲区相加，再写回。先沿这个完整过程理解 HIVM 的作用，再看内存规划与自动同步怎样消费它保留的信息。

## 1. 一个完整的块级计算

下面根据上游样例统一变量名，将三项独立分配集中展示，保留计算、搬运、类型和 entry 属性：

```text
func.func @add(%a: memref<16xi16, #hivm.address_space<gm>>,
               %b: memref<16xi16, #hivm.address_space<gm>>,
               %c: memref<16xi16, #hivm.address_space<gm>>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %ua = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
  %ub = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
  %uc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
  hivm.hir.load ins(%a : memref<16xi16, #hivm.address_space<gm>>)
    outs(%ua : memref<16xi16, #hivm.address_space<ub>>)
  hivm.hir.load ins(%b : memref<16xi16, #hivm.address_space<gm>>)
    outs(%ub : memref<16xi16, #hivm.address_space<ub>>)
  hivm.hir.vadd ins(%ua, %ub : memref<16xi16, #hivm.address_space<ub>>,
                              memref<16xi16, #hivm.address_space<ub>>)
    outs(%uc : memref<16xi16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%uc : memref<16xi16, #hivm.address_space<ub>>)
    outs(%c : memref<16xi16, #hivm.address_space<gm>>)
  return
}
```

先只追踪内容：`a→ua`、`b→ub`，然后逐位置形成 `uc=ua+ub`，最后 `uc→c`。`gm` 表示这条路径的片外存储，`ub` 表示用于向量计算的片上 Unified Buffer。

这里的一条 `vadd` 表示一块数据的加法，不是“启动 16 个 GPU 线程”。操作数仍是 MemRef；参与计算的范围和存储空间显式保留，后续编译才继续处理布局、尺寸与指令限制。

## 2. 搬运、计算与输出的执行管线

在本章采用的 memory-based 向量路径中，可以把上例对应为：

```text
GM → UB 的 load       搬入管线 MTE2
UB 中的 vadd          向量管线 V
UB → GM 的 store      搬出管线 MTE3
```

这不是从名字猜出的关系。固定源码中 `LoadOp` 带 `OpPipeTrait<PIPE_MTE2>`，向量操作基类带 `OpPipeTrait<PIPE_V>`，store 对应搬出管线。这些操作属性和接口给后续分析提供目标执行信息。

编译器需要知道“两条操作访问同一缓冲区”，也需要知道“它们在哪条 pipe 上执行”。源文件中 load 写在 vadd 前面，不等于跨管线的数据依赖已经由硬件自动建立。把执行关系降到目标时还要插入恰当同步。

## 3. 数据依赖怎样成为同步

上例中 vadd 读 ua/ub，必须等两次 load 完成；store 读 uc，必须等 vadd 写完。沿这两类边，可以推导需要从生产管线通知消费管线：

```text
MTE2 搬入完成 → 发出事件 → V 等待该事件 → 执行 vadd
V 计算完成   → 发出事件 → MTE3 等待该事件 → 搬出结果
```

HIVM 用 `hivm.hir.set_flag`、`hivm.hir.wait_flag` 表达相应配对。以下是 MTE2→V 的形式示意，事件 ID 必须由完整调度统一安排：

```text
hivm.hir.set_flag  [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, #hivm.event<EVENT_ID0>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, #hivm.event<EVENT_ID0>]
```

两条操作在相应管线上承担不同角色。不能把它们替换成 GPU workgroup barrier：同步参与者、事件资源和执行协议不同。同一 pipe 的顺序约束、跨 pipe 依赖、Cube/Vector 跨核同步也需要分别处理。

固定提交的 pipeline 可以根据选项选择 GraphSyncSolver，或回退到 InjectSync。前者会把读写、控制流和 pipe 关系转成分析使用的图，选择同步关系并分配事件，再生成 HIVM 同步操作。首次阅读只需沿“哪个写→哪个读→哪些 pipe→生成什么依赖”追一条边，不必先通读整个求解器。

## 4. 片上分配与 Bufferization 的区别

上例已经用 MemRef 表达可变存储，因此不是还在做 Tensor 值到 Buffer 的转换。但 `%ua/%ub/%uc` 仍需要得到符合目标约束的片上地址。

PlanMemory 处理容量、对齐、生命周期与地址复用等问题。不能把这里的 `memref.alloc` 简单想成目标程序运行时调用通用 malloc；高层分配点可以在后续变成已规划的片上地址表达。

假如 vadd 允许某种原地计算，编译器也要确认旧内容后续是否仍被读取，以及相应指令是否允许重叠。只因 ua 以后“不在源代码中再出现”，还不足以省略已经提交但尚未完成的消费者。这就是存储生命周期与异步依赖必须共同分析的原因。

## 5. 多缓冲如何进入真实 Pipeline

扩大到循环处理多个块后，可以用两份 UB slot 交替承载输入。前一篇推导的 `copy(k)→compute(k)` 和 `compute(k)→copy(k+2)` 仍成立，只是现在必须落实成片上地址和 pipe 事件。

固定源码的 post-bufferization 路径中，可以找到以下阶段，期间还穿插其他合法化与清理：

```text
识别/标记 multi-buffer 候选
    ↓
PlanMemory 规划多份地址
    ↓
展开所需计算与访问，插入目标同步
    ↓
EnableMultiBuffer 按迭代选择当前 slot
```

这条链解释了为什么多缓冲不只是修改循环下标：地址分配要留足空间，同步要区分轮转的 slot，最终访问才能选对地址。更多缓冲也可能带来容量溢出、迫使 tile 变小或触发回退，收益不能只由“管线可并行”推出。

具体 Pass 顺序属于版本与目标配置。本章没有把上述摘录写成所有 Ascend 芯片统一、固定不变的流水线。

## 6. Cube 与 Vector 的不同消费需求

VecAdd 主要涉及向量计算；Matmul 还会进入矩阵计算单元及相应局部存储和格式。某一块数学上是 M×K，不意味着其普通行优先布局就已经适合目标矩阵指令。

在含矩阵和向量阶段的融合算子里，编译器还要安排计算核之间的数据交接、workspace 与同步。此前的分块、访问映射、布局、别名、生命周期和 Pass 顺序都成为具体约束，而不只是框架 API。

AscendNPU-IR 的 HFusion 保留较高层算子语义，HIVM 更直接表达计算、搬运与同步，HACC 提供异构函数/入口等信息。可以在不同层接入：已经手写 HIVM 的 VecAdd 不必再先转回 HFusion；来自其他前端的高层表示则可以选择适当转换入口。

## 7. 硬件代际与版本边界

本章围绕 GM/UB/pipe 的 memory-based 路径建立模型，与当前学习计划中的 A2 方向相接。所读仓库也包含更新的 RegBase/SIMT 路径；不能因此声称所有 NPU 都没有线程/warp 概念，或把一代硬件的存储结构推广到全部产品。

这里固定 AscendNPU-IR 提交 `37c6ebc33d1789500014e6a7b2b62164b3cb566b`。该提交记录的 LLVM 子模块是 Ascend 维护的版本，gitlink 为 `06913395ef8aca66a9814723f4ccc616570b0fed`，不是本教程标准 MLIR 实验所用的 20.1.8。因此标准 `mlir-opt` 不能直接替代 `bishengir-opt/compile` 来验证上述自定义语法和完整 pipeline。

当前完成的是源码与定义/消费路径核验，未构建该项目的工具链、未运行 CANN/NPU。上游 VecAdd 集成测试期待输出 1—16，那是上游测试的期望值，不作为本机实测结果。

## 8. 阅读检查与后续

1. `hivm.hir.vadd` 的操作数是 MemRef，为什么仍然可以表达块级向量计算？
2. 同一个缓冲区上的写和读分别在 MTE2/V，编译器还缺什么信息才能正确安排执行？
3. Tensor 已经 Bufferize，为什么还需要 PlanMemory？
4. 两份 slot 的地址已经分配好，为什么仍不能省掉复用前的等待？

后续项目贯通会选一条有限路径追到实际定义、改写、测试和下游消费。当前先把已学 MLIR 机制放回这条编译链，避免将研究目标变成通读整个 NPU 框架。

固定源码入口：[VecAdd 输入](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/test/Integration/HIVM/VecAdd/add.mlir)、[HIVM Pipeline](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp)、[同步定义](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)、[架构与代际说明](https://github.com/Ascend/AscendNPU-IR/blob/37c6ebc33d1789500014e6a7b2b62164b3cb566b/docs/source/zh_cn/introduction/architecture.md)。
