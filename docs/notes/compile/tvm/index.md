---
order: 40
title: TVM 全链路编译
updated: 2026-11-23
---

# TVM 全链路编译

TVM 将张量模型编译成可以反复调用的程序。这一组教程以 `relu(x @ weight + bias)` 为主要模型，解释编译器怎样表示计算、选择实现、优化循环，最后执行模型并返回结果。

## 编译与执行的完整流程

执行一个模型，需要完成两层工作。模型层决定先算什么、后算什么，以及怎样传递中间结果；计算层负责矩阵乘法、加法等运算内部的循环和数据访问。TVM 用 **Relax** 表达模型层的计算关系，用 **TensorIR** 描述具体计算。一个 IRModule 可以同时包含这两类函数。

沿着编译过程，Relax 中的运算逐渐变成对计算函数的调用，TensorIR 中的循环则逐渐变成目标机器执行的指令。下面展示本系列采用的 TVM v0.20.0 CPU 推理流程。

```text
模型的计算图
    本系列从 PyTorch FX 图导入
    ↓
TVM 前端导入
    将源框架操作转换为 Relax 操作
    记录模型参数、张量形状、类型和计算依赖
    ↓
Relax 模型图
    main 描述模型中的运算及其先后依赖
    例如：matmul → add → relu
    ↓
生成计算函数并进行图优化
    将张量运算展开为 TensorIR 函数，用 call_tir 调用（LegalizeOps）
    标记逐元素、广播、归约等计算模式（AnnotateTIROpPattern）
    提前算出只依赖常量的结果（FoldConstant）
    根据依赖和计算模式选择融合组（FuseOps）
    将组内的 TensorIR 函数组合为一个函数（FuseTIR）
    ↓
包含 Relax 和 TensorIR 函数的 IRModule
    Relax main 调用各个计算函数，并传递中间结果
    TensorIR 函数通过循环、归约和 buffer 读写完成计算
    ↓
优化 TensorIR 中的循环与存储访问
    调整循环顺序、分块大小、向量化方式和缓存位置
    手工调度：开发者通过 Schedule 指定变换
    自动调优：MetaSchedule 生成候选程序，测量并选择方案
    ↓
生成可执行代码
    ├─ Relax 部分
    │    将调用展开为准备输出、传递参数和调用计算函数等步骤
    │    加入形状检查和存储管理代码，生成 VM 字节码
    │
    └─ TensorIR 部分
         将计算块和 buffer 访问逐步展开为循环、地址计算和读写
         生成 LLVM IR，再由 LLVM 生成 CPU 代码
    ↓
将 VM 程序与编译好的计算函数打包为可执行产物
    ↓
运行时传入真实数据
    VM 执行模型入口 main
      → 检查输入并准备输出张量
      → 调用编译好的计算函数
      → 返回结果
```

这两层程序在一次函数调用中配合起来。例如，Relax 中的 `result = call_tir(matmul, ...)` 表示调用矩阵乘法并取得结果。编译器将它展开为“准备输出张量，再把输入和输出传给计算函数”。运行时，VM 执行这些步骤，计算函数向输出张量写入数据，`main` 最后返回结果。第七章详细解释这次调用。

`FuseOps` 决定哪些计算放在同一组，`FuseTIR` 将它们组合到一个函数中；调度继续调整函数内部的循环和存储访问。中间结果保存在哪里、被读取几次，可以从变换后的 TensorIR 中看出来。

计算实现还可以选用已有库或外部后端。这时，模型中的相应运算会改成对这些实现的调用，第六章介绍这种选择。图中的 Pass 组合可在 [Relax pipeline](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/pipeline.py) 中查看，字节码与计算函数的构建过程见 [VM 构建代码](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/vm_build.py)。

## 阅读顺序

| 章节 | 核心变化 |
|---|---|
| [1. 从张量模型到 TVM 执行程序](/notes/compile/tvm/01_execution) | 数学模型→Relax→TensorIR→CPU结果 |
| [2. 模型导入、Relax 数据流与形状信息](/notes/compile/tvm/02_relax) | 源图→绑定、StructInfo、参数与符号形状 |
| [3. 操作实现转换与图融合](/notes/compile/tvm/03_graph_optimization) | 高层操作→实现调用→分组→融合函数 |
| [4. TensorIR 的循环、计算块与存储契约](/notes/compile/tvm/04_tensor_ir) | 张量公式→空间/归约轴、初始化和buffer |
| [5. TensorIR 调度：程序变换、计算位置与存储](/notes/compile/tvm/05_schedule) | 把连续访存与缓存策略落实为IR变换，检查计算范围和写回位置 |
| [6. 自动调度、测量数据库与外部实现](/notes/compile/tvm/06_implementation_selection) | 候选方案→测量/选择/复用，以及库调用边界 |
| [7. TIR Lowering、目标代码与 Relax VM](/notes/compile/tvm/07_runtime) | main准备输出、调用compute并返回结果；分别追踪后端与VM的实现 |
| [8. Relax 改写、调度修改与问题定位](/notes/compile/tvm/08_investigation) | DPL函数Pass→编译验证；调度修改与最早偏差定位 |

第1章用一个具体模型展示编译和调用；第2—3章解释模型图怎样变化，第4—6章解释计算函数怎样优化，第7章把编译产物与运行过程连起来，第8章练习修改编译器并定位问题。

## 阅读目标与共同基础

学完本系列，应能沿模型图、TensorIR 和生成代码追踪一次优化，解释变换的依据，并完成一项图改写或调度修改。

已学过 IR、SSA 和 Pass 的读者可以从第1章开始，通用概念可结合 `ai-basics` 查阅。正文以矩阵乘法加偏置和 ReLU 为主要模型，也用逐元素运算解释融合，用 Softmax 解释归约。

正文包含理解过程所需的代码、IR 和结果；labs 提供可运行的参考工程。实践时可以先观察变换前后的程序，再修改其中一项规则，检查生成代码和计算结果怎样变化。

## 版本与实验范围

采用 TVM **v0.20.0**，commit `1c60502cfba5ca4ed948e07bf6c064426418187d`，LLVM 14 CPU 工具链。代码示例使用该版本的 `tvm.tir` API。

CPU 实验已覆盖静态 FX 导入、手写符号 Relax 动态 batch、逐阶段融合、手工调度、cache 与 compute_at、DPL 改写、实际 TIR lowering 及 LLVM 输出、8 次 MetaSchedule 测量与结果复用、外部 PackedFunc 调用、共享库保存和加载。外部调用使用 Python/NumPy 演示参数传递与调用过程；生产 BYOC 后端属于后续扩展，GPU/NPU 设备运行尚待验证。

| 范围 | 本轮达到的解释与观察 | 深入时继续的问题 |
|---|---|---|
| 图层 | 导入、shape/参数、legalization与融合 | 复杂控制流、数据相关shape、自定义优化 |
| TensorIR | 块/轴/存储、合法调度、缓存位置、尾部与向量化 | 多级分块与tensorize有设计说明；设备性能和异步流水另学 |
| 实现选择 | 规则/搜索/数据库、外部接口与BYOC职责 | 大模型调优、真实外部后端、布局成本 |
| 运行时 | 输出分配、生命周期、VM、产物加载 | C++服务、异构设备、内存池与复杂动态行为 |

对应 [TVM labs](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e) 保存观察、练习、环境和生成物定位。接着可以阅读 [PyTorch Compiler](/notes/compile/pytorch-compiler/)，使用同一个模型比较不同系统的设计。
