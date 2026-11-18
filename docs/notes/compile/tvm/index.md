---
order: 40
title: TVM 全链路编译
updated: 2026-10-30
---

# TVM 全链路编译

这一组教程用 `relu(x @ weight + bias)` 走通模型表示、实现转换、图融合、张量调度、代码生成和运行。先建立整条数据与编译过程，再逐章解释其中的决定；以推理为主，不把完整实现一个新硬件后端作为初读前提。

## 阅读顺序

| 章节 | 核心变化 |
|---|---|
| [1. 从张量模型到 TVM 执行程序](/notes/compile/tvm/01_execution) | 数学模型→Relax→TensorIR→CPU结果 |
| [2. 模型导入、Relax 数据流与形状信息](/notes/compile/tvm/02_relax) | 源图→绑定、StructInfo、参数与符号形状 |
| [3. 操作实现转换与图融合](/notes/compile/tvm/03_graph_optimization) | 高层操作→实现调用→分组→融合函数 |
| [4. TensorIR 的循环、计算块与存储契约](/notes/compile/tvm/04_tensor_ir) | 张量公式→空间/归约轴、初始化和buffer |
| [5. TensorIR 调度：程序变换、计算位置与存储](/notes/compile/tvm/05_schedule) | Schedule改写IR；cache、compute_at与写回位置的取舍 |
| [6. 自动调度、测量数据库与外部实现](/notes/compile/tvm/06_implementation_selection) | 候选方案→测量/选择/复用，以及库调用边界 |
| [7. TIR Lowering、目标代码与 Relax VM](/notes/compile/tvm/07_runtime) | 实际TIR passes→LLVM；模型分配→VM调用与加载 |
| [8. Relax 改写、调度修改与问题定位](/notes/compile/tvm/08_investigation) | DPL函数Pass→编译验证；调度修改与最早偏差定位 |

首次先读第1章，知道完整任务；第2—3章展开图层，第4—6章展开张量程序与实现选择，第7章补齐运行，第8章回到整体修改和调查。

## 阅读目标与共同基础

本系列要求能够解释设计选择、沿实际表示追踪一次优化，并在参考工程上修改有限行为。每章先建立具体任务，再解释决定使用哪些事实、改变了什么、怎样验证；源码只选能确认该决定的入口，不以接口数量衡量深度。

`ai-basics` 可按需查阅，不是必须先读完的前置课程。已学过IR、SSA、Pass的读者直接沿本系列前进；影响当前结论的通用原理会在正文中补齐。主要模型仍是矩阵乘法加偏置和ReLU；逐元素链用于隔离融合，Softmax用于观察归约，换例时说明要隔离的问题。

正文独立提供关键代码、IR、状态变化和结果。labs中的参考观察用于复现和查证，练习只选择少量有边界的修改；运行参考脚本不等于独立实现，也不要求每章新增练习。

## 本轮范围与证据

采用TVM **v0.20.0**，commit `1c60502cfba5ca4ed948e07bf6c064426418187d`，LLVM14 CPU工具链。它是固定教学基线，不代表最新版API；本系列使用`tvm.tir`，新版本中的包拆分和接口变化应按对应源码重新确认。

CPU实验已覆盖静态FX导入、手写符号Relax动态batch、逐阶段融合、手工调度、cache与compute_at、DPL改写、实际TIR lowering及LLVM输出、8次MetaSchedule测量与结果复用、外部PackedFunc调用、共享库保存和加载。外部函数使用Python/NumPy演示协议，没有实现生产BYOC后端；GPU/NPU没有设备运行证据。

| 范围 | 本轮达到的解释与观察 | 深入时继续的问题 |
|---|---|---|
| 图层 | 导入、shape/参数、legalization与融合 | 复杂控制流、数据相关shape、自定义优化 |
| TensorIR | 块/轴/存储、合法调度、缓存位置、尾部与向量化 | 多级分块与tensorize有设计说明；设备性能和异步流水另学 |
| 实现选择 | 规则/搜索/数据库、外部接口与BYOC职责 | 大模型调优、真实外部后端、布局成本 |
| 运行时 | 输出分配、生命周期、VM、产物加载 | C++服务、异构设备、内存池与复杂动态行为 |

长期范围保留，当前练习先要求能解释一条完整链并独立修改有限行为。文章已编写、作者已验证与读者已掌握是不同状态。

对应 [TVM labs](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e) 保存观察、练习、环境和生成物定位。接着可以阅读 [PyTorch Compiler](/notes/compile/pytorch-compiler/)，使用同一个模型比较不同系统的设计。
