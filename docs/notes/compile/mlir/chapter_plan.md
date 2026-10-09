---
order: 3
title: IR 定义之后的阅读顺序
updated: 2026-10-04
---

# IR 定义之后的阅读顺序

这份路线从已理解基础 IR、Pass / Pattern 和 IR 定义开始，依次连接表示转换、张量计算、存储、执行、优化与目标项目。**可以按本页从上往下读：每一站按表内顺序推进，完成本站目标后进入下一站。** 不需要先读完所属目录中的所有扩展文章。

本页统一维护后续文章的阅读次序；[学习路径](./learning_path)记录阶段标准与已反馈进度，[覆盖表](./coverage)保留长期知识范围与深度。侧栏按知识分类排列，文件中的 `order` 只决定同级菜单位置，不能跨目录当作学习顺序。

## 路线总览

| 顺序 | 阅读阶段 | 本阶段形成的认识 |
|---|---|---|
| 1 | Conversion 基础 | 已定义的操作和类型怎样转换成下一种表示 |
| 2 | 编译流程导读 | 同一个计算怎样经过多层 IR 走向执行 |
| 3 | Tensor 与 Linalg | 公式、张量值、迭代空间和访问关系怎样对应 |
| 4 | MemRef 与 Bufferization | 张量更新怎样落实为存储复用、复制与释放 |
| 5 | 转换边界、LLVM 与 CPU 执行 | 类型与调用边界怎样保持一致，怎样得到可调用程序 |
| 6 | Math / Index 与 Softmax | 用完整算法连接计算、存储和数值结果 |
| 7 | 分析与优化合法性 | 编译器凭什么判断一次移动、合并或删除成立 |
| 8 | Affine、结构化优化、Vector 与 Transform | 怎样改变迭代、数据复用和执行粒度 |
| 9 | Matmul 与 Attention | 把调度与存储选择放回完整算法 |
| 10 | GPU / NPU 映射 | 并行工作、存储和依赖怎样落实到目标设备 |
| 11 | 真实项目源码链 | 把共同机制映射到选定编译器的实际实现 |

第 1—5 站形成第一条 CPU 编译链，第 6 站用 Softmax 巩固。第 7—9 站进入优化，第 10—11 站进入目标项目。每完成一段可以停下来做有限修改，不必一次读完全部章节。

## 1. Conversion 基础

延续熟悉的 clamp：此前用普通 Pattern 描述展开，现在进一步规定哪些结果被目标接受，以及类型改变后怎样连接使用者。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 1.1 | [操作转换与目标合法性](./compiler/conversion/operation_conversion) | 规则、目标、driver；partial/full 的完成条件 |
| 1.2 | [类型转换与函数签名](./compiler/conversion/type_conversion) | range 到 i32；TypeConverter、adaptor 和最小返回边界 |

读完应能解释一次转换为什么成功或失败，以及类型变化为何必须照顾使用者。混合表示和复杂边界安排在第 5 站；1:N 与框架进阶参考在文末按需回访。

可选实践：07 操作转换与 08 类型转换实验中，先选择一个目标声明或输入类型修改，预测结果和失败原因。各篇文末提供工程链接。

## 2. 编译流程导读

阅读[从张量计算到执行结果](./tutorials/pipelines/overview)。先跟随一个 2×3 矩阵加法，辨认高层计算、结构化计算、存储访问、循环和低层表示各自保留的信息。

这一站只建立整条路径的位置感。遇到尚不熟悉的 Tensor、Linalg 或 descriptor，先辨认它们承担的工作，后面逐站展开。本文使用的 CPU 路径是一个具体选择，不要求所有编译器采用相同的方言顺序。

## 3. Tensor 与 Linalg

先解释张量的值、形状和索引，再把熟悉的数组计算写成迭代与访问关系。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 3.1 | [Tensor 值语义与形状](./dialects/tensor/values_shapes) | 旧值和新值、动态尺寸、dim 与 empty |
| 3.2 | [切片与张量更新](./dialects/tensor/slices) | offset/size/stride、取出与插回 |
| 3.3 | [形状变换与元素对应](./dialects/tensor/reshape) | 展平与拆维；区分 reshape、转置与 cast |
| 3.4 | [迭代空间与访问映射](./dialects/linalg/iteration_maps) | indexing_maps、iterator_types 与区域标量计算 |
| 3.5 | [广播、归约与矩阵乘法](./dialects/linalg/structured_computations) | 从公式推导访问投影、归约维和初值 |
| 3.6 | [Destination-Passing Style 与结果构造](./dialects/linalg/destination_style) | outs、初始内容、结果 Value 与存储复用的区别 |

读完应能给定一个输出坐标，说明读取哪些输入、进行什么计算，以及结果为何仍是一个新的 SSA Value。09 实验用于观察和修改小矩阵、窗口或归约；此时不要求实现 tiling。

想提前接触真实项目，可以穿插[PyTorch 矩阵乘法到 Linalg](./tutorials/pipelines/frontend)，先读导入和 `mm` 转换这一段；其完整边界回到第 11 站再串联。

## 4. MemRef 与 Bufferization

从二维窗口的地址计算进入别名，再解释张量值怎样由可变内存实现。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 4.1 | [存储对象与索引访问](./dialects/memref/storage) | 分配、初始化、访问与生命周期 |
| 4.2 | [布局、偏移与步长](./dialects/memref/layout) | 根据坐标计算元素偏移 |
| 4.3 | [Subview 与别名关系](./dialects/memref/views) | 嵌套窗口、共享分配、复制与释放责任 |
| 4.4 | [Tensor 到 Buffer 的表示变化](./compiler/memory/bufferization) | 新旧张量值在可变存储中的实现 |
| 4.5 | [One-Shot Bufferization 的决策](./compiler/memory/one_shot) | 读写冲突、可写性、alias/equivalence 与复用选择 |
| 4.6 | [函数边界、所有权与释放](./compiler/memory/ownership) | 参数借用、返回存储、逃逸与条件释放 |

读完应能解释一次更新为什么需要或不需要复制，以及哪个函数负责释放返回的存储。10 实验提供窗口与复用对照。自定义 Bufferizable 接口留到已经理解标准操作的决策之后回访。

## 5. 转换边界、LLVM 与 CPU 执行

现在已经见过函数、张量和存储之间的不同约定，再回到 Conversion 的边界问题，接着落实低层表示与实际调用。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 5.1 | [混合表示与边界衔接](./compiler/conversion/materialization) | 新旧类型共存、materialization、占位 cast 与消解 |
| 5.2 | [调用与区域边界转换](./compiler/conversion/boundaries) | caller/callee、Block 参数与 Region 入口/出口 |
| 5.3 | [结构化计算到显式循环](./compiler/lowering/loops) | indexing maps 到访存，循环状态到 CFG 传值 |
| 5.4 | [LLVM 类型、指针与低层操作](./dialects/llvm/operations) | GEP、load 和低层函数 |
| 5.5 | [DataLayout 与 MemRef 描述符](./dialects/llvm/data_layout) | 基址、offset、sizes/strides 与 index 位宽 |
| 5.6 | [类型降低、ABI 与函数边界](./compiler/lowering/abi) | descriptor 展开、C wrapper 与返回约定 |
| 5.7 | [LLVM Dialect 与 LLVM IR](./dialects/llvm/translation) | 两种对象模型、Block 参数与 PHI 的对应 |
| 5.8 | [Translation、JIT 与 AOT](./compiler/lowering/execution) | 代码生成、链接、加载和调用的分工 |
| 5.9 | [Linalg 到 CPU 的最小执行链](./tutorials/pipelines/cpu) | 将上述阶段接起来，核对输入、输出与生命周期 |

第一轮到这里应能说明一个小计算如何变成可调用程序，并区分 verifier、转换、链接和数值检查。实践选择 11 实验中的一个小输入或布局变化，不要求实现完整 LLVM 后端。

## 6. Math / Index 与 Softmax

先补算法真正需要的数学和尺寸计算，再把前面的机制串进稳定 Softmax。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 6.1 | [标量数学、指数与数值实现](./dialects/math/scalar_math) | exp 的目标实现、误差和数值条件 |
| 6.2 | [Index 运算与尺寸计算](./dialects/math/index_arithmetic) | 分块尺寸、余块、位宽与访问边界 |
| 6.3 | [归约与 Softmax 的结构化表示](./tutorials/kernels/reduction_softmax) | max/exp/sum/normalize、归约初值与广播 |
| 6.4 | [Softmax 的存储与融合](./tutorials/kernels/softmax_storage) | 中间分配、按行处理和输出暂存 |

这两篇 Softmax 先解释具体计算重组，通用 fusion 机制在第 8 站展开。13 实验可以修改行列数或数据分布，对照两种实现；数值验证方法遇到问题时查[数值、边界与性能证据](./guides/numerics_performance)。

## 7. 分析与优化合法性

已有算法和存储直觉后，再问编译器需要哪些事实，才能自动执行相似变换。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 7.1 | [支配、活跃性与操作移动](./compiler/analysis/dominance_liveness) | 值可用性、控制关系与移动范围 |
| 7.2 | [别名、内存效果与依赖](./compiler/analysis/alias_effects) | 共享存储、访问影响与重排限制 |
| 7.3 | [分析结果的缓存与失效](./compiler/analysis/caching) | IR 修改后哪些事实仍然有效 |
| 7.4 | [MLIR 数据流分析](./compiler/analysis/dataflow) | 常量、可达性、格、solver 与区域传播 |
| 7.5 | [CSE、DCE 与 LICM 的合法性](./compiler/analysis/legality) | 将效果、支配和推测执行用于具体判断 |

已有传统编译器背景时，可以快速回顾算法名称，把重点放在 MLIR 的 Region、Interface 和失效约定。12 实验用于预测真实分析查询和优化前后变化。

## 8. 结构化优化、Vector 与 Transform

本阶段先描述域与依赖，再改变循环和数据组织，最后用显式调度串联变换。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 8.1 | [Affine 映射](./dialects/affine/maps) → [整数域与边界](./dialects/affine/domains) → [循环与依赖分析](./dialects/affine/loops_analysis) | 依次阅读三篇，将访问关系连接到迭代域和合法性 |
| 8.2 | [Tiling 与边界处理](./compiler/optimization/tiling) | tile、局部访问和余块 |
| 8.3 | [Fusion 与计算重排](./compiler/optimization/fusion) | producer/consumer、减少流量与重复计算 |
| 8.4 | [循环交换、并行化与归约次序](./compiler/optimization/interchange) | 访问顺序、依赖与浮点约束 |
| 8.5 | [向量值](./dialects/vector/values) → [Transfer 与 Mask](./dialects/vector/transfer_mask) → [归约与收缩](./dialects/vector/reduction_contract) | 依次阅读三篇，理解向量计算和有效元素 |
| 8.6 | [向量化与目标降低](./compiler/optimization/vectorization) | 编译器怎样产生向量表示并降低 |
| 8.7 | [Transform Dialect 与显式调度](./compiler/optimization/transform) | handle/payload、匹配、消费与失败 |
| 8.8 | [调度选择与性能解释](./compiler/optimization/performance) | 连接访问量、汇编、计时和适用范围 |

14/15 实验对应这段路线。一次只修改一个调度选择，先检查数值，再解释性能；没有测量时保留为访问或计算量推演。

## 9. Matmul 与 Attention

依次阅读：

1. [Matmul 的分块、累加与复用](./tutorials/kernels/matmul)：把 tiling 放进 M/N/K 三维计算，追踪跨 K 累加与尾块。
2. [Attention 的计算与中间存储](./tutorials/kernels/attention)：连接 QK、归一化和 WV，明确中间矩阵。
3. [在线 Softmax 与分块 Attention](./tutorials/kernels/online_attention)：推导运行中的最大值、归一化量和输出状态，以及重新缩放。

17 实验提供 CPU 数值和错误递推反例。当前先理解算法状态与存储变化；高性能设备实现还需要第 10 站的映射与同步条件。

## 10. GPU / NPU 映射

先认识执行层次，再安排数据和依赖。SCF 的并行结果协议在映射之前补上。

| 顺序 | 文章 | 阅读重点 |
|---|---|---|
| 10.1 | [并行归约与张量结果](./dialects/scf/parallel_results) | parallel 的贡献/合并与 forall 的切片组装 |
| 10.2 | [GPU 执行层次](./dialects/gpu/execution) | launch、block、thread 与工作分配 |
| 10.3 | [并行工作到执行层次](./compiler/targets/parallel_mapping) | SCF 工作到设备映射 |
| 10.4 | [GPU 地址空间](./dialects/gpu/memory_spaces) → [设备布局与搬运](./compiler/targets/layout_transfer) | 依次连接存储归属、线程访问和数据组织 |
| 10.5 | [GPU 同步语义](./dialects/gpu/synchronization) → [流水依赖](./compiler/targets/pipelining) | 依次连接可见性、异步完成与缓冲区复用 |
| 10.6 | [Host / Device 编译与启动](./compiler/targets/host_device) | kernel、目标产物、参数和运行时 |
| 10.7 | [NPU 表示与执行约束](./compiler/targets/npu_constraints) | 计算单元、存储、搬运和 pipe 约定 |

16 实验已经有编译与逻辑模型证据，当前没有 GPU/NPU 设备执行结果。GPU/NPU 不强制采用同一编译路径；学习共性后，下一站回到项目自己的表示和版本。

## 11. 真实项目源码链

先读[有限源码阅读](./guides/source_reading)，用一条 `x+0` 的变化练习“事实由谁产生、由谁消费、如何改变结果”的读法，再选择目标项目。

| 路径 | 文章 | 建议进入时机 |
|---|---|---|
| 前端路径 | [PyTorch 矩阵乘法到 Linalg](./tutorials/pipelines/frontend) | 第 3 站后可提前读转换片段；第 5 站后更容易串起边界 |
| GPU 路径 | [Triton 加法编译路径](/notes/triton/compiler_path) | Vector、布局与 GPU 映射之后 |
| NPU 桥接路径 | [Triton-Ascend 编译路径](/notes/triton/ascend_compiler_path) | Triton 表示与 NPU 约束之后 |
| NPU 操作与消费 | [HIVM 向量加法源码路径](/notes/compile/ai-compiler/npu-ir/source_path) | IR 定义后可提前看 ODS；理解 NPU 约束后继续追变换和消费 |

表格给出默认展开顺序；实践时先选一个项目完成一次有限修改，不要求四个项目全部通读。18 实验保存固定源码入口和阶段快照，源码检查与项目构建、运行分开记录。TVM 仍是 AI Compiler 总路线中的并行项目，不并入 MLIR 方言必读清单。

## 主线之外的回访入口

下面的文章已经有正文，按当前任务选读。它们的目录位置表示知识归属，不意味着必须在 Conversion 之前读完。

| 需要解决的任务 | 阅读顺序或入口 | 适合回访的时机 |
|---|---|---|
| 自定义操作接入分析与内存机制 | [类型/形状推导](./compiler/ir_definition/type_shape_inference) → [自定义 Bufferization](./compiler/memory/custom_operations)；[区域/循环/调用接口](./compiler/ir_definition/region_interfaces) | 第 4 / 7 站之后，能说明消费者需要什么事实时 |
| 更复杂的操作与参数存储 | [复杂 ODS](./compiler/ir_definition/complex_ods) → [参数存储与接口](./compiler/ir_definition/storage_interfaces) | 出现多组 operands、Properties 或自定义参数所有权需求时 |
| 一个值拆成多个值 | [1:N 转换](./compiler/conversion/one_to_many)；[复杂转换参考](./compiler/conversion/advanced) | 第 5 站之后，已经能维护函数/调用/Block 边界时 |
| 声明式规则 | [DRR / PDL / PDLL](./compiler/transforms/declarative_patterns) | 已有可解释的 C++ Pattern，希望比较规则表达方式时 |
| 复杂区域和结构维护 | [多 Block 与多路选择](./dialects/scf/regions_switch)、[CFG 与符号维护](./compiler/transforms/ir_maintenance)、[资源与阶段化效果](./compiler/analysis/resources_effects) | 编辑 CFG、符号或实现依赖效果的变换时 |
| 观察、调试和回归 | [阅读与验证 IR](./guides/inspecting_ir) → [失败与最小复现](./guides/debugging) → [回归测试](./guides/regression_tests) | 从第 1 站起随实验使用，不必等学完整条主线 |
| 数值与性能判据 | [数值、边界与性能证据](./guides/numerics_performance) | 第 5—6 站开始使用，计时前再次回访 |
| 应用接入 | [C API](./guides/c_api)、[Python 绑定](./guides/python_bindings) | 需要用外部程序构造、修改或处理 IR 时 |
| 持久化和动态加载 | [Bytecode](./guides/bytecode) → [插件](./guides/plugins) | 需要方言版本兼容或工具扩展时 |
| 领域表示 | [Quant](./dialects/extensions/quantization)、[Sparse](./dialects/extensions/sparse)、[Shape](./dialects/extensions/shape)、[IRDL](./dialects/extensions/irdl)、[StableHLO/TOSA](./dialects/extensions/graph_ir)、[Complex](./dialects/math/complex) | 遇到量化、稀疏、动态形状或交换表示任务时，选择相应分支 |

`ir_definition` 中 `order: 60/70/80/90` 分别对应形状推导、区域接口、复杂 ODS 和参数存储，用较大的排序值把扩展放在基础五篇之后。数字不是课号、难度等级或前置要求；跳号不意味着漏了几十篇文章。

## 每一站的阅读与实践

先读原理并解释主例，再从对应实验 README 选一个有限修改：写出预测，改变一个条件，观察前后 IR 或数值，最后解释原因。不要把运行验证脚本得到 passed 当成已经掌握。

遇到新 API 时，先确认它在当前过程中改变哪个对象、提供什么事实；具体拼写和实现入口按文末资料查阅。主线能连起来后，再决定是否需要深入框架内部。

目前的明确起点是第 1 站的[操作转换与目标合法性](./compiler/conversion/operation_conversion)。已经完成的 IR 定义基础按需回查，阅读这份新路线不要求从头重学。
