---
order: 2
title: 计划覆盖范围与缺口
updated: 2026-09-06
---

# 计划覆盖范围与缺口

本表约束后续章节的广度和深度。目标是支撑 AI Compiler 的阅读、开发、转换与优化工作，不以穷举所有上游 Dialect 和 API 为完成条件。新项目发现超出范围的机制时，先增补此表，再编写正文。

## 状态与深度

- **D1 辨识**：说明用途、输入输出、与相邻机制的边界。
- **D2 推演**：独立解释完整实例、约束、边界与反例，并定位规范。
- **D3 实现**：完成最小实现或修改，读到相关 ODS/C++/测试，验证成功和失败路径。
- **D4 分析**：解释正确性条件、分析/优化的取舍，给出复现和性能或工程证据。

表中的“本次正文”只记录写作状态；“后续目标”才是长期深度要求。`已展开 D2` 不代表用户已经掌握，也不等同于相应 C++ API 已实现验证。

示例证据按四类记录：P 解析与 verifier，G 通用打印再解析，L 指定 lowering，N 预期失败诊断。证据清单由[文档验证脚本](./guides/inspecting_ir#文档示例的复现)生成；没有机器码执行的地方不记录数值测试通过。纯概念条目的依据是各章列出的固定版本规范与源码。

## 核心 IR

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 尚未覆盖 |
|---|---|---|---|---|
| C01 | 文本、内存对象、bytecode 的对应；parser/printer；Context | — | [对象模型](./core/ir_model)，D2；P/G | D3：构造、解析 API；bytecode 版本机制见 X03 |
| C02 | Operation 字段、Op 包装类、递归所有权；四种关系 | C01 | [对象模型](./core/ir_model)，D2；P/G | D3：walk、插入/移动、erase 生命周期 |
| C03 | Region/Block/terminator；SSACFG 与 Graph；空 Region | C02 | [对象模型](./core/ir_model)，D2 | D3：RegionKindInterface 与克隆/内联边界 |
| C04 | OpResult/BlockArgument；operand、use、user；多结果 | C02 | [SSA](./core/values_ssa)，D2；P/G | D3：use-list、RAUW、IRMapping |
| C05 | CFG 支配、合流、回边、层次支配与作用域 | C03—04 | [SSA](./core/values_ssa)，D2；P/N | D3：DominanceInfo 与修改 CFG 后的维护 |
| C06 | 标量/容器/函数类型；动态 shape；index；类型相等与 cast | C04 | [类型与静态信息](./core/types_attributes)，D2；P/G | D3：类型构造、推导与数据布局查询 |
| C07 | Attribute、inherent/discardable、Properties、别名、Location | C02、C06 | [类型与静态信息](./core/types_attributes)，D2；P/G | D3：自定义存储、ODS accessor、验证与打印 |
| C08 | SymbolTable、SymbolRef、查找、可见性、IsolatedFromAbove | C03—04 | [符号与作用域](./core/symbols_scopes)，D2；P/N | D3：符号替换、重命名、未知符号使用的保守处理 |
| C09 | Tensor 值与 MemRef 存储、view/alias、生命周期 | C04、C06 | [效果模型](./core/effects)，D2 边界；P/G | D3—4：完整内存模型、所有权与 bufferization |
| C10 | Read/Write/Allocate/Free；未知效果；递归效果 | C09 | [效果模型](./core/effects)，D2 | D3：MemoryEffectsOpInterface、Resource 与阶段化效果 |
| C11 | 无内存效果、可推测执行、UB、不终止与改写条件 | C05、C10 | [效果模型](./core/effects)，D2；P/G | D3—4：speculation 接口、控制依赖、可移动性证明 |

## 方言语义

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 尚未覆盖 |
|---|---|---|---|---|
| D01 | builtin.module、Graph 容器、类型/属性归属、转换占位 cast | C01—03、C06 | [builtin](./dialects/builtin)，D2 | D3：module 构造；cast reconciliation 的实现 |
| D02 | func 定义/声明、call/return、签名、隔离、间接调用 | C08 | [func](./dialects/func)，直接调用 D2；间接调用 D1 | D3：FunctionOpInterface、CallOpInterface、内联与 ABI |
| D03 | arith constant、整数运算/比较、浮点语义、cast/select | C06—07 | [arith](./dialects/arith)，D2 核心；P/G | D3：完整常用操作族、fold；D4：数值语义与优化 |
| D04 | cf.br/cond_br、参数合流、回边、入口约束 | C05、D03 | [cf](./dialects/cf)，D2；P/N | D3：switch/assert、CFG 修改与分析 |
| D05 | SCF 区域协议、控制转移与 SSA 结果的关系 | C03—05、D04 | [SCF](./dialects/scf/)，D2 | D3：RegionBranchOpInterface、LoopLikeOpInterface |
| D06 | scf.if：分支结果、空 else、捕获、嵌套、select 边界 | D05、D03 | [if](./dialects/scf/if)，D2；P/G/L/N | D3：IfOp canonicalization 与 lowering |
| D07 | scf.for：边界、迭代状态、多结果、零次循环、类型与 CFG | D05 | [for](./dialects/scf/for)，D2；P/G/L/N | D3：ForOp 实现；D4：循环变换合法性 |
| D08 | scf.while：before/after、condition/yield、两套类型、do-while | D05—07 | [while](./dialects/scf/while)，D2；P/G/L | D3：WhileOp verifier、转 CFG |
| D09 | scf.execute_region、index_switch、parallel、forall、归约 | D05—08 | SCF 索引仅定位，未展开 | D2—3：逐个协议；D4：并行归约与映射 |
| D10 | tensor：empty/generate、extract/insert、slice、reshape、dim | C06、C09 | 未展开；效果章只有值语义例子 | D3：形状约束、动态尺寸与值更新 |
| D11 | memref：alloc/alloca、load/store、subview、layout/stride/space | C09—11 | 未展开操作参考；效果章解释基础边界 | D3—4：别名、越界、释放、描述符与地址计算 |
| D12 | linalg：structured op、indexing_maps、iterator_types、区域标量语义 | D10—11、D07 | 未展开 | D3—4：generic/named、DPS、广播/归约、结构化变换 |
| D13 | affine：dim/symbol、AffineMap/IntegerSet、访问与循环限制 | D04、D07 | 未展开 | D2—3：约束与合法性；D4 按循环优化需要深入 |
| D14 | vector：transfer、mask、contract、reduction、layout 与 lowering | D11—12 | 未展开 | D3—4：边界、向量化与目标指令映射 |
| D15 | math/index/complex：语义、展开与目标库调用 | D03 | 未展开 | D2—3：按算子需求选择；数值边界必须说明 |
| D16 | gpu/async：launch、线程层次、空间、同步、异步依赖 | D11、D14 | 未展开 | D3—4：GPU kernel/host 边界、异步与资源生命周期 |
| D17 | llvm 与目标方言：类型、指针、DataLayout、NVVM/ROCDL 等 | D11、M05 | 未展开 | D3：lowering/translation；D4：选定目标的 ABI 与性能 |
| D18 | 项目方言：StableHLO、Triton、Triton-Ascend/AscendNPU IR | C01—11、M05 | 本 MLIR 基础目录不重复项目文档 | D3—4：在目标仓库固定版本，核对实际入口和变换链 |

## 编译器构造、变换与 lowering

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 所需产物 |
|---|---|---|---|---|
| M01 | Context/Registry、Builder、InsertionPoint、walk、IRMapping、替换/删除 | C01—05 | 未展开；对象章给 API 定位 | D3：可编译最小 IR 构造与变换 |
| M02 | ODS、TableGen、Op/Type/Attr、builders、parser/printer、verifier | C02、C06—07、M01 | 未展开 | D3：小 Dialect，正反例与往返测试 |
| M03 | Trait、Op/Type/Attr Interface、外部模型、推导与效果接口 | C08—11、M02 | 未展开 | D3：供通用变换调用的接口实现 |
| M04 | fold、canonicalization、RewritePattern、benefit、driver、收敛、PDL/PDLL | M01、C11 | 未展开；教程展示 fold 现象 | D3：受约束改写与失败用例；D4：交互/收敛分析 |
| M05 | Dialect Conversion、ConversionTarget、动态合法性、TypeConverter、materialization | M02—04 | 未展开 | D3：1:N 类型变化、region/call 边界、partial/full conversion |
| M06 | Pass/OpPassManager、嵌套 pipeline、注册、选项、失败与线程约束 | M01、C08 | 未展开；指南仅使用命令 | D3：独立 Pass 和 lit/FileCheck 测试 |
| M07 | LLVM dialect lowering、ABI、descriptor、translation、JIT/AOT/runtime | D11、D17、M05 | 未展开 | D3：完整 CPU 可执行链；与设备后端边界分开验证 |
| M08 | DPS、One-Shot Bufferize、读写冲突、in-place/out-of-place、未知 Op | D10—12、M03 | 未展开 | D3—4：解释额外拷贝与 alias 分析结论 |
| M09 | buffer deallocation、ownership、逃逸与跨函数边界 | M08、C09 | 未展开 | D3：分配/释放与返回 buffer 的生命周期验证 |
| M10 | 转换 pipeline 的前后置条件、混合方言、失败定位与阶段 IR 契约 | M04—09 | 未展开 | D4：真实项目 pipeline 的约束与最小失败复现 |

## 分析与优化

| ID | 可检查知识项 | 前置 | 本次正文 | 后续目标 / 所需产物 |
|---|---|---|---|---|
| A01 | Dominance/PostDominance、Liveness、Alias、CallGraph | C05、C08—11 | 未展开 API | D3：选用分析并核对适用范围 |
| A02 | AnalysisManager、缓存、preserve/invalidate、Pass 依赖 | M06、A01 | 未展开 | D3：IR 修改后分析是否仍有效的测试 |
| A03 | DataFlowSolver、格与不动点、稠密/稀疏数据流、控制/区域传播 | C05、M03 | 未展开 | D3：小数据流分析；说明收敛与保守性 |
| A04 | CSE、DCE、LICM：等价性、效果、支配、终止与推测执行 | C11、M04、A01 | 效果章解释部分前提，未展开算法 | D3—4：适用条件与错误优化反例 |
| A05 | tiling、fusion、interchange：访问关系、依赖、reduction 与边界 | D12—13、A01 | 未展开 | D4：合法变换、额外内存流量和复用分析 |
| A06 | vectorization/unrolling、布局、mask、精度与指令映射 | D14、A05 | 未展开 | D4：数值/形状边界测试与性能证据 |
| A07 | Transform dialect：handle/payload、匹配、调度、失败与失效 | M04、A05 | 未展开 | D3—4：可复现 schedule 与对比 |
| A08 | GPU/NPU 映射、异步流水、同步、资源占用和代价模型 | D16—18、A05—07 | 未展开 | D4：在实际硬件/后端中解释正确性和性能 |

## 工具、扩展与验证

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 缺口 |
|---|---|---|---|---|
| X01 | mlir-opt、generic print、诊断、verifier、pipeline、IR dump | C01—11 | [阅读与验证](./guides/inspecting_ir)，D2；P/G/L/N | D3：崩溃复现、pass instrumentation 与调试器 |
| X02 | lit/FileCheck、verify-diagnostics、单测、语义/数值/性能验证 | X01、M04 | 指南解释证据分级，测试框架未展开 | D3—4：具有反例和边界的回归测试 |
| X03 | bytecode、版本升级、Dialect version、序列化兼容 | C01、M02 | 对象章仅定位 | D2—3：选择实际版本兼容案例 |
| X04 | C API、Python bindings、所有权/生命周期、调试打印 | M01 | 未展开 | D2—3：与实际工具脚本集成 |
| X05 | mlir-reduce、最小复现、统计/timing、LSP 与诊断定位 | X01 | 未展开 | D3：可复现错误与性能定位 |
| X06 | DataLayout、符号/shape 推导、跨目标约束、插件/动态注册 | C06、M02—03 | 未展开 | D2—3：按目标项目需要选择实现 |
| X07 | quant/sparse/shape/IRDL 等扩展领域 | 相应核心模块 | 范围已登记，未展开 | D1 定位；采用具体项目时升至 D2—3 |

## 单章展开检查表

编写者在生成正文前按下列结构检查，不以用户是否问到某个问题决定它是否重要。

| 维度 | 必须覆盖的内容 | 当前 SCF for 的例子 |
|---|---|---|
| 边界与依赖 | 它属于什么抽象，本章止于哪里 | 顺序结构化循环；并行循环另列 |
| 契约 | 输入、结果、静态信息、Region、Block、terminator | 边界/步长/init、IV/iter_args、yield/result |
| 语义 | 参数绑定、执行顺序、退出与状态传递 | 首轮、回边、末轮、零轮 |
| 约束 | 数量、类型、作用域、动态前提 | 携带值一一对应；正步长；内层值不逃逸 |
| 变体与反例 | 无结果、多结果、边界、错误例子 | 空 yield、多携带值、类型不匹配 |
| 设计与联系 | 为什么这样表达，与相邻表示如何对应 | SSA 状态、CF header 参数与回边 |
| 查证 | 规范、ODS、必要 C++ 与测试入口 | ForOp、SCF.cpp、SCFToControlFlow.cpp |
| 检查 | 能否独立推演；验证确实检查了什么 | 执行表、P/G/L/N；不冒充机器码执行 |

## 更新规则

新增/扩写正文时同步更新对应 ID 的位置、展开深度和证据；发现正文只提到术语时标为概述或缺口。章节完成、示例通过、用户阅读、用户掌握分别记录。当前正式实践尚未开始，因此不存在“表内内容均已掌握”的结论。

范围依据：[官方文档分类](https://mlir.llvm.org/docs/)、[方言目录](https://mlir.llvm.org/docs/Dialects/)、[教程目录](https://mlir.llvm.org/docs/Tutorials/)。API/语义的具体深度按固定版本源码及所选项目需要核对。
