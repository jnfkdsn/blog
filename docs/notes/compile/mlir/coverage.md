---
order: 2
title: 计划覆盖范围与缺口
updated: 2026-09-14
---

# 计划覆盖范围与缺口

本表约束后续章节的广度和深度。目标是支撑 AI Compiler 的阅读、开发、转换与优化工作，不以穷举所有上游 Dialect 和 API 为完成条件。新项目发现超出范围的机制时，先增补此表，再编写正文。

机制模块入口为[IR 变换基础](./compiler/transforms/)与[定义 IR 抽象](./compiler/ir_definition/)。总览解释各章关系，下面各 ID 保留精确正文与范围，不因目录重组改变学习完成标准。

## 当前进度

阶段 A 的基础阅读与最小 IR 实验已完成，阶段 B 原理与小 Pass 工程已完成理解，阶段 C 的操作定义也已完成阅读，当前继续接口、类型/属性和区域定义。[Pass 与 pipeline](./compiler/transforms/passes)已读完；[C++ IR API](./compiler/transforms/ir_api)核心问题已讨论，配套 `03-ir-api` 观察实验已提供。[PatternRewriter](./compiler/transforms/rewriting)已按单条规则的完整过程重写，主线与可理解性已获肯定；[改写驱动](./compiler/transforms/rewrite_drivers)保留多规则与驱动细节，两篇的独立 C++ 示例已验证；[小 Pass 贯通教程](./tutorials/first_pass)与 `04-small-pass` 工程已验证，学习者已完成理解，独立修改尚待验证。[操作定义](./compiler/ir_definition/op_definition)及 `05-op-definition` 的 ODS/C++ 示例已验证，学习者已完成阅读。[定义 IR 抽象](./compiler/ir_definition/)接口、类型/属性、Region 与解析打印四章核心正文及 `06-ir-definition` 工程，已验证、待阅读。原先 14 篇正文保留为基础知识与查阅入口。

阶段完成按[学习路径](./learning_path)的基础目标判断，不把本表全部长期深度要求追加成同一阶段的关卡。C 调用演示已经由助手验证，完整 ABI、转换实现和所有权分析仍属于后续内容。

## 状态与深度

- **D1 辨识**：说明用途、输入输出、与相邻机制的边界。
- **D2 推演**：独立解释完整实例、约束、边界与反例，并定位规范。
- **D3 实现**：完成最小实现或修改，读到相关 ODS/C++/测试，验证成功和失败路径。
- **D4 分析**：解释正确性条件、分析/优化的取舍，给出复现和性能或工程证据。

表中的“本次正文”只记录写作状态；“后续目标”才是长期深度要求。`已展开 D2` 不代表用户已经掌握，也不等同于相应 C++ API 已实现验证。

示例证据按四类记录：P 解析与 verifier，G 通用打印再解析，L 指定 lowering，N 预期失败诊断。证据清单由[文档验证脚本](./guides/inspecting_ir#文档示例的复现)生成；没有机器码执行的地方不记录数值测试通过。纯概念条目的依据是各章列出的固定版本规范与源码。

C++ IR API 章额外由 `validate_ir_api.py` 编译并运行操作 IR 的 C++ 程序，记录源码片段对应、输出、未匹配输入和支配/隔离诊断。这属于编译器侧实现证据，不是把生成的算子 lower 成机器码后的数值证据，也不等于学习者已完成独立实验。

PatternRewriter 主章与改写驱动篇由 `validate_rewriting.py` 编译并核对单条规则的前后 IR、未匹配推演及规则与 driver 的输出、成功轨迹、未匹配输入、fold/DCE 对照、benefit 选择和有界停止。失败证据包含 driver 未收敛但 IR 仍合法的情况，不能将它误记为 verifier 拒绝或数值错误。

小 Pass 章由 `validate_small_pass.py` 核对正文与实际 C++ 工程、输入输出、重复应用和错误 pipeline 层级，并通过 lit/FileCheck 检查多 use、嵌套 Region、函数声明与合法未匹配输入。它提供编译器侧工程证据，不将作者已实现的示例记为学习者已独立完成。

操作定义章使用 `lab-example` / `lab-invalid` 标记，由 `validate_op_definition.py` 用注册 Lab dialect 的工具单独验证；标准方言文档脚本不处理未知自定义操作。验证包括生成源码与正文对应、打印往返、builder 及自定义验证、展开输出与 lit 正反例，不作为目标机器码执行或学习者独立实现证据。

定义 IR 抽象后四章使用 `irdef-example` / `irdef-invalid` 标记，由 `validate_ir_definition.py` 核对正文、生成源码和 15 组预期诊断，并检查外部模型有无、通用死代码删除、类型身份与 checked 构造、区域与文本往返。这里的 D3 作者工程证据不包含范围比较优化、RegionBranch 实现或自定义操作的目标执行。

## 核心 IR

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 尚未覆盖 |
|---|---|---|---|---|
| C01 | 文本、内存对象、bytecode 的对应；parser/printer；Context | — | [对象模型](./core/ir_model)，D2；P/G；[IR API](./compiler/transforms/ir_api)补 Context/Registry、拥有关系与解析程序 | D3：独立构造与解析；bytecode 版本机制见 X03 |
| C02 | Operation 字段、Op 包装类、递归所有权；四种关系 | C01 | [对象模型](./core/ir_model)，D2；P/G；[IR API](./compiler/transforms/ir_api)补句柄、walk、插入、删除，C++ 已验证 | D3：独立实现；跨 Block/Region 移动的完整合法性仍待展开 |
| C03 | Region/Block/terminator；SSACFG 与 Graph；空 Region | C02 | [对象模型](./core/ir_model)，D2；[区域操作定义](./compiler/ir_definition/regions_assembly)补单 Block 传值、空列表与空 Region 区别及验证 | D3：RegionKindInterface 与克隆/内联边界 |
| C04 | OpResult/BlockArgument；operand、use、user；多结果 | C02 | [SSA](./core/values_ssa)，D2；P/G；[IR API](./compiler/transforms/ir_api)补 use-list、RAUW、参数/结果映射克隆与反例 | D3：独立改写；多 Block/复杂 Region 映射仍待展开 |
| C05 | CFG 支配、合流、回边、层次支配与作用域 | C03—04 | [SSA](./core/values_ssa)，D2；P/N | D3：DominanceInfo 与修改 CFG 后的维护 |
| C06 | 标量/容器/函数类型；动态 shape；index；类型相等与 cast | C04 | [类型与静态信息](./core/types_attributes)，D2；P/G | D3：类型构造、推导与数据布局查询 |
| C07 | Attribute、inherent/discardable、Properties、别名、Location | C02、C06 | [类型与静态信息](./core/types_attributes)，D2；P/G | D3：自定义存储、ODS accessor、验证与打印 |
| C08 | SymbolTable、SymbolRef、查找、可见性、IsolatedFromAbove | C03—04 | [符号与作用域](./core/symbols_scopes)，D2；P/N | D3：符号替换、重命名、未知符号使用的保守处理 |
| C09 | Tensor 值与 MemRef 存储、view/alias、生命周期 | C04、C06 | [效果模型](./core/effects)，D2 边界；P/G | D3—4：完整内存模型、所有权与 bufferization |
| C10 | Read/Write/Allocate/Free；未知效果；递归效果 | C09 | [效果模型](./core/effects)，D2；[接口章](./compiler/ir_definition/traits_interfaces)补真实 DCE 与缺失效果信息对照；区域章使用递归效果 | D3：MemoryEffectOpInterface、Resource 与阶段化效果 |
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
| M01 | Context/Registry、Builder、InsertionPoint、walk、IRMapping、替换/删除 | C01—05 | [C++ IR API](./compiler/transforms/ir_api)已展开；单 Block 函数构造、改写、克隆的 C++ 实例已编译运行，含 N | D3：学习者独立实现；复杂 CFG/Region 修改及 driver 协议后续展开 |
| M02 | ODS、TableGen、Op/Type/Attr、builders、parser/printer、verifier | C02、C06—07、M01 | [操作定义](./compiler/ir_definition/op_definition)、[Type/Attr](./compiler/ir_definition/types_attributes)、[Region](./compiler/ir_definition/regions_assembly)、[解析打印](./compiler/ir_definition/assembly_format)展开 D2 主干，作者 D3 示例已编译：参数化存储、checked 构造、两层验证、单 Block/单组 variadic、隔离及手写 parser/printer | D3：学习者独立定义；多组/嵌套 variadic 实现、复杂语法、自定义/可变存储仍待深入 |
| M03 | Trait、Op/Type/Attr Interface、外部模型、推导与效果接口 | C08—11、M02 | [Trait/Interface](./compiler/ir_definition/traits_interfaces)展开 D2：效果/DCE 决策、协议与消费、直接及外部 Op 模型、Context 注册对照，作者 C++ 示例已验证；Type/Attr 与推导/区域接口为 D1 定位 | D3：学习者独立实现；Type/Attr Interface、InferType、RegionBranch 及资源/阶段化效果模型后续深入 |
| M04 | fold、canonicalization、RewritePattern、benefit、driver、收敛、PDL/PDLL | M01、C11 | [PatternRewriter](./compiler/transforms/rewriting)展开匹配、替换、调用与修改通知；[改写驱动](./compiler/transforms/rewrite_drivers)展开原地更新、Walk/Greedy、benefit、fold 与收敛；C++ 规则和对照已验证，PDL/PDLL 仅定位 | D3：学习者独立实现；Op fold/canonicalization 注册随 M02 展开；D4：复杂规则交互；声明式规则专题待展开 |
| M05 | Dialect Conversion、ConversionTarget、动态合法性、TypeConverter、materialization | M02—04 | 未展开 | D3：1:N 类型变化、region/call 边界、partial/full conversion |
| M06 | Pass/OpPassManager、嵌套 pipeline、注册、选项、失败与线程约束 | D2：C02、C08；D3：M01 | [Pass 与 pipeline](./compiler/transforms/passes)，运行模型 D2；[小 Pass 教程](./tutorials/first_pass)展开函数 Pass、注册、工具集成及失败处理，C++ 工程已验证 | D3：独立 Pass 和 lit/FileCheck 测试 |
| M07 | LLVM dialect lowering、ABI、descriptor、translation、JIT/AOT/runtime | D11、D17、M05 | 未展开 | D3：完整 CPU 可执行链；与设备后端边界分开验证 |
| M08 | DPS、One-Shot Bufferize、读写冲突、in-place/out-of-place、未知 Op | D10—12、M03 | 未展开 | D3—4：解释额外拷贝与 alias 分析结论 |
| M09 | buffer deallocation、ownership、逃逸与跨函数边界 | M08、C09 | 未展开 | D3：分配/释放与返回 buffer 的生命周期验证 |
| M10 | 转换 pipeline 的前后置条件、混合方言、失败定位与阶段 IR 契约 | M04—09 | 未展开 | D4：真实项目 pipeline 的约束与最小失败复现 |

## 分析与优化

| ID | 可检查知识项 | 前置 | 本次正文 | 后续目标 / 所需产物 |
|---|---|---|---|---|
| A01 | Dominance/PostDominance、Liveness、Alias、CallGraph | C05、C08—11 | 未展开 API | D3：选用分析并核对适用范围 |
| A02 | AnalysisManager、缓存、preserve/invalidate、Pass 依赖 | M06、A01 | Pass 章已解释缓存有效性的动机与 CSE 示例；完整 API 未展开 | D3：IR 修改后分析是否仍有效的测试 |
| A03 | DataFlowSolver、格与不动点、稠密/稀疏数据流、控制/区域传播 | C05、M03 | 未展开 | D3：小数据流分析；说明收敛与保守性 |
| A04 | CSE、DCE、LICM：等价性、效果、支配、终止与推测执行 | C11、M04、A01 | 效果章解释前提；Pass 章展开直线 CSE 主例与必要源码路径，完整算法/LICM 未展开 | D3—4：适用条件与错误优化反例 |
| A05 | tiling、fusion、interchange：访问关系、依赖、reduction 与边界 | D12—13、A01 | 未展开 | D4：合法变换、额外内存流量和复用分析 |
| A06 | vectorization/unrolling、布局、mask、精度与指令映射 | D14、A05 | 未展开 | D4：数值/形状边界测试与性能证据 |
| A07 | Transform dialect：handle/payload、匹配、调度、失败与失效 | M04、A05 | 未展开 | D3—4：可复现 schedule 与对比 |
| A08 | GPU/NPU 映射、异步流水、同步、资源占用和代价模型 | D16—18、A05—07 | 未展开 | D4：在实际硬件/后端中解释正确性和性能 |

## 工具、扩展与验证

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 缺口 |
|---|---|---|---|---|
| X01 | mlir-opt、generic print、诊断、verifier、pipeline、IR dump | C01—11 | [阅读与验证](./guides/inspecting_ir) + [Pass 与 pipeline](./compiler/transforms/passes)，D2；新增真实 pipeline 输出、嵌套选择与失败检查 | D3：崩溃复现、pass instrumentation 与调试器 |
| X02 | lit/FileCheck、verify-diagnostics、单测、语义/数值/性能验证 | X01、M04 | 指南解释证据分级；[小 Pass 教程](./tutorials/first_pass)展开 lit/FileCheck 与正反例，工程已验证；[操作定义](./compiler/ir_definition/op_definition)补充 verify-diagnostics 与分割错误用例；单测框架后续展开 | D3—4：具有反例和边界的回归测试 |
| X03 | bytecode、版本升级、Dialect version、序列化兼容 | C01、M02 | 对象章仅定位 | D2—3：选择实际版本兼容案例 |
| X04 | C API、Python bindings、所有权/生命周期、调试打印 | M01 | 未展开 | D2—3：与实际工具脚本集成 |
| X05 | mlir-reduce、最小复现、统计/timing、LSP 与诊断定位 | X01 | 未展开 | D3：可复现错误与性能定位 |
| X06 | DataLayout、符号/shape 推导、跨目标约束、插件/动态注册 | C06、M02—03 | 未展开 | D2—3：按目标项目需要选择实现 |
| X07 | quant/sparse/shape/IRDL 等扩展领域 | 相应核心模块 | 范围已登记，未展开 | D1 定位；采用具体项目时升至 D2—3 |

## 单章展开检查表

编写者在生成正文前按下列维度检查覆盖，不以用户是否问到某个问题决定它是否重要。它是备课检查表，不是正文必须逐项照搬的章节顺序；实际叙事参照工作区 `writing_method.md`，沿连续案例和推演组织。

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

新增/扩写正文时同步更新对应 ID 的位置、展开深度和证据；发现正文只提到术语时标为概述或缺口。章节完成、示例通过、用户阅读、用户掌握分别记录。阶段 A 的最小实验已完成；后续章节不会仅因生成或通过维护脚本就记为学习者已掌握。Pass 章的行为证据单列在 `artifacts/logs/mlir-docs/2026-09-08-passes/manifest.json`，包括输出、调度范围、日志顺序与失败行为。

范围依据：[官方文档分类](https://mlir.llvm.org/docs/)、[方言目录](https://mlir.llvm.org/docs/Dialects/)、[教程目录](https://mlir.llvm.org/docs/Tutorials/)。API/语义的具体深度按固定版本源码及所选项目需要核对。
