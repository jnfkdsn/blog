---
order: 2
title: 计划覆盖范围与缺口
updated: 2026-10-04
---

# 计划覆盖范围与缺口

本表约束后续章节的广度和深度。目标是支撑 AI Compiler 的阅读、开发、转换与优化工作，不以穷举所有上游 Dialect 和 API 为完成条件。新项目发现超出范围的机制时，先增补此表，再编写正文。

机制模块入口为[IR 变换基础](./compiler/transforms/)、[定义 IR 抽象](./compiler/ir_definition/)与[IR 转换](./compiler/conversion/)。总览解释各章关系，下面各 ID 保留精确正文与范围，不因目录重组改变学习完成标准。

## 当前内容状态

2026-10-04 已完成当前章节目录约定的后续正文与按需扩展，相关作者实现及验证记录见下文。表中保留的长期 D3/D4 目标用于后续独立实践和项目研究，不表示还缺少对应入门或机制正文。

基础 IR、变换基础与 IR 定义模块均已有正文和相应编译器侧验证。2026-10-01 已按 my.add 导读的阅读反馈重构 IR 定义五章：导读负责生成与框架接入，机制章节负责字段、契约、消费者和文本往返。2026-10-02 已将导读并入本模块，统一 `my` 教学命名与对应工程；旧迁移页已移除，站内入口直接指向现位置。clamp 的普通 Pattern 展开已有实现，my.add 工程自身尚未加入该变换；后续学习不以另写一套 my.add 课程为前提。

本表记录文档和实现范围，不将旧版阅读记录直接等同于新版理解，也不将作者验证当作学习者独立完成。阶段深度见[学习路径](./learning_path)，拟定文章及案例见[后续章节目录](./chapter_plan)。

## 状态与深度

- **D1 辨识**：说明用途、输入输出、与相邻机制的边界。
- **D2 推演**：独立解释完整实例、约束、边界与反例，并定位规范。
- **D3 实现**：完成最小实现或修改，读到相关 ODS/C++/测试，验证成功和失败路径。
- **D4 分析**：解释正确性条件、分析/优化的取舍，给出复现和性能或工程证据。

表中的“本次正文”只记录写作状态；“后续目标”才是长期深度要求。`已展开 D2` 不代表用户已经掌握，也不等同于相应 C++ API 已实现验证。

示例证据按四类记录：P 解析与 verifier，G 通用打印再解析，L 指定 lowering，N 预期失败诊断。证据清单由[文档验证脚本](./guides/inspecting_ir#文档示例的复现)生成；没有机器码执行的地方不记录数值测试通过。纯概念条目的依据是各章列出的固定版本规范与源码。

作者验证按模块保留完整命令、退出状态、IR、诊断与文件哈希。基础与Conversion工程覆盖对象修改、定义和合法化；09—17工程进一步提供张量/内存/CPU、分析、算法、结构化优化及设备编译证据；18记录四个项目的有限源码链；19—26补接口、存储、1:N/声明式规则、调试、绑定、版本/插件、领域表示与结构维护。具体检查范围见各章与实验README，不将不同层次的通过数量合并为语义证明。

CPU数值与部分固定配置计时已经执行；GPU部分完成NVVM/PTX/cubin及Host object编译，当前没有GPU/NPU设备运行或性能证据。真实项目为固定源码阅读，未构建项目工具链。宽index_switch截断等固定版本限制单独记录，不计为正确性用例。作者已完成参考实现与验证不表示学习者完成独立练习。

## 核心 IR

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 尚未覆盖 |
|---|---|---|---|---|
| C01 | 文本、内存对象、bytecode 的对应；parser/printer；Context | — | [对象模型](./core/ir_model)，D2；P/G；[IR API](./compiler/transforms/ir_api)补 Context/Registry、拥有关系与解析程序 | D3：独立构造与解析；bytecode 版本机制见 X03 |
| C02 | Operation 字段、Op 包装类、递归所有权；四种关系 | C01 | [对象模型](./core/ir_model)、[IR API](./compiler/transforms/ir_api)与[CFG维护](./compiler/transforms/ir_maintenance)，D2/D3作者实现：拥有/句柄、移动、嵌套捕获及非法支配反例 | 独立实现；复杂跨作用域移动按效果/控制条件证明 |
| C03 | Region/Block/terminator；SSACFG 与 Graph；空 Region | C02 | [对象模型](./core/ir_model)，D2；[区域操作定义](./compiler/ir_definition/regions_assembly)补单 Block 传值、空列表与空 Region 区别及验证 | [区域接口](./compiler/ir_definition/region_interfaces)补 RegionKind 查询、自定义 RegionBranch、内联边界；复杂克隆映射按任务展开 |
| C04 | OpResult/BlockArgument；operand、use、user；多结果 | C02 | [SSA](./core/values_ssa)、[IR API](./compiler/transforms/ir_api)及[CFG维护](./compiler/transforms/ir_maintenance)：use/RAUW、多结果和四Block/嵌套Region克隆映射已验证 | 独立改写；符号映射与SSA映射分别维护 |
| C05 | CFG 支配、合流、回边、层次支配与作用域 | C03—04 | [SSA](./core/values_ssa)与[支配分析](./compiler/analysis/dominance_liveness)；26工程提供非法移动/修复，D2/D3作者查询 | 独立维护CFG与所用分析 |
| C06 | 标量/容器/函数类型；动态 shape；index；类型相等与 cast | C04 | [类型](./core/types_attributes)、[推导](./compiler/ir_definition/type_shape_inference)、[DataLayout](./dialects/llvm/data_layout)、[Index](./dialects/math/index_arithmetic)均有正文与对应查询/转换 | 跨目标实际ABI与运行时尺寸条件按项目验证 |
| C07 | Attribute、inherent/discardable、Properties、别名、Location | C02、C06 | [静态信息](./core/types_attributes)、[复杂ODS](./compiler/ir_definition/complex_ods)与[存储](./compiler/ir_definition/storage_interfaces)，D2/D3作者实现涵盖Properties、参数复制及接口 | 独立实现领域参数、验证和文本/版本兼容 |
| C08 | SymbolTable、SymbolRef、查找、可见性、IsolatedFromAbove | C03—04 | [符号](./core/symbols_scopes)与[维护](./compiler/transforms/ir_maintenance)，D2/D3：真实rename更新caller、克隆保留SymbolRef、未知作用域应用保护 | 复杂符号图和可见性规则按项目维护 |
| C09 | Tensor 值与 MemRef 存储、view/alias、生命周期 | C04、C06 | [MemRef](./dialects/memref/)与[内存模块](./compiler/memory/)已展开布局、alias、复用、复制、跨函数所有权和释放，有CPU对照 | 复杂异步/设备存储生命周期需目标验证 |
| C10 | Read/Write/Allocate/Free；未知效果；递归效果 | C09 | [效果](./core/effects)、[接口](./compiler/ir_definition/traits_interfaces)与[资源/阶段](./compiler/analysis/resources_effects)；真实copy/alloca查询与单独协议实例、DCE对照 | 阶段/资源参与一般优化的证明按任务实现 |
| C11 | 无内存效果、可推测执行、UB、不终止与改写条件 | C05、C10 | [效果](./core/effects)与[合法性](./compiler/analysis/legality)解释副作用、推测、零次循环与UB边界；CPU前后对照 | 独立论证新的移动/投机变换，不以verifier替代语义证明 |

## 方言语义

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 尚未覆盖 |
|---|---|---|---|---|
| D01 | builtin.module、Graph 容器、类型/属性归属、转换占位 cast | C01—03、C06 | [builtin](./dialects/builtin)，D2 | D3：module 构造；cast reconciliation 的实现 |
| D02 | func 定义/声明、call/return、签名、隔离、间接调用 | C08 | [func](./dialects/func)，直接调用 D2；间接调用 D1 | D3：FunctionOpInterface、CallOpInterface、内联与 ABI |
| D03 | arith constant、整数运算/比较、浮点语义、cast/select | C06—07 | [arith](./dialects/arith)，D2 核心；P/G | D3：完整常用操作族、fold；D4：数值语义与优化 |
| D04 | cf.br/cond_br、参数合流、回边、入口约束 | C05、D03 | [CF](./dialects/cf)与[多路选择](./dialects/scf/regions_switch)展开br/cond_br/switch及参数合流；[Shape约束](./dialects/extensions/shape)补cf.assert实际输出 | 一般CFG编辑、异常/断言执行策略按任务验证 |
| D05 | SCF 区域协议、控制转移与 SSA 结果的关系 | C03—05、D04 | [SCF](./dialects/scf/)，D2 | D3：RegionBranchOpInterface、LoopLikeOpInterface |
| D06 | scf.if：分支结果、空 else、捕获、嵌套、select 边界 | D05、D03 | [if](./dialects/scf/if)，D2；P/G/L/N | D3：IfOp canonicalization 与 lowering |
| D07 | scf.for：边界、迭代状态、多结果、零次循环、类型与 CFG | D05 | [for](./dialects/scf/for)，D2；P/G/L/N | D3：ForOp 实现；D4：循环变换合法性 |
| D08 | scf.while：before/after、condition/yield、两套类型、do-while | D05—07 | [while](./dialects/scf/while)，D2；P/G/L | D3：WhileOp verifier、转 CFG |
| D09 | scf.execute_region、index_switch、parallel、forall、归约 | D05—08 | [多Block/选择](./dialects/scf/regions_switch)、[并行结果](./dialects/scf/parallel_results)展开D2；26工程含多出口CF、parallel归约、forall切片组装及CPU参考 | 固定20.1.8宽selector与unique-yield限制已记录；真实设备并行归16及目标项目 |
| D10 | tensor：empty/generate、extract/insert、slice、reshape、dim | C06、C09 | [Tensor 三篇](./dialects/tensor/)，D2：值更新、动态尺寸、窗口/步长/降秩、reshape 与形状前提；09 实验含往返、静态失败和数值证据 | D3：独立修改；更复杂 shape 推导、边界与领域扩展按需补充 |
| D11 | memref：alloc/alloca、load/store、subview、layout/stride/space | C09—11 | [MemRef 三篇](./dialects/memref/)，D2：分配、动态尺寸、地址计算、布局 cast、嵌套/跨步视图与别名；关键结果有本机观察 | D3—4：独立修改、一般布局、运行时检查、descriptor ABI 与目标内存空间 |
| D12 | linalg：structured op、indexing_maps、iterator_types、区域标量语义 | D10、D07；buffer 形式再接 D11 | [Linalg 三篇](./dialects/linalg/)，D2：逐元素、广播、归约、matmul、DPS；generic/named、循环与小矩阵数值已验证 | D3—4：独立实现、复杂映射、tiling/fusion 与目标优化 |
| D13 | affine：dim/symbol、AffineMap/IntegerSet、访问与循环限制 | D04、D07 | [Affine 三篇](./dialects/affine/)展开 D2：窗口映射、scope/绑定、集合/余块、依赖距离、semi-affine 与失败边界；作者真实分析和 CPU 对照已验证 | D3：学习者实现约束查询；D4：一般多面体依赖与变换实现 |
| D14 | vector：transfer、mask、contract、reduction、layout 与 lowering | D11—12 | [三篇](./dialects/vector/)＋15 实验；尾块、类型/操作降低与 CPU 数值已核验；[设备布局](./compiler/targets/layout_transfer)继续解释分布与bank | D3—4：边界、向量化与目标指令映射 |
| D15 | math/index/complex：数学函数、近似、整数范围、index 与复数语义 | D03、C06 | [Math/Index 两篇](./dialects/math/)展开 D2：exp 实现、expm1 消去、fast-math 边界、ceildiv/余块、位宽与折叠；真实 libm/LLVM 对照与 64 位数值已验证 | D3—4：目标数学实现、误差与性能取舍；32 位执行按目标环境补充，[Complex](./dialects/math/complex)已有乘法/abs展开和折叠证据 |
| D16 | gpu/async：launch、线程层次、空间、同步、异步依赖 | D11、D14 | [GPU 三篇](./dialects/gpu/)与 16 实验：真实映射/NVVM/PTX/cubin/Host object 与逻辑依赖模型；设备执行未完成 | D3—4：GPU kernel/host 边界、异步与资源生命周期 |
| D17 | llvm 与目标方言：类型、指针、DataLayout、NVVM/ROCDL 等 | D11、M05 | [LLVM 三篇](./dialects/llvm/)展开指针、布局、descriptor、index 位宽与 translation；CPU 侧有编译与调用证据；[Host/Device](./compiler/targets/host_device)展开 NVVM 与设备对象/启动协议 | D3：lowering/translation；D4：选定目标的 ABI 与性能 |
| D18 | 项目方言：StableHLO、Triton、Triton-Ascend/AscendNPU IR | C01—11、M05 | [Triton](/notes/triton/compiler_path)、[Triton-Ascend](/notes/triton/ascend_compiler_path)、[HIVM](/notes/compile/ai-compiler/npu-ir/source_path)固定源码链已有，未构建/执行项目；[StableHLO/TOSA](./dialects/extensions/graph_ir)定位与TOSA→Linalg已有 | D3—4：在目标仓库固定版本，核对实际入口和变换链 |

## 编译器构造、变换与 lowering

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 所需产物 |
|---|---|---|---|---|
| M01 | Context/Registry、Builder、InsertionPoint、walk、IRMapping、替换/删除 | C01—05 | [C++ IR API](./compiler/transforms/ir_api)、[CFG维护](./compiler/transforms/ir_maintenance)已有构造、克隆、符号和移动作者实现；driver通知见改写主章 | 学习者独立实现所选结构变换及不变量检查 |
| M02 | ODS、TableGen、Op/Type/Attr、builders、parser/printer、verifier | C02、C06—07、M01 | [操作定义](./compiler/ir_definition/op_definition)、[Type/Attr](./compiler/ir_definition/types_attributes)、[Region](./compiler/ir_definition/regions_assembly)、[解析打印](./compiler/ir_definition/assembly_format)展开 D2 主干，作者 D3 示例已编译：参数化存储、checked 构造、两层验证、单 Block/单组 variadic、隔离及手写 parser/printer | D3：学习者独立定义；[复杂 ODS](./compiler/ir_definition/complex_ods)与[存储](./compiler/ir_definition/storage_interfaces)已有作者实现：多组/嵌套 variadic、Properties、数组复制、递归名义身份；复杂领域语法按项目深入 |
| M03 | Trait、Op/Type/Attr Interface、外部模型、推导与效果接口 | C08—11、M02 | [Trait/Interface](./compiler/ir_definition/traits_interfaces)展开 D2：效果/DCE 决策、协议与消费、直接及外部 Op 模型、Context 注册对照，作者 C++ 示例已验证；[类型/形状](./compiler/ir_definition/type_shape_inference)及[区域接口](./compiler/ir_definition/region_interfaces)已补 D2/D3 作者实现：生成 InferType、Reify 消费、RegionBranch/SCCP 与内联；[Type/Attr 接口](./compiler/ir_definition/storage_interfaces)含实际规划消费者与拒绝边界 | D3：学习者独立实现；Type/Attr、InferType、RegionBranch 已有作者实现；[资源/阶段](./compiler/analysis/resources_effects)已有协议与查询；一般优化消费者按项目深化 |
| M04 | fold、canonicalization、RewritePattern、benefit、driver、收敛、PDL/PDLL | M01、C11 | [PatternRewriter](./compiler/transforms/rewriting)展开匹配、替换、调用与修改通知；[改写驱动](./compiler/transforms/rewrite_drivers)展开原地更新、Walk/Greedy、benefit、fold 与收敛；C++ 规则和对照已验证，[声明式改写](./compiler/transforms/declarative_patterns)补 DRR 与 PDLL→PDL 的实际驱动、原生约束及未匹配对照 | D3：学习者独立实现；Op fold/canonicalization 注册随 M02 展开；D4：复杂规则交互；声明式规则已有作者实现，复杂规则选择按项目深入 |
| M05 | Dialect Conversion、ConversionTarget、动态合法性、TypeConverter、materialization | M02—04 | [转换模块](./compiler/conversion/)，D2：4 篇主线含类型、函数/call/region 边界；[进阶参考](./compiler/conversion/advanced)解释 analysis、递归合法性、路径与 1:N；07/08 作者 C++ 正反例已验证 | D3：学习者独立修改；[1:N](./compiler/conversion/one_to_many)已有 pair 跨函数/call/CFG 的作者实现及八个 C 数值；更复杂聚合/区域/ABI 按项目深入 |
| M06 | Pass/OpPassManager、嵌套 pipeline、注册、选项、失败与线程约束 | D2：C02、C08；D3：M01 | [Pass 与 pipeline](./compiler/transforms/passes)，运行模型 D2；[小 Pass 教程](./tutorials/first_pass)展开函数 Pass、注册、工具集成及失败处理，C++ 工程已验证 | D3：独立 Pass 和 lit/FileCheck 测试 |
| M07 | LLVM dialect lowering、ABI、descriptor、translation、JIT/AOT/runtime | D11、D17、M05 | [CPU Lowering](./compiler/lowering/)与[贯通教程](./tutorials/pipelines/cpu)展开循环、ABI、C wrapper、translation、JIT/AOT；11 工程含动态非连续布局、返回分配和分层失败证据 | D3：解释 ABI、布局、完整生命周期和运行时；设备边界单独验证 |
| M08 | DPS、One-Shot Bufferize、读写冲突、in-place/out-of-place、未知 Op | D10—12、M03 | [内存模块前两篇](./compiler/memory/)连接值语义、可写性、读写冲突、operand 决策、alias/equivalence、完整覆盖与窗口复用；有分析输出及未知 Op 正反例 | D3—4：独立解释/修改决策案例、[自定义接口](./compiler/memory/custom_operations)已有作者实现/四组 CPU 数值；复杂控制流分析按项目深入 |
| M09 | buffer deallocation、ownership、逃逸与跨函数边界 | M08、C09 | [所有权与释放](./compiler/memory/ownership)解释局部、返回、借用、别名返回和分支责任；具体结构及数值已验证 | D3：独立追踪完整生命周期；动态私有函数协议、异步与目标内存管理按需深入 |
| M10 | 转换 pipeline 的前后置条件、混合方言、失败定位与阶段 IR 契约 | M04—09 | [流程导读](./tutorials/pipelines/overview)、[CPU全链路](./tutorials/pipelines/cpu)、[前端源码](./tutorials/pipelines/frontend)、[调试复现](./guides/debugging)与[SCF边界](./dialects/scf/regions_switch)已连接阶段契约、混合表示和失败定位 | 真实项目源码已固定；项目构建、设备执行和独立调试按研究任务完成 |

## 分析与优化

| ID | 可检查知识项 | 前置 | 本次正文 | 后续目标 / 所需产物 |
|---|---|---|---|---|
| A01 | Dominance/PostDominance、Liveness、Alias、CallGraph | C05、C08—11 | [可用性](./compiler/analysis/dominance_liveness)与[别名/效果](./compiler/analysis/alias_effects)展开 D2：层次/CFG 支配、后支配、SSA 活跃性、调用边界、ModRef 与视图精度；作者实际 C++ 查询已验证 | D3：学习者按优化任务选用分析；D4：精确访问与跨函数分析 |
| A02 | AnalysisManager、缓存、preserve/invalidate、Pass 依赖 | M06、A01 | [缓存失效](./compiler/analysis/caching)用操作数 9→10 解释按需计算、保留声明、Pass 内过期与依赖；真实 PM 正反例已验证 | D3：学习者独立维护所用分析；更精细增量更新按项目深入 |
| A03 | DataFlowSolver、格与不动点、稠密/稀疏数据流、控制/区域传播 | C05、M03 | [数据流](./compiler/analysis/dataflow)展开 D2：常量合流、可达性、单调收敛、稀疏/稠密与范围；作者现代 solver 查询及变体已验证 | D3：实现自定义传递与域；D4：路径精度、收敛成本和跨函数传播 |
| A04 | CSE、DCE、LICM：等价性、效果、支配、终止与推测执行 | C11、M04、A01 | [合法性](./compiler/analysis/legality)连接重复计算、死结果、写入阻断与零次循环除法反例；结构检查及优化前后各 15 个 C 观察已验证 | D3—4：独立实现、证明额外条件和目标相关取舍；[归约数值](./guides/numerics_performance)和Vector篇补浮点重排边界 |
| A05 | tiling、fusion、interchange：访问关系、依赖、reduction 与边界 | D12—13、A01 | [优化前三篇](./compiler/optimization/)展开 D2：2×3 余块、点式融合/邻域重算、置换 maps、依赖与归约条件；实际变换、分配差异和 33 组 CPU 判据已验证 | [Matmul](./tutorials/kernels/matmul)补调度与展开；D4目标性能需相应硬件实测 |
| A06 | vectorization/unrolling、布局、mask、精度与指令映射 | D14、A05 | [向量化](./compiler/optimization/vectorization)与[性能](./compiler/optimization/performance)已有，59 组数值与固定 x86-64 计时；[Matmul](./tutorials/kernels/matmul)补 K 块二倍展开、余数路径及数值对照；目标特化性能仍按硬件验证 | D4：数值/形状边界测试与性能证据 |
| A07 | Transform dialect：handle/payload、匹配、调度、失败与失效 | M04、A05 | [正文](./compiler/optimization/transform)及真实 tile/vectorize、失效、失败传播/抑制对照已有 | D3—4：可复现 schedule 与对比 |
| A08 | GPU/NPU 映射、异步流水、同步、资源占用和代价模型 | D16—18、A05—07 | [目标五篇](./compiler/targets/)展开映射、bank、slot 依赖、Host/Device 与 HIVM pipe；真实编译、逻辑模型和项目源码证据分开，硬件性能未验证 | D4：在实际硬件/后端中解释正确性和性能 |

## 工具、扩展与验证

| ID | 可检查知识项 | 前置 | 本次正文与位置 | 后续目标 / 缺口 |
|---|---|---|---|---|
| X01 | mlir-opt、generic print、诊断、verifier、pipeline、IR dump | C01—11 | [阅读与验证](./guides/inspecting_ir)、[Pass](./compiler/transforms/passes)与[调试](./guides/debugging)；dump、重放、Instrumentation、统计已有作者运行 | 调试器逐行会话和个人IDE配置按需要完成 |
| X02 | lit/FileCheck、verify-diagnostics、单测、语义/数值/性能验证 | X01、M04 | [回归](./guides/regression_tests)与[数值性能](./guides/numerics_performance)，D3；22工程lit及错误返回值反例，已有算子数值/计时证据 | D4：面向真实项目扩展边界与设备测量 |
| X03 | bytecode、版本升级、Dialect version、序列化兼容 | C01、M02 | [Bytecode](./guides/bytecode)，D3作者工程；24实验真实bias→offset、未来版本拒绝、旧文本与容器版本对照 | 真实方言的复杂Type/Properties/资源兼容按项目深化 |
| X04 | C API、Python bindings、所有权/生命周期、调试打印 | M01 | [C API](./guides/c_api)、[Python](./guides/python_bindings)，D3；23工程两接口42→7、所有权/包装失效与错误阶段均实测 | 独立工具接入与目标版本适配 |
| X05 | mlir-reduce、最小复现、统计/timing、LSP 与诊断定位 | X01 | [失败复现](./guides/debugging)，D3；22工程重放/缩减、统计、Instrumentation、LSP握手 | GDB和IDE集成给出入口，未执行完整会话；真实故障定位独立实践 |
| X06 | DataLayout、符号/shape 推导、跨目标约束、插件/动态注册 | C06、M02—03 | [LLVM布局](./dialects/llvm/)、[形状推导](./compiler/ir_definition/type_shape_inference)、[插件](./guides/plugins)与[IRDL](./dialects/extensions/irdl)已有；作者查询/生成/加载验证分别记录 | D3—4：选定目标的完整ABI与部署配置；跨版本插件兼容不承诺 |
| X07 | quant/sparse/shape/IRDL 等扩展领域 | 相应核心模块 | [领域入口](./dialects/extensions/)展开D1/D2有限例子；25工程解析往返、稀疏/形状/TOSA转换与IRDL反例；模型数值不冒充目标执行 | 采用真实领域项目时扩为完整D3/D4链 |

## 算法贯通证据

[归约与 Softmax](./tutorials/kernels/)已提供四阶段 Tensor/Linalg 表示、归约初值、完整 CPU 路径及按行 exp/sum 融合。两实现对照 double 参考覆盖 10 组形状/数据，含下溢、513 列与空形状；错误 max 初值有负例。这里验证的是数值与存储结构，设备映射、通用 fusion 与性能优化由 A05—A08 继续展开。

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
