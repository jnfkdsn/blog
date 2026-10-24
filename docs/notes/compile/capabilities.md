---
order: 90
title: 编译器能力与项目
excludeFromSidebar: true
updated: 2026-10-17
---

# 编译器能力与项目

这里保存长期能力、项目验收和有限源码阅读方法。开始一轮学习时，先用[总路线](/notes/roadmap)选择任务，再按需要回查本页。

## 学习方向与能力模块

这些方向都值得学习，但不需要同等深度、同时展开。以下阶段保留长期范围，当前执行次序优先使用已展开的课程主线：

```text
当前：MLIR基础 → TVM完整链路 + PyTorch Compiler完整链路
随后：Triton GPU 与 Triton-Ascend 分别深入kernel编译与硬件映射
按项目回访：图优化、动态shape、运行时、部署、StableHLO及其他系统

共同能力：表示与语义 → 分析/变换 → 实现选择 → 目标执行与验证
```

上图描述学习依赖，不表示这些系统构成一条实际工具链。[TVM八章](/notes/compile/tvm/)与[PyTorch Compiler八章](/notes/compile/pytorch-compiler/)以推理为主并补充训练；当前实验证据和深入边界见[课程覆盖表](/notes/compile/compiler_courses_coverage)。下面按能力模块组织，不表示固定先后顺序。

### IR 基础：统一 IR 与 Pass 基础

目标：能读懂 MLIR 风格 IR 和 Pass 代码，并能独立定义、转换和测试一个小型 Dialect/Op。

执行边界（2026-09-30）：以下内容是逐轮建立的长期范围。第一轮先能解释基础 IR，在参考工程上修改一个简单 Op 或 Pattern，并用输入输出和正反例验证，即可开始目标项目的定点阅读。第二轮随一条实际编译链补 Conversion、张量/内存和所需 lowering；第三轮随实现任务深入分析、优化与框架源码。不能把下列所有条目都当成接触 TVM、Triton 或 AscendNPU IR 之前必须完成的关卡。MLIR 的前置知识与阶段标准见 [MLIR 学习路径](/notes/compile/mlir/learning_path)。

学习内容：

- SSA、use-def、CFG、dominance。
- Operation、Region、Block。
- Type、Attribute、Dialect。
- TableGen、ODS、Trait 和 Interface。
- builder、verifier、type/shape inference、parser/printer。
- Pattern Rewrite。
- Pass Manager 和 analysis preservation。
- Dialect Conversion、legality、TypeConverter。
- 前向/后向 dataflow、liveness、alias、side effect 和 memory effect。
- loop dependence、tiling、fusion、interchange、vectorization 的合法性。
- Affine/Presburger/polyhedral 基础。
- Destination-Passing Style、One-Shot Bufferize、ownership 和 deallocation。
- LLVM IR 基础。
- CMake、TableGen target、lit 和 FileCheck。

实践：

```text
定义一个简单 tensor op
  → verifier 与 shape inference
  → canonicalization/CSE
  → conversion 到 linalg/arith
  → bufferization
  → lower 到 LLVM dialect
  → 翻译为 LLVM IR
  → lit/FileCheck 验证
```

完成标准：

- 能读 MLIR 文本。
- 能解释一个 pass 的输入输出。
- 能定义一个小型 Dialect/Op，并编写 rewrite 和 conversion。
- 能解释一个循环变换为什么合法、一次 buffer copy 为什么产生。
- 能使用 IR dump 定位是哪一步改变了程序。
- 能为正常转换、非法输入和诊断信息编写测试。

### TVM：用 TVM 跑通完整链路

本轮正文与CPU观察已展开在[TVM教程](/notes/compile/tvm/)，并安排[PyTorch对照](/notes/compile/pytorch-compiler/)。下列项目继续作为逐步深化的能力清单；完整生产BYOC后端等不因当前已演示外部调用协议就记为完成。

选择一个小模型：

```text
MatMul → Add → ReLU
```

观察：

```text
PyTorch/ONNX
  → Relax
  → graph fusion
  → TensorIR
  → schedule
  → CUDA/LLVM
  → runtime execution
```

至少完成：

- 一个 Relax graph rewrite。
- 一个 TensorIR schedule。
- 一次 fusion 前后 IR 对比。
- 一次 BYOC 示例。
- 一次 Runtime Module/VM 执行。
- 解释 schedule 对应的 loop transform，以及依赖关系为何允许该变换。
- 比较静态 shape、symbolic shape 和 dynamic shape 对编译与 runtime 的影响。
- 为 graph rewrite、schedule 和 BYOC/fallback 各保留至少一个自动化测试。

### Triton：深入 Triton 编译 Pipeline

选择三个 Kernel：

| Kernel | 重点观察 |
|---|---|
| Vector Add | 基础 load/store、mask、program mapping |
| Softmax | reduction、临时值、数值函数 |
| MatMul | layout、`tl.dot`、matrix instruction、pipeline |

对每个 Kernel 保存：

```text
source.py
TTIR
关键 pass 前后 IR
目标适配 IR
最终目标代码
correctness result
benchmark result
```

重点阅读：

- JIT/AST code generator。
- `compiler.py`。
- TTIR op 定义。
- backend stage 注册。
- load/store/reduce/dot lowering。
- `driver.py`。
- specialization、JIT cache 和 autotune cache。
- layout conversion 的产生位置和代价。
- reduction 顺序、accumulation dtype 和 fast-math 等数值语义。
- Pass legality、失败路径和 IR transformation test。

对每个 Kernel 还要回答：

- 哪些 layout、shape、memory 和并行信息在当前 IR 中显式存在？
- 哪些信息由 analysis 推导，哪些由 transformation 决定？
- 额外 copy 或 layout conversion 在哪个 pass 产生，为什么产生？
- 编译时间、首次运行时间和稳定运行时间分别是多少？

### NPU 后端：进入 Triton-Ascend 与 AscendNPU IR

按照真实目录追踪：

```text
kernel.py
  → ttir.mlir
  → compiler.py stage
  → TritonToLinalg/Structured
  → AscendNPU IR
  → bishengir-compile
  → kernel.o
  → driver.py launch
```

第一轮只追踪 Vector Add，第二轮追踪 Softmax，第三轮追踪 MatMul。不要一开始遍历所有 Dialect 和 Pass。

在源码追踪之外，至少完成一次真实修改和调试：

- 使用 `TRITON_DEBUG`、`MLIR_ENABLE_DUMP`、location 和 reproducer 保存最小复现。
- 将问题区分为编译错误、精度错误、运行时错误和性能回退。
- 使用 reference/interpreter 与 NPU 结果进行 differential test。
- 覆盖动态 shape、mask、非对齐 shape、空边界和极端数值。
- 对 pass pipeline 做阶段化定位，必要时使用断言、GDB/LLDB、ASan/UBSan 或 Git bisect。
- 修改一个真实 Pass、Conversion 或 Op，并按照上游风格补齐测试和变更说明。

### 算子优化：并行进行算子优化

建议顺序：

```text
Elementwise / Broadcast
  → Reduction
  → RMSNorm / Softmax
  → MatMul Epilogue
  → Quantized GEMM
  → Attention
```

每个算子同时记录：

- reference 和误差标准。
- dtype、accumulation dtype、shape 和边界条件。
- tiling 和并行映射。
- 理论 FLOPs、memory traffic 和 arithmetic intensity。
- 片上 buffer 和 layout。
- pipeline。
- correctness、数值范围、rounding、overflow/underflow 和 fast-math 影响。
- warmup、同步、缓存、编译时间与执行时间分离。
- P50/P95 latency、吞吐、有效带宽和 profiler 指标。
- IR 和最终编译结果。
- 优化收益的机制解释，以及精度、性能、编译时间和通用性的取舍。

### 部署对照：学习 TensorRT 作为工业对照

TVM 完整链路跑通后，再集中学习：

```text
ONNX
  → TensorRT Network
  → Builder
  → Optimization Profile
  → Tactic Selection
  → Engine
  → ExecutionContext
```

重点比较：

| TVM | TensorRT |
|---|---|
| IR 和 Pass 更开放 | 内部优化相对封闭 |
| 适合研究编译器实现 | 更偏工业部署 |
| TensorIR schedule | tactic/kernel selection |
| BYOC | TensorRT Plugin |
| Relax VM | TensorRT Runtime |

同时补充工程部署视角：

- engine compatibility、序列化和版本管理。
- build time、tactic timing cache 与 runtime latency 的边界。
- dynamic shape、Optimization Profile 切换和资源重新计算。
- plugin ABI、workspace、生命周期和错误处理。
- tactic 选择稳定性、精度退化和性能回退定位。

### 框架入口：PyTorch Compiler 与 StableHLO

这一阶段用于补齐生产框架入口和可移植高层 IR，并与已经掌握的 TVM、Triton 和 AscendNPU IR 建立对应关系。

PyTorch Compiler 至少完成：

- 观察 `torch.compile` 中 Dynamo、FX、AOTAutograd、Inductor 和 Triton/C++ codegen 的边界。
- 构造 graph break、guard failure 和 dynamic shape recompilation 示例。
- 查看一个算子的 decomposition、fusion、Inductor IR 和生成 Kernel。
- 使用 `torch.export`/AOTInductor 生成可部署产物。
- 实现一个最小自定义 backend 或 graph rewrite，并补 correctness test。

StableHLO 至少完成：

- 导出或构造一个带 dynamic shape 的 StableHLO module。
- 阅读 op semantics、verifier、type inference 和 canonicalization。
- 观察 portable artifact/VHLO 的版本处理。
- 在目标版本的 AscendNPU IR 或 torch-mlir 源码中确认 StableHLO 的真实入口与 conversion pipeline。
- 追踪一个 `dot` 或 `reduce` 到 Linalg/HFusion/目标 IR，或者明确记录尚未支持的 lowering 边界。

### 统一验收维度

测试、调试和性能方法不能只在最后补。每个阶段都使用下面五个维度验收：

| 维度 | 验收问题 |
|---|---|
| 理论 | 这一阶段解决什么问题，依赖哪些语义和不变量？ |
| 实现 | 是否能编写或修改一个 Op、Pass、Schedule、Kernel 或 Runtime 组件？ |
| 正确性 | 是否包含正常、边界、动态 shape 和错误输入测试？ |
| 调试 | 是否能构造最小复现，并定位到具体 stage/pass/runtime？ |
| 证据 | 是否保存版本、源码 commit、IR、测试结果和必要的 profiler 数据？ |

## 学习深度与方向选择

下表以 GPU/NPU 编译项目为例列出长期能力。全链路、算子编译和运行时方向可以选择不同的深入重点，不把每项都列为当前入门要求。

下表描述长期能力目标。“核心掌握并能独立实现”指能实现或修改相应组件、解释其语义与约束并测试，不要求从零重写 MLIR 的生成器、Interface 基础设施或 bufferization 框架。源码阅读围绕当前修改与判断展开，API 细节允许查阅。

| 内容 | 建议深度 |
|---|---|
| AscendNPU IR | 核心掌握 |
| Triton-Ascend Backend | 核心掌握 |
| MLIR IR/Pass/Conversion | 核心掌握 |
| MLIR ODS/Interface/Bufferization | 核心掌握并能独立实现 |
| 编译分析、循环合法性与内存语义 | 核心掌握 |
| 测试、调试和最小复现 | 核心掌握 |
| Triton Kernel 与性能优化 | 核心掌握 |
| 数值正确性与性能方法学 | 核心掌握 |
| Ascend 硬件与 CANN Runtime | 核心掌握 |
| StableHLO | 能读写 IR、理解语义并追踪一条 conversion |
| PyTorch Compiler | 能独立追踪 capture 到 codegen 的完整链路 |
| TVM Relax/TensorIR | 能独立跑通完整链路 |
| LLVM IR/Pass/Codegen | 工作级理解 |
| TileLang | 能实现和分析主要 Kernel |
| TensorRT | 理解架构、API、Plugin 与性能机制 |
| LLVM Target Backend 内部 | 岗位需要时再深入 |

学习资源的角色可以压缩成：

```text
MLIR + AscendNPU IR
              主线

StableHLO / PyTorch Compiler
              模型入口

Triton + Triton-Ascend
              Kernel 入口

Triton Kernel + Ascend 硬件性能
              实践线

TVM
              完整链路参考

LLVM
              低层支撑

TensorRT
              工业部署参照

TileLang
              Kernel DSL 横向对照
```

## 综合项目：单算子纵向编译追踪与功能实现

相比继续增加互不相连的概念笔记，更值得完成的是一个可执行、可观察、可修改的纵向项目。

建议选择 Softmax 或 RMSNorm。第一版从可直接运行的 Kernel 入口追踪，以框架实现作为数值 reference；模型图入口作为后续接入实验单独验证。

```text
手写 Triton Kernel
   ↓
TTIR
   ↓
Linalg / Ascend Adapter IR
   ↓
AscendNPU IR
   ↓
NPU executable
   ↓
CANN runtime
   ↓
NPU 输出

数值验证：相同输入分别交给 PyTorch reference 和 NPU Kernel
          → 输出对比 → 正确性结论
性能验证：同设备、同输入条件下测量基线与目标 Kernel
          → benchmark / profiler → 性能与机制分析

后续：选择一条具体模型接入路径
框架程序 → 对应 graph IR → 明确的 backend / lowering / 算子注册
         → 生成或调用上述 Kernel → 框架运行
```

`FX`、`StableHLO`、`Relax` 是不同实验入口。记录各自使用的转换组件、支持范围和失败边界；只有实际验证某个 backend 自动生成了 Kernel，才将该段称为自动编译链。通过自定义算子显式调用手写 Kernel 时，应标为框架集成。StableHLO 到 Linalg/AscendNPU IR 的模型路径也可能独立于 Triton Kernel 路径。

项目文档建议拆成以下部分，其中 `01`、`02` 在模型接入实验时补充：

```text
01_model_graph.md
02_graph_optimization.md
03_triton_kernel.md
04_ttir.md
05_ascend_lowering.md
06_ascend_npu_ir.md
07_runtime.md
08_correctness_and_performance.md
09_compiler_change.md
10_tests_and_debugging.md
```

项目按三个层次推进：

```text
A. 观察
第一版追踪 Kernel 的 TTIR、目标 IR、binary 和 runtime
模型接入实验再补实际 Graph、backend 和转换边界

B. 修改
实现一个自定义 Op、fusion pattern、analysis 或 lowering

C. 交付
补齐 verifier、diagnostic、自动化测试、benchmark、版本和设计说明
```

最终完成标准：

- 能画出真实编译阶段，而不只是概念图。
- 每个阶段都有实际 IR dump 或编译产物。
- 能指出每个关键 pass 的输入、输出和作用。
- 能区分 graph fusion 与 kernel 内 fusion。
- 能解释一个 Triton op 如何映射到 AscendNPU IR。
- 对实际完成的模型接入路径，能解释高层入口与 Kernel/目标编译链如何连接；未实现或未验证的接入明确记录，不作为第一版 Kernel 追踪的前置验收项。
- 能区分 Kernel 算法问题、编译器 lowering 问题和 runtime 问题。
- 能用 correctness、performance、mechanism 三类证据说明结论。
- 能独立定义或修改一个 Op/Pass/Conversion，并说明语义、不变量和失败条件。
- 能为正常路径、边界输入、动态 shape、非法 IR 和诊断信息编写测试。
- 能按上游 PR 的标准给出问题背景、方案、测试、性能结果、取舍和已知限制。

## 阅读源码的方法

不要从仓库第一行开始顺序阅读。使用“一个程序 + 一条数据流”的方法：

```text
找到入口
  → 打印输入 IR
  → 找 stage/pass 注册点
  → 打印 pass 后 IR
  → 找生成目标 op 的 rewrite
  → 找消费目标 op 的 lowering
  → 跟踪到 binary
  → 跟踪 runtime launch
```

每阅读一个 pass，填写下面的模板：

```text
Pass 名称：
运行在哪种 IR 上：
输入中的关键 op/type：
输出中的关键 op/type：
依赖的 analysis：
修改的语义/布局/存储信息：
必须维护的不变量：
失败或 fallback 条件：
诊断和最小复现方法：
对应测试：
性能或编译时间影响：
```

每阅读一个系统，填写下面的模板：

```text
系统输入：
高层 IR：
Kernel IR：
主要优化层：
Library/Kernel 选择边界：
Target lowering：
最终产物：
Runtime：
动态 shape 策略：
调试和 IR dump 方法：
测试体系：
版本兼容与部署边界：
```

