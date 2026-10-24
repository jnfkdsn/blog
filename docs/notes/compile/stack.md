---
order: 5
title: AI 编译器技术全景
updated: 2026-10-17
---

# AI 编译器技术全景

本页按表示、变换、代码生成和运行时解释各系统的职责，供学习时定位与对照。先选阅读方向见[知识库学习路线](/notes/roadmap)，逐步深入的能力清单见[编译器能力与项目](/notes/compile/capabilities)。

## 一、统一的 AI Compiler Pipeline

下面是按职责组织的概念链。具体系统可能跳过、合并或替换其中一些阶段；已有库调用和新 Kernel 生成在运行时执行层汇合。

```text
PyTorch / JAX / ONNX 模型
  → 1. Capture / Import
  → 2. High-level Graph IR
  → 3. Graph Optimization
       decomposition / fusion / layout / partition
  → 4. Kernel Selection / Generation
       │
       ├── 已有库 / 预置 Kernel
       │     cuBLAS / CANN 等
       │     → 调用代码、参数适配和库依赖 → 交给步骤 8
       │
       └── 新 Kernel 生成
             → 5. Tensor / Tile / Loop IR
             → 6. Hardware Scheduling
                  tiling / layout / memory / pipeline
             → 7. Target Lowering / Codegen
                  目标 IR、指令或源码 → binary → 交给步骤 8

8. Runtime Execution（执行上述库调用与生成的 Kernel）
   load / memory / stream / launch / cache
   → NPU / GPU
```

已有库分支仍可能需要 lowering 调用 op、生成 host/device 胶水代码和链接依赖，但通常不会重新编译库内部 Kernel。调度与 lowering 在实际系统中可能交错进行；图中的阶段号用于区分职责，不要求每个阶段只出现一次。

理解任何编译系统时，都先回答五个问题：

1. 输入是什么？
2. 核心 IR 是什么？
3. 在哪一层做什么优化？
4. 输出是什么？
5. 谁负责加载和运行输出？

这五个问题可以把一个庞大的代码仓库还原成若干清晰的编译阶段。

## 二、每一层解决什么问题

### 1. Capture / Import

输入可能是 Python 函数、框架计算图、ONNX 或其他模型格式。目标是把原始程序转换成稳定、可分析的表示。

这一层需要处理：

- Python 动态语义和 graph break。
- 模型参数与常量。
- control flow。
- operator schema。
- shape、dtype 和 alias 的初始信息。

典型实现包括 PyTorch Dynamo、`torch.export`、ONNX parser 和 TVM frontend。

### 2. High-level Graph IR

Graph IR 中的节点通常仍然是高层 tensor op：

```text
x, w, b
  → MatMul(x, w)
  → Add(..., b)
  → Relu(...)
```

这一层保留算子语义，适合进行：

- shape/dtype/layout inference。
- decomposition。
- constant folding。
- DCE、CSE。
- graph fusion。
- device/backend partition。
- quantization rewrite。

详细概念见 [Graph IR 基础](/notes/compile/ai-basics/graph_ir_basics) 和 [Graph Rewrite 与 Pass 基础](/notes/compile/ai-basics/graph_rewrite_pass)。

### 3. Kernel Selection / Generation

图中的一个 op 或融合子图通常有两种实现方式：

```text
调用已有高性能实现
  → cuBLAS / cuDNN / CANN 算子库 / TensorRT tactic

生成新的 Kernel
  → Triton / TileLang / TensorIR / Ascend C
```

选择已有库通常更稳定；生成 Kernel 则更容易支持新的融合、动态 shape 和特殊数据布局。

编译器需要同时考虑：

- 后端是否支持。
- 厂商库是否存在高性能实现。
- fusion 是否会破坏高性能 library call。
- 生成 Kernel 的性能和编译开销。
- 不支持场景如何 fallback。

### 4. Tensor / Tile / Loop IR

进入这一层后，程序不再只描述“计算什么”，还开始描述“怎样组织计算”。

```text
for bm in tiles(M):
  for bn in tiles(N):
    acc = 0
    for bk in tiles(K):
      A_tile = load(A)
      B_tile = load(B)
      acc += dot(A_tile, B_tile)
    store(C, acc)
```

这一层关注：

- loop nest。
- tile/block。
- buffer 和访问模式。
- reduction。
- memory scope。
- parallel axis。
- tensor/matrix intrinsic。

TVM TensorIR、Triton TTIR/目标相关 IR、TileLang IR 和部分 AscendNPU IR 都位于这一带。

### 5. Hardware Scheduling

Scheduling 决定 tensor program 如何映射到具体硬件：

- tile 大小。
- 核、线程、warp 或向量单元映射。
- shared/UB/L1/L0/register 分配。
- 数据布局和 swizzle。
- vectorization。
- double buffering。
- async copy 和软件流水。
- matrix instruction 选择。

同一个数学算子可以有很多 schedule。算法正确不代表 schedule 高效。

### 6. Target Lowering / Codegen

Lowering 将高层 op 逐步转换成目标可以执行的低层操作：

```text
tensor op
  → tiled loop
  → vector/matrix op
  → target intrinsic
  → LLVM IR / 厂商 IR
  → object / binary
```

需要处理：

- op legality。
- dtype conversion。
- address calculation。
- calling convention。
- target intrinsic。
- synchronization。
- kernel ABI。

### 7. Runtime

编译完成不等于可以运行。Runtime 负责：

- 加载 binary/engine。
- device memory 和 workspace。
- stream/event。
- kernel launch。
- dynamic shape guard。
- JIT cache。
- executable 生命周期。
- fallback 和错误处理。

详细概念见 [Lowering 与 Runtime 基础](/notes/compile/ai-basics/lowering_runtime)。

## 三、技术定位总表

| 技术 | 定位 | 主要覆盖层次 | 不应把它理解成什么 |
|---|---|---|---|
| LLVM | 低层编译基础设施 | Target lowering、优化、机器码生成 | AI 图编译器 |
| MLIR | 多层 IR、Dialect、Pass 与 Conversion 基础设施 | Graph IR 到 Target IR 都可承载 | 固定的一套 AI 编译产品 |
| StableHLO | 可移植的高层 ML 算子集和模型编译接口 | Framework import、High-level tensor semantics | Kernel IR 或硬件指令 IR |
| PyTorch Compiler | PyTorch 程序捕获、分解、图优化和后端代码生成体系 | Dynamo、FX、AOTAutograd、Inductor、AOTInductor | 单一后端或单一 IR |
| TVM | 端到端 AI 编译系统 | Graph、TensorIR、Schedule、Codegen、Runtime | 只有算子调优的工具 |
| TensorRT | NVIDIA 推理编译与运行系统 | 图优化、tactic 选择、engine、runtime | 通用且完全开放的编译器教学框架 |
| Triton | Kernel DSL + Kernel Compiler | Kernel 表示、调度、lowering、JIT runtime | 完整模型前端和图编译系统 |
| TileLang | 显式 Tile 级 Kernel DSL | Tile、buffer、pipeline、target codegen | TVM 完整编译链本身 |
| AscendNPU IR | 昇腾目标相关 IR/编译层 | Tensor/Kernel IR 到目标代码生成 | 上层模型 Graph IR |
| CANN/GE | 昇腾编译、算子库与运行时生态 | Graph、算子、执行、runtime | 单一 IR 或单一编译器 |
| Ascend C | 昇腾底层算子编程接口 | Kernel、tiling、片上流水、指令调用 | 图编译系统 |

## 四、LLVM 与 MLIR

### LLVM：低层优化与代码生成

LLVM 更接近传统编译器后端：

```text
LLVM IR
  → scalar/CFG optimization
  → instruction selection
  → register allocation
  → machine code
```

LLVM IR 主要表达：

- SSA value。
- 标量和向量运算。
- load/store。
- basic block、branch、phi。
- function 和 calling convention。
- target triple 和 data layout。

程序 lower 到 LLVM IR 后，通常已经看不到“这是 Softmax”或“这是 Attention”。高层 tensor 语义已经变成循环、访存和低层计算。

AI Compiler 岗位第一阶段需要掌握：

- LLVM IR 基本语法。
- SSA、CFG、dominance、phi。
- `alloca/load/store/getelementptr`。
- Pass Manager。
- Target、DataLayout、ABI。
- LLVM IR 到目标代码的基本路径。

暂时不必深入完整的寄存器分配、SelectionDAG/GlobalISel 或 CPU Target Backend，除非岗位明确要求最底层指令选择。

### MLIR：组织多层抽象和渐进式 Lowering

AI 程序不能一开始就丢进 LLVM IR，否则会过早丢失 tensor、layout、memory space 等信息。MLIR 允许不同抽象层通过 Dialect 共存：

```text
高层 Tensor Dialect
        ↓
Linalg / Structured Dialect
        ↓
SCF / Affine / Vector / GPU
        ↓
LLVM Dialect / Target Dialect
        ↓
LLVM IR 或厂商 IR
```

MLIR 的核心对象：

```text
Operation
Region
Block
SSA Value
Type
Attribute
Dialect
Interface
Rewrite Pattern
Pass
Dialect Conversion
Type Converter
Conversion Target
```

Dialect Conversion 的基本思想：

```text
定义目标中哪些 op/type 合法
  → 为不合法的源 op 提供 rewrite pattern
  → 必要时转换 type 和 region signature
  → 直到 IR 对目标完全合法
```

例如：

```text
tt.dot
  → linalg.matmul
  → vector/matrix intrinsic
  → AscendNPU op
```

因此二者的关系是：

- MLIR 保留并逐步转换多层语义。
- LLVM 接管已经足够低层的程序，完成低层优化和目标代码生成。
- 一个 MLIR 编译器不一定必须经过 LLVM，也可以 lower 到厂商专用 IR/编译器。

### MLIR 编译器组件开发

能阅读 MLIR 文本只是起点。实际开发还需要能够定义和扩展编译器组件：

- 使用 TableGen/ODS 定义 Dialect、Operation、Type 和 Attribute。
- 使用 Trait 表达 op 的固有性质，使用 Interface 提供跨 Dialect 的统一行为。
- 编写 builder、verifier、type/shape inference、canonicalization 和 folding。
- 理解 declarative assembly format，必要时编写 parser/printer。
- 注册 Dialect、Pass、Pipeline 和 external model。
- 使用 CMake 和 TableGen target 组织可独立构建的 MLIR 工程。

一个完整的练习不应只包含 rewrite，而应覆盖：

```text
定义一个 tensor op
  → verifier 与 shape inference
  → canonicalization
  → conversion 到 linalg/arith
  → bufferization
  → LLVM dialect / 目标 Dialect
  → lit/FileCheck 测试
```

### 编译分析、循环变换与内存语义

Pass 是否正确不能只靠 pattern matching。需要补齐支撑优化合法性的分析基础：

- 前向和后向 dataflow analysis。
- liveness、dominance、post-dominance。
- alias analysis、side-effect 和 memory-effect modeling。
- loop dependence 与 fusion、interchange、tiling、skewing、vectorization 的合法性。
- Affine/Presburger/polyhedral model 的基本思想。
- symbolic shape、shape constraint 和 runtime guard。
- tensor 与 memref 的语义边界。
- Destination-Passing Style、One-Shot Bufferize 和 in-place analysis。
- buffer alias/equivalence、ownership、deallocation 和 memory planning。

学习这些内容时，不要求先实现通用 polyhedral optimizer，但要能解释一个循环变换为什么合法、一次 buffer copy 为什么产生，以及某个 value 的生命周期如何影响内存复用。

## 五、TVM：理解完整纵向链路

TVM 适合用于建立端到端 AI Compiler 的整体认识：

```text
PyTorch / ONNX
      ↓ frontend/import
Relax IR
      ↓ graph passes
optimized Relax
      ↓ LegalizeOps / FuseOps
TensorIR PrimFunc
      ↓ schedule
scheduled TensorIR
      ↓ target lowering
LLVM / CUDA / external backend
      ↓
Runtime Module / Relax VM
```

两个核心层次：

- `Relax`：模型或子图级表示，负责 graph optimization。
- `TensorIR`：单算子或融合 Kernel 的 loop、buffer、thread 和 memory 表示。

通过 TVM 应重点学习：

- 模型导入和 IRModule。
- Relax Graph IR。
- `LegalizeOps`。
- `FuseOps/FuseTIR`。
- Relax 到 TensorIR 的边界。
- TensorIR schedule。
- tiling、reorder、bind、cache read/write。
- target codegen。
- VM/runtime。
- BYOC：外部 NPU backend 如何接入。

BYOC 的基本链路：

```text
注册后端支持的 pattern
        ↓
从大图中 partition 子图
        ↓
把子图交给外部 backend codegen
        ↓
runtime 调用外部 backend
```

学习 TVM 的目标不是记住所有 API，而是亲自观察一个模型从 Relax 到 TensorIR 再到 executable 的全过程。

## 六、TensorRT：工业推理编译与部署

TensorRT 的核心链路是：

```text
ONNX / NetworkDefinition
        ↓
Graph optimization
        ↓
Layer fusion / precision selection
        ↓
为 layer/subgraph 选择 tactic
        ↓
序列化 TensorRT Engine
        ↓
ExecutionContext 运行
```

它包含两个明显阶段：

- Build：优化网络、选择 kernel/tactic、生成 engine。
- Runtime：反序列化 engine、管理执行上下文并运行。

学习重点：

- ONNX Parser。
- NetworkDefinition。
- BuilderConfig。
- Optimization Profile 和 dynamic shape。
- FP16/INT8/FP8 精度策略。
- tactic selection。
- engine 和 ExecutionContext。
- workspace、stream、memory。
- plugin：不支持算子如何接入。
- TensorRT-LLM 如何组合模型结构、算子库和运行时。

TensorRT 很适合观察工业推理系统如何在图优化、Kernel 库、自动调优和 runtime 之间取舍，但很多内部优化并不开源，不适合用来作为第一个源码级编译器框架。

## 七、Triton：Kernel Compiler 主线

Triton 的输入通常不是完整模型，而是一个 Kernel：

```python
@triton.jit
def kernel(...):
    ...
```

通用路径可以理解为：

```text
Python AST
   ↓
TTIR
   ↓
Target-aware Triton IR
   ↓
GPU/Target Dialect
   ↓
LLVM IR 或厂商 IR
   ↓
二进制
   ↓
Triton Driver / JIT Runtime
```

不同 backend 的中间路径不一定相同。学习 Triton 时要同时保留两个层次。

### Triton 使用层

- program/block 编程模型。
- `tl.arange/load/store`。
- mask 和边界处理。
- reduction。
- `tl.dot`。
- block pointer。
- autotune。
- `num_warps/num_stages`。
- Softmax、GEMM、Attention Kernel。
- profiler 和生成代码检查。

### Triton 编译器层

- AST 如何生成 TTIR。
- TTIR 中的 op、type 和 attribute。
- Pass Pipeline 如何组织。
- layout encoding 表达什么。
- reduction、dot、load/store 如何 lowering。
- memory allocation、pipeline、vectorization。
- backend `compiler.py`。
- runtime `driver.py`。
- MLIR pass 和 dialect conversion。
- IR 单元测试和 FileCheck。

这两层必须相互连接：一个性能问题可能来自 Kernel 算法，也可能来自 layout 推导、lowering、额外 copy 或目标指令选择。

## 八、TileLang：显式 Tile 级对照

TileLang 同样属于 Kernel DSL，但会更显式地暴露：

- tile。
- shared/local/fragment memory。
- copy。
- pipeline。
- GEMM primitive。
- layout 和硬件映射。

典型路径：

```text
TileLang Python DSL
      ↓
TVM IRModule / TensorIR
      ↓
TileLang passes
      ↓
CUDA / HIP / LLVM / 厂商后端
      ↓
JIT Runtime
```

Triton 与 TileLang 的对照：

| Triton | TileLang |
|---|---|
| block program 抽象 | tile/buffer 抽象 |
| 编译器自动推导较多 | memory/pipeline 控制更显式 |
| `tl.load/tl.dot` | `T.copy/T.gemm/T.Pipelined` |
| layout 编译器色彩较强 | 数据搬运和存储位置表达较直接 |
| PyTorch/Inductor 生态联系紧密 | TVM/TensorIR 联系紧密 |

建议用相同算子横向比较，而不是分别建立两套互不相干的知识：

```text
同一个 Softmax
  → CUDA
  → Triton
  → TileLang
  → Ascend C
```

比较每种抽象暴露了什么、隐藏了什么，以及编译器替程序员完成了什么。

## 九、AscendNPU IR 与 Triton-Ascend 主线

根据 Triton-Ascend 的公开架构，核心链路可以概括为：

```text
Triton Python Kernel
        ↓
TTIR
        ↓ Triton-Ascend backend
Linalg IR / Ascend Adapter IR
        ↓
AscendNPU IR
        ↓ BiSheng Compiler
NPU Kernel Object
        ↓ driver.py
CANN / TorchNPU Runtime
        ↓
Ascend NPU
```

对应的核心代码区域：

- `compiler.py`：注册编译选项并组织各个编译 stage。
- `driver.py`：连接 CANN/TorchNPU runtime，加载并启动 kernel。
- Ascend backend `include/`、`lib/`：目标相关 Dialect、Pass 和 Conversion。
- `AscendNPU-IR/`：继续向昇腾硬件代码生成 lowering。
- tutorials/unit tests：算子样例、迁移和 IR 转换测试。

### 区分 Graph/Operator IR 与 AscendNPU IR

```text
Ascend Graph / Operator IR
  → 模型或算子图级别
  → 节点仍然是 MatMul、Softmax、Conv 等

AscendNPU IR
  → Kernel/目标相关级别
  → 包含更具体的计算、存储、搬运和硬件信息
```

阅读一种 AscendNPU IR op 时，逐项记录：

1. 输入输出是什么 type？
2. 操作对象是 tensor 还是 buffer？
3. 是否已经包含 layout？
4. 是否已经绑定 AIC/AIV 或其他并行单元？
5. 是否表达 GM、UB、L1、L0 等 memory space？
6. DMA、barrier、pipeline 是否显式？
7. 哪个 pass 生成它？
8. 哪个 pass 消费它？
9. 下一层会 lower 成什么？
10. verifier 检查哪些不变量？

相关硬件和算子编程背景见 [Ascend C 算子编程知识库](/notes/kernels/ascend-c/) 和 [NPU Lowering 约束](/notes/compile/npu-ir/lowering_constraints)。

## 十、完整关系图

下面按系统分别画出典型路径。相同数学算子可以用于跨系统比较；系统之间的接入需要明确的 importer、backend 或 conversion，不能由图中位置相邻推导出来。

```text
TVM 模型编译路径
模型 → 对应 frontend → Relax / TensorIR IRModule
     → graph passes / TensorIR schedule / target lowering
     → LLVM、CUDA 等 codegen 或外部 backend
     → Runtime Module / Relax VM → 对应设备

PyTorch Compiler 路径（以 Inductor 为例）
PyTorch → Dynamo / FX → functionalization / decomposition 等
        → Inductor → Triton / C++ / 库调用 → 对应运行环境
torch.export → ExportedProgram → 支持该表示的部署 backend

Triton GPU Kernel 路径
Triton Kernel → TTIR → GPU backend 的 IR / passes / codegen
              → GPU binary → Triton GPU driver/runtime → GPU

Triton-Ascend Kernel 路径（典型 Linalg 路径）
Triton Kernel → TTIR → Linalg / Ascend Adapter IR
              → AscendNPU IR 编译链 → BiSheng / NPU object
              → Ascend driver / CANN runtime → NPU

TileLang Kernel 路径
TileLang Kernel → TVM IRModule / TensorIR → TileLang passes
                → 所选目标的 codegen → JIT runtime → 对应设备

TensorRT 部署路径
ONNX / NetworkDefinition → graph optimization / tactic selection
                        → Engine → TensorRT runtime → NVIDIA GPU

高层模型 IR 接入目标编译器（待按版本验证的接入边界）
StableHLO / Torch dialect / 其他高层 IR
  → 明确命名且支持相应 op 的 conversion / backend
  → 该 backend 接受的 IR → 后续目标编译链
```

阅读这些路径时注意：

- TVM 和 TileLang 的常规路径不以 MLIR 或 AscendNPU IR 为必经阶段；接入特定后端需要相应实现。
- MLIR 是组织多层 IR 和转换的基础设施，可以贯穿 Triton、AscendNPU IR 等编译链，不能只画成几个系统之后的共同单一阶段。
- AscendNPU IR 是包含多个 Dialect 和 Pass 的目标编译体系。HFusion、HIVM 等表示属于其中不同层次，实际经过哪些表示由所选 pipeline 决定。
- PyTorch 的 GPU 编译路径不能自动视为 NPU 编译路径。连接 Triton-Ascend 需要确认目标版本的框架 backend、算子注册或显式调用方式。
- 本图是架构说明，不代表本 workspace 已运行验证。实验中的每条接入边界应记录源码 commit、入口命令、转换组件和实验编号；未验证的边界保留为待验证。

路径依据：[TVM 架构](https://tvm.apache.org/docs/arch/index.html)、[Triton-Ascend 架构](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/architecture_design_and_core_features.md)、[TileLang 概览](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)。源码追踪时使用与实验版本一致的文档和源码。

## 十一、PyTorch Compiler 与 StableHLO

### PyTorch Compiler：生产框架编译入口

TVM 适合观察开放的端到端编译系统，PyTorch Compiler 则用于理解生产框架如何把动态 Python 程序连接到 Kernel Compiler：

```text
PyTorch Program
      ↓ TorchDynamo / torch.export
FX Graph / ExportedProgram
      ↓ AOTAutograd / decomposition
Functional Graph
      ↓ TorchInductor
Inductor IR / scheduler
      ↓
Triton / C++ / external backend
      ↓
JIT cache 或 AOT package
```

学习重点：

- Python bytecode capture、graph break 和 fallback。
- FX Graph、FakeTensor、SymInt 和 symbolic shape。
- guard、specialization、recompilation 和 compile cache。
- decomposition、functionalization 和 AOTAutograd 的职责。
- Inductor IR、fusion、scheduler 和 Triton/C++ codegen。
- `torch.export` 与 AOTInductor 的部署路径。
- 自定义 backend 如何接收 graph、返回 callable，并处理错误和不支持算子。

### StableHLO：模型编译与 AscendNPU IR 的高层入口

StableHLO 是带有稳定语义和兼容性约束的高层 ML 算子集，适合充当框架与编译器之间的可移植接口。它保留 `dot`、`convolution`、`reduce`、`broadcast` 等模型级 tensor 语义，本身不是 Kernel IR，也不描述 Ascend 的 UB、L1、Cube/Vector pipeline 等硬件细节。

对于 AscendNPU IR 方向，StableHLO 有必要学习，但要区分候选模型接入边界和 Triton Kernel 路径：

```text
模型/框架入口（需逐段确认导出器、conversion 和目标支持）
PyTorch / JAX / TensorFlow
        ↓ 对应导出器 / importer
StableHLO / Torch MLIR / 其他高层 Dialect
        ↓ 目标版本提供的 decomposition / legalization / partition
目标 backend 接受的结构化 IR
        ↓ 若已实现相应接入
AscendNPU IR 编译链（具体 Dialect 和 Pass 由 pipeline 决定）

Kernel DSL 入口
Triton Kernel
        ↓
TTIR
        ↓ Triton-Ascend lowering
Linalg / Ascend Adapter IR
        ↓
AscendNPU IR
```

StableHLO 因而属于 AscendNPU IR 生态的高层模型入口知识，但不位于 Triton-Ascend 的 TTIR Kernel 主路径中。具体项目是否直接提供 `StableHLO → AscendNPU IR` conversion、是否先经过 Torch/Linalg/HFusion，以及支持哪些 op，必须以所使用版本的源码和 Pass Pipeline 为准。

需要掌握：

- StableHLO program、op、type、attribute 和 verifier。
- broadcasting、reduction、dot/convolution、control flow 和 token 的语义。
- static、dynamic 和 bounded dynamic shape。
- decomposition、canonicalization 和 legalization。
- StableHLO 与 Linalg/Tensor/目标 Dialect 的转换边界。
- portable artifact、版本兼容与 VHLO 的基本作用。
- 不支持 op、custom call 和 fallback 如何处理。

实践至少完成：

- 导出或构造一个 `MatMul → Add → ReLU` StableHLO module。
- 使用 `stablehlo-opt` 观察 canonicalization、refinement 或 legalization 前后 IR。
- 选择一个目标版本的 AscendNPU IR/torch-mlir 源码，确认 StableHLO 的实际入口、Pass 注册点和下一层 Dialect。
- 追踪一个 `stablehlo.dot` 或 `stablehlo.reduce` 最终如何进入 Linalg/HFusion/目标相关 IR；若该版本不支持，记录失败边界和所需 conversion。

## 参考资料

- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [MLIR Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)
- [MLIR Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)
- [MLIR LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/)
- [StableHLO Specification](https://openxla.org/stablehlo/spec)
- [StableHLO Compatibility](https://openxla.org/stablehlo/compatibility)
- [PyTorch Dynamo Deep-Dive](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html)
- [PyTorch AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
- [Apache TVM Design and Architecture](https://tvm.apache.org/docs/arch/index.html)
- [Apache TVM Relax VM](https://tvm.apache.org/docs/arch/relax_vm.html)
- [Apache TVM Bring Your Own Codegen](https://tvm.apache.org/docs/how_to/tutorials/bring_your_own_codegen.html)
- [NVIDIA TensorRT: How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)
- [Triton Compiler Repository](https://github.com/triton-lang/triton)
- [Triton-Ascend Architecture](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/architecture_design_and_core_features.md)
- [Triton-Ascend Debugging Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/debugging.md)
- [AscendNPU IR Repository](https://github.com/Ascend/AscendNPU-IR)
- [Ascend torch-mlir Repository](https://gitcode.com/Ascend/torch-mlir)
- [TileLang Overview](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/overview.md)
