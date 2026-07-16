---
order: 5
title: Lowering、Codegen、Runtime
updated: 2026-07-16
tags: [compiler, lowering, codegen, runtime, jit]
status: draft
---

# Lowering、Codegen、Runtime

相关入口：[传统编译器](/notes/compile/traditional/) / [IR、CFG、SSA](/notes/compile/traditional/ir_ssa_cfg)

前端和中端解决“程序是什么意思”和“程序能怎么改写”。后端和 runtime 解决“程序怎么在目标机器上执行”。

```text
high-level IR
  -> lowering
  -> low-level IR
  -> instruction selection
  -> register allocation
  -> assembly / object / bytecode / kernel source
  -> runtime execution
```

## Lowering

Lowering 把高层语义逐步变成低层语义。

高层 IR：

```text
%z = tensor.add %x, %y : tensor<1024xf32>
```

Lowering 到 loop IR：

```text
for i in 0..1024:
  z[i] = x[i] + y[i]
```

继续 lowering 到标量 load/store：

```text
ptr_x = base_x + i * 4
ptr_y = base_y + i * 4
ptr_z = base_z + i * 4
v0 = load ptr_x
v1 = load ptr_y
v2 = fadd v0, v1
store v2, ptr_z
```

Lowering 的特点：

- 每下降一层，语义更接近目标机器。
- 高层信息会逐步丢失，例如 tensor shape 可能变成 loop bound。
- 一些优化必须在高层做，例如 op fusion、layout rewrite。
- 一些优化必须在低层做，例如 register allocation、instruction scheduling。

## Instruction Selection

Instruction selection 把 IR op 映射到目标机器指令。

IR：

```text
%2 = add %0, %1
```

目标机器可能有多种实现方式：

```text
ADD r2, r0, r1
LEA r2, [r0 + r1]
FADD v2, v0, v1
```

选择哪条指令取决于：

- 操作类型：integer、float、vector。
- operand 是否在寄存器、内存、立即数。
- 目标 ISA 支持什么寻址模式。
- 指令 latency、throughput、code size。

复杂后端常用 pattern matching：

```text
(add x, const) -> ADDri
(mul x, 2) -> SHL x, 1
(add (mul x, scale), base) -> LEA base, x, scale
```

AI Compiler 的 kernel lowering 也有类似选择：

- `matmul` 下沉到 library call、Triton kernel、CUDA kernel、Ascend C kernel。
- `elementwise` 下沉到 fused loop、vector instruction、SIMT kernel。
- `reduce` 根据 shape 选择 block reduce、warp reduce、multi-stage reduce。

## 案例：MatMul Lowering 到 Tensor Core

Tensor Core 是观察 AI Compiler lowering 的一个完整案例。高层只有一个矩阵乘语义：

```text
%C = matmul %A, %B
    : tensor<MxKxf16>, tensor<KxNxf16> -> tensor<MxNxf32>
```

目标机器却不能直接执行任意大小的 `matmul`。编译器需要逐步补全并行、布局、存储和指令信息：

```text
high-level matmul / tl.dot
  -> choose library call or generated kernel
  -> tile M/N/K into CTA tiles
  -> distribute CTA tile to warps
  -> choose shared-memory layouts for A/B
  -> insert global-to-shared copies and pipeline stages
  -> map warp tiles to MMA instruction tiles
  -> lower to PTX mma.sync / ldmatrix / cp.async
  -> assemble to SASS HMMA and memory instructions
```

### 第一步：决定实现路径

同一个 `matmul` 可以：

```text
call cuBLAS / vendor library
generate Triton kernel
generate CUDA/CUTLASS kernel
fallback to SIMT scalar/vector FMA
```

决策依赖：

- target architecture 是否有匹配的数据类型和 MMA shape。
- M/N/K、batch、transpose 和 layout。
- alignment、stride、dynamic shape 和边界比例。
- library call 开销与生成 kernel 的预期收益。
- 后续是否需要融合 bias、activation、dequantize 等 epilogue。

这一步更接近 target selection / dispatch，不是最终的机器指令选择。

### 第二步：Tiling 与 Parallel Mapping

假设选择生成 kernel，编译器需要把大矩阵分解为：

```text
problem tile
  -> CTA tile
      -> warp tile
          -> MMA instruction tile
```

伪 IR：

```text
for bm in 0..M step BM parallel=block_y:
  for bn in 0..N step BN parallel=block_x:
    acc[BM, BN] = 0
    for bk in 0..K step BK:
      a_tile = load A[bm:bm+BM, bk:bk+BK]
      b_tile = load B[bk:bk+BK, bn:bn+BN]
      acc += dot(a_tile, b_tile)
```

接着还要把 `acc` 的一部分分给 warp，并把 warp tile 拆成目标支持的 MMA tile。这里已经不只是普通 loop lowering，还包括 schedule 和 layout 决策。

### 第三步：Memory Promotion 与 Pipeline

A/B tile 通常从 global memory 提升到 shared memory：

```text
global A/B
  -> shared-memory stage
  -> warp fragment registers
  -> accumulator registers
```

Ampere 上可以把同步 copy 继续 lowering 为 `cp.async`，并插入 commit/wait/barrier，形成多级 software pipeline。此时编译器必须维护：

- producer/consumer 的先后关系。
- stage buffer 的生命周期和复用。
- copy 对齐、predicate 和边界填 0。
- shared-memory layout 与 bank conflict 约束。
- stage 数量带来的 shared memory 成本。

因此 async pipeline 不是把 `load` 文本替换成 `cp.async`，而是一次依赖和调度变换。

### 第四步：MMA Instruction Selection

在 Ampere FP16 输入、FP32 累加的场景，warp-level dot 最终可能选择类似：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

要合法选择该指令，前面的 IR 必须已经明确：

- A/B/C/D 的数据类型。
- instruction tile 的 M/N/K。
- A/B 的 row/col layout。
- operand 如何分布到 warp lanes 和 registers。
- 所有参与 lane 是否在一致控制流中。

如果 shape、layout 或 target 不满足约束，就需要 padding、转换为其他 MMA shape，或者回退到 SIMT 实现。

### 第五步：Epilogue 与 Store

MMA 的 accumulator 分散在各 lane 的寄存器中。Epilogue 需要：

```text
reorder accumulator fragments
  -> alpha * acc + beta * C
  -> optional bias / activation / scale / cast
  -> coalesced global store
```

Epilogue fusion 可以减少中间 tensor 和额外 kernel launch，但也可能增加寄存器压力。编译器需要在融合收益与 occupancy、code size 之间权衡。

### 这个案例连接了哪些编译器概念

| 编译器概念 | Tensor Core lowering 中的实例 |
|---|---|
| legality | dtype、shape、layout、alignment 是否满足 MMA 约束 |
| profitability | Tensor Core 收益能否覆盖 padding、转换和小 shape 开销 |
| tiling | problem -> CTA -> warp -> instruction |
| memory planning | shared-memory stage 和 accumulator 生命周期 |
| instruction selection | warp dot -> `mma.sync` |
| register allocation | fragment/accumulator 分配，spill 与 occupancy |
| scheduling | async copy、fragment load 与 MMA 重叠 |
| runtime dispatch | 根据 shape/dtype/device 选择已编译 kernel 或 library |

对应学习材料：

- [低精度数值与混合精度计算](/notes/cuda/low_precision)
- [Tensor Core 编程](/notes/cuda/tensor_core)
- [Ampere 异步拷贝与软件流水线](/notes/cuda/async_pipeline)
- [Tensor Core GEMM 实践](/posts/tensor_core_gemm)

## Register Allocation

IR 里可以有无限个虚拟寄存器：

```text
%0 = add %a, %b
%1 = mul %0, %c
%2 = sub %1, %d
```

真实机器寄存器有限，register allocation 要把虚拟寄存器分配到物理寄存器。

关键概念：

- live range：一个 value 从定义到最后一次使用的范围。
- interference：两个 value 同时 live，不能放同一个寄存器。
- spill：寄存器不够时，把 value 放到栈上，需要 load/store。

例子：

```text
%0 = add %a, %b   ; %0 live
%1 = mul %0, %c   ; %0 dies, %1 live
%2 = sub %1, %d   ; %1 dies, %2 live
ret %2
```

`%0` 和 `%1` 的 live range 不重叠时，可以复用同一个物理寄存器。

常见算法：

- Linear scan：实现简单，JIT 常用。
- Graph coloring：质量更高，实现更复杂。
- SSA-based allocation：利用 SSA 结构简化冲突分析。

AI Compiler 里如果生成 Triton/CUDA，寄存器分配常交给后端编译器；但 tensor compiler 仍然需要关注寄存器压力，因为它决定 occupancy、spill 和 kernel 性能。

## Calling Convention

调用约定规定函数之间如何传参、返回、保存寄存器。

典型内容：

```text
参数放在哪些寄存器或栈位置
返回值放在哪个寄存器
caller-saved registers
callee-saved registers
stack frame 布局
栈对齐规则
```

Runtime、FFI、JIT 都需要遵守调用约定，否则编译出来的代码无法和系统或库函数正确交互。

## Interpreter

Interpreter 直接执行 AST 或 IR，不生成机器码。

IR interpreter 示例：

```text
env = {}
pc = entry
while true:
  inst = next instruction
  match inst.opcode:
    const: env[inst.result] = inst.value
    add: env[inst.result] = env[a] + env[b]
    br: pc = true_block if env[cond] else false_block
    ret: return env[value]
```

解释器的价值：

- 快速验证语言语义。
- 给优化前后 IR 做 correctness oracle。
- 避免一开始就陷入机器码生成细节。
- 可以作为 JIT 前的 baseline execution engine。

## JIT

JIT 在运行时生成或编译代码。

常见流程：

```text
input graph/function
  -> specialize by dtype/shape/device
  -> optimize
  -> generate code
  -> compile or load from cache
  -> run
```

JIT 需要额外处理：

- cache key：shape、dtype、device、layout、compile options。
- guard：输入条件变化时是否还能复用已编译代码。
- compile latency：编译时间不能压过执行收益。
- fallback：不支持的 graph 或 dynamic case 回退到解释/ eager。
- memory lifetime：编译产物和 runtime buffer 的生命周期。

PyTorch 2.x / TorchInductor 中这些问题非常明显：

```text
TorchDynamo capture
  -> guards
  -> FX graph
  -> Inductor compile
  -> Triton/C++ code
  -> cache
  -> runtime launch
```

## Runtime

Runtime 是编译结果执行时依赖的系统层。

传统语言 runtime 关注：

- 栈和堆管理。
- 函数调用。
- 动态分配。
- 异常。
- GC 或引用计数。
- 线程和同步。

AI Compiler runtime 关注：

- tensor memory planning。
- workspace 分配。
- stream/event 管理。
- kernel launch。
- library call 调度。
- shape guard 和 graph cache。
- data transfer 和 device synchronization。

例如一个 fused kernel 的 runtime 工作：

```text
check shape guards
allocate output tensor
allocate temporary workspace if needed
launch kernel on stream
record event if async dependency exists
return tensor handle
```

Runtime 不是后端之外的杂项。很多 AI Compiler 的性能问题来自 runtime：反复编译、过多 kernel launch、同步过多、workspace 重复分配、cache key 过细或过粗。
