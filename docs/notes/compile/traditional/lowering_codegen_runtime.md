---
order: 5
title: Lowering、Codegen、Runtime
updated: 2026-07-01
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
