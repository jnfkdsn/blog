---
order: 7
title: TIR Lowering、目标代码与 Relax VM
updated: 2026-10-30
---

# TIR Lowering、目标代码与 Relax VM

第五章得到的 sch.mod 仍是一份 TensorIR，不是 CPU 已经能执行的机器码。与此同时，Relax 主函数还要准备输出并组织数值函数调用。TVM 怎样把这两种表示分别编译，再接成一次模型执行？

本章先追踪独立 Matmul 的实际 build，观察高层块、二维 buffer 和向量循环怎样变化；再回到主模型，连接 Relax 的分配、VM 指令和已编译的数值函数。这样可以区分内核生成与模型执行组织，避免把两者都叫作“lowering”后就略过中间过程。

## 1. 从调度输出进入后端

独立 Matmul 的入口为：

```python
lib = tvm.tir.build(sch.mod, target="llvm")
```

在固定版本中，build 会绑定目标，运行 TIR pipeline，完成目标相关处理，再调用相应代码生成器。下面选取与本例有关的实际 pass，省略穿插的其他 pass；它不是要求手工照抄的完整 pipeline：

```text
LowerInitBlock
  → ConvertBlocksToOpaque / LowerOpaqueBlock
  → FlattenBuffer
  → VectorizeLoop
  → StorageRewrite 等后续处理
  → host/device 与调用接口处理
  → 目标 intrinsic 处理、LLVM codegen
```

每一步消解一种尚未落实的结构。归约 init 若仍存在，需要变成显式执行逻辑；高层 Block 完成调度使命后，其循环和访问继续保留，块结构可以降低；二维 buffer 访问转成线性偏移；向量循环转成向量表达式。第五章已经拆出的初始化不需要再由 LowerInitBlock重复生成。

这解释了为什么调度和 lowering 不能混为一件事：前者选择计算组织，后者把选定组织转换成后端可以接收的表示。某些优化也会发生在 lowering 内部，但仅设置 target 不会自动替我们选择所有 cache/compute_at 决策。

## 2. 实际中间表示与 LLVM IR

对第五章八列向量化版本，在 build 的 pass instrument 中保存实际中间模块。FlattenBuffer 之后，核心更新成为：

```text
C[i*32 + jo*8 + ji] += A[i*32 + k] * B[k*32 + jo*8 + ji]
```

这里保留了二维索引的含义，却用一维元素偏移表达。继续经过 VectorizeLoop，内层 ji 被八元素向量计算替代。将实际 TVMScript 摘录简写为：

```text
C[base : base+8] = C[base : base+8]
                  + broadcast(A[i*32+k], 8) * B[k*32+jo*8 : k*32+jo*8+8]
base = i*32 + jo*8
```

实际 LLVM 输出中可以找到 `<8 x float>` 的加载、存储，以及 `llvm.fmuladd.v8f32` 调用。对应含义是八个浮点位置一起乘加；SSA临时名与其他辅助参数略去。此时已经能从调度意图追到目标 IR，而不只根据 Python 中的 vectorize 猜结果。

TVM 负责传递索引、向量和存储等选择；LLVM 还会进行自身优化、指令选择和寄存器分配。LLVM IR 中有八元素向量，不保证目标机器最终用一条同宽指令，也不保证局部值永远不溢出到内存。检查实际机器行为需要继续看汇编或测量。

这一观察是本机CPU构建的结果。缓存案例里仍然较大的 buffer 声明，则还会经过分配位置规划与压缩等处理；判断物理存储时要明确观察的是哪个阶段。

## 3. Host 与设备代码的分工

CPU 数值函数可以直接在 host 上调用。GPU 则通常还需要把主机调用与设备计算分开：设备函数执行线程化程序，host准备参数、调用设备运行库并发起 launch。

固定 TIR pipeline 中的 `SplitHostDevice`、`MakePackedAPI` 和 `LowerDeviceKernelLaunch` 等处理连接这些边界。CUDA 路径还需要支持目标的代码生成与工具链；不能把 CPU 的向量循环仅换一个target字符串，就当作完整GPU调度。

同样，TVM的设备路径并非必须经过Triton。不同目标有各自后端和运行库。当前实际验证是LLVM CPU；GPU线程绑定、共享存储和同步的设计要与真实设备案例一起继续验证。

现在回到 Relax 模型。模型级 lowering 会把 call_tir 的输出准备好，数值 PrimFunc 再走上述后端路径；VM 将模型级调用组织和已编译数值函数连接起来。下面沿主例说明连接协议。

## 4. 两种函数的调用约定

模型层写的是：

```python
result = R.call_tir(Module.fused_matmul_add_relu,
                    (x, weight, bias),
                    out_sinfo=R.Tensor((2, 2), "float32"))
```

底层PrimFunc却接收四个buffer：三个输入和一个输出。连接它们时，需要为result准备存储，并把它作为最后一个参数交给数值函数。

跟踪 result 这个值，转换前后可用下列伪代码对照：

```text
转换前：
    result = call_tir(fused, x, weight, bias)

落实调用协议后：
    result = 准备形状为[2,2]的输出Tensor
    fused(x的数据, weight的数据, bias的数据, result的数据)
    后续计算使用 result
```

调用方准备结果位置，数值函数只负责填入。代码生成不能仅把名字替换成 fused，却漏掉第四个输出参数；否则调用双方的协议就没有接起来。

这类转换将“返回一个张量值”落实为“在已有输出buffer中写入”。模型调用者仍然获得一个结果Tensor，低层实现则采用适合生成代码的参数协议。

如果输出shape动态变化，分配所需大小也必须由运行shape计算。因此shape关系会一直传递到这里，而不是导入完成后就丢弃。

## 5. 输出分配的实际变化

本例输出有4个float32元素，需要16字节的元素存储。经过相应lowering后，实际Relax输出中可以看到下面的核心关系：

```python
storage = R.vm.alloc_storage(R.shape([16]), R.prim_value(0),
                             R.dtype("uint8"), R.str("global"))
output = R.vm.alloc_tensor(storage, R.prim_value(0),
                           R.shape([2, 2]), R.dtype("float32"))
R.vm.kill_object(storage)
Module.fused_matmul_add_relu(x, weight, bias, output)
return output
```

这里区分了原始storage对象和拥有形状、dtype等信息的tensor对象。tensor引用相应存储，并通过偏移定位其数据。

可以按对象关系追踪这个过程，而不只看操作名称：

| 阶段 | 当前程序能够通过什么对象访问结果空间 |
|---|---|
| alloc_storage 后 | storage 代表取得的存储 |
| alloc_tensor 后 | output 具有形状、类型并持有所需存储；storage 局部引用仍存在 |
| kill_object(storage) 后 | 不再需要 storage 这个局部引用，output 仍保持数据有效 |
| 数值函数调用后 | output 中已经写入结果 |
| 返回 output 后 | 调用方持有返回 Tensor，数据继续可用 |

这里的 kill 释放当前程序的一项对象引用，而不是在 output 还要使用时强制释放底层字节。返回对象、别名对象还持有什么，决定存储何时才能真正回收。表格表达逻辑拥有关系，不要求记忆底层引用计数的具体数值。

本例低层融合函数内部还有自己的临时buffer。图层的输出分配与TensorIR函数内部的局部存储属于不同层次，不能只统计这一条16字节分配就声称整个计算只使用16字节工作内存。

## 6. 静态规划与动态需求

若某些张量的大小与生命周期可以在编译时确定，内存规划有机会让不同时存活的张量复用存储。判断复用时，需要考虑最后使用位置、别名、对齐、设备和大小等条件。

例如 A 有四个 float32 元素，最后一次读取发生在步骤2；B 也需要16字节，从步骤3才开始写入。如果设备、对齐和别名等条件兼容，二者的存活区间不重叠，可以考虑同一块空间。

若步骤4还要读取 A，步骤3写入 B 就会破坏 A 的旧值。只看“A、B大小相同”不能决定复用，必须把最后使用位置连接到写入时机。返回 A 也是一种需要继续保留它的使用，不能因本函数内部没有后续读就判定它已经失效。

动态batch也可以保留静态关系：输出始终为 `[B,2]`、float32，所需元素字节数可写成 `8B`。但B的具体数值在运行时取得，最终是否复用以及如何分配，要看对应规划和运行时策略。

因此“静态内存规划”不等于所有模型内存都必须是编译期常量，也不意味着存在任意两个张量就能复用。可以证明的部分提前处理，其余通过运行时协议落实。

## 7. VM 指令与数值函数

本例默认使用bytecode执行组织。实际VM文本的关键部分为：

```text
@main:
  call vm.builtin.check_tensor_info ...
  call vm.builtin.match_shape ...
  call vm.builtin.alloc_storage ... dst: %3
  call vm.builtin.alloc_tensor  ... dst: %4
  call vm.builtin.null_value        dst: %3
  call fused_matmul_add_relu in: %0, %1, %2, %4 dst: %void
  ret %4
```

省略号代表具体检查参数和常量索引，完整输出由实验保存。它展示的是模型级操作：检查三个输入、准备输出、调用数值函数、返回tensor。

把这一段实际指令代入主例：%0、%1、%2 是输入；%3 暂存 storage；%4 是输出 Tensor。创建 %4 后，%3 可以清空；调用 fused 时仍把 %4 传进去，最后 ret 返回 %4。这解释了为什么清空 storage 局部槽位不会让结果失效。

矩阵乘法中的乘加循环没有逐条出现在VM字节码里，而是由LLVM路径编译成数值函数。VM调用该函数完成计算。

这样既可以保留模型层的动态形状或控制组织，又能让计算密集部分使用编译代码。固定版本还支持其他执行组织选项；本章以实际使用的bytecode路径建立认识，不把某种模式当成所有程序唯一形式。

## 8. 保存、加载与部署边界

本例将可执行对象导出为共享库，再加载：

```python
executable.export_library("affine.so")
module = tvm.runtime.load_module("affine.so")
vm = relax.VirtualMachine(module, tvm.cpu())
y = vm["main"](x_nd, weight_nd, bias_nd)
```

本机实际加载后得到与原执行相同的 `[[0.5,7.5],[1.5,0]]`。保存和加载验证了产物边界，不再依赖重新构建那份Relax图来完成调用。

部署环境仍需相容的runtime、系统和目标架构；外部实现还需要能解析相应符号或加载对应模块。前章Python注册的PackedFunc若未在新进程注册，就不会因保存模型而自动变成独立C++实现。

输入输出交换同样有成本。NumPy到NDArray的构造和返回NumPy可能复制数据；DLPack等机制可以在适用条件下共享存储，但所有权、设备和同步契约仍然需要遵守。只测数值kernel，不包含这些成本；端到端应用测量则必须明确包含哪些步骤。

## 9. 推演与实践入口

1. sch.mod 中有 vectorized 循环，后端还要完成什么？FlattenBuffer 与 VectorizeLoop 分别改变什么表示？
2. call_tir 的结果存储由谁准备？VM 为什么不需要逐条解释矩阵乘加？
3. alloc_tensor 后清空 storage 的局部引用为何不使 output 失效？保存共享库又为什么不能自动带走 Python 注册的外部函数？

实验见 [TVM labs 的 runtime](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)。固定源码入口为 [CallTIRRewrite](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/call_tir_rewrite.cc)、[StaticPlanBlockMemory](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/static_plan_block_memory.cc)、[VM构建](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/vm_build.py) 和 [VM runtime](https://github.com/apache/tvm/blob/v0.20.0/src/runtime/relax_vm/vm.cc)。

后端入口可查 [TIR build](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/tir/build.py)、[TIR pipeline](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/tir/pipeline.py) 和 [LLVM codegen](https://github.com/apache/tvm/blob/v0.20.0/src/target/llvm/codegen_llvm.cc)。[design_probe](https://github.com/jnfkdsn/aicompiler/blob/main/tvm-e2e/design_probe.py)从实际build中截取pass输出，并保存LLVM IR；不将手工依次调用几个pass冒充实际完整pipeline。
