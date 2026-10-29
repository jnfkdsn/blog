---
order: 7
title: 编译产物、Relax VM 与内存生命周期
updated: 2026-10-28
---

# 编译产物、Relax VM 与内存生命周期

数值函数已经能计算 `relu(x @ weight + bias)`，但应用还不能只拿着循环体就完成一次模型调用。它需要接收张量、检查结构、准备输出、找到函数入口，并在合适的时机管理存储。

本章沿一次 `vm["main"](x, weight, bias)` 追踪这些工作，最后把编译产物保存为共享库，再重新加载执行。这样，代码生成之后的运行时不再只是流程图末尾的一个名字。

## 1. 两种函数的调用约定

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

## 2. 输出分配的实际变化

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

## 3. 静态规划与动态需求

若某些张量的大小与生命周期可以在编译时确定，内存规划有机会让不同时存活的张量复用存储。判断复用时，需要考虑最后使用位置、别名、对齐、设备和大小等条件。

例如 A 有四个 float32 元素，最后一次读取发生在步骤2；B 也需要16字节，从步骤3才开始写入。如果设备、对齐和别名等条件兼容，二者的存活区间不重叠，可以考虑同一块空间。

若步骤4还要读取 A，步骤3写入 B 就会破坏 A 的旧值。只看“A、B大小相同”不能决定复用，必须把最后使用位置连接到写入时机。返回 A 也是一种需要继续保留它的使用，不能因本函数内部没有后续读就判定它已经失效。

动态batch也可以保留静态关系：输出始终为 `[B,2]`、float32，所需元素字节数可写成 `8B`。但B的具体数值在运行时取得，最终是否复用以及如何分配，要看对应规划和运行时策略。

因此“静态内存规划”不等于所有模型内存都必须是编译期常量，也不意味着存在任意两个张量就能复用。可以证明的部分提前处理，其余通过运行时协议落实。

## 4. VM 指令与数值函数

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

## 5. Target、设备与 Host

当前 `target="llvm"` 生成CPU实现，输入也位于CPU上。更一般的设备程序通常同时需要host侧调用组织与device侧数值代码。

例如GPU kernel需要由host准备参数、选择执行配置并发起launch；设备端执行并行计算。target描述编译面向的能力，runtime device描述数据和执行所处设备，二者需要匹配。

仅把target字符串改成CUDA，并不自动获得一个已经正确映射、能够在任意GPU运行的程序。还需要适用调度、目标工具链、设备运行库与实际执行验证。当前课程已经核实CPU链，未执行GPU/NPU链。

## 6. 保存、加载与部署边界

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

## 7. 推演与实践入口

1. call_tir 在模型层返回 Tensor，而 PrimFunc 接收输出 buffer，谁把两种协议接起来？
2. alloc_tensor 之后清空 storage 的局部引用，为什么 output 仍可被数值函数写入和返回？
3. A 在步骤4仍被读取，B 在步骤3开始写入，大小相同是否足以让二者复用存储？

实验见 [TVM labs 的 runtime](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e)。固定源码入口为 [CallTIRRewrite](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/call_tir_rewrite.cc)、[StaticPlanBlockMemory](https://github.com/apache/tvm/blob/v0.20.0/src/relax/transform/static_plan_block_memory.cc)、[VM构建](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/relax/vm_build.py) 和 [VM runtime](https://github.com/apache/tvm/blob/v0.20.0/src/runtime/relax_vm/vm.cc)。
