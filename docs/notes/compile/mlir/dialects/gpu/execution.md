---
order: 10
title: Kernel 与执行层次
updated: 2026-10-04
---

# Kernel 与执行层次

CPU Vector 例子把相邻四项组成一个向量值。把同一计算放到 GPU 时，还有另一个选择：让不同执行实例各自负责一项，再决定实例如何分组。GPU 方言表示这种启动和执行分工。

本章仍计算 `y[i]=2×(x[i]+1)`，输入长度为 10。先手工划分工作，再把分工写成 IR，最后观察 kernel 从 host 函数中抽离后的形态。

## 1. 从迭代点到执行实例

假设每组四个线程，共启动三组。组内线程编号 `t` 为 0—3，组编号 `b` 为 0—2；负责的元素是 `i=4b+t`。

| block 编号 b | 四个 thread 负责的位置 | 需要执行计算的位置 |
|---|---|---|
| 0 | 0、1、2、3 | 全部 |
| 1 | 4、5、6、7 | 全部 |
| 2 | 8、9、10、11 | 8、9 |

这里的 block 对应 workgroup，thread 对应 work item。第三组仍包含四个执行实例，但后两个不访问输入输出。正确性的要求是每个有效元素恰好被一个实例写入，所有访问都在界内。

四线程一组仅为讲清映射，通常不是高效的实际配置。这个分工也没有指定哪一个物理计算单元先执行哪组；block 编号描述工作身份，不是启动顺序或物理 SM 编号。

## 2. 用 `gpu.launch` 表达分工

下面完整函数接收已经适合设备访问的输入输出。它不负责把普通 host 内存自动变成 device 内存；数据搬运将在后面接上。

```text
func.func @launch(%x: memref<10xf32>, %y: memref<10xf32>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c3, %gy = %c1, %gz = %c1)
    threads(%tx, %ty, %tz) in (%dx = %c4, %dy = %c1, %dz = %c1) {
    %base = arith.muli %bx, %dx : index
    %i = arith.addi %base, %tx : index
    %valid = arith.cmpi ult, %i, %c10 : index
    scf.if %valid {
      %a = memref.load %x[%i] : memref<10xf32>
      %b = arith.addf %a, %one : f32
      %c = arith.mulf %b, %two : f32
      memref.store %c, %y[%i] : memref<10xf32>
    }
    gpu.terminator
  }
  return
}
```

`blocks` 后半部给各维的组数；`threads` 后半部给每组各维的线程数。第一组括号中的 `%bx/%tx` 等，是 Region 内当前执行实例的编号；`%gx/%dx` 等则是启动尺寸。它们都是 `index`，但含义不同。

本例只有 x 维有效，y、z 的启动尺寸明确写成 1。每个实例执行相同的 Region，只是编号不同。`scf.if` 决定这个实例是否读写有效元素；它没有减少已经启动的线程总数。

## 3. Kernel 的显式边界

launch Region 可以引用外层的 `%x`、`%y` 和常量。生成独立设备代码时，编译器必须把这种捕获变成明确的参数边界。`gpu-kernel-outlining` 就完成这一步。

实际结果的结构如下，省略函数体和无用的 y/z 索引读取：

```text
module attributes {gpu.container_module} {
  func.func @launch(...) {
    gpu.launch_func @launch_kernel::@launch_kernel
      blocks in (%c3, %c1, %c1) threads in (%c4, %c1, %c1)
      args(%c10 : index, %x : memref<10xf32>,
           %one : f32, %two : f32, %y : memref<10xf32>)
    return
  }
  gpu.module @launch_kernel {
    gpu.func @launch_kernel(%n: index, %x: memref<10xf32>,
                           %one: f32, %two: f32, %y: memref<10xf32>) kernel {
      %b = gpu.block_id x
      %t = gpu.thread_id x
      %size = gpu.block_dim x
      // 同样计算 i=b*size+t，检查 i<n，再读、算、写。
      gpu.return
    }
  }
}
```

这里有两种函数角色。外层 `func.func` 是 host 侧的启动逻辑；`gpu.module` 内的 `gpu.func ... kernel` 是设备入口。`gpu.launch_func` 持有二级符号引用，把启动与目标入口连接起来。

它还不是普通 `func.call`：启动需要 grid/block 尺寸、参数打包及设备运行时；设备函数本体则要编译成设备指令。此前的函数 ABI、符号、模块边界知识在这里分别承担具体任务。

## 4. Thread、Subgroup 与 Vector

多个 thread 还可能组成硬件执行组，GPU 方言用 subgroup 表达相近的层次。NVIDIA 通常称 warp；具体宽度与执行规则要按目标确认，不能把某个目标的数值当成 GPU 方言通用常量。

这与上一模块的 Vector 有区别：`vector<4xf32>` 表示一个值中有四项，而本章四个 thread 表示四个执行实例。一个 thread 可以持有多个向量元素；某些降低也会把一个逻辑向量分布给一组线程。**向量形状本身没有规定这种分布。**

block 也不是物理核心。实际同时驻留多少组取决于线程数、共享存储、寄存器等资源；组数很多不意味着所有组同一时刻运行。因此不同 block 之间不能依赖“编号小的一定先完成”。

## 5. 动态尺寸与空工作

动态长度 N 的组数可取 `ceildiv(N,B)`，再在 kernel 中检查 `i<N`。这两层分别解决覆盖范围和尾部安全；只做 ceildiv 仍会有额外实例。

当 N=0 时，数学上没有工作。实际运行时是否接受零维启动不是应依赖的通用约定。本课程的完整 host 例子在 `N>0` 时才分配、搬运与启动；空输入直接返回。

输入输出还需要满足相同长度、正确布局和不重叠等约定。IR 能解析并不证明调用者传入了合法的设备地址，也不证明工作划分无遗漏、无数据竞争。

## 6. 阅读检查与衔接

1. 将每组宽度改为 8，长度 10 需要几组？哪些实例只经过条件判断？
2. kernel body 里的 `%tx` 与一个 Vector 的 lane 下标，分别描述哪一层对象？
3. outlining 为什么要把外层 `%x` 改成 kernel 参数？

下一篇解释[地址空间与设备存储](./memory_spaces)，随后进入[同步与异步操作](./synchronization)。从 scf.parallel/forall 自动生成这类分工的过程归入[目标后端模块](../../compiler/targets/)。

本文依据固定 LLVM 20.1.8 [GPUOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td) 与 [KernelOutlining.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp)。文中 outlining 来自实际工具输出；编号表为执行语义推演，不冒充设备运行结果。
