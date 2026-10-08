---
order: 1
title: 标量数学、指数与数值实现
updated: 2026-10-04
---

# 标量数学、指数与数值实现

Softmax 需要把一组实数转换成非负权重。对 `[1,2,3]`，常见的稳定计算先减去最大值 3，再计算 `exp([-2,-1,0])`，最后除以三项之和。加减、比较与除法可以用 Arith 表达，而指数是 Math 方言提供的数学操作。

本章沿这一次指数计算说明三个层次：IR 保留了什么数学含义，lowering 怎样选择实现，最终数值为什么仍需要核对。认识 `math.exp` 不需要先背数学函数目录，但需要避免把一条 IR 操作理解为已经确定的一条机器指令。

## 1. 从移位输入到指数权重

先只看一行数据，不涉及线程或内存布局：

```text
输入 x          [1, 2, 3]
行最大值 m      3
移位 x-m        [-2, -1, 0]
指数 e          [0.135335..., 0.367879..., 1]
和 s            1.503214...
归一化 e/s      [0.090031..., 0.244728..., 0.665241...]
```

对其中一个元素，IR 只需要：

```text
%shift = arith.subf %x, %max : f32
%e = math.exp %shift : f32
```

`%e` 是 `%shift` 的指数结果，不是待调用函数的句柄，也不是一个新循环。Math 的很多操作可以作用于标量、vector 或 tensor；在这些逐元素形式下，每个元素各自应用相同数学函数。它不会自动完成跨元素最大值或求和。

在 Linalg 的 scalar body 中使用 `math.exp`，外层 indexing maps 和迭代器负责选择当前元素，Math 只处理这一对输入与结果。完整的归约和广播在[Softmax 表示](../../tutorials/kernels/reduction_softmax)中连接起来。

## 2. 数学操作与实现路径

把 `%e = math.exp %x : f32` 降低到 CPU，可以选择不同路径。

走 `convert-math-to-libm` 时，固定版本会生成对 `expf` 的函数调用，并声明这个外部函数。省略其他函数的实际关键片段是：

```text
func.func private @expf(f32) -> f32 attributes {llvm.readnone}
func.func @exp_value(%arg0: f32) -> f32 attributes {llvm.emit_c_interface} {
  %0 = call @expf(%arg0) : (f32) -> f32
  return %0 : f32
}
```

这里把“指数”交给目标系统的数学库实现。之后的函数转换负责 ABI，链接阶段需要找到 `expf`；本章本机程序通过 `-lm` 连接数学库。`math.exp` 被替换并不意味着库实现已经包含在 IR 文本里。

另一条路径 `convert-math-to-llvm` 会产生 LLVM intrinsic：

```text
llvm.func @exp_value(%arg0: f32) -> f32 attributes {llvm.emit_c_interface} {
  %0 = llvm.intr.exp(%arg0) : (f32) -> f32
  llvm.return %0 : f32
}
```

intrinsic 把“这是指数计算”的信息交给 LLVM。后端再根据类型、目标能力和配置决定如何实现，可能生成库调用，也可能采用目标支持的其他形式。看到 intrinsic 不能直接推断已经使用硬件指数指令，更不能据此得出性能结论。

这两条路径展示了 MLIR 的渐进降低：高层保留数学意图，后续阶段逐步选择调用、低层指令与目标实现。

## 3. 精度与消去误差

计算 `exp(x)-1` 时，如果 x 很小，真实结果也很小。但直接先把 `exp(x)` 舍入到 f32，可能得到恰好 1，再减 1 就变成了 0。

Math 提供 `expm1` 表达“指数减一”这个整体任务，使后续实现有机会避免这一消去过程。以下完整模块用于比较两种表达：

<!-- softmax-example: math -->
```text
module {
  func.func @exp_value(%x: f32) -> f32 attributes {llvm.emit_c_interface} {
    %r = math.exp %x : f32
    return %r : f32
  }
  func.func @exp_minus_one(%x: f32) -> f32 attributes {llvm.emit_c_interface} {
    %one = arith.constant 1.0 : f32
    %e = math.exp %x : f32
    %r = arith.subf %e, %one : f32
    return %r : f32
  }
  func.func @accurate_small_exp(%x: f32) -> f32 attributes {llvm.emit_c_interface} {
    %r = math.expm1 %x : f32
    return %r : f32
  }
}
```

对传入的 f32 `x=1e-8`，本机实测为：

| 表达与 lowering | 结果 |
|---|---|
| `math.exp` 走 libm，再减 1 | 0 |
| `math.expm1` 走 libm 的 `expm1f` | 约 `9.999999939e-9` |
| `math.expm1` 走当前 MathToLLVM 展开 | 0 |

第三行尤其值得解释。LLVM 20.1.8 的这条 `expm1` lowering 实际展开为 `llvm.intr.exp` 和 `llvm.fsub 1`，重新引入了前面那次消去。保留了更好的高层操作，还需要选择能满足任务数值要求的低层实现。本例因此使用数学库路径建立 Softmax 的 CPU 数值基线。

这份观察不表示所有 LLVM 版本和目标都采取相同行为，也不把 `math.expm1` 的名字当成任意 pipeline 的误差保证。核对实现时应继续追到所用 lowering，而不是在高层操作处停止。

## 4. 稳定算法与目标精度

回到 Softmax，移去最大值使有限输入的指数参数不大于 0，避免直接计算很大正数的指数。若某行是 `[1001,1002,1003]`，直接 f32 指数容易溢出；移位后仍为 `[-2,-1,0]`。

这减少了一类数值问题，但没有让全部浮点计算变成精确数学。指数近似、低精度输入已经丢失的差异、加法归约次序和最终除法都会影响结果。很小的权重还可能下溢为 0。判断是否接受需要结合算法的误差目标，而不能只看输出是否有限。

若使用 `exp2` 实现 exp，还要乘以 `log2(e)`；乘法与常数表示本身也会引入误差。若改用查表、多项式或目标近似指令，就应把有效输入区间、最大误差、吞吐和特殊值处理一起核对。这样的选择属于目标实现和数值契约，不能只由“这个操作看起来更快”决定。

## 5. Fast-Math 与允许的变化

浮点优化可能需要额外契约，例如允许重排加法、忽略某些 NaN/Infinity 情况、采用近似函数或倒数。相关 fast-math 标记向编译器说明哪些变化被允许；它们不会自动证明输入满足假设。

例如某个上游阶段确实保证输入有限，才可以考虑相应事实如何传递。如果实际数据包含 NaN，却仍给操作添加“没有 NaN”的约定，后续看似合法的变换就可能产生与原始期待不同的结果。对 Softmax 的分母归约，允许 reassociation 还会影响并行求和顺序和误差。

本批基线没有额外启用这类放宽。先建立清楚的有限输入、f32 运算与 double 参考对照，后续向量化和设备实现再逐项说明新增的取舍。比较实现时应同时记录 lowering 路径、类型和数值选项。

## 阅读自查

1. `math.exp` 被转换成 `llvm.intr.exp` 后，为什么还不知道最终是否调用数学库？
2. 高层使用 `math.expm1`，为什么当前例子仍要比较不同 lowering 的结果？
3. 减去最大值解决了 Softmax 的哪类问题，哪些误差仍然存在？

## 实现依据与实践

本章按 LLVM/MLIR 20.1.8 的 [MathOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Math/IR/MathOps.td)、[MathToLibm.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/MathToLibm/MathToLibm.cpp)和 [MathToLLVM.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp)核验。可选[Softmax 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/13-softmax)保存两个 lowering 的实际 IR 和本机数值；[Math 官方目录](https://mlir.llvm.org/docs/Dialects/MathOps/)用于查找其他操作，版本细节以所用源码为准。
