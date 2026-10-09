---
order: 30
title: 复数语义与标量展开
updated: 2026-10-04
---

# 复数语义与标量展开

图像与信号算法常在频域使用复数。对于编译器，复数不仅是“两项浮点数装在一起”：乘法、绝对值和除法都有自己的数学与数值约定，不能直接按两个lane独立运算。

本章从(1+2i)(3+4i)开始，追踪Complex方言如何保留这种语义，再展开成标量算术并接入低层表示。

## 1. 一个值同时保留实部与虚部

```text
%a = complex.create %one, %two : complex<f32>
%b = complex.create %three, %four : complex<f32>
%p = complex.mul %a, %b : complex<f32>
%re = complex.re %p : complex<f32>
%im = complex.im %p : complex<f32>
```

`complex<f32>`是一个SSA值，其元素类型为f32；create组合实部、虚部，re/im提取分量。与`vector<2xf32>`相比，表示中的数目虽然相同，操作语义却不同。

复数乘法为：

```text
real = a.real × b.real - a.imag × b.imag
imag = a.imag × b.real + a.real × b.imag
```

所以主例得到实部-5、虚部10。逐lane相乘只会得到[3,8]，那是另一项计算。

## 2. 高层操作展开为标量关系

固定版本convert-complex-to-standard将一般complex.mul展开为re/im、四条arith.mulf、一条subf、一条addf和complex.create。输入仍为complex类型，输出也仍重新组合成complex值。

对于完整常量主例，再canonicalize后实际得到：

```text
func.func @example() -> (f32, f32) {
  %r = arith.constant -5.000000e+00 : f32
  %i = arith.constant 1.000000e+01 : f32
  return %r, %i : f32, f32
}
```

这是编译器中的展开和常量折叠结果。处理一般输入时，四次乘法和两次加减仍会交给后续编译阶段。把高层语义展开后，也要继续遵守浮点舍入与fast-math约束，不能无条件换成任意代数等价公式。

## 3. 绝对值中的数值算法

复数绝对值在实数数学上等于sqrt(re²+im²)。直接平方可能让本来可表示的结果在中间溢出，因此固定lowering对分量绝对值先取max/min，采用缩放形式：

```text
hi = max(abs(re), abs(im))
lo = min(abs(re), abs(im))
result = hi × sqrt(1 + (lo/hi)²)
```

实际IR还包含NaN检测和选择，以处理退化情况；不能删除这些代码后仍宣称实现了完整操作语义。对于两个分量都为零，简单公式会遇到0/0，额外路径就有实际作用。

这说明lowering不一定只是把一个符号机械替换成最短数学公式。它还可能选择具有不同中间范围和特殊值处理的算法。除法和其他复数函数也应沿具体实现核对，不能从乘法例子推断全部行为。

## 4. 计算展开与类型 Lowering

convert-complex-to-standard主要处理计算，仍可能留下complex.create/re/im和complex类型。继续到LLVM时，需要相应ComplexToLLVM规则选择分量的低层表示，再处理函数签名和标量运算。

当前固定工程核对ComplexToStandard的实际输出和常量值，未运行复数目标程序。若与外部C/C++复数函数连接，还需核对目标ABI；不能仅凭LLVM里使用一个包含两个分量的结构，就断言与所有平台的语言复数调用约定相同。

## 依据与实践

固定来源：[ComplexOps](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Complex/IR/ComplexOps.td)、[ComplexToStandard](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/ComplexToStandard/ComplexToStandard.cpp)、[上游转换测试](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/test/Conversion/ComplexToStandard/convert-to-standard.mlir)。

[25 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/25-domain-extensions)提供一般乘法、绝对值和常量主例。有限练习：将第二个输入改为3-4i，先手算结果，再观察折叠，解释实部/虚部符号怎样变化。
