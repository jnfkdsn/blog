---
order: 40
title: IRDL 中的定义与约束求值
updated: 2026-10-04
---

# IRDL 中的定义与约束求值

[ODS](../../compiler/ir_definition/op_definition)把声明翻译成C++，再把操作实现编译进工具。如果某个工具希望在运行时加载结构定义，可以把“定义操作这件事”本身写成IR。IRDL就是这样一层表示：其程序描述方言、类型、属性与操作的约束。

本章定义一个只接受i32的my.add，观察加载前后的区别，再说明这些结构约束为何还不足以让编译器把它当成算术加法。

## 1. 定义程序与被定义的程序

定义文件包含：

```text
module {
  irdl.dialect @my {
    irdl.operation @add {
      %i = irdl.is i32
      irdl.operands(lhs: %i, rhs: %i)
      irdl.results(result: %i)
    }
  }
}
```

它描述my.add有两个i32输入、一个i32结果。这里的%i是一项类型约束，不是运行时整数，也不是add的结果。

另一个输入文件才是待编译程序：

```text
func.func @sum(%a: i32, %b: i32) -> i32 {
  %r = "my.add"(%a, %b) : (i32, i32) -> i32
  return %r : i32
}
```

两个文件都使用MLIR语法，却处在不同层次：前者定义结构规则，后者声明符合这些规则的操作实例。把两者分清，就不会误以为执行irdl.is会在目标程序里生成整数计算。

## 2. 加载与检查的实际过程

固定mlir-opt通过`--irdl-file=definition.mlir`加载定义，在Context中建立动态方言和操作。随后读取generic形式的my.add，并按定义检查输入/结果。

原来的i32程序能够打印往返。把函数参数、操作输入和结果都改为f32，普通函数结构仍然一致，但IRDL验证报告：

```text
expected 'i32' but got 'f32'
```

错误来自已加载的定义，而不是操作名的拼写。相比之下，允许unregistered dialect只是在某些情况下容纳未知结构，不会自动提供这项i32约束。

## 3. 约束值的共享关系

irdl.is表达固定类型；更一般的约束还可以使用any_of、all_of、base、parametric等。关键不只是“某个输入属于某个集合”，还包括多个位置是否共享同一个约束变量。

例如同一个any_of(f32,f64)结果同时用于两个operand和result，它在一次验证中选择一致的具体类型。这要求三者相等，不是允许第一个f32、第二个f64、结果任意选一个。

这种SSA共享让类型间关系出现在定义IR中，方便生成和分析。固定上游说明也强调IRDL主要面向可生成、可分析的表示，不要求把它当成所有手写方言开发的默认方式。[IRDL定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/IRDL/IR/IRDL.td)说明了约束求值关系。

## 4. 结构定义没有自动产生计算语义

名称my.add只是名字。本例没有声明Pure效果，没有fold，也没有lowering。即使结果无人使用，canonicalize之后这个动态操作仍然保留，因为编译器没有足够信息证明删除安全。

因此，仅加载上述定义不会自动得到：

```text
my.add → arith.addi → LLVM → 可执行函数
```

若要完成这条链，还必须规定语义，并提供对应变换及需要的接口/效果事实。结构schema帮助识别和验证对象，不能替代这些工作。

[插件](../../guides/plugins)则可以加载完整原生C++实现。两者都属于扩展工具的方法，但一个主要加载声明约束，另一个可以携带Pass、接口和其他原生行为；应根据实际任务选择。

## 依据与实践

固定来源：[IRDLOps](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/IRDL/IR/IRDLOps.td)、[动态加载测试](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/test/Dialect/IRDL)。[在线参考](https://mlir.llvm.org/docs/Dialects/IRDL/)用于查询操作类别。

[25 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/25-domain-extensions)包含定义文件、成功/错误类型及未使用操作保留的对照；没有伪造my.add的执行实现。有限练习：把定义改为i64，在不改函数IR时预测哪项约束失败，再同步修改输入。
