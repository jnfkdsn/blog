---
order: 1
title: Builtin：模块与公共基础
updated: 2026-09-06
---

# Builtin：模块与公共基础

前置：[对象模型](../core/ir_model)、[类型](../core/types_attributes)、[符号](../core/symbols_scopes)。Builtin 提供广泛共用的操作、类型和属性；其名称不能简单理解为“所有语言的基础指令集”。

## 1. module 是容器 Operation

完整模块：

<!-- mlir-example: builtin-module -->
```text
module @example {
  func.func @identity(%x: i32) -> i32 {
    return %x : i32
  }
}
```

`module` 是 `builtin.module` 的简洁写法。它拥有一个 body Region，其中有一个无参数 Block；module 本身没有普通 SSA operands/results。可选的 `@example` 是符号名，不是一个返回 module 对象的 SSA 结果。

| 契约 | LLVM 20.1.8 中的含义 |
|---|---|
| Graph Region | 定义的排版顺序不表示依次调用它们 |
| 单 Block、无 Region 参数 | 模块体不是函数调用时接收实参的入口 |
| NoTerminator | 不要求在模块末尾写 return/yield |
| SymbolTable | 组织并解析直接容纳的符号 |
| IsolatedFromAbove | 不能从 module 外部捕获普通 SSA 值 |

module 是常用顶层容器，但 MLIR 的数据结构并不要求每份可处理 IR 都必须以这个特定 Op 为根；工具和 pipeline 可以选择相应的容器约定。本文示例显式写 module，方便解析、导航和验证。

## 2. 类型的归属与使用方言不同

`i32`、`index`、TensorType、MemRefType、FunctionType 等公共类型由 Builtin 提供。`arith.addi` 使用 i32，不会把 i32 变成 arith 专属类型；tensor/memref 方言定义操作，相关容器类型则有公共基础定义。

同一个 module 可以混合 `func`、`arith`、`scf`、`tensor`、`memref`。Dialect 是扩展语义的组织单位，不等于一个互斥的编译阶段。一次 lowering 可能只替换某些 Op，留下其他方言继续参与后续变换。

公共属性包括整数/浮点、字符串、数组、字典、符号引用、类型属性、位置等。它们为方言构建静态信息提供可复用结构；具体哪些字段必须存在、怎样影响计算，由操作契约解释。

## 3. unrealized_conversion_cast 是转换中的占位桥接

逐步做类型转换时，有些使用者已改用新类型，另一些尚未转换。`builtin.unrealized_conversion_cast` 可以在转换过程中表达暂未兑现的类型衔接。

下面是局部语法示意，不是可执行的数值转换实现：

```text
%new = builtin.unrealized_conversion_cast %old : !source.type to !target.type
```

仅仅写出 cast 不能证明源类型与目标类型的数值或布局转换已经实现。它不同于有明确扩展语义的 `arith.extsi`。最终 pipeline 通常需要由实际 conversion/materialization 消解这类边界，或在可抵消情形下清理占位 cast。

本阶段只认识它的职责；TypeConverter、materialization 与 reconciliation 的算法和失败路径在 Dialect Conversion 章节展开。

## 理解检查与源码

说明 module 的 Block 为什么没有 return；为什么两个函数在模块中换排版顺序通常不改变调用逻辑；为什么混合方言 IR 不一定是“编译到一半的错误状态”。

依据：[BuiltinOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/BuiltinOps.td)、[BuiltinTypes.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/BuiltinTypes.td)、[BuiltinAttributes.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/BuiltinAttributes.td)。优先阅读 ModuleOp 的 traits 与描述，再按需要进入 `BuiltinOps.cpp`。完整模块参与 P/G，占位 cast 片段不参与。

下一章：[Func：定义、调用与返回](./func)。
