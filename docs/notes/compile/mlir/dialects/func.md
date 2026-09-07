---
order: 2
title: Func：定义、调用与返回
updated: 2026-09-06
---

# Func：定义、调用与返回

前置：[符号与作用域](../core/symbols_scopes)、[builtin](./builtin)。本章完整解释直接函数调用链；间接调用与接口先定位，ABI、内联算法和目标调用约定在后续实现/lowering 阶段展开。

## 1. 三个操作构成调用协议

| 操作 | SSA 输入/结果 | Region | 主要静态信息 |
|---|---|---|---|
| `func.func` | 无普通 SSA operand/result | 一个 body；声明时为空 | 符号名、function_type、可见性等 |
| `func.call` | 调用实参 → 调用结果 | 无 | callee 符号引用 |
| `func.return` | 要交给调用者的值 → 无结果 | 无 | 与所属函数签名一致的约束 |

完整模块同时展示多结果和调用：

<!-- mlir-example: func-multiple-results -->
```text
module {
  func.func @sum_diff(%a: i32, %b: i32) -> (i32, i32) {
    %sum = arith.addi %a, %b : i32
    %diff = arith.subi %a, %b : i32
    return %sum, %diff : i32, i32
  }
  func.func @use_pair(%x: i32, %y: i32) -> i32 {
    %pair:2 = func.call @sum_diff(%x, %y) : (i32, i32) -> (i32, i32)
    return %pair#1 : i32
  }
}
```

调用发生时，`%x/%y` 对应 callee 入口参数 `%a/%b`；callee 的 return operands 按位置对应 call 的结果。`%pair#0` 和 `%pair#1` 是 call 的两个 OpResult，`%pair` 不代表把它们打包成 tuple 的另一个 Value。

## 2. 函数类型与 Operation 类型签名

`func.func @sum_diff(...) -> (i32, i32)` 描述函数被调用时的类型。用通用语法看，这个定义 Op 仍是 `() -> ()`，函数类型放在 `function_type` 中。

`func.call` 的通用输入输出则真的对应 SSA operands/results。把两者混淆，容易误以为 `%a` 是 func.func 的普通 operand，或函数定义本身产生两个 SSA 结果。

定义的入口 Block 参数须与函数输入类型一致；每个 `func.return` 的 operands 须与函数输出类型一致。函数可以有多个 Block 和多个 return，只要各路径满足函数/CFG 约束。

## 3. 声明没有函数体

完整模块：

<!-- mlir-example: func-external -->
```text
module {
  func.func private @external(i32) -> i32
  func.func @caller(%x: i32) -> i32 {
    %y = func.call @external(%x) : (i32) -> i32
    return %y : i32
  }
}
```

`@external` 有符号与签名，但 body Region 中没有 Block。空 Region 不等于一个空 Block；后者在函数定义中仍需要合法终结方式。

这个模块可通过静态解析/验证，但不代表已有外部实现可以链接执行。直接调用 verifier 可以核对可见声明和类型，无法提供实际库函数。外部实现的效果、ABI 和链接由后续链路处理。

## 4. 调用类型不匹配的反例

下面故意让 call 使用不同于声明的输入类型：

<!-- mlir-invalid: func-bad-call | operand type mismatch -->
```text
module {
  func.func private @expects_i32(i32) -> i32
  func.func @bad(%x: i64) -> i32 {
    %r = func.call @expects_i32(%x) : (i64) -> i32
    return %r : i32
  }
}
```

局部 call 的 operand 与它自己写出的类型相符，仍会在解析目标符号后发现与 callee 不一致。这说明某些合法性检查需要跨操作/符号信息，不能只看单行语法。

若需要转换，应选择语义正确的 cast 并相应修改调用；不能只把文字中的 i64 改成 i32，因为 `%x` 的真实定义类型没有改变。

## 5. 调用与包含、间接调用与接口

`func.call` 不拥有 callee 的函数体；递归调用也不会让静态 IR 无限嵌套。只有内联等变换才会把相应计算克隆/合并到调用位置，并处理 return、参数、符号和控制流。

`func.constant` 可以把函数引用物化为函数类型的 SSA Value，`func.call_indirect` 再消费这样的 Value 作为 callee。间接调用的数据依赖与直接符号引用不同，可能需要额外分析才能确定目标。本阶段不展开完整间接调用程序和优化。

通用编译器组件不必硬编码所有 func 操作：`FunctionOpInterface`、`CallOpInterface` 等接口描述可以跨方言使用的行为。认识接口的职责即可；具体实现方法属于后续 ODS/Interface 章节。

## 理解检查与源码

独立指出例子中函数定义、call 和 return 各自的 operands/results；说明声明为何可验证却未必可运行；解释修改调用一行的类型文本为何不足以完成类型转换。

依据：[FuncOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Func/IR/FuncOps.td)、[FuncOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Func/IR/FuncOps.cpp)、[Func 测试目录](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/test/Dialect/Func)。阅读路线为 FuncOp/CallOp/ReturnOp 定义 → CallOp 的符号使用验证 → 对应错误测试。完整模块参与 P/G 或预期失败检查，未链接执行 external。

下一章：[Arith：标量计算的语义](./arith)。
