---
order: 4
title: 符号、作用域与隔离
updated: 2026-09-06
---

# 符号、作用域与隔离

前置：[对象模型](./ir_model)、[SSA](./values_ssa)、[类型与静态信息](./types_attributes)。本文解释 `%value` 与 `@symbol` 两种引用机制，以及函数为何可以引用外部定义却不能捕获任意外部 SSA 值。

## 1. SSA 引用与符号引用

完整模块把 caller 写在 callee 前面：

<!-- mlir-example: symbols-forward-call -->
```text
module {
  func.func @caller(%x: i32) -> i32 {
    %r = func.call @callee(%x) : (i32) -> i32
    return %r : i32
  }
  func.func @callee(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %r = arith.addi %x, %one : i32
    return %r : i32
  }
}
```

`%x` 是调用参数，是 SSA operand。`@callee` 通过符号引用标识要调用的函数，是调用操作的静态信息。查找 callee 不要求其函数定义按 SSA 支配 caller；但定义必须能在适当符号表里解析，调用类型必须满足操作约束。

符号引用不建立普通 Value 的 use-list 边。删除没有 SSA 结果的 `func.func` 之前，不能用“没有 Value user”证明函数未被调用；还要考虑符号引用和对外可见性。

## 2. SymbolTable 决定名字在哪查找

`builtin.module` 是常见的 SymbolTable。符号操作直接位于该符号表的容纳区域中，以 `sym_name` 定义名字，同一表内名字必须唯一。

通用 `SymbolRefAttr` 从引用点最近的祖先 SymbolTable 开始解析。嵌套引用如 `@outer::@inner` 需要中间对象既是符号又提供下一层符号表。不要把它想成 C++ 的任意字符串命名空间：具体引用操作可能只接受更受限的形式，例如直接 `func.call` 使用平坦函数符号引用。

嵌套 SymbolTable 形成新的查找边界。默认查找失败后，不会自动沿所有外层表逐层搜索直到找到同名函数。下面的失败例子说明这一点：

<!-- mlir-invalid: symbols-nested-lookup | does not reference a valid function -->
```text
module {
  func.func private @outside()
  module @inner {
    func.func @caller() {
      func.call @outside() : () -> ()
      return
    }
  }
}
```

`caller` 的调用在 `@inner` 的符号表中解析不到 `@outside`。这与 `%value` 的普通 CFG 支配错误不同；修复需要考虑符号组织和引用协议，不是把函数在文本中向上挪一行。

## 3. 可见性与定义/声明

| 属性/形式 | 语义 |
|---|---|
| public（默认） | 可能被当前可见 IR 外部引用，不能假设已看见所有 uses |
| private | 只允许在当前符号表内引用 |
| nested | 可通过符合可见性条件的命名符号表层次访问，但不越过可见 IR 边界 |
| 函数定义 | 有函数体，IR 中能分析其内部操作 |
| 函数声明 | 无函数体，真正实现由当前 IR 外部提供 |

函数声明常写为 `func.func private @external(i32) -> i32`。这里 private 描述 MLIR 的符号可见性，不意味着目标链接器中该外部实现一定拥有同名的私有链接属性。最终 ABI 和链接语义由目标 lowering 决定。

对外调用可能有内存效果或其他行为；看不到函数体不能当作纯函数处理。符号可见性、调用类型检查、效果推断是不同问题。

## 4. IsolatedFromAbove 限制 SSA 捕获

在普通嵌套 Region 中，只要外层 Value 的使用满足层次支配和操作约束，内部可以直接使用它。`scf.for` 中捕获外层常量是常见情况。

`func.func` 则带有 `IsolatedFromAbove`：函数内部不能直接引用该函数 Operation 外部定义的 SSA Value。下面是故意失败的模块：

<!-- mlir-invalid: symbols-isolation | outside the region -->
```text
module {
  %outside = arith.constant 7 : i32
  func.func @bad_capture() -> i32 {
    return %outside : i32
  }
}
```

虽然 `%outside` 出现在函数之前，函数仍不允许这种隐式捕获。可按真实含义选择修复：把值作为函数参数传入；在函数内部物化常量；或通过有明确语义的全局符号访问操作访问全局数据。它们表达不同程序结构，不能无条件互换。

`func.call @callee` 不违反隔离，因为它使用的是符号引用；它没有捕获 callee 的 SSA 结果。函数的参数也不违反隔离，因为参数本来就在该函数入口 Block 中定义。

## 5. 隔离为何有助于编译器工程

普通 SSA Value 的 use-list 会连到使用它的 Operation。如果多个函数任意捕获共享外层 Value，分别变换函数体时就可能同时修改同一个外层定义的 use-list。

隔离把这类 SSA 依赖限制在区域边界内，让以隔离操作为单位的分析/变换更容易局部化，也为并行编译提供条件。符号引用为全局定义提供另一种引用机制。

这不意味着“有隔离 trait 就能任意并行修改所有 IR”。PassManager 的调度、对祖先/兄弟操作的访问限制、符号表更新和分析失效仍须遵守各自规则；实现阶段再讨论这些约束。

## 6. 三种边界不能混用

| 要判断的行为 | 主要依据 |
|---|---|
| `%v` 能否作为当前操作的 operand | 定义位置、RegionKind、层次支配、隔离及额外约束 |
| `cf.br` 能否跳到 `^target` | 目标是否在同一 Region、入口限制、目标参数对应 |
| `@name` 能否被解析/访问 | 最近 SymbolTable、引用形式、可见性和操作 verifier |

不能通过符号引用任意拿到另一个函数体内的局部 SSA Value，也不能通过 branch 跳进另一个 Region 的 Block。每种连接都有单独契约。

源码级变换中，SSA 替换与符号替换也使用不同工具：`Value::replaceAllUsesWith` 处理 SSA uses；`SymbolTable::getSymbolUses`、`replaceAllSymbolUses` 等处理符号引用。直接重命名 `sym_name` 而不更新引用，会破坏调用关系。

## 理解检查

1. caller 写在 callee 前面为什么能合法调用？
2. 函数没有 SSA result，能否因此删除它？
3. 外层常量比函数先定义，为什么函数仍不能直接捕获？
4. 在嵌套 module 里查找失败，为什么不是 CFG 支配错误？

<details>
<summary>核对要点</summary>

符号引用不使用 SSA 定义顺序；要检查符号 uses 和外部可见性；隔离比普通层次支配更严格；符号查找由 SymbolTable 的边界决定。

</details>

依据：[Symbols and Symbol Tables](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/SymbolsAndSymbolTables.md)、[Traits](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Traits/_index.md)、[SymbolTable.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/SymbolTable.h)、[FuncOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Func/IR/FuncOps.td)。完整合法模块参与 P/G；两个失败模块检查对应诊断。

继续阅读：[builtin](../dialects/builtin)、[func](../dialects/func) 与 [arith](../dialects/arith)。
