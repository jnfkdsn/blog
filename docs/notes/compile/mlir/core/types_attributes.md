---
order: 3
title: 类型、Attribute 与 Properties
updated: 2026-09-06
---

# 类型、Attribute 与 Properties

前置：[Value 与 SSA](./values_ssa)。本章区分“程序传递的值”和“IR 对它的静态描述”，再解释类型、形状、属性、Properties 与 Location。自定义 Type/Attr 的存储和 ODS 生成留到编译器实现阶段。

## 1. 三种信息承担不同职责

完整模块：

<!-- mlir-example: types-constant -->
```text
module {
  func.func @add_seven(%x: i32) -> i32 {
    %seven = arith.constant 7 : i32
    %y = arith.addi %x, %seven : i32
    return %y : i32
  }
}
```

| 文本 | 类别 | 解释 |
|---|---|---|
| `%seven` | SSA Value | 常量操作的结果，可以作为 operand |
| `i32` | Type | 对值的表示和合法运算作静态约束 |
| `7 : i32` | IntegerAttr | 常量的静态内容；无需另一条运行时计算提供 7 |
| `arith.constant` | Operation 名称 | 规定如何把静态常量内容表示成可使用的 SSA 结果 |

所以“常量是 Attribute”和“常量可以作为 SSA Value 传递”不矛盾：前者是 constant Op 存储的内容，后者是它产生的结果。Attribute 可以携带 Type，如 `IntegerAttr`；有 Type 的对象不一定就是 SSA Value。

## 2. Type 不仅是标量位宽

MLIR 类型系统可由 Dialect 扩展。基础阅读先区分以下层次：

| 类型示例 | 静态知道的内容 | 不能据此断言的内容 |
|---|---|---|
| `i32`、`f32` | 标量类别与位宽 | `i32` 的每个运算都采用有符号解释 |
| `index` | 索引/尺寸计算使用的整数类型 | 在所有目标上恒为 i64 |
| `tensor<4x8xf32>` | rank=2、静态 shape、元素类型 | 已分配 128 字节 GPU 内存 |
| `tensor<?x8xf32>` | rank=2、第一维动态 | 第一维必然等于另一个 tensor 的动态维 |
| `tensor<*xf32>` | 元素类型已知、rank 未知 | rank 已知只是尺寸未知 |
| `memref<4x8xf32>` | 可寻址存储视图的类型及默认布局等信息 | 指针不与其他 memref 别名 |
| `vector<4xf32>` | 向量形状与元素类型 | 必然一条某种 GPU 向量指令完成 |
| `(i32) -> i32` | 函数输入/输出类型列表 | 自己是一个函数定义或可调用 SSA Value |

类型表达抽象层次的约束，实际存储、指令和 ABI 还需要 lowering 与目标信息。特别是 `index` 的最终位宽受目标/转换约定影响；当前示例用小范围非负 index，避免把宿主机位宽默认为所有后端的语义。

## 3. 静态形状与运行时尺寸

`?` 表示该维度的尺寸不在类型里固定。程序可以通过 `tensor.dim` 等操作取得运行时尺寸，再作为 index Value 使用。

<!-- mlir-example: types-dynamic-dim -->
```text
module {
  func.func @rows(%t: tensor<?x8xf32>) -> index {
    %axis = arith.constant 0 : index
    %n = tensor.dim %t, %axis : tensor<?x8xf32>
    return %n : index
  }
}
```

`%axis` 的定义是运行时 IR 中的 SSA 值，但编译器能从 constant 看出它恒为 0；`%n` 的数值由本次传入 tensor 的尺寸决定。静态/动态是在说明信息如何表示与是否已知，不等于“编译器完全不知道这个 Value 的任何性质”。

两个同类型 `tensor<?x8xf32>` 的 Value，第一维可以分别是 3 和 7。类型相同不能证明两个动态维相等；需要操作契约、形状分析或运行时条件提供额外依据。相反，一个 tensor 的尺寸被某次变换推导出来后，类型或程序也可能被进一步精化。

## 4. 类型相同与显式转换

操作会约束输入/结果类型，例如常用标量 `arith.addi` 要求两个输入和结果类型相同。不能像高级语言那样默认把 i32 与 i64 相加并自动提升。

<!-- mlir-example: types-explicit-cast -->
```text
module {
  func.func @widen_add(%x: i32, %y: i64) -> i64 {
    %wide = arith.extsi %x : i32 to i64
    %sum = arith.addi %wide, %y : i64
    return %sum : i64
  }
}
```

`extsi` 选择符号扩展；`extui` 选择零扩展，两者不是拼写替换。对于高位为 1 的 i32 位模式，扩展结果不同。把 i32 的 bits 重新解释成 f32 的 `bitcast` 与按数值把整数转为浮点的 `sitofp` 也不同。

类型相等只是合法性的一个条件。两个同类型数值的相除仍可能遇到除零，两个同类型 memref 仍可能别名；verifier 不负责解决所有运行时前提。

## 5. Attribute、inherent 与 discardable

Attribute 保存 IR 中不通过 SSA 动态传入的结构化数据，例如比较谓词、符号名、函数类型、常量内容、稠密元素数组和方言特定配置。

- inherent attribute 属于该 Op 自身的定义与语义，例如 `arith.cmpi` 的比较谓词。
- discardable attribute 的语义由操作外部的方言/基础设施约定，例如带方言命名空间的附加标注。

这一区分描述谁定义、验证和解释信息，不能按“名字看起来像注释”猜测是否可以删除。`discardable` 也不意味着所有 pipeline 在任何位置删除它都没有工程后果；要核对相应方言对该信息的契约。

例如 `arith.cmpi slt, %a, %b : i32` 的 `slt` 是静态谓词，不是 i1 条件值。操作执行比较后产生的 i1 结果，才可以传给 if 或 cond_br。

## 6. Properties 与 Attribute 的关系

在采用 Properties 的操作中，inherent 信息可以放入操作专属的 Properties 存储，而顶层 attribute dictionary 保存 discardable attributes。Properties 还可以容纳该操作定义的其他数据；它提供通用序列化所需的 Attribute 表达。

所以不能把 Properties 讲成“另一种动态输入”，也不能断言所有静态信息都必然放在 `getAttrDictionary()` 里。某个 Op 是否迁移到 Properties、其访问器如何生成，要看当前版本定义。

下面是与本章第一个例子等价的完整通用格式模块：

<!-- mlir-example: types-generic-properties -->
```text
"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "add_seven"}> ({
  ^bb0(%x: i32):
    %seven = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %y = "arith.addi"(%x, %seven) : (i32, i32) -> i32
    "func.return"(%y) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

`<{...}>` 是 Properties 的通用文本位置；普通 `{...}` 的 attribute dictionary 是另一个位置。自定义语法的常量数字、函数名与函数签名，是 printer 为这些结构化数据选择的简洁写法。通用形式让存储分类更清楚，但不代表更低层的计算 IR。

## 7. 别名与 Location

文本可使用 `!` 类型别名、`#` 属性别名来减少重复。别名是文本层的引用便利，不会创建新的 SSA Value，也不保证产生一种与被引用类型不同的 nominal type。方言类型的 `!dialect.type` 与本地类型别名也要按实际定义区分。

Location 描述 IR 与源位置或生成位置的联系，用于诊断和变换后溯源。操作在内存中有 Location，即使普通 printer 没有显示它；没有准确源位置时可用 unknown location。下面是完整模块：

<!-- mlir-example: types-location -->
```text
module {
  func.func @located() -> i32 {
    %v = arith.constant 7 : i32 loc("example.py":4:8)
    return %v : i32
  }
}
```

`loc(...)` 不是 operand，也不参与整数计算。查看位置通常需要 `--mlir-print-debuginfo`。复杂变换可能需要融合多个来源或保留调用位置链，不能只在报错时才考虑 Location。

## 理解检查

1. `7`、`%seven`、`i32`、`value` 的含义分别是什么？
2. 两个 `tensor<?x8xf32>` 是否一定可逐元素相加？还缺什么信息？
3. 为什么读取所有顶层 attributes 可能漏掉某个 Op 的语义信息？
4. 改变 `%v` 名字和把 `slt` 改为 `ult`，哪一个可能改变程序结果？

<details>
<summary>核对要点</summary>

常量内容、SSA 结果、类型、Properties 中的字段名；动态维相等或操作允许的广播规则仍需证明；inherent 数据可能存于 Properties；改变比较谓词会改变部分位模式上的比较结果，SSA 名字变化本身不会。

</details>

依据：[LangRef](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/LangRef.md) 的 Types/Attributes/Properties；[BuiltinTypes.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/BuiltinTypes.td)、[BuiltinAttributes.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/BuiltinAttributes.td)；[ArithOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)。完整模块参与 P/G；类型/API 实现未作为 C++ 编译实验报告。

下一章：[符号、作用域与隔离](./symbols_scopes)。
