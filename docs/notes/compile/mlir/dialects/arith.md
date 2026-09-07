---
order: 3
title: Arith：标量计算与数值语义
updated: 2026-09-06
---

# Arith：标量计算与数值语义

前置：[类型与静态信息](../core/types_attributes)。Arith 提供整数/浮点常量、算术、比较、选择和类型转换。本文完整展开读控制流所需的标量核心，并说明整数溢出、浮点与推测执行边界；不逐项罗列整个方言。

## 1. constant 与二元运算

完整模块：

<!-- mlir-example: arith-affine-expression -->
```text
module {
  func.func @twice_plus_one(%x: i32) -> i32 {
    %two = arith.constant 2 : i32
    %one = arith.constant 1 : i32
    %scaled = arith.muli %x, %two : i32
    %r = arith.addi %scaled, %one : i32
    return %r : i32
  }
}
```

constant 没有 operand，静态常量内容产生一个结果；addi/muli 各使用两个输入并产生一个结果。这里输入/结果同为 i32。Arith 很多操作也支持相应 vector/tensor 形式，但是否支持、形状和元素类型要满足哪些条件，须查该 Op 的定义，不能对所有操作一概而论。

## 2. signless integer 与运算解释

`i32` 是 signless integer。加法本身可以按固定宽度位运算理解；需要区分有符号与无符号时，由具体操作或谓词表达，例如 `divsi/divui`、`slt/ult`、`extsi/extui`。

完整模块用同一 i8 位模式展示两种比较：

<!-- mlir-example: arith-signed-unsigned -->
```text
module {
  func.func @compare_bits() -> (i1, i1) {
    %minus_one = arith.constant -1 : i8
    %zero = arith.constant 0 : i8
    %signed = arith.cmpi slt, %minus_one, %zero : i8
    %unsigned = arith.cmpi ult, %minus_one, %zero : i8
    return %signed, %unsigned : i1, i1
  }
}
```

`-1` 的 8 位模式是全 1。有符号解释为 -1，无符号解释为 255，所以两个比较分别为 true、false。结果差异由比较谓词引入，不是 `%minus_one` 运行时改变了 Type。

## 3. 比较产生条件值

`arith.cmpi` 的静态谓词规定比较规则，结果是 i1 条件。标量常用谓词如下：

| 谓词 | 解释 |
|---|---|
| `eq` / `ne` | 位值相等 / 不相等 |
| `slt` / `sle` / `sgt` / `sge` | 按有符号解释进行大小比较 |
| `ult` / `ule` / `ugt` / `uge` | 按无符号解释进行大小比较 |

`arith.cmpf` 用另一套浮点谓词，须考虑 NaN。比如 ordered 比较要求参与值不是 NaN；unordered 类比较会把无序情形纳入对应规则。不能把整数 `slt` 直接替换成浮点拼写，也不能假定浮点比较满足所有整数式逻辑变换。

## 4. 固定位宽溢出、poison 与 UB

不带溢出标志的 `arith.addi` 按固定位宽保留结果低位，即模 `2^N` 的加法。它不是无限精度整数加法；也不能直接套用 C 有符号溢出的语言规则。

`overflow<nsw>` 或 `overflow<nuw>` 为有符号/无符号不溢出提供约定；违背相应约定会产生 poison。添加这些标志需要证明，不能作为“想让编译器更快”的无依据开关。poison 是 IR 语义中的特殊状态，不等同于工具立即抛出异常。

除法有另外的前提：`divui` 除数不能为零；`divsi` 还涉及最小负数除以 -1 的溢出情形。未定义行为与产生 poison 是不同规则，应按具体操作规范阅读，不能统一讲成“结果截断”。

本系列整数执行表使用小数值，保证索引计算可表示；需要测试溢出行为时另给明确类型、标志和预期。

## 5. 浮点运算不能直接套用实数代数

`arith.addf`、`mulf` 等遵循相应浮点与标志语义。有限精度舍入、NaN、无穷、正负零会影响等价变换。

例如将 `(a+b)+c` 改成 `a+(b+c)`，数学实数下恒等，浮点下可能改变舍入结果。带 fastmath 标志时允许的变换依赖具体标志；不能把“允许某个假设”推成“所有数学恒等式都可使用”。

这会直接影响后续 Softmax/attention：归约顺序、近似指数、累加精度和容差都是语义/数值设计的一部分。当前先建立边界，后续在具体算子项目中测量误差。

## 6. select 选择值，if 选择执行区域

完整模块中的两个候选值已经在 select 前可用：

<!-- mlir-example: arith-select -->
```text
module {
  func.func @max_signed(%a: i32, %b: i32) -> i32 {
    %take_a = arith.cmpi sge, %a, %b : i32
    %r = arith.select %take_a, %a, %b : i32
    return %r : i32
  }
}
```

select 没有 then/else Region。它根据条件选择两个 SSA operand 中的一个，不会把产生这两个 operand 的操作自动变成按需执行。

如果候选值来自有副作用、可能除零或不能随意执行的计算，就不能简单把 `scf.if` 两个分支提前算完再 select。[SCF if](./scf/if) 和[效果模型](../core/effects)给出具体解释。select 对未选中 poison operand 的规则也不能挽救先前已执行操作产生的即时 UB。

## 7. cast 表达明确的转换选择

| 操作族 | 含义 | 需要注意 |
|---|---|---|
| `extsi` / `extui` | 符号扩展 / 零扩展整数 | 扩展结果取决于选定解释 |
| `trunci` | 截断整数位宽 | 信息可能丢失 |
| `sitofp` / `uitofp` | 按有符号/无符号数值转浮点 | 可能发生舍入 |
| `fptosi` / `fptoui` | 浮点转整数 | 范围与异常输入需查契约 |
| `index_cast` 等 | index 与整数表示之间转换 | 目标位宽、扩展/截断规则要明确 |
| `bitcast` | 兼容位宽下重新解释位模式 | 不等于数值转换 |

本章用于辨认和选择语义；每种边界值的完整测试放在使用该转换的后续章节，不宣称已经覆盖所有 cast 细节。

## 8. 从操作定义读到优化实现

Arith 的 ODS 定义通常先列类型约束、trait/interface，再描述数值规则和 assemblyFormat。常量折叠（fold）利用已知输入推导结果，但折叠必须服从当前类型、溢出和浮点标志。

读源码时可以选择一个闭合问题：例如“同为 i8 的 slt 和 ult 为什么折叠成不同 i1”。先读 CmpIOp 契约，再找 C++ fold 中谓词和 APInt 比较的分支，最后找常量折叠测试。不要从 Arith.cpp 第一行顺读到最后一行。

## 理解检查

解释 i32 为什么不等于所有运算都按 signed 解释；指出 addi 的普通溢出与带 nsw 的区别；说明浮点重结合为何需要额外许可；说明 select 为什么不能保护先前已执行的除零。

依据：[ArithOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)、[ArithOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Arith/IR/ArithOps.cpp)、[Arith 测试](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/test/Dialect/Arith)。完整模块参与 P/G；比较结果是规范推演，验证脚本另检查其常量化结果。

下一章：[CF：显式控制流](./cf)。
