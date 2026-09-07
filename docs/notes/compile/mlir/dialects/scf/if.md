---
order: 1
title: SCF If：条件区域与结果
updated: 2026-09-06
---

# SCF If：条件区域与结果

前置：[SCF 概览](./)、[SSA 与支配](../../core/values_ssa)、[arith](../arith)。本文展开 if 的输入、区域、结果、捕获、无结果形式及与 CF/select 的关系。

## 1. 操作契约

`scf.if` 消费一个 i1 条件，可以产生零个或多个结果。它拥有 then 和 else 两个 Region。then 必须有一个 Block；else 在存在时有一个 Block，允许省略的形式中为空 Region。分支 Block 没有参数，读取条件之外的值通常通过合法外层捕获完成。

有结果时，两个分支都必须提供匹配结果类型/数量的 `scf.yield`；无结果时可以省略 else，分支用空 yield 终结，简洁语法还可以省略可隐式补出的空 yield。

if 的两个区域不会在普通顺序语义中都执行：条件为 true 执行 then，为 false 执行存在的 else；无 else 且条件为 false 时跳过内部计算。

## 2. 由所选分支形成结果

完整模块：

<!-- mlir-example: if-result -->
```text
module {
  func.func @adjust(%take_add: i1, %x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %result = scf.if %take_add -> (i32) {
      %a = arith.addi %x, %one : i32
      scf.yield %a : i32
    } else {
      %b = arith.subi %x, %one : i32
      scf.yield %b : i32
    }
    return %result : i32
  }
}
```

假设 x=10：

| 条件 | 执行区域 | 局部结果 | yield 接收方 | 函数返回 |
|---|---|---:|---|---:|
| true | then | `%a`=11 | if 形成 `%result` | 11 |
| false | else | `%b`=9 | if 形成 `%result` | 9 |

`%a/%b` 不是同一个 Value 的两个赋值，也不能在 if 外直接引用。`%result` 是父 if 的结果。`scf.yield` 把控制和值交还当前 if，后续仍执行函数中的 `func.return`。

then/else 可以捕获 `%x/%one`，因为它们在外层合法可用且 if 不施加函数那样的隔离约束。分支没有 Block 参数，并不意味着它不能使用任何输入。

## 3. 多结果是逐项对应

完整模块按条件重新安排两个输入：

<!-- mlir-example: if-multiple -->
```text
module {
  func.func @maybe_swap(%swap: i1, %a: i32, %b: i32) -> (i32, i32) {
    %r:2 = scf.if %swap -> (i32, i32) {
      scf.yield %b, %a : i32, i32
    } else {
      scf.yield %a, %b : i32, i32
    }
    return %r#0, %r#1 : i32, i32
  }
}
```

两个分支均提供两个 i32，按位置对应两个 if 结果。这里没有隐式元组构造。静态类型同样的两项互换可能保持 verifier 通过，却改变程序语义。

## 4. 无结果的 if 可以只有 then

完整模块在条件成立时写 buffer：

<!-- mlir-example: if-no-result -->
```text
module {
  func.func @maybe_store(%enabled: i1, %buffer: memref<1xi32>, %x: i32) {
    %zero = arith.constant 0 : index
    scf.if %enabled {
      memref.store %x, %buffer[%zero] : memref<1xi32>
      scf.yield
    }
    return
  }
}
```

条件为 false 时没有值需要选出，也没有写入，所以 else 可以为空。无结果不意味着无行为：如果删除这个 if，就删除了可能发生的存储效果。

如果 if 有 i32 结果却没有 else，false 路径就没有结果来源，因此不合法。下面保留一个不同的错误：两个分支提供不同类型。

<!-- mlir-invalid: if-yield-type | should match input type -->
```text
module {
  func.func @bad_if(%cond: i1, %x: i32, %y: i64) -> i32 {
    %r = scf.if %cond -> (i32) {
      scf.yield %x : i32
    } else {
      scf.yield %y : i64
    }
    return %r : i32
  }
}
```

这里没有隐式 i64→i32 截断；必须明确决定转换语义，再显式表示。

## 5. 嵌套时按直接所属区域解释 yield

for body 中可以包含 if，if then 中还可包含另一个 if。每个区域有自己的 terminator：内层 yield 先给内层 if 形成结果；外层继续运行，最后由外层 yield 给外层结构传值。

下面是概念片段，不可单独解析：

```text
for 的 body:
  %selected = scf.if ... {
    scf.yield %branch_value   // 交给 if
  } else { ... }
  scf.yield %selected        // 交给 for
```

不能把内层 yield 解释成“跳出所有循环”或“从函数返回”。需要提前返回或一般 CFG 时，应选择支持该语义的表示/变换，不能在 if 区域里任意用 func.return 替代合法 terminator。

## 6. Lowering 到 CF 的对应

概念结构为：外层 Block 中计算条件 → `cf.cond_br` → then/else 同层 Block → 携带值跳到 merge → 使用 merge BlockArgument。

| SCF | CF 对应 |
|---|---|
| if 的条件 operand | cond_br 的条件 |
| 两个区域的计算 | 两组同层 Block 中的计算 |
| 分支 yield operand | 跳到 merge 的边参数 |
| if result | merge 参数及其后续使用 |

这解释了结构化结果如何变成 CFG 合流。实际 pass 可能简化某些 Block 或直接选择更简单形式；不要用固定 Block 名字定义转换正确性。[贯通教程](../../tutorials/reading_ir)会把 if 嵌在 for 中跟踪这条关系。

## 7. 与 select 的边界

if 选择执行区域；select 选择已经可用的值。若把两边计算都挪到 if 之前，必须证明扩大执行范围仍安全，并考虑成本。整数除法、可能越界的 load、store 和可能不终止的计算不能随意提前；详见[效果模型](../../core/effects)。

即使某次优化将 if 化简为 select，也不能反推两种操作对所有程序都可无条件互换。

## 理解检查与源码

能解释无结果 if 的 false 路径、有结果 if 为什么需要 else、多结果对应、嵌套 yield 的归属，以及 if/select 的执行差别，即完成本章 D2 阅读目标。

依据：[SCFOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td) 的 IfOp/YieldOp；[SCF.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/SCF/IR/SCF.cpp) 中 IfOp/YieldOp verifier；[SCFToControlFlow.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp) 的 IfLowering。完整合法模块参与 P/G/L，错误模块参与 N，执行表不是机器执行结果。

下一章：[For：循环状态与结果](./for)。
