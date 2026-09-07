---
order: 4
title: CF：显式控制流与沿边传值
updated: 2026-09-06
---

# CF：显式控制流与沿边传值

前置：[SSA 与支配](../core/values_ssa)、[func](./func)、[arith](./arith)。CF 的 branch 操作在同一个 SSACFG Region 内建立 Block 之间的跳转。本章展开 br/cond_br、合流和回边；switch/assert 留在后续操作参考。

## 1. Terminator 显式指定后继

| 操作 | 输入 | 后继 | SSA 结果 |
|---|---|---|---|
| `cf.br` | 传给目标参数的值 | 一个目标 Block | 无 |
| `cf.cond_br` | i1 条件、两条边各自的实参 | true/false 目标 | 无 |
| `func.return` | 函数结果值 | 无同层 Block 后继；交回调用方 | 无 |

branch 使用 Value、引用目标 Block，但没有 Region。它不会把目标 Block 包含进自己；目标 Block 与来源 Block 在同一 Region 内。

## 2. 条件与合流

完整模块：

<!-- mlir-example: cf-merge -->
```text
module {
  func.func @choose(%cond: i1, %a: i32, %b: i32) -> i32 {
    cf.cond_br %cond, ^left, ^right
  ^left:
    cf.br ^join(%a : i32)
  ^right:
    cf.br ^join(%b : i32)
  ^join(%r: i32):
    return %r : i32
  }
}
```

执行路径是 entry → left/right → join。join 参数 `%r` 的值由实际进入的边决定。`cf.cond_br` 自己没有选中值的 SSA result；合流结果由目标 BlockArgument 表示。

在本例中可以进一步简化 CFG，但原表示已经合法。canonicalization 是否选择某个具体输出形式，与该 IR 是否具有明确语义是两个问题。

## 3. 循环是 header、条件和回边的组合

完整模块将一个 i32 初值递增四次：

<!-- mlir-example: cf-loop -->
```text
module {
  func.func @increment_four(%initial: i32) -> i32 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %one = arith.constant 1 : i32
    cf.br ^header(%lb, %initial : index, i32)
  ^header(%i: index, %acc: i32):
    %more = arith.cmpi slt, %i, %ub : index
    cf.cond_br %more, ^body, ^exit
  ^body:
    %next = arith.addi %acc, %one : i32
    %next_i = arith.addi %i, %step : index
    cf.br ^header(%next_i, %next : index, i32)
  ^exit:
    return %acc : i32
  }
}
```

| 进入 header 的次数 | 参数 `%i` | 参数 `%acc`（initial=10） | 下一步 |
|---|---:|---:|---|
| 1 | 0 | 10 | body 计算 11，再回边 |
| 2 | 1 | 11 | body 计算 12，再回边 |
| 3 | 2 | 12 | body 计算 13，再回边 |
| 4 | 3 | 13 | body 计算 14，再回边 |
| 5 | 4 | 14 | 条件假，exit 返回 14 |

exit 中能使用 `%acc`，因为 header 支配 exit，且 `%acc` 是 header 的参数。body 中 `%next` 的定义并不支配所有退出路径：如果第一次条件就为假，body 根本不执行。这正是退出使用“当前状态”而非“body 新结果”的原因。

若把 ub 改为 0，第一次 header 就转 exit，结果是 `%initial`。结构化 for 的零次迭代规则与此相呼应。

## 4. 结构和类型约束

分支目标必须与当前分支位于同一 Region；不能用 cf.br 直接跳进 scf.if 的内部区域或另一个函数。入口 Block 也不能成为本 Region 内分支的目标；需要回边时使用独立 header。

每个合法路径必须有明确 terminator。Block 的文字排列并不产生隐式 fallthrough；把两个标签写在相邻位置不会自动建立边。

下面是参数类型不匹配的失败模块：

<!-- mlir-invalid: cf-argument-type | type mismatch -->
```text
module {
  func.func @bad_branch(%x: i32) -> i64 {
    cf.br ^target(%x : i32)
  ^target(%arg: i64):
    return %arg : i64
  }
}
```

branch operand 自身是合法 i32，target 参数自身是合法 i64，组合却违反沿边传值约束。这不是“所有 Value 都有类型”就足以保证的局部性质。

## 5. CF 与 SCF 的表达取舍

CF 把控制流写成显式边，适合表达一般 CFG 和继续降低到低层表示。循环边界、步长和归约结构不再都集中在一个 Op 的字段中，优化往往需要分析 CFG 才能恢复。

SCF 把这些协议封装为带 Region 的 Operation，保留“它是一个循环/条件结构”的信息。因此高层变换常愿意在 SCF 或更高层表示上完成，再 lower 到 CF。并非所有编译器必须严格依次经过这些方言；实际 pipeline 由目标任务选择。

## 理解检查与源码

独立画出循环的四个 Block 和五次 header 绑定，解释零轮为何有结果；说明为什么不能直接返回 `%next`，为什么入口 Block 不能被当作普通回边目标。

依据：[ControlFlowOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td)、[ControlFlowOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp)、[LangRef](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/LangRef.md)。源码先查 BranchOp/CondBranchOp 的 successor/operand 约束，再看 BranchOpInterface；不要求本阶段实现 CFG 分析。完整模块参与 P/G 或失败诊断检查，执行表是语义推演。

下一章：[结构化控制流 SCF](./scf/)。
