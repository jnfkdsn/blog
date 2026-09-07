---
order: 3
title: SCF While：两区域循环协议
updated: 2026-09-06
---

# SCF While：两区域循环协议

前置：[SCF for](./for)。While 不通过固定 lb/ub/step 推导是否继续，而由区域中的计算产生条件。本文解释 before/after、condition/yield、两套参数类型、首次判假和 do-while 形式；不展开循环优化算法。

## 1. 两个区域和两个终结操作

`scf.while` 拥有 before 和 after 两个 Region，各有一个 Block。输入首先绑定到 before 参数。before 执行后，由 `scf.condition` 根据 i1 条件决定下一步。

```text
while operands / initial values
             │
             ▼
       before Block 参数 ◀──────────────┐
             │                          │
       计算条件与传出值                  │
             │                          │
       scf.condition(%cond) values       │
             ├── false → while results  │
             └── true → after 参数      │
                            │           │
                         执行 after     │
                            │           │
                       scf.yield values ┘
```

condition 的第一个 operand 是控制条件；后面的 operands 才是传递的数据。条件为 true，它们传给 after；为 false，它们直接成为整个 while 的结果。after 的 yield 总是传回 before，不直接成为 while result。

因此不能沿用“for 最后一轮 yield 就是最终结果”的表述来解释 while。while 的退出值由最后一次 condition 提供。

## 2. 普通 while：先判断再执行主体

完整模块从 0 计数到 limit，使用可表示的小范围输入：

<!-- mlir-example: while-count -->
```text
module {
  func.func @count_to(%limit: index) -> index {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result = scf.while (%i = %zero) : (index) -> index {
      %more = arith.cmpi slt, %i, %limit : index
      scf.condition(%more) %i : index
    } do {
    ^bb0(%current: index):
      %next = arith.addi %current, %one : index
      scf.yield %next : index
    }
    return %result : index
  }
}
```

`%i` 是 before 参数，`%current` 是 after 参数，它们是不同 Value。第一个 `%i` 来自 zero；after 中算出的 next 交给下一次 before。

当 limit=2：

| before 的 `%i` | condition | condition 传出 `%i` 的接收方 | after 的行为 |
|---:|---|---|---|
| 0 | true | after `%current`=0 | 算出 1，yield 给 before |
| 1 | true | after `%current`=1 | 算出 2，yield 给 before |
| 2 | false | while `%result`=2 | 不执行 after |

若首次就判假，after 执行零次，但 before 已经执行一次。不能笼统说“while 的两个 Region 都执行零次”。如果 before 在判断之前做了其他计算或效果，它们也已经发生。

## 3. 两套类型分别闭合

While 可以有与输入不同的结果类型。把类型列表记为 A 和 B：

| A 组，必须对应 | B 组，必须对应 |
|---|---|
| while operands | while results |
| before Block 参数 | after Block 参数 |
| after 的 yield operands | condition 除 i1 条件外的 operands |

同一组内部数量与类型按位置对应，A 与 B 不必相同。before 可以从 A 计算出 B，after 再从 B 计算出下一轮 A。

下面的完整模块以 index 保存 before 状态，用 i32 作为 after 状态和最终结果。它只用于展示类型协议，假设 limit 在 0—100 范围内：

<!-- mlir-example: while-different-types -->
```text
module {
  func.func @count_as_i32(%limit: index) -> i32 {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %result = scf.while (%i = %zero) : (index) -> i32 {
      %current_i32 = arith.index_cast %i : index to i32
      %more = arith.cmpi slt, %i, %limit : index
      scf.condition(%more) %current_i32 : i32
    } do {
    ^bb0(%current: i32):
      %next_i32 = arith.addi %current, %one : i32
      %next = arith.index_cast %next_i32 : i32 to index
      scf.yield %next : index
    }
    return %result : i32
  }
}
```

如果把 after 的 yield 改成 i32，它虽然与 after 参数相同，却不符合要回到的 before 参数类型。检查类型时必须顺着控制去向看接收方，不能只比较附近的声明。

这种机制也允许 before 同时计算条件与 after 要使用的中间量，以参数传过去，避免在 after 重算。两个 Region 是兄弟区域，不能直接引用另一边内部定义的 SSA Value 绕过协议。

## 4. do-while：主体在 before 中

`before/after` 的命名是相对于条件检查。把主体计算放在 before 中，就能表达至少执行一次主体的 do-while：

<!-- mlir-example: while-do-while -->
```text
module {
  func.func @increment_at_least_once(%limit: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %result = scf.while (%current = %zero) : (i32) -> i32 {
      %next = arith.addi %current, %one : i32
      %more = arith.cmpi slt, %next, %limit : i32
      scf.condition(%more) %next : i32
    } do {
    ^bb0(%forwarded: i32):
      scf.yield %forwarded : i32
    }
    return %result : i32
  }
}
```

limit=0 时 before 仍先把 0 加成 1，再判假并返回 1。limit=3 时依次计算 1、2、3，最后返回 3。这里 after 只负责把数据传回 before，不承载主体计算。

这也解释为什么 while 不能凭关键词直接翻译成某一种 C while 模板：先看两个区域分别做什么，再还原普通 while 或 do-while 的程序。

## 5. 作用域、终止与 CF 对照

外层 limit、one 可由两个区域捕获，前提是满足层次支配及操作约束。before 的局部 next 只能通过 condition 传到 after；after 的局部结果只能通过 yield 回到 before。

条件可能永远为真，或者更新没有朝终止方向推进。Verifier 可以检查结构与类型，不会普遍证明循环终止。即使没有内存效果，移动或删除一个可能不终止的循环也涉及额外语义条件。

CF 对照可按四类 Block 理解：入口把初值传给 before；before 计算后条件跳到 after 或 exit，分别传 B；after 通过无条件分支把 A 传回 before；exit 的参数承接 B 并供外部使用。具体 pass 可以折叠多余 Block，但两套类型关系必须保留。

## 理解检查

1. 第一次 condition 为 false 时，哪些区域已经执行？while result 从哪里来？
2. 为何 after 的 yield 类型对应 while 输入，而不是 while 输出？
3. 第三个模块 limit=0 时为什么返回 1？
4. 把 before 的局部 SSA Value 直接拿到 after 使用，错在哪里？

<details>
<summary>核对要点</summary>

before 执行过，condition 的 trailing operands 成为结果；yield 回到 before，必须匹配其参数；主体加法在条件检查前；兄弟 Region 的局部值不能直接越过作用域，需按参数协议传递。

</details>

依据：[SCFOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td) 的 WhileOp/ConditionOp/YieldOp；[SCF.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/SCF/IR/SCF.cpp) 的 WhileOp verifier；[SCFToControlFlow.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp)。三个模块参与 P/G/L，推演表未作为机器执行报告。

继续阅读：[内存、效果与优化边界](../../core/effects)，然后进入[贯通教程](../../tutorials/reading_ir)。
