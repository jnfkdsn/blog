---
order: 1
title: 贯通教程：从算法读懂 MLIR
updated: 2026-09-06
---

# 贯通教程：从算法读懂 MLIR

前置：[学习路径](../learning_path)中的核心概念、基础方言与 SCF。本文用一个完整算法串起已经定义的机制，不逐章重复定义。目标是学会从算法、结构、数据流、执行与变换五个角度阅读陌生 IR。

版本：LLVM 20.1.8。完整模块经过解析/验证；后面的 CF 程序来自本地工具真实转换。文中的手工执行表不是 CPU/GPU/NPU 机器码运行结果。

## 1. 先确定算法与有效输入

算法遍历一维整数数组，只把严格大于 0 的元素累加起来。先用数学整数理解意图：

```text
sum = 0
for i in [0, length(input)):
    if input[i] > 0:
        sum = sum + input[i]
return sum
```

本教程假设输入指向有效存储，执行期间没有其他线程并发修改它，维度与索引运算可表示，并且正数之和不超出 i32 有符号范围。这样算法的数学求和与这里不带溢出标志的 i32 加法结果一致。超出该求和范围时，IR 中 addi 仍按其固定位宽语义解释，不能继续把它称为无界整数求和。

空输入的结果为 0；负数和 0 不增加结果。这些边界先由算法确定，再检查 IR 是否真的表达了它们。

## 2. 完整 MLIR 程序

<!-- mlir-example: tutorial-sum-positive -->
```text
module {
  func.func @sum_positive(%input: memref<?xi32>) -> i32 {
    %zero_i = arith.constant 0 : index
    %one_i = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %n = memref.dim %input, %zero_i : memref<?xi32>
    %sum = scf.for %i = %zero_i to %n step %one_i
        iter_args(%acc = %zero) -> (i32) {
      %x = memref.load %input[%i] : memref<?xi32>
      %positive = arith.cmpi sgt, %x, %zero : i32
      %next = scf.if %positive -> (i32) {
        %added = arith.addi %acc, %x : i32
        scf.yield %added : i32
      } else {
        scf.yield %acc : i32
      }
      scf.yield %next : i32
    }
    return %sum : i32
  }
}
```

这里同时使用 builtin、func、arith、memref、scf。它们在同一程序中分别表达容器、函数、标量计算、可变存储访问与结构化控制流，组合并不矛盾。

选择 MemRef 是为了显式展示运行时存储读取；这不是从某个 tensor 方言自动 bufferize 得到的输出。后续张量课程会展示同一计算如何在更高层表示与内存表示之间转换。

## 3. 第一遍：画静态包含结构

```text
module Op
└── Region → Block
    └── func.func @sum_positive
        └── 函数体 Region → 入口 Block（参数 %input）
            ├── 3 个 constant
            ├── memref.dim → %n
            ├── scf.for → %sum
            │   └── body Region → Block（参数 %i、%acc）
            │       ├── memref.load → %x
            │       ├── arith.cmpi → %positive
            │       ├── scf.if → %next
            │       │   ├── then Region → Block → addi、yield
            │       │   └── else Region → Block → yield
            │       └── scf.yield %next
            └── func.return %sum
```

函数体只有一个同层 Block；for body 也只有一个 Block。运行时存在循环与条件，不要求函数在当前表示下就拥有多个同层 Block，因为控制行为由两个 SCF Operation 封装。

图中没有函数调用；`func.func` 是定义。函数参数也不是一个“读取参数”的 Operation，它是入口 Block 的参数。

## 4. 第二遍：列输入、状态与输出

| Value | 定义来源 | 角色 | 可用范围 |
|---|---|---|---|
| `%input` | 函数入口 BlockArgument | 输入存储引用 | 函数内部，满足嵌套约束的位置 |
| `%n` | memref.dim 的 OpResult | 运行时长度 | 后续 for 及内部可捕获位置 |
| `%i` | for body BlockArgument | 本轮索引 | for body 内 |
| `%acc` | for body BlockArgument | 处理前 i 个元素后的状态 | for body 内及合法嵌套区域 |
| `%x` | load 的 OpResult | 本轮读到的标量 | 本轮 body 内的后续使用 |
| `%positive` | cmpi 的 OpResult | i1 分支条件 | if 的 operand |
| `%added` | then 中 addi 的 OpResult | 正数路径的新和 | then 内；经 yield 传出 |
| `%next` | if 的 OpResult | 本轮处理后的状态 | for body 中 if 后的位置 |
| `%sum` | for 的 OpResult | 整个遍历的结果 | for 后、函数 return |

这里有三种特别容易混淆的“结果”：then 的 `%added`、if 的 `%next`、for 的 `%sum`。它们是三个不同层级的 Value，靠各层传值协议连接，不能互相替换名字绕过作用域。

`%n` 是 for 的上界 operand，却不是 body 的 BlockArgument。for 负责解释上界，不会把每个 operand 都原样变成一个入口参数。

## 5. 第三遍：逐轮执行

令输入为 `[3, -2, 0, 5]`，n=4：

| i | 进入时 acc | load 的 x | positive | if yield 的值，即 next | for yield 后 |
|---:|---:|---:|---|---:|---|
| 0 | 0 | 3 | true | 3 | 下一轮 acc=3 |
| 1 | 3 | -2 | false | 3 | 下一轮 acc=3 |
| 2 | 3 | 0 | false | 3 | 下一轮 acc=3 |
| 3 | 3 | 5 | true | 8 | 循环结束，sum=8 |

else 不用 addi，而是把当前 `%acc` 原样交给 if。循环携带值允许下一轮等于本轮，不要求每轮都产生不同数值。

if then 的 yield 只结束 then；if 完成后，for body 继续执行自己的 yield。最后一轮结束才得到 `%sum`，再由函数 return 交给调用方。

## 6. 用不变量解释算法是否对

循环不变量是：进入索引 i 的 body 时，acc 等于 input 的前 i 个元素中所有正数之和。

1. 初始化：i=0，尚未处理元素，初始 acc=0，命题成立。
2. 保持：读取 input[i]。若它为正，则加到 acc；否则保持 acc。yield 的 next 因此等于前 i+1 个元素的正数和。
3. 退出：当 i 到达 n，不再进入 body；for 的结果为前 n 个元素的正数和。

在给定有效输入前提下，这段推理说明了算法与 IR 的对应。Verifier 只知道结构/类型等约束，不会自动证明这条求和不变量。

这个方法也适用于更复杂的归约和 attention：先说清每个循环状态代表哪一部分计算，再分析每次更新是否保持含义。项目变复杂时需要更完整的数值、并行和内存证明。

## 7. 边界与错误修改

| 情形或修改 | 应如何解释 |
|---|---|
| n=0 | for 不进入 body，无 load，无 if；sum 直接等于初始 0 |
| 所有值非正 | 每轮选择 else，状态始终为 0 |
| 把 `sgt` 改为 `sge` | 0 也走 then；本例对结果不变，但执行路径不同，不能一般化到有其他行为的分支 |
| 把比较改为 `ugt` | 负整数位模式可能被当作大的无符号数，改变算法 |
| for 最后 yield `%acc` 而非 `%next` | 类型合法，却丢弃每轮更新，结果保持初始 0 |
| return `%added` | 内层 Value 越过 Region 作用域，非法 |
| 把 load 提到 for 外 | 索引不可用；零轮也可能引入原本没有的读取，不能直接这样做 |

“类型一致”“例子算对”“对所有有效输入正确”是不同强度的结论。讲优化时需要说明是哪一种。

## 8. 转为 CF 后继续追同一条数据流

下面是本地 `mlir-opt --convert-scf-to-cf` 对第 2 节模块的输出。保留工具生成的名字和前驱注释，没有额外 canonicalization：

<!-- mlir-example: tutorial-sum-positive-cf -->
```text
module {
  func.func @sum_positive(%arg0: memref<?xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?xi32>
    cf.br ^bb1(%c0, %c0_i32 : index, i32)
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb6
    %2 = arith.cmpi slt, %0, %dim : index
    cf.cond_br %2, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %3 = memref.load %arg0[%0] : memref<?xi32>
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    cf.cond_br %4, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %5 = arith.addi %1, %3 : i32
    cf.br ^bb5(%5 : i32)
  ^bb4:  // pred: ^bb2
    cf.br ^bb5(%1 : i32)
  ^bb5(%6: i32):  // 2 preds: ^bb3, ^bb4
    cf.br ^bb6
  ^bb6:  // pred: ^bb5
    %7 = arith.addi %0, %c1 : index
    cf.br ^bb1(%7, %6 : index, i32)
  ^bb7:  // pred: ^bb1
    return %1 : i32
  }
}
```

现在函数 body 中出现多个同层 Block。按语义给它们取阅读别名：

| 工具标签 | 阅读角色 | 与原 SCF 的关系 |
|---|---|---|
| 入口 | 准备初值 | 给 header 传入 0 索引和 0 状态 |
| `^bb1(%0, %1)` | loop header | `%0` 对应 i，`%1` 对应 acc；判断范围 |
| `^bb2` | 本轮 load 与正负判断 | 对应 for body 的前半部分 |
| `^bb3` / `^bb4` | then / else | 分别传更新值或旧状态给合流块 |
| `^bb5(%6)` | if merge | `%6` 对应 if 的 next |
| `^bb6` | latch / 回边 | 更新索引，把 `%6` 作为下一轮状态 |
| `^bb7` | exit | 返回 header 的当前状态 |

结构化 for 的隐式回边变成 `cf.br`；if 的结果变成合流 Block 参数；for 最终结果的使用变成 exit 中对 header 状态的使用。

保留 `^bb5 → ^bb6` 这样可进一步简化的边仍然合法。转换正确性不要求立即获得最少 Block；不同版本或额外 Pass 可能选择不同排版/命名/简化结果。

## 9. 降低了什么，还没有降低什么

本次转换消去了 SCF，将结构化控制协议表达为 CF。memref.load/dim、arith、func 仍在。这是多层表示中的一次局部降低，不是“生成机器码完成”。

要继续执行，还需要选择目标、转换剩余类型/操作、处理函数 ABI、运行时与入口数据。GPU/NPU 还涉及线程映射、内存空间、同步和目标工具链。本阶段先确认每个表示中的算法含义，不提前把设备问题混入对象模型。

## 10. 从教程返回知识库与源码

遇到问题先定位所属机制：不懂 `%next` 的作用域查 SSA；不懂两个 yield 的接收方查 SCF；不懂 buffer 与 SSA 查效果模型；不懂转换后 Block 数量则进入对应 conversion 源码。

追一次 lowering 的最短源码路径是：SCFOps.td 的 ForOp/IfOp 契约 → SCFToControlFlow.cpp 中 ForLowering/IfLowering → SCFToControlFlow 转换测试。阅读时回答“边界值怎样进入 header、yield 怎样变成分支实参、旧 result uses 怎样被替换”，当前无需完整读完所有 conversion。

## 阅读完成检查

关闭本页后，能否自行重画包含树和 CF 图，解释 n=0、负数输入、两层 yield，并说出一个 verifier 无法排除的错误修改？这些解释完成后，再讨论剩余疑问并安排正式最小 IR 实验。

依据：[SCFOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)、[SCFToControlFlow.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp)、[转换测试](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/test/Conversion/SCFToControlFlow)。正文算法、执行表与不变量分析为本教程的推演；转换输出来自本地 LLVM 20.1.8。

下一步：[阅读与验证 IR](../guides/inspecting_ir)，理解不同验证结果各自说明什么。
