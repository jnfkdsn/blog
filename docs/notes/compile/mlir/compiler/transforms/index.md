---
order: 10
title: IR 变换基础
updated: 2026-09-13
excludeFromSidebar: true
---

# IR 变换基础

本模块把 Pass、C++ IR API、PatternRewriter 和 driver 放回同一个执行过程：**一份 IR 进入编译器后，一条局部改写怎样被找到、应用，并形成可检查的结果。** 初读时用这篇总览定位各章，读完后沿调用链回顾；具体算法与 API 在对应章节展开。

## 从你已经运行过的命令开始

[小 Pass 教程](../../tutorials/first_pass)中使用了这条命令，在 workspace 根目录运行：

```bash
artifacts/builds/mlir-small-pass/lab-opt \
  aicompiler-labs/llvm-mlir/04-small-pass/input.mlir \
  --pass-pipeline='builtin.module(func.func(lab-remove-add-zero))' \
  --verify-each
```

输入文件的主要计算是先得到 `%a = x + 0`，再得到 `%r = %a + x`，最后返回 `%r`。变换后，剩余加法的两个输入都指向函数参数 x。这里改变的是编译器进程中的 IR 对象；工具并没有接收一个具体整数并执行 `twice` 函数。

要得到这项变化，需要完成不同层次的工作。命令里的 Pass 名称负责选择变换，`func.func` 指定它的运行层级，而“这条加法右边是零，所以可以用左边替代”是变换内部的一条局部规则。

## 沿一次修改走过各层

```text
lab-opt：注册方言和 Pass，读取文件并构造 IR
  ↓
PassManager：按 pipeline 安排函数层级的 Pass
  ↓
runOnOperation()：取得当前函数，准备规则集合
  ↓
driver：选择待处理 Operation，尝试候选 Pattern
  ↓
Pattern：检查操作类型、operand 和静态信息
  ↓
PatternRewriter：更新使用关系、删除旧操作，通知 driver
  ↓
driver：按策略继续处理或结束
  ↓
Pass 返回；PassManager 验证 IR；工具打印结果
```

先看最外层。`lab-opt` 提供解析和打印等工具能力，Pass 注册使 pipeline 能通过 `lab-remove-add-zero` 找到相应的构造方法。PassManager 根据嵌套层级，把所选函数交给函数 Pass；这就是 `runOnOperation()` 中 `getOperation()` 的来源。

进入函数 Pass 后，算法还需要找到具体的加法。本例把这项工作交给 driver：它在函数内部选择操作，尝试以加法为 root 的 Pattern。Pattern 随后检查 RHS 是否由零常量定义，以及类型和 flags 是否满足当前规则的范围。

匹配成功后，rewriter 把旧结果的全部 use 接到左 operand，并删除旧加法。这个动作在对象层面涉及上一章 IR API 中的 Value、OpOperand、use-list 和 Operation 生命周期；在改写框架中还要发送修改通知，使 driver 能维护自己的处理记录。

最后，driver 决定是否继续。Greedy 可以重新处理受影响或新建的操作；Walk 执行一次遍历。它们应用的是 Pattern。PassManager 则在外层安排整个 Pass，pipeline 的顺序与 driver 的规则应用顺序因此属于两层调度。

本例中，Pattern 删除了加零操作，Greedy 的简单 DCE 清理了无用的零常量。随后 `--verify-each` 检查输出 IR 的合法性，FileCheck 测试再检查是否形成了约定的数据关系。两种证据共同帮助定位实现是否符合目标；数值语义仍需要规则本身的论证。

## 每一章补上了哪一段

| 阅读顺序 | 章节 | 放回执行链中的作用 |
|---|---|---|
| 1 | [Pass 与 pipeline](./passes) | 解释变换怎样组织、运行在什么对象上，以及嵌套、分析有效性和失败 |
| 2 | [C++ IR API](./ir_api) | 解释输入、结果、使用与拥有关系，以及创建、替换、删除怎样改变对象 |
| 3 | [PatternRewriter](./rewriting) | 将局部变换拆为匹配与修改，解释为什么修改要经过 rewriter |
| 4 | [改写驱动](./rewrite_drivers) | 解释多条规则怎样协作，以及 Walk/Greedy、fold、优先级和收敛的行为 |
| 5 | [实现并测试一个小 Pass](../../tutorials/first_pass) | 将规则、函数 Pass、注册、工具和 lit/FileCheck 接成完整工程 |

第一次学习可以先读驱动篇的规则协作与真实轨迹，再进入小 Pass 工程；benefit、fold 干扰和停止预算等细节留作深入查阅。完整 C++ 类定义可在需要实现时回查，不要求记住全部接口。

`first_pass.md` 位于 `tutorials/`，因为它将多个机制组成一个完整案例。本模块通过链接安排阅读顺序，机制正文仍各自维护主要定义。

## 哪些可以复用，哪些仍由具体算法决定

这套基础支持你在 MLIR 上实现编译器变换。例如算子分解可以用 Pattern 表达局部展开；融合和循环优化可能先使用分析推导合法性，再修改 IR；lowering 可以通过一组 Pass 分阶段改变表示。具体工作会变复杂，但运行范围、对象修改、注册与回归测试仍有共同基础。

一个 Pass 也可以直接遍历、调用分析或实现专门算法，无需把所有逻辑都写成 Pattern。Pattern 是组织局部改写的一种方式，driver 为它提供应用策略。选择哪一种结构取决于算法需要的信息和处理过程。

框架不会替任意算法证明正确性。它能帮助组织调用、维护修改通知和检查 IR 约束；能否交换两次内存访问、移动一次计算、融合两个循环，还需要相应的语义、依赖或数值分析。后续分析与优化章节会逐步补充这些依据。

## 遇到问题时沿哪一层回查

| 观察到的现象 | 先检查什么 | 阅读入口 |
|---|---|---|
| pipeline 不认识 Pass 名称 | 实现是否编译链接，注册函数是否执行 | [小 Pass 的注册与工具入口](../../tutorials/first_pass) |
| Pass 不能放到当前 pipeline 层级 | Pass 的操作类型与所在 PassManager 是否一致 | [Pass 与 pipeline](./passes) |
| Pass 运行了，目标操作却没变化 | 处理范围、候选类型与匹配条件是否满足 | [PatternRewriter](./rewriting) |
| 第一处修改后，新机会没有继续处理 | driver 的重新处理策略、规则集合与修改通知 | [改写驱动](./rewrite_drivers) |
| 替换后 verifier 报错或访问旧句柄出错 | 类型、支配、作用域、使用关系与对象生命周期 | [C++ IR API](./ir_api) |
| IR 通过 verifier，结果仍不符合预期 | 改写是否满足语义前提，测试是否约束正确的数据关系 | [小 Pass 的测试与证据](../../tutorials/first_pass) |

这张表用于选择排查入口。比如“没有变化”也可能只是该输入已经处于不动点，需要结合实际输入判断，而不是直接认定 driver 出错。

## 从变换已有操作走向定义新操作

前面一直使用已经定义好的 `arith.addi`。接下来进入[定义 IR 抽象](../ir_definition/)：操作的输入、结果、属性、类型约束和访问器从哪里来，框架怎样验证它，通用变换又怎样获取它的语义信息。

两部分会在 `lab.clamp` 中会合：操作定义规定区间限制的含义与合法形式，已有的 Pattern/Pass 将它展开成 arith 运算。之后的 Trait / Interface、Dialect Conversion 与分析机制继续建立在这条链路上。

模块范围对应[覆盖表 M01、M04、M06 与 X02](../../coverage)，个人学习状态仍由[学习路径](../../learning_path)记录。总览串联已学机制，不额外增加阶段完成要求。
