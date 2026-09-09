---
order: 1
title: Pass 与 pipeline：组织一次 IR 变换
updated: 2026-09-08
---

# Pass 与 pipeline：组织一次 IR 变换

你已经能编写带分支和循环的 MLIR，也见过 SCF 转为 CF 后的样子。接下来把视角移到编译器这一边：一份 IR 进入工具后，谁负责改变它，怎样连续做几次处理，又怎样知道处理发生在哪个函数上？

我们从一段有冗余的小程序出发，先跟踪两次具体简化，再把函数增加到多个、放入嵌套 module，最后对照运行日志和源码理解调度。本章先建立 Pass 的运行模型，C++ 对象修改和完整 Pass 工程在后续章节展开。示例依据本地 LLVM 20.1.8。

## 1. 从一段可以简化的程序开始

下面的函数先算两遍 x+0，再把两次结果相加：

<!-- mlir-example: passes-base -->
```text
module {
  func.func @twice(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %a = arith.addi %x, %zero : i32
    %b = arith.addi %x, %zero : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}
```

假设 x=7，a 和 b 都是 7，r 为 14。更一般地，按这里不带溢出标志的 i32 运算语义，最终结果等价于直接计算 x+x。

现在先不考虑任何工具，人工观察有两条简化理由。

第一，a 和 b 使用相同输入、执行相同操作。第二次计算可以复用第一次的结果。第二，加零本身没有改变值，因此 a、b 的使用也可以直接改为使用 x。

这两条理由需要不同的信息：前者比较两个操作及其依赖，后者使用加法的具体语义。它们最终都改变同一份 IR 的定义与使用关系，但不必由同一段算法完成。

还要区分两个执行层次。运行这段函数时，输入是一个 i32，输出也是一个 i32；运行编译器简化时，输入是包含函数的 IR 对象，结果是经过处理的 IR。编译器不需要先知道 x=7，才能证明 x+0 可以化简。

## 2. 先运行 CSE：复用已经算过的值

CSE 是 Common Subexpression Elimination，即公共子表达式消除。对这个直线程序，可以把它理解为：遇到一个满足条件的计算时，检查前面是否已有等价、可复用的计算结果。

把上面的完整模块保存为 `twice.mlir`，在 workspace 根目录可以使用：

```bash
MLIR_OPT="$PWD/artifacts/builds/mlir-20.1.8/bin/mlir-opt"
"$MLIR_OPT" twice.mlir --pass-pipeline='builtin.module(func.func(cse))'
```

先关注最后的 `cse`，外面的括号稍后拆解。以下是本地工具的实际输出：

<!-- mlir-example: passes-cse-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.addi %arg0, %c0_i32 : i32
    %1 = arith.addi %0, %0 : i32
    return %1 : i32
  }
}
```

名字被 printer 改成了 `%arg0`、`%0`、`%1`。对照数据流，真正发生的是原来的 `%b` 被删除，而 `%r` 的第二个 operand 改为使用原来的 `%a`：

```text
处理前：a = addi(x, zero)      b = addi(x, zero)
                    \       /
                    r = addi(a, b)

处理后：a = addi(x, zero)
                    │ 使用两次
                    r = addi(a, a)
```

这里有一个具体的安全理由：a 与 b 在同一 Block 中使用相同 SSA 输入和相同运算语义，a 的定义又位于 b 以及替换后的使用点之前。保留 a 可以满足这些使用的支配要求。

CSE 并不只是比较打印出来的两行字符串。若某个输入引用的是另一个 Value，或会影响语义的属性不同，两条操作就不能仅凭名字相同而合并。若是两个 load，中间的 store 还可能让它们读到不同内容。前面学过的支配和效果信息，正是在这里成为变换的依据。

注意工具仍保留了 x+0。这次 CSE 找到了重复计算，并没有因此承担所有整数代数简化。现在正好可以让另一项处理接着工作。

## 3. 再运行 canonicalize：利用操作自身的简化规则

将两个 Pass 连起来：

```bash
"$MLIR_OPT" twice.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))'
```

其实际输出为：

<!-- mlir-example: passes-canonical-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

沿着上一步的 IR 看，canonicalize 将加零结果的使用替换为 x，最后一个加法变为 x+x。常量 zero 失去用途，也随简化过程被清理。这里描述的是输入/输出之间可解释的变化，不是在假定内部 worklist 必须按某个固定顺序访问这些节点。

canonicalize 是组织规范化简化的 Pass。具体操作和方言提供自己的折叠或改写规则，通用驱动负责反复尝试，直到当前配置下不再变化或达到处理上限。因此，它可以处理多种方言，也可能一次触发多种清理行为；它不保证找到所有数学等价式中的最优程序。

在本例中，单独运行 canonicalize 也能得到 x+x：

```bash
"$MLIR_OPT" twice.mlir \
  --pass-pipeline='builtin.module(func.func(canonicalize))'
```

这并不使 CSE 的职责消失。它说明不同 Pass 的效果可能重叠，而且特定输入可能不需要全部 Pass。不能用一个小例子推出“所有程序只用 canonicalize 即可”，也不能为了说明 pipeline 顺序而声称交换这两个 Pass 必然得到不同终态。

本例经过了三种可观察状态：

| 位置 | 剩余的加法 | 能解释的变化 |
|---|---|---|
| 原始 IR | x+0、x+0、a+b | 两次重复计算和加零 |
| CSE 后 | x+0、a+a | 复用已有结果 |
| canonicalize 后 | x+x | 消除加零并清理无用常量 |

之后每个 Pass 看到的都是前一步留下的状态。这是理解 pipeline 顺序的基础：前面的处理可能创造、消除或改变后面的处理机会。两种顺序是否产生不同结果，必须结合具体输入和具体 Pass 检查。

## 4. 从一次变换理解 Pass 的工作单位

刚才我们调用了两个名字：`cse` 和 `canonicalize`。在实现中，它们对应能够被框架调度的 Pass 对象。一次 Pass 运行以某个 Operation 为当前对象，在允许的范围内分析或处理它所包含的 IR。

在前面的 pipeline 中，当前对象是 `func.func @twice`，不是其中某一个 addi，也不是文本文件路径。CSE 自己遍历函数内部，决定哪些计算可以复用。PassManager 不负责替它判断 x+0 的含义。

可以把一次运行画成：

```text
当前对象：func.func @twice
             │
             ▼
        CSE.runOnOperation()
             │
       读取函数内部 IR
       利用支配、等价性和效果条件
       替换使用并删除冗余操作
             │
             ▼
        原函数对象中的 IR 已改变
```

图中的方法名会在源码部分看到；现在先记住，它是框架进入这个 Pass 实现的入口。

运行对象与内部遍历范围是两回事。函数级 Pass 可以处理函数内的循环和分支；模块级 Pass 也可能自己递归处理多个函数。具体处理哪些内部操作，由 Pass 算法决定。Pipeline 指定调度位置，不自动规定算法的每一步遍历。

Pass 也不一定改变 IR：它可以检查、打印，或者在当前输入上找不到可做的变换。成功且输出不变是正常情况。MLIR 中供 Pass 查询的 Analysis 则通常是单独管理的分析对象，并不是要求你在 pipeline 里先运行一个同名“分析 Pass”。第 9 节会用 CSE 说明这个区别。

## 5. 为什么需要 PassManager

如果编译器只有这两个固定处理，当然可以在 C++ 中手动依次调用两个函数。随着编译链增长，还会出现共同需求：给 Pass 配置选项，在合适的 IR 层级运行，记录前后状态，检查失败，并让后续处理取得仍然有效的分析信息。

如果每个编译器阶段都自己实现这些管理逻辑，很容易出现不一致。PassManager 把这些共同工作集中起来，而 Pass 保留自己的具体算法。

对单个函数，先使用下面这个概念过程理解它：

```text
取得当前函数
    ↓
记录 CSE 运行前的 IR
    ↓
调用 CSE 的实现
    ↓
处理分析有效性、必要的验证及成功/失败通知
    ↓ 成功
调用 canonicalize 的实现
    ↓
完成这个函数的处理序列
```

这里的“输入输出”通常是同一份内存 IR 被逐步修改，而不是每个 Pass 必须读一个新文件、再写一个新文件。IR dump 是观察窗口，文件不是 Pass 之间交接数据的必经介质。

**Pass 负责一项处理，PassManager 负责组织它的运行，pipeline 则描述采用哪些处理及其结构和顺序。** 有了这三个角色，再看括号语法就有了对应对象。

## 6. 增加函数与嵌套 module：括号到底选择了什么

把同一段计算放入三个函数。其中 f、g 直接位于外层 module，h 位于内层 module：

<!-- mlir-example: passes-nested -->
```text
module {
  func.func @f(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %a = arith.addi %x, %zero : i32
    %b = arith.addi %x, %zero : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
  func.func @g(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %a = arith.addi %x, %zero : i32
    %b = arith.addi %x, %zero : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
  module @inner {
    func.func @h(%x: i32) -> i32 {
      %zero = arith.constant 0 : i32
      %a = arith.addi %x, %zero : i32
      %b = arith.addi %x, %zero : i32
      %r = arith.addi %a, %b : i32
      return %r : i32
    }
  }
}
```

对应的包含树是：

```text
builtin.module
├── func.func @f
├── func.func @g
└── builtin.module @inner
    └── func.func @h
```

现在对这个模块运行：

```text
builtin.module(func.func(cse,canonicalize))
```

最外层 `builtin.module` 指定 pipeline 根对象的类型。里面的 `func.func(...)` 在该层直接包含的操作中安排函数级序列。因此 f、g 被处理，h 不在这一层的选择范围内。本地输出中 f、g 各剩一个 addi，h 仍有三个。

如果希望处理 h，需要沿它的实际包含层级继续嵌套：

```text
builtin.module(builtin.module(func.func(cse,canonicalize)))
```

这次处理的是内层 module 下的 h；外层 f、g 不会因为名字也是 func.func 就自动被这一条嵌套路径选中。实测结果是 h 剩一个 addi，f、g 各保留三个。

两部分都需要时，可以在外层序列中同时写出这两条路径：

```text
builtin.module(
  func.func(cse,canonicalize),
  builtin.module(func.func(cse,canonicalize))
)
```

括号表达的是 IR 的包含层次，与函数运行时的调用图无关。它也不是在整个模块中递归搜索所有同名操作的通配语法。

如果改成 `builtin.module(cse,canonicalize)`，两个 Pass 的当前对象变成根 module。由于这两个具体 Pass 能遍历内部区域，它们也可以简化内部函数；但其调度单位、分析范围与函数级嵌套不同。不能因为这次终态碰巧相同，就认为两种 pipeline 在工程上完全等价。

## 7. 一个函数先走完整个序列

对 f 和 g 的函数级 pipeline，可以分别画出：

```text
f：CSE → canonicalize

g：CSE → canonicalize
```

在关闭多线程的本次运行中，日志显示先完成 f 的两个 Pass，再完成 g 的两个 Pass。它不是先对全部函数执行 CSE，然后统一对全部函数执行 canonicalize。

这使编译器可以在一个函数的 IR 仍被频繁访问时完成一组处理，也便于按函数安排并行任务。开启多线程后，不应依赖不同函数之间固定的先后；每个函数内部声明的处理顺序仍须遵守。

这解释了函数级 Pass 的一项重要限制：处理 f 时不能随意查看或修改兄弟函数 g 的状态。g 可能正被另一条任务修改。如果变换确实需要比较多个函数、改写它们之间的关系，就需要选择合适的更高层运行范围，并遵守那一层的访问约束。

运行单位也不能任意选成 addi 或普通 scf.for。固定版本的 Pass 基础设施要求调度锚点已注册并具有 `IsolatedFromAbove`，同时必须满足该 Pass 的类型限制。函数级运行常见，是因为 func.func 具备这类边界；它的 body 又足以容纳局部优化的工作范围。

`IsolatedFromAbove` 禁止捕获外层 SSA Value，从而减少局部处理时对外部 use-list 的牵连。但它不是任意修改父对象、兄弟对象的许可证。完整的线程与对象修改规则在实现自己的 Pass 时继续展开。

## 8. 用日志观察运行，而不是猜测

将第 6 节模块保存为 `nested.mlir`。下面命令只增加观察选项，没有增加新的优化：

```bash
"$MLIR_OPT" nested.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  --mlir-disable-threading \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all
```

`--mlir-disable-threading` 在这里让教学日志的次序容易核对，不是让 Pass 才能正确运行的必需配置。实际日志可以整理为下列事件序列；这是摘要，不是逐字复制完整 stderr：

```text
Before CSE             @f：三个 addi
After CSE              @f：两个 addi
Before Canonicalizer   @f：两个 addi
After Canonicalizer    @f：一个 addi
Before CSE             @g：三个 addi
After CSE              @g：两个 addi
Before Canonicalizer   @g：两个 addi
After Canonicalizer    @g：一个 addi
```

中间 IR 通常写到 stderr，最终模块写到 stdout。观察时先定位“哪个 Pass、哪个函数、前还是后”，再追踪具体 Value 的定义与使用。只看两份文件的行数，很容易把打印格式变化当成优化效果。

选项还可以配置某一个 Pass 实例：

```text
builtin.module(func.func(canonicalize{max-iterations=4},cse))
```

这里的 4 限制 canonicalizer 内部的相应迭代过程，不是让整个 pipeline 对函数执行四遍。`canonicalize` 的命令行名字和这些选项由 Pass 注册及定义连接到实现；在本地源码中它们集中定义在 `mlir/include/mlir/Transforms/Passes.td`。当前先能读懂配置含义，注册 API 在小 Pass 工程中再实际使用。

## 9. CSE 为什么还需要分析信息

回到主例子：第一条加法位于第二条之前，很容易确认它的结果可以复用。若函数有多个 Block，单靠文本前后位置就不够，必须判断定义是否支配替换后的使用。

CSE 可以向框架请求 `DominanceInfo`。这类 Analysis 描述当前 IR 的某些事实，通常按需计算并缓存。另一个 Pass 再需要相同事实时，若缓存仍然有效，就可以复用。

问题在于 IR 会变化。假设某个 Pass 删除一条 CFG 边，旧支配关系可能不再成立；如果后续 CSE 继续使用旧结果，就可能作出错误判断。因此，运行 Pass 后还要处理 analysis preservation/invalidation：哪些分析仍然有效，哪些需要失效并在下次重新计算。

这不是要求每个 Pass 都手动维护所有分析算法。默认按框架约定使未保留的缓存失效，Pass 只有在能够保证时才声明保留相关分析。本地 CSE 实现中，没有改动时保留全部分析；发生它所支持的局部修改时，按其实现保证保留支配相关分析。

这一点足以解释 PassManager 为什么不只是一个字符串列表。更细的 AnalysisManager API、跨分析依赖和 CFG 分析算法，在后续相应章节展开。

## 10. 成功、没有变化和失败是三种不同的观察

一个 Pass 成功，并不要求删除一个操作。把已经简化成 x+x 的函数再次交给 CSE，输出不变也可以成功。反过来，“工具认识这个 Pass 名字”也不保证配置或运行必然成功。

先看调度范围错误：

```bash
"$MLIR_OPT" twice.mlir \
  --pass-pipeline='builtin.module(func.func(convert-func-to-llvm))'
```

本地 LLVM 20.1.8 在构造 pipeline 时拒绝它，诊断的关键内容是：

```text
Can't add pass 'ConvertFuncToLLVMPass' restricted to 'builtin.module'
on a PassManager intended to run on 'func.func'
```

这是根据日志抽取的诊断片段。这个 Pass 虽然名字里有 func，其运行锚点却是 module；算法要转换哪种操作，与 Pass 要在哪个容器上运行，不是同一件事。此时还没有进入它的变换主体。

再看运行过程中主动失败。为了观察，可以专门使用 canonicalizer 的测试配置：

```bash
"$MLIR_OPT" twice.mlir \
  --pass-pipeline='builtin.module(func.func(canonicalize{max-iterations=1 test-convergence=true},cse))' \
  --mlir-disable-threading \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-failure
```

在本章输入与固定版本上，canonicalizer 已将程序简化到 x+x，但在这个过小的迭代预算下未通过收敛检查，于是报告 Pass 失败。日志出现 `IR Dump After Canonicalizer Failed`，退出码为 1，后面的 CSE 没有开始。

这个测试说明两件事。第一，失败条件由 Pass 的契约和配置决定，不只来自非法 IR；默认 canonicalize 是尽力简化，不能把这个特意打开的测试选项当成其正常默认行为。第二，失败前 IR 可能已经修改，框架不会普遍替任意 Pass 自动回滚到运行前状态。

对当前这一个函数的序列，失败后不会继续运行其后面的 Pass，失败也会向上返回。多个子任务尤其是并行任务中，不能据此推断其他已经启动的函数处理一定都尚未发生。

除了主动失败，框架也能通过变换后的 verifier 发现不变量被破坏。`mlir-opt` 默认开启变换间验证，实际实现可以根据保留信息避免冗余验证。但合法性检查仍不证明算法等价：把 x+x 错改为 x，类型和结构可能仍然合法。

## 11. 对照源码：调度怎样到达具体算法

现在已经知道希望源码解释什么：选中一个函数，按顺序调用 Pass，CSE 在函数内复用已有计算，并把失败传给调度方。沿这条路径读几个关键位置即可，不需要从 Pass.cpp 第一行顺读到最后。

先看 [Pass.cpp 的 runPipeline](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Pass/Pass.cpp#L571)。其中最关键的循环是：

```cpp
for (Pass &pass : pm.getPasses())
  if (failed(run(&pass, op, am, verifyPasses, parentInitGeneration)))
    return failure();
```

这是源码摘录。`pm` 提供这个层级的 Pass 序列，`op` 是当前函数或模块，`am` 提供其分析管理。循环中的 `run` 不只是直接调用算法，还准备运行状态、发出 instrumentation 通知、处理分析失效及验证。普通 Pass 最终通过 `pass->runOnOperation()` 进入具体实现；嵌套管理则走专门的 adaptor 路径。

再看同文件的 [runOnOperationImpl](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Pass/Pass.cpp#L719)。它遍历当前对象的 Region、Block 和直接包含的 Operation，为匹配的对象找到嵌套 PassManager，再运行那一层 pipeline。这正对应第 6 节中 f/g 被选择、h 需要再嵌套一层的现象。

最后进入 [CSE::runOnOperation](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/CSE.cpp#L396)，开头三行是：

```cpp
IRRewriter rewriter(&getContext());
CSEDriver driver(rewriter, &getAnalysis<DominanceInfo>());
bool changed = false;
```

接着执行：

```cpp
driver.simplify(getOperation(), &changed);
```

把它们与本章图对应起来：`getOperation()` 取得当前函数；`getAnalysis<DominanceInfo>()` 请求支配信息；driver 使用修改 IR 的工具，执行实际简化并记录是否改变。`IRRewriter` 的方法如何维护对象关系，留给 C++ IR API 与改写章节。

进一步追到同文件的 `CSEDriver::simplifyOperation`，在处理本例这种无内存效果操作的分支中，会查询 `knownValues.lookup(op)`；找到可复用对象后调用 `replaceUsesAndDelete`，否则记录当前操作供后续比较。实际相等性比较通过 `OperationEquivalence` 完成。这就把第 2 节“b 的使用改为 a”的图，连到了实现中的查找与替换。

支配区域遍历、带内存效果操作和终结操作还有各自处理分支。本章只追完主例子经过的必要路径，不把这个简化描述冒充完整 CSE 算法。

canonicalize 的对应入口是 [Canonicalizer.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Canonicalizer.cpp)：初始化阶段收集方言/操作提供的模式，运行阶段用 `applyPatternsGreedily` 处理当前对象。第 10 节的主动失败来自其 `testConvergence` 检查。为什么这些模式需要 driver、怎样表达一条模式，会在 PatternRewriter 章节接着解释。

## 12. 把运行模型带回 sum_positive

你已经用 `--convert-scf-to-cf` 处理过 sum_positive。现在可以把这次经历拆开：选定的 conversion Pass 读取结构化循环与分支，在内存中建立相应 CFG 并替换使用，工具最后打印处理后的 IR。

CSE/canonicalize 则尝试简化已有计算，并不承诺把它变成 CF 或 LLVM IR。对没有明显冗余的 sum_positive，输出变化可能很少；这并不说明 Pass 没有运行。反过来，一个名为 conversion 的 Pass 成功也不普遍保证整份程序已变成单一方言，具体允许保留什么取决于转换契约。

因此，阅读真实编译链时要给每一阶段写出两个判断：它被安排在哪个 IR 层级，以及它对该阶段输入希望完成什么处理。这样能同时读懂 pipeline 的结构和每个算法的作用。

## 本章读完应能解释的内容

本章的目标是形成运行模型。能脱离原文解释下面四点，就可以继续学习 C++ IR API：

1. CSE 后为什么仍有加零，而 canonicalize 后只剩 x+x？
2. 为什么函数级嵌套能处理 f/g，却没有自动选择内层 module 的 h？
3. 为什么运行日志按一个函数的完整序列展开，Pass 也不能任意访问兄弟函数？
4. 为什么成功可以没有变化，失败却可能已经改变了 IR？

需要查命令或源码时再回看相应段落。完整 C++ Pass 编写、分析 API 和测试框架不是本章额外的阅读关卡。

## 依据与复现

规范以 [LLVM 20.1.8 PassManagement](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/PassManagement.md) 和 [Canonicalization](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Canonicalization.md) 为准；Pass 的名称、构造器与选项见 [Transforms/Passes.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/Passes.td)。正文中的 C++ 是定位实现用的摘录，没有将其作为独立 C++ 工程编译。

维护脚本 `aicompiler-labs/llvm-mlir/docs/validate_passes.py` 从本页提取模块，核对 CSE/canonicalize 输出、嵌套选择、串行日志次序、配置及失败行为。默认结果保存到 `artifacts/logs/mlir-docs/<日期>-passes/manifest.json`。这些是本章讲解的验证证据，不是要求读者完成的另一组作业。

下一章计划：[编译器机制目录](./)中的 **C++ IR API**，从一次 Value 替换和 Operation 删除开始，把本章的“修改 IR”展开成具体对象操作。
