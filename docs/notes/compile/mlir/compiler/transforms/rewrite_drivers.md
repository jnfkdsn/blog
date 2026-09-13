---
order: 4
title: 改写驱动：规则如何协作与收敛
updated: 2026-09-13
---

# 改写驱动：规则如何协作与收敛

[上一章](./rewriting)已经把一次加零消除拆成匹配、替换和驱动调用，并解释了修改通知的作用。本篇继续研究：多条规则共同工作时，怎样决定下一步处理谁，怎样解释最终输出，又怎样保证处理能停下来。

这里逐步扩展 `twice`，比较真实运行结果。规则都由同一套 Pattern 接口表达；变化的是规则集合和驱动配置。可以先读前四节理解协作过程，其余各节在需要选择策略或排查行为时查阅。

## 1. 从一次修改扩展到一组规则

在上一章输入的基础上，再增加一处加零，让 `%r` 的两个输入分别来自 `%a` 和 `%b`：

<!-- mlir-example: rewriting-input -->
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

这一轮准备三条规则：

| 规则 | 匹配与改写 | 为后续创造的机会 |
|---|---|---|
| A：RemoveAddZero | `addi(x, 0) → x` | `%r` 的两个输入可能变成同一个 Value |
| B：DoubleToMul | `addi(x, x) → muli(x, 2)` | 新建一个可以继续处理的乘法 |
| C：MulTwoToShift | `muli(x, 2) → shli(x, 1)` | 本组规则不再处理这个移位 |

三条规则都限定标量 signless i32、无 overflow flags，常量通过实际常量操作识别。按固定宽度回绕语义，加倍与左移一位在这里一致；这是一组用于解释驱动的规则，不据此断言某种表示在所有硬件上更快。

如果先访问 `%r`，它的输入还是两个不同 Value，B 不匹配。处理 A 之后，`%r` 的输入改变，原先的结论就过时了。即使原始 IR 按定义顺序访问，能及时处理 `%r`，B 新建的乘法也不一定属于原先收集的操作列表，C 仍可能错过它。

因此，要完成整组改写，除了规则本身，还需要保存待处理对象，并在 IR 变化后更新这份记录。这个待处理集合通常称为 **worklist**。上一章解释的修改通知，在这里用于维护工作表，让刚刚出现的机会能够继续被处理。

加零规则 `RemoveAddZero` 的完整定义见[上一章的实现附录](./rewriting#完整实现与复现)。下文补上另外两条规则，并让它们作用于同一个函数。

## 2. 创建新操作后，怎样把控制权交回 driver

第二条规则把两个相同输入的加法变成乘法：

<!-- cpp-example: double -->
```cpp
struct DoubleToMul : OpRewritePattern<arith::AddIOp> {
  explicit DoubleToMul(MLIRContext *context, unsigned benefit = 1)
      : OpRewritePattern(context, benefit) {
    setDebugName("DoubleToMul");
  }
  LogicalResult matchAndRewrite(arith::AddIOp op,
                               PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32) ||
        op.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
        op.getLhs() != op.getRhs())
      return failure();
    llvm::errs() << "APPLY DoubleToMul\n";
    auto two = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 2, 32);
    auto mul = rewriter.create<arith::MulIOp>(op.getLoc(), op.getLhs(), two);
    rewriter.replaceOp(op, mul.getResult());
    return success();
  }
};
```

`op.getLhs() == op.getRhs()` 比较的是两个 operand 是否引用同一个 Value，而不是编译器已经证明两个不同 Value 在数学上相等。如果输入仍然分别是 `%a` 和 `%b`，这条规则就不会自行执行 CSE 或 GVN。

常用 driver 在尝试规则时把 rewriter 的插入点设到 root 之前，本例据此插入常量和乘法。若规则改变插入点，仍需自己保证所需输入在那里可用。新乘法使用 `%x` 和新常量，没有使用旧加法的结果，因此可以在转接旧结果的使用后删除旧加法。

第三条规则消费刚创建的乘法：

<!-- cpp-example: shift -->
```cpp
struct MulTwoToShift : OpRewritePattern<arith::MulIOp> {
  explicit MulTwoToShift(MLIRContext *context) : OpRewritePattern(context, 1) {
    setDebugName("MulTwoToShift");
  }
  LogicalResult matchAndRewrite(arith::MulIOp op,
                               PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32) ||
        op.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
        !isIntegerConstant(op.getRhs(), 2))
      return failure();
    llvm::errs() << "APPLY MulTwoToShift\n";
    auto one = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, 32);
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(), one);
    return success();
  }
};
```

`replaceOpWithNewOp` 是“创建一个替代操作，然后替换旧操作”的便利接口。它和前一段显式 `create` 再 `replaceOp` 的关系很直接，不需要把它当作另一类优化机制。

A、B、C 的代码都没有自己维护循环，也没有递归调用其他 Pattern。每条规则完成局部修改并返回，是否继续应用另一条规则由 driver 决定。

## 3. 将规则交给 Greedy driver

用 `RewritePatternSet` 收集规则实例：

<!-- cpp-example: population -->
```cpp
patterns.add<RemoveAddZero, DoubleToMul, MulTwoToShift>(&context);
```

这里的 `patterns` 是 `RewritePatternSet`，构造时已关联当前 Context。随后通过 `FrozenRewritePatternSet frozen(std::move(patterns))` 得到用于应用的规则集合。“Frozen”指规则集合固定，不是冻结待修改的 IR；这个步骤也不会自动搜集所有方言的 canonicalization 规则。

为了明确观察 A、B、C 的作用，本篇主例采用以下配置：

<!-- cpp-example: driver -->
```cpp
GreedyRewriteConfig config;
config.fold = false;
config.cseConstants = false;
config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
```

在 LLVM 20.1.8 中，Greedy driver 的 folding 和常量 CSE 默认启用；这里显式关闭它们及 Region 简化，减少其他机制对主例的影响。简单的死代码清理仍然存在，稍后会看到它删除无用常量。

应用规则时传入函数 `fn`：

<!-- cpp-example: apply -->
```cpp
result = applyPatternsGreedily(fn, frozen, config, &changed);
llvm::errs() << "converged=" << succeeded(result)
             << " changed=" << changed << "\n";
```

这个 Operation 入口处理 `fn` 各 Region 内的操作，不把 `func.func` 本身当作待匹配 root；容器需要满足相应的隔离要求。当前函数的 body 是工作范围。不要把“传入函数句柄”理解成只对函数操作运行一次 A/B/C。

本地版本还保留了旧接口 `applyPatternsAndFoldGreedily`，其包装会开启 folding。本文使用 `applyPatternsGreedily` 并显式配置行为，具体接口以固定版本头文件为准。

## 4. 沿一次真实运行追踪 worklist

对本篇输入，`greedy` 模式打印的成功轨迹是：

```text
APPLY RemoveAddZero
APPLY RemoveAddZero
APPLY DoubleToMul
APPLY MulTwoToShift
converged=1 changed=1
```

可以据此还原以下数据变化：

| 修改 | 发生的变化 | 需要 driver 获知的信息 |
|---|---|---|
| A 替换第一条加零 | `%r` 的一个输入改成 `%x` | 使用旧结果的操作发生修改 |
| A 替换第二条加零 | `%r` 的两个输入都成为 `%x` | 原先不匹配 B 的位置现在可能匹配 |
| B 替换 `%r` | 创建 `muli(x, 2)` | 新操作需要进入后续处理范围 |
| C 替换乘法 | 创建 `shli(x, 1)` | 旧乘法被删除，新操作与其依赖可能需要处理 |

这张表表达的是依赖变化，不是保证所有输入、所有配置下都采用同一个全局遍历顺序。Greedy 的初始遍历顺序只决定最初怎样准备 worklist；随后的通知还会把符合范围和 strictness 条件的操作加入工作表。

例如 `GreedyPatternRewriteDriver.cpp` 的 `notifyOperationInserted` 和 `notifyOperationModified` 都会走到 `addToWorklist(op)`。删除操作时，driver 还会维护工作表和相关记录，并考虑受影响的生产者；转接 use 时，对相应用户的修改通知也会影响后续处理。

因此，worklist 既不是永远不变的原始节点列表，也不是每一步都无条件扫描整个函数。它利用修改信息，让相关操作获得重新尝试的机会。具体入队细节仍受 driver 的实现和配置约束。

最终输出是：

<!-- mlir-example: rewriting-greedy-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.shli %arg0, %c1_i32 : i32
    return %0 : i32
  }
}
```

A/B/C 没有显式删除原来的零常量和乘法使用过的常量 2。它们失去用途后，由 Greedy driver 的简单 DCE 清理。解释这份输出时，要同时识别“Pattern 替换了操作”和“driver 清理了无用操作”这两类行为。

## 5. 同一组规则换成 Walk driver 会发生什么

现在保持规则集合不变，改用 `walkAndApplyPatterns(fn, frozen)`。这个 driver 做一次后序遍历，不重新访问被修改或新替换出的操作，也不做 folding 与 DCE。

本例原有操作按序被处理，A 改变了 `%r` 的输入，因此走到原来的 `%r` 时 B 能成功；但 B 新创建的乘法不会在同一轮被继续匹配，C 没有应用。

实际输出为：

<!-- mlir-example: rewriting-walk-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.muli %arg0, %c2_i32 : i32
    return %0 : i32
  }
}
```

这里仍有乘法和无用的零常量。它们不是 verifier 错误，而是这次驱动策略没有继续做相应工作。Walk 也有自己的修改范围限制，例如删除应局限于当前匹配 root 及其子结构；不能把 Greedy 上可运行的复杂规则集合不加检查地换到 Walk。

驱动的选择取决于规则的需求：只需一次局部遍历时，Walk 较直接；需要重新处理被修改或新建的操作时，Greedy 提供相应机制。Dialect Conversion 则在另一章引入目标合法性与类型转换，不等于给 Greedy 换一个名字。

## 6. 原地更新同样需要通知

不一定每次都创建替代 Operation。有时只需改变原操作的 operands 或属性。例如将 `addi(0, x)` 规范到 `addi(x, 0)`，让只检查 RHS 的 A 接着处理：

<!-- cpp-example: inplace -->
```cpp
struct MoveZeroToRhs : OpRewritePattern<arith::AddIOp> {
  explicit MoveZeroToRhs(MLIRContext *context) : OpRewritePattern(context, 1) {
    setDebugName("MoveZeroToRhs");
  }
  LogicalResult matchAndRewrite(arith::AddIOp op,
                               PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32) ||
        op.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
        !isIntegerConstant(op.getLhs(), 0) || isIntegerConstant(op.getRhs(), 0))
      return failure();
    Value lhs = op.getLhs(), rhs = op.getRhs();
    llvm::errs() << "APPLY MoveZeroToRhs\n";
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(ValueRange{rhs, lhs}); });
    return success();
  }
};
```

`modifyOpInPlace` 把修改包在 start/finalize 通知之间。底层的 `setOperands` 仍然负责重接 use-list；外层通知使 driver 知道这个 root 已改变。直接调用 `setOperands` 而不通知，就可能漏掉重新处理它的机会。

这里还需要一个终止条件：如果两边都是零，就不交换；原地交换后，左边不再是零，当前规则不应再次匹配同一状态。不能每次调换左右输入都返回 success，否则可能永远宣称发生了新变化。

配套例子的输入是：

<!-- mlir-example: rewriting-left-zero -->
```text
module {
  func.func @twice(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %r = arith.addi %zero, %x : i32
    return %r : i32
  }
}
```

实际成功轨迹为 `MoveZeroToRhs → RemoveAddZero`，最终函数直接返回 `%x`。第一次修改保留了原 Operation 身份，第二次才将它替换并删除。

这组接口有 transaction-like 的命名，但不应把它理解为任意 IR 修改的自动回滚。`modifyOpInPlace` 的基础实现是 start、调用修改函数、finalize；基础 `cancelOpModification` 也没有替调用者保存并恢复旧状态。普通 Greedy Pattern 仍应先完成检查，修改后返回 success；不要先创建一批操作，发现不匹配后直接返回 failure。

## 7. Benefit 排的是候选优先级，不是 pipeline 顺序

给同一个 root 再增加一种方案：直接把 `addi(x, x)` 改成 `shli(x, 1)`，命名为 `DoubleToShift`。它与 B 都能匹配这个加法。

<!-- mlir-example: rewriting-double-input -->
```text
module {
  func.func @twice(%x: i32) -> i32 {
    %r = arith.addi %x, %x : i32
    return %r : i32
  }
}
```

只提供这两条竞争规则、关闭 folding，实际对照结果为：

| 配置 | 命中的规则 | 输出中的计算 |
|---|---|---|
| DoubleToMul benefit=2，DoubleToShift benefit=1 | DoubleToMul | `muli(x, 2)` |
| DoubleToMul benefit=1，DoubleToShift benefit=2 | DoubleToShift | `shli(x, 1)` |

在常用默认策略下，driver 对当前 root 优先尝试 benefit 更高的候选；高 benefit 候选不匹配时仍可以尝试较低的候选。Benefit 属于构造好的 Pattern 实例，可以在创建实例时根据目标等信息设置，但不是每个匹配位置上自动测量的性能收益。

它没有表达“先在整个函数运行规则 B，再在整个函数运行 C”，也没有比较所有可能改写序列的最终代价。Greedy 在当前选择中取局部优先方案，通常不会回溯重试所有历史分支。因此 benefit 更高不等于已经证明全局最优或硬件执行更快。

相同 benefit 的规则也不应依赖某个未明确约定的应用顺序来维持正确性。正确性来自各规则的语义前提；若顺序本身是算法要求，应显式设计阶段或选择能够表达所需约束的驱动方式。

## 8. Fold、canonicalize 与自定义 Pattern 的交界

还有一种容易误判的现象：规则集合为空，IR 仍然可能变化。

在同一主例上，配套程序的 `empty` 模式关闭 folding，`fold-only` 模式开启 folding；两者都不加入 A/B/C，也关闭常量 CSE 和 Region 简化。实测如下：

| 模式 | 自定义 Pattern 成功记录 | 结果 |
|---|---|---|
| empty | 无 | 当前主例保持原样，`converged=1 changed=0` |
| fold-only | 无 | 加零折叠，函数剩下 `addi(x, x)`，`converged=1 changed=1` |

fold-only 的实际输出为：

<!-- mlir-example: rewriting-fold-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

加零可以由操作自己的 fold hook 处理，无用常量再被清理。如果开启默认 folding，却声称“看到加零消失就证明 A 匹配成功”，就混淆了输出与执行路径。

三者的关系可以这样区分：

| 机制 | 提供什么 | 谁应用它 |
|---|---|---|
| Op 的 fold hook | 受限制的局部折叠，返回已有 Value 或常量 Attribute，也可支持 root 原地折叠 | Builder、Greedy driver 等调用者 |
| RewritePattern | 可以创建新操作、匹配较大结构的改写规则 | 选定的 Pattern driver |
| canonicalize Pass | 汇集注册方言/操作的 canonicalization patterns，并配置通用规范化过程 | PassManager 调度，内部使用 Greedy driver |

fold hook 本身不能任意创建新操作；返回常量 Attribute 时，调用者负责按方言设施物化所需常量。自定义 Pattern 仅仅被加入本章的局部集合，不会因此自动成为 `mlir-opt --canonicalize` 的规则；如何把规则附到 Op 的 canonicalization 入口，后续随操作定义展开。

关闭 folding 也不是关闭所有清理：即使空规则集合，只要输入包含可删除的死操作，Greedy 仍可能清理它。因此 `changed=true` 只能说明这次驱动改变了 IR，不能单独证明某个 Pattern 命中。

## 9. 停下来、没有变化与收敛失败

对当前 A/B/C，可以提出一个终止理由：先看剩余加法数量，再看剩余乘法数量，组成字典序度量。A 减少加法；B 虽增加乘法，但减少加法；C 在加法数量不变时减少乘法。没有规则重新制造加法，度量会下降，因此这组规则没有加法/乘法之间的往返循环。

现在把 C 换成反向规则 `muli(x, 2) → addi(x, x)`，就得到：

```text
addi(x, x) → muli(x, 2) → addi(x, x) → muli(x, 2) → ...
```

每一步都可以保持本例的算术语义，但整组规则不能收敛。单条规则合法，不等于组合后的编译过程一定终止。

配套程序的 `cycle` 模式只运行这两条规则，并同时设置 `maxIterations=2` 和 `maxNumRewrites=4`。本地执行记录了八次交替改写，然后返回：

```text
converged=0 changed=1
```

外层 iteration 是“处理工作表、Region 简化等阶段再继续”的循环，不能当作“最多应用一次 Pattern”。如果两个规则一直在同一工作表处理过程中制造新候选，只设外层 iteration 上限未必能及时停止，所以该反例同时限制每轮改写次数。

未收敛时 IR 已经被修改；本例打印的结果仍通过 verifier，甚至保留了来不及清理的死常量。停止上限不是事务回滚，也不能作为规则终止性的证明。

`setHasBoundedRewriteRecursion()` 用于声明某条规则能安全地重复应用，例如每次剥离一轮且剩余可剥离次数下降。它不设置迭代上限，不会自动证明递减量，更不能修复两条规则之间的互相撤销。实际实现仍需要检查规则及其组合的终止条件。

最后区分三个层次的返回语义：

| 层次 | success / failure 的含义 |
|---|---|
| 单条 Pattern | 已完成一次改写 / 当前规则不匹配 |
| Greedy driver | 在配置范围内完成收敛 / 未能在限制内完成收敛 |
| Pass | 由 Pass 的整体契约决定是否失败；可按需求将 driver failure 转为 `signalPassFailure()` |

某个操作没有任何规则匹配，可以是成功的不动点。它不等于“所有不希望存在的操作都被消除”；需要强制目标合法性时，后续使用 Dialect Conversion 的 legality 契约。

## 10. 用有限的源码路径解释一次改写

源码不必从头读到尾。围绕“规则 B 创建乘法，规则 C 为什么接着执行”，选择下面这条路径：

1. `PatternApplicator` 选择候选、设置插入点并调用 `matchAndRewrite`。
2. B 通过 rewriter 创建乘法、替换旧加法，进入 `RewriterBase` 的修改方法。
3. Greedy driver 收到插入、修改或删除通知，维护 worklist。
4. 工作表取出新乘法，为这个 root 尝试相应类型的规则。

需要解释“为什么没有改”时，先定位是哪一种情况：root 类型未进入候选、匹配条件不满足、操作不在 driver 范围内、被其他机制提前处理，或者达到停止上限。然后再查看 debug name、成功轨迹和 `notifyMatchFailure` 提供的原因。

`-debug-only=greedy-rewriter` 是 LLVM 调试日志入口，是否可用取决于工具构建及命令行支持；它不是每个自定义二进制天然识别的参数。本章工具使用明确的 `APPLY ...` 日志展示成功规则，输出 IR 用于核对实际变化。

规则还可以用 DRR、PDL/PDLL 等声明方式表达。它们改变规则的编写或表示方式，仍需要 driver 应用，也仍受相应改写协议与合法性条件约束；本章不把这些语法作为开始写 C++ Pattern 的前置条件。

## 把这些机制带回一个 Pass

现在可以把执行过程连起来：Pass 准备规则集合与范围，driver 选择 root 和候选规则，Pattern 完成局部改写，rewriter 把修改通知回 driver。Driver 再按照配置决定是否继续处理，并把收敛结果交还 Pass。

接下来[实现并测试一个小 Pass](../../tutorials/first_pass)把这些调用放入 `runOnOperation()`，补上注册、构建和正反例测试。优先掌握能够实现这一闭环的部分，其他驱动配置可在实验中按需回查。

## 依据与复现

配套源码位于 `aicompiler-labs/llvm-mlir/docs/rewriting/rewriting_demo.cpp`，正文标记的 C++ 片段与实际编译源码核对。输入输出来自固定版本运行，日志表示示例显式记录的成功轨迹，不是对任意程序遍历顺序的保证。

维护者从 workspace 根目录运行：

```bash
python3 aicompiler-labs/llvm-mlir/docs/validate_rewriting.py
```

生成物放在 `artifacts/builds/mlir-rewriting-docs/` 和 `artifacts/logs/mlir-docs/<日期>-rewriting/`。正文已展示关键过程；需要自行观察时，可按配套 README 直接运行 `rewriting-demo`，终端会分别给出实际 IR 与成功规则日志。通过计数属于回归证据，不替代学习过程。

这里编译并运行的是操作 IR 的 C++ 工具，没有执行改写后函数的机器码，也没有性能测试。匹配边界通过未匹配输入检查，结构通过 verifier 与打印结果核对，算术正确性的说明限定于正文写明的类型与 flags。

固定版本源码入口：

- [PatternMatch.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/PatternMatch.h) / [PatternMatch.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/IR/PatternMatch.cpp)：Pattern、替换/删除通知、原地修改协议。
- [PatternApplicator.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Rewrite/PatternApplicator.cpp)：候选选择、benefit 与匹配调用。
- [GreedyPatternRewriteDriver.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h) / [实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp)：范围、配置、worklist、修改通知与停止条件。
- [WalkPatternRewriteDriver.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h)：单轮遍历与不重新访问的契约。
- [Canonicalizer.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Canonicalizer.cpp)：汇集方言/操作规则并调用 driver。

官方概念说明可对照 [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) 与 [Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)。在线资料中的名称和配置可能演进，实际行为优先以本地固定版本的头文件和运行证据核对。
