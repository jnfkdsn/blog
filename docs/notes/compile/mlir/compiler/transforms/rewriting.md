---
order: 3
title: PatternRewriter：把一次 IR 修改写成可应用的规则
updated: 2026-09-13
---

# PatternRewriter：把一次 IR 修改写成可应用的规则

上一章中，我们拿到一个 Operation，检查它的输入，再用 C++ API 替换结果、删除旧操作。现在希望把这段修改交给编译器反复使用：无论这个操作出现在函数中的什么位置，只要条件满足，就能应用同样的变换。

**本章要建立的是“局部改写规则”的概念：规则说明在哪里可以改、改成什么，框架负责找到操作并调用规则。** 我们先把一次已经会做的加零消除完整走通，再从这次修改对其他操作的影响，理解为什么需要 PatternRewriter。

## 1. 先把要完成的修改看清楚

仍然使用 `twice`，这次只保留一处加零：

<!-- mlir-example: rewriting-single-input -->
```text
module {
  func.func @twice(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %a = arith.addi %x, %zero : i32
    %r = arith.addi %a, %x : i32
    return %r : i32
  }
}
```

`%a` 的值就是 `%x`，因此 `%r` 可以直接使用 `%x`。修改的目标是：

```text
修改前：%a = addi(%x, 0)        修改后：
        %r = addi(%a, %x)              %r = addi(%x, %x)
```

这里包含两个动作。先把 `%r` 对 `%a` 的引用改成对 `%x` 的引用，再删除产生 `%a` 的加法。只删除加法会留下没有定义的使用；只转接使用则会留下一个已经无用的加法。

如果已经拿到第一条加法的句柄 `op`，上一章的底层 API 足以完成这两个动作：

```cpp
// 已经确认 op 是可以消除的加零操作。
Value replacement = op.getLhs();
op.getResult().replaceAllUsesWith(replacement);
op.erase();
```

RAUW 改的是使用者的 operand。它不会把原结果对象“变成”另一个 Value；`erase()` 随后才销毁旧操作及其结果。`%x` 是函数参数，在这两个加法处都可用，而且不依赖 `%a`，所以这里的替换满足支配要求，也不会引入自引用。

不过，这三行还不能直接作为一条可复用的优化规则：如果传入的是第二条加法，删除它就会把 `2*x` 错改成 `x`。我们还需要明确地表达“什么时候可以执行这三行”。

## 2. 从一段修改代码中提取规则

对本例，可以把允许的变换写成：

```text
匹配：一条右操作数由整数常量 0 定义的加法
改写：用左操作数替换这条加法的结果，并删除加法
```

这就是一条 rewrite pattern（改写规则）。它描述一个局部结构和对应的变换，不包含“遍历完整函数”这样的外层工作。

匹配从一条操作出发。这条作为入口的操作叫 **root Operation**。在本例中，root 是 `arith.addi`；检查右操作数时，还要顺着 Value 找到定义它的 `arith.constant`。因此“以加法为 root”并不意味着只能检查加法自己的字段。

沿输入具体判断一次：

- 第一条加法的 RHS 是 `%zero`。它由 `arith.constant 0 : i32` 定义，规则可以应用。
- 第二条加法的 RHS 是函数参数 `%x`。它没有这样的常量定义，规则不适用。

`%zero` 这个名字不参与判断。把它打印成 `%c0_i32` 或 `%17`，都不改变匹配结果。编译器沿内存中的定义—使用关系读取常量属性。

我们将规则的范围固定为**无 overflow flags 的标量 signless `i32`**。在这个范围内，整数加零保持原值。实现会检查这个范围；其他类型与属性组合留给有相应语义论证的规则。匹配条件由此承担两项工作：找出目标结构，并检查本规则保持语义所依赖的前提。

到这里，我们已经知道规则的内容。接下来只需要把这两部分放入 MLIR 约定的接口。

## 3. 用 matchAndRewrite 表达“先判断，再修改”

MLIR 为规则提供 `matchAndRewrite` 回调。框架把一个候选 root 传进来，规则检查它，并在适用时完成一次改写。省去类声明、范围检查和日志后，本例的核心只有下面几行；完整实现放在文末：

```cpp
LogicalResult matchAndRewrite(arith::AddIOp op,
                             PatternRewriter &rewriter) const override {
  // 完整实现先检查标量 i32 与 overflow flags。
  if (!isIntegerConstant(op.getRhs(), 0))
    return failure();

  rewriter.replaceOp(op, op.getLhs());
  return success();
}
```

`op` 是本次收到的加法。辅助函数 `isIntegerConstant` 完成上一节的查找：查看 RHS 的定义操作，再读取整数属性。检查失败时返回 `failure()`，表示这条规则不适用于当前加法，框架可以继续处理其他候选。

检查成功时，`rewriter.replaceOp(op, op.getLhs())` 把旧结果的所有使用转接到左操作数，并删除旧操作。它完成了第一节 RAUW 与 erase 对应的工作。随后返回 `success()`，表示这次改写已经完成。

这里的 `success()` 描述的是**发生了一次修改**。例如 `%r = addi(%a, %x)` 的 RHS 不是零，我们返回 failure，函数本身依然完全合法。未匹配是规则执行中的普通结果，不是编译失败。

框架正是通过这个返回值判断有没有应用规则，因此检查和修改需要有清楚的分界：

```text
读取并检查 IR ──不满足──→ failure，IR 保持原样
      │
    满足
      ↓
通过 rewriter 完成修改 ──→ success
```

如果先创建或修改操作，再发现条件不满足并返回 failure，框架收到的“没有应用”就与真实 IR 不一致。普通 Pattern 改写没有自动恢复任意修改的保证，所以应当在第一次修改前完成可能失败的匹配检查。

现在先把 `rewriter` 理解为框架交给规则的修改入口。它为什么必须由框架提供，我们在看到规则实际运行后再展开。

## 4. 让框架真正应用这条规则

定义好 `RemoveAddZero` 类后，还需要一段调用代码：收集规则，并把函数交给负责应用规则的程序。后者称为 **rewrite driver（改写驱动）**。

本例只需要遍历一次函数，使用 `walkAndApplyPatterns` 即可：

<!-- cpp-example: single-driver -->
```cpp
static void applyAddZero(func::FuncOp fn, MLIRContext &context) {
  RewritePatternSet patterns(&context);
  patterns.add<RemoveAddZero>(&context);
  FrozenRewritePatternSet frozen(std::move(patterns));
  walkAndApplyPatterns(fn, frozen);
}
```

`patterns.add` 创建并加入我们的规则。`FrozenRewritePatternSet` 将收集好的规则集合固定下来，供 driver 使用；待处理的 IR 仍然可修改。最后一行开始遍历，并在合适的操作上尝试规则。这个调用不会凭空带上其他算术优化规则。

对于 `twice`，过程可以完整推演如下：

| 当前操作 | 规则如何判断 | 发生的变化 |
|---|---|---|
| `arith.constant` | 不是规则接受的加法类型 | 无 |
| `%a = addi(%x, %zero)` | RHS 是整数零常量 | `%r` 的第一个输入改为 `%x`，旧加法被删除 |
| `%r = addi(%x, %x)` | RHS 是函数参数 | 无 |
| `return %r` | 不是加法类型 | 无 |

完整程序运行后，打印出下面的 IR：

<!-- mlir-example: rewriting-single-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

打印器重新选择了 SSA 名称。对照操作之间的连接，可以看到我们预期的变化：第一条加法消失，剩余加法的两个输入都指向函数参数。

零常量仍然存在，因为这条规则只替换加法，而本次 Walk driver 不顺带做死代码清理。一个没有使用者的常量是合法 IR；后续清理可以删除它。这样，我们可以把眼前的输出完整归因于刚才写下的规则。

配套程序显式打印一次 `APPLY RemoveAddZero`，确认回调确实执行了修改。到这里，“描述规则 → 加入集合 → driver 应用 → 获得新 IR”已经走通。你暂时不需要了解其他 driver，就能解释这次变换。

## 5. 为什么修改要经过 PatternRewriter

回到第一节：底层 RAUW 和 erase 也能得到这份输出，为什么回调要改用 `rewriter.replaceOp`？

关键在于，执行过程中除了 IR，**driver 还可能持有与 IR 有关的处理记录**。当规则修改 IR 时，这些记录也要跟着变化。

继续观察刚才的 `%r`。改写前，它的输入是两个不同 Value：`%a` 和 `%x`。改写后，它的两个输入成为同一个 `%x`。假设我们再提供一条规则，将 `addi(v, v)` 改成 `muli(v, 2)`，那么 `%r` 就从“不匹配”变成了“匹配”。

第二条规则不会被第一条规则主动调用。要发现这个新机会，driver 必须在合适的时机检查 `%r`。如果它先前已经检查过 `%r`，就需要知道 `%r` 的输入刚刚发生了变化，安排重新检查。

可以把这类 driver 的待处理记录想成一张工作表：

```text
尝试 %r：两个输入不同，暂时不匹配
    ↓
处理 %a：将它的使用替换成 %x
    ↓
获知“%r 的输入已改变”
    ↓
把 %r 放回待处理集合，再次尝试规则
```

这是一条说明通知用途的执行情景，不是上一节 Walk 运行的实际顺序。上一节中 `%r` 正好在修改之后才被访问；支持重新处理的 Greedy driver 则使用工作表维护后续机会。

删除也有同样的问题。如果 driver 的工作表中保存着某个操作的指针，规则删除它时就必须让 driver 清除相关记录。否则，IR 已经没有这个对象，待处理列表却还可能保留它。

因此，一次框架内的修改需要同时完成两件事：改变 IR 的对象关系，并告诉 driver 哪些对象发生了变化。`PatternRewriter` 把修改接口与通知连接在一起。在我们的例子中：

```text
RemoveAddZero
  → rewriter.replaceOp(旧加法, %x)
      → 转接旧结果的使用，并通知相关用户发生修改
      → 删除旧加法，并通知删除
  → 返回 success
```

本地实现中的 `replaceOp` 正是通过 rewriter 的替换与删除方法完成这些步骤。Greedy driver 接收通知后维护工作表；其他 driver 可以根据自身策略处理通知。因此，同一条规则不需要知道调用者用哪种数据结构保存待处理操作。

直接调用 Value 的 RAUW 或 Operation 的 erase，可以维护 IR 自身的 use-list，却绕过了这条框架通知路径。**在 Pattern 内，应当用传入的 rewriter 创建、替换、删除或更新 IR。** 需要原地改变 operands 或属性时，用 `modifyOpInPlace` 包住更新，让 driver 也能获知变化。

Rewriter 接管的是修改与通知这部分工作。证明 `x+0` 可以替换成 `x`、保证替代 Value 在所有使用处可用，仍由规则负责。并且 `replaceOp` 已删除旧操作，调用之后应直接使用提前保存的替代对象，不再访问旧 `op`。

## 6. 把规则放回编译器的整体流程

现在再看你已经学过的 Pass，它与这套机制怎样连接？

一个 Pass 可以在 `runOnOperation()` 中拿到当前函数，准备规则集合，调用 driver，然后处理驱动结果。我们刚才的独立 C++ 工具承担了这层调用工作；接入 Pass 时，局部匹配与改写本身可以保留。

```text
Pass 的 runOnOperation()
  │  取得函数，准备规则集合
  ↓
Driver
  │  选择操作与候选规则
  ↓
Pattern 的 matchAndRewrite()
  │  判断适用条件，通过 rewriter 修改 IR
  ↓
修改通知返回 driver，driver 继续处理或结束
```

每一层现在都能对应到主例中的具体动作：Pass 决定对哪个函数运行；driver 找到第一条加法；Pattern 判断 RHS 是零；rewriter 将 `%r` 的输入接到 `%x` 并删除旧加法。

当规则变多时，这种分工的收益才更明显。我们可以继续增加“两个相同输入的加法”等规则，而不必让每条规则重新实现遍历、候选调度和删除通知。不同的变换需求也可以选择不同 driver。

Walk 和 Greedy 就是两种应用规则的策略。前者执行一次遍历；后者可以重新处理受影响或新建的操作，直到在配置范围内没有进一步变化。它们调度的是 **Pattern 的应用**。Pass pipeline 则安排整个 Pass 的执行顺序，是外层的另一项工作。

## 7. 用一个变化检查是否真的理解

把输入中的常量 `0` 改成 `1`，暂时不要运行程序，沿刚才的链路推演：

1. driver 仍会把第一条加法交给 `RemoveAddZero` 吗？
2. 回调会在哪一步停止？此时有没有创建、替换或删除对象？
3. 最终 IR 应当保留哪些操作？

<details>
<summary>展开推演</summary>

加法类型没有变化，它仍然是候选 root。规则读到 RHS 的整数属性是 `1`，零常量检查失败，返回 failure。此前只有读取，没有修改；两条加法和常量都保留。第二条加法的 RHS 依然是函数参数，也不匹配。

这说明找到候选操作、匹配成功和完成改写是三个先后发生的步骤。能够区分这三步，就不会把“driver 访问过操作”误认为“这个操作一定被优化”。

</details>

本章的核心已经完整：把一次合法的局部修改封装成规则，由 driver 应用，再通过 rewriter 保持 IR 与驱动记录同步。接下来可以阅读[改写驱动：规则如何协作与收敛](./rewrite_drivers)的前四节，继续追踪第二条规则怎样接上这次修改。随后[实现并测试一个小 Pass](../../tutorials/first_pass)会把本章调用接入正式工程；benefit、fold 和停止预算的细节可以按需回查。

## 完整实现与复现

<details>
<summary>展开完整规则：把前面的判断与修改放进 C++ 类</summary>

`OpRewritePattern<arith::AddIOp>` 声明 root 类型，因此回调收到的是加法操作。构造参数 `benefit=1` 给规则一个候选优先级；这里只有一条规则，无需用它安排顺序。`setDebugName` 用于标识规则。

<!-- cpp-example: zero -->
```cpp
struct RemoveAddZero : OpRewritePattern<arith::AddIOp> {
  explicit RemoveAddZero(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/1) {
    setDebugName("RemoveAddZero");
  }
  LogicalResult matchAndRewrite(arith::AddIOp op,
                               PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32) ||
        op.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return rewriter.notifyMatchFailure(op, "expected unflagged scalar i32");
    if (!isIntegerConstant(op.getRhs(), 0))
      return rewriter.notifyMatchFailure(op, "RHS is not an integer zero constant");
    llvm::errs() << "APPLY RemoveAddZero\n";
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
```

辅助函数如下：

```cpp
static bool isIntegerConstant(Value value, int64_t expected) {
  auto op = value.getDefiningOp<arith::ConstantOp>();
  auto attr = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr();
  return attr && attr.getValue() == expected;
}
```

`notifyMatchFailure` 返回 failure，并附带未匹配原因供调试设施使用。普通运行不一定打印这个原因。`APPLY` 行是示例自己加入的成功记录，放在所有条件检查通过之后。

把类声明、辅助函数与第四节调用对应起来即可；API 名称可以在写代码时查阅。

</details>

正文按工作区 LLVM/MLIR 20.1.8 编译运行核对。源码与构建说明在 `aicompiler-labs/llvm-mlir/docs/rewriting/`；`validate_rewriting.py` 验证本文和进阶篇的源码片段及输入输出。它运行的是操作 IR 的编译器工具。

若要观察本章输出，在 workspace 根目录运行：

```bash
python3 aicompiler-labs/llvm-mlir/docs/validate_rewriting.py
artifacts/builds/mlir-rewriting-docs/rewriting-demo single artifacts/logs/mlir-docs/2026-09-13-rewriting/rewriting-single-input.mlir
```

第二条命令使用本次验证生成的输入，直接打印成功规则与修改后的 IR；后续日期的日志目录以脚本输出为准。阅读正文无需先运行这些命令。

继续查阅：[官方 Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)；[Toy 第三章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)用连续转置给出另一条规则的完整例子。对应本章修改通知的固定版本实现见 [PatternMatch.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/IR/PatternMatch.cpp)，驱动接口见 [WalkPatternRewriteDriver.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h)。