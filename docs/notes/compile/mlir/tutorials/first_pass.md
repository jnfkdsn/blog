---
order: 2
title: 实现并测试一个小 Pass：从规则到编译器工具
updated: 2026-09-13
---

# 实现并测试一个小 Pass：从规则到编译器工具

上一章的 `RemoveAddZero` 已经会判断一条加法能否消除，并通过 rewriter 完成替换。我们还写了一段独立程序，主动把函数交给 driver。

现在希望像运行 CSE 那样，通过一条 pipeline 命令应用自己的规则。这要求把三件已学过的事情接起来：**PassManager 找到要处理的函数，Pass 调用改写框架，测试确认这个工具确实完成了预期变换。** 本章沿这个过程建立一个可以构建、运行和修改的小工程。

## 1. 从一个明确的输入输出约定开始

使用上一章的 `twice`：

<!-- mlir-example: first-pass-input -->
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

要改变的是第一条加法。它的 RHS 由零常量定义，因此 `%r` 可以直接使用 `%x`；第二条加法继续表达加倍。规则仍只接受无 overflow flags 的标量 signless i32，不在这一章扩大算术语义范围。

从调用者的角度，新的接口应当像这样：

```text
lab-opt input.mlir
  --pass-pipeline='builtin.module(func.func(lab-remove-add-zero))'
```

`lab-opt` 是我们准备构建的编译器工具，`lab-remove-add-zero` 是自己注册的 Pass 名称。命令行中的输入是 IR 文件；Pass 运行时面对的是解析后的 Operation 对象。

输入输出约定还要说明不匹配的情况。把 RHS 改成常量 1，原加法必须保留；把同一结果交给多个使用位置，替换必须更新所有这些位置。这样，后面的测试就有了来自需求的判据。

## 2. 让 Pass 接住当前函数

先考虑 PassManager 已经选中了 `@twice`。我们不需要再次按名字查找它，`getOperation()` 就能取得本次处理的操作。

将规则和 driver 调用放进 `runOnOperation()`：

<!-- cpp-example: pass -->
```cpp
struct RemoveAddZeroPass
    : PassWrapper<RemoveAddZeroPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveAddZeroPass)

  StringRef getArgument() const final { return "lab-remove-add-zero"; }
  StringRef getDescription() const final {
    return "Remove unflagged scalar i32 additions with zero on the RHS";
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveAddZero>(&getContext());
    GreedyRewriteConfig config;
    config.fold = false;
    config.cseConstants = false;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    if (failed(applyPatternsGreedily(fn, std::move(patterns), config)))
      signalPassFailure();
  }
};
```

这里的核心动作仍然是准备规则集合，再调用 `applyPatternsGreedily`。变化在于函数从哪里来：上一章由独立程序查找 `@twice`，这里由 PassManager 交给当前 Pass。因此同一个实现可以应用于模块中的多个函数，不把某个符号名写死在算法中。

`OperationPass<func::FuncOp>` 表达运行对象的类型约束。它让 `getOperation()` 返回函数句柄，也让 PassManager 能检查这个 Pass 是否被放到了正确层级。`PassWrapper` 提供实现一个具体 Pass 所需的包装；Type ID 宏用于框架识别这个 C++ Pass 类型。第一次阅读先把它们与“接收一个函数”联系起来，实际编写时可以复用这段声明。

函数可能只是外部声明，没有函数体，所以 `fn.isExternal()` 时直接结束。对有函数体的情况，driver 会处理函数内部的操作，包括符合其范围约束的嵌套 Region；它并不是只看入口 Block 中的一层加法。

本例沿用改写驱动篇的观察配置：关闭 folding、常量 CSE 和 Region 简化，使 RHS 非零或 flags 等未匹配测试能检查自己的规则。Greedy 仍有简单死代码清理，因此无用的零常量会被删除。完整机制可回查[改写驱动篇](../compiler/rewrite_drivers)，这里重点看这段调用怎样进入 Pass。

如果 driver 返回 failure，当前实现调用 `signalPassFailure()`，把“没有在限制内完成收敛”作为 Pass 失败处理。规则对某个加法返回 failure 则只是未匹配，不会直接触发这里的失败分支。这两个返回值处在不同层次。普通 Pass 失败也不会自动回滚已经发生的 IR 修改。

## 3. 让命令行名称找到 Pass 的构造方法

现在 C++ 中已经有 `RemoveAddZeroPass` 类，文本 pipeline 却还不知道 `lab-remove-add-zero` 代表什么。

需要注册一项映射：读到这个名称时，创建相应的 Pass 实例。本工程提供一个明确调用的注册函数：

<!-- cpp-example: registration -->
```cpp
void registerLabPasses() {
  PassRegistration<RemoveAddZeroPass>();
}
```

`PassRegistration` 将构造方法交给注册系统，名称与描述来自类中的 `getArgument()` 和 `getDescription()`。注册的作用是让工具能够识别并创建 Pass；实际执行仍要由 pipeline 安排。

接下来写工具入口：

<!-- cpp-example: main -->
```cpp
int main(int argc, char **argv) {
  registerLabPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR learning optimizer\n", registry));
}
```

`MlirOptMain` 提供类似 `mlir-opt` 的命令行、文件读写、解析和 PassManager 驱动。我们提供自己的 Pass 注册和要读取的方言，就能使用现成的基础设施。

这两种注册服务于不同阶段。方言注册让解析器认识 `arith.addi`、`func.func`、`scf.if` 等操作；Pass 注册让 pipeline 解析器认识 `lab-remove-add-zero`。只注册方言，不会得到这个优化；只注册 Pass，也无法完整读取缺失定义的方言操作。

本规则直接复用已有 Value，没有创建新方言操作。以后 Pass 若会创建新的方言操作，应通过 `getDependentDialects` 声明相应依赖，使框架能在运行前准备好它们。当前先完成这条已有操作上的改写链路。

## 4. 构建工具，看到第一次完整结果

工程保存在 `aicompiler-labs/llvm-mlir/04-small-pass/`：

```text
LabPass.cpp       规则、函数 Pass、注册入口
lab-opt.cpp       方言注册与工具 main
CMakeLists.txt    编译并链接 MLIR 库
input.mlir       本章输入
```

CMake 的工作是把前两份 C++ 文件编译成同一个程序，并链接它们调用的 MLIR 库。例如 `MLIROptLib` 提供工具入口，`MLIRPass` 提供 Pass 基础设施，方言库提供操作定义，`MLIRTransforms` 提供这里使用的驱动。完整依赖以配套 CMake 文件为准。

在 workspace 根目录运行，复用已有 LLVM/MLIR 20.1.8 构建：

```bash
cmake -S aicompiler-labs/llvm-mlir/04-small-pass \
  -B artifacts/builds/mlir-small-pass -G Ninja \
  -DMLIR_DIR="$PWD/artifacts/builds/mlir-20.1.8/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/artifacts/builds/mlir-20.1.8/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang
cmake --build artifacts/builds/mlir-small-pass -j 2
```

随后运行自己的工具：

```bash
artifacts/builds/mlir-small-pass/lab-opt \
  aicompiler-labs/llvm-mlir/04-small-pass/input.mlir \
  --pass-pipeline='builtin.module(func.func(lab-remove-add-zero))' \
  --verify-each
```

实际输出为：

<!-- mlir-example: first-pass-output -->
```text
module {
  func.func @twice(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

沿数据流对照：规则把 `%a` 的 use 接回函数参数，再删除产生 `%a` 的旧加法；Greedy 随后清理失去用途的零常量。第二条加法保留，两个 operand 都指向同一个参数，函数仍然计算两倍输入。

`--verify-each` 让 PassManager 在各 Pass 之后检查 IR 的合法性。它能发现类型、结构、支配等约束被破坏，却不能仅凭这份合法输出证明变换保持语义。例如错误地把函数改成直接返回参数，也可能是合法 IR。我们需要为“结果究竟改成了什么”再补测试。

如果希望同时观察输入和输出，可以使用配套 `observe.py`，它会构建并打印真实前后 IR；上述输出已经足够支持正文阅读，无需先运行命令才能继续。

## 5. 沿命令追到那条加法

现在把刚才成功的命令从外向内展开：

```text
main 注册 Pass 和方言
  → MlirOptMain 读取 input.mlir，构造并验证模块
  → pipeline 在 builtin.module 下安排函数层级的 PassManager
  → 对所选 func.func 运行 RemoveAddZeroPass
  → runOnOperation() 取得当前函数
  → Greedy driver 尝试 RemoveAddZero
  → PatternRewriter 转接使用并删除旧加法
  → driver 完成，Pass 返回，工具打印 IR
```

这一条链路说明了为什么本章需要注册、嵌套 pipeline 和 driver：注册解决文本名称到实例的连接，嵌套解决 Pass 的运行对象，driver 负责在函数内部应用局部规则。

我们可以主动拆掉其中一层来验证理解。把 pipeline 写成 `builtin.module(lab-remove-add-zero)`，就相当于直接要求这个函数 Pass 运行在模块上。工具拒绝添加它，诊断包含：

```text
Can't add pass ... restricted to 'func.func' on a PassManager
intended to run on 'builtin.module', did you intend to nest?
```

这里省略了诊断中的 C++ 类名，只保留关键片段。错误发生在 pipeline 构建阶段，尚未走进 `matchAndRewrite`。排查时应修正运行层级，而不是修改加零条件。

如果换成未注册的 Pass 名称，则连构造实例这一步都无法完成。配套测试同时检查这两种错误，确保命令接口与运行对象的约定确实成立。

## 6. 用 FileCheck 约束输出的数据关系

先把第一节的成功输入写成回归测试。只检查“输出包含 `arith.addi`”不够，因为未改写的输入也包含它。我们真正需要确认的是：函数只剩一条加法，它的两个输入都是原函数参数，返回值使用这条加法的结果。

`tests/rewrite.mlir` 对这个函数写了以下检查：

```text
// CHECK-LABEL: func.func @twice(
// CHECK-SAME: %[[X:.*]]: i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[X]], %[[X]] : i32
// CHECK-NEXT: return %[[SUM]] : i32
// CHECK-NEXT: }
```

`CHECK-LABEL` 定位函数，后面的 `CHECK-SAME` 在同一行捕获参数名。`[[X:.*]]` 中的 `X` 是测试变量名，正则匹配实际打印名；后续 `[[X]]` 要求再次出现相同文本。因此 printer 把 `%x` 改名为 `%arg0` 不会影响判据。

`CHECK-NEXT` 要求下一行就是预期内容。这让测试同时排除函数体中多余的常量或加法。这里的输出很短，连续行检查直接表达了完整目标；较复杂的变换应选取稳定的语义关系，避免过度依赖无关的排版或合法调度顺序。

测试文件开头还有一条运行指令：

```text
// RUN: %lab-opt %s --pass-pipeline='builtin.module(func.func(lab-remove-add-zero))' --verify-each | %FileCheck %s
```

这条指令左侧运行编译器，右侧用 FileCheck 检查标准输出。`%s` 表示当前测试文件；该文件同时保存输入 IR 和检查注释。编译器忽略注释，FileCheck 从注释中取得预期。

FileCheck 本身不会发现所有测试文件，也不会自动理解 `%lab-opt` 的路径替换。下一步需要一个测试运行器来执行这些 `RUN` 行。

## 7. 用 lit 把正反例一起运行

LLVM 的 `lit` 发现测试文件、展开替换并运行 `RUN` 指令。本工程的 `tests/lit.cfg.py` 配置了 `.mlir` 与 `.test` 后缀，并把 `%lab-opt`、`%FileCheck` 映射到本地工具。临时结果放入 artifacts 构建目录。

构建完成后执行：

```bash
artifacts/builds/mlir-20.1.8/bin/llvm-lit -v \
  aicompiler-labs/llvm-mlir/04-small-pass/tests
```

目前有三个测试文件，内部覆盖多组输入：

| 测试文件 | 具体检查 | 为何需要 |
|---|---|---|
| `rewrite.mlir` | `twice`、同一 Value 的多个 use、SCF 内部加法、函数声明、再次应用 | 检查真实替换关系、处理范围与稳定输出 |
| `no-match.mlir` | RHS=1、零在左侧、i64、overflow flags、未知 RHS | 检查合法输入中的不匹配情况是否保留 |
| `pipeline.test` | help 中可见、错误运行层级、未知 Pass 名称 | 检查注册与命令接口 |

实际运行三个文件全部通过。计数表示测试文件数量，不等于覆盖了所有程序。这里更值得查看的是每个文件表达的约束。

以 RHS=1 为例，预期输出继续保留常量 1、加法以及对加法结果的返回。它不是非法 MLIR，也不该触发编译失败。同样，左侧为零、i64 和带 flags 的加法只是超出当前规则范围；将来可以在论证后扩展，当前保留它们符合约定。

这些反例使用结果确实被返回的活跃操作。Greedy 可以删除无用操作，因此不能用一个无人使用的加法来证明“规则没有匹配”；它可能通过独立的死代码清理消失。

重复应用检查的是：这组输入运行两次后仍满足预期输出。规则每次删除一条加法，不创建新加法，也给出了它自身不会反复制造机会的终止理由。测试与推理提供的是不同证据，应当结合使用。

## 8. 测试通过之后，怎样判断已经做对

我们已经获得一条完整工程链路，但仍要清楚每份证据说明什么：

| 证据 | 支持的结论 |
|---|---|
| C++ 编译并链接成功 | 实际版本提供这些 API，程序可以构建 |
| 工具识别 pipeline，错误层级被拒绝 | 注册与运行对象的契约生效 |
| verifier 接受输出 | 输出满足所检查的 IR 合法性约束 |
| FileCheck 匹配结果和使用关系 | 这些用例产生约定的 IR；必要计算没有被错误删掉 |
| `x+0=x` 与类型/flags/支配条件的论证 | 说明这条替换为什么保持适用范围内的语义 |

当前替代 Value 是原加法的左 operand。对这里的有效函数体 IR，它在 root 处已经可用，在旧结果的合法使用处也可用；替换没有引入对旧结果的依赖。`replaceOp` 更新全部使用并删除 root，随后不再访问旧句柄。于是算术语义、SSA 可用性和对象生命周期三个方面都有对应理由。

这些测试没有执行生成函数的机器码，也没有测量性能。后续涉及数值算法、布局与内存时，需要补充对应执行证据。当前工程的价值是让你能定位并解释一次真实编译器变换，而不仅是看到一个 Pass 名称。

## 9. 阅读之后做一个有限的修改

现有工程是完整讲解示例。你的任务可以很小：**在保持当前类型和 flags 范围的前提下，让规则也接受 `0+x`。**

先修改 `left_zero` 的期望，确认测试失败；再修改规则，重新运行测试。这样你能看到新需求怎样进入测试，以及实现怎样满足它。随后补一个 `0+0` 用例，解释替代 Value 的选择和终止条件。

还应检查：原来的 `x+0` 是否仍能处理，`x+1` 是否仍保留，同一结果有多个 use 时是否都被更新。最后记录一组前后 IR、适用条件、一个未匹配输入和测试结果。

能够独立完成并解释这次修改，就可以作为阶段 B“实现或修改一个小变换并验证”的证据。之后进入自定义 Op/ODS 与 verifier：我们已经会在已有操作上写优化，下一阶段再定义操作本身的结构和语义。阶段 A 的完成状态不因这个工程增加新要求。

## 完整规则与源码入口

<details>
<summary>展开 RemoveAddZero 的完整定义</summary>

<!-- cpp-example: pattern -->
```cpp
struct RemoveAddZero : OpRewritePattern<arith::AddIOp> {
  explicit RemoveAddZero(MLIRContext *context) : OpRewritePattern(context, 1) {}

  LogicalResult matchAndRewrite(arith::AddIOp op,
                               PatternRewriter &rewriter) const override {
    if (!op.getType().isSignlessInteger(32) ||
        op.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return failure();
    auto constant = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return failure();
    auto value = dyn_cast<IntegerAttr>(constant.getValue());
    if (!value || !value.getValue().isZero())
      return failure();
    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
```

这里直接读取 RHS 的常量定义与整数属性，对应上一章的 `isIntegerConstant` 辅助逻辑；去掉了教学成功日志，运行过程通过输入输出与测试观察。

</details>

完整工程与观察方法见 `aicompiler-labs/llvm-mlir/04-small-pass/README.md`。从 workspace 根目录运行 `python3 aicompiler-labs/llvm-mlir/04-small-pass/observe.py --test`，可以依次看到构建、前后 IR 与 lit 结果。作者维护检查为 `docs/validate_small_pass.py`，它核对本文代码片段和实际工程，再检查正文输入输出与测试。

本章采用本地 LLVM/MLIR 20.1.8。需要沿源码确认时，顺着实际调用查四个入口即可：

- [PassRegistry.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Pass/PassRegistry.h)：`PassRegistration` 如何登记构造方法。
- [MlirOptMain.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp)：解析输入、构建并运行 pipeline 的工具入口。
- [Pass.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Pass/Pass.cpp)：PassManager 的对象约束与执行。
- [TestingGuide](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/TestingGuide.md)、[FileCheck](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/docs/CommandGuide/FileCheck.rst)：测试文件、指令与匹配语义。
