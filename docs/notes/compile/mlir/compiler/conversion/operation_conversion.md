---
order: 1
title: Dialect Conversion：操作转换与目标合法性
updated: 2026-10-02
---

# Dialect Conversion：操作转换与目标合法性

假设编译器的前一阶段保留了 `my.clamp`，方便表达“将一个整数限制在给定区间”。接下来的阶段只准备处理基本算术计算，因此需要把 clamp 展开成常量、最大值和最小值操作。

我们已经知道怎样写这条替换规则。现在还需要解决一个阶段之间的协作问题：**什么时候可以宣布转换完成，把 IR 交给下一阶段？** 如果某条 clamp 没被处理，输入程序可能依然符合自己的定义，却超出了下一阶段接受的范围。

本章沿同一个 clamp 实例走完这个过程：先确定转换后的计算，再规定允许保留的操作，最后将规则与完成条件一起交给转换框架。MLIR 将这套机制称为 **Dialect Conversion**。它也能约束同一方言内部的表示变化，不只用于把操作名字换到另一个方言。

## 1. 编译阶段的输入与输出

### 1.1 保留高层操作的输入

下面的函数接收一个 i32，将它限制到有符号区间 `[-4, 7]`：

<!-- conversion-example: input -->
```text
module {
  func.func @clip(%x: i32) -> i32 {
    %r = my.clamp %x bounds(-4, 7) : i32
    return %r : i32
  }
}
```

`%x` 是计算时才能取得的值，`-4` 和 `7` 是已经保存在操作属性中的静态参数。`my.clamp` 约定：输入小于下界时返回下界，大于上界时返回上界，否则返回输入；上下界必须满足 `lower <= upper`。

这是一段合法的源 IR。它的 verifier 可以确认字段和类型正确、上下界有序。但 verifier 没有理由要求程序中不存在 clamp，因为这个操作本来就是当前表示允许表达的计算。

### 1.2 下一阶段接受的表示

目标阶段希望看到下面的计算。为便于追踪，这里使用有含义的 SSA 名字：

<!-- conversion-example: expanded -->
```text
module {
  func.func @clip(%x: i32) -> i32 {
    %lower = arith.constant -4 : i32
    %upper = arith.constant 7 : i32
    %bounded = arith.maxsi %x, %lower : i32
    %r = arith.minsi %bounded, %upper : i32
    return %r : i32
  }
}
```

变化集中在函数体内：一个 clamp 展开成两条常量和两条算术操作，函数参数和返回类型保持 i32。`func.func`、`func.return` 和外层 module 继续保留，承载相同的程序结构。

本次转换的约定可以完整写成一句话：

> 保持函数的计算含义，将 clamp 展开成指定的 arith 操作；转换结束后，参与转换的 IR 必须全部属于本阶段允许的操作集合。

这里的“目标”是一个编译阶段接受的 IR 形式，并不表示已经到了 GPU 指令或机器码。多个阶段可以依次采用不同的目标。

## 2. 计算展开与数据流替换

### 2.1 从区间语义推导算术操作

对满足 `lower <= upper` 的上下界，有：

```text
clamp(x, lower, upper) = min(max(x, lower), upper)
```

先取最大值，保证中间结果不小于下界；再取最小值，保证最终结果不大于上界。两次比较都必须使用**有符号整数**解释，这就是本例选择 `arith.maxsi` 和 `arith.minsi` 的原因。

用三个输入推演 `[-4,7]` 的计算：

| 输入 x | max(x, -4) | min(中间结果, 7) | 对应 clamp 的分支 |
|---|---|---|---|
| -9 | -4 | -4 | 小于下界，返回 -4 |
| 2 | 2 | 2 | 在区间内，返回输入 |
| 12 | 12 | 7 | 大于上界，返回 7 |

这张表是语义推演；编译器改写 IR 时，并没有执行这三次函数调用。对于任意合法上下界，也可以按“小于下界、位于区间、大于上界”三种情况得到同样的结论。上下界相等时，两次操作最终总会选出该边界。

如果错误使用无符号最大值，`-4` 的位模式会被解释为一个很大的正整数，以上推导就不成立。即使生成的 IR 类型全部正确，转换也可能改变结果。因此，写出规则前仍需确认计算语义。

### 2.2 属性、Value 与使用者的对应

转换时要处理的不只是操作名称。原来两个静态属性需要成为 arith 操作使用的 SSA 值，而原结果的使用者需要接到新的计算结果：

```text
原来的表示：
  输入 %x ──→ my.clamp ──→ 原结果 ──→ return
               ↑
           属性 -4、7

新的表示：
  属性 -4 → constant → %lower ──┐
  输入 %x ─────────────────────┴→ maxsi → %bounded ──┐
  属性  7 → constant → %upper ───────────────────────┴→ minsi → 新结果 → return
```

这里有三项实际工作：创建常量、创建算术操作、更新原结果的使用。打印文本即使继续使用 `%r`，内存中也已经是新的结果 Value；SSA 名字不决定对象身份。

上一模块的普通 Pattern 已经能表达这三项工作。Dialect Conversion 继续使用这种局部规则，同时增加了关于整个阶段的完成条件。

## 3. ConversionTarget 与操作合法性

### 3.1 两种不同的检查

考虑一种故障：作者忘记将 clamp 的展开规则加入规则集合。一次普通改写过程可能没有匹配到任何内容，正常结束，并原样留下 `my.clamp`。

这不能单凭“没有修改”判断为程序错误：许多优化本来就允许没有适用机会。但对当前转换阶段，留下 clamp 意味着工作没有完成。要区分这两种情况，需要把目标要求写出来。

MLIR 用 `ConversionTarget` 保存这样的约定。本章声明：

<!-- conversion-source: target -->
```cpp
target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
target.addLegalOp<arith::ConstantOp, arith::MaxSIOp, arith::MinSIOp>();
target.addIllegalOp<my::ClampOp>();
```

第一行保留程序容器和函数边界，第二行接受展开所需的算术操作，第三行要求消除 clamp。这里的 `illegal` 是“该阶段不允许留下”，并不表示源程序违反了 clamp 自己的定义。

| 检查 | 问题 | 当前输入中的 clamp |
|---|---|---|
| IR verifier | 这个实例是否符合操作定义和通用 IR 约束？ | 合法：类型正确，-4 ≤ 7 |
| ConversionTarget | 这个实例是否是当前转换阶段接受的最终形式？ | 不合法：本阶段要求展开它 |

同一个操作在前者看来合法，在后者看来不合法，是渐进降低过程中的正常情况。转换的任务就是在保持含义的同时跨过这个边界。

### 3.2 允许父操作不等于允许整个子树

`func.func` 必须保留，否则还需要额外提供函数的转换规则。不过，允许函数本身留下，不会让函数体中的 clamp 自动获得许可。转换仍要检查嵌套操作。

因此，上面的目标能同时表达“函数外壳保持”和“函数体中的 clamp 必须消失”。只有显式采用递归合法性设置时，才会把某个操作的整个子树作为已接受的范围；本章不采用这种设置。

目标也没有生成代码的能力。声明 clamp 不合法后，如果没有规则将它转成可接受的计算，框架只能报告失败，不会从操作名推导出 max/min。

## 4. ConversionPattern 与结果替换

现在完成条件已经明确，可以把第 2 节的展开写成转换规则：

<!-- conversion-source: pattern -->
```cpp
struct ClampToArith : OpConversionPattern<my::ClampOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(my::ClampOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    auto lower = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getLowerAttr());
    auto upper = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getUpperAttr());
    auto bounded = rewriter.create<arith::MaxSIOp>(op.getLoc(), adaptor.getInput(), lower);
    rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, bounded, upper);
    return success();
  }
};
```

前两行构造从属性取得的常量；第三行将当前输入与下界比较；最后的替换创建 minsi，并把 clamp 原结果对应到新结果。这与普通 Pattern 中熟悉的计算展开一致。

区别在于，这条规则工作在转换过程里：`OpConversionPattern` 接收转换框架提供的输入，`ConversionPatternRewriter` 将创建和替换纳入该过程的管理。修改必须经过 rewriter，框架才能跟踪产生了哪些操作、原结果由谁替代，以及这条转换路径是否满足目标。

对于这种不改变类型的展开，也可以向转换框架提供普通 RewritePattern；完成标准来自 target 与转换 driver。这里采用 `OpConversionPattern`，是为了明确接入转换输入，并为后续类型变化保留相同的规则组织方式。

### 4.1 原操作与 adaptor

规则里为什么既有 `op`，又有 `adaptor`？单看一条 clamp，它们的输入似乎没有区别。再增加一条依赖前者的 clamp：

```text
%a = my.clamp %x bounds(-4, 7) : i32
%b = my.clamp %a bounds(-2, 3) : i32
```

第一条规则把 `%a` 替换成一个新 minsi 的结果。处理第二条时，新的 maxsi 应当使用这个替代值，才能接入已经转换的计算链。

`op` 用于访问被匹配操作的定义信息，例如原来的上下界属性；`adaptor` 提供转换框架为当前规则准备的输入值。于是，规则可以继续从 `op` 读取 `lower`、`upper`，从 `adaptor.getInput()` 取得接到新计算上的输入。

这件事现在还没有改变类型，却已经涉及原 Value 与替代 Value 的对应。后续发生类型转换时，这个区分会更重要。当前无需深入映射表的实现，也不要把 adaptor 理解成“又复制了一条 clamp”。

### 4.2 规则成功与阶段成功

末尾的 `success()` 表示这条规则已经完成自己的匹配和改写。它还不能说明整段转换成功：新建的 arith 操作是否符合目标、其他位置是否还留下不能处理的操作，都需要继续检查。

反过来，一条规则返回 `failure()` 也不一定立刻使整个阶段失败；可能还有其他规则能够完成这项转换。最终要看框架是否能将要求处理的操作合法化。

本例没有额外的匹配条件，是因为所有通过 verifier 的 `my.clamp` 都满足 max/min 展开的前提。若只支持某种属性或形状，就应在规则中检查条件，并在不满足时报告不匹配。

## 5. Pass 中的一次完整转换

Pass 仍负责组织这项工作。本例运行在 module 上，其核心逻辑如下：

<!-- conversion-source: pass -->
```cpp
void runOnOperation() override {
  ConversionTarget target(getContext());
  configureClampTarget(target);
  RewritePatternSet patterns(&getContext());
  populateClampPatterns(patterns);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

`configureClampTarget` 填入第 3 节的目标声明；`populateClampPatterns` 用 `patterns.add<ClampToArith>(patterns.getContext())` 加入第 4 节的规则。这里先使用 `applyFullConversion`，要求本次范围内的操作最终全部被目标接受。

把这段代码放回最初的 `@clip`，可以沿下列过程理解它的决定：

| 当前对象或动作 | 根据什么作决定 | 结果 |
|---|---|---|
| module 和 func.func | 目标允许这两种容器 | 保留外壳，继续处理内部 |
| my.clamp | 目标明确禁止保留 | 寻找可用的展开规则 |
| 应用 ClampToArith | 字段与语义满足规则前提 | 创建两条常量及 max/min，记录结果替换 |
| 检查产生的操作 | 目标接受 constant、maxsi、minsi | 这条展开路径成立 |
| func.return | 操作被允许，返回值接到新结果 | 函数结构与结果类型保持 |
| 完成转换 | 整个范围满足 full conversion 的要求 | 返回成功，Pass 可以结束 |

这是用于理解责任与数据流的过程表，不要求依赖内部每条创建、替换的提交时机。转换框架会管理改写路径；使用者应通过它的 API 修改 IR。

实际工具打印得到下面的结果，自动选择的 SSA 名字与第 1 节不同，但引用关系相同：

<!-- conversion-output: expanded -->
```text
module {
  func.func @clip(%arg0: i32) -> i32 {
    %c-4_i32 = arith.constant -4 : i32
    %c7_i32 = arith.constant 7 : i32
    %0 = arith.maxsi %arg0, %c-4_i32 : i32
    %1 = arith.minsi %0, %c7_i32 : i32
    return %1 : i32
  }
}
```

结果只包含本阶段接受的操作。框架不要求把它们进一步优化成某个唯一形式，也不负责执行函数。

## 6. 转换失败的两条路径

### 6.1 没有消除源操作

保持同一个输入和目标，只把规则集合留空。输入仍能通过 verifier，但 full conversion 报告：

```text
failed to legalize operation 'my.clamp'
```

原因已经可以从前面推导：目标要求 clamp 消失，而当前没有可用路径消除它。Pass 将这个失败传给 pipeline，后续依赖转换结果的阶段就不能当作它已完成而继续。

这与 `bounds(8, 7)` 被 verifier 拒绝不同。后者是输入本身违反定义；这里是合法输入无法转换到所声明的目标。排查时应先判断失败属于哪一层，再检查字段、规则或目标。

### 6.2 新操作没有获得目标许可

再恢复规则，但从目标中漏掉 `arith.constant`。规则仍然知道怎样创建 max/min，然而它需要的两个常量没有获得目标许可，也没有其他规则能将常量继续变成被允许的操作。

因此，整个 clamp 展开路径仍然无法成立。这说明定义规则时需要同时观察两端：

```text
源端：哪些操作必须消除？
路径：现有规则会生成哪些操作？
目标端：这些操作能直接被接受，还是需要继续转换？
```

规则不一定一步生成最终合法形式，也可以经过中间表示，再由其他规则处理。本例只用一层展开，目的在于让目标、规则和结果之间的关系清楚可见。

目标合法性也不会证明展开的算术含义正确。如果把最后一步误写成 maxsi，目标仍可能接受它；第 2 节的语义论证和相应行为测试依然有独立价值。

## 7. Partial 与 Full Conversion

### 7.1 同一规则，不同的阶段承诺

假设输入除了 clamp，还包含一条加法：

<!-- conversion-example: mixed -->
```text
module {
  func.func @clip_then_add(%x: i32) -> i32 {
    %r = my.clamp %x bounds(-4, 7) : i32
    %s = arith.addi %r, %x : i32
    return %s : i32
  }
}
```

本章目标只列出了三种 arith 操作，没有说明 `arith.addi` 应当保留还是消除。工具当然认识这条加法，它也符合自己的定义；**未知的是它在当前 ConversionTarget 中的合法性**，不是它没有注册到 MLIR。

此时可以提出两种不同的阶段要求：

- 当前只负责清除明确禁止保留的 clamp，输入中其他未表态的操作允许留给后续阶段。这对应 partial conversion。
- 当前要为整段 IR 提供明确的可接受形式，不能留下未获目标许可的操作。这对应 full conversion。

在同一个目标与规则集合下，分别调用：

```cpp
applyPartialConversion(operation, target, std::move(patterns));
// 或者，在另一次独立转换中：
applyFullConversion(operation, target, std::move(patterns));
```

partial 成功：clamp 展开，原有 addi 留下，其输入改接到新的 minsi 结果。full 失败：addi 未被目标接受，也没有对应转换规则。这里两种调用是独立对照，并非将同一个已移动的规则集合连续使用两次。

### 7.2 未知与明确不合法的区别

将前面几个实验放在一起，观察实际成功条件：

| 输入与配置 | Partial | Full | 原因 |
|---|---|---|---|
| 只有 clamp，目标和展开规则齐全 | 成功 | 成功 | 最终操作均被接受 |
| 有 clamp，但没有展开规则 | 失败 | 失败 | 明确禁止保留的 clamp 无法消除 |
| clamp 后有未表态的 addi | 成功，保留 addi | 失败 | 对原有未知操作的要求不同 |
| 有展开规则，但漏掉目标中的 constant | 失败 | 失败 | 新建常量无法合法化，clamp 的转换路径不成立 |

因此，partial 不表示“尽量做，失败也算成功”，也不是“只执行一轮”。它仍要求完成那些明确规定不能留下的部分。对本章缺少常量许可的例子，也不能将新生成的常量当作原有未知操作而放过。

有了这一区分，选择模式取决于阶段职责：是只处理一类操作，还是要保证整个范围都进入规定的表示。full 成功也只表示满足声明的目标；如果目标允许某种高层操作，full 不会强迫它降低到机器级。

## 8. 动态合法性与实例条件

前面的合法性按操作种类决定。实际目标还可能对同一种操作的不同实例提出要求，例如：只接受 i32 的 maxsi，暂不接受 i64 的 maxsi。

这时可以将 maxsi 的目标规则写成：

<!-- conversion-source: dynamic-target -->
```cpp
target.addDynamicallyLegalOp<arith::MaxSIOp>([](arith::MaxSIOp op) {
  return op.getType().isInteger(32);
});
```

`dynamic` 表示在编译期间检查当前 IR 实例的字段，不是在目标程序运行时判断实参。本例查询结果类型；其他规则可以检查形状、属性或更具体的表示条件。

最初的 clamp 仍产生 i32 maxsi，符合目标。下面的函数则有另一种结果：

<!-- conversion-example: wide -->
```text
module {
  func.func @wide(%a: i64, %b: i64) -> i64 {
    %r = arith.maxsi %a, %b : i64
    return %r : i64
  }
}
```

这段 IR 可以通过 verifier。若使用原来的静态合法声明，转换会保留它；改用上述动态规则后，谓词返回 false，本实例需要被转换。当前没有规则处理它，因此 partial 和 full 都失败。

这个例子与 addi 的区别很关键：addi 是目标没有表态，i64 maxsi 是目标明确检查后拒绝。合法性回调返回 false 不能靠换成 partial 来绕过。

动态规则只负责判定，它不会自动将 i64 截成 i32。如何表示原值、如何修改其使用者、怎样保持含义，仍需要转换设计与实现。

## 9. 从操作转换到类型转换

到这里，已经可以完整解释一次转换：输入是合法的源 IR；规则保持计算含义地改变表示；目标规定阶段完成条件；driver 组织合法化，Pass 将成功或失败交给 pipeline。

本章所有展开都保留 i32，所以函数参数、返回类型和使用者的类型无需变化。回到此前的 `my.limit`，情况就不同了：它产生的是 `!my.range<-4,7>`，后续算术操作却希望使用普通 i32。

如果只创建一条 i32 结果操作，原来要求 range 类型的使用者和函数签名还没有处理。这将使“怎样替换一条操作”扩展为“怎样让整条数据流及其边界使用一致的新表示”。下一章[类型转换与函数签名](./type_conversion)围绕这项工作引入 TypeConverter、adaptor 和函数边界转换。

阅读后可以先用三个变化检查自己的推理：

1. 将 `my.clamp` 在目标中声明为合法，却仍加入展开规则。为什么不能再依靠这个 conversion 来保证 clamp 消失？
2. 对第 7 节的输入，明确将 addi 声明为合法或不合法，partial/full 的结果分别会怎样改变？
3. 两条 clamp 前后相接时，第二条展开后的输入应当来自哪里？为什么只检查输出中没有 `my.clamp` 还不足以验证替换正确？

## 实现与查阅

本章实现以 LLVM `llvmorg-20.1.8` 核对。正文提供理解所需的 IR、关键 C++ 和结果；[配套工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/07-dialect-conversion)提供构建、实际 Pass 运行，以及相同输入下更换目标和模式的对照。工程验证解析、目标合法性、数据流和预期失败，不把 IR 打印当作机器码的数值执行。

需要确认精确契约时，可阅读固定版本的 [Dialect Conversion 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DialectConversion.md)和 [DialectConversion.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/DialectConversion.h)。若要追查 partial 为什么拒绝动态 false，可定位 [DialectConversion.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Utils/DialectConversion.cpp) 中的 `ConversionTarget::isIllegal` 与 `OperationConverter::convert`，先沿这一个具体判断阅读。

类型转换、materialization、函数/Region 边界和更复杂的转换路径属于后续内容，见[本模块路线](./)。当前没有使用递归合法性，也没有要求通读转换框架内部实现。
