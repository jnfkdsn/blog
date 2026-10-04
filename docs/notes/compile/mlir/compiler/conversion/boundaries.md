---
order: 4
title: 调用与区域边界转换
updated: 2026-10-02
---

# 调用与区域边界转换

一个值的使用者不一定就在定义后面。它可以作为实参进入另一个函数，也可以绑定为区域中的 Block 参数，再由 yield 送回外层。

前两章已经解决了 range 与 i32 的表示选择，以及新旧表示暂时共存的衔接。本章将同一套转换扩展到更长的使用链：**每跨过一个边界，就检查传出的值、接收位置和边界声明是否一致。** 我们仍然转换 `my.limit` 的结果，不引入另一套计算。

## 1. 函数调用中的同一份契约

先增加一个只返回参数的函数 `@keep`：

<!-- range-example: calls -->
```text
module {
  func.func @keep(%v: !my.range<-4, 7>) -> !my.range<-4, 7> {
    return %v : !my.range<-4, 7>
  }
  func.func @caller(%x: i32) -> !my.range<-4, 7> {
    %r = my.limit %x bounds(#my.bounds<-4, 7>) : !my.range<-4, 7>
    %s = func.call @keep(%r) : (!my.range<-4, 7>) -> !my.range<-4, 7>
    return %s : !my.range<-4, 7>
  }
}
```

从 `%r` 向后追踪：它是 call 的实参；被调用函数的 `%v` 接收它；keep 的 return 形成 call 结果 `%s`；caller 再返回 `%s`。

```text
limit 结果 → call 实参 → keep 参数 → keep 返回
                                      ↓
caller 返回 ← call 结果 ←───────────────┘
```

符号名 `@keep` 负责找到函数，但调用时参数和结果怎样解释，由调用点与函数签名共同约束。仅仅保持符号名不变，并不能自动同步所有类型。

## 2. 函数定义、调用点与返回值的协调

如果先把 keep 改成 `(i32) -> i32`，原 call 还声明传入和返回 range，两端就不再遵守同一接口。完整转换需要覆盖：

| 位置 | 原来 | 转换后 |
|---|---|---|
| keep 的参数与返回签名 | range → range | i32 → i32 |
| keep 的入口参数 | `%v : range` | `%v_new : i32` |
| caller 内 limit 的结果 | range | 新 minsi 结果 i32 |
| call 实参和结果 | range | i32 |
| 两个函数的 return | 接收 range | 接收相应 i32 值 |

目标为 call 增加类型条件：

<!-- range-source: call-target -->
```cpp
target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
  return converter.isLegal(op);
});
```

然后复用标准调用转换规则：

<!-- range-source: call-pattern -->
```cpp
populateCallOpTypeConversionPattern(patterns, converter);
```

这条规则根据 converter 计算新的结果类型，使用转换准备好的输入，并保留被调用符号。它与上一章的函数签名、return 规则一起工作，不能用其中一条替代其余部分。

实际输出如下：

<!-- range-output: calls -->
```text
module {
  func.func @keep(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
  func.func @caller(%arg0: i32) -> i32 {
    %c-4_i32 = arith.constant -4 : i32
    %c7_i32 = arith.constant 7 : i32
    %0 = arith.maxsi %arg0, %c-4_i32 : i32
    %1 = arith.minsi %0, %c7_i32 : i32
    %2 = call @keep(%1) : (i32) -> i32
    return %2 : i32
  }
}
```

两个函数仍保留，调用也没有被内联。变化只发生在表示和接口上。对原来满足 range 输入契约的程序，keep 仍返回相同数值；转换后的公开 i32 签名本身不再表达范围前提，调用方仍需遵守原约定。

若当前只转换一个函数，另一个函数来自无法修改的外部库，就不能擅自改变它的 ABI。此时要选择保持接口并提供真实桥接，或者通过明确的包装函数连接两个表示。这是实际工程需要决定的编译边界。

## 3. Region 的入口与出口

再把同一个 limit 结果交给 `my.scope`。它的语义是执行一次内部区域，并将 yield 值返回给父操作；内部不直接捕获外部 SSA 值。

<!-- range-example: scope -->
```text
module {
  func.func @scoped(%x: i32) -> !my.range<-4, 7> {
    %r = my.limit %x bounds(#my.bounds<-4, 7>) : !my.range<-4, 7>
    %s = "my.scope"(%r) ({
    ^bb0(%local: !my.range<-4, 7>):
      my.yield %local : !my.range<-4, 7>
    }) : (!my.range<-4, 7>) -> !my.range<-4, 7>
    return %s : !my.range<-4, 7>
  }
}
```

这里有两条不同的对应关系：

```text
入口：scope 的 operand %r → body 的 Block 参数 %local
出口：yield 的 operand   → scope 的结果 %s
```

入口绑定不是普通的 SSA use 替换：`%local` 有自己的定义位置。它虽然代表传入值，却不是 `%r` 的另一个名字。出口也不是让 yield 产生一个 OpResult，而是由区域执行协议把值交给 scope 的结果。

转换因此要同时维护父操作、Block 参数和 terminator 之间的约定。

## 4. 区域边界的五个位置

对于上面的单输入、单输出例子，逐项列出目标类型：

| 位置 | 转换动作 |
|---|---|
| scope 的 operand | 使用 limit 对应的新 i32 值 |
| body 的 Block 参数 | 从 range 变为 i32，并更新内部引用 |
| yield 的 operand | 使用转换后的 Block 参数 |
| scope 的 OpResult | 类型变为 i32 |
| 外部 return 的 operand | 使用新的 scope 结果，函数签名同步 |

只修改 scope 的结果列表不够：yield 可能仍传出 range。只修改 Block 参数也不够：父操作仍可能接收 range，而定义要求入口类型逐项一致。

本章让 scope 继续存在，仅转换它所传递的类型。我们还没有把 scope 的执行语义降低成另一种控制流或机器代码。

## 5. 构造新的父操作并转换 Region 参数

先获得转换后的结果类型和输入，再创建一个同名的新 scope，将原区域移入它，最后转换区域参数：

<!-- range-source: scope-pattern -->
```cpp
struct ConvertScope : OpConversionPattern<my::ScopeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(my::ScopeOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    OperationState state(op.getLoc(), my::ScopeOp::getOperationName());
    state.addOperands(adaptor.getInputs());
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    state.addRegion();
    auto replacement = cast<my::ScopeOp>(rewriter.create(state));
    rewriter.inlineRegionBefore(op.getBody(), replacement.getBody(), replacement.getBody().end());
    if (failed(rewriter.convertRegionTypes(&replacement.getBody(), *getTypeConverter())))
      return failure();
    rewriter.replaceOp(op, replacement.getOutputs());
    return success();
  }
};
```

可以按两层理解这段代码。

第一层是边界外壳：`adaptor.getInputs()` 取得新输入，`convertTypes` 计算新结果类型，新的 OperationState 保存这些信息。本例 scope 没有需要额外迁移的 Properties；可保留的普通属性复制到新操作。

第二层是内部程序：`inlineRegionBefore` 把已有 Region 中的 Block 移入新操作，保留内部计算；`convertRegionTypes` 使用同一 converter 改写 Block 参数并维护映射。它不会替所有内部操作定义转换规则，yield 等使用者仍需要自己的模式。

最后 `replaceOp` 建立旧 scope 结果与新结果的对应。整个过程通过 ConversionPatternRewriter 完成，使 driver 能管理重建、移动、引用关系及失败时的恢复。它不是对每个 Value 直接 `setType`，也不依赖在修改一半时整份 IR 已经满足最终 verifier。

## 6. Terminator 与动态合法性

yield 没有结果，但它的 operands 参与了父操作的结果契约，因此需要转换：

<!-- range-source: yield-pattern -->
```cpp
struct ConvertYield : OpConversionPattern<my::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(my::YieldOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<my::YieldOp>(op, adaptor.getValues());
    return success();
  }
};
```

这里使用 adaptor 中的新值重建 yield。输入例子里，旧 `%local : range` 对应新的 i32 Block 参数，新 yield 就引用这个参数。

再让目标分别检查父操作和出口：

<!-- range-source: region-target -->
```cpp
target.addDynamicallyLegalOp<my::ScopeOp>([&](my::ScopeOp op) {
  return converter.isLegal(op) && converter.isLegal(&op.getBody());
});
target.addDynamicallyLegalOp<my::YieldOp>([&](my::YieldOp op) {
  return converter.isLegal(op);
});
```

scope 的检查包括普通输入/结果，以及 body 的 Block 参数；yield 的检查包括其传出值的类型。`converter.isLegal(&region)` 检查 Block 参数类型，并不代替每个内部操作的合法化。父子之间的类型匹配、terminator 位置与隔离等约束仍由操作定义及最终 verifier 检查。

本例实际输出为：

<!-- range-output: scope -->
```text
module {
  func.func @scoped(%arg0: i32) -> i32 {
    %c-4_i32 = arith.constant -4 : i32
    %c7_i32 = arith.constant 7 : i32
    %0 = arith.maxsi %arg0, %c-4_i32 : i32
    %1 = arith.minsi %0, %c7_i32 : i32
    %2 = "my.scope"(%1) ({
    ^bb0(%arg1: i32):
      my.yield %arg1 : i32
    }) : (i32) -> i32
    return %2 : i32
  }
}
```

现在沿入口和出口两条线看，都是 i32；scope 仍包围同一段内部程序。对多个输入/结果，按位置应用相同原则；对零个输入/结果，也仍需保留合法的区域和 terminator。

## 7. 从失败定位遗漏边界

配套工程保留了两个只删除一类规则的对照：

| 删除的规则 | 其他规则仍在做什么 | 观察到的失败 |
|---|---|---|
| call 转换 | 函数签名与 limit 已有转换路径 | `failed to legalize operation 'func.call'` |
| yield 转换 | scope 与 Block 参数已有转换路径 | `failed to legalize operation 'my.yield'` |

失败的操作往往就是没有跟上表示变化的接口位置。此时先画出它的输入从哪里来、结果交给谁，再检查对应的类型目标和规则，比不断给目标添加静态 legal 更能定位问题。

多 Block 的 CFG 也有类似边界：`cf.br` 传出的值要与目标 Block 参数匹配。工程另外提供一个 `limit → cf.br → Block 参数 → return` 的例子，使用标准 BranchOpInterface 转换规则验证它。这里只覆盖这条直接分支路径；不同区域控制流协议仍需要按各自语义选择转换方式。

## 理解检查与下一步

如果 scope 的两个结果类型不同，只交换 yield 的顺序，为什么需要同时检查值的位置与类型？如果只将 func 声明为递归合法，为什么可能“转换成功”却完全没有改变函数里的 range？

后一项涉及框架的进阶控制，见[复杂转换路径与框架行为](./advanced)。读完本章，主线已经形成：定义类型映射，转换生产者与使用者，协调函数和区域边界，再检查目标与 IR 约束。接下来可以进入张量计算，不必先通读 driver 实现。

## 实现与查阅

[配套工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/08-type-conversion)包含 calls、scope 和 CFG 输入，以及缺失规则的失败对照。转换后的 scope 仍需要后续实现执行路径；这里验证的是编译器侧结构、类型与引用关系。

固定版本的函数规则见 [FuncConversions.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp)，区域与签名转换入口见 [DialectConversion.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/DialectConversion.h)。
