---
order: 4
title: Region 操作：执行协议、传值与验证
updated: 2026-10-01
---

# Region 操作：执行协议、传值与验证

前面的 clamp 把一项标量计算表达为操作。函数、循环和设备执行区域还需要容纳一段内部程序：操作不仅有输入和结果，也拥有 Region。

本章定义一个教学用的 `lesson.scope`，用来包装并执行一次内部计算。重点是确定四件事：什么时候进入区域、输入怎样绑定、结果怎样返回，以及哪些结构与引用合法。例子先使用普通 i32，类型与属性的自定义存储不参与这条主线。

## 1. 区域语义与数据传递

我们为 scope 约定：它有一个非空 Region，其中恰好一个 Block；进入时将输入按位置绑定到 Block 参数，执行其中的计算，最后把 `lesson.yield` 的值传给父操作结果。它不允许内部直接捕获外部 SSA 值。

下面把已经熟悉的 clamp 放进去：

<!-- irdef-example: scope-basic -->
```text
module {
  func.func @clip_inside(%x: i32) -> i32 {
    %r = "lesson.scope"(%x) ({
    ^bb0(%local: i32):
      %clipped = lab.clamp %local bounds(-4, 7) : i32
      lesson.yield %clipped : i32
    }) : (i32) -> i32
    return %r : i32
  }
}
```

通用语法的 `(%x)` 是 scope 输入列表，`({ ... })` 包含它的 Region，末尾 `(i32) -> i32` 描述父操作的输入和结果类型。

假设函数输入为 12，按我们规定的执行语义，值的传递为：

| 位置 | 数值 | 在 IR 中的角色 |
|---|---:|---|
| `%x` | 12 | 父操作的输入引用 |
| `%local` | 12 | 内部 Block 的入口参数 |
| `%clipped` | 7 | 区域内 clamp 的结果，也是 yield 的输入 |
| `%r` | 7 | 父操作结果，随后交给 return |

这四个名字对应不同的 Value。入口与出口处的数值联系来自 scope 的执行约定，不是因为这些 Value 自动互为别名。

这张表是语义推演。当前工程能够解析和验证这种表示，但尚未提供 scope 的执行器或 lowering；读取文本不会实际执行表中的计算。

## 2. Region 结构与 ODS 定义

Region 本身提供容器，不规定执行次数或输入输出绑定。`scf.for` 可以重复执行内部区域，`scf.if` 选择分支，函数体则在调用时执行。我们给 scope 规定执行一次，需要在定义与后续实现中保持这项约定。

操作声明如下：

<!-- source-example: scope-ops -->
```text
def ScopeOp : Op<Lesson_Dialect, "scope", [SingleBlock, IsolatedFromAbove, RecursiveMemoryEffects]> {
 let arguments = (ins Variadic<AnyType>:$inputs);
 let results = (outs Variadic<AnyType>:$outputs);
 let regions = (region AnyRegion:$body);
 let hasVerifier = 1;
 let hasRegionVerifier = 1;
}
def YieldOp : Op<Lesson_Dialect, "yield", [Pure, Terminator, HasParent<"ScopeOp">]> {
 let arguments = (ins Variadic<AnyType>:$values);
 let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}
```

`regions` 声明一个名为 body 的 Region；`Variadic<AnyType>` 分别声明一组输入和一组结果。本例的组长度为一，稍后可以变成零或多个。AnyType 表示不把元素类型固定为 i32，入口和出口的对应关系仍由 verifier 检查。

`SingleBlock` 限制 Region 的 Block 数量；本版本该 trait 允许零或一个 Block，所以本操作还要补充“body 非空”的检查。`IsolatedFromAbove` 禁止相应的外部 SSA 捕获。

`lesson.yield` 是区域的出口。`Terminator` 给它 Block 终结操作的结构角色，`HasParent<"ScopeOp">` 限制其所在父操作。即使 yield 没有普通结果，它也不能像无用的算术操作一样被删除，否则区域将失去约定的出口。

这些结构设施让框架能够检查 IR 的形状；“进入一次、按位置绑定、按位置传出”仍是 scope 的语义协议，需要后续执行或 lowering 落实。

## 3. 入口验证与区域验证

### 3.1 入口参数对应

已有一个 i32 的 scope 输入，并不足以保证区域里恰好声明了一个 i32 参数。可以误写为 i64，也可以多写一个参数。因此需要检查两条序列：

```text
scope 的输入类型序列 == 入口 Block 的参数类型序列
```

先检查 body 非空，之后才安全取得入口 Block。

### 3.2 出口结果对应

内部操作验证完成后，再确认 Block 最后确实是 lesson.yield，并检查：

```text
yield 的输入类型序列 == scope 的结果类型序列
```

这要求读取内部操作，所以放在区域验证阶段。实际实现为：

<!-- source-example: scope-verify -->
```cpp
LogicalResult ScopeOp::verify() {
 if (getBody().empty())
   return emitOpError("requires a nonempty body");
 Block &entry = getBody().front();
 if (entry.getArgumentTypes() != getInputs().getTypes())
   return emitOpError("entry argument types must match input types");
 return success();
}
LogicalResult ScopeOp::verifyRegions() {
 Block &entry = getBody().front();
 if (entry.empty())
   return emitOpError("requires a lesson.yield terminator");
 auto yield = dyn_cast<YieldOp>(entry.back());
 if (!yield)
   return emitOpError("requires a lesson.yield terminator");
 if (yield.getValues().getTypes() != getOutputs().getTypes())
   return emitOpError("yield types must match result types");
 return success();
}
```

普通 `verify()` 检查外壳和入口；`verifyRegions()` 在内部操作经过验证后读取出口。这样的顺序让后面的检查能够使用前面已经建立的前提，例如 body 存在、内部 yield 的字段可以合法访问。

比较类型序列也同时比较长度。只检查第一个类型会漏掉多结果、遗漏参数和空列表等情况。

### 3.3 出口不匹配的反例

<!-- irdef-invalid: scope-yield-mismatch | yield types must match result types -->
```text
module {
  func.func @bad(%x: i32) -> i64 {
    %r = "lesson.scope"(%x) ({
    ^bb0(%local: i32):
      lesson.yield %local : i32
    }) : (i32) -> i64
    return %r : i64
  }
}
```

入口的 i32 对应没有问题，但父操作宣称结果为 i64，yield 却交出 i32。工具因此报告 `yield types must match result types`。这不是 parser 无法读取语法，而是读出的结构没有满足操作协议。

## 4. 区域隔离与显式依赖

回到第一节，把内部 clamp 的输入从 `%local` 改成外部 `%x`。在那一次数值推演中它们都是 12，但两种写法在依赖结构上不同：

```text
显式绑定：外部 x → scope operand → 内部参数 local → clamp
直接捕获：外部 x ────────────────────────────→ clamp
```

本例选择隔离，因此后一种写法违反 `IsolatedFromAbove`。这样一来，区域需要的 SSA 输入集中在 scope 的 operand 列表中，克隆或独立处理区域时更容易确定依赖。

隔离不自动证明任意搬移都是安全的，也不禁止所有形式的外部引用。它约束 SSA 捕获；若内部使用符号调用函数，还需遵守符号可见性及调用语义。标准 SCF 的许多区域允许捕获，不能把本例的设计推广成所有 Region 的统一规则。

## 5. 可变参数与多结果传递

将输入组和结果组各扩展为两个值，不改变刚才的基本协议：

<!-- irdef-example: scope-swap -->
```text
module {
  func.func @swap(%x: i32, %y: i64) -> (i64, i32) {
    %r:2 = "lesson.scope"(%x, %y) ({
    ^bb0(%a: i32, %b: i64):
      lesson.yield %b, %a : i64, i32
    }) : (i32, i64) -> (i64, i32)
    return %r#0, %r#1 : i64, i32
  }
}
```

进入时，x 对应 a，y 对应 b；退出时，第一个 yield 输入 b 对应父结果 0，第二个 a 对应父结果 1。所以入口类型为 `(i32,i64)`，出口类型为 `(i64,i32)`，各自在自己的边界上对应。

不能为了省事给 scope 加上“所有输入与结果同类型”的约束，那会排除这个合法程序。约束应来自执行协议，而不是看到多个值就套用一个现成 trait。

长度也可以为零：

<!-- irdef-example: scope-empty -->
```text
module {
  func.func @empty() {
    "lesson.scope"() ({
      lesson.yield
    }) : () -> ()
    return
  }
}
```

这里没有输入和结果，但 Region 仍有一个 Block 和所需出口。零个结果、空 Block、空 Region 是三种不同结构。yield 的声明式格式使用可选组，在 values 为空时省略值与类型，得到裸 `lesson.yield`。

### 多组可变字段的分段问题

本例每个 operand/result 列表只有一个可变组，边界明确。如果同一个 operand 列表包含两组可变字段，仅知道总数无法判断各组长度：三个值可能分成 1+2，也可能分成 2+1。

这时需要额外的分段约定，例如 `AttrSizedOperandSegments`，或在语义确实要求等长时使用 `SameVariadicOperandSize`。多组 optional/variadic 的布局及访问器必须依据这种约定生成；本工程没有实现多组分段，不能把单组的结果直接当作验证证据。

## 6. 结构验证、效果与控制流接口

现在表示能够通过结构、类型和隔离验证，但通用分析仍需知道怎样理解内部计算。

`RecursiveMemoryEffects` 表示需要考察内部操作的效果。外层只是容器，并不意味着内部任意读写都可以被忽略；yield 自身的无内存效果也不能抹去前面操作的效果。

另一类分析要知道控制和数据怎样进入、离开区域。它不会从“恰好一个 Block”自动推导出“执行一次”，也不会仅凭入口和出口类型相等就知道 Value 的映射。相关消费者通常通过 `RegionBranchOpInterface` 等协议获取这些信息。

因此可以把几种职责放回同一条 scope：

| 层次 | 向工具提供什么 |
|---|---|
| 结构与自定义 verifier | 当前对象是否符合约定的形状和对应关系 |
| 效果信息 | 内部是否有分析或优化必须考虑的行为 |
| 区域控制流接口 | 通用分析所需的进入、退出及值传递关系 |
| lowering 或执行实现 | 实际落实“进入一次并传出结果”的语义 |

当前工程实现前两类，后两类留给后续任务。定义完整的语义不等于全部消费者已经实现对它的理解。

## 理解检查

在 swap 示例中，只交换 yield 的两个输入，保持父结果类型不变：哪一步验证会失败？再删除空结果示例的 yield：为什么“没有结果要传出”仍不足以允许缺少出口？

如果将第一节内部的 clamp 换成返回自定义范围类型的 limit，入口与出口可以使用不同类型，只需在各自边界上对应。这样可以复用本章协议，而不必在第一次理解区域时同时掌握类型存储机制。

## 实现与依据

本章新增的普通整数示例和已有复杂类型、零/多结果用例均由 LLVM `llvmorg-20.1.8` 下的[IR 定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/06-ir-definition)验证。传值表为语义推演，未执行 scope 的目标代码。

[ODS 验证顺序](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)、[Verifier.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/IR/Verifier.cpp)和 [ControlFlowInterfaces.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/ControlFlowInterfaces.td)分别用于确认验证阶段及后续区域协议。下一章[解析与打印](./assembly_format)将专门解释表示与文本的对应。
