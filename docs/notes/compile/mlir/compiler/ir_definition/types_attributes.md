---
order: 3
title: 自定义 Attribute 与 Type：参数与结果契约
updated: 2026-10-01
---

# 自定义 Attribute 与 Type：参数与结果契约

前两章用两个整数属性保存 clamp 的上下界，接口再读取它们，告诉消费者结果范围。现在考虑两个不同的设计需求：多个操作都要保存同样结构的区间；结果离开定义操作后，仍希望其类型明确表达范围保证。

第一个需求适合用自定义 Attribute 组织静态参数，第二个涉及自定义 Type。本章分别建立它们的用途，再将两者放回同一条计算中。它们是可选择的 IR 设计工具，不能因为已经实现一个 Dialect，就推断还必须定义新的类型和属性。

## 1. 区间参数与自定义 Attribute

原来的表示是：

```text
%r = lab.clamp %x bounds(-4, 7) : i32
```

其中 lower 与 upper 是两个独立整数属性。若许多操作都使用区间，每个操作都需要组织两个字段，并检查顺序与范围。可以将这组静态数据定义为一个有明确含义的属性对象：

```text
#lesson.bounds<-4, 7>
```

它表示一对有序的、可由有符号 i32 表达的上下界。它本身不执行 clamp，也不产生 SSA Value；操作持有这个对象，用它决定自己的计算参数。

实际 ODS 定义为：

<!-- source-example: bounds-attr -->
```text
def BoundsAttr : AttrDef<Lesson_Dialect, "Bounds"> {
 let mnemonic = "bounds";
 let parameters = (ins "int64_t":$lower, "int64_t":$upper);
 let assemblyFormat = "`<` $lower `,` $upper `>`";
 let genVerifyDecl = 1;
}
```

`lower` 和 `upper` 是属性参数，`mnemonic` 决定文本里的 bounds 名字，`assemblyFormat` 决定两个参数在尖括号中的顺序。`int64_t` 是保存参数的 C++ 类型；本例仍要求其数值落在 i32 的有符号范围内，下一节验证机制会补足这个约束。

这样，操作可以拥有一个 bounds 属性，而不用各自定义两份整数。共享属性对象的意义在于复用这组数据的结构与合法性规则，计算含义仍由使用它的操作规定。

## 2. 结果范围与自定义 Type

现在换到使用者一侧。一个值的类型如果只有 i32，类型本身只表达整数位宽等信息，不包含“它一定落在 [-4,7]”这一保证。范围接口可以回到定义操作查询，但也可以选择把范围纳入值的类型：

```text
!lesson.range<-4, 7>
```

本教学类型的语义是：一个具有有符号 i32 整数含义、且数值保证落在指定闭区间内的值。它不规定目标机器上的新数据布局；后续仍需定义怎样转换和执行这种表示。

类型的 ODS 为：

<!-- source-example: range-type -->
```text
def RangeType : TypeDef<Lesson_Dialect, "Range"> {
 let mnemonic = "range";
 let parameters = (ins "int64_t":$lower, "int64_t":$upper);
 let assemblyFormat = "`<` $lower `,` $upper `>`";
 let genVerifyDecl = 1;
}
```

形式与属性很像，但两者所处的位置不同：

| 对象 | 附着在哪里 | 本例表达什么 |
|---|---|---|
| `#lesson.bounds<-4,7>` | 操作的静态字段 | 此次 clamp 使用哪个区间 |
| `!lesson.range<-4,7>` | 结果 Value 的类型 | 使用者可以依赖怎样的数值保证 |

两者都保存两个整数，却不能互相替代。只知道结果位于 `[-4,7]`，不能决定它究竟怎样由 x 计算而来；很多不同计算都可能产生该范围内的值。属性在本例中规定计算参数，类型描述结果契约。

## 3. 操作参数与结果类型的组合

将两项设计放回一个新的操作 `lesson.limit`：

<!-- irdef-example: types-input -->
```text
module {
  func.func @clip(%x: i32) -> !lesson.range<-4, 7> {
    %r = lesson.limit %x bounds(#lesson.bounds<-4, 7>) : !lesson.range<-4, 7>
    return %r : !lesson.range<-4, 7>
  }
}
```

从左到右，`%x` 是普通 i32 输入；bounds 属性给出区间；`%r` 的类型携带结果保证。以输入 12 作语义推演，操作应把它限制为 7，因此输出满足 `!lesson.range<-4,7>`。

实际定义如下：

<!-- source-example: limit-op -->
```text
def LimitOp : Op<Lesson_Dialect, "limit", [Pure, DeclareOpInterfaceMethods<StaticBounds>]> {
 let summary = "Clamp an i32 and return a range-refined integer value";
 let arguments = (ins I32:$input, BoundsAttr:$bounds);
 let results = (outs RangeType:$result);
 let assemblyFormat = "$input `bounds` `(` $bounds `)` attr-dict `:` type($result)";
 let hasVerifier = 1;
}
```

`BoundsAttr:$bounds` 与 `RangeType:$result` 分别约束字段种类。StaticBounds 接口继续提供公共查询，所以前一章的消费者仍能向操作索取范围。

本例选择了一个方便检查的强约定：结果类型的两个端点必须与 bounds 属性完全相等。这会在文本中重复写一遍范围，但清楚地区分了参数和结果类型的责任。之后可以改进构造与类型推导，减少这种重复书写。

仅检查字段种类还不能保证两组范围一致。下面要把对象自身的合法性与它们组合使用的合法性分开。

## 4. 参数验证与操作验证

### 4.1 类型和属性的参数约束

BoundsAttr 和 RangeType 各自要求：

```text
INT32_MIN <= lower <= upper <= INT32_MAX
```

无论某个对象被哪条操作使用，`<8,7>` 都违反这项约定。两种对象的参数验证因此共享一个辅助检查：

<!-- source-example: type-attr-verify -->
```cpp
LogicalResult RangeType::verify(function_ref<InFlightDiagnostic()> emitError, int64_t lower, int64_t upper) {
 return verifyBounds(emitError, lower, upper);
}
LogicalResult BoundsAttr::verify(function_ref<InFlightDiagnostic()> emitError, int64_t lower, int64_t upper) {
 return verifyBounds(emitError, lower, upper);
}
```

`verifyBounds` 比较上下界顺序和 i32 范围，失败时报告 `expected ordered bounds within signed i32 range`。ODS 中的 `genVerifyDecl = 1` 请求生成这些验证方法的声明，实现由我们提供。

这一层先保证“单个区间对象成立”。例如 `[7,7]` 合法，但下界 8、上界 7 不合法。

### 4.2 不同字段之间的一致性

下面的两个区间各自合法：

<!-- irdef-invalid: mismatched-bounds | result range must match bounds attribute -->
```text
module {
  func.func @bad(%x: i32) -> !lesson.range<-4, 8> {
    %r = lesson.limit %x bounds(#lesson.bounds<-4, 7>) : !lesson.range<-4, 8>
    return %r : !lesson.range<-4, 8>
  }
}
```

问题在于属性要求限制到 `[-4,7]`，结果类型却写成 `[-4,8]`，不符合本例选择的完全一致规则。操作验证检查的是这层关系：

<!-- source-example: limit-verify -->
```cpp
LogicalResult LimitOp::verify() {
 auto range = getResult().getType();
 auto bounds = getBounds();
 if (range.getLower() != bounds.getLower() || range.getUpper() != bounds.getUpper())
   return emitOpError("result range must match bounds attribute");
 return success();
}
int64_t LimitOp::getMinimum() { return getBounds().getLower(); }
int64_t LimitOp::getMaximum() { return getBounds().getUpper(); }
```

前半段取得结果类型与属性，逐项比较端点；后面的两个方法实现 StaticBounds，将属性里的范围交给通用消费者。

`[-4,7]` 确实包含在 `[-4,8]` 中，从数值保证看，放宽范围并非必然不安全。但我们的操作定义选择了更严格的相等约定，便于避免不一致和推导歧义。Verifier 检查的是明确制定的契约，设计者可以选择其他规则，但必须一并说明语义与消费者如何理解它。

验证顺序由此清楚了：先确认参数对象和字段种类，再检查操作中几份信息的对应关系。类型正确不等于操作正确，操作验证通过也不自动证明后续 lowering 的数值实现正确。

## 5. 类型身份与共享存储

一万个 Value 都可能使用 `!lesson.range<-4,7>`。若每个 Value 都复制一份区间数据，既浪费存储，也不利于比较。MLIR 因此在同一个 Context 中按类型种类与参数共享存储，这一机制称为 uniquing。

对于当前类型，可以沿一次请求理解：

```text
请求 RangeType(-4, 7)
  → 按 RangeType 种类及参数 (-4,7) 查找
  → 已有存储：返回引用它的句柄
  → 尚无存储：创建并登记，再返回句柄
```

实际 C++ 观察为：

<!-- source-example: uniquing -->
```cpp
auto a = lesson::RangeType::get(&context, -4, 7);
auto b = lesson::RangeType::get(&context, -4, 7);
auto c = lesson::RangeType::get(&context, -4, 8);
llvm::outs() << "same_parameters=" << (a == b) << " different_parameters=" << (a == c) << "\n";
```

输出：

```text
same_parameters=1 different_parameters=0
```

相同参数的两次请求得到相等句柄，改变一个端点则得到另一种类型。这里的 `get` 取得一个类型对象，不是创建一条 IR 操作，也不计算运行时整数。

这些普通不可变类型和属性的存储由 Context 管理。复制句柄不复制全部参数，调用者不应单独 delete 它；Context 销毁后句柄不能继续使用。想换一个范围，应请求另一种类型并合法地调整相关 IR，而不是修改共享对象中的 lower。

如果把参数扩展成字符串或数组，还需要保证存储拥有足够长的生命周期。`StringRef`、`ArrayRef` 本身是非拥有视图，不能让类型存储长期引用解析器的临时缓冲区；参数分配与复制规则需要与 uniquing 一起设计。当前两个整数按值保存，所以不存在这项额外问题。

## 6. 合法构造与 checked 构造

先前请求的三个 RangeType 都合法。若参数来自输入文本或其他尚未验证的数据，就可能遇到 `<8,7>`，这时需要能返回失败的构造入口：

<!-- source-example: checked-construction -->
```cpp
auto invalid = lesson::RangeType::getChecked(
    [&]() { return emitError(UnknownLoc::get(&context)); }, &context, int64_t(8), int64_t(7));
llvm::outs() << "invalid_is_null=" << !invalid << "\n";
```

实际输出 `invalid_is_null=1`，同时报告上下界非法。调用者由空句柄知道构造失败，应停止继续创建依赖该类型的对象。

对本例而言，`get` 用于调用者已经保证参数合法的情况；非法参数可能触发断言，不能把 Release 中未必存在的断言当作输入验证。`getChecked` 则显式执行参数验证，以诊断和空句柄表示失败。生成的文本 parser 使用 checked 路径，因而非法区间会在构造类型或属性时被拒绝。

这与上一章操作 builder 的行为不同。操作可以先被构造，再接受完整 IR 验证；类型或属性的 checked 构造会先检查自己的参数。操作里多份信息是否一致，仍需操作 verifier 另行确认。

## 7. 类型相等、范围包含与转换

数学上，`[-4,7]` 是 `[-4,8]` 的子集；在本例的普通类型相等检查中，两组参数不同，因此是不同类型。MLIR 不会因为区间包含关系自动给它们建立子类型规则。

这会带来实际影响：

```text
%r : !lesson.range<-4,7>
  → 不能自动当作 !lesson.range<-4,8>
  → 也不能自动交给只接受已有整数类型的 arith.addi
```

若需要放宽范围或转为 i32，必须规定相应操作或转换，并正确处理使用者、函数签名和区域边界。不能只改一个打印字符串，把尚不匹配的其他对象留在原处。

因此，自定义类型的收益与成本相伴：契约可以跟随 Value 显式传播，但生产者、消费者和 lowering 都要支持它。另一种设计是保留 i32，将范围存在分析结果中；这更容易复用已有操作，但分析需要维护传播和失效。

本章提供的是一个类型设计例子，并不主张所有数值范围都应进入类型系统。

## 8. 结果类型推导

回到第 3 节的重复端点。既然操作要求类型与属性完全一致，构造者可以从属性生成结果类型：

```text
BoundsAttr(-4,7)
  → 读取两个端点
  → 请求 RangeType(-4,7)
  → 用该类型创建操作结果
```

专用 builder 可以封装这个过程；希望通用基础设施也能请求推导时，可以提供 `InferTypeOpInterface`。计算方式仍需作者实现，字段同名不会自动建立关系。

推导回答“应该生成什么类型”，验证回答“现有对象是否满足约定”。即使有推导，文本仍可能给出显式类型，其他 C++ 调用者也可能构造对象，所以跨字段验证仍有意义。本工程保留显式类型，尚未实现该推导接口。

## 理解检查

依次判断三种变化：属性变成 `<8,7>`；属性不变但结果类型改成 `<-4,8>`；新增操作声明产生 `!lesson.range<-4,7>`，实际 lowering 却返回 100。前两项可由本章哪一层验证发现？第三项为什么还需要语义实现及行为检查？

后面的 [Region 操作](./regions_assembly)讨论把一段计算放入操作内部。它先使用普通整数，不要求先熟悉本章的类型存储实现。

## 实现与依据

完整代码和观察程序见[IR 定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/06-ir-definition)，按 LLVM `llvmorg-20.1.8` 验证参数、字段对应、句柄比较及失败构造。当前没有自定义类型的机器码 lowering。

[AttributesAndTypes 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/AttributesAndTypes.md)说明声明与存储规则；[StorageUniquerSupport.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/StorageUniquerSupport.h)用于确认 `get` 和 `getChecked`。高级可变存储、Type/Attribute Interface 等按具体设计需求继续展开。
