---
order: 3
title: 自定义 Type 与 Attribute：把领域信息放进 IR
updated: 2026-09-14
---

# 自定义 Type 与 Attribute：把领域信息放进 IR

`lab.clamp %x bounds(-4, 7) : i32` 保存了限制区间，但它的结果类型仍是普通 i32。沿 use-def 找到定义操作时，分析可以查询上一章的范围接口；若这个值穿过函数参数等边界，单看类型便不再知道它具有这个范围。

这一章做一次明确的抽象设计：**用属性表达操作要求的限制区间，用类型表达计算结果向后续使用者提供的范围保证。** 然后沿文本解析、参数验证、Context 内存储与操作验证走完一条链。这里的自定义类型服务于教学中的领域 IR；实际编译器也可能保留 i32，把范围放在分析状态中，类型细化不是每项优化的必经之路。

## 1. 同一个区间，在操作与值上承担不同职责

新操作 `lesson.limit` 仍执行有符号 i32 clamp，但结果使用专门的范围类型：

<!-- irdef-example: types-input -->
```text
module {
  func.func @clip(%x: i32) -> !lesson.range<-4, 7> {
    %r = lesson.limit %x bounds(#lesson.bounds<-4, 7>) : !lesson.range<-4, 7>
    return %r : !lesson.range<-4, 7>
  }
}
```

先按含义阅读，不急着看 C++：

- `%x` 是本次执行得到的 i32 值，属于运行时数据。
- `#lesson.bounds<-4, 7>` 是静态属性，规定这次计算将输入限制到哪个闭区间。
- `!lesson.range<-4, 7>` 是结果类型，表示一个具有 i32 整数含义、且有符号值保证处在该区间内的值。
- 函数返回类型携带相同的保证，调用边界因而能表达这种约束。

例如输入为 12，按操作语义结果是 7；输入为 -2，结果为 -2。这是我们规定的操作执行含义。**写出一个带范围的类型不会在运行时插入比较、截断或检查。** 编译器作者还需在 lowering 中实现这些计算，并确保每个能产生这种类型的操作都维持承诺。

正常打印这个操作时，生成器已知道属性必须是 BoundsAttr、结果必须是 RangeType，可以省略它们的种类前缀，得到 `lesson.limit %arg0 bounds(<-4, 7>) : <-4, 7>`。上面的完整写法也可解析；函数签名等通用位置仍写 `!lesson.range`。这种缩写改变的是文本冗余，不改变对象种类。

范围信息在这里出现两次是有意的。属性回答“操作做哪种计算”，类型回答“结果是什么样的值”。本例采用一个简单契约：结果类型的上下界必须与属性完全一致。后面可以从属性推导类型，减少重复书写，但不能因为希望语法简洁就省略契约本身。

## 2. 定义参数化类型，先决定什么算同一种类型

我们把 `!lesson.range<L, U>` 限定为有序的、可由有符号 i32 表示的整数闭区间。它没有独立位宽参数：i32 是这个教学类型的固定语义。

<!-- source-example: range-type -->
```text
def RangeType : TypeDef<Lesson_Dialect, "Range"> {
 let mnemonic = "range";
 let parameters = (ins "int64_t":$lower, "int64_t":$upper);
 let assemblyFormat = "`<` $lower `,` $upper `>`";
 let genVerifyDecl = 1;
}
```

这份定义包含三个相互配合的决定：

1. `TypeDef` 指定它是 Lesson dialect 下的一种类型，生成 C++ `RangeType`。
2. `lower`、`upper` 是类型身份的参数。上下界不同，得到不同类型。
3. `assemblyFormat` 规定参数的文本形式；`genVerifyDecl` 请求生成参数验证函数的声明。

`!lesson.range` 中的 `lesson` 标识方言，`range` 来自 mnemonic。角括号里的两个整数按声明顺序解析。没有 SSA operand，没有 IR 中的计算，也没有为每个使用它的 Value 生成一个“类型操作”。

接着用相同的两个参数定义属性：

<!-- source-example: bounds-attr -->
```text
def BoundsAttr : AttrDef<Lesson_Dialect, "Bounds"> {
 let mnemonic = "bounds";
 let parameters = (ins "int64_t":$lower, "int64_t":$upper);
 let assemblyFormat = "`<` $lower `,` $upper `>`";
 let genVerifyDecl = 1;
}
```

`AttrDef` 生成 `BoundsAttr`。`#` 与 `!` 的不同不是装饰：parser 需要知道此处正在读 Attribute 还是 Type，它们进入不同的对象体系。两个对象即使保存相同的整数，也不能互相替代。

## 3. 类型本身合法，与操作使用它合法，是两层检查

先检查每个区间对象独立成立所需的条件：

```text
INT32_MIN ≤ lower ≤ upper ≤ INT32_MAX
```

RangeType 与 BoundsAttr 的参数验证共享这个规则，核心入口如下：

<!-- source-example: type-attr-verify -->
```cpp
LogicalResult RangeType::verify(function_ref<InFlightDiagnostic()> emitError, int64_t lower, int64_t upper) {
 return verifyBounds(emitError, lower, upper);
}
LogicalResult BoundsAttr::verify(function_ref<InFlightDiagnostic()> emitError, int64_t lower, int64_t upper) {
 return verifyBounds(emitError, lower, upper);
}
```

`verifyBounds` 在完整源码中比较上下界与 i32 范围，失败时发出 `expected ordered bounds within signed i32 range`。因此 `!lesson.range<8, 7>` 和 `#lesson.bounds<-4, 2147483648>` 各自就不合法，无需先把它们放进 `lesson.limit`。

再把两种对象放到操作中：

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

ODS 生成的约束先确认输入是 i32、属性是 BoundsAttr、结果属于 RangeType。这还不够。下面的两种范围各自都合法，却不满足本操作“完全一致”的约定：

<!-- irdef-invalid: mismatched-bounds | result range must match bounds attribute -->
```text
module {
  func.func @bad(%x: i32) -> !lesson.range<-4, 8> {
    %r = lesson.limit %x bounds(#lesson.bounds<-4, 7>) : !lesson.range<-4, 8>
    return %r : !lesson.range<-4, 8>
  }
}
```

这个跨字段关系由操作 verifier 检查，而非类型 verifier：

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

最后两个成员函数还完成上一章的接口实现：消费者通过 `StaticBounds` 取得上下界，不必知道字段已经换成自定义属性。

沿完整解析过程整理一下：文本先生成并验证类型/属性参数；操作构造出输入、静态数据和结果；操作验证再检查字段种类与彼此关系。每一层拒绝不同种类的错误。`verify` 通过也只说明已实现的这些规则成立，不能证明未来的 lowering 一定正确执行 clamp。

## 4. 为什么反复 get 不会反复复制一个类型对象

如果一万个值都使用 `!lesson.range<-4, 7>`，每个值保存一份独立区间结构既浪费空间，也让类型比较更昂贵。MLIR 在同一个 Context 中对类型和属性进行 uniquing：按种类及参数查找存储，已有就复用，否则创建并保存。

以本章生成的 RangeType 为例：

```text
RangeType::get(context, -4, 7)
  → 用 (-4, 7) 作为本类型的存储 key 查询
  → 已存在：返回指向既有 storage 的轻量句柄
  → 不存在：分配、初始化并登记 storage，再返回句柄
```

C++ probe 展示这个过程在对象比较上的结果：

<!-- source-example: uniquing -->
```cpp
auto a = lesson::RangeType::get(&context, -4, 7);
auto b = lesson::RangeType::get(&context, -4, 7);
auto c = lesson::RangeType::get(&context, -4, 8);
llvm::outs() << "same_parameters=" << (a == b) << " different_parameters=" << (a == c) << "\n";
```

实际输出是：

```text
same_parameters=1 different_parameters=0
```

两项比较的含义是：相同 Context 中，相同类型种类与参数得到相等的句柄；改一个参数就得到不同类型。可打开生成的 `LessonTypes.cpp.inc`，查看 `RangeTypeStorage` 的 `KeyTy`、比较、hash 和构造函数，它们把这一机制落实成存储代码。

句柄不是由调用者 `delete` 的独立类型对象。普通不可变参数存储由 Context 管理，Context 销毁后，保留的句柄也不能继续使用。本章定义没有可变存储部分，不能通过给 `a` 的 lower 赋新值来修改所有共享类型；应请求另一类型，再显式处理相关 IR 变化。

跨 Context 的同文本类型不具有这里的句柄身份保证。不能把不同 Context 的对象任意混装进同一份 IR。

## 5. get 与 getChecked 怎样选择

刚才三次请求的参数都合法。现在把条件改成下界 8、上界 7：这个区间不存在，我们希望得到可处理的失败，而不是一个非法类型。probe 的下一步因此改用 checked 构造：

<!-- source-example: checked-construction -->
```cpp
auto invalid = lesson::RangeType::getChecked(
    [&]() { return emitError(UnknownLoc::get(&context)); }, &context, int64_t(8), int64_t(7));
llvm::outs() << "invalid_is_null=" << !invalid << "\n";
```

实际结果是 `invalid_is_null=1`，同时出现上下界非法的诊断。调用者拿到空句柄，便能停止构造依赖这个类型的操作。

对于带参数验证的本章类型：

- `get` 面向调用者已经保证参数合法的构造，非法参数可能触发断言；Release 构建不能靠断言承担可靠的输入校验。
- `getChecked` 调用验证，失败时发出诊断并返回空句柄，调用者必须处理这个结果。
- 本章生成的文本 parser 使用 checked 路径，从而将坏参数转成解析阶段的诊断。

这与上一章“Operation builder 可以先构造，完整 verifier 另行调用”的讨论有联系，但不能机械等同。Type/Attribute 的参数合法性会参与 checked 构造；Operation 中诸如上下界字段与结果类型对应的关系，仍由操作验证负责。

## 6. 一个整数类型有子集，不代表 MLIR 自动拥有子类型系统

数学上，`[-4, 7]` 是 `[-4, 8]` 的子集。但本章两个 RangeType 的参数不同，所以 MLIR 的普通类型相等检查会认为它们不同。

把 `%r : !lesson.range<-4, 7>` 直接放到要求 `!lesson.range<-4, 8>` 的位置，不会自动生成“安全放宽”。把它交给 `arith.addi` 也不会自动当作 i32 使用：arith 的操作约束不认识这个自定义标量类型。

这意味着类型设计同时创造了后续工作：

1. 明确哪些操作可生产、消费这种类型。
2. 如需范围放宽，定义有清楚语义和验证规则的操作或转换。
3. 降到已有整数表示时，转换结果类型、使用者以及函数/区域边界。
4. 保证新表示仍执行原先计算，而不是只把类型字符串换掉。

也可以选择让范围只存在于分析状态中，保留所有值为 i32。这样更容易复用现有操作，但跨边界传播和分析失效维护由分析负责。类型和分析是两种不同设计工具，应由抽象边界决定选择。

## 7. 回到起点：能否只写一次区间

现在已经知道，bounds 属性规定 clamp 怎样计算，RangeType 则描述结果保证。主例要求两组参数完全一致，因此构造时可以先读取属性中的上下界，再请求同参数的 RangeType，最后用这个类型创建结果。这样调用者只需提供一次区间，操作内部仍保留各自承担职责的属性与类型。

这就是一个简单的结果类型推导过程：

```text
BoundsAttr(-4, 7)
  → 读取 lower=-4、upper=7
  → 请求 RangeType(-4, 7)
  → 以此作为新操作的结果类型
```

专用 builder 可以执行这个过程。若希望通用工具也能请求推导，则通过 `InferTypeOpInterface` 提供协议。计算逻辑仍由作者实现，字段同名并不会让 ODS 自动理解它们的关系。

推导回答“应构造什么类型”，验证回答“当前对象是否满足契约”。从文本读入显式类型，或由其他 C++ 调用者创建操作时，仍然需要后者。因而主例保留严格相等的 verifier；减少书写不等于放松语义要求。

## 阅读后的一个小推演

把主例的结果类型改为 `!lesson.range<-4, 8>`，属性保持不变：RangeType 的参数验证会通过，LimitOp 的字段关系验证会失败。再把属性改为 `#lesson.bounds<9, 8>`：这次还没到跨字段对应检查，属性自身的参数已经非法。

最后思考：如果新增一个声称能产生 RangeType 的操作，却实际返回区间外数值，谁能发现？普通类型相等和本章参数 verifier 无法检查其所有动态执行。需要正确的操作语义、实现，以及相应转换与数值验证。这也是为什么自定义类型的核心是建立契约，而不只是增加一种文本写法。

<details>
<summary>实现时查阅：Properties、复杂参数的所有权与 C++ 重载</summary>

**属性与 Properties 的关系。** BoundsAttr 是 Context 中可共享的静态数据对象。在这个固定版本生成的 Op 实现中，作为 inherent 字段的 `bounds` 由操作的 Properties 保存对应属性句柄。Properties 是操作拥有的数据容器，Attribute 是其中可引用的一类数据；两者不是两套互斥的属性格式。使用生成的访问器读字段，可避免把“字段一定在普通字典里”写死在消费者中。

**把两个整数换成字符串或数组。** 当前 `int64_t` 按值保存，没有外部缓冲区寿命问题。如果新增 `StringRef`/`ArrayRef` 参数，裸 C++ 引用可能指向解析器的临时内存。应选用具有合适分配/复制规则的 `StringRefParameter`、`ArrayRefParameter` 等参数描述，或实现定制存储构造；同时明确比较与 hash 使用哪些内容。否则 uniquing 得到的“长期存储”可能引用已经失效的数据。

C++ 示例给 `getChecked` 传入显式 `int64_t` 实参，以对应生成重载的参数类型。这里不要求记模板重载细节；构造报错时，查生成声明比猜参数类型更直接。

</details>

## 配套观察与依据

`06-ir-definition/observe.py` 展示 `types.mlir`、通用格式、范围查询以及 `type-probe` 的实际输出。本例保留显式结果类型；本节的类型推导为设计推演，尚未实现 InferType 接口。高级自定义/可变存储留待实际参数需求展开。运行 `docs/validate_ir_definition.py` 可回归合法/非法参数、字段对应、函数返回类型不匹配和通用格式往返；这些是编译器侧证据，未执行 `lesson.limit` 的目标代码。

固定版本 LLVM `llvmorg-20.1.8` 的查阅入口：

- [AttributesAndTypes.md](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/AttributesAndTypes.md)：AttrTypeDef、参数存储、builders 与验证。
- [StorageUniquerSupport.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/StorageUniquerSupport.h)：`get`、`getChecked` 与存储共享入口。
- [AttrTypeBase.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/AttrTypeBase.td)：参数、生成器与 parser/printer 控制项。
- 配套 build 的 `LessonTypes.cpp.inc`、`LessonAttrs.cpp.inc`、`LessonOps.h.inc`：本例实际生成代码。

下一章[定义带 Region 的操作](./regions_assembly)把这个计算放入一个新操作的 Region，解释新类型怎样沿区域输入与结果协议流动。
