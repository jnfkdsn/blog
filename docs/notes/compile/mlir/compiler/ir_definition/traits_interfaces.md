---
order: 2
title: Trait 与 Interface：让通用代码理解操作
updated: 2026-09-14
---

# Trait 与 Interface：让通用代码理解操作

上一章的 `lab.clamp` 已经能被读取、构造、验证，还能由专用 Pattern 展开。不过，编译器里大量代码并不认识 `lab.clamp`：通用死代码删除不应为每个方言增加一个操作名单，区间分析也不应逐一硬编码所有带范围信息的操作。

这一章沿着两个决策展开：**一个没有用户的操作能否删除；一个不认识操作名字的消费者怎样获得结果范围。** 前者使用已有的效果接口，后者实现一个小接口。最终要看清的是操作定义怎样把语义事实交给通用代码使用。

## 1. 两个结果都没人用，为什么只删除一个

先沿用上一章的有符号 i32 闭区间限制。下面两种操作约定相同的计算：小于 -4 返回 -4，大于 7 返回 7，否则返回输入。`lesson.opaque_clamp` 是本章特意定义的另一种表示，它有输入、属性、结果及上下界验证，但没有声明效果或推测执行信息。

<!-- irdef-example: interfaces-input -->
```text
module {
  func.func @unused(%x: i32) {
    %a = lab.clamp %x bounds(-4, 7) : i32
    %b = "lesson.opaque_clamp"(%x) <{lower = -4 : i32, upper = 7 : i32}> : (i32) -> i32
    return
  }
}
```

手工分析很容易：两项计算的结果都没有使用者；这些标量计算不写内存、不发出 I/O，也没有合法输入上的额外可观察行为，删除它们不改变函数的可观察结果。

通用代码面对的事实却更少。它可以遍历 use-list，确认两项结果都无人使用；仅凭 Operation 的输入、结果和属性，它看不出操作是否打印日志、更新设备状态，或做了别的事情。因此，**“没有 use”只能解决计算结果是否有用，不能解决执行这个操作是否有用。**

配套工具先调用 `isOpTriviallyDead` 报告判断，再运行 `canonicalize`。先只看报告中的 `dead` 字段（下面摘录这一列）：

```text
lab.clamp            dead=1
lesson.opaque_clamp  dead=0
```

实际清理后的函数仍含有 `lesson.opaque_clamp`，`lab.clamp` 已消失。`dead=0` 表示工具在这里没有证明它是可平凡删除的死操作；它不表示数学上已经证明该操作有副作用。

为什么已有的 `lab.clamp` 能被识别？上一章在 ODS 中给了它 `Pure`。

## 2. 从 Pure 走到删除决定

在本工作区使用的 MLIR 中，`Pure` 组合了 `NoMemoryEffect` 和 `AlwaysSpeculatable`。它向基础设施提供两类不同的信息：

| 信息 | 消费者能据此询问什么 |
|---|---|
| 无内存效果 | 该操作报告哪些内存读写、分配、释放效果？对于这里的 clamp，集合为空 |
| 总是可以推测执行 | 将本来可能不执行的计算提前执行，是否会引入禁止的行为？这里所有合法 i32 输入的 clamp 都有定义 |

这些是操作作者依据语义作出的承诺。ODS 不会分析一段未来的机器代码，再证明声明属实。若把写设备状态的操作错误标成无效果，结构 verifier 可能通过，优化却会错误删除它。

本例的删除决策可以逐步还原：

1. 检查 `lab.clamp` 的所有结果，没有 use。
2. 它不是必须保留的 terminator 等结构性操作。
3. 查询内存效果接口，得到空效果集合。
4. 这条操作满足这里的平凡死代码条件，清理过程可以删除它。

对 `lesson.opaque_clamp`，第 3 步无法取得效果模型，工具走保守路径，保留操作。

这里有一个容易混淆的边界：**本例的平凡 DCE 判断并不是先调用 `isPure`，也不是要求同时通过一项 speculation 检查。** `isOpTriviallyDead` 检查 use，再调用处理效果与结构边界的辅助逻辑。它还处理某些读、分配及递归效果情形，不能把完整算法缩写成“所有非 Pure 操作都不能删除”。

`AlwaysSpeculatable` 在移动计算、把条件执行变成无条件执行等决策中另有作用。例如，除法结果无人使用和把除法提前到分支外是两个不同问题；后者可能让原来不会遇到的非法除数被执行。学习接口时应始终把“提供的事实”和“当前消费者的决策”分开。

## 3. Trait 与 Interface 分别填补哪一层信息

现在回头看两种扩展机制，就有了具体用途。

**Trait 把一项共性附着到操作定义上。** 比如 `SingleBlock` 约束区域的 Block 结构，`IsolatedFromAbove` 约束区域向外捕获 SSA 值。一个 trait 可以包含验证逻辑和共享方法；它不只是没有行为的标签。通用代码可通过 `hasTrait<...>()` 查询某类共性。

**Interface 定义一套消费者可以调用的协议。** 比如 `MemoryEffectOpInterface` 的消费者能请求效果集合，每种操作用自己的数据回答。通用代码拿到的是接口视图，不必知道该操作具体属于哪个方言。

两者不是互斥选项。ODS 会通过 trait 形式将接口接入生成的操作类，`Pure` 本身又是若干已有能力的组合。设计时最有用的区别是：

- 若要复用同一项结构约束或共享实现，先考虑 trait。
- 若消费者需要向不同对象提出同一个问题，并得到对象相关的回答，先考虑 interface。

不要据此认为 trait 只能回答“是/否”，或 interface 一定涉及复杂虚函数代码。我们接下来只让一个接口返回两个整数。

## 4. 先规定范围问题，再定义接口

假设后续分析想知道操作的结果范围。例如，它看到一个结果与常量 100 作有符号小于比较，如果结果保证位于 `[-4, 7]`，便具备了将比较判为真的一项事实。

这个消费者需要的契约是：

> 对一个具有单一整数含义结果的操作，返回一个静态、闭合、可靠的有符号范围；所有有定义执行得到的结果都落在这个范围中。边界不要求是最紧的估计。

本例只为单个具有整数含义的结果提供已知范围；取得接口后，两项查询都必须有可靠答案。没有这项能力的操作由消费者识别为未知。

协议确定之后，ODS 才有东西可以声明：

<!-- source-example: bounds-interface -->
```text
def StaticBounds : OpInterface<"StaticBounds"> {
 let cppNamespace = "::mlir::lesson";
 let methods = [
  InterfaceMethod<"Inclusive signed lower bound of the result.", "int64_t", "getMinimum">,
  InterfaceMethod<"Inclusive signed upper bound of the result.", "int64_t", "getMaximum">
 ];
}
```

这段定义生成 `lesson::StaticBounds` 接口类以及实现模型所需的结构。它尚未让任何操作自动获得上下界。还需要分别提供“每个操作怎样回答”和“哪个消费者来问”。

## 5. 在操作中实现，再走完一次查询

先在自己维护的操作中实现协议。我们定义 `lesson.clamp`，计算语义、输入、上下界属性和结果都沿用上一章的 `lab.clamp`，只增加范围接口。这一步不需要自定义类型。

<!-- source-example: direct-op -->
```text
def ClampOp : Op<Lesson_Dialect, "clamp", [Pure, DeclareOpInterfaceMethods<StaticBounds>]> {
 let arguments = (ins I32:$input, I32Attr:$lower, I32Attr:$upper);
 let results = (outs I32:$result);
 let assemblyFormat = "$input `bounds` `(` $lower `,` $upper `)` attr-dict `:` type($result)";
 let hasVerifier = 1;
}
```

`DeclareOpInterfaceMethods<StaticBounds>` 将接口接到操作类，并生成两项方法的声明。生成器知道消费者可以调用什么，但怎样从字段中得到答案，要由操作实现：

<!-- source-example: direct-methods -->
```cpp
LogicalResult ClampOp::verify() {
 if (getLowerAttr().getValue().sgt(getUpperAttr().getValue()))
   return emitOpError("requires lower <= upper");
 return success();
}
int64_t ClampOp::getMinimum() { return getLowerAttr().getInt(); }
int64_t ClampOp::getMaximum() { return getUpperAttr().getInt(); }
```

前面的 verifier 保证上下界有序。后面两个方法读取已经验证的 i32 属性，返回带符号整数。这里能直接把属性当作结果范围，是因为 clamp 的计算语义保证输出不会超出上下界；换一种操作，就必须重新证明答案可靠。

现在给这个操作一个具体输入模块：

<!-- irdef-example: direct-input -->
```text
module {
  func.func @direct(%x: i32) -> i32 {
    %r = lesson.clamp %x bounds(-4, 7) : i32
    return %r : i32
  }
}
```

消费端的核心代码如下，`op` 是当前遍历到的 `Operation *`：

<!-- source-example: consumer -->
```cpp
if (auto bounds = dyn_cast<lesson::StaticBounds>(op))
  llvm::errs() << " bounds=[" << bounds.getMinimum() << "," << bounds.getMaximum() << "]";
else
  llvm::errs() << " bounds=unknown";
```

`dyn_cast` 在这里尝试取得接口；成功后调用统一方法，失败则报告未知。消费端没有 `if (isa<lesson::ClampOp>(op))`，也没有直接读取名为 `lower` 的属性。这样才能容纳用其他字段保存相同事实的操作。

对这份输入，报告工具实际给出：

```text
lesson.clamp effect_interface=1 speculatable=1 dead=0 bounds=[-4,7]
```

沿一次调用追踪：Pass 遍历到 `lesson.clamp` → `dyn_cast` 取得它支持的 StaticBounds 接口 → 调用接口方法 → 分派到 ClampOp 成员函数 → 读出 lower、upper → 消费者得到 `[-4,7]`。结果被 return 使用，所以 `dead=0`；这不影响范围查询。

到这里，协议、实现、消费者和实际回答已经接起来。之后新增一个用其他字段保存范围的操作，只需提供满足相同协议的实现，报告程序便可继续使用。

## 6. 当操作属于另一个组件时，补上外部模型

刚才可以直接修改自己维护的 `lesson.clamp`。现在换一个条件：要让上一章独立工程中的 `lab.clamp` 也支持查询，但不希望基础方言反过来依赖这个工具的范围接口。此时把实现放在工具侧，通过外部模型接入：

<!-- source-example: external-model -->
```cpp
struct ClampBoundsModel : lesson::StaticBounds::ExternalModel<ClampBoundsModel, lab::ClampOp> {
 int64_t getMinimum(Operation *op) const { return cast<lab::ClampOp>(op).getLowerAttr().getInt(); }
 int64_t getMaximum(Operation *op) const { return cast<lab::ClampOp>(op).getUpperAttr().getInt(); }
};
```

模型拿到原始操作，从已经验证的 i32 属性中读取带符号整数。模型的方法多了 `Operation *` 参数，是因为这次实现不放在 `ClampOp` 成员函数中；接口调用的底层分派会把目标对象交给模型。

随后把模型接到使用它的 Context：

<!-- source-example: attach -->
```cpp
registry.addExtension(+[](MLIRContext *context, lab::LabDialect *) {
  lab::ClampOp::attachInterface<ClampBoundsModel>(*context);
});
```

这里的 registry extension 会在相关 dialect 加载时应用。把注册封装在 registry 中，工具不必猜测 Lab dialect 何时首次加载。**接口模型的可用性与 Context 的注册状态有关**，并不是链接进一个 `.cpp` 后，所有 Context 就自动知道该接口。

配套工程从同一份源码构建两个工具：一个添加扩展，一个不添加。它们读取完全相同的 `interfaces.mlir`，查询同一个 `lab.clamp`：

| 工具配置 | 范围查询 | 平凡死代码判断 |
|---|---|---|
| 注册外部范围模型 | `bounds=[-4,7]` | `dead=1` |
| 不注册外部范围模型 | `bounds=unknown` | `dead=1` |

范围信息缺失并没有改变已有的效果声明，所以本例 DCE 的结果不受影响。这个对照说明：不同消费者使用不同协议，添加一个范围接口不会自动修改所有优化的行为。

直接实现把方法放在操作类中；外部模型把方法放在另一个组件中，再接到 Context。两种方式最终供同一个消费者调用。外部模型仍必须遵守原操作语义，不能通过新增接口改变计算含义。

## 阅读后的一个小推演

先不运行程序，考虑把 `lab.clamp` 的结果返回给函数调用者：范围报告会怎样变化，`dead` 又会怎样变化？应当是范围仍为 `[-4,7]`，但 use 不再为空，`dead` 变成 0。反过来，只移除外部范围模型，会失去哪条信息，又保留哪条信息？

能够解释这两个变化，就已抓住本章主题：**操作提供受语义约束的事实，消费者将它与当前 IR 状态结合，再作具体决定。** 下一章[自定义 Type 与 Attribute](./types_attributes)从操作字段继续深入，解释范围本身怎样成为可构造、验证和复用的数据对象。

<details>
<summary>延伸查阅：其他接口与模型注册边界</summary>

常用接口可以按消费者来理解，而不是按名字背诵：

| 消费者需要作的决定 | 相关协议 | 仍需额外理解的内容 |
|---|---|---|
| 从输入和静态信息构造结果类型 | `InferTypeOpInterface` | 推导如何失败，显式结果类型怎样检查兼容性 |
| 穿过结构化区域传播数据流 | `RegionBranchOpInterface` | 哪些区域可能先后执行，边上的值怎样映射 |
| 判断张量操作可否复用存储 | `BufferizableOpInterface` | 读写、别名及分析得到的冲突 |
| 查询某类自定义类型的公共性质 | Type Interface | 查询对象变成 Type，模型返回类型参数相关事实 |
| 查询某类静态数据的公共性质 | Attribute Interface | 查询对象变成 Attribute，与所属 Op 分开 |

本章已实现的是 Op Interface 的直接/外部接入及消费。Type/Attribute Interface 共享协议与模型的思路，但其具体生成、注册代码不在本例中；bufferization 和区域分析的协议随相应机制深入。这样区分是为了知道可迁移的原理，以及还需实际补做的实现。

范围报告仅查询边界，没有实现比较折叠。完整范围分析还需要处理未知状态、多结果、位宽和溢出。某些消费者要求模型必须存在，或用 promised interface 诊断遗漏；本章报告程序采用的 unknown 回退不代表所有接口都允许缺失。

</details>

## 配套观察与依据

观察入口：`aicompiler-labs/llvm-mlir/06-ir-definition/README.md`。执行 `observe.py` 会显示原 IR、两种注册配置的报告及清理后的 IR；不需要先运行才能阅读本章。实现使用 LLVM `llvmorg-20.1.8`，复用上一章 Lab dialect，未修改原实验。

固定版本源码入口：

- [SideEffectInterfaces.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/SideEffectInterfaces.td)：Pure 的组合与效果/推测执行协议。
- [SideEffectInterfaces.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Interfaces/SideEffectInterfaces.cpp)：`isOpTriviallyDead`、效果处理及保守分支。
- [Interfaces.md](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Interfaces.md)：接口模型、外部模型及 Context 注册。
- [Traits.md](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Traits.md)：共享 trait 的方法与验证。

本章的删除与模型对照由 `docs/validate_ir_definition.py` 回归；这里只验证编译器侧行为，未执行自定义操作的机器码。
