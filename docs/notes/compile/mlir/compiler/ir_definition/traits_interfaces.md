---
order: 2
title: 通用语义：Trait 与 Interface
updated: 2026-10-02
---

# 通用语义：Trait 与 Interface

上一章的展开 Pattern 直接匹配 `my.clamp`，知道它的上下界存在哪两个属性里。专用变换这样写很自然，但编译器还有大量需要处理不同方言的通用代码：判断操作能否删除、查询结果范围、分析区域之间的数据传递。

如果每个消费者都维护所有操作的名字与字段，增加一个新操作就可能需要修改许多地方。本章从“查询 clamp 的结果范围”出发，说明怎样把语义事实交给通用代码；随后再用死代码删除观察现成协议如何影响优化。

## 1. 通用消费者与语义事实

对于 `clamp(x, -4, 7)`，即使不知道 x，也知道任何合法执行的结果都落在 `[-4, 7]`。假设后续代码判断结果是否小于 100，这个范围足以支持“比较恒为真”的推导。

消费端真正需要的是结果的上下界，并不一定需要知道操作叫 clamp，也不需要知道上下界保存为两个属性还是一个复合属性。因此可以先约定一个问题：

> 对具有单一整数含义结果的操作，查询一个可靠的静态有符号闭区间。所有有定义执行的结果都在其中，区间不必是最紧的估计。

这是一项语义契约。某个操作返回 `[-4, 7]`，就承诺其输出不会超出该范围；没有这种信息的操作可以被当前消费者视为未知。

整个协作过程是：

```text
消费者需要结果范围
  → 询问操作是否支持这项查询
  → 调用统一的上下界方法
  → 操作自己的实现从字段中取得答案
  → 消费者使用答案作决定
```

MLIR 的 Interface 用来表达这类公共协议。我们先实现一个只返回两个整数的接口，把一次查询完整走通。

## 2. Op Interface 的定义与实现

### 2.1 查询协议

用 ODS 定义 `StaticBounds`：

<!-- source-example: bounds-interface -->
```text
def StaticBounds : OpInterface<"StaticBounds"> {
 let cppNamespace = "::mlir::my";
 let methods = [
  InterfaceMethod<"Inclusive signed lower bound of the result.", "int64_t", "getMinimum">,
  InterfaceMethod<"Inclusive signed upper bound of the result.", "int64_t", "getMaximum">
 ];
}
```

这段定义规定两个方法的名字、返回类型和用途，生成接口类及接入结构。它没有提供上下界数值；不同操作需要根据自身语义实现这些方法。

这里使用有符号 `int64_t` 容纳本例 i32 的上下界。若要覆盖任意位宽、多结果或运行时范围，协议本身还需要重新设计，不能只换几个操作名就认为通用范围分析已经完成。

### 2.2 操作提供答案

继续使用上一章的 `my.clamp`。它的计算、输入和结果保持原样；现在为操作增加范围查询接口，让通用消费者也能取得它的上下界。

<!-- source-example: direct-op -->
```text
def ClampOp : Op<My_Dialect, "clamp", [Pure, DeclareOpInterfaceMethods<StaticBounds>]> {
 let arguments = (ins I32:$input, I32Attr:$lower, I32Attr:$upper);
 let results = (outs I32:$result);
 let assemblyFormat = "$input `bounds` `(` $lower `,` $upper `)` attr-dict `:` type($result)";
 let hasVerifier = 1;
}
```

新增的 `DeclareOpInterfaceMethods<StaticBounds>` 将协议接到操作上，并声明待实现的方法。原来的字段、结果和上下界验证仍然保留：

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

两个查询方法读取上下界属性。能直接把属性值作为结果范围，依据是 clamp 的计算语义：小于下界时返回下界，大于上界时返回上界，其他情况返回区间内的输入。对于别的操作，即使也有名为 lower 和 upper 的属性，也必须重新确认它们是否意味着输出范围。

到这里已有协议和实现，还需要一个真正使用它们的消费者。

## 3. 接口查询与调用过程

先用一个完整程序确定被查询的对象：

<!-- irdef-example: direct-input -->
```text
module {
  func.func @direct(%x: i32) -> i32 {
    %r = my.clamp %x bounds(-4, 7) : i32
    return %r : i32
  }
}
```

消费端在遍历到一个 `Operation *op` 后，执行：

<!-- source-example: consumer -->
```cpp
if (auto bounds = dyn_cast<my::StaticBounds>(op))
  llvm::errs() << " bounds=[" << bounds.getMinimum() << "," << bounds.getMaximum() << "]";
else
  llvm::errs() << " bounds=unknown";
```

`dyn_cast` 尝试取得这个操作的接口视图；成功后通过统一方法取值，失败则输出 unknown。注意消费端没有把 op 转成 `my::ClampOp`，也没有自行搜索 lower 属性。

对于本例，工具实际报告中包含 `bounds=[-4,7]`。把调用过程放回具体对象：

| 步骤 | 当前发生的动作 |
|---|---|
| 取得接口 | `my.clamp` 已声明支持 StaticBounds |
| 查询下界 | 接口分派到 `ClampOp::getMinimum()` |
| 读取字段 | lower 属性保存 -4，方法返回 -4 |
| 查询上界 | `getMaximum()` 从 upper 属性取得 7 |
| 消费结果 | 报告程序打印 `[-4,7]` |

当前工程实现的是查询和报告。前面“与 100 比较恒为真”是说明消费者用途的语义推演，尚未在这个报告 Pass 中实现比较折叠。

如果后来有另一种操作通过不同字段计算出可靠范围，它可以实现相同接口，消费代码无需随字段布局改变。这正是定义协议的意义：操作负责提供事实，消费者负责把事实用于自己的算法。

## 4. Trait 与共享操作约束

Interface 面向“可以向不同操作提出什么问题”。另一类需求是让许多操作共同遵守同一种结构约束，例如某个区域必须只有一个 Block。重复编写这一检查并没有帮助，MLIR 因此也提供 Trait 来复用共性、方法和验证。

例如 `SingleBlock` 可以附着到不同的区域操作上，使它们共享单 Block 的结构要求。`IsolatedFromAbove` 表达禁止某类外部 SSA 捕获的约束；这些约束的具体作用会在 Region 章通过输入与区域参数说明。

两种机制在设计上的关注点可以这样区分：

| 需求 | 使用方式 |
|---|---|
| 多个操作共享某种约束或实现 | 附加合适的 Trait |
| 通用消费者需要对象相关的答案 | 通过 Interface 调用操作提供的方法 |

Trait 可以包含方法和验证逻辑，不能仅理解成一个标签；Interface 也可能提供默认实现。在 ODS 中，接口接入本身会以 trait 的形式出现，因此它们不是互斥的两个列表。先从需求判断它们各自承担什么，比先研究模板继承关系更有帮助。

上一章定义里出现的 `Pure` 就组合了现有能力。下面用它观察通用消费者真正作出一次删除决定。

## 5. 内存效果与死代码删除

### 5.1 未使用结果的删除条件

下面两种操作约定相同的 clamp 计算，但 `my.opaque_clamp` 没有声明内存效果与推测执行信息：

<!-- irdef-example: interfaces-input -->
```text
module {
  func.func @unused(%x: i32) {
    %a = my.clamp %x bounds(-4, 7) : i32
    %b = "my.opaque_clamp"(%x) <{lower = -4 : i32, upper = 7 : i32}> : (i32) -> i32
    return
  }
}
```

两项结果都没有使用者。人知道它们只是比较并选择整数，因此可以删除；通用优化只看到两个不被使用的结果，却还不知道操作是否写内存或输出日志。

这里缺少的事实是执行本身是否有需要保留的效果。`my.clamp` 的 `Pure` 包含 `NoMemoryEffect`，通过内存效果接口报告空效果集合。对这个例子，`isOpTriviallyDead` 的决定可以按以下过程理解：

```text
结果无人使用
  → 排除 terminator 等不能按普通死操作删除的结构
  → 查询效果，确认没有需要保留的行为
  → 可以删除 my.clamp
```

`my.opaque_clamp` 没有提供相应效果模型，工具采取保守处理，保留它。实际报告中的 dead 字段为：

```text
my.clamp            dead=1
my.opaque_clamp  dead=0
```

运行 canonicalize 后，前者被清理，后者仍在。这里的 dead=0 表示未能证明它可被平凡删除，不表示证明了它一定有副作用。

### 5.2 无效果与推测执行

`Pure` 在当前版本组合 `NoMemoryEffect` 与 `AlwaysSpeculatable`。后者涉及另一项决定：把原本可能不执行的计算提前执行，是否会引入不允许的行为。

例如某个除法原来位于不会进入的分支中，提前计算可能遇到非法除数。它与“结果无人使用时是否删除”不是同一个问题。本例的平凡 DCE 判断并不要求调用 `isPure` 或额外执行 speculation 检查；完整算法还处理某些读、分配和递归效果情形，不能归纳成“只有 Pure 才能删除”。

这些声明必须由真实语义支持。若把写设备状态的操作错误声明为无效果，结构 verifier 未必发现错误，但优化可能据此删除必要行为。接口使消费者能够使用事实，事实是否可靠仍是操作作者的责任。

## 6. 外部接口模型

现在已有两种独立信息：效果信息影响删除，范围信息回答数值范围。`my.clamp` 已经直接实现了范围接口；第 5 节的 `my.opaque_clamp` 则还没有。假设一个工具希望查询后者的范围，却不修改它的 ODS 声明，可以将范围实现放在工具侧，用外部模型接入。

这里继续使用同一个 `my` 方言。外部模型中的“外部”指实现放在操作定义之外，不要求属于另一个方言。实际项目可以用这种方式将某个消费者需要的接口接入已有操作，减少组件间的依赖。

<!-- source-example: external-model -->
```cpp
struct OpaqueClampBoundsModel : my::StaticBounds::ExternalModel<OpaqueClampBoundsModel, my::OpaqueClampOp> {
 int64_t getMinimum(Operation *op) const { return cast<my::OpaqueClampOp>(op).getLowerAttr().getInt(); }
 int64_t getMaximum(Operation *op) const { return cast<my::OpaqueClampOp>(op).getUpperAttr().getInt(); }
};
```

模型仍读取同一份属性，只是方法不再写在操作类内部，所以显式接收待查询的 `Operation *`。再通过 registry extension，在 My dialect 加载时附加它：

<!-- source-example: attach -->
```cpp
registry.addExtension(+[](MLIRContext *context, my::MyDialect *) {
  my::OpaqueClampOp::attachInterface<OpaqueClampBoundsModel>(*context);
});
```

这个动作把模型接到相应 Context 中，单纯编译一个模型类还不够。对于同一份 `my.opaque_clamp` 输入，两个工具配置得到：

| 配置 | 范围报告 | 本例 DCE 判断 |
|---|---|---|
| 注册外部范围模型 | `bounds=[-4,7]` | `dead=0` |
| 未注册外部范围模型 | `bounds=unknown` | `dead=0` |

有了范围模型，操作仍然没有提供内存效果信息，所以 DCE 继续保留它。与此同时，直接实现接口的 `my.clamp` 在两个配置下都能回答范围查询。这个对照也说明：接口不是“添加后所有优化自动变聪明”的开关；必须有消费者真正查询并使用它。

直接实现与外部模型供同一个协议消费。实际开发中先掌握直接实现即可，遇到组件依赖或无法修改原操作时，再采用外部接入。

## 理解检查

把第 5 节 `my.clamp` 的结果作为函数返回值，范围仍是 `[-4,7]`，但它不再是未使用结果。解释范围查询为何仍成立、删除判断为何改变。再设想一个错误模型返回 `[0,7]`：输入 -9 时，哪项语义事实会被违反？

下一章[自定义 Attribute 与 Type](./types_attributes)把注意力转向事实如何保存在 IR 中：区间可以是一组计算参数，也可以成为附着在结果上的类型保证，两者服务不同用途。

## 实现与深入范围

代码与查询、删除对照基于 LLVM `llvmorg-20.1.8`，完整实现见[IR 定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/06-ir-definition)。本例实现 Op Interface；Type/Attribute Interface、类型推导、区域控制流和 bufferization 协议在对应主题中继续展开。当前的 unknown 回退是报告程序的策略，不意味着所有消费者都允许所需接口缺失。

固定版本的 [Interfaces 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Interfaces.md)和 [Traits 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Traits/_index.md)用于确认接入方式；[SideEffectInterfaces.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Interfaces/SideEffectInterfaces.cpp)用于核对平凡死代码判断的具体分支。
