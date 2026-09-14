---
order: 1
title: 一个操作是怎样被定义出来的：ODS、C++ 与验证
updated: 2026-09-14
---

# 一个操作是怎样被定义出来的：ODS、C++ 与验证

前面写 Pass 时，我们可以直接使用 `arith::AddIOp`，调用 `getLhs()`，再通过 rewriter 替换结果。这些代码之所以成立，是因为 `arith.addi` 已经有了定义：框架知道它叫什么、有哪些输入和结果，以及哪些形式是合法的。

现在从使用者走到定义者这一侧。我们准备增加一个操作，完整追踪它怎样从一份语义约定变成工具能读取、C++ 能构造、verifier 能检查、Pass 能处理的 IR 节点。

## 1. 先决定这个操作表达什么

考虑把整数限制到一个闭区间。以区间 `[-4, 7]` 为例，小于 -4 的输入得到 -4，大于 7 的输入得到 7，中间的值保持原样。

用伪代码表达：

```text
clip(x, lower, upper):
    if x < lower: return lower
    if x > upper: return upper
    return x
```

假定三个数都按有符号 i32 解释，区间下界不大于上界。对于固定区间 `[-4, 7]`，可以手工推演：

| 输入 x | 判定 | 结果 |
|---:|---|---:|
| -9 | 小于下界 | -4 |
| -4 | 正好等于下界 | -4 |
| 2 | 位于区间内 | 2 |
| 7 | 正好等于上界 | 7 |
| 12 | 大于上界 | 7 |

现在给这项计算一个 IR 名称：`lab.clamp`。希望写出的程序是：

<!-- lab-example: op-definition-input -->
```text
module {
  func.func @clip(%x: i32) -> i32 {
    %r = lab.clamp %x bounds(-4, 7) : i32
    return %r : i32
  }
}
```

这里先保留“限制到区间”这个整体含义。后续 Pass 可以选择何时将它展开成比较、选择或 min/max 等较基础的计算。把计算保持为一个操作，能够让编译器在展开之前直接识别它的意图。

这个学习用操作只处理标量整数。它不声称等同于某个框架中具有浮点、广播或张量语义的完整 clamp 算子。

## 2. 区分一次调用传入的值与操作保存的静态信息

看 `%r = lab.clamp %x bounds(-4, 7) : i32`，有三种不同角色：

- `%x` 来自函数参数，本次调用的数值还不知道，应当作为 SSA operand。
- `-4` 和 `7` 是这条操作自身固定的区间，应当保存为静态属性。
- `%r` 是计算后提供给使用者的 SSA result，类型为 i32。

因此，这个操作只有**一个 operand、两个属性、一个 result**。数字出现在操作语法里，不表示它们自动成为两条 SSA 输入边。

选择把上下界放在属性中，也选择了这项抽象的能力范围：同一条操作的上下界在编译时可知。若将来需要每次调用传入不同上下界，就应该设计带相应 SSA operands 的操作。修改 C++ 中某个属性会改变 IR 所描述的程序，却不是目标程序在运行时给属性赋值。

我们还需要三个约束：输入是 i32、结果是 i32、两个属性各自保存 i32 整数；此外，两属性之间必须满足 `lower <= upper`。接下来把这些信息写成机器能够使用的定义。

## 3. 用 ODS 写下一份操作定义

MLIR 的 ODS（Operation Definition Specification）使用 TableGen 描述操作。TableGen 在构建编译器时读取 `.td` 文件并生成 C++，让我们不用为每个操作重复手写输入访问器、通用约束和解析打印代码。

先定义容纳操作的 Dialect：

<!-- source-example: dialect -->
```text
def Lab_Dialect : Dialect {
  let name = "lab";
  let cppNamespace = "::mlir::lab";
  let summary = "Operations for the MLIR learning examples";
}
```

`name = "lab"` 对应 IR 名称的 `lab.` 前缀；`cppNamespace` 对应生成 C++ 类所在的命名空间。Dialect 负责组织和注册一组相关定义，本例先只放一个操作。

接着定义 `lab.clamp`：

<!-- source-example: clamp -->
```text
def Lab_ClampOp : Op<Lab_Dialect, "clamp", [Pure]> {
  let summary = "Clamp a signed i32 value to inclusive constant bounds";
  let description = [{
    Interpret input, lower and upper as signed 32-bit integers.
    Return lower if input < lower, upper if input > upper, otherwise input.
    Require lower <= upper. Equal bounds are valid.
  }];
  let arguments = (ins I32:$input, I32Attr:$lower, I32Attr:$upper);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $input `bounds` `(` $lower `,` $upper `)` attr-dict `:` type($input)
  }];
  let hasVerifier = 1;
}
```

先沿着上一节的约定读三处：`I32:$input` 对应输入，`I32Attr:$lower/$upper` 对应两个属性，`I32:$result` 对应结果。`arguments` 是 ODS 中统一描述 operands 与 attributes 的字段；放在其中并不意味着都变成 SSA operand，具体类别由约束类型决定。

`assemblyFormat` 声明我们希望看到的简洁文本格式，`hasVerifier` 表示还要提供额外的 C++ 验证。`Pure` 声明这项有效的整数计算没有内存效果且可推测执行：它只比较并选择已有整数，不读写存储、不会除零，也没有不终止的内部循环。该声明应当由操作语义支持，不能仅为了获得优化而添加。

配套工程编译并注册后，用 `lab-dialect-opt` 读取第一节程序，实际打印为：

<!-- lab-example: op-definition-custom -->
```text
module {
  func.func @clip(%arg0: i32) -> i32 {
    %0 = lab.clamp %arg0 bounds(-4, 7) : i32
    return %0 : i32
  }
}
```

到这里，工具已经能认识并保存这个操作。操作仍然是 `lab.clamp`，不会因为 description 中写了伪代码就自动执行 clamp 或展开为机器指令。**ODS 把结构、约束和接口接入框架；语义说明还需要由后续变换和执行实现遵守。**

## 4. 从 .td 追到 getInput() 的来源

构建过程有两步：先生成 C++，再编译和链接。

```text
LabOps.td
  ├─ mlir-tblgen -gen-op-decls → LabOps.h.inc
  ├─ mlir-tblgen -gen-op-defs  → LabOps.cpp.inc
  └─ dialect 生成选项          → LabDialect.h.inc / .cpp.inc
                                     ↓
                  与手写的 LabOps.cpp、工具入口一起编译
                                     ↓
                           lab-dialect-opt
```

生成文件在 `artifacts/builds/mlir-op-definition/` 中。`LabOps.h.inc` 声明 `mlir::lab::ClampOp` 以及访问器；`LabOps.cpp.inc` 包含 builder、约束验证、解析和打印等实现。`.inc` 是供手写 C++ 文件包含的生成代码，不是另一套运行时 IR。

对应关系可以直接沿生成文件查到：

| ODS 字段 | 生成接口 | 访问的内容 |
|---|---|---|
| `I32:$input` | `getInput()` | operand 0 引用的 Value |
| `I32Attr:$lower` | `getLowerAttr()` | 下界 IntegerAttr |
| `I32Attr:$upper` | `getUpperAttr()` | 上界 IntegerAttr |
| `I32:$result` | `getResult()` | 这个操作产生的结果 Value |
| `arguments` 与 `results` | `build(...)` | 填充创建操作所需的状态 |
| 类型/属性约束 | `verifyInvariantsImpl()` 等生成实现 | 检查实际对象是否符合声明 |

访问器的名字来自字段名，所以之前的 `arith::AddIOp::getLhs()` 也有类似来源。生成的 `ClampOp` 是对底层 Operation 的类型化包装；它的输入仍是普通 operand 槽位，结果仍是 OpResult。

本版本还为 `I32Attr` 生成返回 `uint32_t` 的 `getLower()` 等便利接口。我们的语义需要有符号比较，因此手写验证使用 `getLowerAttr().getValue()` 取得 APInt，再明确调用 signed 比较。这样负数下界的解释就由代码清楚指定，而不依赖宿主 C++ 整数转换。

修改操作定义应回到 `.td`，然后重新生成和编译。直接修改 `.inc` 会被下一次构建覆盖，也会使定义与实现失去一致性。

## 5. 自动生成的检查还缺少什么

现在把输入的边界改为 `bounds(8, 7)`。两个数字仍然是 i32，但已经不能构成我们定义的有效区间。

ODS 中的两个 `I32Attr` 分别检查属性的存在与类型，并没有写下它们之间的大小关系。要补足这个跨字段条件，定义 `hasVerifier = 1` 声明的成员函数：

<!-- source-example: verify -->
```cpp
LogicalResult ClampOp::verify() {
  if (getLowerAttr().getValue().sgt(getUpperAttr().getValue()))
    return emitOpError("requires lower <= upper (signed i32)");
  return success();
}
```

`sgt` 按有符号整数比较。下界大于上界时，`emitOpError` 产生诊断并返回失败；相等时接受，因为闭区间 `[a, a]` 合法，计算结果恒为 a。

让工具读取下面的错误输入：

<!-- lab-invalid: op-definition-reversed | requires lower <= upper (signed i32) -->
```text
module {
  func.func @bad(%x: i32) -> i32 {
    %r = lab.clamp %x bounds(8, 7) : i32
    return %r : i32
  }
}
```

实际诊断包含：

```text
error: 'lab.clamp' op requires lower <= upper (signed i32)
```

这次拒绝来自手写验证。若输入改成 i64、缺少 lower 或 lower 的属性类型是 i64，则会先被相应结构/生成约束拒绝。正常的整套验证会先建立结构、类型和必要属性等前提，再进入本例的手写验证，因此这里可以使用已经满足 I32Attr 约束的访问器。

沿当前生成代码，还能直接找到 `ClampOp::verifyInvariants()`：它先调用 `verifyInvariantsImpl()`，成功后才调用我们的 `verify()`。这解释了声明与手写条件怎样共同生效。实际验证还有 Trait/Interface 等层次；带 Region 的验证顺序在后续操作和接口章节继续展开。

应通过框架的完整 `verify(...)` 检查任意待验证 IR。直接调用这个手写成员函数，只会运行我们补充的上下界检查，不能替代所有生成约束和结构验证。

## 6. 同一对象为什么能有两种文本写法

现在再看 `assemblyFormat`：

```text
$input `bounds` `(` $lower `,` $upper `)` attr-dict `:` type($input)
```

其中 `$input/$lower/$upper` 引用已声明的字段；反引号中的内容是固定语法。`type($input)` 显式打印输入类型，result 的类型在本例中可以从固定 I32 约束得到，因此没有要求再打印一次结果类型。`attr-dict` 为未在指定位置打印的额外属性保留通道。

这段描述生成 parser/printer，使 `bounds(-4, 7)` 能与两个属性相互转换。它改变可读写法，不改变 clamp 的计算含义。

对第一节程序添加 `--mlir-print-op-generic`，实际得到：

<!-- lab-example: op-definition-generic -->
```text
"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "clip"}> ({
  ^bb0(%arg0: i32):
    %0 = "lab.clamp"(%arg0) <{lower = -4 : i32, upper = 7 : i32}> : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

圆括号里只有 `%arg0`，所以这里确实只有一个 operand。上下界位于 `<{...}>` 的 Properties 文本位置：本版本将本操作的 inherent 属性存放在生成的 Properties 中，`getLowerAttr()` 正是从那里读取。这也给前面[类型与属性](../../core/types_attributes)章节中的分类找到了具体实现。

把这份通用格式重新交给同一个工具，仍能打印出第三节的简洁格式。它们经过 parser 构造相同结构，再由不同 printer 路径呈现。通用格式省去了对特定简洁语法的依赖，但注册后的操作仍要满足自身 verifier，不能用通用写法绕过类型和上下界条件。

## 7. 通过生成的 builder 直接创建操作

现在我们已经知道输入、属性和结果在对象中怎样存储，也可以不经过文本 parser，直接使用 C++ 创建它。

配套 `lab-build` 先创建一个参数和结果均为 i32 的 `@clip` 函数，再在入口 Block 中调用：

<!-- source-example: builder -->
```cpp
auto clamp = builder.create<lab::ClampOp>(
    loc, builder.getI32Type(), x,
    builder.getI32IntegerAttr(invalid ? 8 : -4), builder.getI32IntegerAttr(7));
builder.create<func::ReturnOp>(loc, clamp.getResult());
```

这里 `invalid` 是工具用于观察错误构造的开关，正常运行时为 false。传入的参数依次提供 location、结果类型、输入 Value、下界属性和上界属性。生成的 `ClampOp::build` 将它们加入 `OperationState`，随后 `OpBuilder::create` 创建并插入实际对象。`clamp.getResult()` 才是 return 可以使用的 SSA 值。

正常运行打印的访问结果包含：

```text
builder completed; input_type=i32 lower=-4 upper=7
```

随后显式 `verify(module->getOperation())` 成功，构造的函数与第一节输入表达相同结构和计算。

将 `invalid` 设为 true，则先出现：

```text
builder completed; input_type=i32 lower=8 upper=7
error: 'lab.clamp' op requires lower <= upper (signed i32)
```

这段顺序说明当前生成的 builder 已经完成对象构造，随后完整 verifier 才拒绝错误的上下界。Builder 负责组织构造参数，它不自动证明所有操作契约，也不执行目标程序。编译器在构造或改写后，需要在合适的边界验证形成的 IR。

parser 与 builder 于是构成两条入口：

```text
文本格式 → parser ────┐
                     ├→ Operation → 完整 verifier → 供 Pass 使用
C++ 参数 → builder ───┘
```

两条入口汇入同一个对象模型，所以后续 Pattern 不需要关心它最初来自文件还是来自 C++ 构造。

## 8. 把生成的定义装进工具

C++ 类已经生成，还需要让 Context 在加载 Lab dialect 时认识这些操作。本例的 Dialect 初始化代码为：

<!-- source-example: initialize -->
```cpp
void LabDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LabOps.cpp.inc"
      >();
}
```

`GET_OP_LIST` 让生成文件在这个位置展开为操作类型列表；`addOperations` 将这些操作注册到 Dialect。另一处 `GET_OP_CLASSES` 则用于引入类定义/实现。这些宏是在选择生成文件提供的哪一部分内容。

工具再提供 Dialect 注册：

<!-- source-example: main -->
```cpp
int main(int argc, char **argv) {
  registerLabExpansion();
  mlir::DialectRegistry registry;
  registry.insert<mlir::lab::LabDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Operation definition examples\n", registry));
}
```

因此整条连接是：`.td` 生成类与实现，C++ 编译链接进工具，工具登记可加载的 Dialect，Dialect 初始化登记操作，parser 和后续基础设施便能找到该操作的定义。

这与上一章的 Pass 注册是两条配合使用的链路：Dialect/Op 注册使工具认识 IR，Pass 注册使工具认识要执行的变换名称。把新源码放到目录中而没有编译、链接和注册，都不足以让一个已有 `mlir-opt` 自动理解它。

本章使用独立工具 `lab-dialect-opt`，既保留标准方言，也加入 Lab dialect。更大工程还会注册更多类型、属性和接口，本章先把一项操作的连接完整走通。

## 9. 让已经学过的 Pass 处理新操作

前面的定义让工具能够理解 clamp 的结构与合法性。现在希望将其表示展开为标准算术操作，语义可以写成：

```text
bounded = max_signed(x, lower)
result  = min_signed(bounded, upper)
```

为什么这与第一节一致？当 x 小于 lower，第一步得到 lower，因 lower <= upper，第二步仍得 lower；当 x 位于区间内，两步都保留 x；当 x 大于 upper，第一步保留 x，第二步返回 upper。负数也必须遵循同样的有符号比较，所以使用 `maxsi/minsi`。

将这个推导写成 Pattern：

<!-- source-example: expand -->
```cpp
struct ExpandClamp : OpRewritePattern<lab::ClampOp> {
  explicit ExpandClamp(MLIRContext *context) : OpRewritePattern(context, 1) {}
  LogicalResult matchAndRewrite(lab::ClampOp op,
                               PatternRewriter &rewriter) const override {
    auto lower = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getLowerAttr());
    auto upper = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getUpperAttr());
    auto bounded = rewriter.create<arith::MaxSIOp>(op.getLoc(), op.getInput(), lower);
    rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, bounded, upper);
    return success();
  }
};
```

这里已经能看见前面几章的直接用途：通过生成的访问器取得输入和属性，使用 rewriter 创建标准操作，用替代操作更新所有使用并删除旧 root，再由函数 Pass 把规则交给 driver。

还有一个细节：上下界原来是属性，`arith.maxsi/minsi` 需要的却是 SSA operands。因此展开规则先创建两条 constant，将静态数据变成这些算术操作可以使用的 Value。这正是第二节的表示选择在转换时产生的实际工作。

运行 `builtin.module(func.func(lab-expand-clamp))` 后，第一节输入变成：

<!-- lab-example: op-definition-expanded -->
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

配套 Pass 使用一次 Walk 应用本规则，新建的都是 arith 操作，不需要重新匹配 clamp。它还通过 `getDependentDialects` 声明会创建 arith 操作。其余 Pass/注册代码与上一章相同，可在 `ExpandClamp.cpp` 中对照阅读。

这个例子展示了“将一种表示展开为另一种表示”的工作。它仍使用普通 Pattern；后续 Dialect Conversion 才引入哪些操作必须消失、哪些形式算合法，以及类型变化如何跨边界传播的契约。

现在可以将两部分学习放在一起看：**Operation 定义表达什么，以及什么形式有效；Pass 决定在某个编译阶段怎样处理和变换这些操作。** 分析结果、硬件规则和性能目标以后会成为变换的更多依据，而对象修改、规则调度和测试仍然使用前面建立的基础。

## 10. 阅读后的推演与下一步

先改变同一个例子的上下界：`(-4, -4)`、`(7, -4)`、`(-2147483648, 2147483647)`。判断各自能否通过 verifier，并说明接受时计算表达什么。再把输入类型改为 i64，区分这次拒绝与下界颠倒的拒绝分别来自哪里。

随后沿生成文件找一次 `getInput()` 和 `ClampOp::build`，说明访问器怎样连接字段，以及 builder 为什么能先构造非法区间。理解的是对象关系与验证过程，方法名可以查阅。

<details>
<summary>核对推演</summary>

`(-4, -4)` 合法，所有输入都得到 -4；`(7, -4)` 被手写有符号上下界检查拒绝；完整 i32 有符号范围合法，保持输入不变。输入改为 i64 则违反 ODS 中的 I32 约束，不需要等到上下界比较才拒绝。完整范围测试验证了 IR 的边界表示与接受情况，没有实际执行所有 i32 输入。

</details>

下一章沿这个操作继续学习 [Trait / Interface](./traits_interfaces)：通用优化怎样通过效果接口判断删除条件，范围消费者怎样查询不同操作。再读[自定义 Type / Attribute](./types_attributes)与[带 Region 的操作](./regions_assembly)，把领域数据和区域协议补齐，再读[解析与打印](./assembly_format)，随后进入 Dialect Conversion。整个章节组的关系见[模块总览](./)。

## 实现与验证入口

配套工程为 `aicompiler-labs/llvm-mlir/05-op-definition/`。在 workspace 根目录运行：

```bash
python3 aicompiler-labs/llvm-mlir/05-op-definition/observe.py --test
```

观察脚本复用 LLVM/MLIR 20.1.8 构建，依次展示自定义/通用格式、builder 构造、Pass 展开和验证失败。完整构建命令与可编辑输入见工程 README。原理阅读可以直接使用本文已经展示的结果。

lit 测试检查四类行为：打印往返与合法边界；输入/结果/属性类型、字段缺失、operand 数量及区间错误；builder 的有效与无效构造；展开后的操作和多处使用关系。`--verify-diagnostics` 检查错误输入是否产生预期诊断，预期诊断全部匹配时测试成功。

本文自定义方言示例由 `docs/validate_op_definition.py` 使用注册 Lab dialect 的工具核对，并检查正文片段与实际编译源码、生成代码入口和展示输出。例子的数值表来自语义推演，未执行生成函数的机器码，也没有进行性能测试。

固定版本的查证路径：

- [Operations / ODS](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)：字段、builder、assembly format 与 verifier 顺序。
- [OpBase.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/OpBase.td)：操作声明基础；[CommonAttrConstraints.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/CommonAttrConstraints.td)：整数属性约束与生成访问器类型。
- [Standalone 示例](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/examples/standalone)：生成文件、Dialect 初始化和工具集成。
- [SideEffectInterfaces.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/SideEffectInterfaces.td)：本章 Pure 声明的组成，下一章继续解释通用消费者。
