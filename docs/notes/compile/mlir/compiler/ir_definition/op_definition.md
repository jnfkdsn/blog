---
order: 1
title: 操作定义：语义、字段与验证
updated: 2026-10-02
---

# 操作定义：语义、字段与验证

[my.add 导读](./dialect_basics)已经说明，一份操作定义会生成 C++ 能力，并通过注册交给框架使用。本章继续向前：当一种计算包含输入、静态参数和额外约束时，应该怎样设计它的表示，怎样保证不同入口构造的对象都遵守约定？

我们用整数区间限制作为例子。它比加法多了两个静态参数，也产生了一个需要检查的跨字段关系；这些需求会自然引出 Attribute、verifier 和 builder。生成文件的包含与工具注册沿用导读中的接入过程，本章集中解释操作本身。

## 1. 操作语义与抽象边界

设计算 `clamp(x, lower, upper)` 将有符号 i32 整数 x 限制在闭区间内：

```text
x < lower  → 返回 lower
x > upper  → 返回 upper
其余情况   → 返回 x
```

以 `[-4, 7]` 为例，输入 -9、2、12 分别得到 -4、2、7。这里的上下界必须有序；`[7, 7]` 也是合法区间，所有输入都得到 7。

给它一个操作名 `my.clamp`，程序可以写成：

<!-- opdef-example: op-definition-input -->
```text
module {
  func.func @clip(%x: i32) -> i32 {
    %r = my.clamp %x bounds(-4, 7) : i32
    return %r : i32
  }
}
```

在当前层次保留这条操作，编译器能够直接识别“将输入限制到某个区间”这一意图。之后再选择合适的时机，把它展开为基础算术操作。自定义操作因此同时确定了**当前保留的抽象**和**后续实现需要保持的含义**。

本例只讨论标量整数与静态上下界。实际框架的 clamp 还可能涉及浮点、广播和张量，这些语义应另外定义，不能仅因操作名字相似就沿用本章结论。

## 2. Operand、Attribute 与 Result

观察这一行：

```text
%r = my.clamp %x bounds(-4, 7) : i32
```

`%x` 表示计算时由其他地方提供的输入值；`-4` 和 `7` 是这条操作保存的固定参数；`%r` 是交给后续计算的结果。对应的结构为：

```text
函数参数 %x ──→ operand 0
               my.clamp   ──→ result 0 ──→ return
               lower = -4
               upper =  7
```

两个整数属性没有各自形成一条 SSA 输入边。执行 `@clip` 时，x 可以随调用改变，而同一条操作中的上下界保持为 -4 和 7。编译器当然可以修改属性，但那是在修改所描述的程序。

选择 Attribute 也限制了这种操作能够表达的计算。若上下界来自运行时的其他计算，就应改用相应的 operands。反过来，operand 也可能由常量操作产生；区分两者的依据是表示与传值方式，不能简单记成“operand 一定未知、attribute 一定是数字”。

这个设计给出一份可检查的约定：一个 i32 输入、两个 i32 整数属性、一个 i32 结果，以及 `lower <= upper`。接下来把约定交给框架。

## 3. ODS 字段声明与生成接口

下面是实际操作定义。所属 `My_Dialect` 使用名称 `my` 和 C++ 命名空间 `mlir::my`，其生成、编译和注册方式与 my.add 一致。

<!-- source-example: clamp -->
```text
def My_ClampOp : Op<My_Dialect, "clamp", [Pure]> {
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

先看 `arguments` 与 `results`：`I32:$input` 声明一个 operand；`I32Attr:$lower` 和 `I32Attr:$upper` 声明属性；`I32:$result` 声明结果。ODS 的 `arguments` 同时容纳输入与静态属性，具体类别由约束决定。

从这些声明生成的接口直接对应刚才的图：

| 访问方法 | 取得什么 | 当前例子 |
|---|---|---|
| `getInput()` | 第一个 operand 引用的 Value | 函数参数 `%x` |
| `getLowerAttr()` | 下界 IntegerAttr | 保存 -4 的属性 |
| `getUpperAttr()` | 上界 IntegerAttr | 保存 7 的属性 |
| `getResult()` | 操作定义的结果 Value | 被 return 使用的值 |

这些方法访问 IR，不计算某次函数调用的结果。后续的 verifier 与 Pattern 都通过它们取得需要的信息。

`assemblyFormat` 规定属性怎样显示在 `bounds(...)` 中，`hasVerifier` 请求额外验证方法的声明。`Pure` 则为这种标量比较和选择声明无内存效果及可推测执行的性质；其依据是这里的计算语义。它怎样被通用优化使用，在下一章展开。

输入、属性和结果的类别检查可以由声明生成，但“下界不得大于上界”还没有出现在任何类型约束中。这正是手写验证需要补上的一层。

## 4. 生成约束与自定义验证

### 4.1 单个字段与字段间关系

把上下界改成 8 和 7，每个属性单独看仍是合法的 i32 整数，组合起来却不再是有效区间。因此验证应分成两类：

```text
字段约束：lower 和 upper 都是 i32 整数属性
    ↓
关系约束：将二者按有符号整数解释，检查 lower <= upper
```

第二类条件由 `hasVerifier = 1` 对应的实现补充：

<!-- source-example: verify -->
```cpp
LogicalResult ClampOp::verify() {
  if (getLowerAttr().getValue().sgt(getUpperAttr().getValue()))
    return emitOpError("requires lower <= upper (signed i32)");
  return success();
}
```

`getValue()` 取得属性里的整数位模式，`sgt` 明确按有符号整数比较。这样 -4 小于 7 的解释来自代码，不依赖生成的便利访问器在宿主 C++ 中采用哪种整数类型。本版本的 `getLower()` 返回 `uint32_t`，直接拿它比较负数上下界容易偏离操作语义。

### 4.2 完整验证与失败位置

<!-- opdef-invalid: op-definition-reversed | requires lower <= upper (signed i32) -->
```text
module {
  func.func @bad(%x: i32) -> i32 {
    %r = my.clamp %x bounds(8, 7) : i32
    return %r : i32
  }
}
```

工具会拒绝这段程序，诊断中包含：

```text
'my.clamp' op requires lower <= upper (signed i32)
```

若缺少 lower，或其属性类型不是 i32，则会先被字段检查拒绝。对于本例，生成的 `verifyInvariants()` 先检查生成约束，再调用手写的 `verify()`；进入上面的比较时，所需字段已经满足相应前提。

这也说明，手写成员函数只负责补充条件。检查任意待验证 IR 时，应调用框架的完整验证入口，而不能仅调用 `ClampOp::verify()` 就认定整个对象合法。

验证检查的是表示是否满足已实现的规则。它不会枚举所有 x，更不会证明尚未编写的目标代码一定正确计算 clamp。动态行为仍需要由定义和后续实现共同保证。

## 5. 对象构造与验证边界

文本 parser 是构造操作的一条入口。另一条入口来自编译器内部：前端或 Pass 可以直接使用 C++ 参数创建同样的对象。

假设已经创建参数和返回类型均为 i32 的 `@clip`，变量 x 是其入口参数，builder 的插入点在函数体中。实际构造代码为：

<!-- source-example: builder -->
```cpp
auto clamp = builder.create<my::ClampOp>(
    loc, builder.getI32Type(), x,
    builder.getI32IntegerAttr(invalid ? 8 : -4), builder.getI32IntegerAttr(7));
builder.create<func::ReturnOp>(loc, clamp.getResult());
```

`invalid` 是演示工具的开关，正常情况下为 false。调用依次提供结果类型、输入 Value 和两个属性；生成 builder 填写操作状态，`OpBuilder::create` 创建并插入对象。随后 return 使用新对象的结果。

正常构造后，通过访问器读到：

```text
builder completed; input_type=i32 lower=-4 upper=7
```

若把开关设为 true，builder 仍能先构造出下界 8、上界 7 的对象；随后完整验证才报告前面的区间错误。当前生成 builder 没有自动执行这项跨字段检查。

所以两条入口的关系是：

```text
文本 → parser ─────┐
                   ├→ 具有输入、属性和结果的 Operation → 验证 → 供变换处理
C++ 参数 → builder ┘
```

验证后的 Pattern 无需区分对象最初来自哪条入口。这是框架采用统一 IR 对象模型的直接收益。

## 6. 操作展开与语义保持

定义好对象后，已有的改写框架就能处理它。我们选择以下展开：

```text
bounded = max_signed(x, lower)
result  = min_signed(bounded, upper)
```

当 x 小于 lower，第一步将它提高到 lower，第二步因为 lower 不大于 upper 而保留它；当 x 位于区间内，两步都保留 x；当 x 大于 upper，第二步把结果限制为 upper。刚才验证的区间顺序正是这个推导的前提。

下面的 Pattern 实现该过程：

<!-- source-example: expand -->
```cpp
struct ExpandClamp : OpRewritePattern<my::ClampOp> {
  explicit ExpandClamp(MLIRContext *context) : OpRewritePattern(context, 1) {}
  LogicalResult matchAndRewrite(my::ClampOp op,
                               PatternRewriter &rewriter) const override {
    auto lower = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getLowerAttr());
    auto upper = rewriter.create<arith::ConstantOp>(op.getLoc(), op.getUpperAttr());
    auto bounded = rewriter.create<arith::MaxSIOp>(op.getLoc(), op.getInput(), lower);
    rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, bounded, upper);
    return success();
  }
};
```

上下界原先是属性，`arith.maxsi/minsi` 却需要 SSA 输入，所以先用两个常量操作把属性中的整数变成 Value。随后读取原输入，创建 max，最后用 min 的结果替换旧 clamp 的结果。这里的表示变化可以逐步追踪：

```text
静态属性 -4、7 → 常量操作的结果
原 operand x  → maxsi 的输入
maxsi 的结果  → minsi 的输入
旧 clamp 的使用者 → 改为使用 minsi 的结果
```

在已构建工具的目录中，将第一节程序保存为 `input.mlir`，下面的命令运行对应 Pass：

```bash
./my-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(my-expand-clamp))' --verify-each
```

实际输出为：

<!-- opdef-example: op-definition-expanded -->
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

至此，语义约定、字段设计、验证和变换已经相连。这个例子使用普通 Pattern 完成固定 i32 的局部展开，没有处理自定义类型转换；后续 Dialect Conversion 还需要解决目标合法性和跨边界类型衔接。

## 7. 文本表示与检查入口

同一条 clamp 可以按简洁格式或通用格式打印。正常解析后输出如下：

<!-- opdef-example: op-definition-custom -->
```text
module {
  func.func @clip(%arg0: i32) -> i32 {
    %0 = my.clamp %arg0 bounds(-4, 7) : i32
    return %0 : i32
  }
}
```

使用 `--mlir-print-op-generic` 时输出为：

<!-- opdef-example: op-definition-generic -->
```text
"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "clip"}> ({
  ^bb0(%arg0: i32):
    %0 = "my.clamp"(%arg0) <{lower = -4 : i32, upper = 7 : i32}> : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

重点是 `"my.clamp"(%arg0)` 只有一个 operand；上下界在本版本的 `<{...}>` 属性/Properties 表示中。使用生成的 `getLowerAttr()` 访问它们，可以避免消费者猜测属性存放在哪个内部容器中。

两种语法汇入相同对象，接受同一组验证规则。通用写法不会绕过上下界检查。文本与字段之间的完整过程在[解析与打印](./assembly_format)展开，本章不再重复生成 parser 的全部实现。

## 理解检查

将上下界改成 `[7, 7]`，解释 verifier 为什么接受，以及 max/min 两步为何对任何合法 x 都得到 7。再考虑把上界改成运行时输入：除了文本写法，字段类别、验证条件和展开 Pattern 分别需要发生什么变化？

下一章[通用语义：Trait 与 Interface](./traits_interfaces)讨论另一个问题：专用 Pattern 已经认识 clamp，但不认识其名字的通用优化怎样使用它的语义事实？

## 实现与依据

本章按 LLVM `llvmorg-20.1.8` 核对。关键代码来自[操作定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/05-op-definition)，工程保存构建、完整外壳与正反例；本文展示的展开输出经过工具验证，未执行目标机器码。

固定版本的 [ODS 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)用于确认字段、builder 与验证顺序；[my.add 导读](./dialect_basics)解释定义如何生成并注册到工具，无需在这里再走一遍工程链。
