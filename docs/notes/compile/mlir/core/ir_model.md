---
order: 1
title: IR 对象模型与表示
updated: 2026-09-06
tags: [mlir, ir, operation, region, block, symbol]
status: draft
---

# IR 对象模型与表示

前置知识：能阅读简单函数和加法即可。本文建立所有方言共同使用的容器模型，解释表示形式、对象归属及 Region 类别；SSA 的合法使用规则在[下一章](./values_ssa)展开。

文中 `%x` 表示一个有类型的值（Value），operand 是操作使用的输入，result 是操作定义的输出。暂时用这三个直观含义阅读结构即可。

版本：LLVM 20.1.8。完整模块、示意树和局部语法分别标注；验证范围见[使用指南](../guides/inspecting_ir)。

## 1. 同一份 IR 的几种表示

MLIR 有可读文本、编译器内存对象和用于保存/传输的 bytecode。它们表达同一套 IR 内容，不是两套需要分别编译的语言。

```text
文本 .mlir ── parser ──→ 内存 IR 对象 ── printer ──→ 文本 .mlir
                              ↑   ↓
                           分析与变换
                              ↕
                      bytecode 读写（序列化）
```

这里的“内存”是运行编译器的进程内存，存放 Operation、Block 等 C++ 对象；与目标程序在 GPU/NPU 上分配的 tensor buffer 是两回事。前端也可以用 IR API 直接创建这些对象，不必先写出文本再解析。

以 `%y = arith.addi %x, %one : i32` 为例：parser 识别操作名和语法，把 `%x`、`%one` 解析为对已定义 Value 的引用，创建加法 Operation 及其结果。Pass 可以修改输入、替换结果的使用或删除节点；printer 再为对象分配可读名字并输出。因此改写通常不是字符串替换，打印后的 `%0`/`%1` 名称变化也不代表算法变化。

文本中的自定义格式和通用格式仍然描述同一批对象。通用格式显式写出输入类型、结果类型、属性/Properties 和 Region；自定义格式可以隐藏可推导信息、隐式 terminator 或入口 Block 标签。下文用同一个函数展示两种形式。

`MLIRContext` 提供方言注册/加载和类型、属性等共享基础设施；模块 Operation 保存这份程序的结构。Context 不是模块、不是执行设备，也不等同于 PassManager。多个模块可以使用同一 Context，但操作的所有权仍由各自 IR 结构管理。

## 2. 为什么把函数和加法都叫 Operation

如果把 Operation 理解为“加、减、乘这样的简单指令”，那么“Operation 拥有函数体”确实显得矛盾。MLIR 对 Operation 的抽象范围更大：它是一个有名称、有语义、可以被编译器分析和变换的 IR 节点。

这个节点可以表达一次计算，也可以表达一项定义，或者表达一个带内部程序的控制结构。

| Operation | 它表达什么 | 是否拥有 Region |
|---|---|---|
| `arith.addi` | 一次整数加法 | 0 个 |
| `func.call` | 对某个函数的一次调用 | 0 个 |
| `func.func` | 函数定义或声明 | 1 个；外部声明的 Region 没有 Block |
| `scf.for` | 一个结构化循环 | 1 个，用于循环体 |
| `scf.if` | 一个结构化条件结构 | 2 个，分别用于 then/else；允许省略 else 的形式中 else Region 为空 |
| `builtin.module` | 模块结构 | 1 个，用于容纳定义等内容 |

这套设计让编译器可以统一处理“计算节点”和“结构节点”。例如，通用遍历可以先访问一个 Operation，再递归进入其内部 Region。各个节点的具体合法性和执行方式仍由对应操作规定。

“都是 Operation”不意味着它们有相同执行语义，也不意味着它们都对应一条机器指令。

## 3. 三种对象的精确定义

### Operation：拥有语义的节点

一个 Operation 可以有输入、结果、静态信息、Location、后继 Block 和嵌套 Region；具体有哪些内容，由它的定义决定。

若一个 Operation 带有内部程序，它通过 Region 来容纳这些程序。例如，`scf.for` 的内部程序是循环体，`func.func` 的内部程序是函数体。

### Region：由所属 Operation 解释的程序区域

Region 由某个 Operation 拥有，里面组织着 Block。Region 本身不是“函数”这个语义对象。

- `func.func` 拥有的 Region 被解释为函数体。
- `scf.for` 拥有的 Region 被解释为循环体。
- `scf.if` 拥有的两个 Region 被解释为两个分支区域。

**Region 表示什么、何时进入、参数怎样绑定、怎样结束，要看它属于哪个 Operation。** 仅仅知道某对象是 Region，还不足以判断它如何执行。

Region 可以为空，也可以包含一个或多个 Block，具体数量受父 Operation 的约束。例如 `scf.for` 的循环体要求恰好一个 Block；函数体可以包含多个 Block。

### Block：带参数的 Operation 序列

一个 Block 包含一列 Operation，也可以有 Block Argument。

在本课涉及的 SSACFG Region 中，Block 对应基本块：同层操作按顺序执行，最后由 terminator 规定是否转移到同层其他 Block，或退出当前区域。

Block 不专属于条件跳转。最普通的直线计算函数也有入口 Block。Block 中还可以出现拥有内部 Region 的 Operation；内部区域的控制流不等于这个 Block 的同层控制流。

在 Graph Region 中，不能直接套用上述按顺序执行的规则。具体 Region 的类别和语义，需要结合其所属 Operation 判断。

## 4. 用没有分支的函数看清包含关系

下面的完整模块只有一个函数：

<!-- mlir-example: ir-model-1 -->
```text
module {
  func.func @add_one(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %y = arith.addi %x, %one : i32
    return %y : i32
  }
}
```

可以把它的内存对象画成：

```text
builtin.module                         Operation
└── 模块 Region
    └── 模块 Block
        └── func.func @add_one          Operation
            └── 函数体 Region
                └── 入口 Block
                    ├── 参数：%x: i32
                    ├── arith.constant Operation
                    ├── arith.addi     Operation
                    └── func.return    Operation
```

这个函数没有条件跳转，依然有 Block。Block 是组织函数体的结构，不是出现 `if` 后才产生的特殊对象。

程序中的两个大括号层次分别是模块 Region 和函数体 Region。Block 的存在被简洁语法隐藏了，所以不能通过数大括号来数 Block。

### 隐藏的入口 Block 在哪里

把函数单独以通用格式表达，会更容易看到它。下面是一个完整模块，和上例表达相同计算；属性中 `function_type` 声明函数签名：

<!-- mlir-example: ir-model-2 -->
```text
"builtin.module"() ({
  "func.func"() <{sym_name = "add_one", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %y = "arith.addi"(%x, %one) : (i32, i32) -> i32
    "func.return"(%y) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

这里的 `^bb0(%x: i32):` 显式声明函数入口 Block。自定义函数语法把 `%x` 放到了函数参数列表中，并省去了 Block 标签。

函数定义这一 Operation 的通用签名是 `() -> ()`：它没有普通 SSA operand 和 result。函数将来调用时的输入输出类型放在 `function_type` 中，不能把这两类签名混在一起。

## 5. 为什么 Operation 能包含另一个 Operation

从对象结构看，它们不是直接任意嵌套，而是通过固定层级组织：

```text
Operation
  → Region
    → Block
      → Operation
        → Region
          → Block
            → ...
```

这不是类型定义上的矛盾，而是一种递归容器结构。上面的 `func.func` 和下面的 `arith.addi` 都是 Operation 的具体实例，处在不同层级、具有不同语义。

例如：函数中有一个循环，循环中有一个加法。

```text
func.func
└── 函数体 Region
    └── 函数入口 Block
        ├── 准备边界等值的 Operation
        ├── scf.for
        │   └── 循环体 Region
        │       └── 循环体 Block
        │           ├── arith.addi
        │           └── scf.yield
        └── func.return
```

函数入口 Block 直接包含的是 `scf.for` 这个节点。加法通过循环体 Region 和 Block 间接位于函数中。

这个结构同时保留“它是循环”和“循环内部计算什么”，后续 Pass 才可以针对整个循环做分析或变换。

## 6. 函数定义和函数调用的区别

**Operation 包含 Region 的函数例子，指的是函数定义 `func.func`，并不是函数调用 `func.call` 把被调用函数嵌入自己。**

看下面的完整模块：

<!-- mlir-example: ir-model-3 -->
```text
module {
  func.func @callee(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %y = arith.addi %x, %one : i32
    return %y : i32
  }

  func.func @caller(%input: i32) -> i32 {
    %output = func.call @callee(%input) : (i32) -> i32
    return %output : i32
  }
}
```

它的结构是两个并列的函数定义：

```text
模块 Block
├── func.func @callee
│   └── Region → Block → 常量、加法、返回
└── func.func @caller
    └── Region → Block
        ├── func.call @callee
        └── func.return
```

`func.call` 保存对 `@callee` 的符号引用，并使用 `%input` 作为参数、产生 `%output`。它自身没有 Region，也没有拥有 `@callee` 的函数体。

运行时发生调用，是执行语义；IR 中谁拥有谁，是静态结构。调用次数增加，不会自动让 IR 中出现更多被调用函数体。编译器若执行内联变换，才会进一步改写调用点并引入对应计算。

## 7. 不要把 C 大括号直接对应为 MLIR Block

C 中 `if (...) { ... }` 的花括号表示源语言语法结构。MLIR 中同一个条件行为可以选择不同层次的表示：

| 表示 | 同层结构 | 分支代码在哪里 |
|---|---|---|
| `scf.if` | 外层 Block 中的一条结构化 Operation | 它拥有的 then/else Region 中 |
| `cf.cond_br` | 当前 Block 的终结操作 | 同一 Region 内的目标 Block 中 |

例如一个 `scf.if` 可以位于函数的唯一入口 Block 中：这个 Block 先运行条件结构，再运行后面的操作。虽然执行时存在条件选择，函数体却不必因此拥有多个同层 Block。

将结构化条件转换为显式 CFG 后，通常就会出现多个同层 Block。计算行为可以相同，IR 的组织方式发生了变化。

## 8. 四种关系分别看

阅读 IR 时，可以分别画四张小图：

| 关系 | 例子 | 要回答的问题 |
|---|---|---|
| 包含关系 | `func.func → Region → Block → arith.addi` | 这个操作属于哪个结构？ |
| 数据依赖 | 常量的结果被加法使用 | 输入从哪里来，结果被谁使用？ |
| 控制流 | `cf.br` 指向另一个 Block | 同一 Region 内下一步进入哪里？ |
| 符号引用 | `func.call @callee` | 通过名字引用了哪项定义？ |

不要因为函数运行时调用了另一个函数，就在包含图中把后者画成前者的孩子；也不要因为一个 Operation 使用了外层 Value，就把 Value 的定义移动到它的 Region 中。

## 9. 结构边界与合法性

理解通用容器结构后，再加入具体约束：

- Region 的入口 Block 参数如何获得值，由父 Operation 规定。函数参数来自调用，循环参数来自边界与初始值等。
- 在 SSACFG Region 中，非入口 Block 的参数通常通过前驱分支传入。
- `func.func` 的函数体带有隔离约束，不能直接捕获其外层 SSA Value。
- `scf.for` 的循环体可以使用合法的外层 Value，例如循环外定义的常量。
- 内层 Region 的 Value 不能直接越过其作用域供外层使用；应通过父 Operation 规定的结果传递机制。
- `func.return` 和 `scf.yield` 的语义不同，不能随意替换。它们各自需要满足所在结构的约束。

后续阅读一个陌生 Operation，先检查“输入、结果、Region 数量、Block 约束、终结操作、传值规则”这六项，通常就能建立它的结构。

## 10. Region 类别与终结约束

SSACFG Region 用顺序与 CFG 描述执行：正常进入非空区域时从入口 Block 开始，操作的结果在执行后供后续操作使用，Block 末尾的 terminator 决定同层跳转或交还控制。入口 Block 不能成为该 Region 内分支的目标；需要循环回边时，另建 header Block。

Graph Region 适合没有这种块内顺序约束的图表示。在 LLVM 20.1.8 中 Graph Region 限于一个 Block，OpResult 在本区域内的使用不要求普通 CFG 的先定义后使用顺序；这不等于随意忽略嵌套区域、隔离或操作自身的约束。

`builtin.module` 的 body 是一个常见的 Graph Region：它组织函数和符号定义，函数在文本中写在前面或后面通常不决定调用先后。`func.func` 的 body 则是 SSACFG Region。两者都通过 Region/Block 容纳 Operation，但不能套用同一执行规则。

Block 通常需要 terminator；例外由所属结构明确声明。`builtin.module` 采用无需 terminator 的约定；`func.func` 使用 `func.return` 或 CF 跳转；`scf.for` 使用 `scf.yield`。自定义语法省略一个可隐式补出的 terminator，与操作根本不要求 terminator 是不同情况。

Region 自身没有结果列表、类型或属性字典。所谓“区域返回值”由 terminator 的 operand 表达，再由父 Operation 解释；真正可供外部 SSA 使用的结果属于父 Operation。这正是 `scf.yield` 与 `scf.for` result 不属于同一对象的原因。

## 11. 从对象模型定位源码

C++ 中 `Operation *` 表示通用节点；`arith::AddIOp`、`func::FuncOp` 等是面向某种操作的类型化包装，提供有语义的访问器。它们不是与 Operation 并存的另一份 IR。类型检查/cast 后，仍在访问同一个底层节点。

| 要确认的关系 | 入口 | 本次要读到的程度 |
|---|---|---|
| Op 属于哪个 Block、拥有哪些 Region | `Operation.h`：`getBlock`、`getRegions` | 对照容器图理解访问方向 |
| Region 拥有哪些 Block、父 Op 是谁 | `Region.h`：`getBlocks`、`getParentOp` | 区分父对象与控制流目标 |
| Block 参数与操作列表 | `Block.h`：`getArguments`、`getOperations` | 参数不在 Operation 列表里 |
| 某种 Op 的结构限制 | 相应 `.td` 中 operands/results/regions/traits | 将通用容器与具体约束对应 |

以后做 IR 变换时，移动/删除节点、克隆 Region 和替换 Value 都需要维护引用与作用域。本章建立这些操作必须尊重的结构；具体 C++ 生命周期和 builder 用法在实现阶段展开。

## 12. 阅读完成标准

能够独立解释下面的结构，再进入 Value/SSA：

1. 一个没有分支的函数为什么依然有 Region 和 Block。
2. 一个循环为什么可以作为 Operation，出现在函数 Block 中，并拥有自己的 Region 和 Block。
3. `func.call` 为什么引用函数定义，却不拥有被调用函数体。
4. 同样有一个 Block，为什么 module 与函数体不能都解释为顺序执行？
5. parser/printer 的输入输出是什么？打印名字改变，为什么不能据此判断数据依赖改变？

<details>
<summary>检查提示</summary>

没有条件分支仍需入口 Block；循环的内部程序由它拥有的 Region 容纳；call 通过符号引用 callee。Region 的种类和父 Op 的契约决定解释方式，文本名字只是在打印时引用 Value 的表示。

</details>

## 源码与验证依据

- [LLVM 20.1.8 LangRef](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/LangRef.md)：Operation、Region、Block 与不同关系的定义。
- [FuncOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Func/IR/FuncOps.td)：函数定义、调用、入口 Block 参数与隔离约束。
- [SCFOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)：for/if 的 Region 和 Block 约束。
- [Operation.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/Operation.h)、[Region.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/Region.h)：需要读实现时，从 Region 访问与 Block 列表开始。

完整模块参与解析、verifier 和通用打印往返检查；结构树和流程图不是可执行输入。Graph/module 的具体约束还应对照 [BuiltinOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/BuiltinOps.td)，Context/Block 的入口为 [MLIRContext.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/MLIRContext.h) 与 [Block.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/Block.h)。

下一章：[Value、SSA 与支配关系](./values_ssa)。
