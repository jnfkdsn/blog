---
order: 1
title: 01：自定义操作与 Dialect 接入
updated: 2026-10-01
---

# 01：自定义操作与 Dialect 接入

写 Pass 时，我们已经会找到一条 `arith.addi`，取得它的输入，再替换它的结果。那时，加法操作是现成的：框架知道它的名字、输入输出和合法形式，变换只需要使用这些信息。

现在换一个位置：假如要给编译器增加一种操作，这些信息从哪里来？我们需要写一个怎样的定义，才能让工具读取它，让 C++ 代码处理它？

这一章用 `my.add` 回答这个问题。它仍然表示两个整数相加，刻意沿用熟悉的计算，好把注意力放在“编译器怎样认识一种新操作”上。实际项目已有 `arith.addi` 时通常无需再造一个加法；自定义操作的价值在于表达项目所需的抽象，例如暂时保留一个高层计算，等合适的阶段再展开。

我们先看工具需要认识什么，再解释定义怎样变成工具的能力，最后沿一次实际的读取与检查把它们接起来。

## 1. 操作定义与 IR 实例

希望工具能够处理下面这段程序：

<!-- my-source: input.mlir -->
```text
module {
  func.func @test(%a: i32, %b: i32) -> i32 {
    %0 = my.add %a, %b : i32
    return %0 : i32
  }
}
```

我们约定 `my.add` 对两个 32 位整数做加法，超出位宽时保留低 32 位，即结果按模 `2^32` 计算。函数的含义因此很简单：接收两个整数，返回它们的和。例如实参为 2 和 3 时，按照这项约定应返回 5。

不过，此时我们在写的是供编译器处理的程序表示。读取这段文本时，工具还没有收到运行 `@test` 的实参。它首先需要建立这样的关系：

```text
函数的第一个参数 ──→ my.add 的第一个输入
函数的第二个参数 ──→ my.add 的第二个输入
                    my.add 的结果 ──→ return 的输入
```

这些结构 MLIR 本来就能保存。通用的 `Operation` 可以保存操作名、operand、result 及类型等信息，`Value` 可以把生产者和使用者连起来。我们不需要为每增加一种计算重新实现一套数据流图。

需要补充的是：**名为 `my.add` 的操作应当怎样使用这套通用结构。** 比如它必须有两个输入和一个结果，三者都是 i32；文本中的两个名字分别指向两个输入；C++ 代码应能取得左输入和右输入。

这里有两个不同层次：

| 层次 | 本例中的含义 |
|---|---|
| 一种操作的定义 | 所有 `my.add` 都遵守的结构、约束，以及可供框架调用的方法 |
| 程序中的一次操作 | 上面那一条具体加法，输入引用 `%a` 和 `%b`，结果被 return 使用 |

定义一次之后，一个程序可以包含许多条 `my.add`，每条各有自己的输入和结果。接下来要完成的是前一行的工作，使工具能够正确处理后一行的对象。

## 2. ODS 与声明式操作定义

先设想完全手工实现。我们至少要告诉工具：

- 操作的名字是 `my.add`，这样读取文本时才能找到对应定义。
- 左输入位于第一个 operand，右输入位于第二个 operand，方便变换取值。
- 合法对象有两个 i32 输入和一个 i32 结果，以便检查错误。
- `my.add %a, %b : i32` 中每段文本怎样对应这些字段，以便解析和打印。

这些工作中有很多重复关系。一旦已经声明“第一个输入叫 lhs，类型为 I32”，生成一个 `getLhs()` 访问器和对应的类型检查就有了确定依据。若每个操作都手写这些重复代码，很容易出现声明、访问和验证不一致。

MLIR 因而提供 **ODS（Operation Definition Specification）**：使用基于 TableGen 的描述，集中写下操作的信息，再生成相应的 C++ 代码。

这里的“生成”可以按字面理解：一个工具读取描述文件，把类声明和方法实现写进源码文件，随后由 C++ 编译器编译。生成器掌握的是 MLIR 已经规定的结构和实现规则。

计算语义还需要开发者约定。例如“结果等于两个输入之和”不能由名字 `add` 自动推断。本章先让工具拥有表示和检查这项计算的能力；真正把它变成可执行计算，需要后续的变换或执行实现。

## 3. 字段约束与文本格式

下面是本例完整的 `MyOps.td`。先沿着“名字、输入、结果、文本写法”读，就能把它与前面的需求对应起来：

<!-- my-source: MyOps.td -->
```text
#ifndef MY_OPS_TD
#define MY_OPS_TD
include "mlir/IR/OpBase.td"

def My_Dialect : Dialect {
  let name = "my";
  let cppNamespace = "::mlir::my";
}

def My_AddOp : Op<My_Dialect, "add"> {
  let summary = "Add two i32 values with wraparound semantics";
  let description = [{
    The result is the sum modulo 2^32. This stage defines representation
    and verification only; execution/lowering is introduced in a later stage.
  }];
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
#endif
```

`My_Dialect` 定义一组操作所属的方言。这里的方言名是 `my`，操作在其中叫 `add`，两者组成 IR 里的完整名字 `my.add`。`cppNamespace` 则安排生成的 C++ 类放在 `mlir::my` 命名空间中。方言在这个例子里先只组织这一种操作。

真正规定加法结构的是下面两行：

```text
let arguments = (ins I32:$lhs, I32:$rhs);
let results = (outs I32:$result);
```

第一行按顺序声明两个输入，分别叫 lhs 和 rhs，类型约束都是 I32；第二行声明一个 i32 结果。这里的 I32 指 32 位 signless integer。操作如何解释这些位，由计算语义规定；本例采用前面约定的回绕加法。

`lhs` 是**定义中的字段名**，`%a` 是**某段程序里被引用的值的文本名字**。对于开头那条加法，lhs 对应 `%a`；另一条加法可以让 lhs 对应 `%x`。所以把字段叫 lhs，不是在创建一个名为 `%lhs` 的变量。

现在回看文本写法：

```text
$lhs       `,`      $rhs       attr-dict       `:`     type($result)
 ↓          ↓        ↓         本例为空         ↓            ↓
%a          ,       %b                         :           i32
```

`assemblyFormat` 把字段安排到文本里：先写左输入、逗号、右输入，再写可选的额外属性字典，最后写冒号和结果类型。当前输入没有额外属性，所以看不到字典。操作名和结果赋值部分由外围的操作打印流程处理，不需要再次写进这行格式。

这样，工具需要的结构与语法已经有了来源。`summary` 和 `description` 记录含义；输入、结果及格式声明则能用于生成具体方法。接下来看看其中一项如何变成可用的 C++。

## 4. TableGen 生成代码

假设当前目录保存着刚才的 `MyOps.td`，使用对应版本的 `mlir-tblgen`，并让 `MLIR_INCLUDE_DIR` 指向包含 `mlir/IR/OpBase.td` 的 MLIR include 目录。下面的命令会生成操作类声明及部分类内方法：

<!-- my-command: tblgen -->
```bash
mlir-tblgen --gen-op-decls MyOps.td \
  -I "$MLIR_INCLUDE_DIR" -o MyOps.h.inc
```

命令里的输入是 **操作定义 `MyOps.td`**，输出是 **C++ 源码片段 `MyOps.h.inc`**。此时还没有读取开头那个 `@test`，也没有创建它里面的加法对象。

在生成的 `mlir::my::AddOp` 类中，实际可以看到：

<!-- my-generated: MyOps.h.inc -->
```cpp
::mlir::TypedValue<::mlir::IntegerType> getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}
```

`getODSOperands(0)` 取得定义中的第一个输入组。本例这一组只有一个输入，所以取其第一个元素就是 lhs。返回值是一个表示整数类型 Value 的句柄，指向这条操作已有的输入。

现在把它放回开头的 IR：若 C++ 代码中的 `add` 是那条 `my.add` 的访问句柄，`add.getLhs()` 得到的就是函数参数 `%a` 所对应的 Value。它不会读取一个运行时整数，也没有执行加法。`getRhs()` 同理取得 `%b`。

这就接上了前面写 Pass 的经验：过去使用现成的 `arith::AddIOp` 读取输入；现在 ODS 帮我们为自己的操作生成了相应访问方式。底层仍是通用 Operation 保存的那组 operand，生成的 AddOp 提供符合这项定义的访问接口。

构造操作也有同样的对应。选择生成操作实现的后端后，`MyOps.cpp.inc` 中有这样一个实际方法：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
void AddOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, ::mlir::Type result,
                  ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(result);
}
```

这个 builder 把调用者提供的两个 Value 按顺序写入待构造状态，并记录结果类型。它与访问器正好相接：构造时放在第一个位置的值，之后由 `getLhs()` 取得。这里的 `build` 是构造 IR 对象所需信息的方法，与编译链接工程的 build 是两回事。

可以看到，这段 builder 没有执行类型检查；操作验证另有对应方法。构造、访问和验证各自完成不同工作，ODS 让它们依据同一份定义生成。

本例工程还从同一个 `.td` 生成操作实现、方言声明和方言实现。四个输出的组织方式是：

| 输出 | 保存的 C++ 内容 |
|---|---|
| `MyOps.h.inc` | AddOp 声明、访问器等 |
| `MyOps.cpp.inc` | 构造、解析、打印和约束检查等实现，以及操作类型列表 |
| `MyDialect.h.inc` | MyDialect 类声明 |
| `MyDialect.cpp.inc` | MyDialect 构造函数等实现 |

它们共同为工具提供处理 `my.add` 的代码。首次理解这一过程，关键是知道声明如何影响生成的方法，而不需要逐行阅读全部生成产物。

## 5. 方言注册与工具集成

生成的 `.inc` 仍然是 C++ 文本。手写的头文件和实现文件通过 `#include` 把它们纳入普通 C++ 编译。例如，头文件中有：

```cpp
#include "MyDialect.h.inc"
#define GET_OP_CLASSES
#include "MyOps.h.inc"
```

`GET_OP_CLASSES` 是预处理时的选择开关，让生成文件中相应的操作类内容进入当前源码。实现文件采用同样方式包含 `.cpp.inc` 中的实现。编译和链接完成后，这些能力就成为工具的一部分；工具处理 IR 时调用的是已经编译好的代码。

但“代码已编译进去”还缺少一层联系。解析器读到字符串 `my.add` 时，需要知道应该使用哪套解析、验证和打印方法。**注册就是把操作名与处理这种操作的代码联系起来。**

本例方言初始化时登记它包含的操作。将生成的单项类型列表展开后，核心代码等价于：

<!-- my-equivalent: initialize -->
```cpp
void MyDialect::initialize() {
  addOperations<AddOp>();
}
```

这次调用登记的是 **AddOp 这类操作的定义信息**。它没有向某个函数插入一条加法。框架可以从 AddOp 获得操作名及对应方法；其中操作名也是生成的：

<!-- my-generated: MyOps.h.inc -->
```cpp
static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("my.add");
}
```

工具入口还需要告诉 MLIR，它能提供哪些方言。下面是 `main` 中的相关部分：

<!-- my-source-fragment: my-opt.cpp -->
```cpp
mlir::DialectRegistry registry;
registry.insert<mlir::my::MyDialect, mlir::func::FuncDialect>();
return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "My dialect: stage 01\n", registry));
```

Registry 提供 My 和 Func 方言的加载信息。工具使用的 `MLIRContext` 可以据此加载 MyDialect；其生成的构造函数调用 `initialize()`，将 AddOp 登记进去。`MlirOptMain` 提供常见的读入 IR、运行所选 Pass、验证和打印的工具流程，因此这里不用再手写文件读取和命令行驱动。

现在已经有了完整的接入关系：

```text
MyOps.td 描述操作
  → mlir-tblgen 写出 C++ 代码
  → 手写源码包含生成代码，编译链接为 my-opt

运行 my-opt
  → Registry 使 MyDialect 可被加载
  → 加载方言时登记 AddOp
  → 框架能够按 my.add 找到相应处理方法
```

上半段使工具拥有代码，下半段使运行中的框架能够使用它。接下来让这些方法处理一次具体输入。

## 6. IR 解析、验证与打印

假设已经构建出 `my-opt`，当前目录下有这个可执行文件，以及保存第一节程序的 `input.mlir`。直接运行：

<!-- my-command: run -->
```bash
./my-opt input.mlir
```

本例没有指定变换 Pass，工具读取、验证并重新打印这份 IR，实际输出为：

<!-- my-output: custom -->
```text
module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = my.add %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

这次运行中，前面准备的各项能力按下面的关系配合：

1. 读取函数参数时，建立两个 i32 的 BlockArgument；文本里的 `%a`、`%b` 指向它们。
2. 读到 `my.add`，框架找到已注册的操作定义，使用生成的专用 parser 读取后面的文本。
3. parser 按格式读到两个输入引用，找到对应 Value，并记录结果类型。框架据此构造具体的 Operation。
4. 读取 return 时，将它的输入接到新加法的结果上。工具验证 IR，其中包括输入、结果数量及 I32 类型约束。
5. printer 从当前对象读取输入、结果和类型，再按规定的文本格式输出。

打印器为函数参数选择了 `%arg0`、`%arg1` 这样的名字，引用关系保持不变。**整个过程完成的是让编译器读懂并检查一项计算的表示。** 它没有调用 `@test(2, 3)`，所以输出是 IR，而不是数字 5。

换一种打印方式，更容易直接看见操作的结构：

<!-- my-command: generic -->
```bash
./my-opt input.mlir --mlir-print-op-generic
```

下面只摘录其中的加法行：

<!-- my-output-fragment: generic -->
```text
%0 = "my.add"(%arg0, %arg1) : (i32, i32) -> i32
```

通用格式明确列出两个输入类型和一个结果类型。它与前面的简洁格式描述同一个对象。再次读取通用格式时，通用 parser 按统一语法填写操作字段；之后仍然使用已注册操作的约束检查。这给了我们一个直接观察“结构可以写出来，但不符合定义”的机会。

## 7. 类型约束与错误诊断

只把第一个参数改为 i64，并用通用语法显式记录这个类型：

<!-- my-invalid: operand-type | operand #0 must be 32-bit signless integer -->
```text
module {
  func.func @bad(%a: i64, %b: i32) {
    %0 = "my.add"(%a, %b) : (i64, i32) -> i32
    return
  }
}
```

这段文本足以描述一个通用 Operation：有名字、两个 operand、一个结果，引用本身也找得到。但它违背了我们给 `my.add` 规定的 I32 输入约束。

用同一个工具读取这段输入，会得到非零退出状态，诊断中包含：

```text
'my.add' op operand #0 must be 32-bit signless integer, but got 'i64'
```

这条信息可以追溯到 ODS 的 `I32:$lhs`。生成的验证代码取得第 0 个 operand 的类型，检查它是否为 32 位 signless integer；遇到 i64 就报告错误。数量约束也有相应检查，例如只提供一个 operand 会被拒绝。

于是，“在 `.td` 中写下 I32”已经不只是文档说明：它产生了代码，代码接入工具，并在读取不符合约定的 IR 时影响了结果。

回到第二节的语义约定，也能看出验证的边界。检查两个输入是不是 i32，并不能证明未来的实现真的把它们相加。如果后续错误地把 `my.add` 转换为减法，类型检查可能仍然通过；变换作者还需要保持计算含义，并用相应测试检查。

## 8. 操作定义与 IR 变换

现在工具已经能保存一个 `my.add`，C++ 代码可以取得其两个输入和结果，错误的结构或类型也会被拒绝。下一步，前面学过的 Pattern 和 Pass 就有了可以处理的对象。

对于本例约定的 i32 回绕加法，可以把目标变换写成下面的局部示意：

```text
变换前：%r = my.add     %a, %b : i32
变换后：%r = arith.addi %a, %b : i32
```

实现时会读取原操作的两个 operand，创建一条无 overflow flags 的 `arith.addi`，将旧结果的使用转接到新结果，再删除原操作。这里打印成相同的 `%r` 只是方便对照；内存中创建了新的结果 Value，并更新使用关系。

这就是定义与变换的连接：**定义使一种计算具有可构造、可检查、可访问的表示；变换再根据它的含义，将这份表示改成下一阶段需要的形式。** 生成 AddOp 类承担前一项工作，不能自动替我们完成后一项。

当前最小工程尚未加入这条 Pattern。接下来直接进入[操作定义：语义、字段与验证](../../compiler/ir_definition/op_definition)，其中用 clamp 给出实际的 Pattern 展开，并继续解释静态属性与验证。这里不再另建一套以 my.add 重复各项机制的课程。即使变成了 `arith.addi`，仍需后续编译路径或执行设施才能得到机器上的运行结果。

读到这里，可以用下面两个小变化检查自己建立的关系：

- 将 ODS 的 lhs 改名为 left，同时更新 assemblyFormat 的对应字段引用。输入仍在第一个位置，那么生成的访问器与打印出来的 IR，分别可能发生什么变化？
- 在一个函数中写两条 `my.add`，是不是需要登记两次 AddOp？这两条操作会共用什么，又各自保存什么？

它们分别检验“定义字段与程序中的值”的区别，以及“一种操作与一次操作”的区别，不要求记住生成文件里的全部方法。

## 实现、实践与查阅

本章输入、输出和生成代码摘录对应 LLVM `llvmorg-20.1.8`。工具验证的是编译器侧的表示与约束；第 8 节的变换是后续实现目标，未记为当前工具已有功能。

需要亲手构建时，使用 [最小工程与构建说明](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)；想验证字段改名的预测，可做 [字段改名练习](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/tasks/01-rename.md)。这些材料提供完整依赖、命令和源码，正文中的理解不依赖先运行实验。

继续查阅可以按具体问题选择：

- 更多字段与验证关系：[操作定义](../../compiler/ir_definition/op_definition)。
- 文本怎样变成对象，再打印回来：[解析与打印](../../compiler/ir_definition/assembly_format)。
- ODS 的准确字段与生成规则：[固定版本 Operations 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)。
- 工程的生成、编译与接入：[固定版本 Creating a Dialect](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Tutorials/CreatingADialect.md)。

若要确认注册的实现，可以从固定版本的 `Dialect::addOperations` 进入 `RegisteredOperationName::Model`：前者登记操作类型，后者把框架的解析、打印和验证入口接到该类型的方法。追到这层即可核对第五节的关系，不必为理解本章继续展开全部模板实现。
