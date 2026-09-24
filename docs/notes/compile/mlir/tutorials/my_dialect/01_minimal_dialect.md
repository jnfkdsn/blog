---
order: 1
title: 01：让自己的工具认识 my.add
updated: 2026-09-24
---

# 01：让自己的工具认识 my.add

前面已经会阅读 IR，也见过 Pass 怎样修改已有操作。现在从一个完整的小工程出发，解释工具为什么能认识一个自己定义的操作。**这一篇只走通“定义 → 生成 C++ → 编译注册 → 读取、验证、打印”的过程。**

配套 [lab 入口](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)与本篇对应的 [阶段 01 源码](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect/stages/01-minimal)位于独立仓库。本地路径是 `aicompiler-labs/llvm-mlir/my-dialect/`。本篇展示的关键代码与输出足以解释过程，完整构建配置在 lab 查阅。

## 1. 先看工具最终要读懂什么

<!-- my-source: input.mlir -->
```text
module {
  func.func @test(%a: i32, %b: i32) -> i32 {
    %0 = my.add %a, %b : i32
    return %0 : i32
  }
}
```

我们约定 `my.add` 接收两个 i32，返回按 32 位回绕的和，即模 2³² 加法。这项语义约定为后续转成 `arith.addi` 提供依据。当前版本实现它的表示和合法性检查，没有执行加法。

第一步的成功标准很具体：自己的 `my-opt` 能读取这段文本，并正常打印同一份 IR。现成的标准工具没有注册我们的 My dialect，不能仅凭名字猜出操作定义。

从 workspace 根目录运行：

```bash
python3 aicompiler-labs/llvm-mlir/my-dialect/observe.py
```

脚本先生成并编译工具，再展示生成代码片段、正常打印和通用打印。正常打印的实际输出如下：

<!-- my-output: custom -->
```text
module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = my.add %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

`%a`、`%b` 被 printer 命名为 `%arg0`、`%arg1`，定义和使用关系保持不变。此时先记住：我们已经得到一个认识新操作的工具。接下来沿构建过程查清这种能力从哪里来。

## 2. 先把工程中的文件放到正确位置

参考源码仅有六个文件：

```text
stages/01-minimal/
├── MyOps.td          操作与方言声明
├── MyOps.h           接入生成的 C++ 声明
├── MyOps.cpp         接入生成实现，注册操作
├── my-opt.cpp        工具入口，提供方言注册表
├── CMakeLists.txt    规定生成、编译、链接关系
└── input.mlir        工具处理的输入
```

构建后会出现另一组文件：

```text
MyOps.td
  ├─ 生成 MyDialect.h.inc / MyDialect.cpp.inc
  └─ 生成 MyOps.h.inc / MyOps.cpp.inc
                  ↓ 被手写 .h / .cpp 包含
               MyIR 库
                  ↓ 与工具入口和 MLIR 库链接
                my-opt
```

`.td` 是生成器的输入，`.inc` 是生成器输出的 C++ 片段，`.h/.cpp` 是我们维护的接入代码。它们共同组成一个程序，**不是运行时依次打开的四种文件**。处理 `input.mlir` 时，生成的逻辑早已编译进工具。

## 3. 在 ODS 中写出这一项操作

[MyOps.td](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/stages/01-minimal/MyOps.td) 的完整内容：

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

先沿主例需要的内容逐项对应：

| 声明 | 本例中确定了什么 |
|---|---|
| `name = "my"` | 操作名使用 my 方言前缀 |
| `cppNamespace = "::mlir::my"` | 生成的 C++ 类放在这个命名空间 |
| `Op<My_Dialect, "add">` | 完整操作名为 my.add，生成操作类 AddOp |
| `I32:$lhs, I32:$rhs` | 两个 operand，分别命名为 lhs/rhs，类型均要求 i32 |
| `I32:$result` | 一个 i32 结果 |
| `assemblyFormat` | 文本按“左输入、逗号、右输入、可选属性字典、冒号、结果类型”排列 |

`lhs` 是操作字段的名字；`%a` 是某段文本中的 SSA 名称。把同一份 IR 的 `%a` 换名，不会改变生成类的 `getLhs()`。反过来，重命名 ODS 字段，会影响生成访问器，但不必改变用户程序的 SSA 名字。

这里用了 TableGen 的记录语法，ODS 规定这些字段如何描述 MLIR 操作。我们先使用它声明结构，不需要先学习整个 TableGen 语言。

类型约束直接写成 I32，使两项输入和结果都固定为 i32。后面若支持多种类型，才需要考虑怎样表达它们之间的关系。本篇没有加入自定义 Trait、Interface、Type、Attribute 或 Region。

## 4. 构建器究竟生成了什么

[CMakeLists.txt](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/stages/01-minimal/CMakeLists.txt) 中指定同一份 `.td`，使用不同生成选项产生四个文件：

```cmake
set(LLVM_TARGET_DEFINITIONS MyOps.td)
mlir_tablegen(MyOps.h.inc -gen-op-decls)
mlir_tablegen(MyOps.cpp.inc -gen-op-defs)
mlir_tablegen(MyDialect.h.inc -gen-dialect-decls -dialect=my)
mlir_tablegen(MyDialect.cpp.inc -gen-dialect-defs -dialect=my)
add_public_tablegen_target(MyIncGen)
```

这些 CMake 声明建立调用 `mlir-tblgen` 的构建规则。依赖目标保证编译使用 `.inc` 的 C++ 文件前，生成物已就绪。不要把它与下一阶段要讲的 Op builder 混淆：这里是在构建编译器程序，还没有构造输入程序中的 Operation。

本地打开 `artifacts/builds/my-dialect-01/MyOps.h.inc`，先搜索 `class AddOp`，然后看这个类里的 `getLhs()`、`getRhs()`。它们对应 `.td` 中命名的两个输入。文件前面的 adaptor 也可能有同名访问器，第一次先限定在 AddOp 类内阅读。

继续在 `MyOps.cpp.inc` 搜索：

```bash
rg -n 'AddOp::build|AddOp::parse|AddOp::print|AddOp::verifyInvariants'   artifacts/builds/my-dialect-01/MyOps.cpp.inc
```

你会找到构造状态、解析打印和字段验证的生成方法。本次只确认这些能力有对应的 C++ 代码；不展开所有重载。下一次需要构造或验证的新行为时，再回到相应函数。

这也解释了为什么手写文件中没有 parser 和类型检查函数，工具仍能读懂并检查这个操作：相应实现由声明生成了。

## 5. 生成代码怎样接入手写 C++

先看 [MyOps.h](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/stages/01-minimal/MyOps.h)：

<!-- my-source: MyOps.h -->
```cpp
#ifndef MY_OPS_H
#define MY_OPS_H
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "MyDialect.h.inc"
#define GET_OP_CLASSES
#include "MyOps.h.inc"
#endif
```

前几个 include 提供 MLIR 基础类型；`MyDialect.h.inc` 给出生成的方言类声明；`GET_OP_CLASSES` 选择生成文件中的操作类声明部分，再包含 `MyOps.h.inc`。因此其他 C++ 文件只需包含 MyOps.h，就能使用 MyDialect 和 AddOp。

再看 [MyOps.cpp](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/stages/01-minimal/MyOps.cpp)：

<!-- my-source: MyOps.cpp -->
```cpp
#include "MyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
using namespace mlir;
using namespace mlir::my;
#include "MyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "MyOps.cpp.inc"

void MyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MyOps.cpp.inc"
      >();
}
```

上半部分包含生成实现，下半部分定义方言初始化时要登记哪些操作。`GET_OP_LIST` 选择同一生成文件中的操作类型列表；在这个阶段，列表里只有 `mlir::my::AddOp`，交给 `addOperations` 注册。

同一个 `.inc` 被包含两次，是通过不同宏选择不同片段。它在预处理/编译时拼接 C++ 内容，不是程序运行时读文件，也不是把操作对象创建了两次。

至此，类的声明和实现已经接好，方言知道自己有哪些操作。还差最后一层：让具体工具能够加载这个方言。

## 6. 工具运行时怎样找到这个操作

[my-opt.cpp](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/stages/01-minimal/my-opt.cpp) 的完整入口：

<!-- my-source: my-opt.cpp -->
```cpp
#include "MyOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::my::MyDialect, mlir::func::FuncDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "My dialect: stage 01\n", registry));
}
```

registry 向工具提供 My 与 Func dialect 的加载信息。输入中的 `func.func`/`return` 来自 Func，`my.add` 来自 My；外层 module 是工具支持的 builtin 操作。加载 My dialect 时，前面的 initialize 会登记 AddOp。

现在可以沿本次输入走一遍运行链：

```text
my-opt 接收 input.mlir
  → 解析器需要识别 my.add
  → Context 可从 registry 加载 My dialect，并取得已注册的 AddOp 信息
  → 生成的 parser 读取两个 SSA operand 引用与格式中的类型
  → 构造内存中的 Operation
  → 工具运行完整验证，其中包含生成的操作约束检查
  → printer 将对象写回文本
```

这里没有手写一个“执行加法”的回调。读取 `my.add` 是建立一个表达加法的 IR 节点；将它转换成标准操作、继续生成可执行代码，是后续任务。

可以把构建与运行两条链连起来理解：ODS 使工具在构建后具有认识 AddOp 的代码，注册使运行中的工具能找到这套代码，parser 才能按契约构造具体对象。

## 7. 换一种打印，确认对象结构

观察脚本还显示了通用格式，其中 my.add 一行是：

```text
%0 = "my.add"(%arg0, %arg1) : (i32, i32) -> i32
```

它显式写出两个输入和一个结果的类型。正常格式里的逗号、冒号等写法由 assemblyFormat 规定；通用格式直接展示操作结构，两者最终对应同一种内存对象。脚本将通用输出重新解析并正常打印，结果与第一节一致。

进一步改变一个条件：如果用通用格式给 my.add 提供 i64 输入，生成的 I32 约束会拒绝它。这样既能观察“工具认识名字”，也能观察“认识之后会按声明检查对象”，两者有不同的失败原因。

## 8. 轮到你改一个字段

任务是将第一个 operand 的 ODS 名称从 lhs 改为 left，同时更新 assemblyFormat 中的对应引用，观察生成访问器与 IR 文本是否变化。先写预测，再构建验证。

先创建你的副本：

```bash
python3 aicompiler-labs/llvm-mlir/my-dialect/observe.py --prepare-task
```

编辑 `aicompiler-labs/llvm-mlir/my-dialect/work/01-rename/MyOps.td`，然后运行：

```bash
python3 aicompiler-labs/llvm-mlir/my-dialect/observe.py --task
```

完整要求在 [lab 任务单](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/tasks/01-rename.md)。副本不会覆盖参考版本，重复准备也不会覆盖已存在的改动。修改后记录几行：改了哪里、生成代码如何变化、原输入是否仍成立。把这些交给我审阅即可。

本轮能沿一个字段解释 `.td → .inc → 工具行为`，就有了继续扩展的支点。下一步会把 my.add 接到已学过的 Pattern/Pass，转换为 arith.addi。

## 版本、复现与网页链接

本篇对应 `stages/01-minimal/`；后续功能增加到新的阶段目录，避免旧教程突然指向更复杂的实现。固定 LLVM `llvmorg-20.1.8`，完整构建方法在 [lab README](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/README.md)。

本篇站内链接由 blog 构建，源码链接指向独立的 aicompiler 仓库。`artifacts/...` 是本地复现路径，`.inc` 生成物不要求上传；正文已给出关键观察结果。两边分别提交推送后，对应 GitHub 链接才会存在。main 链接会随分支更新；正式固定版本可改用已发布 commit 的永久链接。

作者验证入口为 `aicompiler-labs/llvm-mlir/docs/validate_my_dialect.py`，检查源码对应、打印往返、诊断及字段改名的影响；不将这些结果记为学习者已完成任务。

机制参考：[操作定义](../../compiler/ir_definition/op_definition)、[解析与打印](../../compiler/ir_definition/assembly_format)。完整字段规范查 [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)，生成与链接关系查 [Creating a Dialect](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)；具体 API 以本地固定版本生成文件为准。
