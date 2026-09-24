---
order: 1
title: 01：让自己的工具认识 my.add
updated: 2026-09-24
---

# 01：让自己的工具认识 my.add

前面已经会阅读 IR，也见过 Pass 怎样修改已有操作。现在从一个完整的小工程出发，解释工具为什么能认识一个自己定义的操作。**这一篇只走通“定义 → 生成 C++ → 编译注册 → 读取、验证、打印”的过程。**

配套 [lab 入口](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)与本篇对应的 [阶段 01 源码](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect/stages/01-minimal)位于独立仓库。本地路径是 `aicompiler-labs/llvm-mlir/my-dialect/`。本篇直接给出分步命令、各步产物和生成代码的关键内容；不需要先打开生成文件才能读懂。所有命令都从 workspace 根目录、在同一个 Bash 终端按顺序执行，使用已经构建好的 LLVM/MLIR 20.1.8。

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

先预览最终结果。下面是工具构建完成后，读取上面输入的实际输出；后文会逐步构建并亲自运行这个工具：

<!-- my-output: custom -->
```text
module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = my.add %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

`%a`、`%b` 被 printer 命名为 `%arg0`、`%arg1`，定义和使用关系保持不变。接下来从源码开始，逐步得到这个结果。

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

## 4. 先配置工程，再单独生成 C++

### 4.1 CMake 先建立构建规则

先为三个已有位置取短名字：源码目录、LLVM 构建目录、本教程的构建目录。这里使用独立的 `my-dialect-01-steps`，便于观察首次构建；此前脚本生成的 `my-dialect-01` 可以保留。

<!-- my-command: configure -->
```bash
MYLAB_SRC="$PWD/aicompiler-labs/llvm-mlir/my-dialect/stages/01-minimal"
MYLAB_LLVM="$PWD/artifacts/builds/mlir-20.1.8"
MYLAB_BUILD="$PWD/artifacts/builds/my-dialect-01-steps"

cmake -S "$MYLAB_SRC" -B "$MYLAB_BUILD" -G Ninja \
  -DMLIR_DIR="$MYLAB_LLVM/lib/cmake/mlir" \
  -DLLVM_DIR="$MYLAB_LLVM/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang
```

成功输出的末尾包含 `Configuring done`、`Generating done` 与构建目录位置。此时得到 `CMakeCache.txt` 和 `build.ninja` 等构建配置，**还没有编译自己的操作实现**。MLIR_DIR/LLVM_DIR 用来找到已有 LLVM/MLIR 的配置、头文件和库，不会在这里重新编译整个 LLVM。

CMake 根据什么建立生成规则？本工程 [CMakeLists.txt](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/stages/01-minimal/CMakeLists.txt) 的这六行规定了四项生成任务：

```cmake
set(LLVM_TARGET_DEFINITIONS MyOps.td)
mlir_tablegen(MyOps.h.inc -gen-op-decls)
mlir_tablegen(MyOps.cpp.inc -gen-op-defs)
mlir_tablegen(MyDialect.h.inc -gen-dialect-decls -dialect=my)
mlir_tablegen(MyDialect.cpp.inc -gen-dialect-defs -dialect=my)
add_public_tablegen_target(MyIncGen)
```

`mlir_tablegen` 是 CMake 辅助函数，它安排构建时调用 `mlir-tblgen`。四次调用读取同一个 `.td`，选择不同生成器；`MyIncGen` 把这些任务归为一个可以单独构建的目标。

<details>
<summary>完整 CMakeLists.txt：需要核对工程配置时展开</summary>

<!-- my-source: CMakeLists.txt -->
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_dialect_stage01 LANGUAGES C CXX)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
set(LLVM_TARGET_DEFINITIONS MyOps.td)
mlir_tablegen(MyOps.h.inc -gen-op-decls)
mlir_tablegen(MyOps.cpp.inc -gen-op-defs)
mlir_tablegen(MyDialect.h.inc -gen-dialect-decls -dialect=my)
mlir_tablegen(MyDialect.cpp.inc -gen-dialect-defs -dialect=my)
add_public_tablegen_target(MyIncGen)
add_library(MyIR STATIC MyOps.cpp)
add_dependencies(MyIR MyIncGen)
target_link_libraries(MyIR PUBLIC MLIRIR)
add_executable(my-opt my-opt.cpp)
target_link_libraries(my-opt PRIVATE MyIR MLIROptLib MLIRFuncDialect)
```

`include_directories` 同时包含源码与构建目录，所以手写文件能找到构建目录里的 `.inc`。`add_dependencies(MyIR MyIncGen)` 保证先生成再编译。最后几行的库与可执行目标会在第 5、6 节分别构建。

</details>

### 4.2 只运行 TableGen，暂不编译 C++

<!-- my-command: generate -->
```bash
cmake --build "$MYLAB_BUILD" --target MyIncGen -j2
```

首次运行会看到下面四项任务；并行执行顺序可能不同，因此这里省略进度编号：

```text
Building MyOps.h.inc...
Building MyOps.cpp.inc...
Building MyDialect.h.inc...
Building MyDialect.cpp.inc...
```

现在构建目录中新增四个 `.inc` 文件。它们还是 C++ 文本，没有变成库或可执行文件。重复运行时若输入未变化，Ninja 会报告 `no work to do`，不必删除产物来重复观察。

想看到完整的生成器调用，可在这条构建命令末尾加 `--verbose`；若已无任务要运行，用 `ninja -C "$MYLAB_BUILD" -t commands MyIncGen` 查看保存的命令。

<details>
<summary>直接调用 mlir-tblgen 是什么样子？</summary>

下面是本例可独立运行的最小生成命令。输入是 MyOps.td，`-I` 用来找到它包含的 MLIR 定义，`-o` 指定输出。为了与 CMake 管理的产物区分，输出放在 `tblgen-preview/`：

<!-- my-command: tblgen -->
```bash
mkdir -p "$MYLAB_BUILD/tblgen-preview"
"$MYLAB_LLVM/bin/mlir-tblgen" "$MYLAB_SRC/MyOps.td" \
  -I "$PWD/upstream/llvm-project/mlir/include" \
  -gen-op-decls -o "$MYLAB_BUILD/tblgen-preview/MyOps.h.inc"
```

切换为 `-gen-op-defs` 生成操作实现；切换为 `-gen-dialect-decls -dialect=my` 或 `-gen-dialect-defs -dialect=my` 生成方言声明或实现，同时更改输出文件名。本例只需上述 include 路径；其他工程可能还需要 LLVM 头文件或生成目录。CMake 的实际命令还包含依赖记录等选项。

</details>

### 4.3 先看方言：名字变成类，构造函数调用初始化

MyDialect.h.inc 中生成了下面这个类，摘录省略文件头、外层命名空间与类型标识宏：

<!-- my-generated: MyDialect.h.inc -->
```cpp
class MyDialect : public ::mlir::Dialect {
  explicit MyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("my");
  }
};
```

`.td` 中的 `name = "my"` 变成 `getDialectNamespace()` 的返回值。类声明提供 `initialize()`，但此处没有它的函数体。

再看 MyDialect.cpp.inc 的构造函数。下面是实际生成内容，仅整理空白：

<!-- my-generated: MyDialect.cpp.inc -->
```cpp
MyDialect::MyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MyDialect>()) {
  initialize();
}
```

它调用 `initialize()`，而工程作者必须提供这个函数，登记当前方言的操作。第 5 节正好补上这一环。此时不要把“生成类”理解成已经创建了方言对象：对象是在工具运行时由 Context 加载方言时创建的。

### 4.4 再看操作：命名字段变成访问器

MyOps.h.inc 中的 AddOp 类记录操作名：

<!-- my-generated: MyOps.h.inc -->
```cpp
static constexpr ::llvm::StringLiteral getOperationName() {
  return ::llvm::StringLiteral("my.add");
}
```

同一个类里的两个访问器如下：

<!-- my-generated: MyOps.h.inc -->
```cpp
::mlir::TypedValue<::mlir::IntegerType> getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IntegerType> getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
}
```

这段代码把 `.td` 的字段与底层 Operation 连了起来：`lhs` 对应第 0 个 operand，`rhs` 对应第 1 个 operand；访问器取得已经存在的 SSA Value。它们没有执行加法，也没有另外存一份 lhs/rhs 数据。结果字段同样生成 `getResult()`。

MyOps.cpp.inc 中还生成了构造状态的方法。这里只取一个重载：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(result);
}
```

它向 OperationState 写入两个 operand 和一个结果类型，供 C++ 构造操作时使用。这里的 `AddOp::build` 与前面的 `cmake --build` 处于不同过程：前者构造 IR 的状态，后者编译编译器程序。读取文本时会走 parser，不是先执行这段 build 再执行 parser。

### 4.5 文本格式与类型约束也有生成实现

`assemblyFormat` 指定先读 lhs、逗号、rhs。生成的 `AddOp::parse` 中就有对应代码，以下为连续摘录：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
lhsOperandsLoc = parser.getCurrentLocation();
if (parser.parseOperand(lhsRawOperand))
  return ::mlir::failure();
if (parser.parseComma())
  return ::mlir::failure();

rhsOperandsLoc = parser.getCurrentLocation();
if (parser.parseOperand(rhsRawOperand))
  return ::mlir::failure();
```

在读完属性字典、冒号与结果类型之后，parser 还需要把这些文本引用解析成实际 Value：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
result.addTypes(resultTypes);
if (parser.resolveOperands(lhsOperands, odsBuildableType0, lhsOperandsLoc, result.operands))
  return ::mlir::failure();
if (parser.resolveOperands(rhsOperands, odsBuildableType0, rhsOperandsLoc, result.operands))
  return ::mlir::failure();
return ::mlir::success();
```

这里 I32 让生成器能够直接构造输入所需的 i32 类型；`resolveOperands` 将先前读到的 SSA 引用解析为 operand，写入状态。文本中的结果类型则由 `result.addTypes` 记录。

反方向的 `AddOp::print` 会通过 `getLhs()`、`getRhs()` 打印 operand 引用，再输出属性字典与结果类型。例如其中的连续摘录是：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
_odsPrinter << getLhs();
_odsPrinter << ",";
_odsPrinter << ' ';
_odsPrinter << getRhs();
```

因此只写一行 assemblyFormat，确实可以得到一对 parser/printer。

类型检查也有对应代码。MyOps.cpp.inc 为 I32 生成的辅助函数如下：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
static ::llvm::LogicalResult __mlir_ods_local_type_constraint_MyOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}
```

生成的 `verifyInvariantsImpl()` 会对两个 operand 和结果分别调用这个检查函数。比如它遍历第一个 operand 对应范围时，调用如下：

<!-- my-generated: MyOps.cpp.inc -->
```cpp
for (auto v : valueGroup0) {
  if (::mlir::failed(__mlir_ods_local_type_constraint_MyOps1(*this, v.getType(), "operand", index++)))
    return ::mlir::failure();
}
```

结构上的“两个输入、一个结果”还有生成到操作类上的数量约束，由验证框架检查。后文将故意给第一个输入传入 i64，观察这里的错误信息。

至此，四个文件的分工已经具体可见：方言声明/构造函数、操作声明/访问器，以及操作构造、解析、打印、验证的实现。下一步把这些 C++ 片段接入库。

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

现在用刚才生成的 C++ 编译方言库：

<!-- my-command: library -->
```bash
cmake --build "$MYLAB_BUILD" --target MyIR -j2
```

首次运行的任务内容是：

```text
Building CXX object CMakeFiles/MyIR.dir/MyOps.cpp.o
Linking CXX static library libMyIR.a
```

MyOps.cpp 包含的 `.inc` 随它一起编译，不会各自编译成独立 `.o`。这里实际生成 `CMakeFiles/MyIR.dir/MyOps.cpp.o`，再归档为 `libMyIR.a`。方言定义已经进入库，但库本身没有 main，不能直接接收 input.mlir。接下来编译工具入口并链接它。

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

编译并链接工具：

<!-- my-command: tool -->
```bash
cmake --build "$MYLAB_BUILD" --target my-opt -j2
```

在前面已构建好 MyIR 的情况下，新增任务是：

```text
Building CXX object CMakeFiles/my-opt.dir/my-opt.cpp.o
Linking CXX executable my-opt
```

CMake 的 `target_link_libraries` 把 MyIR、工具驱动库 MLIROptLib 和 Func dialect 库接到 my-opt。现在才得到可执行文件。下面亲自让它读取本章开头的输入：

<!-- my-command: run -->
```bash
"$MYLAB_BUILD/my-opt" "$MYLAB_SRC/input.mlir"
```

输出就是第 1 节展示的正常 IR：两个函数参数成为 `%arg0`、`%arg1`，my.add 仍然存在，并由 return 使用其结果。这一条命令完成解析、验证和打印；我们没有给工具安排变换 Pass。

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

## 7. 直接观察打印往返，再触发一次类型检查

通用格式让 operand 与 result 的类型全部显式出现：

<!-- my-command: generic -->
```bash
"$MYLAB_BUILD/my-opt" "$MYLAB_SRC/input.mlir" --mlir-print-op-generic
```

实际完整输出：

<!-- my-output: generic -->
```text
"builtin.module"() ({
  "func.func"() <{function_type = (i32, i32) -> i32, sym_name = "test"}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "my.add"(%arg0, %arg1) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

重点看 my.add 那一行：两个输入、一个结果都明确列出。正常格式中的逗号、冒号等由 assemblyFormat 规定；通用格式使用 MLIR 的通用语法表示同一操作结构。

亲自做一次往返，把中间结果保存在本次构建目录：

<!-- my-command: roundtrip -->
```bash
"$MYLAB_BUILD/my-opt" "$MYLAB_SRC/input.mlir" > "$MYLAB_BUILD/custom.mlir"
"$MYLAB_BUILD/my-opt" "$MYLAB_SRC/input.mlir" --mlir-print-op-generic > "$MYLAB_BUILD/generic.mlir"
"$MYLAB_BUILD/my-opt" "$MYLAB_BUILD/generic.mlir" > "$MYLAB_BUILD/roundtrip.mlir"
diff -u "$MYLAB_BUILD/custom.mlir" "$MYLAB_BUILD/roundtrip.mlir"
```

本例中 diff 没有输出，退出状态为 0：通用打印再解析后，正常输出没有变化。

再只改变第一个输入的类型，让它成为 i64。下面通过标准输入提供一个完整模块，不需要先创建错误文件：

<!-- my-command: invalid -->
```bash
"$MYLAB_BUILD/my-opt" <<'MLIR'
module {
  func.func @bad(%a: i64, %b: i32) {
    %0 = "my.add"(%a, %b) : (i64, i32) -> i32
    return
  }
}
MLIR
```

该命令预期失败，诊断的核心内容为：

```text
'my.add' op operand #0 must be 32-bit signless integer, but got 'i64'
```

通用语法显式给出了 i64，因此能先建立这个操作，再由操作验证拒绝它。回看第 4.5 节：第 0 个 operand 的类型进入 `isSignlessInteger(32)` 检查并失败，正好产生这里的错误。这说明生成代码已经参与运行中的决定，而不只是目录里多了几个文件。

## 8. 轮到你改一个字段

任务是将第一个 operand 的 ODS 名称从 lhs 改为 left，同时更新 assemblyFormat 中的对应引用，观察生成访问器与 IR 文本是否变化。先写预测，再构建验证。

先创建你的副本；已有目录时保留它，不覆盖改动：

```bash
MYLAB_WORK="$PWD/aicompiler-labs/llvm-mlir/my-dialect/work/01-rename"
mkdir -p "$(dirname "$MYLAB_WORK")"
if [ ! -e "$MYLAB_WORK" ]; then
  cp -R "$MYLAB_SRC" "$MYLAB_WORK"
fi
```

编辑 `work/01-rename/MyOps.td` 后，给这份源码配置独立构建目录。下面的步骤与参考工程完全相同，只有源码和输出位置变化：

```bash
MYLAB_TASK_BUILD="$PWD/artifacts/builds/my-dialect-01-task"
cmake -S "$MYLAB_WORK" -B "$MYLAB_TASK_BUILD" -G Ninja \
  -DMLIR_DIR="$MYLAB_LLVM/lib/cmake/mlir" \
  -DLLVM_DIR="$MYLAB_LLVM/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang
cmake --build "$MYLAB_TASK_BUILD" --target MyIncGen -j2
rg -n 'getLeft|getLhs|getRhs' "$MYLAB_TASK_BUILD/MyOps.h.inc"
cmake --build "$MYLAB_TASK_BUILD" --target my-opt -j2
"$MYLAB_TASK_BUILD/my-opt" "$MYLAB_WORK/input.mlir"
```

完整要求在 [lab 任务单](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/tasks/01-rename.md)。副本不会覆盖参考版本，重复准备也不会覆盖已存在的改动。修改后记录几行：改了哪里、生成代码如何变化、原输入是否仍成立。把这些交给我审阅即可。

本轮能沿一个字段解释 `.td → .inc → 工具行为`，就有了继续扩展的支点。下一步会把 my.add 接到已学过的 Pattern/Pass，转换为 arith.addi。

## 可选的一键复现

理解分步过程后，可以使用下面的快捷入口重跑参考工程：

```bash
python3 aicompiler-labs/llvm-mlir/my-dialect/observe.py
```

它将配置、生成/编译、打印和往返打包执行，并保存日志；本章已经逐步展示这些动作。一键脚本仍使用 `my-dialect-01`，分步命令使用 `my-dialect-01-steps`，两者读取相同源码。

## 版本、复现与网页链接

本篇对应 `stages/01-minimal/`；后续功能增加到新的阶段目录，避免旧教程突然指向更复杂的实现。固定 LLVM `llvmorg-20.1.8`，完整构建方法在 [lab README](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/my-dialect/README.md)。

本篇站内链接由 blog 构建，源码链接指向独立的 aicompiler 仓库。`artifacts/...` 是本地复现路径，`.inc` 生成物不要求上传；正文已给出关键观察结果。两边分别提交推送后，对应 GitHub 链接才会存在。main 链接会随分支更新；正式固定版本可改用已发布 commit 的永久链接。

作者验证入口为 `aicompiler-labs/llvm-mlir/docs/validate_my_dialect.py`，检查源码对应、打印往返、诊断及字段改名的影响；不将这些结果记为学习者已完成任务。

机制参考：[操作定义](../../compiler/ir_definition/op_definition)、[解析与打印](../../compiler/ir_definition/assembly_format)。完整字段规范查 [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)，生成与链接关系查 [Creating a Dialect](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)；具体 API 以本地固定版本生成文件为准。
