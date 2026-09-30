---
order: 1.1
title: 一个函数怎样从源码进入程序
updated: 2026-09-30
---

# 一个函数怎样从源码进入程序

我们先做一件很小的事：让程序调用 `add(2, 3)`，打印 `5`。然后把同一个函数拆到不同文件，看看它为什么仍能被调用。

本篇需要你认识函数、参数和调用。编译器工具、生成代码和 MLIR 都暂时不出现。正文先解释每个变化；命令用于观察已经解释的现象，可以读完后再执行。[配套源码](https://github.com/jnfkdsn/aicompiler/tree/main/common/cpp-foundations)已准备好。

## 1. 所有代码在一起时，调用关系容易看清

`plain/single.cpp` 的完整内容如下：

<!-- cpp-source: plain/single.cpp -->
```cpp
#include <iostream>

int add(int a, int b) {
  return a + b;
}

int main() {
  std::cout << add(2, 3) << '\n';
}
```

运行进入 main，计算 `add(2, 3)`，再把返回值送给输出流，打印 `5`。`std::cout` 和 `<<` 在这里用于输出，不是本篇要研究的对象。

这里同时存在两件事。C++ 编译器处理源码时，需要知道 add 接受什么参数、返回什么类型，以及函数体怎样实现。最终程序运行时，才会用具体的 2 和 3 执行加法。

先在 workspace 根目录设定源码和输出位置，后面的命令使用同一个终端：

```bash
CPP_LAB="$PWD/aicompiler-labs/common/cpp-foundations"
CPP_OUT="$PWD/artifacts/builds/cpp-foundations"
mkdir -p "$CPP_OUT"
c++ -std=c++17 "$CPP_LAB/plain/single.cpp" -o "$CPP_OUT/single"
"$CPP_OUT/single"
```

输出：

```text
5
```

`c++` 是这里使用的 C++ 编译驱动程序，通常对应 GCC 或 Clang。它默认可以连续完成预处理、编译、汇编和链接。下一步把中间边界拆出来观察。

## 2. main 需要知道函数的接口，但不必在同一文件看到函数体

现在想让其他程序也复用 add，于是把函数体移到 `plain/add.cpp`：

<!-- cpp-source: plain/add.cpp -->
```cpp
#include "add.h"

int add(int a, int b) {
  return a + b;
}
```

main 留在另一个文件 `plain/main.cpp`：

<!-- cpp-source: plain/main.cpp -->
```cpp
#include "add.h"
#include <iostream>

int main() {
  std::cout << add(2, 3) << '\n';
}
```

两份文件都包含的 `plain/add.h` 只有一行：

<!-- cpp-source: plain/add.h -->
```cpp
int add(int a, int b);
```

这一行是**声明**：告诉编译器函数叫 add，接收两个 int，返回 int。末尾只有分号，没有函数体。add.cpp 中带 `{ return a + b; }` 的版本是**定义**，它提供实现；函数定义本身也会声明这个函数。

当编译器检查 main 中的调用时，声明已经足以让它判断参数和返回值的使用方式。它可以先生成调用者的目标代码，留下对 add 的外部引用，等链接时解决。它并不需要此时读到 `a + b` 才能编译 main。

add.cpp 也包含声明，使实现和接口在同一个翻译单元中接受检查。例如把定义的返回类型单独改成 double，就与已有声明发生冲突。

## 3. include 让声明出现在调用者面前

`#include "add.h"` 的基本作用是：预处理时，在这里处理指定头文件的内容。对当前这个没有其他指令的头文件，可以理解成把那行声明放到 include 所在位置。

因此，add.cpp 经过预处理后，关键内容就是：

<!-- cpp-output: preprocessed -->
```cpp
int add(int a, int b);
int add(int a, int b) {
  return a + b;
}
```

可以直接观察这份很短的实际结果：

```bash
c++ -std=c++17 -E -P "$CPP_LAB/plain/add.cpp" -o "$CPP_OUT/add.ii"
cat "$CPP_OUT/add.ii"
```

`-E` 停在预处理之后；`-P` 去掉用于记录原文件位置的行标记，便于阅读。这一步输出的仍然是源码文本，没有运行 add。

main.cpp 的预处理也会得到 add 的声明，同时引入 iostream 的大量声明，所以不必在本篇打印它的全部结果。

**include 没有把 add.cpp 的函数体带进 main.cpp。** 它只处理你指定的 add.h。main 为什么最终能调用另一个文件里的实现，需要下一步的链接来解释。

一个 `.cpp` 连同它包含并预处理得到的内容，组成独立编译所处理的翻译单元。两个 `.cpp` 不会因为互相位于同一个目录，就自动合并成一个翻译单元。

## 4. 两份目标文件怎样合成一个程序

分别编译两个源文件：

```bash
c++ -std=c++17 -c "$CPP_LAB/plain/main.cpp" -o "$CPP_OUT/main.o"
c++ -std=c++17 -c "$CPP_LAB/plain/add.cpp" -o "$CPP_OUT/add.o"
```

这里 `-c` 表示生成目标文件，尚不链接可执行程序。对于本例，可以这样理解得到的两份内容：

| 文件 | 已经提供什么 | 还需要什么 |
|---|---|---|
| main.o | main 的目标代码 | add 的实现，以及输出所需的库支持 |
| add.o | add 的目标代码 | 本例没有额外的业务函数依赖 |

链接时把它们一起交给驱动程序：

```bash
c++ "$CPP_OUT/main.o" "$CPP_OUT/add.o" -o "$CPP_OUT/demo"
"$CPP_OUT/demo"
```

仍然输出 `5`。链接器解析目标文件之间的符号引用，并进行必要的地址修正等处理；驱动程序还安排标准库等链接输入。可执行程序现在同时具备调用者和被调用者的实现。

如果故意只交 main.o：

```bash
c++ "$CPP_OUT/main.o" -o "$CPP_OUT/missing"
```

这个命令预期失败。当前 Linux 工具链报告包含 `undefined reference to` 和 `add(int, int)` 的诊断：声明曾帮助 main.cpp 通过编译，但没有提供可供链接的实现。这与找不到 add.h 是不同阶段的问题。

它也说明：知道“存在这样一个函数”和拥有“这个函数的实现”是两件事。

## 5. 库与 CMake 放回这个已经理解的过程

如果想把 add 的实现交给多个程序使用，可以先把 add.o 归档到静态库中：

```bash
ar rcs "$CPP_OUT/libadd.a" "$CPP_OUT/add.o"
c++ "$CPP_OUT/main.o" "$CPP_OUT/libadd.a" -o "$CPP_OUT/demo-library"
"$CPP_OUT/demo-library"
```

输出还是 `5`。本例中，链接器从库里取出满足 add 引用所需的目标文件。静态库本身不因为被归档就开始执行，也不是另一个需要启动的进程。

当文件很多时，手动维护“先编译哪些文件，再链接哪些库”容易出错。CMake 用来描述这些目标和依赖，再生成 Ninja 等工具能够执行的构建规则。因此以后看到 `add_library`、`add_executable`、`target_link_libraries`，可以先把它们放回刚才的编译和链接动作中理解。详细写法在[编译、链接与调试](./build_debug)中查阅。

到这里，一条完整关系已经建立：

```text
头文件提供声明
  → 调用者可以通过编译，留下对实现的引用
实现文件提供定义
  → 编译为包含实现的目标文件
链接目标文件和库
  → 得到程序
运行程序
  → main 调用 add，得到 5
```

## 用一个变化检查理解

假设只把 add.cpp 的函数体改成 `return a - b;`，而声明和 main 保持原样：新的程序会输出什么？需要重新处理哪些源码或目标文件？为什么只编辑源码、直接运行旧 demo，看不到变化？

本篇能解释这几个问题即可。下一篇[宏和生成代码怎样进入程序](./generated_cpp)继续用同一个加法函数，解释函数声明和定义由另一个程序写出来时，为什么前面的编译过程依然成立。

依据：[GCC include 的处理方式](https://gcc.gnu.org/onlinedocs/cpp/Include-Operation.html)、[编译阶段与选项](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html)。配套例子按 C++17 在本地编译验证；诊断路径、空白和工具链前缀可能不同。
