---
order: 1.2
title: 宏和生成代码怎样进入程序
updated: 2026-09-30
---

# 宏和生成代码怎样进入程序

上一篇已经用声明、定义和链接解释了 add 函数怎样被调用。现在仍然希望 `add(2, 3)` 得到 `5`，只改变一件事：**部分 C++ 源码由一个小程序替我们写出来。**

这能帮助理解大型工程为什么会出现“描述文件、生成器、生成文件、手写代码”。本篇先用普通加法说明过程，最后再连接 MLIR。还不熟悉声明与链接时，先读[一个函数怎样从源码进入程序](./source_to_program)。

## 1. 生成代码首先只是写出了文本

假设有一个文件 function.txt，里面只有：

```text
add
```

我们编写的生成器读取这个名字，写出下面的 C++ 函数：

```cpp
int add(int a, int b) {
  return a + b;
}
```

在这个时刻，没有人调用 add，也没有发生 `2 + 3`。生成器只是把一段字符写入另一个文件。之后 C++ 编译器读取这些字符时，并不因为它们是程序生成的就赋予特殊含义；它们仍需符合普通 C++ 规则。

如果 function.txt 中的名字改成 sum，这个生成器就会写出 `int sum(...)`。**“输入描述发生变化 → 输出源码发生变化”就是本例代码生成的全部意思。** 加法的含义来自我们写在生成器里的规则，它不会自动猜测名字代表什么算法。

## 2. 先理解宏怎样选择一段内容

为了同时保存函数声明和定义，我们准备让生成器把两部分放进同一个文件。在解释这个文件之前，先看两条预处理指令：

```cpp
#define SHOW_DECLARATION

#ifdef SHOW_DECLARATION
int add(int a, int b);
#endif
```

第一行定义一个宏名，这里不带替换内容。`#ifdef` 检查预处理到这里时，这个宏是否已定义。因为已经定义，条件区域里的声明会保留。经过预处理，以上内容剩下：

```cpp
int add(int a, int b);
```

如果删除第一行，且命令行等其他地方也没有定义该宏，条件不成立，这条声明就不会保留下来。

这个过程发生在 C++ 编译之前。它不是运行程序时执行一个 if，也没有创建名为 SHOW_DECLARATION 的 C++ 变量。`#undef SHOW_DECLARATION` 则取消这个宏定义，影响后续的预处理。

## 3. 用一个生成文件保存两种片段

本例生成器实际写出的 generated.inc 如下：

<!-- cpp-output: generated -->
```cpp
#ifdef SELECT_DECLARATIONS
#undef SELECT_DECLARATIONS
int add(int a, int b);
#endif

#ifdef SELECT_DEFINITIONS
#undef SELECT_DEFINITIONS
int add(int a, int b) {
  return a + b;
}
#endif
```

两个区域分别保存声明和定义。进入某个区域后取消选择宏，避免该选择意外影响后面的 include；这不会撤销已经选中的区域。

`.inc` 是工程用来提示“供包含的片段”的文件名后缀。`#include` 不要求目标文件一定以 `.h` 结尾。编译器并不会因为文件名是 `.inc` 就自动包含它，仍然需要手写 include。

现在 `generated/add.h` 的完整内容是：

<!-- cpp-source: generated/add.h -->
```cpp
#ifndef CPP_LESSON_ADD_H
#define CPP_LESSON_ADD_H
#define SELECT_DECLARATIONS
#include "generated.inc"
#endif
```

先看里面两行：定义 SELECT_DECLARATIONS，再包含 generated.inc。它们触发刚才文件里的第一个区域，所以在此处得到 add 的声明；第二个区域没有对应选择宏，不会留下定义。

外面的 `#ifndef` 是“尚未定义时才保留”。第一次包含 add.h 时，它定义 CPP_LESSON_ADD_H；在同一个翻译单元再次包含时，就跳过整个头文件主体。这叫头文件保护。不同 `.cpp` 分别预处理时，不靠这个宏共享状态。

实现文件 `generated/add.cpp` 再选择定义：

<!-- cpp-source: generated/add.cpp -->
```cpp
#include "add.h"

#define SELECT_DEFINITIONS
#include "generated.inc"
```

沿着处理顺序走一遍：

| 处理位置 | 保留下来的 C++ 内容 |
|---|---|
| include add.h | add 的声明 |
| 定义 SELECT_DEFINITIONS | 暂时没有新增 C++ 内容，只改变预处理条件 |
| 再次 include generated.inc | add 的定义 |

因此，这份 add.cpp 预处理后就是：

<!-- cpp-output: preprocessed -->
```cpp
int add(int a, int b);
int add(int a, int b) {
  return a + b;
}
```

它与上一篇手写实现的预处理结果一致。这个例子已经足以说明为什么有效：C++ 编译器最终接收到熟悉的函数声明和函数定义。

选择片段的 generated.inc 故意没有在最外层设置“整个文件只包含一次”的保护，否则第二次包含无法选出另一部分。头文件保护与片段选择宏解决的是不同问题。

## 4. 生成器本身也可以看清

以下是本例完整的 Python 生成器。它只支持 add 和 sum 两个名字，目的是让“输入怎样变成输出”能够直接看见：

<!-- cpp-source: generated/generate.py -->
```python
"""Write C++ text for this lesson's one integer-addition function."""
from pathlib import Path
import sys

name = Path(sys.argv[1]).read_text().strip()
if name not in {"add", "sum"}:
    raise SystemExit("This example accepts only add or sum.")
output = Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("""#ifdef SELECT_DECLARATIONS
#undef SELECT_DECLARATIONS
int @NAME@(int a, int b);
#endif

#ifdef SELECT_DEFINITIONS
#undef SELECT_DEFINITIONS
int @NAME@(int a, int b) {
  return a + b;
}
#endif
""".replace("@NAME@", name))
```

它读取名字，用这个名字替换模板中的 `@NAME@`，然后写文件。这是生成器自己的执行过程。它没有调用正在生成的 C++ 函数。

可以在 workspace 根目录复现；这些命令只是观察上述过程，不要求先运行才能理解正文：

```bash
CPP_LAB="$PWD/aicompiler-labs/common/cpp-foundations"
CPP_OUT="$PWD/artifacts/builds/cpp-foundations"
mkdir -p "$CPP_OUT"
python3 "$CPP_LAB/generated/generate.py" \
  "$CPP_LAB/generated/function.txt" "$CPP_OUT/generated.inc"
cat "$CPP_OUT/generated.inc"
c++ -std=c++17 -I "$CPP_OUT" -E -P "$CPP_LAB/generated/add.cpp" \
  -o "$CPP_OUT/generated-add.ii"
cat "$CPP_OUT/generated-add.ii"
```

两次 cat 分别显示本篇已经列出的生成文件和预处理结果。`-I` 为 include 添加搜索目录，让编译器找到位于输出目录的 generated.inc。

调用者 `generated/main.cpp` 与上一篇一样：

<!-- cpp-source: generated/main.cpp -->
```cpp
#include "add.h"
#include <iostream>

int main() {
  std::cout << add(2, 3) << '\n';
}
```

编译、链接并运行：

```bash
c++ -std=c++17 -I "$CPP_OUT" "$CPP_LAB/generated/main.cpp" \
  "$CPP_LAB/generated/add.cpp" -o "$CPP_OUT/generated-demo"
"$CPP_OUT/generated-demo"
```

输出仍是 `5`。本次用一条 c++ 命令完成两个翻译单元的编译和链接，基本关系与上一篇完全一致。

## 5. 回到“把描述生成 C++”这句话

现在可以把三件事分开：

```text
运行生成器：function.txt → generated.inc（源码文本）
构建程序：手写源码包含 generated.inc → 目标代码 → 可执行程序
运行程序：main → 调用 add(2, 3) → 输出 5
```

生成器退出后，源码文件仍保存在磁盘。构建结束后，相关代码已经进入程序。最终程序不需要重新读取 function.txt 或运行 generate.py 才能调用 add。

MLIR 工程中的 `.td → mlir-tblgen → .inc` 属于第一种过程。区别在于输入遵循 TableGen/ODS 的描述规则，生成器按所选后端生成方言类、操作类及相关方法，远比本例丰富。它可以输出类声明，也可以输出类的成员函数实现；这些输出随后参与普通 C++ 编译。生成类的文本，并不等于在运行中创建了一个类实例。

本例生成普通整数加法的函数体，是我们在生成器中明确写入了 `a + b`。不能由此推断 MLIR 看到名为 add 的 Op 就会自动生成机器加法：ODS 描述操作结构、约束等，操作的计算含义需要开发者约定，并由相应变换或执行机制实现。

在 MLIR 中看到 `GET_OP_CLASSES` 与 include 的组合时，现在可以先判断：它是在预处理时选择生成文件中的某部分。具体选中声明还是定义，还取决于包含的是哪份生成文件及它内部的条件区域。

## 本篇结束时，应该能解释的变化

- 如果生成文件还不存在，为什么编译会在 include 处失败，而不是到程序运行时才失败？
- 如果修改输入描述，但没有重新生成和编译，为什么旧程序的行为不变？
- 如果第二次 include 没有定义 SELECT_DEFINITIONS，最终会缺少什么？

本篇解释的是代码怎样进入程序。模板调用使用类型参数，是另一个问题，接着读[读懂带类型参数的函数调用](./template_calls)。无需在这里追踪 mlir-tblgen 的内部实现。

依据：[GCC 条件预处理](https://gcc.gnu.org/onlinedocs/cpp/Ifdef.html)、[include](https://gcc.gnu.org/onlinedocs/cpp/Include-Operation.html)。[配套源码](https://github.com/jnfkdsn/aicompiler/tree/main/common/cpp-foundations)与正文对应，生成物留在本地 artifacts。
