---
order: 6
title: 编译、链接与调试
updated: 2026-09-09
tags: [cpp, cmake, linking, sanitizer, pybind11]
---

# 编译、链接与调试

路线：[C++ 工程基础](/notes/cpp/)。本篇讲通用 CPU 工程；CUDA 扩展的 `.cu`、架构参数和 Torch 依赖见 [CUDA CMake 实践](/notes/cuda/cmake)。

## 从源文件到程序

```text
头文件展开与预处理 → 每个翻译单元编译 → 目标文件 → 链接 → 可执行文件/共享库
```

头文件通常提供声明，`.cpp` 提供非内联定义。每个源文件连同它包含的内容构成翻译单元。命名空间避免名字冲突，不会自动减少 include 依赖。

```cpp
// tensor.hpp
#pragma once
namespace infra {
class Tensor;
void inspect(const Tensor& tensor);
}
```

前向声明可用于声明指针、引用和部分函数接口；按值保存对象、访问成员等需要完整定义。`unique_ptr<Impl>` 可支持 PImpl，但用默认 deleter 执行删除的地方需要 Impl 完整，常把拥有者的析构定义放到 `.cpp`。

ODR（单一定义规则）约束实体的定义。普通外部链接函数不能在多个翻译单元重复定义；inline 函数、模板等允许满足规则的多处定义。inline 不保证编译器一定内联调用。头文件保护只防止同一翻译单元重复包含，不能解决跨翻译单元的重复定义。

模板定义通常放头文件；有限类型可用显式实例化管理。constexpr 函数的定义要在需要常量求值的地方可达，也通常写在头文件。

## 最小 CMake 工程

下面的三个文件可以独立构建，分别按注释保存。

```cpp
// sum.hpp
#pragma once
#include <vector>
float sum(const std::vector<float>& values);
```

```cpp
// sum.cpp
#include "sum.hpp"
#include <numeric>
float sum(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0f);
}
```

```cpp
// main.cpp
#include "sum.hpp"
#include <iostream>
int main() {
    std::cout << sum({1.0f, 2.0f, 3.0f}) << '\n';
}
```

```cmake
cmake_minimum_required(VERSION 3.16)
project(cpp_notes LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(tensor_core STATIC sum.cpp)
target_compile_features(tensor_core PUBLIC cxx_std_17)
target_include_directories(tensor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(example main.cpp)
target_link_libraries(example PRIVATE tensor_core)
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
./build/example
```

输出应为 6。`CMAKE_BUILD_TYPE` 适用于单配置生成器；多配置生成器使用构建时的 `--config Debug`。`compile_commands.json` 可供 clangd 等工具读取实际编译参数。

target 的 PRIVATE 配置供自身使用，INTERFACE 供消费者使用，PUBLIC 同时作用于两者。例中库头文件暴露 vector，示范把 include 路径与语言要求传播给调用方。线程项目使用 `find_package(Threads REQUIRED)`，再链接 `Threads::Threads`。[CMake target_link_libraries 文档](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)

## 静态库、动态库与 ABI

静态库是目标文件的归档，链接时抽取需要的对象；共享库在运行时加载，除链接成功还要保证加载器能找到它及依赖。Linux 中注意 rpath、库搜索路径和导出符号。

ABI 涉及名称修饰、调用约定、对象布局、标准库和编译选项等。`extern "C"` 提供 C 语言链接，不会让 std::string 等 C++ 类型自动变成稳定的跨语言 ABI。跨边界应使用明确的类型与所有权协议。

| 现象 | 优先检查 |
| --- | --- |
| 找不到头文件 | include 路径、依赖安装和 target 传播 |
| undefined reference | 源文件是否加入 target、依赖链接、签名、模板定义/实例化 |
| multiple definition | 头文件中非 inline 定义、重复编入目标 |
| import 时 undefined symbol | 共享库依赖、导出符号、加载版本与 ABI |

## 调试与 Sanitizer

CPU 单文件实验可以使用以下命令。`main.cpp` 替换为实际入口；GCC/Clang 的支持情况以平台为准。

```bash
g++ -std=c++17 -g -O1 -Wall -Wextra -Wpedantic -fno-omit-frame-pointer -fsanitize=address,undefined main.cpp -o demo_asan
./demo_asan
g++ -std=c++17 -g -O1 -pthread -fsanitize=thread main.cpp -o demo_tsan
./demo_tsan
```

ASan 用于越界、释放后访问等；UBSan 检测部分未定义行为；TSan 检测执行路径中的 CPU data race。TSan 与 ASan 使用独立构建，不组合为一个程序。工具未报错不代表所有路径都正确；CPU Sanitizer 也不能代替 CUDA Compute Sanitizer。[Clang TSan 文档](https://clang.llvm.org/docs/ThreadSanitizer.html)

使用 gdb/lldb 时先拿到调用栈，定位首个无效地址来源，再检查其分配、最后一次合法使用和释放位置。Release/RelWithDebInfo 用于性能分析；Debug 的时序和内联情况不能代表线上性能。

benchmark 应固定输入规模、预热、重复测量，避免结果被优化掉，记录编译选项和分配次数。CPU 可用 perf 等采样分析；GPU 应使用适当 event 或同步测实际执行时间，不能只计 launch 返回耗时。

## Python 绑定与 GIL

pybind11 暴露 C++ 函数时要明确参数类型、返回值所有权和异常转换。对传统带 GIL 的 CPython，pybind11 不会自动释放 GIL；长时间纯 C++ 计算可在明确不访问 Python 对象的范围内使用 `gil_scoped_release`，回调 Python 前要持有 GIL。free-threaded 构建有额外要求，不能把 GIL 当作所有环境中的全局业务锁。[pybind11 GIL 文档](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)

借用 NumPy/Tensor 数据时确认 dtype、shape、stride、device、可写性与生命周期；零拷贝不等于没有所有权责任。异步工作不能只保存临时 Python 对象提供的裸地址。Torch extension 还要检查当前 stream 等框架约定，见 [CUDA 导读](/notes/cuda/cpp)。

## 自查与练习

- 构建上面的多文件工程，再故意遗漏 sum.cpp，区分编译错误与链接错误。
- 用调试器和 ASan 分别定位一个独立的悬空访问实验。
- 将有界队列的多线程实验单独构建为 TSan 版本。
- 解释 shared library 链接成功但 Python import 失败的可能原因。
