---
order: 5
title: 模板与编译期编程
updated: 2026-09-09
tags: [cpp, template, constexpr, traits]
---

# 模板与编译期编程

前置：[基础语法](/notes/cpp/basics)、[移动语义](/notes/cpp/ownership)。目标是读懂 CUDA/CUTLASS 和编译器代码中的类型组合，不以实现复杂元编程库为起点。

## 函数模板、类模板和推导

```cpp
template <typename T>
T add(T a, T b) { return a + b; }

template <typename T, int N>
struct Tile {
    static_assert(N > 0, "tile size must be positive");
    T data[N]{};
};
```

`add(1, 2)` 推导 T 为 int；`add(1, 2.0)` 对同一个 T 推导冲突，不能期待编译器自动挑一个共同类型。`add<double>(1, 2.0)` 显式指定后可以转换参数。

T 是类型模板参数，N 是非类型模板参数。`Tile<float, 16>` 与 `Tile<float, 32>` 是不同类型。运行时变量不能直接作为这里的 N；动态配置通常由 host 代码分派到有限个已编译版本。

## constexpr 与 if constexpr

```cpp
#include <type_traits>

constexpr int square(int x) { return x * x; }
static_assert(square(4) == 16);

template <typename T>
constexpr T relu(T x) {
    static_assert(std::is_arithmetic_v<T>, "arithmetic type required");
    if constexpr (std::is_unsigned_v<T>) {
        return x;
    } else {
        return x > T{0} ? x : T{0};
    }
}
```

`constexpr` 变量的初始化要满足常量表达式要求；constexpr 函数可以参与编译期计算，也可以在运行时执行，例如参数来自输入时。常量表达式上下文要求编译期求值，而不是每次调用都强制编译期执行。C++20 `consteval` 用于要求立即求值的函数。[C++ constexpr 规则](https://eel.is/c%2B%2Bdraft/dcl.constexpr)

`if constexpr` 在模板实例化时选择分支，丢弃的依赖分支可以不实例化，但代码仍需能解析，非依赖错误也不能随意隐藏。普通 if 的两个分支都需要满足对应的编译要求。

## 特化、偏特化和重载

```cpp
template <typename T>
struct TypeInfo { static constexpr bool pointer = false; };

template <typename T>
struct TypeInfo<T*> { static constexpr bool pointer = true; }; // 偏特化

template <>
struct TypeInfo<float> { static constexpr bool pointer = false; }; // 全特化
```

类模板可以偏特化，函数模板不能偏特化，通常用重载或 if constexpr 表达差异。实际项目优先使用标准 traits，如 `is_same_v`、`is_pointer_v`、`remove_reference_t` 和 `remove_cv_t`，避免重复实现。

`std::is_same_v<const float, float>` 为 false；检查时是否去掉 const/reference 取决于接口意图，不能一律抹去限定信息。

## 依赖名：typename 和 template

```cpp
template <typename Container>
typename Container::value_type first(const Container& c) {
    return c.front();  // 前置条件：非空
}
```

`Container::value_type` 依赖模板参数，`typename` 告诉解析器它是类型。在依赖对象上调用成员模板时，可能需要 `obj.template get<T>()` 消除语法歧义。理解这两种写法后再读长模板错误，先定位实际实例化类型与第一处失败约束。

## 参数包与完美转发

```cpp
#include <memory>
#include <utility>

template <typename T, typename... Args>
std::unique_ptr<T> make_owned(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}
```

Args 是类型包，`...` 展开参数。这里的 `Args&&` 因参与推导而是转发引用，可以接收左值或右值；不是所有 `T&&` 都是转发引用。引用折叠中，只要组合包含左值引用就得到左值引用，否则得到右值引用。

具名 args 在函数体内是左值，`std::forward<Args>` 按推导信息恢复调用者的值类别。若全部换成 std::move，就可能意外搬走调用者传入的左值资源。不要重复消费一个已经转发给接管所有权接口的参数。

## SFINAE、concepts 与编译成本

旧代码常用 `enable_if`：某些替换失败会让候选退出重载集合，而不是立即报整个程序错误，这称为 SFINAE；函数体内部任意错误并不都属于这种情况。

C++20 concepts 可更直接表达约束：

```cpp
#include <concepts>

template <std::integral T>
constexpr T twice(T x) { return x + x; }
```

约束限定接口能接受的类型，不自动保证运行时数值不溢出。C++17 项目先会读 traits、enable_if 即可，按项目标准决定是否使用 concepts。

模板通常把定义放在头文件，让实例化处可见；也可以在实现文件显式实例化有限类型。模板参数组合越多，编译时间和二进制体积可能越大。CUDA 中不要把每个运行时维度都模板化，优先选择真正改变代码结构的 dtype、tile、layout 等配置。

## 自查与练习

- 实现一个支持 float/double 的函数，比较隐式推导与显式模板参数。
- 将三种固定 tile 配置封装成模板，由运行时 switch 分派。
- 解释 if constexpr 与普通 if 的编译区别。
- 用重载记录一次完美转发分别选择左值和右值版本的过程。
