---
order: 1
title: 类型、引用与类
updated: 2026-09-09
tags: [cpp, const, lambda]
---

# 类型、引用与类

路线：[C++ 工程基础](/notes/cpp/) → 本篇 → [对象生命周期](/notes/cpp/object_lifetime)。示例采用 C++17。

## 值、指针和引用

指针保存地址，可以为空、可以改为指向其他对象；引用在初始化时绑定对象，之后给引用赋值是在修改被引用对象。两者都不自动延长任意被指向对象的生命周期。

| 参数 | 常见意图 | 注意 |
| --- | --- | --- |
| `int n` | 传递小值 | 修改形参不影响调用方 |
| `const Tensor& x` | 借用只读对象 | 调用方保证对象存活 |
| `Tensor& out` | 借用并修改对象 | 不应绑定空对象 |
| `Tensor* out` | 可选对象或 C API 参数 | 约定空指针是否有效 |
| `std::unique_ptr<Tensor> x` | 接收所有权 | 详见所有权篇 |

```cpp
int x = 1;
int& ref = x;
int* ptr = &x;
ref = 2;               // x 变为 2
*ptr = 3;              // x 变为 3
```

声明中的 `&` 表示引用，表达式中的一元 `&` 取地址。`nullptr` 表达空指针；空指针不能解引用。`void*` 可以接收对象指针转换，但没有元素类型，不能直接解引用，也没有标准 C++ 指针算术。

`T*` 可以转换为 `void*`，**`T**` 不能隐式转换为 `void**`**。二级指针接口会写入调用方的指针，不能把一级转换规则套用过来。见 [C++ 指针转换规则](https://eel.is/c%2B%2Bdraft/conv.ptr)。

## const 约束哪个对象

```cpp
int x = 1;
const int* a = &x;        // 不能通过 a 修改 x；a 本身可以改指向
int* const b = &x;        // b 不能改指向；可以通过 b 修改 x
const int* const c = &x;  // 两者都受限
const int& r = x;         // 不能通过 r 修改 x
```

只读访问路径不代表底层对象永远不变，例如这里仍然可以执行 `x = 2`。`const` 也不是锁，不能让与其他写操作冲突的读取自动变得线程安全。

`const` 成员函数不通过 `this` 修改普通成员，可以在 const 对象上调用；它不保证深层指针所指对象也不可修改。

## 初始化、auto 与类型转换

局部基本类型没有初始化时可能含不确定值，优先写 `int count = 0;`。花括号初始化可以拒绝一些窄化转换，但需要注意 `std::vector<int>{4, 1}` 是两个元素，而 `(4, 1)` 是四个 1。

```cpp
const int n = 8;
auto copy = n;            // int，按值推导丢掉顶层 const
const auto& alias = n;    // const int&，借用
```

`auto` 不意味着自动引用。遍历大对象时使用 `const auto&` 避免复制；需要修改元素时使用 `auto&`。

| 转换 | 适用范围 |
| --- | --- |
| `static_cast<T>` | 明确的数值转换、合法的静态类型转换；不会替你检查整数范围 |
| `dynamic_cast<T*>` | 多态层次中的运行时类型检查，失败返回空指针；需要 RTTI 支持 |
| `reinterpret_cast<T*>` | 底层表示转换；不保证解引用满足对齐、生命周期和别名规则 |
| `const_cast<T*>` | 改变 cv 限定；修改原本定义为 const 的对象仍是未定义行为 |

大小常用 `std::size_t`，固定宽度接口可用 `<cstdint>` 的整数类型。先验证非负和范围再把有符号尺寸转成无符号数，避免 `-1` 变成巨大长度。

## 类、构造与接口

`struct` 默认成员和继承访问权限为 public，`class` 默认为 private。初始化列表直接初始化成员；引用成员和没有默认构造函数的成员等需要在这里初始化。实际初始化顺序按成员声明顺序，不按列表书写顺序。

```cpp
#include <cstddef>

class TensorShape {
public:
    explicit TensorShape(std::size_t batch) : batch_(batch) {}
    std::size_t batch() const noexcept { return batch_; }
    TensorShape& set_batch(std::size_t batch) {
        batch_ = batch;
        return *this;
    }
private:
    std::size_t batch_ = 0;
};
```

`explicit` 限制隐式构造转换；`this` 指向当前对象，`return *this` 可以实现链式调用。构造函数应建立对象的不变量，例如维度合法，而不只是把参数存下来。

运行时可替换的实现可以使用虚函数接口：

```cpp
struct Executor {
    virtual ~Executor() = default;
    virtual void run() = 0;
};
struct CpuExecutor : Executor {
    void run() override {}
};
```

通过基类指针删除派生对象时，这样的接口需要虚析构函数。`override` 帮助检查签名；按值传递可复制的基类对象会切掉派生部分，称为对象切片。编译期已确定的实现也可以通过模板表达，见 [模板篇](/notes/cpp/templates)。

## lambda 捕获与生命周期

语法为 `[捕获](参数) { 函数体 }`。`[=]`、`[&]` 为需要隐式捕获的局部实体提供默认方式，并非无条件捕获所有外部变量。

```cpp
#include <memory>
#include <utility>

auto make_task() {
    auto data = std::make_unique<int>(42);
    return [p = std::move(data)] { return *p; };
}
```

这里把所有权移进闭包，函数返回后闭包仍可使用数据。若改成捕获局部变量的引用，返回后就会悬空。捕获 `this` 只保存指针，不延长当前对象的生命；成员函数中的 `[=]` 也不等于复制整个对象。C++17 的 `[*this]` 才显式复制对象，且仍要分析内部指针的所有权。

`mutable` 允许修改闭包中的按值捕获成员，不会把它们变成调用方变量的引用。捕获 `unique_ptr` 的闭包通常只能移动，不能直接存入要求可复制目标的 C++17 `std::function`。

## 自查与练习

- 为输入、输出、可选参数和所有权转移各设计一个函数签名。
- 解释为什么 `const int*` 不阻止其他路径修改同一个 `int`。
- 把返回悬空引用捕获的 lambda 改为按值捕获或移动捕获。
- 说明虚函数、模板分别在哪个阶段选择实现。
