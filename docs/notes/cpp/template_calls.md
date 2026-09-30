---
order: 1.3
title: 读懂带类型参数的函数调用
updated: 2026-09-30
---

# 读懂带类型参数的函数调用

读 C++ 框架时经常遇到 `someFunction<SomeType>(...)`。本篇先用一个打印名字的程序解释：尖括号里的类型和圆括号里的值，分别给了函数什么信息。它与上一篇的外部代码生成是不同机制。

## 1. 先看 AddTag 是什么

```cpp
struct AddTag {
  static const char *name() { return "add"; }
};
```

这是一个类类型，名为 AddTag；struct 的成员默认公开。这里没有保存数据的成员，只有一个返回名字的函数。`static` 使它成为静态成员函数，可以通过 `AddTag::name()` 调用，不必先创建 AddTag 对象。

`const char *` 在这个例子中指向字符串字面量。可以先把这个返回值理解为供输出的名字，不需要先展开字符串库。

类的定义让编译器知道这种类型具备什么成员。仅仅看到这段类定义，并不会打印 add；必须有调用它并输出结果的代码。

## 2. 用类型参数选定要查询的名字

完整程序如下：

<!-- cpp-source: templates/main.cpp -->
```cpp
#include <iostream>

struct AddTag {
  static const char *name() { return "add"; }
};

template <typename T>
void announce(int count) {
  std::cout << T::name() << ": " << count << '\n';
}

int main() {
  announce<AddTag>(2);
  announce<AddTag>(3);
}
```

`template <typename T>` 声明 T 是一个类型参数。编译器处理 `announce<AddTag>(2)` 时，知道这一版本的 T 是 AddTag，所以函数体中的 `T::name()` 指向 `AddTag::name()`。

你可以把这个具体版本理解成下面的普通函数；这是解释用途的等价示意，不是声称编译器会生成这份文本文件：

```cpp
void announce_for_AddTag(int count) {
  std::cout << AddTag::name() << ": " << count << '\n';
}
```

程序运行时，第一次调用的 count 是 2，第二次是 3，因此输出：

<!-- cpp-output: template -->
```text
add: 2
add: 3
```

两次使用同一个模板特化，传入不同普通参数。**模板参数在编译时确定，不表示函数体一定在编译时执行。** 这里的输出动作在程序运行时发生。

复现命令从 workspace 根目录运行：

```bash
mkdir -p artifacts/builds/cpp-foundations
c++ -std=c++17 aicompiler-labs/common/cpp-foundations/templates/main.cpp \
  -o artifacts/builds/cpp-foundations/template-demo
artifacts/builds/cpp-foundations/template-demo
```

## 3. 类型参数没有偷偷创建对象

在 `announce<AddTag>(2)` 里：

| 位置 | 含义 |
|---|---|
| `AddTag` | 告诉编译器用哪个类型实例化函数模板 |
| `2` | 本次调用传给 count 的普通参数 |

这句代码没有创建一个 AddTag 对象交给 announce。能够调用 name，是因为本例把它定义成静态成员。

如果把类型参数换成 int，编译就会失败：实例化这个函数体需要 `int::name()`，而 int 没有这种成员。模板实现对类型有具体要求，并不是任意类型都能代入。

## 4. 回到框架中的写法

以后看到 `addOperations<AddOp>()` 时，可以先读出语法：调用一个函数模板，把 AddOp 作为类型参数，没有普通实参。它内部能利用 AddOp 提供的静态信息做登记，是函数实现规定的行为；尖括号本身不具备“注册”的特殊含义。

同样，`create<AddOp>(...)` 怎样构造操作，需要看 create 的约定。名字不同的模板函数可以完成完全不同的工作。不能只因为都写了 `<AddOp>`，就把它们理解成同一操作。

本例 announce 只输出名字，没有实现注册表、查找或回调，也不是 MLIR 注册代码的缩小实现。它只用来解决类型参数如何参与函数逻辑这个 C++ 阅读障碍。

更复杂的参数包、继承、CRTP 与模板约束，在遇到具体需要时再深入[模板与编译期编程](./templates)。本篇的目标是能区分类型参数、普通参数、类型定义和对象构造，并认识到模板函数也可以在运行时执行。

试着解释：如果把 AddTag::name 的返回值改为 `"addition"`，为什么需要重新构建程序？如果只把第一次调用的 2 改为 8，改变的是类型参数还是普通参数？
