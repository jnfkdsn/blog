---
order: 2
title: 对象生命周期与内存布局
updated: 2026-09-09
tags: [cpp, lifetime, memory, alignment]
---

# 对象生命周期与内存布局

前置：[类型、引用与类](/notes/cpp/basics)。本篇讨论对象与存储；线程间可见性见 [并发内存模型](/notes/cpp/memory_model)。

## 存储存在与对象存活

存储是容纳字节的空间，对象还有类型、初始化和生命周期。一般类对象在初始化完成后进入正常生命周期，在析构开始时结束生命周期；构造和析构期间有专门规则。不能因为地址还能打印或内存尚未覆盖，就认为对象仍能访问。

| 存储期 | 常见例子 | 管理方式 |
| --- | --- | --- |
| 自动 | 普通局部对象 | 离开作用域时销毁，常以栈实现 |
| 静态 | 全局变量、局部 static | 通常持续到程序结束；注意跨翻译单元初始化顺序 |
| 线程 | `thread_local` | 每个线程独立实例 |
| 动态 | `new`、容器内部存储 | 由所有者决定释放时间，常以堆实现 |

标准 C++ 局部内建数组的长度需是编译期常量；运行时长度优先使用 `std::vector<T>`。局部 vector 对象本身是自动存储期，它管理的元素通常在动态存储中，因此“局部变量都在栈上”不足以描述实际布局。

`new T` 分配并初始化对象，`delete` 销毁并释放；`new T[n]` 对应 `delete[]`。`malloc/free` 管理原始存储，不调用类的构造/析构函数。它们不能混用。placement new、手动生命周期管理留到 allocator 实现时再深入。

## 悬空、越界和失效

```cpp
#include <vector>

void safe_growth() {
    std::vector<float> data(4, 1.0f);
    data.reserve(1024);           // 可能重分配，应在获取借用指针之前做
    float* p = data.data();
    data.push_back(2.0f);         // 本例容量足够，已有元素的指针仍有效
    p[0] = 3.0f;
}
```

如果先保存指针再触发重分配，指针会悬空。`reserve` 也不能无限保证稳定性：超过 capacity 的增长仍会重分配，擦除等操作有各自的失效规则。

常见未定义行为包括：解引用空指针、越界、use-after-free、有符号整数溢出、访问不满足要求的对齐地址、data race。未定义行为不是“保证崩溃”，优化后的程序可能表现不同。

绑定临时对象到某些局部 const 引用可以延长临时对象生命周期，但不是通用机制：函数返回局部对象的引用仍然悬空，非拥有视图也不会自动保活底层数据。

## 大小计算也需要正确性

分配 tensor 时检查维度、元素数与字节数的每次乘法，避免先溢出再检查。

```cpp
#include <cstddef>
#include <limits>
#include <stdexcept>

std::size_t checked_bytes(std::size_t count, std::size_t item_size) {
    if (item_size != 0 &&
        count > std::numeric_limits<std::size_t>::max() / item_size) {
        throw std::overflow_error("buffer size overflow");
    }
    return count * item_size;
}
```

无符号整数回绕有定义，但回绕后的分配大小通常不符合程序意图，随后写入会越界。此类逻辑问题不能只依赖 Sanitizer 自动发现。

## 对齐、padding 与数据布局

`sizeof(T)` 包含对象的 padding，`alignof(T)` 表示类型对齐要求，`alignas(N)` 可以指定更严格的有效对齐。不要假设结构体大小等于成员大小之和，也不要依赖未经验证的 ABI 布局写文件或跨语言传输。

```cpp
struct alignas(16) Vec4 {
    float x, y, z, w;
};
static_assert(alignof(Vec4) >= 16);
```

强制把任意地址转为 `Vec4*` 不会让它自动对齐，也不会自动建立一个可访问的 `Vec4` 对象。对齐、类型访问规则、生命周期和边界都要满足。读取对象表示可用字符/字节类型；同尺寸、可平凡复制类型间的位表示转换可研究 C++20 `std::bit_cast`，不要用不合法的指针解引用替代。

AoS 将一个对象的各字段相邻存放，SoA 将同一字段的多个元素连续存放。选择由访问模式决定：批量只读某字段时，SoA 可能减少无用搬运；总是一起访问多个字段时，AoS 也可能合适。用实际访问和测量判断，不把某种布局当成固定答案。

## 缓存、分配与 false sharing

连续遍历通常比追逐分散指针更有利于缓存。性能分析除算法复杂度外，还要看分配次数、访问局部性和工作集大小。

多个 CPU 线程修改同一缓存行上的不同对象，可能产生 false sharing：语言层面可以没有 data race，但缓存一致性流量仍使性能下降。可尝试线程局部累加后合并、分离高频写字段，再测量；缓存行大小不能当成所有平台恒定的 64 字节。

内存池/arena 通过批量分配和统一回收降低开销，但统一回收会让所有借用同时失效。对象池复用前还要保证旧使用者已结束，这个问题在异步 CUDA buffer 中同样存在。

## 自查与练习

- 比较 `sizeof(std::vector<float>)` 与元素占用字节数，解释差异。
- 构造一次 vector 重分配后的悬空访问，在独立实验中用 ASan 定位，再修复。
- 为三维 tensor 的元素数计算逐步加入溢出检查。
- 比较连续数组和链式节点的遍历耗时，记录规模与编译优化选项。

语义参考：[C++ 对象生命周期](https://eel.is/c%2B%2Bdraft/basic.life)、[对象表示](https://eel.is/c%2B%2Bdraft/basic.types)。调试命令见 [构建与调试](/notes/cpp/build_debug)。
