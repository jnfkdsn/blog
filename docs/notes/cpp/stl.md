---
order: 4
title: STL、算法与非拥有视图
updated: 2026-09-09
tags: [cpp, stl, containers, views]
---

# STL、算法与非拥有视图

前置：[所有权](/notes/cpp/ownership)。日常所说的 STL 常泛指标准库中的容器、迭代器、算法等。本篇按访问模式、生命周期和成本组织。

## 常用容器怎么选

| 类型 | 常见用途 | 成本与限制 |
| --- | --- | --- |
| `array<T, N>` | 固定大小元数据 | 元素内嵌，N 在编译期确定 |
| `vector<T>` | 动态连续序列 | 随机访问 O(1)，尾部追加摊销 O(1)，中间插删 O(n) |
| `deque<T>` | 两端增长的序列 | 两端插入 O(1)，存储不保证整体连续 |
| `list<T>` | 已持有迭代器的节点插删 | 节点分配与指针追逐成本；按值寻找位置仍 O(n) |
| `unordered_map<K,V>` | 按 key 查任务或缓存项 | 查找平均 O(1)、最坏 O(n)，需要合适的 hash/equality |
| `map<K,V>` | 有序遍历、范围查找 | 查找和插入 O(log n) |
| `priority_queue<T>` | 优先级调度 | top O(1)，push/pop O(log n)，不提供任意项更新接口 |
| `string` | 拥有字符串 | 不要依赖特定实现的小字符串优化阈值 |

默认先考虑 vector，只有具体访问需求再选择更复杂的结构。复杂度不能替代测量：节点分配、缓存局部性、key 长度和锁竞争可能主导耗时。

## size、capacity 与插入

```cpp
#include <vector>

std::vector<int> make_ids() {
    std::vector<int> ids;
    ids.reserve(4);       // 预留存储，不创建元素，size 仍是 0
    ids.push_back(7);
    ids.resize(4, -1);    // 创建后续元素，size 变为 4
    return ids;
}
```

`reserve(4)` 后直接写 `ids[0]` 仍然越界。`operator[]` 不检查边界；`at()` 对越界抛异常。`emplace_back(args...)` 在容器中构造元素，但不保证比 push_back 更快，也不阻止扩容或其他元素的移动。

对 unordered_map，`operator[]` 在 key 不存在时会插入默认值；只查询用 `find()`，C++20 也可用 `contains()`。`try_emplace()` 可在缺失时原位构造值，但传参表达式仍会先求值，不是任意昂贵计算的惰性执行器。

## 迭代器和借用何时失效

| 操作 | 关键失效规则 |
| --- | --- |
| vector 重分配 | 所有元素指针、引用和迭代器失效 |
| vector 无重分配的尾部插入 | 已有元素引用和迭代器保留，旧 end 失效 |
| vector erase | 被删位置及其后的迭代器和引用失效 |
| unordered_map rehash | 迭代器失效，已有元素的指针和引用保留 |
| map/list erase | 指向被删元素的借用失效，其他元素通常保持有效 |

每类容器有独立契约，特别不要把 vector 的规则套到 deque。即使指针在容器操作后仍有效，也不等于该操作允许与读线程并发执行。

## 算法与比较器

```cpp
#include <algorithm>
#include <numeric>
#include <vector>

int positive_sum(std::vector<int> values) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](int x) { return x <= 0; }), values.end());
    std::sort(values.begin(), values.end());
    return std::accumulate(values.begin(), values.end(), 0);
}
```

示例假设总和不溢出 int。`remove_if` 先移动保留元素并返回逻辑终点，erase 才真正缩短容器。`accumulate` 的初始值决定累加类型；浮点累加用 `0.0f` 或 `0.0`，不要误用整数 `0`。

`sort` 比较器必须满足严格弱序，不能写 `a <= b`；浮点 NaN 也会影响比较前提，需要明确处理策略。`lower_bound` 要求范围按对应比较关系分区，通常先排序；找到的是插入位置，是否等于目标还需判断。

## 非拥有视图

`string_view`（C++17）保存字符范围，`span<T>`（C++20）保存连续元素范围。它们不拥有、不释放，也不延长底层数据生命。复制视图只复制描述信息。

```cpp
#include <string_view>

bool is_cuda(std::string_view name) {
    return name == "cuda";
}
```

函数内同步读取临时 string 对应的视图可以安全，但把该视图保存到以后使用可能悬空。返回指向局部 string 的 string_view 也不安全。string_view 的 `data()` 不保证在视图末尾有 NUL，不能直接当作任意 C 字符串 API 的输入。

C++20 示例：

```cpp
#include <span>

float sum(std::span<const float> values) {
    float result = 0;
    for (float x : values) result += x;
    return result;
}
```

`span<const T>` 表达元素只读；`const span<T>` 只是视图对象不可修改，元素仍可以写。span 不保证自动做越界检查，也不能描述任意带 stride 的 tensor，需要额外布局信息。

LLVM 的 `StringRef`、`ArrayRef` 同样强调借用，`SmallVector` 则拥有元素并提供内嵌容量；阅读接口时先分清这两类。[LLVM Programmer’s Manual](https://www.llvm.org/docs/ProgrammersManual.html)

## 常用值类型与回调

`optional<T>` 表达可能没有值，访问前检查；`variant<A,B>` 表达有限类型中的一种，可用 `std::visit` 分派；`pair/tuple` 表达组合值，C++17 结构化绑定便于解包。它们不替代所有错误处理：optional 的空值不携带详细错误原因。

`std::function` 对回调做类型擦除，可能分配，调用也存在间接开销；C++17 要求目标可复制。模板回调更容易内联但增加实例化。选择应由 API 需求和测量决定。

## 自查与练习

- 分别设计按 ID 查询任务、FIFO 等待队列和优先级任务队列。
- 解释为什么 reserve 后不能按 capacity 下标写元素。
- 找出一个保存临时字符串视图的悬空问题。
- 比较预留容量前后的分配次数，而不只看单次运行耗时。

容器契约参考：[vector 修改操作](https://eel.is/c%2B%2Bdraft/vector.modifiers)、[无序关联容器要求](https://eel.is/c%2B%2Bdraft/unord.req)、[string_view](https://eel.is/c%2B%2Bdraft/string.view)。
