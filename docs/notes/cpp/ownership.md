---
order: 3
title: 所有权、RAII 与移动语义
updated: 2026-09-09
tags: [cpp, raii, smart-pointer, move]
---

# 所有权、RAII 与移动语义

前置：[对象生命周期](/notes/cpp/object_lifetime)。资源包括内存、文件、锁和 CUDA stream。首先明确谁负责释放，再选择类型。

## 拥有与借用

| 类型 | 通常表达的关系 |
| --- | --- |
| `T`、`std::vector<T>` | 值或容器拥有其内容 |
| `std::unique_ptr<T>` | 独占所有权，可转移 |
| `std::shared_ptr<T>` | 共享所有权，最后一个拥有者释放对象 |
| `std::weak_ptr<T>` | 观察共享对象，不增加强引用计数 |
| `T*`、`T&`、`string_view`、`span` | 现代接口中通常表达借用，需由其他对象保活 |

裸指针本身不编码所有权；遇到 C API 必须阅读契约，不能单凭 `T*` 判断是否需要释放。优先独占所有权，只在确实需要共享生命周期时使用 shared_ptr，见 [C++ Core Guidelines R.21](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#Rr-unique)。

## RAII 和异常安全

RAII 把资源持有与对象生命周期绑定，析构时清理资源；提前返回和异常展开也会销毁已经构造完成的局部对象。如果构造函数抛出，当前对象的析构函数不会执行，但已构造完成的成员和基类会被销毁，因此应让成员本身管理资源。

常用异常保证：基本保证是对象仍有效且不泄漏；强保证是失败后保持原状；不抛异常保证用于必须可靠完成的操作。析构函数不应让异常逃逸；可失败的关闭操作可提供显式接口，让调用方处理错误，析构作为清理兜底。

## unique_ptr 的常用操作

```cpp
#include <memory>
#include <utility>

void unique_example() {
    auto owner = std::make_unique<float[]>(1024);
    float* observer = owner.get();  // 借用，不转移所有权
    observer[0] = 1.0f;
    auto next = std::move(owner);   // owner 为空，next 拥有原数组
    next.reset();                  // 释放；observer 从此悬空
}
```

`get()` 借出地址，`reset()` 释放旧资源并可接管新资源，`release()` 放弃管理并返回地址，调用方必须接手释放责任。不要从 `get()` 返回的地址再构造另一个独立拥有者，否则可能双重释放。

`unique_ptr<T>` 默认使用 `delete`，数组特化 `unique_ptr<T[]>` 使用 `delete[]`。其他资源必须匹配自定义 deleter：

```cpp
#include <cstdio>
#include <memory>
#include <stdexcept>

struct FileCloser {
    void operator()(std::FILE* file) const noexcept {
        if (file) std::fclose(file);
    }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

File open_read(const char* path) {
    File file(std::fopen(path, "rb"));
    if (!file) throw std::runtime_error("open failed");
    return file;
}
```

这个只读示例在析构中不报告关闭错误；需要可靠落盘的写文件接口应显式检查写入、刷新和关闭结果。

## 拷贝、移动与 Rule of Zero/Five

拷贝通常建立独立值或按类型契约共享资源；移动允许从源对象转移资源。`std::move` 只是产生可供移动重载使用的表达式，本身不搬数据。具名的右值引用变量作为表达式仍是左值。

`std::move(const_object)` 通常不能调用接收非 const 右值引用的移动构造，可能退回拷贝。标准库对象移动后通常有效但状态未指定，除非类型有更强保证；不能把 unique_ptr 移动后为空推广到所有类型。

Rule of Zero：让 vector、unique_ptr 等成员管理资源，尽量不手写特殊成员函数。若类型需要定制析构、拷贝构造、拷贝赋值、移动构造或移动赋值，应整体考虑这五个函数，即 Rule of Five，不意味着五个都必须手写。

下面有意维护“空指针对应长度 0”的移动后不变量，因此显式实现移动操作：

```cpp
#include <cstddef>
#include <memory>
#include <utility>

class Buffer {
public:
    explicit Buffer(std::size_t size)
        : data_(size ? std::make_unique<float[]>(size) : nullptr),
          size_(size) {}
    ~Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept
        : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {}
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);  // 先前持有的资源自动释放
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    float* data() noexcept { return data_.get(); }
    const float* data() const noexcept { return data_.get(); }
    std::size_t size() const noexcept { return size_; }
private:
    std::unique_ptr<float[]> data_;
    std::size_t size_;
};
```

`noexcept` 是承诺，异常逃逸会触发终止；它不是自动吞异常。容器扩容时，为满足异常保证，可能优先使用不抛异常的移动，或在可以拷贝时选择拷贝。返回局部值通常直接写 `return value;`，不必强行 `std::move`，以保留返回值优化机会。

## shared_ptr、weak_ptr 与线程边界

shared_ptr 通常通过控制块管理计数和 deleter；复制会增加共享拥有者，存在计数维护成本。`make_shared` 通常合并对象和控制块分配，但不应把具体分配次数当成所有实现的保证。

循环强引用会使计数无法归零。树的子节点可由父节点持有，回指父节点使用 weak_ptr：

```cpp
#include <memory>
#include <vector>

struct Node {
    std::vector<std::shared_ptr<Node>> children;
    std::weak_ptr<Node> parent;
};

bool has_parent(const Node& node) {
    if (auto parent = node.parent.lock()) {
        return true;  // parent 在这个作用域中保活父节点
    }
    return false;
}
```

`lock()` 一步尝试取得强引用；不要把 `expired()` 检查当成后续对象必然存活的保证。

共享同一控制块的**不同 shared_ptr 实例**可以并发维护各自的所有权；这不意味着可以无同步地修改同一个 shared_ptr 变量，也不意味着可以并发修改 `*ptr`。C++20 的 `atomic<shared_ptr<T>>` 解决指针变量的原子访问，仍不保护 T 的可变状态，也不保证实现无锁。

## 异步任务如何保活资源

把借用地址提交给线程池或 CUDA kernel 后，当前函数返回不等于任务完成。可以让任务持有 owner，或由调度器维护 in-flight 资源并在完成后释放。GPU 不会因收到裸地址自动增加 CPU 上的 shared_ptr 计数。

CUDA 分配必须匹配 CUDA 释放接口，并考虑 stream 使用顺序，见 [CUDA C++ 导读](/notes/cuda/cpp)。RAII 能管理清理入口，但正确的释放时机仍由异步协议决定。

## 自查与练习

- 验证 Buffer 移动后源对象为空、自移动不破坏对象、移动赋值释放旧资源。
- 解释 `get()` 与 `release()` 分别是否改变所有权。
- 写出一个循环引用并用 weak_ptr 解除。
- 为异步任务画出提交、最后一次访问、完成通知和销毁四个时刻。
