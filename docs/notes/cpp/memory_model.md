---
order: 8
title: C++ 并发内存模型
updated: 2026-09-09
tags: [cpp, atomic, happens-before, memory-order]
---

# C++ 并发内存模型

前置：[多线程与任务队列](/notes/cpp/concurrency)。本篇讨论哪些跨线程访问有定义，以及一个线程的写入如何安全地供另一个线程读取。

## data race 与顺序关系

不同线程对同一内存位置的冲突访问中至少有一次写，若涉及非原子访问且没有所需的 happens-before 关系，会产生 data race，属于未定义行为。线程恰好在不同时间片运行、测试几次结果正常，都不能代替同步。

建立推理链时区分：

- **sequenced-before：**单线程内部由语言规定的求值顺序。
- **synchronizes-with：**由同步操作建立的跨线程关系，例如同一 mutex 的解锁与随后获得该锁。
- **happens-before：**连接这些关系，判断前面的副作用能否安全地被后续操作使用。

它们描述语言层面的保证，不等于某条指令简单地“刷新所有缓存”。语义参考：[C++ 多线程执行与 data race](https://eel.is/c%2B%2Bdraft/intro.races)。

## mutex 和 join 已经建立同步

线程 A 在 mutex 下修改共享数据并解锁，线程 B 随后获取同一个 mutex 再读取，可以通过这条同步链访问数据。只在 A 中加锁而 B 不加锁则没有这条保证。

线程执行结束与成功 join 返回建立同步，因此 [线程示例](/notes/cpp/concurrency) 在 join 后读取 result，不需要额外 atomic。能用这种高层协议解决时，优先保持简单。

## atomic 保证什么

`std::atomic<T>` 为支持的类型提供原子访问与可选内存序。一个 atomic 上的修改存在修改顺序，但两个独立 atomic 不会自动合成一个事务。

```cpp
#include <atomic>

std::atomic<int> completed{0};

void count_one() {
    completed.fetch_add(1, std::memory_order_relaxed);
}
```

fetch_add 是一次读改写；分开的 load 加 store 即使各自原子，也可能丢失更新。比较交换用于“值仍为预期值才更新”，`compare_exchange_weak` 可以伪失败，常放在循环中；失败时 expected 会被写成观察到的值。

atomic 不一定 lock-free；可以查询 `is_lock_free()`。lock-free 也不等于每个线程都在有界步数内完成，后者是更强的 wait-free 性质。

## 常用内存序

| 内存序 | 核心保证 | 入门用途 |
| --- | --- | --- |
| `seq_cst` | 对 seq_cst 操作提供单一全序，并具有相应 acquire/release 语义 | 默认起点，仍需正确算法 |
| `relaxed` | 原子性和该原子对象的修改顺序；不建立其他数据的发布关系 | 独立统计计数 |
| `release` | 与读取其发布值的匹配 acquire 建立同步 | 发布已准备好的数据 |
| `acquire` | 与匹配 release 建立同步，使先前写入可用于后续读取 | 消费发布的数据 |
| `acq_rel` | 读改写同时具有 acquire/release 语义 | 需要双向顺序的 RMW 协议 |

不是所有操作都能使用所有内存序，例如普通 store 不能使用 acquire。初学不必把 consume 和独立 fence 作为默认工具。

## 一次性发布示例

下面是完整 C++17 程序：payload 是非原子变量，但 release/acquire 建立了从写入到读取的 happens-before。

```cpp
#include <atomic>
#include <cassert>
#include <thread>

int main() {
    int payload = 0;
    std::atomic<bool> ready{false};
    std::thread producer([&] {
        payload = 42;
        ready.store(true, std::memory_order_release);
    });
    std::thread consumer([&] {
        while (!ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        assert(payload == 42);
    });
    producer.join();
    consumer.join();
}
```

顺序链为：写 payload → release store → 读到 true 的 acquire load → 读 payload。关键不是“出现了 acquire”，而是它读到了匹配的发布值。更一般的规则还涉及 release sequence，见 [C++ 原子操作顺序](https://eel.is/c%2B%2Bdraft/atomics.order)。

若把 ready 的操作都换为 relaxed，就失去保护 payload 的发布关系。此例只发布一次，producer 发布后不再修改 payload；不能简单重置 ready 并反复覆写 payload，否则还需要消费者确认和 buffer 复用协议。

这是解释内存序的自旋示例，长时间等待应考虑条件变量或 C++20 atomic wait/notify。yield 不保证公平，也不提供内存同步。

## 常见误解

- `volatile` 不提供线程间原子性和同步，不能代替 atomic/mutex。
- 原子计数达到某个值，不一定发布其他数据；必须审查实际内存序和协议。
- shared_ptr 的计数维护不保护被管理对象的字段。
- 没有 data race 仍可能有逻辑竞争，例如“检查余额”和“扣款”分开加锁。
- x86 上偶然观察到的执行顺序不是可移植 C++ 的证明，编译器优化也参与其中。

## 与 CUDA 同步的边界

CPU 原子操作不能自动证明 GPU kernel、DMA 或跨 stream 的顺序。CUDA 扩展包含线程 scope 与设备相关条件，需要按 CUDA 契约选择 API。[NVIDIA CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)

同样，`cp.async` 完成等待和 block 线程 barrier 处理不同层次的问题，见 [异步流水线](/notes/cuda/async_pipeline)。不要把所有“同步”压缩成一个概念。

## 自查与练习

- 给发布示例标出 sequenced-before 和 synchronizes-with 边。
- 解释原子 load + store 为什么不能替代 fetch_add。
- 设计双向确认后才能复用 buffer 的协议，先用 mutex 实现。
- 把无锁队列、ABA、hazard pointer 和内存回收列为后续专题；在需要之前不以它们替代可靠的有锁队列。
