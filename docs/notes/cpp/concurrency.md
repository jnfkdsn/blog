---
order: 7
title: 多线程与任务队列
updated: 2026-09-09
tags: [cpp, thread, mutex, condition-variable]
---

# 多线程与任务队列

前置：[RAII 与所有权](/notes/cpp/ownership)、[STL](/notes/cpp/stl)。先建立可解释的有锁实现，再学习 [并发内存模型](/notes/cpp/memory_model)。

## 线程生命周期

`std::thread` 创建线程执行函数，参数默认按值保存；需要借用可显式使用 `std::ref`，但必须保证被引用对象存活。`join()` 等待线程结束；`detach()` 脱离管理，不解决对象生命周期和程序退出问题。

```cpp
#include <thread>

int compute_in_thread() {
    int result = 0;
    std::thread worker([&] { result = 42; });
    worker.join();
    return result;  // join 后读取，不与写操作并发
}
```

joinable 的 thread 对象析构会调用 `std::terminate`，即使底层线程已经结束但尚未 join。实际代码要处理提前返回和异常路径，可使用 RAII 线程管理。C++20 `std::jthread` 析构时请求停止并 join，但停止是协作式的，工作函数必须检查 stop token 或使用支持停止的等待方式。

线程入口未捕获异常会终止程序。线程池应捕获任务异常，通过 future、错误队列等向调用方传播。

## mutex 与锁的范围

`std::mutex` 保护一组共享状态及其不变量。对这些状态的所有冲突访问都要遵守同一个同步协议，不能只给写操作加锁而让读操作裸读。

- `lock_guard`：简单作用域加锁。
- `unique_lock`：支持解锁、重新加锁，适合条件变量等待。
- `scoped_lock`（C++17）：可同时获取多个 mutex，使用避免死锁的获取策略。

锁内尽量只修改受保护状态，不执行耗时任务、I/O 或未知回调。多把锁采用一致的获取协议；递归 mutex 不能自动解决锁顺序问题。`shared_mutex` 适合部分读多写少场景，但调度和写者等待成本需要测量。

## 条件变量等待的是条件

条件变量可能伪唤醒，因此用谓词形式 `cv.wait(lock, predicate)`。它在等待时释放 mutex，醒来重新获取，再检查条件。通知本身不保存一份可供将来领取的任务；共享状态才记录“是否有任务”。修改状态与检查谓词必须遵守同一个锁协议。

## 可关闭的有界队列

下面是完整的 C++17 教学类，元素固定为 int，让并发协议不被泛型元素抛异常等问题干扰。容量上限提供背压：满时生产者等待；关闭后拒绝新任务，消费者排空旧任务后退出。

```cpp
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("zero capacity");
    }
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    bool push(int value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_) return false;
        queue_.push_back(value);
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    std::optional<int> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return std::nullopt;
        int value = queue_.front();
        queue_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }
private:
    const std::size_t capacity_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<int> queue_;
    bool closed_ = false;
};
```

正确性依赖三个条件：队列和 closed 都由同一 mutex 保护；所有 wait 都检查谓词；关闭通知两类等待者。close 可以重复调用，但**不能在还有线程使用队列时析构队列**。

正常排空流程是启动消费者与生产者，等待生产者结束，close，等待消费者结束，最后销毁队列。提前取消可先 close，让阻塞中的 push 返回 false，再 join 所有线程。不要在没有消费者的情况下等待一个被满队列阻塞的生产者结束。

这是正常执行路径的同步示例；deque 分配仍可能失败。生产环境需捕获工作线程异常、触发统一关闭并 join。推广到 T 时还要明确入队/出队移动抛异常的保证，不能机械替换 int。

## future、线程池与退出协议

`promise<T>` 写入结果或异常，`future<T>::get()` 等待并取出，异常会在 get 时重新抛出。`packaged_task` 包装可调用对象与结果通道。`std::async` 默认策略允许延迟执行，不能把它当作固定大小线程池；明确异步时可指定 `std::launch::async`，但仍需管理并发数量。

线程池通常由工作线程、队列和完成通道组成。取任务后释放队列锁再执行任务。需要明确：关闭是排空还是丢弃？任务异常如何传播？队列满了怎么办？工作线程是否可能等待同池中无法获得执行机会的子任务？

CPU 队列为空不代表 GPU 任务执行完毕。若任务只是提交 kernel，完成通道必须区分“已提交”和“GPU 已完成”。

## 自查与练习

- 验证空队列关闭会唤醒消费者，满队列关闭会唤醒生产者。
- 多生产者提交唯一 ID，多消费者收集结果，验证每项恰好消费一次。
- 验证关闭后旧任务仍能取出、新 push 返回 false、重复 close 不出错。
- 用 TSan 检查执行过的并发路径；工具没有报错不构成正确性证明。

参考：[C++ 条件变量](https://eel.is/c%2B%2Bdraft/thread.condition.condvar)、[thread 生命周期](https://eel.is/c%2B%2Bdraft/thread.thread)、[Clang ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html)。
