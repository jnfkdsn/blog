---
order: 1
title: CUDA 所需的 C++ 前置知识
updated: 2026-09-09
tags: [cpp, cuda, raii, template]
---

# CUDA 所需的 C++ 前置知识

通用 C++ 已整理为 [C++ 工程基础](/notes/cpp/)。本篇保留 CUDA 所需的知识入口与 host/device 边界；GPU 线程和内存层次见 [CUDA 基础语法](/notes/cuda/cuda_basic_syntax)。

## 写 kernel 前先掌握什么

| C++ 主题 | CUDA 中的用途 | 入口 |
| --- | --- | --- |
| 指针、引用与 const | 输入输出地址、长度、只读访问路径 | [基础语法](/notes/cpp/basics) |
| 对象生命周期与对齐 | buffer 边界、字节数、向量访问条件 | [对象与存储](/notes/cpp/object_lifetime) |
| RAII、移动与所有权 | 管理 device buffer、stream、event | [所有权](/notes/cpp/ownership) |
| vector 与视图 | host 元数据、shape、输入数据管理 | [STL](/notes/cpp/stl) |
| 模板与 constexpr | dtype、tile size、layout 的编译期配置 | [模板](/notes/cpp/templates) |
| 编译链接与绑定 | launcher、共享库和 Python 扩展 | [构建调试](/notes/cpp/build_debug)、[CUDA CMake](/notes/cuda/cmake) |

CPU [多线程](/notes/cpp/concurrency) 和 [内存模型](/notes/cpp/memory_model) 在开发调度器与 runtime 时进一步学习，不必作为第一个 kernel 的全部前置。

## 地址不等于可在任意处理器上解引用

在典型独立 GPU 上，host 和 device 使用不同的物理存储；UVA、managed memory、映射内存等提供不同地址与访问机制，不能简单归纳为所有 CPU/GPU 指针完全隔离或完全通用。持有一个地址，仍需知道它属于何种分配、谁可以访问、何时可访问。

`cudaMalloc` 将分配结果写入调用方的指针；C 形式接口是 `cudaError_t cudaMalloc(void**, size_t)`。普通 C++ **不允许 `float**` 隐式转换为 `void**`**；CUDA C++ 提供的类型化模板重载允许常见的 `cudaMalloc(&typed_ptr, bytes)` 写法。不能把这种 API 包装解释成语言的二级指针转换规则。

下面显式使用 void* 中间变量，检查大小与错误，并用 deleter 配对释放。这是 host 端资源管理示例，需要 CUDA Toolkit 编译环境。

```cpp
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>

inline void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

struct CudaDeleter {
    void operator()(float* ptr) const noexcept {
        if (ptr) {
            const auto error = cudaFree(ptr);
            if (error != cudaSuccess) {
                std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(error));
            }
        }
    }
};
using DeviceBuffer = std::unique_ptr<float, CudaDeleter>;

DeviceBuffer make_device_buffer(std::size_t count) {
    if (count == 0) return DeviceBuffer{};
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("device buffer size overflow");
    }
    void* raw = nullptr;
    cuda_check(cudaMalloc(&raw, count * sizeof(float)));
    return DeviceBuffer(static_cast<float*>(raw));
}
```

这里的 unique_ptr 在 CPU 上管理 GPU 地址；不能在 CPU 上用 `*buffer` 读取 device 数据。deleter 不向外抛异常，实际项目可以使用日志或显式关闭接口报告清理失败。

CUDA 内存接口与地址空间说明见 [NVIDIA CUDA Runtime Memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)。

## kernel 中的 const 与模板

```cpp
template <typename T, int BLOCK_SIZE>
__global__ void relu_kernel(const T* input, T* output, int n) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024,
                  "invalid block size");
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) output[i] = input[i] > T{0} ? input[i] : T{0};
}
```

这是用于理解参数的示例，假定一维 launch 的 `blockDim.x == BLOCK_SIZE`、n 非负、索引不溢出且分配足够大。host 在 n 为 0 时跳过 launch；非空时例如选择 `relu_kernel<float, 256>`，并使用 256 个线程。模板参数不会自动设置实际 launch 配置，也不替代设备限制检查。

`const T*` 限制通过 input 写入，不保证没有其他别名写同一地址。多个线程或多个 kernel 的冲突仍需通过布局和同步协议避免。`constexpr` 函数可以运行时执行；非类型模板参数则要求相应编译期实参，详见模板篇。

## host 标准库与 device 代码的边界

host 可以用 vector 保存输入、用 unique_ptr 管理资源、用线程池组织任务；普通 host 标准库对象不能直接当作 GPU 可用容器传入 kernel。kernel 通常接收 device 数据指针和简单元数据；device 可调用接口受 CUDA 工具链与库支持限制，不能认为所有 std 接口都可用。

模板只是生成代码的机制，不会把 host-only 函数变成 device 函数。CUDA 修饰符、受支持的标准库设施和目标架构仍要分别满足。

## RAII 与异步完成

资源拥有者应覆盖资源最后一次使用。launch 返回通常只说明完成提交；任务的输入、输出和异步拷贝所用 host buffer 都有各自的使用期限。

- 同一 stream 的操作有相应顺序；跨 stream 的依赖通常通过 event 等机制建立。
- 用完成事件判断能否复用 buffer，不能只根据提交函数已经返回。
- 传统 cudaFree 可能引入隐式同步；不要把析构释放当作可扩展的调度协议。
- 使用 cudaMallocAsync/cudaFreeAsync 时，要明确分配、访问、释放在 stream 中的顺序，尤其是跨 stream 使用。
- GPU 拿到裸指针不会增加 host 上 shared_ptr 的引用计数，调度器仍需保活 owner。

入门阶段可以在已知完成点后让资源出作用域，性能工程再设计 in-flight 资源回收。kernel 内的异步 copy 与 stage 复用见 [Ampere 异步流水线](/notes/cuda/async_pipeline)。

## 自查

1. 为什么 typed pointer 的 cudaMalloc 调用不是 float** 到 void** 的隐式转换？
2. 为什么 unique_ptr 的 get() 可以作为 kernel 参数，却不能证明异步使用期间资源一直存活？
3. 模板 block size 与 launch block size 不一致会发生什么？
4. 为什么 host vector 和 CUDA device buffer 需要不同的分配与释放路径？
5. 一个 Python 扩展如何验证 dtype、device、连续性并选择正确 stream？继续阅读 [CUDA CMake 实践](/notes/cuda/cmake) 和 [绑定基础](/notes/cpp/build_debug)。
