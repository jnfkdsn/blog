---
order: 1
title: C++ 前置知识
updated: 2026-05-18
tags: [cpp, cuda, raii, template]
status: draft
---

# C++ 前置知识

## 0.1指针与内存管理

### 为什么 CUDA 需要指针
CPU(HOST),CUDA(DEVICE)有各自独立的内存空间
1. 在 GPU 上分配内存（`cudaMalloc` 返回一个指向 GPU 内存的指针）
2. 把数据从 CPU 拷贝到 GPU（通过指针指定源和目标地址）
3. 把 GPU 指针作为参数传给 kernel 函数
CUDA Runtime API 主要通过指针表达内存位置和数据搬运。PyTorch 中的 `model.to(device)` 可以理解成把参数和 buffer 迁移到 GPU，但底层还涉及 allocator、stream、异步 copy 等机制，不完全等价于用户手写逐个 `cudaMemcpy`。

### 动态内存分配
栈上的数组大小必须在编译期确定，当需要运行时确定大小的数组时需要动态分配(堆分配)
```c++
int n; std::cin >> n; 
float* data = new float[n]; 
delete[] data;
```

### void* 类型转换
void*可以指向任何类型的内存
cudaMalloc原型：
```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size);
```
cudaMalloc 需要修改调用者的指针变量。C 函数要修改外部变量，必须传该变量的指针。如device_data 是 float*，它的指针就是 float**（传入后隐式转为 void**）。
如果是void*按值传递只能修改副本，不能修改原变量，修改原始变量必须传递指针or引用。
& 出现在类型声明里就是引用，出现在表达式里就是取地址。
类型转换：
```cpp
void* generic_ptr;
cudaMalloc(&generic_ptr, 1024);
float* float_ptr = static_cast<float*>(generic_ptr);
```

## 0.2引用与const
```cpp
// input 用 const（只读），output 不用 const（写入）const表示不可修改
__global__ void relu_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}
```

## 0.3结构体与类
### struct
### class
类和结构体在 C++ 中几乎相同，唯一区别：`struct` 默认成员 `public`，`class` 默认 `private`。
构造函数初始化列表更高效
```cpp
MyClass(int a, int b) : x(a), y(b) {}
```
### this指针
在成员函数内部，`this` 指向当前对象本身
两种必须用 this 的场景：
1. 参数名和成员名相同时（消除歧义）
推荐使用size_来命名成员避免冲突
2. 返回自身引用（链式调用）
```c++
class TensorBuilder {
    int batch_, seq_len_;
public:
    TensorBuilder& set_batch(int b)   { batch_ = b;   return *this; }
    TensorBuilder& set_seq_len(int s) { seq_len_ = s;  return *this; }
};
// 链式调用
TensorBuilder builder;
builder.set_batch(32).set_seq_len(512);  // 每个函数返回 *this，所以能接着调用
```

## 0.4 RAII与智能指针

### RAII（资源获取即初始化）
核心思想：
> 把资源的获取（分配内存/打开文件/锁）放在对象的构造函数中，把资源的释放放在析构函数。不再需要手动管理内存

### 禁止拷贝
避免浅拷贝复制地址，释放相同的位置

### 智能指针 std::unique_ptr

C++ 标准库提供了 `unique_ptr`——自动管理资源生命周期的智能指针，本质就是 RAII 的通用封装：

```cpp
#include <memory>
// unique_ptr 独占所有权：同一时间只有一个 unique_ptr 指向某个对象
auto host_data = std::make_unique<float[]>(1024);
// 离开作用域时自动 delete[]
// 不能拷贝，只能移动
// auto copy = host_data;                // 编译错误
auto moved = std::move(host_data);       
// 此时 host_data 变成 nullptr
```

### std::shared_ptr
允许多个指针共享一个对象，内部维护引用计数。

### 用 RAII 管理 CUDA 资源

`std::unique_ptr` 默认只会调用 `delete`，不能直接释放 `cudaMalloc` 得到的指针。CUDA 资源也可以用 RAII 思路管理，但需要自定义 deleter：

```cpp
struct CudaDeleter {
    void operator()(float* p) const {
        if (p) cudaFree(p);
    }
};

std::unique_ptr<float, CudaDeleter> make_device_buffer(size_t n) {
    float* ptr = nullptr;
    cudaMalloc(&ptr, n * sizeof(float));
    return std::unique_ptr<float, CudaDeleter>(ptr);
}
```

真实项目里通常还会把 `cudaMalloc` 的返回值接入 `CUDA_CHECK`，避免分配失败后继续使用空指针。

## 0.5 模板

CUDA kernel经常需要支持多种数据类型和多种配置，模板可以用一份代码，编译器自动生成特化版本
函数模板，类模板
`template <typename T>`
```cpp
template <typename T>
__global__ void add(T* a, T* b, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// 调用时指定类型
add<float><<<grid, block>>>(a_float, b_float, c_float, n);
```
### 非类型模板参数 
编译期常量值，如CUDA block size等 
`template <int BLOCK_SIZE>`
```cpp
// BLOCK_SIZE 是一个编译期确定的 int 值
template <int BLOCK_SIZE>
__global__ void reduce_kernel(float* input, float* output, int n) {
    __shared__ float shared_data[BLOCK_SIZE];  // 编译期确定大小
    //
}
// 使用——编译成不同版本
reduce_kernel<256><<<grid, 256>>>(input, output, n);  // BLOCK_SIZE=256
reduce_kernel<512><<<grid, 512>>>(input, output, n);  // BLOCK_SIZE=512
```
多个模板参数：
```cpp
template <
    typename ElementA,          // A 矩阵的元素类型 (float, half)
    typename ElementB,          // B 矩阵的元素类型
    typename ElementC,          // C 矩阵的元素类型
    typename LayoutA,           // A 的内存布局 (RowMajor, ColumnMajor)
    typename LayoutB,
    int TileM, int TileN, int TileK  // Tile 大小
>
```
### 模板特化
某种类型需要特殊处理
```cpp
// 通用版本
template <typename T>
struct TypeName {
    static const char* name() { return "unknown"; }
};
// 特化版本
template <>
struct TypeName<float> {
    static const char* name() { return "float32"; }
};
std::cout << TypeName<float>::name();   // 输出 "float32"
```

## 0.6 constexpr
### constexpr 编译期计算
`constexpr`关键字标记的函数或变量，要求其在编译期就能确定
constexpr 变量：编译期常量
constexpr 函数：编译期求值

### if constexpr（C++17）
在编译期做条件分支，不满足的分支直接消除

### std::is_same_v和类型traits
`std::is_same_v<T, U>` 在编译期判断两个类型是否相同

## 0.7 命名空间与头文件管理
### 命名空间namespace 
避免不同库之间名字冲突
namespace A{
namespace B{
    void func(){}
}
}

### 头文件 include
头文件.h/.hpp包含声明，源文件.cpp/.cu包含定义
模板函数和constexpr函数的实现在头文件，编译器在编译 main.cpp 时就要看到函数实现，不能等到链接
### 前向声明
前向声明是在头文件里先告诉编译器“有这个类型/函数”，但暂时不给完整定义。它可以减少头文件互相 include，降低编译依赖。

```cpp
// tensor.hpp
class Tensor;

void launch_kernel(const Tensor& x);
```

只有在需要知道对象大小或访问成员时，才必须 include 完整定义：

```cpp
// tensor.cpp
#include "tensor.hpp"
#include "tensor_impl.hpp"
```

如果只是使用 `Tensor*`、`Tensor&` 或声明函数参数，前向声明通常够用；如果要按值保存 `Tensor` 或访问 `x.shape()`，就需要完整定义。

## 0.8 lambda函数
- Lambda 语法：`[捕获](参数) { 函数体 }`
- `[=]` 按值捕获，`[&]` 按引用捕获 所有外部参数

## 0.9 cmake和pybind11

CUDA 算子接到 Python 时通常会出现三层代码：

- `.cu`：写 kernel 和 C++ launcher。
- `.cpp`：用 pybind11 或 Torch extension 暴露 Python 接口。
- `CMakeLists.txt` / `setup.py`：负责把 C++/CUDA 编译成 Python 能 import 的 `.so`。

这部分和 [CMake 构建实践](/notes/cuda/cmake) 连起来看。
