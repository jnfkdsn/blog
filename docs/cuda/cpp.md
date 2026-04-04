# CPP

## 0.1指针与内存管理

### 为什么 CUDA 需要指针
CPU(HOST),CUDA(DEVICE)有各自独立的内存空间
1. 在 GPU 上分配内存（`cudaMalloc` 返回一个指向 GPU 内存的指针）
2. 把数据从 CPU 拷贝到 GPU（通过指针指定源和目标地址）
3. 把 GPU 指针作为参数传给 kernel 函数
CUDA通过指针通信(model.to(device)即对每个参数cudaMemcpy(HostToDevice))

### 动态内存分配
栈上的数组大小必须在编译期确定，当需要运行时确定大小的数组时需要动态分配(堆分配)
```c++
int n; std::cin >> n; 
float* data = new float[n]; 
delete[] data 
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
```cpp
struct dim3{
    unsigned int x,y,z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1)
        :x(x),y(y),z(z) {}
};
dim3 grid(16, 16);    // grid.x = 16, grid.y = 16, grid.z = 1
```
### class
类和结构体在 C++ 中几乎相同，唯一区别：`struct` 默认成员 `public`，`class` 默认 `private`。
```cpp
class BlockManager {
private:
    std::vector<CacheBlock> blocks_;
    int num_free_blocks_;
public:
    // 构造函数
    BlockManager(int total_blocks, int block_size)
        : num_free_blocks_(total_blocks) {
        blocks_.resize(total_blocks);
        for (int i = 0; i < total_blocks; i++) {
            blocks_[i].block_id = i;
            blocks_[i].ref_count = 0;
            cudaMalloc(&blocks_[i].key_data, block_size * sizeof(float));
            cudaMalloc(&blocks_[i].value_data, block_size * sizeof(float));
        }
    }
    // 析构函数：对象销毁时自动调用
    ~BlockManager() {
        for (auto& block : blocks_) {
            cudaFree(block.key_data);
            cudaFree(block.value_data);
        }
    }
};
```
构造函数初始化列表更高效
```cpp
MyClass(int a, int b) : x(a), y(b) {}
```
### this指针
在成员函数内部，`this` 是一个隐式指针，指向当前对象本身
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

## RAII与智能指针
