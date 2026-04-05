# CUDA 基础语法
 
## 2. 函数修饰符
| 修饰符 | 在哪执行 | 谁能调用 | 典型用途 |
|--------|---------|---------|---------|
| `__global__` | GPU | CPU（通过 `<<<>>>` ） | kernel 入口函数 |
| `__device__` | GPU | GPU | kernel 内部的工具函数 |
| `__host__` | CPU | CPU | 普通 C++ 函数(不加修饰符等于host) |
| `__host__ __device__` | 两边 | 两边 | 通用工具函数 |

## 3. 内存管理API
显存分配与释放：
cudaMalloc(&d_data,size);
cudaFree(d_data);

数据传输
```cpp
// cudaMemcpy(目标地址, 源地址, 字节数, 传输方向)
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```
四种传输方向：
| 方向枚举 | 含义 | 场景 |
|---------|------|------|
| `cudaMemcpyHostToDevice` | CPU → GPU | 把输入数据传到 GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU | 把计算结果取回 CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU | GPU 内部数据搬移 |
| `cudaMemcpyHostToHost` | CPU → CPU | 很少用（等于 memcpy） |

初始化显存：
cudaMemset(d_data,0,size); //清0

## 4. kernel
### 4.1 线程层次结构
```
Grid（网格）
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 255
├── ...
└── Block N-1
    └── ...
```
- **Grid**：一次 kernel 启动产生的所有线程
- **Block**：一组线程。同一个 Block 内的线程可以通过 shared memory 通信、可以同步
- **Thread**：最小执行单位
kernel_function<<<grid_size, block_size>>>(参数...);
总线程数=grid_size x block_size

### 4.2 内置变量
| 变量 | 类型 | 含义 |
|------|------|------|
| `threadIdx.x` | `uint3` | 当前线程在 Block 内的索引 |
| `blockIdx.x` | `uint3` | 当前 Block 在 Grid 内的索引 |
| `blockDim.x` | `dim3` | 每个 Block 的线程数 |
| `gridDim.x` | `dim3` | Grid 中的 Block 数 |
**计算全局线程 ID 的公式**：
```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

### 4.3 多维线程配置
Grid Block可以是1D,2D,3D的
```cpp
//2D
dim3 block(16, 16);       // 每个 block 16×16 = 256 个线程
dim3 grid(
    (width + 15) / 16,    // x 方向的 block 数
    (height + 15) / 16    // y 方向的 block 数
);
matrix_kernel<<<grid, block>>>(d_matrix, width, height);
__global__ void matrix_kernel(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = row * width + col;  // 行主序索引
        matrix[idx] *= 2.0f;
    }
}
```

- 每个 Block 最多 **1024** 个线程
- 多维时各维度限制不同：x ≤ 1024, y ≤ 1024, z ≤ 64，且 x × y × z ≤ 1024
- Block size 应该是 **32 的倍数**（warp size），否则有线程浪费

## 5. 同步机制

