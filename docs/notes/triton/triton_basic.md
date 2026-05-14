---
order: 2
---
## 1. triton介绍

Triton 是 OpenAI 开发的GPU 编程语言和编译器，旨在简化并优化在GPU上执行的复杂操作的开发。它的定位介于 CUDA 和 PyTorch 之间：

```
PyTorch             ← 最高层：用 Python 写算子，底层自动调用 cuBLAS/cuDNN
  │                    优点：简单
  │                    缺点：无法自定义 kernel，无法做 kernel fusion
  │
Triton              ← 中间层：用 Python 写 GPU kernel，编译器自动优化
  │                    优点：代码简洁、自动处理共享内存/向量化/合并访问
  │                    缺点：无法精确控制 warp 级行为
  │
CUDA C++            ← 最底层：直接控制 thread/warp/block/memory
                       优点：性能最高
                       缺点：代码量大、出错概率高、优化周期长
```


```
  PyTorch 模型代码
        │
        ▼
  torch.compile()
        │
        ▼
  TorchDynamo (图捕获)
        │
        ▼
  AOTAutograd (自动微分)
        │
        ▼
  TorchInductor (代码生成)     ← Inductor 的默认后端就是 Triton
        │
        ▼
  Triton Kernel (自动生成)
        │
        ▼
  Triton 编译器
        │
        ▼
  PTX / SASS (GPU 机器码)
```
调用 `torch.compile(model)` 时，PyTorch 会自动把计算图编译成 Triton kernel。

## 2.triton编程模型
在 CUDA 中，kernel被多个thread执行，需要设置每个thread做什么
在 Triton 中，kernel被多个program执行。每个program处理一个数据块（block），只需要描述一个 program 如何处理它负责的那一块数据。

### 示例：
```python
@triton.jit
def my_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Step 1: 确定当前 program 处理哪个块
    pid = tl.program_id(axis=0)                    
    # Step 2: 计算这个块内的元素偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 每个program处理一个blocksize的向量
    # Step 3: 生成边界 mask（防止越界访问）
    mask = offsets < N
    # Step 4: 从 global memory 加载数据（向量化 load）
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Step 5: 计算 + 写回
    y = x * 2.0
    tl.store(output_ptr + offsets, y, mask=mask)
```
### 常用元语
`tl.program_id(axis)`：获取当前program的ID，axis=0/1/2三个维度
`tl.arange(0,N)`:生成 `[0, 1, ..., N-1]` 整数向量
`tl.load(ptr, mask, other)`:带mask的向量化加载
`tl.store(ptr, val, mask)`:带mask的向量化写入
`tl.max(x, axis)`:向量归约求最大值
`tl.sum(x, axis)`:向量归约求和
`tl.constexpr` : 编译期常量标记

### Launch语法
CUDA 用 `<<<grid, block>>>` 启动 kernel，Triton 用 `kernel[grid](args)` ：
```python
# CUDA
# my_kernel<<<(N + 255) / 256, 256>>>(d_input, d_output, N);

# Triton
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
my_kernel[grid](input, output, N, BLOCK_SIZE=256)
```
grid 通过 `meta` 字典接收 kernel 的 `tl.constexpr` 参数
## 3. vector_add
```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,n_elements,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets<n_elements
    x = tl.load(x_ptr+offsets,mask=mask)
    y = tl.load(y_ptr_offsets,mask=mask)
    output = x+y
    tl.store(output_ptr+offsets,output,mask=mask)

def add(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta:(triton.cdiv(n_elements,BLOCK_SIZE=1024))
    add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE=1024)
    return output

```

## 4. triton softmax
```python
@triton jit
def softmax_kernel(output_ptr,input_ptr,input_row_stride,output_row_stride,n_cols,BLOCK_SIZE:tl.constexpr):
    row_idx = tl.pregram_id(0)
    # 当前行起始地址
    row_start_ptr = input_ptr+row_idx*input_row_stride
    col_offsets = tl.arange(0,BLOCK_SIZE)
    # 加载一行数据到寄存器
    input_ptrs = row_start_ptr + clo_offsets
    mask = col_offsets < n_cols
    row = tl.load(inputs_ptrs,mask=mask,other=-float('inf'))
    # softmax计算
    row_max = tl.max(row,axis=0)
    numerator = tl.exp(row-row_max)
    denominator = tl.sum(numerator,axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + output_row_stride * row_idx + col_offsets
    tl.store(output_ptrs,softmax_output,mask=mask)

def softmax(x:torch.Tensor) -> torch.Tensor:
    n_rows,n_cols = x.shape
    # BLOCK_SIZE 必须是 2 的幂，且 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 限制最大值避免寄存器溢出
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    # 控制每个 program 用多少个 warp
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    output = torch.empty_like(x)
    # 一个 program 处理一行 → grid = 行数
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

### 对比CUDA softmax
1. tl.max和tl.sum 编译器自动选择最优归约策略，warp shuffle归约
2. row = tl.load()直接把整行加载到寄存器，并且自动选择向量化存取


## 5. autotune 自动调参
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
    ],
    key=['n_cols'],  # 当 n_cols 变化时重新搜索
)
```
### 关键参数

#### `BLOCK_SIZE`
每个 program 处理的元素数量。更大的 BLOCK_SIZE 意味着：
- 每个 program 处理更多数据 → 需要更多寄存器
- program 数量更少 → GPU 可能没跑满
- 有时候更大的 block 能更好地利用带宽（更少的内存事务）

#### `num_warps`
每个 program 使用多少个 warp（每个 warp = 32 线程）。
```
num_warps=4  → 128 线程/program
num_warps=8  → 256 线程/program
num_warps=16 → 512 线程/program
```
`num_warps` 影响 **occupancy**：
- 太少：SM 上活跃的线程不够，无法隐藏内存延迟
- 太多：每个 warp 分到的寄存器变少，可能导致寄存器溢出（spill to local memory）
经验法则：
- **Memory-bound kernel**（如 softmax）：更多 warp（8-16）有助于隐藏内存延迟
- **Compute-bound kernel**（如 GEMM）：适度 warp（4-8）+ 足够寄存器给 Tensor Core 用

#### `num_stages`
控制**软件流水线（software pipelining）**的深度。
```
num_stages=1: 无流水线
  Load_0 → Compute_0 → Load_1 → Compute_1 → ...

num_stages=2: 双缓冲（加载下一批数据时计算当前批）
  Load_0 → Load_1 → Compute_0 → Load_2 → Compute_1 → ...
             ↑ 预加载                ↑ 预加载

num_stages=3: 三缓冲
  Load_0 → Load_1 → Load_2 → Compute_0 → Load_3 → Compute_1 → ...
```

更多 stages = 更多预取 = 更好的延迟隐藏，但需要更多 shared memory 来暂存预取的数据。

经验法则：
- Softmax 这种 kernel 通常 `num_stages=2` 就够
- GEMM 等需要循环加载多个 tile 的 kernel 需要 `num_stages=3-4`

#### key
`key` 指定哪些参数变化时需要重新搜索最优配置

## 6. triton局限性

### 1. 需要精确控制 Warp 级行为

```python
# Triton 中没有等价物：
# __ballot_sync()   — warp 投票
# __shfl_sync()     — 精确控制 lane 间通信
# __match_any_sync() — 值匹配
```
Triton 的 `tl.max` / `tl.sum` 虽然底层可能用 warp shuffle 实现，但无法控制具体的 shuffle pattern。

### 2. 需要精确控制 Shared Memory 布局

在 CUDA 中可以精确控制 shared memory 的 bank 映射（如手动 swizzle / padding）。Triton 中 shared memory 的使用完全由编译器决定，无法手动分配和布局。

```cpp
// CUDA：精确控制 shared memory 布局
__shared__ float smem[32][33];  // 33 列  padding
smem[threadIdx.y][threadIdx.x] = ...;  // 精确控制访问模式
```

### 3. BLOCK_SIZE 必须是 2 的幂

```python
# 这会报错
offsets = tl.arange(0, 100)  

# 必须这样
offsets = tl.arange(0, 128)   # 128 = 2^7
mask = offsets < 100          # 用 mask 处理多余的部分
```

这意味着对某些特殊形状的 tensor，Triton 可能浪费一些计算在 mask 为 False 的无效位置上。

### 4. 跨 Block 通信

Triton 的编程模型是**每个 program 独立**的，没有直接支持跨 program（跨 block）通信的机制, **它天然适合一个 program 能独立完成的计算**。需要跨 program 通信的场景，CUDA 更灵活

如果需要全局归约（如对整个 tensor 求 sum），需要：
- 先在每个 program 内做局部归约
- 用 `tl.atomic_add` 原子操作累加到全局变量
- 或者分两个 kernel 完成（先局部归约，再最终归约）

CUDA 中可以用 cooperative groups 做更灵活的跨 block 同步。

### 5. 编译时间

Triton 的 JIT 编译在首次调用时需要几秒到几十秒（尤其是 autotune 搜索所有 config 时）。对于需要快速启动的场景（如推理服务的冷启动），这可能是问题。

解决方案：
- 预编译 + 缓存（Triton 默认会缓存编译结果到 `~/.triton/cache/`）
- AOT（Ahead-of-Time）编译

### 6. 性能天花板

在极致优化场景下（如追求 cuBLAS 级别的 GEMM 性能），手写 CUDA 仍然能比 Triton 快 10-20%。原因：
- Triton 编译器的 shared memory 布局选择可能不是最优的
- 无法使用 PTX 内联汇编做微指令级优化
- 无法精确控制 warp scheduling

