---
order: 30
title: 同步与异步操作
updated: 2026-10-04
---

# 同步与异步操作

共享存储让线程可以交换数据，异步启动让 host 可以提交工作后继续执行。两者都需要描述“哪项工作完成以后，下一项才可以开始”，但同步范围不同。

先继续四线程轮转，再把整个 kernel 放进“复制输入→计算→复制输出”的流程。本章分别解释 block 内的数据可见性与 host/device 之间的完成依赖，避免把所有等待都理解成同一种 barrier。

## 1. Block 内的生产者与消费者

轮转中线程 t 先写 `tile[t]`，再读 `tile[(t+1)%4]`。只保证同一个线程先写后读还不够：

```text
线程 0：写 tile[0]=10 → 读 tile[1] → ...
线程 1：                      尚未写 tile[1]=20
```

这是一种不应被程序允许的执行次序。四个线程的写入虽然没有冲突，读取仍依赖其他线程的写入。

把 `gpu.barrier` 放到写和读之间：

```text
各线程写自己的 tile[t]
           ↓
     gpu.barrier
           ↓
各线程读邻居的 tile[(t+1)%4]
```

barrier 要求 workgroup 的参与者到达同一个同步点，并建立此前内存访问对组内线程的可见性。它不只是让某一个线程“慢一点”，也不是只对发出指令的线程刷新缓存。

这一步之后可以推导输入 `[10,20,30,40]` 对应输出 `[20,30,40,10]`。若只在单线程解释器中按“所有写再所有读”的固定顺序执行，即使删去 barrier 也可能看似正确；这不能证明并行程序安全。

## 2. 收敛执行与尾部线程

GPU barrier 的重要约定是同一 workgroup 的线程需要收敛地执行：要么都经过这次 barrier，要么都不经过。不能让一部分线程直接返回，另一部分一直等待它们。

对于有尾块的数据，常见结构是：

```text
每个线程计算 valid
若 valid：从输入读取
否则：为自己负责的共享位置准备算法允许的填充值
每个线程写自己的共享位置
所有线程经过 barrier
按算法消费共享数据；只让有效输出位置写回
```

这里条件保护的是内存访问，barrier 留在所有参与者都能到达的位置。并非所有算法都需要无效线程填值：取决于后续有没有人读它对应的位置。但若确实会读取，就必须保证内容已经定义且填充值不改变算法语义。

普通 IR verifier 通常不会证明任意控制流里的 barrier 都满足运行时收敛，也不会检查每一对共享读写的竞争。合法语法、合法 SSA 和并行语义正确是不同层次。

## 3. Barrier 的范围

同一 block 的 barrier 不能用来等待所有 block 完成。不同 block 可能以任意顺序或分批调度；一个 block 等待另一个尚未被调度的 block，甚至可能使进展停住。

若需要全体工作完成后再进入下一阶段，最直接的结构是先结束一个 kernel，再通过有依赖的第二次启动执行后续阶段。某些目标提供更专门的协作启动或同步机制，但必须满足其单独契约，不能从 `gpu.barrier` 推断这些能力。

subgroup 内的通信和同步又是更小范围。即使目标的线程以 warp 组织，也不能仅凭“它们在同一 warp”就省略算法所需的内存同步。使用 shuffle、subgroup reduction 等操作时，要看参与者与有效性的规定。

## 4. Host 提交与设备完成

现在考虑 kernel 外部。设备输入必须先有内容，host 读取结果也必须等复制完成。一个最小异步链是：

```text
%begin = gpu.wait async
%copied_in = gpu.memcpy async [%begin] %device_x, %host_x
  : memref<?xf32>, memref<?xf32>
%computed = gpu.launch_func async [%copied_in] @kernels::@compute
  blocks in (%grid, %one, %one) threads in (%block, %one, %one)
  args(%device_x : memref<?xf32>, %device_y : memref<?xf32>)
%copied_out = gpu.memcpy async [%computed] %host_y, %device_y
  : memref<?xf32>, memref<?xf32>
gpu.wait [%copied_out]
```

这是核心依赖片段，假设参数、kernel 和设备存储已经准备好。`gpu.memcpy` 的操作数顺序是目标、来源。

`!gpu.async.token` 表示前一项异步工作完成的依赖，不装载计算结果。`%computed` 在编译器的数据流中已经存在，不意味着运行时 kernel 已经算完。它让下一条 memcpy 知道必须等什么。

最后没有 `async` 的 `gpu.wait` 是 host 可观察的等待：返回之后，此链中相关的内存操作已经完成，host 才能安全读取 `%host_y`。带 `async` 的 wait 则产生新的依赖 token，不等于立即阻塞 host。

## 5. 多个依赖与资源生命周期

如果一项计算需要两份独立准备的输入，可以用 token 合流表示必须同时等待：

```text
%ready = gpu.wait async [%left_ready, %right_ready]
// 后续计算依赖 %ready。
```

这表达的是前置关系。是否真的并行执行两次复制，还取决于运行时 streams/events、内存类型和设备资源。若把每次复制、计算、再复制全部串在同一条依赖链中，写了 `async` 也不会自动得到重叠。

释放存储同样属于依赖图。最后一个消费者完成前不能释放或复用缓冲区。完整实验的顺序为：

```text
分配 device_x → 分配 device_y → H2D
    → kernel → D2H → 释放 device_x → 释放 device_y → host wait
```

这个保守顺序便于首次核对所有权。后续可以缩短生命周期，但必须分别证明哪项工作已经不再使用每个 buffer。

## 6. 从抽象 Token 到运行时

固定版本的 GPU-to-LLVM host 路径将这些操作接到 `mgpuStreamCreate`、`mgpuMemcpy`、`mgpuStreamSynchronize` 等 wrapper。CUDA wrapper 再调用驱动 API。

因此 token 不是必须存在于目标程序里的一个统一“token 对象”；lowering 根据依赖结构把它落实为 stream/event 等表示。也不能从 IR 的 async 关键字推断每一个 runtime API 都无阻塞：例如该版本 CUDA 分配/释放 wrapper 的具体实现与异步 memcpy 并不相同。

同理，`gpu.barrier` 在设备侧可转成 NVVM barrier，而 host wait 进入运行时同步调用。它们分别约束 kernel 内部参与者和 host 提交的工作链；看见二者都叫“同步”，不能在优化时互相替换。

## 7. 阅读检查与衔接

1. 四个线程写不同位置，为何读取邻居时仍需要 barrier？
2. 尾块里把 barrier 放到 `if valid` 内，需要额外证明什么？
3. token 已经作为 SSA Value 产生，为什么 host 仍不能立即读取设备结果？
4. 两个异步动作之间有依赖，是否还可能同时执行？怎样判断希望的重叠是否被依赖禁止？

后续在[目标后端模块](../../compiler/targets/)继续双缓冲、Host/Device 编译及实际目标约束。本章依据 [GPUOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td)、[GPU host lowering](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp) 与 [CUDA runtime wrapper](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp)，区分依赖语义、编译结果和需要硬件测量的实际重叠。
