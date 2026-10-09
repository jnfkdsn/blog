---
order: 40
title: Host 与 Device 编译及启动
updated: 2026-10-04
---

# Host 与 Device 编译及启动

设备 kernel 描述每个执行实例怎样计算，但应用程序还需要把数据放到可访问的存储、加载设备代码、打包参数并启动执行。生成一段正确的 GPU IR，只完成了这条链的一部分。

本章把长度可变的 `y[i]=2×(x[i]+1)` 接到 host 的输入输出。沿同一程序看两条编译路径怎样分开，又怎样在运行时重新连接。

## 1. 完整运行需要两部分代码

先不考虑文件格式，程序需要完成：

```text
Host：准备输入 → 设备分配 → H2D → 启动 → D2H → 等待 → 读取输出
                                    │
Device：                            └→ 每个实例计算自己的 i，检查边界，读算写
```

其中 H2D/D2H 分别表示 host 到 device、device 到 host 的内容传输。若平台使用其他共享访问机制，步骤可能改变；本章用显式复制建立可核对的边界。

Host 与 device 不使用同一套目标机器指令。Host 代码要在 CPU 上运行，device kernel 要符合所选设备架构；二者通过入口名字、启动尺寸、参数 ABI 和运行时连接。

## 2. 动态长度的资源与依赖

读取输入长度 N 后，本例只在 N>0 时启动工作。先注册 host 缓冲区供所选 CUDA 复制路径使用，再分配两个 device buffer；它们都连续、等长、不重叠。

核心依赖可读为：

```text
%begin = gpu.wait async
%dx, %alloc_x = gpu.alloc async [%begin] (%n) : memref<?xf32>
%dy, %alloc_y = gpu.alloc async [%alloc_x] (%n) : memref<?xf32>
%copy_in = gpu.memcpy async [%alloc_y] %dx, %host_x
  : memref<?xf32>, memref<?xf32>
// kernel 依赖 %copy_in，结果 token 为 %computed。
%copy_out = gpu.memcpy async [%computed] %host_y, %dy
  : memref<?xf32>, memref<?xf32>
%free_x = gpu.dealloc async [%copy_out] %dx : memref<?xf32>
%free_y = gpu.dealloc async [%free_x] %dy : memref<?xf32>
gpu.wait [%free_y]
```

`%dx` 是缓冲区描述符对应的 Value，`%alloc_x` 是异步分配完成的依赖；这两个结果承担不同职责。最后等待结束后才解除 host 注册并返回，调用者随后可以读取输出。

这个例子用一条保守链保证内容和生命周期，重点是正确连接，不宣称复制和 kernel 已重叠。输入为零时不创建零维 launch；host 输入输出的实际有效范围仍由调用者保证。

## 3. 抽离设备 Region

初始代码可以把计算放在 `gpu.launch` Region 内。Outlining 后，host 留下 `gpu.launch_func`，device body 进入 `gpu.module` 里的 kernel。

外层捕获的 N、输入输出以及标量常量成为设备函数参数。一个动态一维 MemRef 不只是数据指针：默认转换还涉及 allocated/aligned 指针、offset、size、stride。Host 打包与 device 参数展开必须相互匹配。

如果使用 bare-pointer 约定，就要满足相应形状和布局限制，并在两侧一致配置。单独把设备签名改成一个指针，而 host 仍传完整描述符字段，会让参数解释错位；这种问题与加法本体是否正确无关。

## 4. 设备侧：GPU 到 NVVM

以 NVIDIA 路径为例，GPU 执行信息被转换到 NVVM/LLVM：

| 原始信息 | 当前路径中的低层形式 |
|---|---|
| `gpu.thread_id x` | `nvvm.read.ptx.sreg.tid.x`，再转换到需要的 index 表示 |
| `gpu.block_id x` | 对应 block ID 的 NVVM 特殊寄存器读取 |
| `gpu.barrier` | `nvvm.barrier0` |
| MemRef 访问 | 描述符字段、指针运算与 LLVM load/store |
| kernel 标记 | 设备入口相关属性和后续翻译信息 |

实际 load/store 仍要保留正确地址空间、对齐与边界。SCF 条件被降低后，尾部谓词仍控制有效访问；类型变低并不允许删除越界保护。

接下来由 NVPTX 后端为选定 chip/features 生成设备表示。PTX 是面向 NVIDIA 的虚拟 ISA 文本；cubin 则包含面向具体架构的设备机器代码。一个文件叫 binary 并不必然意味着里面已经是最终机器码，需要看所选 object format。

## 5. 设备产物与 `gpu.binary`

MLIR 的 GPU 编译路径通过 target attribute 记录如何序列化设备模块。结构上可以看成：

```text
gpu.module @kernels [#nvvm.target<chip = "sm_80", features = "+ptx70">] {
  // 已降低的设备函数
}
           ↓ gpu-module-to-binary
gpu.binary @kernels [设备对象及其目标、格式、内容]
```

这是结构示意，不把省略的对象内容当作可运行示例。一个 binary 可以承载多个目标对象，再由 offloading handler 选择。编译配置应与实际部署架构匹配，不能因为主机安装了某版 CUDA 就推断目标 chip。

对于学习中的编译验证，可以先产生 PTX，检查入口和关键访存/同步，再交给 `ptxas` 检查并形成 cubin。这条过程不需要真的执行 kernel，但它也无法证明运行时数据准备正确、没有竞争或性能符合预期。

四线程轮转在本次 sm_80/+ptx70 编译中产生 `.visible .entry rotate`、16 字节 shared 数组，以及共享写入、`bar.sync 0`、共享读取的顺序。CUDA 12.4 的 ptxas 接受该 PTX，报告使用 12 个寄存器、16 字节 shared memory、无 spill。这是当前编译配置的资源报告；它既不是运行时间，也不说明换一组目标选项仍有相同资源需求。

## 6. Host 侧的两次转换

Host 侧不只是把所有 GPU 操作一次性改成普通函数调用。固定版本的实际处理中，`gpu-to-llvm` 先将分配、复制、等待等改成 runtime wrapper 调用；`gpu.launch_func` 还可以保留，参数则已展开为低层类型。

本例这一阶段可观察到：

```text
llvm.call @mgpuMemHostRegisterMemRef(...)
llvm.call @mgpuStreamCreate(...)
llvm.call @mgpuMemAlloc(...)
llvm.call @mgpuMemcpy(...)
gpu.launch_func ... args(展开后的指针、整数与标量)
llvm.call @mgpuMemcpy(...)
llvm.call @mgpuMemFree(...)
llvm.call @mgpuStreamSynchronize(...)
llvm.call @mgpuMemHostUnregisterMemRef(...)
```

当设备模块已经成为 `gpu.binary`，后续 GPU LLVM translation interface 才能解析启动引用、选择设备对象、嵌入其内容，并生成加载/取函数/启动相关的 LLVM IR 调用。

这也是 Translation 扩展接口的实际用途：支持的非 LLVM 方言操作可以通过自己的翻译接口进入 LLVM IR。这里不是随意让混合 IR 绕过检查，而是 GPU binary 与 launch 有明确的下游消费者和协议。

## 7. 参数数组与运行时加载

在所选默认 offloading handler 中，Host 为每个 kernel 参数准备存储，再形成指向各参数的指针数组。运行时拿到这个数组，按 kernel 的 ABI 传递值。

整个过程可归纳为：

```text
设备对象内容 → mgpuModuleLoad / 相应 JIT 路径
入口名       → mgpuModuleGetFunction
grid/block + 参数数组 + stream → mgpuLaunchKernel
```

CUDA wrapper 再进入驱动 API。Host 目标文件因而需要链接相应运行时和驱动依赖；只有 device 目标文件并不足以执行完整程序。MLIR 的 CUDA wrapper 构建选项及库属于这条运行路径的依赖，普通 CPU 实验不自动具备它们。

如果设备对象是 PTX，驱动还可能在装载时完成 JIT。首次装载和重复调用的成本可能不同；性能测量要区分编译、装载、分配、传输和 kernel 时间，不能把只测 kernel 的数值当作整个应用耗时。

## 8. 按失败边界定位问题

| 现象 | 优先核对的边界 |
|---|---|
| 剩余 GPU/SCF 操作无法转换 | 当前 pipeline 是否覆盖这种表示和控制流 |
| 找不到持有 kernel 的 binary | 是否已序列化设备模块，符号引用是否一致 |
| 序列化失败或目标不被支持 | 后端是否编入、chip/PTX 特性及工具链是否兼容 |
| Host 链接缺 mgpu 符号 | wrapper 库与其运行时依赖是否加入 |
| 装载或启动失败 | 设备访问、目标对象、入口、参数 ABI、启动资源限制 |
| 启动成功但结果错误 | 数据内容、地址与尺寸、尾部条件、竞争、同步、生命周期 |

前几项能在没有硬件时做相当多检查，后两项需要真实运行条件。只打印出一段 NVVM 或 PTX，不能覆盖整张表。

## 9. 阅读检查与衔接

1. 为什么 outlining 完成后，Host 和 Device 还需要分别降低？
2. `gpu-to-llvm` 之后还看到 launch_func，是否必然表示转换失败？它在什么时候被消费？
3. PTX 已经通过汇编检查，为什么还可能在实际启动时报错？
4. 动态 MemRef 的 kernel 参数与一个裸指针参数，调用者需要保持哪项一致性？

下一篇转向[NPU 表示与执行约束](./npu_constraints)，用固定项目说明同样的编译问题怎样采用不同的表示和运行时。

本文已核验 outlining、GPU→NVVM、host wrapper 转换、PTX 序列化、ptxas 生成 cubin，以及 Host LLVM IR/object 编译。当前 PTX 路径的 host.o 留有 `mgpuModuleLoadJIT`、`mgpuModuleGetFunction`、`mgpuLaunchKernel` 等外部符号，需要部署时的运行时链接和设备访问；本环境未执行 GPU 或测量性能。固定实现入口：[GPUToNVVM pipeline](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp)、[GPU translation](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.cpp)、[对象选择与启动生成](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp)。
