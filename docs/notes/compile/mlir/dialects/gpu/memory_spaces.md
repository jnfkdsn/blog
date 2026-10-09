---
order: 20
title: 地址空间与设备存储
updated: 2026-10-04
---

# 地址空间与设备存储

上一章各个 thread 独立读写自己的元素。如果一个 thread 要使用另一个 thread 读入的数据，问题就变成了：这份数据放在哪里、谁能看到、什么时候可以使用？

用一个四项轮转说明这件事。输入为 `[10,20,30,40]`，四个线程各读一项，再让线程 t 输出线程 `(t+1)%4` 读入的值。期望输出是 `[20,30,40,10]`。真实轮转未必值得专门用共享存储实现，这里用它观察跨线程通信的存储契约。

## 1. 三种可见范围

首先为数据划定归属：

```text
设备输入 x、输出 y：所有参与者都可按约定访问
中间 tile：同一 block 的线程共同使用
局部 scratch：每个 thread 各自一份
```

固定版本 GPU 地址空间属性分别表示 `global`、`workgroup`、`private`。在 MemRef 类型里，形状/布局描述坐标怎样对应地址，memory space 则描述这种存储属于什么空间。

```text
memref<4xf32, #gpu.address_space<global>>
memref<4xf32, #gpu.address_space<workgroup>>
memref<1xf32, #gpu.address_space<private>>
```

空间不是所有权证明。一个 global 参数仍需调用者保证实际分配、有效范围和生命周期；一个 private 临时也不意味着一定放进物理寄存器。寄存器分配与 spill 是目标实现的另一层选择。

## 2. Workgroup 存储的产生

Kernel 可以在函数级声明中间存储：

```text
gpu.func @rotate(%x: memref<4xf32, #gpu.address_space<global>>,
                 %y: memref<4xf32, #gpu.address_space<global>>)
  workgroup(%tile: memref<4xf32, #gpu.address_space<workgroup>>)
  private(%scratch: memref<1xf32, #gpu.address_space<private>>) kernel {
  // ...
}
```

`%tile` 不是 host 传进来的第三个参数，而是 workgroup memory attribution。每个 block 有自己的四项 tile；同一 block 的线程看到同一份存储，不同 block 不能通过这个名字共享数据。

`%scratch` 则按线程私有。四个线程都访问 `scratch[0]`，访问的是各自的局部存储，不是四个线程争用同一个元素。本例不需要使用它，保留声明仅用于核对 private attribution 的降低；实际优化可以删除未使用存储。

这种声明把存储归属与 kernel 结构绑定，不能把 tile 的内容作为跨 kernel 调用保存状态的手段。跨调用保留的数据应放到生命周期足够长的存储，通过参数传递。

## 3. 从各自读取到交换数据

轮转的 kernel body 是：

```text
%t = gpu.thread_id x
%c1 = arith.constant 1 : index
%c4 = arith.constant 4 : index
%v = memref.load %x[%t] : memref<4xf32, #gpu.address_space<global>>
memref.store %v, %tile[%t] : memref<4xf32, #gpu.address_space<workgroup>>
gpu.barrier
%next = arith.addi %t, %c1 : index
%j = arith.remui %next, %c4 : index
%w = memref.load %tile[%j] : memref<4xf32, #gpu.address_space<workgroup>>
memref.store %w, %y[%t] : memref<4xf32, #gpu.address_space<global>>
gpu.return
```

此 kernel 的启动约定是一个 block、x 维四个线程、其他维度为 1。第一阶段线程 t 填 `tile[t]`；第二阶段读 `tile[(t+1)%4]`。每个写入位置唯一，但读取依赖另一个线程完成写入，所以二者之间需要同步。

共享空间只回答“能否访问同一份数据”，不回答“对方是否已经写好”。如果去掉 barrier，线程 0 可以在线程 1 写入之前读取 `tile[1]`。下一篇会把存储通信和时间顺序合起来分析。

## 4. 地址空间进入目标类型

将本例转换为 NVVM/LLVM 后，可以实际看到：

```text
global 输入/输出参数 → !llvm.ptr<1>
workgroup tile       → addr_space = 3 的 LLVM global
                         在 kernel 内得到 !llvm.ptr<3>
private attribution  → kernel 内的 llvm.alloca
```

这里的 LLVM global 是目标代码中的存储声明，不能因为名字是 global 就把它误认成前面 GPU 的 global 空间；决定空间的是 `addr_space=3`。在所选 NVVM 路径中它承载 workgroup/shared 存储。

同样，private attribution 在这条实现里先成为 alloca，不能机械写成“一定是 `!llvm.ptr<5>`”。具体转换会结合目标和 alloca 的规则处理。地址空间数字属于目标约定；GPU 属性帮助前面阶段保留抽象含义。

转换后的 kernel 参数还会包含 MemRef 的指针、偏移、尺寸和步长字段。设备函数同样存在 ABI：只有在源和调用方一致采用更受限的约定时，才可以简化成 bare pointer。

## 5. 类型、分配与搬运

把一个普通 host MemRef 的类型写成 global，不会自动完成设备分配或复制。这里需要三种动作：

| 动作 | 改变的对象 |
|---|---|
| 分配 | 创建实际存储并建立描述符/句柄 |
| 搬运 | 将内容从一个存储位置传到另一个位置 |
| 类型或地址空间转换 | 改变编译器对地址的表示及访问规则 |

例如 `gpu.alloc` 返回可供后续设备操作使用的存储，`gpu.memcpy %dst, %src` 表达复制，释放则由 `gpu.dealloc` 或选定运行时协议安排。地址空间 cast 不能替代 memcpy。

设备页映射、统一内存、host 注册等还会改变 host/device 的可访问关系，但都需要对应平台机制。正文先使用显式分配和复制形成清楚边界，再在项目里判断哪些步骤可以省略。

## 6. 阅读检查与衔接

1. 两个 block 各有一个 `%tile`，是否能用它传递 block 0 的结果给 block 1？
2. 四个 thread 都访问 private 的位置 0，为什么不等于访问同一地址？
3. `addr_space=3` 的 LLVM global 为什么在本例中代表共享存储，而非 GPU global memory？

接着读[同步与异步操作](./synchronization)。更复杂的连续访问、共享存储布局与填充，在[目标后端模块](../../compiler/targets/)结合二维小块分析。

本文按 LLVM 20.1.8 核对三种 GPU 地址空间及实际 NVVM 转换；使用当前在线文档时需注意新版已扩展的空间枚举。依据：[GPU 设计说明](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/GPU.md)、[GPUToNVVM](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp)。轮转结果是语义推演，设备访问在当前环境尚未执行验证。
