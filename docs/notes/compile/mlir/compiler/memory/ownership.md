---
order: 3
title: 函数边界、所有权与释放
updated: 2026-10-04
---

# 函数边界、所有权与释放

Bufferization 已经让旧值和新值各自读到正确内容，但 preserve 例子多出了一次 `memref.alloc`。如果函数反复执行，而这片临时存储一直不释放，即使每次计算结果都正确，程序仍会持续消耗内存。

本章沿这片存储继续追踪：**谁负责释放它，责任如何跨函数或分支传递，释放应放在哪里。** 这里的 ownership 表示释放责任，而不是“只有一个 SSA Value 能指向它”。许多视图可以共享存储，释放责任仍必须协调为一次正确的释放。

## 1. 局部临时 Buffer 的最后使用

preserve 函数将新张量的内容放到独立 buffer，最后只返回三个标量。既然标量已经读出，临时 buffer 就没有继续存活的必要。

将两阶段连接起来：

```bash
mlir-opt preserve.mlir --one-shot-bufferize \
  --buffer-deallocation-pipeline --verify-each
```

实际输出为：

<!-- memory-output: preserve-dealloc -->
```text
module {
  func.func @preserve(%arg0: memref<4xi32>) -> (i32, i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c99_i32 = arith.constant 99 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
    memref.copy %arg0, %alloc : memref<4xi32> to memref<4xi32>
    memref.store %c99_i32, %alloc[%c0] : memref<4xi32>
    %0 = memref.load %arg0[%c0] : memref<4xi32>
    %1 = memref.load %alloc[%c0] : memref<4xi32>
    %2 = memref.load %alloc[%c1] : memref<4xi32>
    memref.dealloc %alloc : memref<4xi32>
    return %0, %1, %2 : i32, i32, i32
  }
}
```

新增的 dealloc 位于最后一次 load 之后、return 之前。返回值不再引用临时 buffer，所以释放不会破坏它们。输入 `%arg0` 没有被释放：它来自调用方，当前函数只是借用。

对于这种直线代码，最后使用很容易观察。复杂程序中，结果可能沿分支离开当前 Block、作为函数返回值交给调用者，或通过视图保持可达；不能看到一个 alloc 就机械地在同一个函数末尾加 dealloc。

## 2. 函数参数与返回值的责任约定

逐个函数处理时，释放 pass 需要一套共同的函数边界约定，否则它不知道被调用函数是否接管了参数，也不知道返回 buffer 应由谁清理。

本章使用的 ownership-based deallocation 默认约定是：

| 边界行为 | 释放责任 |
|---|---|
| 将 memref 作为参数传入 | 被调用者借用，调用方继续负责 |
| 返回 memref | 责任交给接收返回值的调用方 |
| 返回与某个参数共享底层分配的 memref | 需要消除这种别名返回，必要时产生独立副本 |

这是**这套释放流程采用的 ABI 约定**，不是 `func.func` 类型系统对所有 MLIR 程序的普遍限制。项目可以设计别的协议，但编译器、外部函数和运行时必须一致地遵守。普通 memref 参数没有一个类型位能够自动说明“这次调用已转移所有权”。

先看最直接的责任转移：

<!-- memory-example: call -->
```text
module {
  func.func @make() -> memref<4xi32> {
    %c0 = arith.constant 0 : index
    %v = arith.constant 42 : i32
    %m = memref.alloc() : memref<4xi32>
    memref.store %v, %m[%c0] : memref<4xi32>
    return %m : memref<4xi32>
  }
  func.func @consume() -> i32 {
    %c0 = arith.constant 0 : index
    %m = func.call @make() : () -> memref<4xi32>
    %v = memref.load %m[%c0] : memref<4xi32>
    return %v : i32
  }
}
```

`@make` 分配并初始化第一项，随后返回整个 buffer。`@consume` 接收它，读取第一项，最后仅返回标量。

运行释放 pipeline 后：

<!-- memory-output: call-dealloc -->
```text
module {
  func.func @make() -> memref<4xi32> {
    %c0 = arith.constant 0 : index
    %c42_i32 = arith.constant 42 : i32
    %alloc = memref.alloc() : memref<4xi32>
    memref.store %c42_i32, %alloc[%c0] : memref<4xi32>
    return %alloc : memref<4xi32>
  }
  func.func @consume() -> i32 {
    %c0 = arith.constant 0 : index
    %0 = call @make() : () -> memref<4xi32>
    %1 = memref.load %0[%c0] : memref<4xi32>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %0 : memref<4xi32> -> memref<i32>, index, index, index
    memref.dealloc %base_buffer : memref<i32>
    return %1 : i32
  }
}
```

`@make` 里面没有释放，因为返回之后还要使用；`@consume` 的 load 之后出现释放。这里的 `extract_strided_metadata` 取得用于清理的底层分配信息，与前面学习的访问元数据是同一套 MemRef 表示。

如果外部 C 或运行时函数返回 memref，这套流程会假定外部实现也满足相同约定。不能在不检查运行时协议的情况下，把某个借用视图当作“收到就拥有”的返回值接进来。

## 3. 返回参数为什么可能引入副本

考虑一个看似无需任何工作的函数：

<!-- memory-example: return-alias -->
```text
module {
  func.func @return_alias(%m: memref<4xi32>) -> memref<4xi32> {
    return %m : memref<4xi32>
  }
}
```

作为普通 MemRef IR，它就是返回传入的视图。但放进上述默认所有权协议时，会产生歧义：调用方既保留原参数的释放责任，又按约定取得返回值的释放责任，两者其实指向同一次分配。

释放 pipeline 将它改成：

<!-- memory-output: return-alias-dealloc -->
```text
module {
  func.func @return_alias(%arg0: memref<4xi32>) -> memref<4xi32> {
    %0 = bufferization.clone %arg0 : memref<4xi32> to memref<4xi32>
    return %0 : memref<4xi32>
  }
}
```

`bufferization.clone` 在这里是释放流程插入的边界处理。在本系列使用的后续 `convert-bufferization-to-memref` lowering 中，它会落实为分配和数据复制。返回子视图也不能仅靠“不重叠”绕过这个约定：它依然与参数共享同一个 allocated base。

还需区分中间操作契约与当前 lowering：`bufferization.clone` 的通用契约允许实现共享视图，并规定 clone 后修改其源或结果属于未定义行为。因此，不应把它当作任意可变程序中的“深拷贝 API”。手写一份需要独立修改的副本时，明确使用 `memref.alloc` 与 `memref.copy`；本章通过只读内容和具体 lowering 后的分配关系观察边界处理。

这说明观察到 copy 时，不能一律归因为 Tensor 旧值冲突。**函数的存储所有权协议也会要求复制。** 若目标接口适合由调用方提供输出 buffer，可以另行设计显式输出参数；若返回值与参数等价且调用方式允许，也可能先用专门变换消除冗余返回。选择哪一种取决于对外 ABI，不能只在局部删掉 clone。

## 4. Tensor 函数边界的衔接

前两章刻意使用 memref 参数和标量结果，让读者先集中理解函数内部的存储选择。真实 Tensor 程序也需要转换函数签名。例如：

<!-- memory-example: signature -->
```text
module {
  func.func @updated(%t: tensor<4xi32>) -> tensor<4xi32> {
    %c0 = arith.constant 0 : index
    %v = arith.constant 99 : i32
    %u = tensor.insert %v into %t[%c0] : tensor<4xi32>
    return %u : tensor<4xi32>
  }
}
```

使用以下配置，把函数边界也纳入 bufferization，并在本例中选择 identity layout：

```bash
mlir-opt signature.mlir \
  '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'
```

得到：

<!-- memory-output: signature-buffer -->
```text
module {
  func.func @updated(%arg0: memref<4xi32>) -> memref<4xi32> {
    %c0 = arith.constant 0 : index
    %c99_i32 = arith.constant 99 : i32
    memref.store %c99_i32, %arg0[%c0] : memref<4xi32>
    return %arg0 : memref<4xi32>
  }
}
```

这时函数参数和返回值已变为 memref，内部更新可以原地执行。接下来若使用默认释放协议，返回值与参数的别名关系又需要按上一节处理。**计算可以原地实现，与返回值可以按某种 ABI 无成本地转交，是两个相邻但不同的问题。**

本例固定形状且采用连续布局，方便看清责任变化。更一般的函数边界可能使用动态 offset/strides；过早要求 identity layout 可能限制调用方可传的视图或带来布局转换成本。布局选择不能只当作让打印结果更简洁的选项。

函数调用和定义需要协同转换；递归、外部声明与复杂调用图还受具体版本的函数 bufferization 支持范围约束。后续 CPU lowering 再展开 descriptor 与实际机器 ABI，本章先把 tensor-to-buffer 边界和 ownership 边界区分清楚。

## 5. 分支中的动态所有权

当程序根据条件选择“新分配”或“借用参数”，只跟踪一个 memref 不足以决定要不要释放：

<!-- memory-example: branch -->
```text
module {
  func.func @choose(%m: memref<4xi32>, %useFresh: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %v = arith.constant 77 : i32
    %selected = scf.if %useFresh -> memref<4xi32> {
      %fresh = memref.alloc() : memref<4xi32>
      memref.store %v, %fresh[%c0] : memref<4xi32>
      scf.yield %fresh : memref<4xi32>
    } else {
      scf.yield %m : memref<4xi32>
    }
    %r = memref.load %selected[%c0] : memref<4xi32>
    return %r : i32
  }
}
```

两条路径分别是：

| 条件 | `%selected` 的来源 | 读出的值 | 当前函数是否负责释放 |
|---|---|---|---|
| true | 本函数新分配 | 77 | 是 |
| false | 借用 `%m` | 输入第一项 | 否 |

Ownership pass 在需要时让 memref 与一个 `i1` 责任标志一起经过区域边界。可以把中间关系理解为下列示意，具体自动生成 IR 还包含别名与保留集合处理：

```text
true 分支  → (新 buffer, true)
false 分支 → (借用 buffer, false)
                   ↓
            (selected, owned)
                   ↓
             读取 selected
                   ↓
          owned 为真时释放其分配
```

完整 pipeline 简化后，本例的实际输出是：

<!-- memory-output: branch-dealloc -->
```text
module {
  func.func @choose(%arg0: memref<4xi32>, %arg1: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c77_i32 = arith.constant 77 : i32
    %0 = scf.if %arg1 -> (memref<4xi32>) {
      %alloc = memref.alloc() : memref<4xi32>
      memref.store %c77_i32, %alloc[%c0] : memref<4xi32>
      scf.yield %alloc : memref<4xi32>
    } else {
      scf.yield %arg0 : memref<4xi32>
    }
    %1 = memref.load %0[%c0] : memref<4xi32>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %0 : memref<4xi32> -> memref<i32>, index, index, index
    scf.if %arg1 {
      memref.dealloc %base_buffer : memref<i32>
    }
    return %1 : i32
  }
}
```

释放条件直接化简为原来的分支条件 `%arg1`。选择借用参数时不会释放输入；选择新分配时，先读出 77，再释放它。

这比“函数退出时统一释放所有 memref”多了一条关键信息：当前执行路径是否承担责任。更复杂的合流不能总是静态化简，需要把责任条件保留到运行时。

## 6. 从所有权到具体释放操作

内部的 `bufferization.dealloc` 同时描述待处理的 buffers、相应责任条件，以及必须保留的 buffers。其语义需要避免对同一分配重复释放，也需要避免释放仍由后续值使用的存储。

以 `@make` 为例，分配确实由本函数创建，但返回值属于需要保留并传出的集合；因此不能在返回前释放。以分支为例，责任条件随路径改变。别名关系无法完全静态确定时，还可能需要运行时比较；已知关系则可化简。

这也是使用完整 `buffer-deallocation-pipeline` 的原因：它不仅插入抽象的释放请求，还安排相关规范化、释放简化与 lowering，尽可能消除冗余条件和检查。直接把未经简化的抽象 dealloc 降低，可能得到不必要的运行时开销。

该流程应在需要它管理的 bufferization 完成之后运行。输入一般不应混入已经手工安排好的 `memref.dealloc`；后续 pass 若再创建新的分配，也不能假定前面运行过的释放 pass 会自动回头处理。示例中的手动释放基础实验与自动释放实验因此分别验证。

## 7. 生命周期闭环与后续 Lowering

到这里，同一个张量更新已经有一条完整的存储解释：

```text
旧值与新值的语义
    ↓
复用许可、读写冲突与布局
    ↓
分配 / 复制 / 访存
    ↓
参数借用、返回责任与分支传递
    ↓
在安全位置释放
```

这条路径解决 CPU 程序正确执行所需的内存基础。它还不是性能最优的生命周期安排：分配提升、栈提升、内存池和更早释放可以继续优化；异步设备任务则还需要把完成事件和生命周期联系起来。当前主线接到 [CPU Lowering](../lowering/)，继续解释循环、descriptor、LLVM IR、机器码和运行时如何衔接。

## 阅读自查

1. `@make` 的 alloc 为什么不能在 `@make` 返回前释放？责任最后在哪个函数结束？
2. 参数借用与返回所有权采用上述约定时，直接返回参数为什么需要特别处理？
3. `scf.if` 的两个分支产生相同 memref 类型，为什么仍要追踪额外的责任条件？
4. One-Shot 没有产生 copy，是否足以证明后续完整 pipeline 也不会产生 copy？

## 资料与实践

本文使用 LLVM 20.1.8 的默认所有权协议，未启用 private-function dynamic ownership 扩展。契约和 pipeline 见[固定版本 Ownership-based Buffer Deallocation](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/OwnershipBasedBufferDeallocation.md)，相关接口模型见[BufferDeallocation 源码目录](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/lib/Dialect/Bufferization/Transforms)。

可选实践：[函数返回、借用与分支释放](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/10-memory-bufferization)。实验验证具体 IR 结构和本机数值，不据此声称完成任意程序的内存安全证明或性能测量。
