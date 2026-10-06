---
order: 1
title: Tensor 到 Buffer 的表示变化
updated: 2026-10-04
---

# Tensor 到 Buffer 的表示变化

一个长度为 4 的张量，原内容为 `[10,20,30,40]`。我们希望把第一项改为 99。Tensor IR 很容易表达这个计算：产生一个值为 `[99,20,30,40]` 的新张量。

真正执行时却必须作一个选择：能否直接覆盖原数组，还是需要另一片存储？如果程序随后还要读取旧张量第一项，直接覆盖就会把本应得到的 10 变成 99。

Bufferization 要完成的工作，就是**把张量值的计算落实成对 buffer 的访问，同时保留程序对旧值和新值的所有有效观察**。它包含存储选择，也包含操作改写；不是把类型名字从 `tensor` 替换为 `memref` 就结束。

## 1. 同一次更新的两个实现

先忽略具体 API，比较两种实现。下面的箭头表示结果由哪片存储承载：

```text
只需要新值：
  t = [10,20,30,40] ──覆盖第一项──→ [99,20,30,40] ← u

仍需要旧值：
  t → A = [10,20,30,40]
       │ 复制
       ↓
      B = [10,20,30,40] ──覆盖第一项──→ [99,20,30,40] ← u
```

第一种只用一片存储。第二种保留 A，把新结果放到 B；旧值从 A 读，新值从 B 读。

这两种实现都可以符合 Tensor 的值语义，取决于程序会观察哪些内容。Tensor 的“不可变”约束的是值的含义，不要求每次产生新 SSA Value 都在运行时复制整个数组。只要旧内容不再被需要，编译器就有机会复用它的存储。

## 2. 连接已有 Buffer 与 Tensor 计算

为了只观察这一项选择，我们让函数接收一个已有 buffer。通过 `bufferization.to_tensor`，将它接入 Tensor 计算：

<!-- memory-example: reuse -->
```text
module {
  func.func @reuse(%m: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %v = arith.constant 99 : i32
    %t = bufferization.to_tensor %m restrict writable : memref<4xi32> to tensor<4xi32>
    %u = tensor.insert %v into %t[%c0] : tensor<4xi32>
    %new = tensor.extract %u[%c0] : tensor<4xi32>
    return %new : i32
  }
}
```

这段函数的输入存储可以是 `[10,20,30,40]`。`tensor.insert` 产生 `%u`，后面的 extract 只读取 `%u[0]`，不再读取 `%t` 的旧内容。

边界上的两个属性分别给出必要前提：

- `writable` 允许 bufferization 生成对这片存储的写入。
- `restrict` 承诺该结果是 Tensor IR 访问这片 memref 存储及其别名的唯一入口，使分析能沿这一条 tensor use-def 关系追踪访问。

这些是调用与构造 IR 时需要保证的契约，不是运行时检查，也不是“加上就自动消除别名”的优化开关。特别是，不应为同一片存储构造两个彼此别名的 `to_tensor restrict` 入口。`restrict` 也不表达释放责任。

示例没有夹杂其他通过 memref 对输入的访问；外部调用者还需要接受 `writable` 所允许的输入修改。接入更复杂的混合程序时，必须一致地维护 Tensor 边界及 MemRef 侧访问的语义。

对完整模块执行：

```bash
mlir-opt reuse.mlir --one-shot-bufferize --verify-each
```

得到如下实际输出：

<!-- memory-output: reuse-buffer -->
```text
module {
  func.func @reuse(%arg0: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c99_i32 = arith.constant 99 : i32
    memref.store %c99_i32, %arg0[%c0] : memref<4xi32>
    %0 = memref.load %arg0[%c0] : memref<4xi32>
    return %0 : i32
  }
}
```

原来的 tensor 值和边界操作已经消失，更新成为对输入 buffer 的 store，取值成为 load。函数返回 99，输入存储的第一项也变成 99。没有新增分配，是因为这个具体程序不再需要保留旧内容，并且边界允许写入。

## 3. 旧值使用引入复制

现在只改变计算的观察方式：更新之后，既读取旧张量第一项，又读取新张量的第一项和第二项。

<!-- memory-example: preserve -->
```text
module {
  func.func @preserve(%m: memref<4xi32>) -> (i32, i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 99 : i32
    %t = bufferization.to_tensor %m restrict writable : memref<4xi32> to tensor<4xi32>
    %u = tensor.insert %v into %t[%c0] : tensor<4xi32>
    %old = tensor.extract %t[%c0] : tensor<4xi32>
    %new = tensor.extract %u[%c0] : tensor<4xi32>
    %unchanged = tensor.extract %u[%c1] : tensor<4xi32>
    return %old, %new, %unchanged : i32, i32, i32
  }
}
```

输入仍为 `[10,20,30,40]`，正确结果必须是 `(10,99,20)`。若仍然直接写输入 buffer，`%old` 会错误地读到 99。

One-Shot Bufferize 的实际输出变成：

<!-- memory-output: preserve-buffer -->
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
    return %0, %1, %2 : i32, i32, i32
  }
}
```

可以沿每条数据流检查这次改写：

1. `%alloc` 为新结果准备存储。
2. `memref.copy` 保留全部初始内容。
3. store 只改变新存储的第一项。
4. 旧值 load 继续从 `%arg0` 读取，新值的两个 load 改为读取 `%alloc`。

复制的原因不只是“需要一个新数组”。`tensor.insert` 只更新一个位置，其余位置仍需继承旧张量内容，因此新存储还必须得到那些内容。返回新张量第二项得到 20，正好检查了这部分语义。

输出中的对齐属性来自本次工具配置；它不决定这个例子为什么必须保护旧值。此处也还没有 dealloc：**One-Shot Bufferize 本身不负责释放**，生命周期将在本模块第三章接续处理。

## 4. 表示转换与计算展开

这次转换没有把“插入一个元素”变成完全不同的算法，而是为已有语义选定了存储实现。同样，Linalg 的 tensor 形式可以转成 memref 形式，同时保留计算结构：

```text
%r = linalg.fill ins(%v : i32)
    outs(%t : tensor<4xi32>) -> tensor<4xi32>

       ↓ 选定 %r 对应的 buffer 后

linalg.fill ins(%v : i32) outs(%buffer : memref<4xi32>)
```

第二种形式不再返回张量值，因为结果通过内存写入表达。至于 fill 如何展开为循环、循环如何降低为控制流，再由后续 passes 处理。

这也连接前面学习的 Dialect Conversion：转换框架讨论“目标接受哪些形式、怎样替换操作与类型”；Bufferization 则有额外的领域问题——怎样基于张量数据流作安全的复用决定。不能仅给 TypeConverter 添加一条 tensor-to-memref 映射，就替代这部分分析。

## 5. 从正确实现到存储决策

本章用两个具体结果建立了判断标准：新旧值都必须读对，复用只是满足语义的实现选择。下一步才适合深入分析的依据：

```text
知道操作读什么、写什么、结果可能共享谁的存储
                      ↓
检查候选写入会不会破坏仍然需要的读取
                      ↓
决定复用或另行分配，必要时复制
                      ↓
按照决定改写成 buffer 操作
```

[One-Shot Bufferization 的决策](./one_shot)将沿同一例子展开。新分配不一定需要复制；没有旧值读取，也不保证输入一定允许写入。这两种条件都可以通过改变当前例子逐一看清。

## 阅读自查

1. `%t` 和 `%u` 是两个不同 Tensor Value，为什么仍有可能使用同一片物理存储？
2. preserve 例子为什么同时需要 alloc 和 copy？分别去掉哪一步会破坏什么？
3. `to_tensor` 上只有 `restrict`，没有 `writable`，是否足以允许覆盖输入？

## 资料与实践

本文输入和输出按 LLVM 20.1.8 验证；两条路径均有本机数值观察。固定版本说明见[Bufferization](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Bufferization.md)与[`to_tensor` 操作契约](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Bufferization/IR/BufferizationOps.td)。[在线文档](https://mlir.llvm.org/docs/Bufferization/)可能使用更新的名称与语法，查接口时应对照版本。

可选实践：[复用与保留旧值实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/10-memory-bufferization)。
