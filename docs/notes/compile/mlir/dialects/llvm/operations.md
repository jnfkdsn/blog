---
order: 1
title: LLVM 类型、指针与低层操作
updated: 2026-10-04
---

# LLVM 类型、指针与低层操作

前面的 MemRef 可以用 `%m[%i,%j]` 读取二维数组。CPU 执行时，指令接收的却是某个地址，而不是带 shape 的二维数组。降低到 LLVM Dialect，就是逐步把这种高层信息落实成地址计算、标量操作、跳转和调用。

先从更小的任务进入：给定一个指向两个 `i32` 的指针，读取这两个整数并求和。这样可以先认识低层操作，再把它们放回矩阵访问的完整过程。

## 1. 地址计算与数值计算

下面是一个完整 LLVM Dialect 模块：

<!-- cpu-example: pointer -->
```text
module {
  llvm.func @sum_pair(%p: !llvm.ptr) -> i32 {
    %q = llvm.getelementptr %p[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %x = llvm.load %p : !llvm.ptr -> i32
    %y = llvm.load %q : !llvm.ptr -> i32
    %s = llvm.add %x, %y : i32
    llvm.return %s : i32
  }
}
```

设 `%p` 指向连续的 `[3,-2]`，执行过程如下：

| 操作 | 得到什么 | 本例含义 |
|---|---|---|
| `llvm.getelementptr` | 一个新地址 `%q` | 指向第 1 个 `i32` |
| 第一次 `llvm.load` | 标量 `%x` | 读出 3 |
| 第二次 `llvm.load` | 标量 `%y` | 读出 -2 |
| `llvm.add` | 标量 `%s` | 计算 1 |
| `llvm.return` | 函数结果 | 返回 1 |

GEP 是 getelementptr 的常用缩写。它只计算地址，不读取该地址的内容。要得到整数，仍要执行 load。这个区分和 MemRef 的“视图与内容”一致，只是现在不再由 MemRef 类型统一携带访问结构。

GEP 末尾的 `i32` 指定用于计算元素偏移的类型。在常见 CPU 布局中，向后移动 1 个 `i32` 等于移动 4 字节。`[1]` 不是任意地址加 1 字节；元素大小需要结合目标数据布局理解。

## 2. Opaque Pointer 与访问类型

本系列使用的 `!llvm.ptr` 是 opaque pointer：指针类型本身不写“指向 i32”或“指向 f32”。操作在需要的地方携带这部分信息：

```text
llvm.getelementptr ... , i32     // 用 i32 的布局计算偏移
llvm.load %p : !llvm.ptr -> i32  // 读取一个 i32
llvm.store %v, %p : i32, !llvm.ptr
```

这不意味着任何指针都可以随意读取任意类型。地址必须在可访问的存储范围内，访问满足相应的布局、对齐与生命周期前提；例如一个已经释放的指针，即使它的类型正确，也不能用于正常读取。

因此，LLVM Dialect verifier 能检查操作的结构与类型关系，却通常不能仅凭这一段函数证明调用方真的提供了两个可读整数。与普通 C 接口一样，这部分属于调用契约。

## 3. 高层信息在哪里消失

比较两种读取的局部形式：

```text
%v = memref.load %m[%i, %j] : memref<?x?xi32, strided<[?, ?], offset: ?>>

                 ↓ 取得布局信息并计算地址

%p = llvm.getelementptr %base[%linear] : (!llvm.ptr, i64) -> !llvm.ptr, i32
%v = llvm.load %p : !llvm.ptr -> i32
```

前者仍然知道访问对象有两个逻辑维度。后者只知道从 `%base` 移动 `%linear` 个 `i32` 再读取。这里的 `%linear` 必须由原来的 offset、indices 和 strides 计算出来，不能随意丢弃它们。

降低改变的是信息的组织方式：shape 和 stride 可以成为低层结构体字段，维度访问成为整数计算，复杂计算成为循环和调用。某些静态已知字段则可以直接折叠成常量。

这也是为什么通常在较高层进行 tiling 等变换：当“这是二维矩阵计算”仍被直接表达时，操作变换容易取得结构信息。到低层再恢复它往往更困难。具体阶段安排由编译器 pipeline 决定。

## 4. 整数、浮点与低层语义

`i32` 和 `f32` 等兼容的内建类型仍可以出现在 LLVM Dialect 中。类型降低不是强制把所有类型改成带 `!llvm` 前缀的名字。

对本例，`llvm.add` 表达整数加法；浮点加法使用 `llvm.fadd`。即使两个操作都看起来像加号，也不能把整数回绕、浮点舍入和附加的优化假设混为一谈。

特别是，溢出相关标记、fast-math 属性、GEP 的 inbounds 等都是有语义的条件。它们会给优化器额外假设，不是单纯的性能开关。转换必须保持原程序已经提供的保证，不能为了生成更漂亮的低层 IR 擅自加强前提。

## 5. LLVM Dialect 仍然是 MLIR

`llvm.func` 仍是 Operation，函数体仍由 Region 和 Block 组成，结果依然遵循 SSA。Pass 可以通过前面学习的 IR API 和 Pattern 修改它。

它与 LLVM IR 的关系是：用 MLIR 的对象模型表达接近 LLVM 指令语义的操作。它还没有变成 LLVM 自己的 `llvm::Module` / `llvm::Instruction` 对象，也没有成为机器码。

下一步有两个问题需要分别解决：[数据布局与描述符](./data_layout)解释高层访问信息如何进入低层类型；[LLVM Dialect 与 LLVM IR](./translation)解释对象表示怎样被翻译到 LLVM 的世界。

## 阅读自查

1. GEP 指向第二个元素之后，为什么还需要一次 load？
2. `!llvm.ptr` 没写元素类型，元素大小由哪一处信息参与确定？
3. MemRef load 降成地址加 load 时，原来的 stride 能否直接删除？

## 资料与实践

语法和行为按 LLVM 20.1.8 核验；固定版本[LLVM Dialect 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/LLVM.md)、[LLVM 操作定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td)用于确认具体契约。

可选实践：[11-cpu-lowering](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/11-cpu-lowering)，`pointer.mlir` 已通过 C 调用验证 `[3,-2] → 1`。
