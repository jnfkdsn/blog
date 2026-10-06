---
order: 2
title: DataLayout 与 MemRef 描述符
updated: 2026-10-04
---

# DataLayout 与 MemRef 描述符

一个二维窗口可以由 offset=5、sizes=[2,2]、strides=[4,1] 描述。上一模块已经推导：窗口 `[1,1]` 对应线性元素偏移 `5+1×4+1=10`。

现在把这个函数交给 LLVM 后端，需要回答一个实现问题：窗口带来的这些信息如何成为可以传递、读取和计算的低层值？答案是给它们约定具体表示，常见形式就是 MemRef descriptor。

## 1. 将一个视图拆成字段

对本章使用的 64 位 CPU 配置，二维 ranked strided memref 的默认描述符为：

```text
!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
```

各字段的职责如下：

| 字段 | 窗口示例中的值 | 使用它的阶段 |
|---|---|---|
| allocated pointer | 原分配的起点 | 释放那次分配 |
| aligned pointer | 用于访问的基址 | 计算元素地址 |
| offset | 5 | 从访问基址到视图第一项 |
| sizes | `[2,2]` | 维度查询与循环范围 |
| strides | `[4,1]` | 将逻辑坐标变成元素偏移 |

为什么需要两个指针？为了满足对齐要求，分配得到的起点和实际用于访问的基址可能不同。释放时需要匹配原始分配；读取时则使用访问基址加偏移。把两者都写成同一个值适用于很多普通数组，但不是通用要求。

描述符自身只是包含地址与元数据的对象。复制一个描述符会复制这些字段，不会复制它指向的整个矩阵。

## 2. 从字段恢复一次访问

下面这个完整函数接受一般二维 strided 视图，读取 `[i,j]`：

<!-- cpu-example: read2d -->
```text
module {
  func.func @read2d(%m: memref<?x?xi32, strided<[?, ?], offset: ?>>,
                    %i: index, %j: index) -> i32 attributes {llvm.emit_c_interface} {
    %v = memref.load %m[%i, %j] : memref<?x?xi32, strided<[?, ?], offset: ?>>
    return %v : i32
  }
}
```

它的 lowering 在结构上完成以下过程。为突出字段含义，下面使用语义变量名表示实际 IR 中的取字段结果：

```text
base       = descriptor.aligned
offset     = descriptor.offset
rowStride  = descriptor.strides[0]
colStride  = descriptor.strides[1]
linear     = offset + i * rowStride + j * colStride
address    = GEP(base, linear, elementType=i32)
result     = load i32 from address
```

传入前面的窗口和 `[1,1]` 时，先得到元素偏移 10，再使用目标的 i32 布局形成字节地址。实际 lowering 可能先为 offset 做一次 GEP，再对两个 stride 乘积之和做第二次 GEP；这与一次合并计算表达相同的访问。

`sizes` 没有出现在这次纯地址公式里，但仍有用途：它为维度查询、循环边界和调用约定保留形状。普通 load 不自动把 sizes 变成运行时范围检查，调用者仍要提供合法坐标。

## 3. 静态类型与运行时描述符

若类型写成 `memref<2x3xf32>`，维度大小和默认步长在编译期已知。默认描述符形式仍包含 sizes/strides，外部构造它时应填入与类型一致的数值；内部 lowering 可以根据静态类型直接使用常量。

这一点对 C 接入尤其重要：不能把 declared type 为连续 `2×3` 的函数，当作能够任意传入 `[4,1]` 步长窗口的函数。即使自己在描述符里填了 4，转换后的函数也可能已按静态行步长 3 生成地址计算。

要接受一般窗口，应在 MLIR 函数签名中保留动态 strided 布局，例如本章 `read2d` 的类型。接口需要表达的自由度，应在降低前就明确。

## 4. Index 位宽由目标约定决定

尺寸、步长、偏移和索引都使用 `index` 的降低类型。本文的默认 CPU 路径使用 i64，但 `index` 本身不是“永远 64 位”的别名。

用一个简单索引递增函数观察：

<!-- cpu-example: index -->
```text
module {
  func.func @advance(%i: index) -> index {
    %one = arith.constant 1 : index
    %next = arith.addi %i, %one : index
    return %next : index
  }
}
```

在本机默认配置下，转换得到 `@advance(i64) -> i64`。若对参与转换的 arith 与 func passes 一致指定 32 位，则得到：

<!-- cpu-output: index32 -->
```text
module {
  llvm.func @advance(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(1 : index) : i32
    %1 = llvm.add %arg0, %0 : i32
    llvm.return %1 : i32
  }
}
```

位宽改变后，可表示范围、溢出行为和边界 ABI 都要一起检查。只把函数参数改成 i32，而访问或描述符仍按 i64 生成，会让不同部分对同一值采用不一致的协议。

这里展示的是显式转换配置的对照。完整编译器通常从目标 DataLayout 和统一的类型转换配置推导这些选择，不由各个 Pattern 任意决定。

## 5. DataLayout 与目标信息

DataLayout 描述类型的大小、对齐、指针表示等目标相关事实；目标 triple 则标识体系结构、系统和环境等目标信息。二者共同影响后端处理，但都不等价于具体的优化调度。

对本章的 GEP，DataLayout 参与确定 i32 元素的大小。对 C 结构体，指针宽度、整数宽度和字段对齐决定字段偏移。如果一侧按 32 位指针布局结构体，另一侧按 64 位布局读取，就算字段名称和顺序一致，也不能正确互通。

MLIR 中可以通过相应布局接口与属性描述目标信息，LLVM Dialect 模块也支持 LLVM 的 data layout 和 target triple 属性。将这些配置接入转换与最终代码生成，是编译器的责任；只给某个模块写上“另一个目标”的字符串，不会自动重建整个匹配的工具链和运行时。

## 6. 默认描述符的适用范围

这里讲的是默认 ranked strided memref 表示。一般布局需要先整理成后续 lowering 支持的形式。Unranked memref 则额外传递 rank 和指向 ranked descriptor 的指针，因而还需要考虑描述符对象自身的生命周期。

Bare-pointer 调用约定可以在受限条件下省去部分描述符传参，但调用双方必须能够恢复所需元数据；它不自动适用于任意动态形状和动态布局。具体限制应随所选后端和接口核验。

掌握字段含义后，就可以理解[函数 ABI](../../compiler/lowering/abi)为什么有时把描述符拆成多个参数，有时又额外生成一个接收结构体指针的 wrapper。

## 阅读自查

1. allocated pointer 与 aligned pointer 不同的时候，load 和 free 分别应该使用谁？
2. 为什么不能把静态连续 MemRef 函数仅通过填一个不同 stride，就变成一般窗口函数？
3. 若将 index 配置成 32 位，除了循环变量，还需要检查哪些边界信息？

## 资料与实践

示例基于 LLVM 20.1.8、本机 64 位 CPU；32 位 index 对照验证转换结构，不作为跨目标执行测试。依据见[LLVM IR Target](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/TargetLLVMIR.md)、[DataLayout](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DataLayout.md)与[MemRef 描述符实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/LLVMCommon/MemRefBuilder.cpp)。

可选实践：[描述符与非连续窗口的 C 调用](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/11-cpu-lowering)。
