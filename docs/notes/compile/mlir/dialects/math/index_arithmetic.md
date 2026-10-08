---
order: 2
title: Index 运算与尺寸计算
updated: 2026-10-04
---

# Index 运算与尺寸计算

把长度为 10 的一行按每块 4 个元素处理，需要 3 块，最后一块只有 2 个有效元素。算法本身很简单，但编译器需要把块数、起点、边界和地址尺寸表示为可执行计算，还要保证这些计算在选定目标的位宽下成立。

本章沿这个分块边界解释 Index 方言。它操作的是已经见过的 builtin `index` 类型；“类型”和“提供该类型运算的方言”是不同层次。以前用 Arith 对 index 做加法并没有错，Index 方言进一步为这类目标相关整数计算提供专门操作与折叠约定。

## 1. 块数、起点与余块

设长度为 N、块大小为 T，当前讨论 `N≥0,T>0`。所需块数是向上取整的 `N/T`：

```text
N=10，T=4
块编号       0       1       2
起点         0       4       8
有效长度     4       4       2
覆盖范围     [0,4)   [4,8)   [8,10)
```

块 k 的起点为 `k*T`，有效长度为 `min(T,N-k*T)`；这些公式的使用条件是 k 为有效块编号，且相关乘法没有越过所采用的可表示范围。它们产生的结果可以用于切片、循环上界或 vector mask。

当 N=0 时，块数为 0，没有最后一块。对长度恰好为 12 的情况，块数为 3，最后一块仍有 4 个元素。边界逻辑应从这些条件自然得到，不额外假设每次都存在余块。

## 2. 向上整除与溢出边界

常见公式 `(N+T-1)/T` 在无限精度非负整数上成立，但固定宽度的 `N+T-1` 可能溢出。Index 方言可以直接保留“无符号向上整除”的意图：

```text
%count = index.ceildivu %n, %tile
```

`ceildivu` 的输入和结果都是 index，所以语法省略了重复类型。除数仍必须非零；类型检查不会替运行时调用方保证 `%tile>0`。

固定版本的降低利用类似下面的等价分解处理非零 N，并对 N=0 选择结果 0：

```text
N > 0 时：1 + (N-1)/T
N = 0 时：0
```

这样的分解避免了先做 `N+T-1`。实际 lowering 中可能先计算多个 SSA 中间值再 select；因此还要按低层整数与除法语义检查，例如不能用 select 遮盖一个零除数。

完整观察模块还包含下一节的位宽例子：

<!-- softmax-example: index -->
```text
module {
  func.func @tiles(%n: index, %tile: index) -> index attributes {llvm.emit_c_interface} {
    %count = index.ceildivu %n, %tile
    return %count : index
  }
  func.func @width_sensitive() -> index attributes {llvm.emit_c_interface} {
    %large = index.constant 4294967298
    %two = index.constant 2
    %r = index.divu %large, %two
    return %r : index
  }
}
```

在 64 位宿主上的实际 tiles 结果为 `(0,4)→0`、`(10,4)→3`、`(12,4)→3`。实验还用无符号上界验证算术实现，采用精确整数比较；那是运算边界测试，不表示真实系统能分配如此大的张量。

## 3. Index 位宽与折叠

`index` 用来表达目标相关的尺寸和索引，不能在尚未确定目标时简单视为 i64。观察 `4294967298 / 2`：

| 最终 index 位宽 | 降低后的操作数 | 无符号除法结果 |
|---|---|---|
| 64 位 | 4294967298 与 2 | 2147483649 |
| 32 位 | 大常量截断为 2，除数仍为 2 | 1 |

如果在不知道目标位宽时先按 64 位折叠成 2147483649，再把这个结果截成 32 位，得到的仍是 2147483649，与真正 32 位计算得到的 1 不同。

因此，本章固定版本的 `canonicalize` 保留该 `index.divu`，没有把它统一替换为一个常量。指定 `index-bitwidth=32` 降低后，关键操作变为：

```text
%0 = llvm.mlir.constant(2 : i32) : i32
%1 = llvm.udiv %0, %0 : i32
```

64 位降低则保留 4294967298 与 2。这里展示的是转换后的输入与计算；32 位案例只做 IR 转换检查，64 位案例另有本机执行结果。

Index 方言的折叠会检查结果是否与其支持的目标位宽解释一致。加法低位通常满足“先算后截断”和“先截断再算”一致，右移、除法等操作则更容易受到高位影响。这解释了为什么某些看似全常量的表达式仍保留在高层 IR 中。

## 4. 有符号解释与类型转换

`index` 本身是 signless，某次运算采用有符号还是无符号解释，由操作决定。非负尺寸常用无符号除法；偏移或差值可能为负，需要按实际语义选择。例如 `divs` 向零舍入，`floordivs` 向负无穷舍入，`ceildivs` 向正无穷取整，负数时结果可能不同。

把固定宽度整数转成 index 也要明确扩展方式：

```text
%s = index.casts %x : i32 to index
%u = index.castu %x : i32 to index
```

若目标 index 为 64 位、`%x` 的位模式为 `0xffffffff`，前者符号扩展，得到全 1 的 64 位模式；后者零扩展，得到 4294967295。反向窄化会截断，不能自动证明原值适合目标范围。

所以不能把“把 i32 改成 index”当成完整的索引安全方案。应先明确输入是否非负、是否允许负偏移、形状乘积是否溢出、后端位宽以及函数边界的表示，再选择相应操作。若来自外部的尺寸不可信，运行时检查还需要显式表达。

## 5. 从尺寸运算到实际访问

块数为 3 只是调度信息，最后一次 load 能否合法，还取决于使用的是有效长度 2，还是仍然盲目访问 4 个元素。前者可以形成缩短的循环，后者则需要 mask 或其他边界处理。

对 N=10 的最后一块：

```text
base=8
lane=0 → index=8，有效
lane=1 → index=9，有效
lane=2 → index=10，无效
lane=3 → index=11，无效
```

一个比较 `base+lane < N` 可以产生有效位，但编译器必须让无效 lane 的内存访问真正受到保护。先无条件读取再对结果 select，并不能撤销已经发生的越界访问。这个区别会在 Vector transfer/mask 和 GPU 线程边界中再次出现。

尺寸、循环和内存访问因而构成一条连续链：Index 表达边界计算，控制流或 mask 约束实际执行，MemRef/目标布局决定地址。单独把尺寸公式算对，只完成了其中一段。

## 阅读自查

1. 为什么 `(N+T-1)/T` 需要额外考虑位宽，而 `N=0` 还需要单独在等价分解中处理？
2. 上面的常量除法为什么在不知道目标位宽时不能直接折叠？
3. 最后一块的 mask 已经算出，为什么仍需检查 load 本身是否受保护？

## 实现依据与实践

位宽与折叠依据 [IndexDialect.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Index/IR/IndexDialect.td)，操作契约依据 [IndexOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Index/IR/IndexOps.td)，降低依据 [IndexToLLVM.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/IndexToLLVM/IndexToLLVM.cpp)。[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/13-softmax)保留 canonicalize、32/64 位转换和 64 位本机观察；与 ABI 的连接可回查 [DataLayout](../llvm/data_layout)。
