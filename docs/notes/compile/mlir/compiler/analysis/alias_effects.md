---
order: 2
title: 别名、效果与内存依赖
updated: 2026-10-04
---

# 别名、效果与内存依赖

对整数 SSA 值 `%x`，两次读取这个值不会让它自己改变。对 `%p: memref<1xi32>`，描述符还是同一个 SSA 值，背后的内存内容却可能已经被写过。于是，“两条 load 的 operands 完全相同”不能单独保证它们返回相同结果。

本章围绕一次具体优化展开：能否删除第二次 load，改用第一次读出的值？我们先追踪内存实际怎样改变，再把所需证据拆成地址关系与读写行为，最后说明通用分析能证明到什么程度。

## 1. 两次读取之间的写入

考虑这个完整函数：

```text
func.func @load_store(%p: memref<1xi32>, %q: memref<1xi32>, %v: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %a = memref.load %p[%c0] : memref<1xi32>
  memref.store %v, %q[%c0] : memref<1xi32>
  %b = memref.load %p[%c0] : memref<1xi32>
  %r = arith.addi %a, %b : i32
  return %r : i32
}
```

调用方可以传入两个不同数组，也可以把同一个数组同时传给 `%p` 和 `%q`。函数签名没有自动赋予两个参数“不重叠”的保证。

设 p 指向的元素初始为 3，`v=9`：

| 执行步骤 | p、q 指向同一元素 | p、q 指向不同分配 |
|---|---|---|
| 第一次 load | a=3 | a=3 |
| store 到 q | p 的元素也变成 9 | 只改变 q，p 仍是 3 |
| 第二次 load | b=9 | b=3 |
| 返回 a+b | 12 | 6 |

若无条件用 `%a` 替代 `%b`，函数会在两种情况下都返回 6，破坏第一种调用。这是一个具体的数据依赖错误；IR 的类型、支配和 verifier 都可能仍然正常。

## 2. 别名查询与可证明的地址关系

编译器需要先回答：两个值可能指向同一存储吗？这类问题称为 alias analysis，别名分析。

对没有其他约定的 p/q 参数，分析通常只能给出 `MayAlias`。意思是尚不能排除重叠；既不表示一定重叠，也不表示概率较高。优化必须同时适用于允许的所有调用，因此需要保留可能影响结果的依赖。

如果把函数换成内部独立分配两个 buffer：

```text
%a = memref.alloc() : memref<4xi32>
%b = memref.alloc() : memref<4xi32>
```

这两个仍在存活的独立分配提供了更强的证据，查询可以得到 `NoAlias`。若查询同一个 SSA 值与自身，则能得到 `MustAlias`。常见结果分类还包括 `PartialAlias`，用来表达已知的部分重叠；但一个实现不一定会精确产生全部分类。

固定版本的默认分析，对观察程序中的三组输入实际返回：

```text
arguments p/q: MayAlias
allocations a/b: NoAlias
same value a/a: MustAlias
```

结果的用途在于缩小优化必须保留的可能性。`NoAlias` 可以排除某个写入影响另一个分配，但不能排除另一个未知操作、释放行为或同步要求。一次查询只提供所问关系的证据。

## 3. 操作效果与内存位置

知道 p/q 的关系还不够，还要知道两次 load 之间的操作做了什么。纯整数加法不会改变 p；store 会写目标；一个没有效果信息的自定义操作则可能读写外部状态。

效果接口把这些行为暴露给消费者。对 `memref.store %v, %a[...]`，可以取得“对 a 的写效果”，再和关心的位置 b 做别名判断：

```text
store 写 a
   ↓
a 与 b 不别名
   ↓
这次写入不修改 b 所描述的存储
```

`AliasAnalysis::getModRef(op, location)` 将两类信息组合起来。观察程序对独立分配 a/b 的实际结果是：

```text
store(a) vs a: Mod
store(a) vs b: NoModRef
```

`Mod` 表示可能修改，`Ref` 表示可能读取，`ModRef` 表示两者都可能，`NoModRef` 表示查询没有发现对该位置的读写影响。这里查询的是 buffer 层面的关系；它并没有自动纳入每次访问的精确索引，也不能替代生命周期分析。

例如在这一版本的 LocalAliasAnalysis 中，ModRef 查询会跳过 Allocate/Free 效果。于是不能依据“没有 Mod/Ref”就把 load 跨过 dealloc；读取一片已经释放的存储本来就不合法。做移动的消费者必须同时检查读写、分配/释放以及目标操作涉及的其他资源。

## 4. 内存依赖与操作重排

现在可以把主例的两个方向都说清楚。第一次 load 必须在 store 前面，才能读到旧值；第二次 load 必须在 store 后面，才能读到新值。如果已知 p/q 重叠，两个约束都要保持。

用 R 表示读、W 表示写：

| 原顺序 | 重排可能改变什么 | 常用名称 |
|---|---|---|
| W → R | 后面的读取本应看见前面的写入 | 真依赖 / RAW |
| R → W | 前面的读取本应看见写入前的内容 | 反依赖 / WAR |
| W → W | 最终留下的写入值 | 输出依赖 / WAW |
| R → R | 普通内存读取通常不相互改变内容 | 单凭这两次普通读没有写入依赖 |

这些关系还要结合访问的实际位置、控制流和操作语义。涉及 volatile、原子、设备同步或其他可观察资源时，不能套用普通内存读写的简化模型。

在 AI 编译器里，fusion、循环交换和并行化都会改变访问顺序。例如两个 tile 看上去独立，若输出窗口重叠，就可能引入 WAW；若第二个 tile 读取第一个刚写出的内容，就有 RAW。后续优化章节将把这里的一维例子推广到索引集合与迭代域。

## 5. 视图与查询精度

[MemRef 视图](../../dialects/memref/views)已经说明：同一分配可以产生两个不重叠窗口。下面的 left 覆盖元素 0、1，right 覆盖元素 2、3：

```text
%left = memref.subview %a[0] [2] [1]
  : memref<4xi32> to memref<2xi32, strided<[1]>>
%right = memref.subview %a[2] [2] [1]
  : memref<4xi32> to memref<2xi32, strided<[1], offset: 2>>
```

按坐标计算，两个窗口的元素集合没有交集。但本章固定版本的通用 LocalAliasAnalysis 会沿 ViewLike 接口追溯到源 `%a`，随后按同一 underlying value 处理；实际查询输出是：

```text
views left/right: MustAlias
```

这个结果必须结合实现层次理解：这里追踪到的是同一底层来源，分析没有证明两个 subview 的逐元素关系。不能据此宣称 `left[0]` 与 `right[0]` 是同一个地址，更不能直接用前者的 load 替换后者。精确访问判断还要结合 offset、size、stride、坐标映射与边界。

这也是阅读分析 API 时必须问“它分析到哪一层”的原因。名字叫别名分析，不意味着自动包含所有仿射索引推理；能给出一种分类，也不意味着消费者可以忽略其适用范围。需要更精确的优化时，应使用匹配访问模型的依赖分析，或补充自己能够证明的条件。

## 6. 未知操作与可扩展协议

假设在两次 load 中间插入一个自定义操作。编译器不能仅凭它没有结果，就断定它不会修改内存；写文件、启动设备任务、写入 buffer 都可能没有 SSA 结果。

安全的信息流应该是：操作准确声明效果，分析根据效果与别名关系得出保守结论，变换只在所需条件成立时实施。缺少接口时，许多通用消费者会保守放弃优化；错误地声明“无效果”，则会使消费者得到错误依据。

对于带 Region 的操作，还要表达内部效果如何计入父操作。给父操作一个空效果列表、却在 body 中写外部内存，会让跨区域变换失去正确依据。[接口章](../ir_definition/traits_interfaces)中的效果消费者和[区域章](../ir_definition/regions_assembly)中的递归效果约定，可以在这里看到实际用途。

未知信息使优化减少，错误信息使程序算错。扩展 Dialect 时，效果模型因此属于操作语义的一部分，而不是仅用于提高优化效果的附加标签。

## 7. 证明能力与具体 Pass 的选择

即使我们已经证明某次 store 与 load 无关，也不表示所有现成 Pass 都会使用这个证明。LLVM 20.1.8 的通用 CSE 对只读操作采用有限路径：同一 Block 内寻找相同操作，并检查中间是否有阻止合并的效果；它没有把本章所有别名推理自动接入 load CSE。

因此可以出现“语义上允许合并，但当前 Pass 没有合并”。这时要区分三个层次：事实是否成立，分析是否表达了这个事实，变换是否消费了它。后一篇[优化合法性](./legality)会把实际输出与这些边界放在一起观察。

## 阅读自查

1. p/q 为 `MayAlias` 时，为什么不能“按通常不同数组”来删除第二次 load？
2. `NoModRef` 为什么不足以许可把读取跨过释放？
3. 两个 view 来自同一个 allocation，怎样进一步判断两个具体元素访问是否重叠？

## 实现依据与实践

主例的别名和 ModRef 结果由配套[分析实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/12-analysis)调用真实分析取得。行为依据为固定版本的 [AliasAnalysis.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Analysis/AliasAnalysis.h)、[LocalAliasAnalysis.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Analysis/AliasAnalysis/LocalAliasAnalysis.cpp)和 [CSE.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/CSE.cpp)。这里给出的 view 结果是该实现的具体行为，不能脱离版本与消费者推广为精确地址等式。
