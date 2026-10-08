---
order: 1
title: Affine Map、维度与符号
updated: 2026-10-04
---

# Affine Map、维度与符号

从一张图像中裁出一个窗口，需要把输出坐标映射回输入坐标。如果窗口起点是 `(row0,col0)`，输出 `(i,j)` 读取输入 `(i+row0,j+col0)`。这个关系既能直接计算地址，也能让编译器推理不同迭代是否访问同一元素。

Affine Map 保留的就是这样的索引关系。本章从一个 3×4 窗口的裁剪开始，说明映射中的变量代表什么、怎样绑定到 SSA 值，以及为什么某些表达式能被统一分析。

## 1. 裁剪中的坐标对应

源数组形状为 6×8，输出为 3×4，当前窗口起点为 `(1,2)`：

```text
输出 (0,0) → 输入 (1,2)
输出 (0,3) → 输入 (1,5)
输出 (2,0) → 输入 (3,2)
输出 (2,3) → 输入 (3,5)
```

如果源元素定义为 `src[r,c]=100*r+c`，完整输出应为：

```text
102 103 104 105
202 203 204 205
302 303 304 305
```

输出坐标在迭代中变化；窗口起点由函数输入给定，在这次遍历中固定。这两类角色分别对应 map 的 dimension 和 symbol：

```text
affine_map<(i,j)[row0,col0] -> (i+row0,j+col0)>
```

圆括号声明维度，方括号声明符号，箭头后给出结果坐标。这是一个映射描述，不是正在执行的函数，也没有因此产生一个新的 SSA Value。

## 2. 映射描述与 SSA 绑定

真正执行映射需要操作把它绑定到当前值。为分别计算行、列，下面使用两个单结果 `affine.apply`；当前操作要求 map 具有一个结果：

<!-- structured-example: crop -->
```text
module {
  func.func @crop(%src: memref<6x8xi32>, %out: memref<3x4xi32>, %row0: index, %col0: index) attributes {llvm.emit_c_interface} {
    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 4 {
        %row = affine.apply affine_map<(i)[r0] -> (i + r0)>(%i)[%row0]
        %col = affine.apply affine_map<(j)[c0] -> (j + c0)>(%j)[%col0]
        %v = affine.load %src[%row, %col] : memref<6x8xi32>
        affine.store %v, %out[%i, %j] : memref<3x4xi32>
      }
    }
    return
  }
}
```

在 `%i=2,%row0=1` 时，第一条 apply 将 map 中 i 绑定为 2，r0 绑定为 1，得到 `%row=3`。下一条得到列坐标，再由 load/store 完成元素搬运。

同一个 map 可以被多次使用，每次绑定不同 SSA 值；printer 也可能把重复 map 提到模块前并命名为 `#map`。这个文本别名用于复用描述，不是内存地址，也不是运行时存储。

接口条件必须同时成立：本例要求 `0≤row0≤3`、`0≤col0≤4`，输入输出存储有效且不重叠。Affine 表达式可分析，不等于工具自动给动态窗口做越界检查。错误起点可能仍通过结构验证，却产生非法访问。

## 3. Dimension 与 Symbol 的有效范围

在当前函数的循环里，i/j 作为不断变化的迭代坐标，row0/col0 作为相对这片 AffineScope 的参数。这是理解二者最方便的例子，但不能把规则缩减为“循环变量一定是维度，其他 SSA 值随便当符号”。

所有 SSA 值都只定义一次，然而某个值可能在循环体的每次动态执行中产生不同结果。是否可以绑定为 symbol，要检查值的来源和使用位置。函数这种 AffineScope 的参数、该作用域顶层定义的适当值、常量，以及由有效符号经纯操作产生的值等，具有相应合法路径。

包围当前操作的 affine 循环 IV 可以用作 dimension，却通常不能直接在同一函数 scope 内被伪装成不随循环变化的 symbol。把 `%i` 从圆括号挪到方括号不会改变它的语义，只会让绑定不满足协议。

维度也可以接收某些本可作为符号的值。两种角色用于表达和约束分析中的变量，而不是两种不同 MLIR 类型；它们的实际 SSA 类型都是 index。准确规则由值所在的 AffineScope 和定义/使用关系决定。

验证的位置也要分清：此版本的 affine.apply 主要检查参数数量与单结果，不会在这里拒绝所有不满足 symbol 条件的绑定；affine.if、循环边界等消费者会进一步检查标识符有效性。因此把 IV 填进方括号并解析成功，仍不能作为它是有效 symbol 的证明。实验的拒绝用例在 affine.if 上观察这个条件。

## 4. 可分析的表达式

最基本的仿射表达式由整数常量、变量的常数倍以及加减构成，例如 `2*i+j+3`。MLIR 还允许对正整数常量做 floordiv、ceildiv、mod，因此可以描述固定大小分块：

```text
块编号 = i floordiv 4
块内位置 = i mod 4
```

`i=10` 对应块编号 2、块内位置 2。相比把这些关系展开成很多普通整数操作，显式 map 保留了整体结构，方便组合、化简与约束推理。

两个随迭代变化的量相乘，如 `i*j`，不属于普通仿射表达式。运行时步长参与乘法，如 `i*stride`，可以进入更宽的 semi-affine 表达形式，但不是所有基于线性整数约束的分析都支持它。

因此需要分清三个边界：语法能否表示，某个操作是否允许绑定，所用分析是否能够求解。能解析一个 map 不能推出所有循环优化都能处理它。遇到间接索引 `x[index_array[i]]`，索引本身来自内存，更不能仅改写 map 文本就获得仿射关系。

## 5. 与 Linalg 和 MemRef 的对应

之前 Linalg 使用 `(i,j)->(i)` 把二维迭代投影到行统计结果，语义仍是“给定迭代坐标，访问哪个 operand 坐标”。裁剪则使用带平移的映射，二者都保留访问关系。

MemRef 布局描述的是下一层：取得元素坐标后，怎样变成底层线性偏移。例如当前输入按连续行主序存储，输入 `(3,5)` 的元素偏移为 `3*8+5=29`。裁剪 map 先把输出 `(2,3)` 变成输入 `(3,5)`，布局再把 `(3,5)` 变成存储位置。

```text
迭代/输出坐标 → operand 元素坐标 → 存储偏移 → 实际读写
```

优化可能改变前一层的遍历或映射，也可能改变后一层的布局。保持两层区别，才能说明“交换循环顺序”和“转置数据”为什么是不同动作。

## 阅读自查

1. map 描述与 affine.apply 的结果 Value 分别是什么？
2. 把当前循环 IV 写进 symbol 列表，为什么不能让它变成循环不变量？
3. 输出 `(2,3)` 到输入地址，需要经过哪两层关系？

下一篇[整数集合与循环域](./domains)从“一个点访问哪里”进入“哪些点应当执行”。

## 实现依据与实践

语义按 LLVM/MLIR 20.1.8 的 [Affine 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/Affine.md)及 [AffineOps.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Affine/IR/AffineOps.cpp)核验。[结构化优化实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)提供完整裁剪输入、绑定失败变体与本机坐标对照。
