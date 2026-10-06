---
order: 3
title: Linalg 到 CPU 的最小执行链
updated: 2026-10-04
---

# Linalg 到 CPU 的最小执行链

现在把前面分别学习的计算、存储、低层表示和调用约定放回一个任务：让 C 程序调用 MLIR 定义的二维矩阵加法，得到与参考计算相同的结果。

输入是两个 `2×3` 矩阵：

```text
A = [[1,2,3],       B = [[10,20,30],
     [4,5,6]]            [40,50,60]]

期望 Out = [[11,22,33],
            [44,55,66]]
```

本例采用调用方提供输出 buffer 的接口。它适合看清“哪些存储来自 C，编译出的函数写入哪里”，并避免为第一次 C 接入同时引入返回新分配的责任。返回新数组的对应路径已在 [ABI 章节](../../compiler/lowering/abi)展开。

## 1. 计算接口与前提

下面是一份完整的 MLIR 输入，可保存为 `add-into.mlir`：

<!-- cpu-example: add-into -->
```text
module {
  func.func @add_into(
      %a: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %b: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %out: memref<?x?xf32, strided<[?, ?], offset: ?>>)
      attributes {llvm.emit_c_interface} {
    linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : memref<?x?xf32, strided<[?, ?], offset: ?>>,
                   memref<?x?xf32, strided<[?, ?], offset: ?>>)
      outs(%out : memref<?x?xf32, strided<[?, ?], offset: ?>>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }
    return
  }
}
```

函数的任务很明确：按相同坐标读取两个输入，相加后写入输出。它不负责分配或释放三个参数。

调用方保证三个视图形状相同，所访问元素有效，输入已初始化，输出可写；本例还保证输出与输入不重叠。动态 strided 类型允许不同物理布局，但不会自动检查任意传入元数据是否正确。

如果从 Tensor 程序开始，前面还要通过 Bufferization 选择对应存储。这篇直接从已经明确存储接口的 Linalg 开始，重点闭合后半段执行链。

## 2. 计算与地址的逐层落实

对一个坐标 `[i,j]`，完整过程可以用同一组对象追踪：

```text
Linalg：三个 identity maps 确定 A[i,j]、B[i,j]、Out[i,j]
    ↓
循环：i 遍历行，j 遍历列
    ↓
MemRef：每个视图独立使用 offset、strides 计算元素位置
    ↓
LLVM：GEP + 两次 load + 浮点 add + 一次 store
```

先前的[显式循环章节](../../compiler/lowering/loops)已经展示实际完整 SCF 与 CF 输出。对当前 `[0,1]`，连续布局下三者元素偏移都是 1，读取 2 和 20，写入 22。

如果 A、B 和 Out 有不同 stride，三者的逻辑坐标仍相同，但各自的线性位置不同。因而不能先算一个物理偏移，再不加区分地用于所有 operands；每个视图都带有自己的布局。

## 3. 从 C 提供存储

下面是与本文 64 位默认 ABI 匹配的完整 C 驱动，可保存为 `driver.c`：

<!-- cpu-c: minimal_driver.c -->
```c
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} F32View2;

extern void _mlir_ciface_add_into(F32View2 *, F32View2 *, F32View2 *);

int main(void) {
  float a[6] = {1, 2, 3, 4, 5, 6};
  float b[6] = {10, 20, 30, 40, 50, 60};
  float out[6] = {0};
  F32View2 av = {a, a, 0, {2, 3}, {3, 1}};
  F32View2 bv = {b, b, 0, {2, 3}, {3, 1}};
  F32View2 ov = {out, out, 0, {2, 3}, {3, 1}};
  _mlir_ciface_add_into(&av, &bv, &ov);
  int failed = 0;
  for (int k = 0; k < 6; ++k) {
    printf("%.0f\n", out[k]);
    if (out[k] != a[k] + b[k]) failed = 1;
  }
  return failed;
}
```

数组负责保存数据；`av/bv/ov` 三个结构体负责描述怎样访问它们。每个描述符都说明大小为 `[2,3]`、行步长为 3、列步长为 1、偏移为 0。

调用的是 `_mlir_ciface_add_into`。它由 `llvm.emit_c_interface` 生成，接收描述符指针，解包字段后调用内部函数。这里不用手写内部那个展开为许多指针和整数参数的函数原型。

三个数组都在 main 的栈帧中，调用期间有效；被调用函数只借用，不释放它们，因此不需要额外 free。`out` 初始化为 0 便于观察，但逐元素加法完整覆盖输出，不依赖其旧内容。

## 4. Lowering、Translation 与链接

将输入降低到 LLVM Dialect：

```bash
mlir-opt add-into.mlir \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-to-llvm --reconcile-unrealized-casts \
  --verify-each -o add-llvm.mlir
```

前几个 passes 让计算、索引与控制流显式化；`convert-to-llvm` 通过已注册的方言转换接口处理相关低层操作与类型；最后清理能够消解的转换桥接。该命令链针对本例输入，不是任何 MLIR 程序都适用的通用编译命令。

随后完成翻译和代码生成，再链接 C 驱动：

```bash
mlir-translate --mlir-to-llvmir add-llvm.mlir -o add.ll
llc -filetype=obj -relocation-model=pic -O0 add.ll -o add.o
clang -std=c11 driver.c add.o -o run
./run
```

这些命令假定工具已在 PATH 中。输入中没有分配、数学库调用或通用复制路径，因此这个最小函数不额外依赖 MLIR 的辅助运行时库。增加相应操作后，应重新检查生成代码需要哪些符号。

程序实际依次输出：

```text
11
22
33
44
55
66
```

驱动同时逐项比较 `out[k]` 与 C 参考 `a[k]+b[k]`，不一致时返回失败。所选数值能够被 f32 精确表示，本例的直接相加也精确；这不能外推为复杂浮点算法都应做逐位相等比较。

## 5. 非连续窗口复用同一个函数

只测试连续数组，还不能确认动态 stride 被正确使用。继续使用同一个编译结果，改变 C 侧描述符和数据摆放：

| 视图 | offset | sizes | strides |
|---|---:|---|---|
| A | 5 | `[2,3]` | `[7,1]` |
| B | 2 | `[2,3]` | `[8,2]` |
| Out | 3 | `[2,3]` | `[9,2]` |

三个底层数组各准备足够空间，把逻辑矩阵 A、B 按相应公式写入。以 `[1,2]` 为例：

```text
A 的位置   = 5 + 1×7 + 2×1 = 14，内容为 6
B 的位置   = 2 + 1×8 + 2×2 = 14，内容为 60
Out 的位置 = 3 + 1×9 + 2×2 = 16，写入 66
```

前两处位置恰好都为 14，但它们位于两个不同数组中；这不意味着共享存储。其他坐标的偏移也并不相同。

按逻辑坐标读取 Out，仍应得到同样六个数。实验还把窗口以外的位置初始化为哨兵，并检查没有被写入，避免只验证“期望位置对了”却漏掉额外错误写入。

## 6. 空域、接口与所有权的对照

动态接口还接受合法的 `0×3` 或 `2×0` 形状。对本例，循环没有元素计算，因此不应该产生输出写入。这个边界同时检查动态 dim 和循环条件是否被保留。

同批实验还分别验证：

- 从非连续 i32 窗口读取 `[1,1]`，核对 offset 与两个 stride。
- 使用不同的 allocated/aligned pointer 表达同一访问位置。
- 求和循环的 carried value，验证 `[3,-2,5,7] → 13`。
- 从返回新分配的函数接收结果描述符、读取内容并匹配释放。

这些对照各改变一个边界条件。它们比增加许多相同连续矩阵，更能定位“计算正确但接口或布局有误”的问题。

## 7. 这条链提供的基础

现在，一段 MLIR 不再只是能被打印的文本：它已经进入循环和低层函数，被翻译为 LLVM IR，生成目标文件，并通过清晰的 ABI 与普通 C 程序交换数据。

后续优化可以在这条可验证的路径上改变一个环节。例如 tiling 调整循环组织，vectorization 改变一次处理的元素数量，目标映射改变执行资源。但每次改动都应重新检查表示前提、地址关系、数值和生命周期。

下一阶段的[分析模块](../../compiler/analysis/)解释“编译器依据什么允许变换”；[算子贯通模块](../kernels/)将以归约和 Softmax 提供更贴近 AI 计算的任务。

## 阅读自查

1. 相同 `[i,j]` 为什么在 A、B、Out 中可以对应不同线性偏移？
2. 为何调用方创建了三个描述符，但没有复制三个矩阵？
3. 六个结果都正确，为什么还检查窗口外没有写入？
4. 加入一个返回新分配的函数后，C 端需要额外维护哪项责任？

## 版本与复现

正文命令链按 LLVM 20.1.8 与本机 64 位 C ABI 验证。最小驱动逐项检查六个结果，扩展驱动检查 33 个标量观察；这覆盖本例的接口和数据边界，不是性能结论或一般内存安全证明。

完整源码、直接命令、观察脚本与有限任务见[11-cpu-lowering](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/11-cpu-lowering)。LLVM 调用边界依据[固定版本 TargetLLVMIR](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/TargetLLVMIR.md)。
