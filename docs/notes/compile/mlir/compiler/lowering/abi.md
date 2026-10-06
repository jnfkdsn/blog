---
order: 2
title: 类型降低、ABI 与函数边界
updated: 2026-10-04
---

# 类型降低、ABI 与函数边界

函数内部已经知道如何根据 stride 找到元素，但 C 程序调用它时，只传一个 `float *` 还不够：二维视图的尺寸、偏移和步长必须一起抵达函数。

ABI 要解决的就是这项跨边界约定：参数如何表示和排列，结果怎样返回，调用双方用什么类型布局解释同一组数据。本章沿 `read2d` 和一个返回新数组的函数，观察 MLIR 如何生成这层接口。

## 1. 默认内部调用约定

考虑已经认识的签名：

```text
func.func @read2d(%m: memref<?x?xi32, strided<[?, ?], offset: ?>>,
                  %i: index, %j: index) -> i32
```

在本文的 64 位默认配置中，二维 MemRef 对应两个指针、offset、两个 sizes 和两个 strides。作为默认内部函数参数时，这些描述符字段会展开成各自的参数，再加上 `%i` 和 `%j`：

```text
llvm.func @read2d(
  allocated: ptr, aligned: ptr, offset: i64,
  size0: i64, size1: i64, stride0: i64, stride1: i64,
  i: i64, j: i64) -> i32
```

这里为解释字段用了语义名称，具体合法文本中的参数名由工具打印。函数体可将参数重新组装成 descriptor 供通用转换使用，再从中取访问需要的字段；后续优化有机会清理冗余组装。

因此，即使原函数只有三个 MLIR 参数，低层也可能出现九个参数。直接猜测 C 原型为 `int read2d(int *p, int i, int j)` 会让两边对参数位置产生完全不同的解释。

## 2. C Wrapper 的职责

为函数加上 `llvm.emit_c_interface` 后，Func-to-LLVM 转换会生成 `_mlir_ciface_read2d`。它接受一个指向描述符结构体的指针，以及两个标量索引：

```text
llvm.func @_mlir_ciface_read2d(ptr, i64, i64) -> i32
```

这层 wrapper 读取结构体字段，拆开后调用内部 `@read2d`，最后返回标量。调用关系为：

```text
C 调用者准备 descriptor
        ↓ 传 descriptor 的地址
_mlir_ciface_read2d
        ↓ load 字段，按内部参数顺序传递
read2d
        ↓ 计算元素地址并读取
i32 结果沿两层调用返回
```

它没有把矩阵复制进 wrapper，也没有为参数取得新的内存所有权。这里 load 的是小结构体中的字段，后面的元素 load 才读取矩阵内容。

## 3. 构造一个可调用的描述符

在与本文配置一致的 64 位 C 环境中，可采用：

```c
typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} I32View2;

extern int32_t _mlir_ciface_read2d(I32View2 *, int64_t, int64_t);

int32_t data[16];
for (int k = 0; k < 16; ++k) data[k] = k;
I32View2 window = {data, data, 5, {2, 2}, {4, 1}};
int32_t result = _mlir_ciface_read2d(&window, 1, 1);  // 10
```

这段是实际驱动中的核心片段，完整驱动还包含标准头文件与检查。数组包含 `0…15`，窗口对应中间四个位置，读取 `[1,1]` 得到 10。

各字段必须共同描述合法视图。将 offset 设为 5 后，再把 aligned pointer 自行移动 5 个元素，就会重复计算偏移。若选择不同的访问基址，应重新推导 offset，确保最终地址关系一致。

本例两根指针都来自栈数组，但函数只借用它们，不执行 free；只要调用期间数组仍有效，就符合当前接口。若另一个接口会接管并释放参数，便不能直接套用这种栈数组接入。

## 4. 返回 MemRef 的结果约定

返回标量只需一个整数返回值。返回 MemRef 则还要携带描述符，而且新存储的释放责任要有明确归属。

下面的完整函数显式分配并复制一个矩阵：

<!-- cpu-example: make-copy -->
```text
module {
  func.func @make_copy(%src: memref<2x3xf32>) -> memref<2x3xf32>
      attributes {llvm.emit_c_interface} {
    %out = memref.alloc() : memref<2x3xf32>
    memref.copy %src, %out : memref<2x3xf32> to memref<2x3xf32>
    return %out : memref<2x3xf32>
  }
}
```

内部低层 `@make_copy` 返回一个描述符结构体。生成的 C wrapper 把结构体结果改成**第一个指针参数**：调用方提供一块存放结果描述符的空间，wrapper 将结果写进去。

对应 C 原型和使用方式为：

```c
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} F32View2;

extern void _mlir_ciface_make_copy(F32View2 *result, F32View2 *source);

float values[6] = {1, 2, 3, 4, 5, 6};
F32View2 input = {values, values, 0, {2, 3}, {3, 1}};
F32View2 output;
_mlir_ciface_make_copy(&output, &input);
// 使用 output.aligned / offset / strides 读取或修改结果。
free(output.allocated);
```

`output` 结构体本身由调用者提供，数据 buffer 由 MLIR 函数分配。free 的对象是 `output.allocated` 指向的分配，而不是 `&output` 这个局部描述符地址。

这里显式使用 alloc+copy，结果可以独立修改；不要将它与上一模块中有特殊后续修改约束的 `bufferization.clone` 混淆。输入仍归调用方，本函数不释放它。该分配的默认 CPU lowering 与 C `free` 匹配；更换 allocator 或运行时后，释放方式也必须匹配。

## 5. ABI 与所有权共同构成接口

参数位宽和字段顺序正确，只解决“怎样解释传来的比特”。接口还需要约定“谁可以修改内容、谁负责释放、存储保持有效多久”。

例如上述 make_copy，输入 shape 静态为 `2×3` 且布局连续，调用方必须提供与类型一致的字段。返回的是新分配，按所用所有权协议将责任交给调用方。若改成返回输入的 subview，生命周期和别名规则都发生变化，不能只改一个 C 返回类型就完成接入。

对于异步设备执行，还需要存储保持有效直到设备完成。当前 CPU 同步调用没有这层等待协议，后续设备章节再建立它。

## 6. 接口一致性的检查顺序

遇到“MLIR 能编译，但 C 调用结果异常”时，可沿数据传递逐项核对：

1. 检查实际导出的名字，是内部函数还是 `_mlir_ciface_` wrapper。
2. 对照实际低层签名，核对参数数量、位宽、指针与结构体结果。
3. 核对描述符布局和字段值，尤其是 aligned pointer、offset、strides。
4. 核对形状、别名、初始化和生命周期前提。

这些检查都应以生成结果为依据。C/C++ 头文件里的声明如果只是凭印象手写，即使编译器接受，也不能证明它与另一侧 ABI 匹配。

下一章进入[代码生成、链接与执行](./execution)，将已经一致的接口交给目标工具链。

## 阅读自查

1. 一个二维 MemRef 参数为何会展开为七个低层参数？
2. wrapper 接收描述符指针时，是否复制了整个矩阵？
3. 返回新 MemRef 时，结果结构体空间与数据 buffer 分别由谁提供？
4. 为什么释放使用 allocated pointer，而不是描述符地址或窗口首元素地址？

## 资料与实践

本文 ABI 由 LLVM 20.1.8 的实际输出及本机 C 调用验证，不能直接作为任意目标的通用结构体定义。依据见[LLVM Target 调用约定](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/TargetLLVMIR.md)、[FuncToLLVM.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp)、[CRunnerUtils 描述符类型](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h)。

可选实践：[实际 C 驱动](https://github.com/jnfkdsn/aicompiler/blob/main/llvm-mlir/11-cpu-lowering/driver.c)。
