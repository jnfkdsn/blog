---
order: 1
title: 从张量计算到执行结果
updated: 2026-10-03
---

# 从张量计算到执行结果

考虑两个形状相同的矩阵相加：`C[i,j] = A[i,j] + B[i,j]`。写算法时，我们关心元素对应和结果；真正执行时，处理器还需要知道数据在哪里、怎样遍历、怎样读写、怎样传递函数参数。

MLIR 编译的一条主线，就是逐步补上这些执行细节。前面学习的 Pass、Pattern 和 Conversion 在这里各有用途：一些 Pass 改变计算组织，一些规则把操作展开到下一种表示，而每一阶段都有自己的输入要求与完成条件。

本章沿这一项矩阵加法观察几个关键阶段。先建立整体认识，后面的 Tensor、Linalg、存储和 lowering 章节再分别解释其中的决策。

## 1. 同一项计算的不同信息层次

输入取一个容易核对的小例子：

```text
A = [[1, 2, 3],       B = [[10, 20, 30],
     [4, 5, 6]]           [40, 50, 60]]

C = [[11, 22, 33],
     [44, 55, 66]]
```

这个结果由六次逐元素加法得到。编译器可以选择不同执行方法，只要保持所要求的语义。为了看清过程，这里选择普通 CPU 循环，不加入分块、向量化或 GPU 映射。

```text
矩阵加法的公式
    ↓ 表达计算范围、访问关系和元素运算
Linalg + Tensor：张量上的结构化计算
    ↓ 为张量值安排存储
Linalg + MemRef：访问 buffer 的结构化计算
    ↓ 将计算展开成显式循环
SCF + MemRef + Arith：循环、读写和标量运算
    ↓ 降低控制流、地址和函数接口
LLVM dialect → LLVM IR
    ↓ 目标代码生成、链接和调用
CPU 执行结果
```

图中每一行描述一份 IR 内共同出现的表示。Linalg 与 Tensor 可以一起存在；换成 MemRef 后，Linalg 也仍然可以存在。因此，不能仅按方言名字把它们排成彼此替代的固定层级。

## 2. 用结构化计算保留算法关系

先看这个计算在 tensor 语义下的完整表示：

<!-- tensor-example: add -->
```text
module {
  func.func @add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %empty = tensor.empty() : tensor<2x3xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<2x3xf32>)
      outs(%empty : tensor<2x3xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
}
```

暂时只沿三条线阅读。

第一条线是数据：函数接收两个 `tensor<2x3xf32>`，返回同样形状的 tensor。`tensor.empty` 提供结果形状，它的元素内容未指定；本例会计算每个结果元素，不使用这些初始内容。

第二条线是坐标：三个 `(i,j) -> (i,j)` 分别描述 A、B 和输出。对于迭代点 `(1,2)`，读取 A[1,2] 和 B[1,2]，把算出的值交给结果的 [1,2]。

第三条线是计算：Region 的 `%x`、`%y` 是当前迭代点取出的两个标量，`arith.addf` 求和，`linalg.yield` 提交这一点的结果。

IR 还没有展开两个 `for`，但已经保存了循环空间、访问映射和元素计算之间的联系。后续做分块或融合时，正是这些关系帮助编译器定位计算怎样组织，而不必先从零散的地址操作中重新猜测。

## 3. 从张量值到可读写存储

张量结果是一个新的值。CPU 最终需要把它的元素放在存储中。这个例子采用的 bufferization 路径为结果安排一个 buffer，将张量接口改成 MemRef 接口。实际输出为：

<!-- tensor-output: buffer-add -->
```text
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x3xf32>, memref<2x3xf32>) outs(%alloc : memref<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<2x3xf32>
  }
}
```

现在 `%alloc` 表示结果存储，`outs(%alloc)` 指向写入目的地，函数返回这个 MemRef。Linalg 操作不再产生 tensor SSA 结果，而是在执行时向输出 buffer 写入元素。

关键变化不是把文本里的 `tensor` 换成 `memref`，而是从“得到一个结果值”进入了“怎样实现这个值的存储”。在更复杂的程序里，还要判断哪些存储能复用、哪些旧值必须保留、何时需要复制。这些问题由 Bufferization 专题展开。

本例返回新分配的结果，谁负责释放还需要调用约定。上述 IR 只展示存储安排，不包含完整的所有权与释放方案。

## 4. 从结构化计算到显式循环

继续将 buffer 形式的 Linalg 展开成循环，得到：

<!-- tensor-output: loops-add -->
```text
module {
  func.func @add(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) -> memref<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<2x3xf32>
        %1 = memref.load %arg1[%arg2, %arg3] : memref<2x3xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %alloc[%arg2, %arg3] : memref<2x3xf32>
      }
    }
    return %alloc : memref<2x3xf32>
  }
}
```

此时原来隐含在 Linalg 中的关系变成了具体操作：

- 形状 `2×3` 给出两个循环的范围。
- 输入映射给出两次 load 的 `[i,j]`。
- Region 中的加法进入循环体。
- 输出映射和 yield 对应结果 buffer 的 store。

沿 `(i,j)=(1,2)` 执行，读取 6 和 60，计算 66，再写入结果的 [1,2]。这个过程与第一节公式一致，只是执行细节已经显式化。

这里的循环形式是一种选择。另一个 pipeline 可以先 tile、vectorize，再展开成不同结构；Linalg 的 `parallel` 也不要求这里必须产生并行线程。

## 5. 从循环和地址到目标代码

显式循环仍没有完成全部工作。SCF 要降低成低层控制流，MemRef 中的形状与布局信息要落实为地址计算和调用约定，标量操作也要进入目标支持的形式。

例如前一节的两次 load、一条 add 和一次 store，在低层仍对应“读两个地址、相加、写一个地址”。区别是低层需要知道元素地址如何从基地址、偏移与索引计算出来。

LLVM dialect 让这些低层操作继续作为 MLIR 对象存在；translation 再把它们转换成 LLVM IR。LLVM 后端完成目标指令选择等代码生成工作，链接与运行时则让分配、外部调用等依赖可用。最终，程序接收输入并产生结果。

不需要现在就记住 MemRef 描述符每个字段。先明确这一步的任务：前面还由高层类型隐含的信息，必须进入具体地址、参数和返回值约定。后续 [CPU lowering](../../compiler/lowering/)会沿同一个边界解释。

## 6. 转换正确与执行正确的证据

这条路径可以分层检查：

| 检查 | 能说明什么 |
|---|---|
| parser / verifier 通过 | IR 满足已实现的结构、类型与操作约束 |
| 某阶段转换成功 | 这条 pipeline 能处理该输入并形成结果 |
| 观察关键前后 IR | 计算与使用关系按预期展开，阶段边界没有明显遗漏 |
| 编译并执行，核对全部结果元素 | 对选定输入，实际运行与参考结果一致 |

本章的矩阵例子已走到 LLVM IR，并编译为本机程序核对六个结果元素。这个小测试支持的是该例子的执行结论；更一般的动态形状、别名、浮点边界与所有权，仍需要针对性设计测试。

## 7. 进入张量与结构化计算

接下来先解决两组紧密相关的问题：

1. [Tensor](../../dialects/tensor/)：一个张量值怎样更新，形状怎样描述，切片与 reshape 怎样对应元素。
2. [Linalg](../../dialects/linalg/)：整个计算域怎样与各个张量坐标相连，怎样表达广播、归约和结果初始值。

随后再进入 MemRef、Bufferization 和执行链的详细机制。这样能先读懂“计算了什么”，再解释“怎样安排存储与执行”。

如果从 PyTorch 开始，还需要某个实际前端将程序导入并转换到适合后端的表示。那是特定项目的职责，不由本章这段 Linalg 代码自动完成；实际前端路径保留为本目录的后续教程。

## 观察与查阅

[配套实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/09-tensor-linalg)保留每个阶段的直接命令、完整输出和数值检查。正文不要求先运行脚本。

本例以 LLVM 20.1.8 为依据。结构化计算见固定版本 [Linalg 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/Linalg/_index.md)，存储转换见 [Bufferization](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Bufferization.md)，低层边界见 [LLVM dialect 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/LLVM.md)。
