---
order: 60
---

# Triton 与 Triton-Ascend

这一系列把块级编程模型与编译路径放在一起：先说明程序表达什么，再追踪 IR、布局和代码生成怎样落实到目标执行。已有基础和两条固定版本源码路径；更完整的编译器优化系列将沿同一入口扩展。

## 推荐路径

1. [Python 前置知识](/notes/compile/triton/python_basic)
2. [Triton 基础](/notes/compile/triton/triton_basic)
3. [Triton GEMM 优化](/posts/triton_gemm)
4. [Triton 加法算子的编译路径](/notes/compile/triton/compiler_path)：固定 v3.2.0，沿块、encoding、load lowering 与 NVIDIA pipeline 阅读源码。

5. [Triton-Ascend 编译路径](/notes/compile/triton/ascend_compiler_path)：固定 v3.2.1，追踪 adapter、NPU 编译器与运行时的交接。

## 编程与编译的连接

编程部分帮助理解程序语义，编译路径解释这些语义怎样被实现。NVIDIA 与 Ascend 路径使用不同的后端与运行时，分别核验；跨领域阅读时从[算子优化](/notes/kernels/)或[推理系统](/notes/inference/)链接到本系列，不另建一份相同教程。

## 和 CUDA 的对照

- [CUDA 基础语法](/notes/kernels/cuda/cuda_basic_syntax)：对照 thread/block/grid、shared memory、warp 级操作。
- [Softmax 算子实现与优化](/posts/softmax)：对照 CUDA softmax 和 Triton softmax 的表达方式。
- [CUDA 矩阵乘法](/posts/GEMM)：对照手写 CUDA GEMM 和 Triton `tl.dot`。

## 相关地图

- [算子编程与优化](/notes/kernels/)
