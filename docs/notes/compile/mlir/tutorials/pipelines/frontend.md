---
order: 20
title: 从 PyTorch 矩阵乘法到 Linalg
updated: 2026-10-04
---

# 从 PyTorch 矩阵乘法到 Linalg

前面的 CPU 贯通从已经写好的 Linalg 开始。实际模型则常从 PyTorch 进入编译器：Python 程序中的 `torch.mm(a,b)` 怎样成为能够分块、Bufferize 和降低的矩阵乘法？

本章固定 torch-mlir 的一条 Export/FX 导入路径，随后追踪 `torch.aten.mm` 到 Linalg 的规则。关注同一个计算携带的信息怎样变化，不把全部前端算子和后端选项同时展开。

## 1. 计算入口与导出结果

源程序可以很小：

```python
class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.mm(a, b)
```

取 a 为 2×3、b 为 3×2，结果应为 2×2。Python 定义了执行计算的程序，但下游 MLIR Pass 需要显式操作、类型和数据流，不能直接对任意 Python 执行过程做 Linalg 分块。

选定版本的 `export_and_import` 先获得 `torch.export.ExportedProgram`；若传入的已经是 ExportedProgram，就复用它。对于普通 Module 则调用 export，再按所选 decomposition table 处理图。

导出承担把受支持程序转成显式图及其约束的工作；分解把某些高层操作改写成所选集合。两步都还不同于“已经生成设备代码”。动态 shape、状态和 mutation 也有对应契约，不能把一次静态 mm 示例推广成任意 Python 都可导入。

## 2. 从图节点到 Torch 方言

FxImporter 读取图节点的操作 schema、输入引用、输出信息，创建相应 MLIR Operation。普通 ATen schema 的名字经映射形成 `torch.aten.*`，有 overload 时继续附加 overload 名称。

忽略位置和外围包装，静态 mm 的 Torch IR 形式可以写成：

```text
%r = torch.aten.mm %a, %b
  : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,2],f32>
    -> !torch.vtensor<[2,2],f32>
```

这是按该操作约定整理的示意，不是本机运行 importer 后截取的输出。`vtensor` 仍是 Torch 方言的 value tensor 类型，记录 shape/dtype 信息；它不是已分配的 GPU 数组，也还没有选定局部 tile 或线程布局。

默认冻结导入路径与显式启用 mutation 的路径不同。此处无参数状态、无原地修改，只沿最简单的 `import_frozen_program` 继续。

## 3. 转换规则先检查可表示性

`ConvertAtenMmOp` 是一个真实的 `OpConversionPattern<AtenMmOp>`。它通过 adaptor 取得已经适配为后端 Tensor 的操作数，随后检查这两个输入是否能被目标表示。

本例要求 rank=2、元素类型兼容。规则还处理量化和不同整数符号等分支，但普通 f32 情形不需要先理解这些分支。判断失败时返回匹配失败信息，不能先构造一个不合法的 Linalg matmul，再期待 verifier 替源程序兜底。

动态形状还带来运行时条件：a 的第二维必须等于 b 的第一维。固定源码在没有采用严格符号形状假设的路径中创建：

```text
ka = tensor.dim a, 1
kb = tensor.dim b, 0
condition = arith.cmpi eq, ka, kb
cf.assert condition, "mismatching contracting dimension for torch.aten.mm"
```

这是省略 SSA 类型标注的过程示意。条件既不是 tiling 的选择，也不是 Bufferization 的别名问题，而是源算子成立所需的形状约束。若前端契约已经允许假定该关系，则可能走不生成这一 guard 的分支；不能把“省了检查”直接当成更强的编译器证明。

## 4. 结果形状与初始值

规则从 a 的第 0 维和 b 的第 1 维得到 M、N，建立零初始化结果，再创建 Linalg matmul：

```text
tensor.empty(M,N)
    ↓ linalg.fill 0
zero_init
    ↓ linalg.matmul a,b outs zero_init
matmul_result
    ↓ 必要的元素类型转换与 tensor.cast
源操作所需的结果类型
```

这里 fill 的理由与[Matmul 章节](../kernels/matmul)相同：Linalg matmul 读取并累加 outs 的初值。只把 torch.mm 名字改成 linalg.matmul、却提供未初始化的结果张量，会改变计算。

创建 empty 时使用动态尺寸，结果类型可能先是 `tensor<?x?xf32>`。源结果若已知某一维，末尾 cast 恢复对应的静态类型信息。cast 的作用是衔接类型契约，不是重新计算或转置矩阵。

选定的 f32 情形继续使用相应累加类型；其他元素类型可能先采用更宽的默认累加类型，再转换回要求的结果类型。读代码时应沿自己选定的 dtype 分支推演，而不是略过精度变化。

## 5. 一个 Pass 后仍有混合表示

上游 `basic.mlir` 的单 Pass 测试明确检查以下边界：

```text
!torch.vtensor
    ↓ torch_c.to_builtin_tensor
tensor → fill / matmul / cast
    ↓ torch_c.from_builtin_tensor
!torch.vtensor
```

也就是说，局部操作已经转换，函数签名却可以暂时保留 Torch 类型。这个结果与前面学习的 materialization 完全对应：先连接新旧表示，再由后续阶段统一边界。

`ConvertTorchToLinalg` 注册目标方言、类型转换和各类规则，实际调用 `applyPartialConversion`。它负责这一段的转换，并不承诺单个 Pass 就把整个模型变成纯 Linalg、清掉全部 Torch 操作或生成 CPU 程序。

## 6. Pipeline 如何闭合边界

固定版本的 Linalg-on-tensors 后端 pipeline 除 TorchToLinalg 外，还包含 TorchToTMTensor、TorchToSCF、TorchToArith、TorchToTensor 等转换和清理。原因是一个模型同时含计算、控制、标量、shape 和其他操作，不是每项工作都适合变成 Linalg。

尾部再进行函数后端类型转换、最终边界转换，并检查 Linalg-on-tensors 后端契约。阅读这段 pipeline 时，可以只跟踪三项变化：

| 阶段 | 当前 mm 案例要完成什么 |
|---|---|
| 操作转换 | 用后端支持的 Tensor/Linalg/Arith 表达计算与 guard |
| 边界转换 | 参数、返回值、调用和桥接保持类型一致 |
| 契约检查 | 确认交给下游的模块满足后端接受条件 |

此时仍以 Tensor 计算为主，随后由选定编译后端安排优化、Bufferization、目标映射和执行。不能把整条路线写成固定的 `Torch→Linalg→Tensor`：Linalg 和 Tensor 本来就可以共同组成这一阶段。

## 7. 与已有 CPU 证据连接

完成上述路径后，矩阵乘法进入了本课程已经能处理的表示。可以接[分块 Matmul](../kernels/matmul)和[CPU 执行链](./cpu)，继续观察存储、LLVM ABI 与数值。

两段证据需分清：标准 Linalg 的本机编译与 C 数值已验证；本章前端部分核对的是固定源码、调用链和上游测试期望。当前 Python 环境没有 torch/torch_mlir 包，未运行此提交的 exporter 或项目 Pass，因此文中的 Torch 片段与检查项不冒充本机前端日志。

固定 torch-mlir 提交为 `528d3dcb9a2288743b62df9f4937b1689787e38a`，LLVM 子模块 gitlink 为 `10dbfc4863c9aea2fb237022c3f47fc24c02265e`。复现时应使用这个项目自身的兼容工具链，不能直接用标准 MLIR 20.1.8 替代它。

## 8. 一条有限的源码阅读路线

只沿 mm 阅读即可形成闭环：

1. [fx.py](https://github.com/llvm/torch-mlir/blob/528d3dcb9a2288743b62df9f4937b1689787e38a/python/torch_mlir/fx.py)：找 export、decomposition、import 和后端 lowering 的接续。
2. [FxImporter](https://github.com/llvm/torch-mlir/blob/528d3dcb9a2288743b62df9f4937b1689787e38a/python/torch_mlir/extras/fx_importer.py)：确认 schema 到操作名及已有图 Value 到 SSA 引用的映射。
3. [Linear.cpp](https://github.com/llvm/torch-mlir/blob/528d3dcb9a2288743b62df9f4937b1689787e38a/lib/Conversion/TorchToLinalg/Linear.cpp)：跟 ConvertAtenMmOp 的 rank、shape、初始化与替换。
4. [Passes.cpp](https://github.com/llvm/torch-mlir/blob/528d3dcb9a2288743b62df9f4937b1689787e38a/lib/Conversion/Passes.cpp)与[basic.mlir](https://github.com/llvm/torch-mlir/blob/528d3dcb9a2288743b62df9f4937b1689787e38a/test/Conversion/TorchToLinalg/basic.mlir)：解释单 Pass 混合类型和完整后端边界的区别。

阅读后应能回答：为什么需要 fill；动态 K 不一致在哪里处理；一个局部转换成功后函数为什么还可能带 Torch 类型。把这三个答案连接起来，就已经能用此前学过的 MLIR 机制解释一项真实前端转换。
