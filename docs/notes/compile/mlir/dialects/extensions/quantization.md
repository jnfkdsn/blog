---
order: 10
title: 量化类型与整数计算的边界
updated: 2026-10-04
---

# 量化类型与整数计算的边界

把一个浮点张量换成8位整数存储，可以减少数据量，但整数本身并不能说明原来的数值。若存储值128表示实数0，就需要把这个对应关系保留下来，让后续计算正确处理乘法、累加和输出缩放。

MLIR 的 Quant 类型表达这种数值映射。它不是一条自动把任意模型变成高效整数 kernel 的 pipeline；本章先追踪一个数怎样变为整数、怎样恢复，再说明还需要哪些编译工作。

## 1. 存储值与表达值

本例采用均匀量化，scale=0.25，zero point=128，存储范围0到255：

```text
real ≈ (q - 128) × 0.25
q = clamp(round(real / 0.25) + 128, 0, 255)
```

选择明确的最近偶数舍入约定后，几个例子为：

| real | q | 恢复值 |
|---:|---:|---:|
| -0.5 | 126 | -0.5 |
| 0 | 128 | 0 |
| 0.75 | 131 | 0.75 |
| 0.3 | 129 | 0.25 |
| 40 | 255 | 31.75 |

前两种误差来源不同：0.3落在量化网格之间，需要舍入；40超过表示范围，需要饱和。不能把所有误差都称为“少了几位精度”。[Quant 设计](https://mlir.llvm.org/docs/Quantization/)给出存储值、表达值、scale和zero point的对应。

## 2. 类型保存映射，操作表达边界

对应的 MLIR 类型为：

```text
!quant.uniform<u8:f32, 0.25:128>
```

u8描述存储范围，f32是表达类型，后面保存scale与zero point。一个小片段是：

```text
%q = quant.qcast %x
    : tensor<4xf32> to tensor<4x!quant.uniform<u8:f32, 0.25:128>>
%y = quant.dcast %q
    : tensor<4x!quant.uniform<u8:f32, 0.25:128>> to tensor<4xf32>
```

qcast引入量化，dcast回到表达类型。两者连起来也不一般等于原输入：表中的0.3已经变为0.25。编译器不能像删除两个无损cast那样无条件消去它们。

scast则连接量化类型与原始整数存储。例如从该量化张量得到`tensor<4xi8>`，仍保留同样的整数位模式；它不执行上面的反量化公式。固定实验把目标改为i16会被拒绝，因为存储类型不匹配。

## 3. 整数 Matmul 还缺什么

若A、B分别使用scale sA/sB和zero point zA/zB，实数乘积和可表示为：

```text
acc = Σ (qA[k] - zA) × (qB[k] - zB)
real_result ≈ sA × sB × acc
```

为了输出到另一个量化网格，还需根据输出scale重新缩放、舍入、加输出zero point并裁剪。累加位宽也必须足够，不能把输入的8位存储直接当成8位累加器。

因此从Quant类型到整数硬件路径，需要选择/构造支持相应规则的操作，再lower到目标指令或库。不同项目可以把参数保存在类型或操作属性中；重要的是消费者在计算时取得正确约定，而不是必须出现某一固定方言链。

per-axis量化进一步为某个轴的不同位置使用不同参数，例如不同输出通道有不同scale。此时reshape、transpose和布局变换必须同步解释参数轴，不能只移动数据类型名称。

## 4. 校准、混合精度与验证

scale和zero point从哪里来，是另一个问题：可能来自模型显式参数、校准统计或训练过程。Quant类型保存选定结果，不自动证明校准分布适合未来输入。

不支持整数实现的部分也可以保留浮点计算，在边界插入转换。优化需要同时考虑精度损失、转换开销和目标支持。仅减少张量位宽不保证端到端变快。

本章IR经过固定版本解析和往返，数值表由独立公式模型核对。没有执行量化模型或测量整数kernel；下一阶段若选择真实量化项目，应补算子约定、输出误差、校准数据与目标指令证据。

## 依据与实践

固定定义：[QuantTypes](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Quant/IR/QuantTypes.h)、[QuantOps](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Quant/IR/QuantOps.td)。[25 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/25-domain-extensions)包含qcast/dcast/scast及存储类型反例。有限练习：改变zero point后重新计算实数0的整数表示，解释padding应该填什么。
