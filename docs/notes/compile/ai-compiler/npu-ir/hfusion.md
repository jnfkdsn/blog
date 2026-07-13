---
order: 9
title: HFusion IR
updated: 2026-07-13
tags: [ai-compiler, npu, mlir, hfusion, fusion]
status: draft
---

# HFusion IR

相关入口：[NPU IR 融合知识库](/notes/compile/ai-compiler/npu-ir/)

HFusion IR 把 arith / math / linalg 等来源的表达整理成更适合融合、tiling、bufferization 和 NPU lowering 的结构化 op。它的重点不是写得短，而是让后端能直接看到：

```text
op 类别
输入 / 输出
输出目标或 init tensor
shape / dtype
具体计算函数
reduce axis 或特殊属性
```

## 基本形状

一个常见的 HFusion op：

```text
%empty = tensor.empty() : tensor<1024xf32>

%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
       ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>)
       outs(%empty : tensor<1024xf32>)
       -> tensor<1024xf32>
```

这段 IR 表示：

```text
逐元素二元计算：maxf
输入：%arg0, %arg1
输出目标：%empty
结果类型：tensor<1024xf32>
```

`ins / outs / ->` 是读 HFusion op 时最重要的三个位置：

- `ins`：读哪些输入。
- `outs`：写到哪个 destination / init tensor。
- `->`：返回的 tensor 类型。

## Structured Op

很多 HFusion op 继承 `HFusionStructuredBase_Op`，并带有类似接口：

```text
DestinationStyleOpInterface
LinalgStructuredInterface
ReifyRankedShapedTypeOpInterface
```

这类 op 接近 linalg structured op：有明确输入、输出、迭代空间和 shape 推导能力。

它适合支撑：

- fusion
- tiling
- bufferization
- shape 推导
- auto schedule
- lowering 到 NPU kernel 或 library boundary

典型 structured op：

```text
hfusion.elemwise_binary
hfusion.elemwise_unary
hfusion.cast
hfusion.compare
hfusion.reduce_with_index
hfusion.arange
hfusion.gather
```

## 常见 Op

### elemwise_unary

一输入一输出，逐元素操作。

```text
%empty = tensor.empty() : tensor<128xf32>

%0 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
       ins(%arg0 : tensor<128xf32>)
       outs(%empty : tensor<128xf32>)
       -> tensor<128xf32>
```

语义：

```text
%0[i] = sqrt(%arg0[i])
```

来源可能是 `math.sqrt`、`linalg.generic` 或 `torch.aten.sqrt` 一类表达。

### elemwise_binary

二输入一输出，逐元素操作。

```text
%empty = tensor.empty() : tensor<128xf32>

%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
       ins(%x, %y : tensor<128xf32>, tensor<128xf32>)
       outs(%empty : tensor<128xf32>)
       -> tensor<128xf32>
```

语义：

```text
%0[i] = powf(%x[i], %y[i])
```

HFusion 也可以表达 tensor + scalar-like input：

```text
%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
       ins(%x, %cst : tensor<?x256xf32>, f32)
       outs(%out : tensor<?x256xf32>)
       -> tensor<?x256xf32>
```

这里不需要先把 scalar broadcast 成 tensor。HFusion op 本身已经记录了一个输入是 tensor、另一个输入是 scalar。

### cast

类型转换。

```text
%empty = tensor.empty() : tensor<512xf32>

%0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
       ins(%arg0 : tensor<512xbf16>)
       outs(%empty : tensor<512xf32>)
       -> tensor<512xf32>
```

语义：

```text
tensor<512xbf16> -> tensor<512xf32>
round mode = rint
```

`LegalizeBF16Pass` 这类 pass 可能会生成这种形式。

### compare

比较操作，输出通常是 `i1` tensor。

```text
%empty = tensor.empty() : tensor<32xi1>

%0 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
       ins(%a, %b : tensor<32xf32>, tensor<32xf32>)
       outs(%empty : tensor<32xi1>)
       -> tensor<32xi1>
```

语义：

```text
%0[i] = (%a[i] != %b[i])
```

可能来源：

```text
arith.cmpf une, %a, %b : tensor<32xf32>
```

### reduce_with_index

用于 `argmax / argmin` 这类同时需要 reduce value 和 index 的操作。

```text
%reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input : tensor<256x64xf32>)
    outs(%init_val, %init_idx : tensor<256xf32>, tensor<256xi32>)
    dimensions = [1]
    -> tensor<256xf32>, tensor<256xi32>
```

语义：

```text
沿第 1 维做 max reduce
同时返回最大值和最大值所在 index
```

这类 op 的 lowering 比普通 elementwise 更复杂，因为它涉及 reduce axis、并行规约、tie-break 规则和 index dtype。

## 转换关系

HFusion 常作为更通用 dialect 的 lowering 目标之一。

例如 arith compare：

```text
%1 = arith.cmpf une, %arg1, %arg2 : tensor<32xf32>
```

可以变成：

```text
%empty = tensor.empty() : tensor<32xi1>

%ret = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
         ins(%arg1, %arg2 : tensor<32xf32>, tensor<32xf32>)
         outs(%empty : tensor<32xi1>)
         -> tensor<32xi1>
```

转换后的表达更长，但信息更规整：

```text
op 类别：compare
compare function：vne
输入 tensor：%arg1, %arg2
输出目标：%empty
输出类型：tensor<32xi1>
```

常见转换入口：

```text
test/Conversion/ArithToHFusion/arith-to-hfusion.mlir
test/Conversion/MathToHFusion/math-to-hfusion.mlir
test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir
```

## 为什么更啰嗦

普通表达：

```text
%0 = arith.addf %a, %b : tensor<1024xf32>
```

写起来短，但后端还要继续推断：

```text
结果 shape 是多少
输出 buffer 是谁
能不能 tiled
能不能 fusion
是不是 elementwise
是不是 scalar + tensor
```

HFusion 写法：

```text
%empty = tensor.empty() : tensor<1024xf32>

%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<addf>}
       ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
       outs(%empty : tensor<1024xf32>)
       -> tensor<1024xf32>
```

信息更适合后端 pass：

```text
这是 elementwise binary
函数是 addf
输入 shape 明确
输出目标明确
结果类型明确
可以走 linalg-style tiling / fusion / bufferization 分析
```

这里的“规整”不是短，而是语义分类、输入输出、shape 和后端可分析信息都明确。

## 规范化和 Pipeline

HFusion 相关 pass 通常会继续做规范化，目标是减少后续 pass 需要处理的 IR 形态。

常见测试入口：

```text
test/Dialect/HFusion/hfusion-normalize-ops.mlir
test/Dialect/HFusion/hfusion-inline-brc.mlir
test/Dialect/HFusion/hfusion-normalize-slice-ops.mlir
test/Dialect/HFusion/ops.mlir
```

源码入口：

```text
include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td
include/bishengir/Dialect/HFusion/IR/HFusionOps.td
lib/Dialect/HFusion/Pipelines/HFusionPipelines.cpp
```

`preProcess` 一类逻辑可以理解成把不同来源的表达整理成 HFusion 风格：分类清楚、输入输出明确、shape 可推导、后续可融合和可调度。

## 阅读检查项

看到一个 HFusion op，按下面顺序读：

```text
op 名字：elemwise_binary / cast / reduce_with_index
attrs：fun / round_mode / compare_fn / dimensions
ins：读哪些 value
outs：写到哪个 destination 或 init tensor
->：结果类型
```

检查一个 HFusion op 是否适合继续参与融合：

- 输入输出 shape 是否明确。
- dtype 是否与后续 op 兼容。
- scalar-like input 是否需要 broadcast 语义。
- reduce axis 是否明确。
- destination-style outs 是否能 bufferize。
- op attrs 是否足以还原原始语义。
- 后端是否支持对应 lowering。
