**HFusion IR **

---

**HFusion IR 的基本形状**

先看一个最常见的 HFusion op：

```mlir
%empty = tensor.empty() : tensor<1024xf32>

%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
       ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>)
       outs(%empty : tensor<1024xf32>)
       -> tensor<1024xf32>
```

你可以把它读成：

```text
用 maxf 做逐元素二元计算：
输入：%arg0、%arg1
输出 buffer / init tensor：%empty
结果类型：tensor<1024xf32>
```

重点看这三个部分：

```mlir
ins(...)
outs(...)
-> tensor<...>
```

这是 MLIR linalg 风格的 destination-style 写法。HFusion 很多 op 也沿用了这个形式。

为什么要有 `outs`？

因为后端很关心“结果写到哪里”。对于 bufferization、tiling、fusion 来说，这比纯 SSA 表达更接近真实内存行为。

普通数学表达可能是：

```text
c = a + b
```

HFusion 更喜欢表达成：

```text
把 a 和 b 读进来，把结果写到 out 这个目标里
```

所以写成：

```mlir
ins(%a, %b)
outs(%out)
```

---

**第 2 课：HFusion structured op 是什么**

在 [HFusionStructuredOps.td](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td) 里，很多 HFusion op 都继承了类似这样的基类：

```tablegen
HFusionStructuredBase_Op
```

它带了几个重要接口：

```text
DestinationStyleOpInterface
LinalgStructuredInterface
ReifyRankedShapedTypeOpInterface
```

暂时不用怕这些名字。你先记成：

```text
HFusion structured op = 像 linalg 一样有输入、输出、迭代空间、shape 推导能力的 op
```

这类 op 适合做：

- fusion
- tiling
- bufferization
- shape 推导
- auto schedule

典型例子：

```mlir
hfusion.elemwise_binary
hfusion.elemwise_unary
hfusion.cast
hfusion.compare
hfusion.reduce_with_index
hfusion.arange
hfusion.gather
```

---

**第 3 课：先学最核心的 5 类 HFusion op**

我建议你先只学这几类，不要一口气看完所有 op。

第一类：`hfusion.elemwise_unary`

一输入一输出，逐元素操作。

```mlir
%empty = tensor.empty() : tensor<128xf32>

%0 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
       ins(%arg0 : tensor<128xf32>)
       outs(%empty : tensor<128xf32>)
       -> tensor<128xf32>
```

意思是：

```text
%0[i] = sqrt(%arg0[i])
```

对应来源可能是：

```mlir
math.sqrt
linalg.generic
torch.aten.sqrt
```

第二类：`hfusion.elemwise_binary`

二输入一输出，逐元素操作。

```mlir
%empty = tensor.empty() : tensor<128xf32>

%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
       ins(%x, %y : tensor<128xf32>, tensor<128xf32>)
       outs(%empty : tensor<128xf32>)
       -> tensor<128xf32>
```

意思是：

```text
%0[i] = powf(%x[i], %y[i])
```

注意，HFusion 允许有些输入是 scalar-like：

```mlir
%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>}
       ins(%x, %cst : tensor<?x256xf32>, f32)
       outs(%out : tensor<?x256xf32>)
       -> tensor<?x256xf32>
```

这就是我前面说的“更规整”：不用先把 scalar broadcast 成 tensor，而是让 HFusion op 直接知道“一个输入是 tensor，一个输入是 scalar”。

第三类：`hfusion.cast`

类型转换。

```mlir
%empty = tensor.empty() : tensor<512xf32>

%0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
       ins(%arg0 : tensor<512xbf16>)
       outs(%empty : tensor<512xf32>)
       -> tensor<512xf32>
```

意思是：

```text
bf16 tensor 转成 f32 tensor
```

`LegalizeBF16Pass` 经常会制造这种形式。

第四类：`hfusion.compare`

比较操作，输出通常是 `i1` tensor。

```mlir
%empty = tensor.empty() : tensor<32xi1>

%0 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
       ins(%a, %b : tensor<32xf32>, tensor<32xf32>)
       outs(%empty : tensor<32xi1>)
       -> tensor<32xi1>
```

意思是：

```text
%0[i] = (%a[i] != %b[i])
```

它可能来自：

```mlir
arith.cmpf une, %a, %b : tensor<32xf32>
```

第五类：`hfusion.reduce_with_index`

用于 `argmax / argmin` 这种“既要 reduce 值，又要 index”的操作。

```mlir
%reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <max>
    ins(%input : tensor<256x64xf32>)
    outs(%init_val, %init_idx : tensor<256xf32>, tensor<256xi32>)
    dimensions = [1]
    -> tensor<256xf32>, tensor<256xi32>
```

意思是：

```text
沿第 1 维做 max reduce
同时返回最大值和最大值所在 index
```

---

**第 4 课：读懂 HFusion IR 的固定套路**

看到一个 HFusion op，你按这个顺序读：

```text
1. op 名字：hfusion.elemwise_binary / hfusion.cast / hfusion.reduce_with_index
2. 属性：{fun = ...}、{round_mode = ...}、dimensions = [...]
3. ins：读哪些输入
4. outs：写到哪个目标 / init tensor
5. ->：结果类型
```

比如：

```mlir
%3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
       ins(%2 : tensor<512xbf16>)
       outs(%0 : tensor<512xf32>)
       -> tensor<512xf32>
```

读法：

```text
%3 是一个 cast
把 %2 从 tensor<512xbf16> 转成 tensor<512xf32>
round mode 是 rint
输出目标是 %0
结果类型是 tensor<512xf32>
```

---

**第 5 课：HFusion 和普通 MLIR 的转换关系**

你可以拿这些测试作为练习：

- arith 到 HFusion：[arith-to-hfusion.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Conversion/ArithToHFusion/arith-to-hfusion.mlir)
- math 到 HFusion：[math-to-hfusion.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Conversion/MathToHFusion/math-to-hfusion.mlir)
- linalg 到 HFusion：[linalg-to-hfusion.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Conversion/LinalgToHFusion/linalg-to-hfusion.mlir)

比如这个转换：

```mlir
%1 = arith.cmpf une, %arg1, %arg2 : tensor<32xf32>
```

会变成：

```mlir
%empty = tensor.empty() : tensor<32xi1>

%ret = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
         ins(%arg1, %arg2 : tensor<32xf32>, tensor<32xf32>)
         outs(%empty : tensor<32xi1>)
         -> tensor<32xi1>
```

这就是“从通用 arith 表达，变成 HFusion 结构化表达”。

---

**第 6 课：为什么 HFusion IR 看起来比普通表达啰嗦**

因为它不是给人手写算法用的，它是给后端优化用的。

普通表达：

```mlir
%0 = arith.addf %a, %b : tensor<1024xf32>
```

很短，但后端要自己推断：

```text
结果 shape 是多少？
输出 buffer 是谁？
能不能 tiled？
能不能 fusion？
是不是 elementwise？
是不是 scalar + tensor？
```

HFusion 写法：

```mlir
%empty = tensor.empty() : tensor<1024xf32>

%0 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<addf>}
       ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
       outs(%empty : tensor<1024xf32>)
       -> tensor<1024xf32>
```

更啰嗦，但信息更适合后端：

```text
这是 elementwise binary
函数是 addf
输入 shape 明确
输出目标明确
结果类型明确
可以被 linalg-style tiling/fusion 分析
```

所以“规整”的意思不是短，而是：

```text
语义分类清楚
输入输出清楚
shape 清楚
后端 pass 不用猜
```

---

**第 7 课：一条推荐学习路线**

你可以按这个顺序学，每一步都对应仓库里的文件。

1. 看 `hfusion.elemwise_binary / unary / cast / compare`
   读：
   [arith-to-hfusion.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Conversion/ArithToHFusion/arith-to-hfusion.mlir)
   [math-to-hfusion.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Conversion/MathToHFusion/math-to-hfusion.mlir)

2. 看 `ins/outs/->` 的 destination-style
   读：
   [HFusionStructuredOps.td](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td)

3. 看特殊 op：`interleave / deinterleave / flip / histogram / matmul_mx`
   读：
   [HFusionOps.td](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td)
   [ops.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Dialect/HFusion/ops.mlir)

4. 看规范化 pass 改了什么
   读：
   [hfusion-normalize-ops.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Dialect/HFusion/hfusion-normalize-ops.mlir)
   [hfusion-inline-brc.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Dialect/HFusion/hfusion-inline-brc.mlir)
   [hfusion-normalize-slice-ops.mlir](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/test/Dialect/HFusion/hfusion-normalize-slice-ops.mlir)

5. 最后再看完整 pipeline
   读：
   [HFusionPipelines.cpp](E:/jnfkdsn/Desktop/compiler/ascendnpu-ir/bishengir/lib/Dialect/HFusion/Pipelines/HFusionPipelines.cpp)

---

**小练习**

你可以先练习读这个：

```mlir
%empty = tensor.empty() : tensor<256xf32>

%0 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
       ins(%arg0 : tensor<256xf32>)
       outs(%empty : tensor<256xf32>)
       -> tensor<256xf32>
```

答案是：

```text
创建一个 256 个 f32 的输出 tensor
对 %arg0 的每个元素做 sqrt
结果写到 %empty 对应的输出里
最终返回 tensor<256xf32>
```

再读这个：

```mlir
%empty = tensor.empty() : tensor<256xi1>

%0 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
       ins(%a, %b : tensor<256xf32>, tensor<256xf32>)
       outs(%empty : tensor<256xi1>)
       -> tensor<256xi1>
```

答案是：

```text
逐元素比较 %a 和 %b 是否不相等
输出是 bool tensor，也就是 tensor<256xi1>
```

你现在学习 HFusion IR，最重要的不是先记所有 op，而是先形成这个阅读习惯：

```text
op 名字告诉你它是哪类计算
attribute 告诉你具体函数/模式
ins 告诉你读什么
outs 告诉你写到哪里
-> 告诉你结果类型
```

后面你再看 `preProcess`，就会发现它做的事情其实很朴素：把各种来源的表达，整理成这种可读、可分析、可调度的 HFusion 风格。