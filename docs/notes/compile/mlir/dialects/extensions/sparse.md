---
order: 20
title: 稀疏编码与按存储坐标生成循环
updated: 2026-10-04
---

# 稀疏编码与按存储坐标生成循环

矩阵中有许多零时，仍逐坐标读取并乘加可能浪费存储和工作量。稀疏表示把逻辑矩阵和实际保存的条目分开：数学上仍是二维张量，物理上却由位置、坐标和值数组描述。

本章沿一个2×3矩阵的乘向量过程，看 sparse_tensor 编码如何让 Linalg 的完整迭代关系变成只遍历已存储条目的循环。

## 1. 从逻辑矩阵到 CSR

取：

```text
A = [[2, 0, 3],     x = [10, 20, 30]
     [0, 4, 0]]
```

按行压缩时，一种CSR表示为：

```text
positions = [0, 2, 3]
coordinates = [0, 2, 1]
values = [2, 3, 4]
```

第0行使用条目区间[0,2)，对应列0、2；第1行使用[2,3)，对应列1。positions是条目边界，coordinates才是逻辑列号。把条目编号p直接当列号，会读错x。

输出从零开始时，第一行得到2×10+3×30=110，第二行得到4×20=80。若初值为[1,2]，则得到[111,82]，因为本例Linalg语义是累加到outs初值。

## 2. Tensor 形状与存储 Level

MLIR用encoding保存这项存储约定：

```text
#csr = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>
```

`tensor<2x3xf32,#csr>`仍有逻辑形状2×3；map描述逻辑维怎样映射到存储level。这里行level为dense，列level为compressed。更复杂编码可以重排或拆分维度，因此dimension与level不应永远当成同义词。

compressed并不等于“运行时自动把所有零删除”。输入需要符合选定格式的坐标、排序和唯一性等契约，显式保存的零也可能存在。格式转换、构造和释放各有相应操作/运行时路径。[SparseTensor规范](https://mlir.llvm.org/docs/Dialects/SparseTensorOps/)定义这些表示。

## 3. 保留原来的数学迭代关系

稀疏输入仍可进入熟悉的linalg.generic：

```text
indexing_maps = [(i,j)->(i,j), (i,j)->(j), (i,j)->(i)]
iterator_types = ["parallel", "reduction"]
```

区域仍然计算acc+A×x。变化发生在如何枚举有效贡献：稀疏编译器结合访问关系与encoding，选择positions/coordinates/values上的遍历。

固定实验先运行 sparse-reinterpret-map，再运行 sparsification。前一步建立该版本稀疏变换使用的映射形式；只运行后一条Pass时，本例仍保留linalg.generic。Pass名称正确并不保证所有前置条件已满足。

## 4. 实际生成的存储循环

转换后可看到以下结构，省略了具体SSA名字：

```text
for i = 0 .. 2:
  acc = out[i]
  begin = positions[i]
  end = positions[i+1]
  for p = begin .. end:
    j = coordinates[p]
    acc += values[p] * x[j]
  out[i] = acc
```

实际IR包含sparse_tensor.positions、coordinates、values，随后是memref.load和scf.for。内层循环界限来自存储条目数量，列号来自coordinates，恰好对应第一节的手算。

该输出仍包含稀疏类型、bufferization边界和存储访问，不是最终机器码。继续编译还需选择稀疏运行时或相应codegen、处理构造/释放与ABI，不能把这条局部转换当成完整可执行pipeline。

## 5. 跳过零的语义与代价

当前有限数值例子可以直接按非零项推演，但一般浮点程序还有NaN/Inf、符号零和归约顺序等边界。省略乘零或重新排序是否允许，必须与源操作及项目采用的稀疏语义一致，不能只依据实数代数。

性能也取决于稀疏率、行长度分布、坐标开销、访问局部性和并行负载。一个很小矩阵或不规则索引可能比密集kernel更慢；编码选择与算子调度应共同评估。

本章验证的是类型、转换与具体存储模型，不包含稀疏运行时执行或性能承诺。学习者可先把第二行设为空行，推演positions为何变为[0,2,2]，以及初值如何保留。

## 依据与实践

固定来源：[Encoding定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td)、[Sparsification实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/SparseTensor/Transforms/Sparsification.cpp)、[完整管线](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/SparseTensor/Pipelines/SparseTensorPipelines.cpp)。[25 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/25-domain-extensions)保留完整generic、实际循环和rank不匹配诊断；110/80及初值变体来自明确标注的存储模型。
