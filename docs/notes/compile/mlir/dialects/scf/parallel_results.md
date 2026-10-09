---
order: 50
title: 并行归约与张量结果组装
updated: 2026-10-04
---

# 并行归约与张量结果组装

顺序scf.for用iter_args把上一轮状态传给下一轮。并行迭代不能普遍依赖这种顺序：多个迭代可能同时进行，谁先完成不由源程序的线性顺序决定。

当结果是一个总和，需要明确怎样合并每个迭代的贡献；当结果是一个张量，需要说明各迭代负责哪一片输出。scf.parallel的reduce与scf.forall的shared_outs分别表达这两类关系，本章用一个整数和与一个四元素张量比较它们。

## 1. Parallel 的贡献与归约函数

希望计算init+0+1+…+(n-1)：

```text
%r = scf.parallel (%i) = (%zero) to (%n) step (%one)
    init(%init) -> i32 {
  %v = arith.index_cast %i : index to i32
  scf.reduce(%v : i32) {
  ^bb0(%a: i32, %b: i32):
    %s = arith.addi %a, %b : i32
    scf.reduce.return %s : i32
  }
}
```

外层body每次产生一个贡献v；reduce中的a、b是待合并的两个值，不是“当前i和上一轮结果”。init参与归约，结果最终交给r。

init=5、n=4时得到5+0+1+2+3=11；n=0时没有贡献，结果保留5。固定CPU参考路径验证了n从0到4的这些边界。

## 2. 并行语义不承诺顺序 Fold

parallel允许迭代按不同顺序执行，归约的组合方式也不能按顺序for的左折叠假定。整数模加法在本例中适合组合；浮点加法改变顺序可能改变舍入，带额外overflow承诺的整数计算也要重新检查条件。

多个归约结果分别对应init、reduce operand和reduce region，三者位置需要一致。无结果parallel仍以无operand、无region的scf.reduce终结；该终结符本身不表示“自动执行求和”。

并行body中若对同一个普通内存位置无同步写入，并不会因为使用parallel就安全。归约协议或目标同步必须表达真实依赖，不能把数据竞争当成合法调度差异。

## 3. Forall 的 Shared Output

另一项任务是生成[0,1,2,3]。各迭代只负责一个不同位置，可以共享一个目的张量：

```text
%r = scf.forall (%i) in (4)
    shared_outs(%out = %empty) -> (tensor<4xi32>) {
  %v = arith.index_cast %i : index to i32
  %tile = tensor.from_elements %v : tensor<1xi32>
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %tile into %out[%i][1][1]
      : tensor<1xi32> into tensor<4xi32>
  }
}
```

out是body中代表共享输出的Block参数，r是forall完成后的结果。parallel_insert_slice声明当前迭代把长度1的tile放到偏移i处，四次写入互不重叠，完整覆盖结果。

empty初值没有有效数据，但这里每个结果位置都被覆盖，后续读取才有依据。若只覆盖部分位置，就需要有效的初始张量，不能让未初始化部分进入结果使用者。

## 4. 结果组装与归约的区别

forall.in_parallel区域表达并行结果的组合动作；它不是普通顺序执行的一组store。parallel_insert_slice也不是重叠写入时自动求和的atomic add。

本例由切片范围证明互不重叠。若两个迭代都写同一个位置，需要另一个明确的合并协议；不能依赖“最后一个写入获胜”，因为并行执行没有这样的确定顺序。

forall还可以携带mapping信息，为后续映射到设备层次提供依据。但没有mapping时，它本身也不会凭空选择GPU block/thread。具体映射继续见[目标并行映射](../../compiler/targets/parallel_mapping)。

## 5. 串行参考 Lowering 与真实并行

作者将含tensor的forall先bufferize，处理分配与释放，再转换SCF到CF并生成CPU代码。这个参考路径将并行表示落为顺序循环，实际读取结果位置0和3，得到0+3=3。

顺序执行是本例无竞争程序的一种合法实现，可以帮助核对结果与生命周期，但不证明GPU/NPU映射或并行性能。反过来，CPU串行结果正确也不能为一个有潜在竞争的并行body提供安全证明。

parallel的归约与forall的张量结果，都需要在最终目标上兑现相应的合并、存储与同步条件。先理解源协议，再选择设备实现，才能区分算法正确性和调度收益。

## 依据与实践

固定来源：[SCFOps](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)、[Tensor切片操作](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td)、[SCF转换](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp)。

[26 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/26-ir-maintenance)保留归约初值/空迭代和forall结果的CPU参考；设备映射证据归16工程。有限练习：把forall改为只更新偶数位置，说明为什么应以有效初值替换empty，并预测未更新位置。
