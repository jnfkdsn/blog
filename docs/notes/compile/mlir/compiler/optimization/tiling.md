---
order: 1
title: Tiling 与边界处理
updated: 2026-10-04
---

# Tiling 与边界处理

一个矩阵计算可以按完整行遍历，也可以先处理一小块，再换到下一块。后者叫 tiling：把原来的迭代域划分为局部块，在块之间和块内部安排执行顺序。

分块的价值来自后续局部复用、向量化或设备映射。首先必须解决正确性：每个有效点执行恰好一次，余块不越界，原有依赖顺序仍然成立。本章用 5×7 逐元素加一推导这一过程，再观察 Tensor 表示中的切片与更新。

## 1. 原始迭代域

输入输出是不同的 5×7 存储，当前每个点读取 src[i,j]，加 1 后写到 out[i,j]。这项计算没有跨迭代依赖：

<!-- structured-example: affine-add -->
```text
module {
  func.func @affine_add(%src: memref<5x7xi32>, %out: memref<5x7xi32>) attributes {llvm.emit_c_interface} {
    %one = arith.constant 1 : i32
    affine.for %i = 0 to 5 {
      affine.for %j = 0 to 7 {
        %v = affine.load %src[%i, %j] : memref<5x7xi32>
        %r = arith.addi %v, %one : i32
        affine.store %r, %out[%i, %j] : memref<5x7xi32>
      }
    }
    return
  }
}
```

这里输入输出需有效且不重叠。若输出窗口覆盖随后还要读取的输入，原先的“各点独立”就不成立，不能沿用当前证明。

## 2. 2×3 分块的执行范围

选择行块大小 2、列块大小 3，块起点为行 `[0,2,4]` 与列 `[0,3,6]` 的组合，共 9 块：

| 块行起点 | 列起点 0 | 列起点 3 | 列起点 6 |
|---|---|---|---|
| 0 | 2×3 | 2×3 | 2×1 |
| 2 | 2×3 | 2×3 | 2×1 |
| 4 | 1×3 | 1×3 | 1×1 |

最后一块只包含坐标 `(4,6)`。实际域应写成 `i∈[bi,min(bi+2,5))` 与 `j∈[bj,min(bj+3,7))`，不能把九块都当成完整 2×3。

为这个输入调用 `affine-loop-tile`，指定 `tile-sizes=2,3`，工具生成四层循环：两层选择块起点，两层选择块内元素。关键结构为：

```text
#rows = affine_map<(bi) -> (bi+2,5)>
#cols = affine_map<(bj) -> (bj+3,7)>
affine.for %bi = 0 to 5 step 2 {
  affine.for %bj = 0 to 7 step 3 {
    affine.for %i = %bi to min #rows(%bi) {
      affine.for %j = %bj to min #cols(%bj) {
        // 原来的 src[i,j]+1 → out[i,j]
      }
    }
  }
}
```

这是按实际输出归一化变量名后的片段，完整可运行输出保存在实验。读它时要区分 step=块大小的外循环与 step=1 的内循环。

## 3. 从全局坐标到局部切片

若计算仍采用 Tensor/Linalg 表示，通常不会直接把 scalar body 展开成四层循环。可以让外两层 SCF 选择 tile，用 extract_slice 取输入块，在块内保留一个较小 Linalg 计算，再把结果 insert_slice 回完整输出。

对于最后一块 `(bi,bj)=(4,6)`：

```text
实际行长 = min(2,5-4) = 1
实际列长 = min(3,7-6) = 1
取输入 slice [4,6] [1,1] [1,1]
在局部 (0,0) 计算，得到全局位置 (4,6) 的结果
把 1×1 结果插回输出的 [4,6]
```

动态 M/N 时，实际 tile 长度仍需在运行时计算。配套调度对二维 elementwise 计算产生的关键序列是 `affine.min → tensor.extract_slice → linalg.generic → tensor.insert_slice`。

输出 tensor 通过 `scf.for iter_args` 逐块传递：前一块写入的新结果成为下一块的初始整体值。随后 Bufferization 可以在满足条件时把这些值更新落到同一输出分配，不能把每个 SSA tensor 名字都解释成一次完整复制。

## 4. 保持原始计算的条件

分块正确需要同时满足域、访问和依赖条件。域要求完整且不重复地覆盖原有效点；访问要求局部坐标与原坐标对应一致；依赖要求新的块顺序和块内顺序没有让消费者越过生产者。

逐元素加一满足独立性，因此当前 2×3 顺序是合法的。对于[跨行递推](../../dialects/affine/loops_analysis)，需要把其 `(1,-1)` 依赖带入新顺序检查；不能仅看到“所有点最终都会被访问”就批准任意 tiling。

归约维度也要区别处理。矩阵乘法对 K 分块时，每个 K tile 必须继续累加到同一输出 tile 的已有结果，不能在每个 K tile 开始都把输出清零。若进一步改成独立部分和再合并，还会涉及浮点次序变化。

## 5. 分块的成本与用途

对当前单次逐元素加一，分块并没有创造新的输入复用：每个元素本来就只需读取一次。额外的循环与 min 还可能增加控制开销。因此这个案例用于验证分块机制和边界，不能拿“多了 tile”当作性能提升。

当一块输入会被多次使用，例如 Matmul 中一个 A 块参与多个输出列，或者 producer 与 consumer 在同一 tile 内结合，分块才可能缩短复用距离、限制工作集并匹配向量或设备资源。

tile 过小会增加控制与边界开销，过大可能超出缓存、寄存器或局部存储预算。合适大小依赖计算、布局和目标，后面的性能篇再用明确的机器与计时条件比较。

## 6. 边界验证

实验同时核对原始和分块后的 5×7 输出，每个坐标都使用不同输入，并用输出前后的哨兵检测越界写。Tensor 版本还核对 1×1、1×7、2×3、7×5 与空形状，避免只在整除 tile 的输入上成立。

原始、融合、分块与交换后的逐元素函数都按同一个 `2*(x+1)` 参考核对。这里的分块脚本保证可复现，Transform 的 handle 与执行协议在后续专篇说明；理解本章需要掌握的是分块前后的计算与边界。

## 阅读自查

1. 最后一块为什么是 1×1，而不是 2×3 或 1×3？
2. Tensor tiling 中，为什么循环要携带并返回完整输出 tensor？
3. 同样覆盖所有点的调度，为什么仍可能破坏递推或归约？

下一篇[融合与计算重排](./fusion)观察 tile 内 producer 与 consumer 怎样减少中间值。

## 实现依据与实践

Affine 实际输出来自 [LoopTiling.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp)，Linalg tiling 的固定版本调度参照 [transform-op-tile.mlir](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/test/Dialect/Linalg/transform-op-tile.mlir)。[结构化优化实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/14-structured-optimization)保存全部命令、阶段与数值，没有把结构正确当成性能结论。
