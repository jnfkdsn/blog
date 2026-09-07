---
order: 5
title: 内存、效果与优化边界
updated: 2026-09-06
---

# 内存、效果与优化边界

前置：SSA、类型、arith 与 SCF if/for。本章解释值语义、存储、别名、内存效果与推测执行，建立判断改写是否可能正确的基础。Tensor/MemRef 的完整操作参考、bufferization 算法与释放流程列入后续覆盖范围。

## 1. SSA 保证引用关系，不保证存储内容不变

完整模块中，两次 load 使用同一个 `%buffer`，中间有一次写入：

<!-- mlir-example: effects-mutable-buffer -->
```text
module {
  func.func @read_write_read(%buffer: memref<1xi32>) -> (i32, i32) {
    %i = arith.constant 0 : index
    %seven = arith.constant 7 : i32
    %before = memref.load %buffer[%i] : memref<1xi32>
    memref.store %seven, %buffer[%i] : memref<1xi32>
    %after = memref.load %buffer[%i] : memref<1xi32>
    return %before, %after : i32, i32
  }
}
```

假设调用前该元素为 3，且存储有效，两次 load 分别得到 3 和 7。`%buffer` 作为 SSA Value 始终由同一个 BlockArgument 定义；通过它引用的内存可以变化。`%before` 的标量结果仍是先前读出的值，不会因后续 store 自动变成 7。

因此不能仅凭“两个 load 的 operand 完全相同”就把第二次 load 替换成第一次。必须分析中间写入是否可能影响它读取的位置。

## 2. Tensor 更新产生新值

与可变存储不同，普通 tensor 操作用值语义描述计算。下例的 `tensor.insert` 返回新 tensor Value，旧 `%t` 的抽象内容保持不变：

<!-- mlir-example: effects-tensor-value -->
```text
module {
  func.func @updated_copy(%t: tensor<4xi32>, %v: i32)
      -> (tensor<4xi32>, tensor<4xi32>) {
    %i = arith.constant 0 : index
    %updated = tensor.insert %v into %t[%i] : tensor<4xi32>
    return %t, %updated : tensor<4xi32>, tensor<4xi32>
  }
}
```

新 Value 不等于实现时必然分配一份全新 buffer。后续 bufferization 可以在证明没有读写冲突等条件时复用存储，也可以选择分配与复制。必须保持的是原来 tensor 程序可观察的值语义。

本例同时返回旧值和更新值，正好提醒我们不能随意原地修改旧内容。是否实际需要复制，还取决于更完整的使用和别名分析，不能只凭此片段预测具体内存分配数量。

## 3. MemRef 是带结构信息的存储视图

MemRef 的抽象不仅有“地址”，还涉及元素类型、rank/shape、layout、stride 和 memory space 等信息。不同 memref Value 可以引用同一片存储，甚至重叠的子区域；Value 不同不等于内存不别名。

`memref.subview` 通常建立已有存储的视图，不是拷贝内容。默认布局的 memref 也不能与所有 strided view 随意互换类型。经过 lowering 后常用描述符表达基址、对齐地址、offset、sizes、strides；具体形式属于目标转换约定，不要把 MLIR 内存对象当成运行时描述符。

存储还具有生命周期：动态分配的 buffer 可能需要释放，自动分配有所属作用域，函数参数通常由调用方提供。memref 的 SSA 引用仍可出现在 IR 中，不代表其运行时存储一定还活着。访问已释放存储、越界或错误对齐等问题不是“SSA verifier 通过”就消失了。

## 4. 内存效果补足纯 SSA 数据流

如果只画 Value 的 use-def 图，`store` 没有结果，看起来像“什么都没产生”。但它改变可观察存储，所以需要额外描述操作效果。

| 效果 | 典型含义 | 例子 |
|---|---|---|
| Read | 读取资源的状态 | `memref.load` |
| Write | 修改资源的状态 | `memref.store` |
| Allocate | 创建资源/存储 | `memref.alloc` |
| Free | 结束存储生命周期 | `memref.dealloc` |

`MemoryEffectsOpInterface` 允许操作声明或查询这些效果，并可关联 Value 或抽象 Resource。效果模型不只用于普通堆内存，也可表达其他可变资源；复杂模型还可区分阶段和作用范围，后续实现章节再展开。

没有实现效果接口的未知操作，不能默认没有效果。`scf.for` 等带 Region 的操作还需要考虑内部操作的效果：循环表面没有结果，也可能在循环体里写入大量内存。

## 5. 无内存效果不等于可无条件提前执行

把条件分支里的计算移动到分支前，会让原本不执行它的路径也执行它。这种变换涉及 speculation（推测执行），不仅是内存读写。

完整模块先检查除数，只有非零时才执行除法：

<!-- mlir-example: effects-guarded-division -->
```text
module {
  func.func @safe_div(%x: i32, %divisor: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %ok = arith.cmpi ne, %divisor, %zero : i32
    %r = scf.if %ok -> (i32) {
      %q = arith.divui %x, %divisor : i32
      scf.yield %q : i32
    } else {
      scf.yield %zero : i32
    }
    return %r : i32
  }
}
```

`divui` 不写内存，但除数为零时行为未定义。原程序的零除数路径返回 0，不执行除法。若先算 `%q` 再用 `arith.select` 选 0，就在该路径引入了除零。这个反例同时说明 `select` 不能自动保护已经被提前执行的计算。

循环也有类似问题：一个不终止的操作被移到原本跳过它的路径，会改变程序终止行为。MLIR 用 `ConditionallySpeculatable` 等接口表达相关条件；`Pure` 组合了无内存效果与可推测执行等约定，不能仅凭看到“没有 store”自行推断。

## 6. 不同变换需要不同证明

| 想做的变换 | 至少要进一步判断 |
|---|---|
| 删除无结果操作 | 是否有必须保留的效果、控制流意义、终止行为 |
| 合并两个计算 | 运算语义/静态信息是否相同、支配是否成立、读到的状态是否相同 |
| 把计算提出循环 | operand 是否可用、是否循环不变、是否引入原本不会发生的执行 |
| 交换两个内存操作 | 是否别名、读写依赖与同步语义是否允许 |
| 把 tensor 更新改为原地存储写 | 后续旧值读取是否受影响、别名和所有权是否允许 |

这是推理清单，不是完整的优化合法性算法。实际变换使用接口、分析结果和操作语义共同判断；通用基础设施无法自动证明任何任意改写都正确。

## 7. Verifier 的保证边界

把加法的 constant 1 改成 constant 2，IR 依旧合法，但结果变了。把上述 guarded division 错误地提前计算，也可能通过结构和类型检查。反过来，一个理论上保持数学结果的变换若破坏支配或类型，仍不是合法 MLIR 改写。

所以需要分层证据：parser/verifier 验证 IR 契约；测试/语义论证检查改写前后行为；性能实验检查目标是否更快。后两者不能由第一层代替。含未定义行为的程序还要明确有效输入范围，不能依靠无效输入比较结果来论证正确性。

## 理解检查

- 两个不同 memref Value 是否一定不重叠？
- `%buffer` 是 SSA，为什么 store 仍能改变第二次 load 的结果？
- `tensor.insert` 为什么允许产生新 Value，而实现可能复用存储？
- 为什么“没有内存效果”不足以证明除法可以移出分支？

<details>
<summary>核对要点</summary>

memref 可以是别名/子视图；SSA 约束引用的定义而不是存储内容；复用需要保持值语义的分析证明；移动会扩大实际执行路径，可能引入原程序没有的未定义行为。

</details>

依据：[Side Effects & Speculation](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Rationale/SideEffectsAndSpeculation.md)、[SideEffectInterfaces.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/SideEffectInterfaces.td)、[Bufferization](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Bufferization.md)、[TensorOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td)、[MemRefOps.td](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)。效果/推测执行的讨论按此版本 SSACFG 模型展开。例子参与 P/G；执行结果是基于语义的推演。

继续阅读：[贯通阅读一段 IR](../tutorials/reading_ir)。
