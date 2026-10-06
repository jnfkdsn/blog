---
order: 2
title: One-Shot Bufferization 的决策
updated: 2026-10-04
---

# One-Shot Bufferization 的决策

上一章已经看到：同一个 `tensor.insert`，只读取新值时可以直接覆盖输入，随后读取旧值时则需要另一片存储。现在要解释的是，编译器怎样从 IR 得到这个结论。

One-Shot Bufferization 先分析张量 use-def 关系，为操作的 tensor operands 决定能否就地使用存储，再按决定改写 IR。这里“先分析再改写”的价值是：**在仍然能区分旧张量值和新张量值时，作出存储选择。** 如果已经把它们都随意映射成同一块内存，原有的值版本信息就很难恢复了。

## 1. Destination 给出复用候选

以这条更新为例：

```text
%u = tensor.insert %v into %t[%i] : tensor<4xi32>
```

`%t` 是 destination，`%u` 是围绕它构造的新结果。因此分析有一个自然的候选：让 `%u` 复用 `%t` 的 buffer。

这连接了 [DPS](../../dialects/linalg/destination_style) 的意义。Destination 并没有事先要求一定原地修改；它给出结果与哪个输入存储候选相关。对这个操作，分析主要在“复用 destination”和“另行分配”之间选择，而不是扫描全函数的每一块空闲内存，寻找全局最优安排。

所以，同样的数学计算，IR 是否提供合适的 destination，会影响这一步容易得到什么实现。后续 tiling/fusion 的结果是否形成清晰的数据流链，也会影响复用机会。

## 2. 读写冲突的具体推导

对 preserve 例子，先假设复用成立，再推演后果：

| IR 中的行为 | Tensor 语义需要什么 | 假设复用后的访存 |
|---|---|---|
| `%t` 来自输入边界 | 原内容第一项为 10 | A 中存着 10 |
| insert 得到 `%u` | 新内容第一项为 99 | 向 A[0] 写 99 |
| extract `%t[0]` | 读取旧值 10 | 从 A[0] 读出 99，冲突 |
| extract `%u[0]` | 读取新值 99 | 从 A[0] 读出 99，正确 |

最后一行虽然正确，也不能抵消第三行的错误。这是一次 read-after-write 冲突：候选写入先发生，随后需要旧内容的读取被破坏。

分析并不是只数“这个 Value 是否有两个 use”。第二个 use 可能只查询形状，也可能在写入之前已经读出标量；它未必与写入冲突。实际判断还依赖读写效果、别名关系、控制流位置和相应操作提供的事实。反过来，旧值也可能先经过 slice，再被读取，不能只检查 destination 的直接使用者。

对直线代码，一个简单的无冲突变体是先把旧值提取为标量，再进行更新：已经取出的标量不依赖旧 buffer 后续保持不变。这个变体有助于理解条件，但不代表分析器会在所有复杂循环和索引情况下证明所有理论上安全的复用。

## 3. 分析输出中的 Operand 决策

可以让工具只分析、不改写，并打印冲突：

```bash
mlir-opt preserve.mlir \
  '--one-shot-bufferize=test-analysis-only print-conflicts'
```

完整分析输出中的关键两处如下，SSA 名字保留工具打印形式，省略无关属性：

```text
%inserted = tensor.insert %c99_i32 into %0[%c0]
  {__inplace_operands_attr__ = ["none", "false", "none"]} : tensor<4xi32>

// 对照 reuse 例子，同一位置是：
{__inplace_operands_attr__ = ["none", "true", "none"]}
```

三个条目按 operands 排列：插入的标量、destination、索引。只有中间的 `%t` 是需要本次存储决策的 tensor operand。因此 `false` 表示 destination 不能原地使用，不是“这整个操作所有输入都必须复制”；`none` 也不是一次失败。

工具还为冲突关联定义、写入和读取，便于定位是哪一条旧值读取阻止了复用。这些带双下划线的属性是分析调试输出，不是操作的算法语义，不应写进正常输入来命令分析器强行原地执行。

## 4. 可写性与值依赖是两种条件

即使删去旧值读取，也可能无法复用。把输入边界的 `writable` 去掉：

<!-- memory-example: readonly -->
```text
module {
  func.func @readonly(%m: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %v = arith.constant 99 : i32
    %t = bufferization.to_tensor %m restrict : memref<4xi32> to tensor<4xi32>
    %u = tensor.insert %v into %t[%c0] : tensor<4xi32>
    %new = tensor.extract %u[%c0] : tensor<4xi32>
    return %new : i32
  }
}
```

这个函数仍然返回 99，但 One-Shot 不允许对输入存储生成写入。结果采用新分配，复制旧内容后再更新；分析输出会标记来源不可写。

这里的 readonly 是本函数边界对 bufferization 的约束。输入在运行时可能是普通可写内存，但编译器不能据此自行违反边界契约。与之相反，`writable` 只提供写入许可，不能覆盖旧值冲突检查：preserve 例子虽然可写，仍然不能直接修改旧值所需的 buffer。

因此，在排查额外复制时，至少先区分：**候选存储不允许写入，还是写入会破坏所需的值？** 两种原因需要调整的条件不同。

## 5. 新分配与复制的区别

再把单元素更新改成完整填充，同时保留旧值读取：

<!-- memory-example: fill-preserve -->
```text
module {
  func.func @fill_preserve(%m: memref<4xi32>) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %v = arith.constant 99 : i32
    %t = bufferization.to_tensor %m restrict writable : memref<4xi32> to tensor<4xi32>
    %u = linalg.fill ins(%v : i32) outs(%t : tensor<4xi32>) -> tensor<4xi32>
    %old = tensor.extract %t[%c0] : tensor<4xi32>
    %new = tensor.extract %u[%c0] : tensor<4xi32>
    return %old, %new : i32, i32
  }
}
```

旧值仍要保护，所以新结果不能覆盖原存储。但是 fill 为每个结果位置都写入 99，不需要读取 destination 的旧元素。实际输出为：

<!-- memory-output: fill-preserve-buffer -->
```text
module {
  func.func @fill_preserve(%arg0: memref<4xi32>) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c99_i32 = arith.constant 99 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
    linalg.fill ins(%c99_i32 : i32) outs(%alloc : memref<4xi32>)
    %0 = memref.load %arg0[%c0] : memref<4xi32>
    %1 = memref.load %alloc[%c0] : memref<4xi32>
    return %0, %1 : i32, i32
  }
}
```

这里有 alloc，没有 copy。可以分两步理解：

```text
旧值还要读取 → 需要独立的结果存储
结果完整覆盖 → 新存储不需要继承旧内容
```

而 tensor.insert 的未更新位置需要继承内容，所以相同的独立存储决定后面会跟随复制。这是为什么 bufferization 必须了解每个操作如何读取和写入，而不能只根据输入输出类型作决定。

## 6. Alias 与 Equivalent 的作用

分析中需要追踪“哪些张量未来可能共享存储”。其中共享一部分存储与对应同一个完整 buffer，是不同强度的事实。

对于 slice，结果可指向源 buffer 的一个子视图：两者 alias，但形状、offset 或覆盖范围不同，不能简单当成同一完整视图。对成功原地处理的 insert，destination 与结果则可具有等价的 buffer 关系。这里的 **Equivalent 指 buffer 关系，不是说更新前后张量数值相等**。

这种区别可以在局部填充中直接看到：

<!-- memory-example: tile-update -->
```text
module {
  func.func @tile_update(%m: memref<4x4xi32>) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %v = arith.constant 99 : i32
    %t = bufferization.to_tensor %m restrict writable : memref<4x4xi32> to tensor<4x4xi32>
    %tile = tensor.extract_slice %t[1, 1][2, 2][1, 1]
      : tensor<4x4xi32> to tensor<2x2xi32>
    %filled = linalg.fill ins(%v : i32) outs(%tile : tensor<2x2xi32>) -> tensor<2x2xi32>
    %u = tensor.insert_slice %filled into %t[1, 1][2, 2][1, 1]
      : tensor<2x2xi32> into tensor<4x4xi32>
    %inside = tensor.extract %u[%c1, %c2] : tensor<4x4xi32>
    %outside = tensor.extract %u[%c0, %c0] : tensor<4x4xi32>
    return %inside, %outside : i32, i32
  }
}
```

数据流是“取窗口 → 填窗口 → 将结果放回同一位置”。没有额外旧值冲突，destination 又允许写入时，窗口可以由 subview 实现。执行 One-Shot 后，再用 CSE 合并相同视图并 canonicalize 清理自复制，得到：

<!-- memory-output: tile-update-buffer -->
```text
module {
  func.func @tile_update(%arg0: memref<4x4xi32>) -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c99_i32 = arith.constant 99 : i32
    %subview = memref.subview %arg0[1, 1] [2, 2] [1, 1] : memref<4x4xi32> to memref<2x2xi32, strided<[4, 1], offset: 5>>
    linalg.fill ins(%c99_i32 : i32) outs(%subview : memref<2x2xi32, strided<[4, 1], offset: 5>>)
    %0 = memref.load %arg0[%c1, %c2] : memref<4x4xi32>
    %1 = memref.load %arg0[%c0, %c0] : memref<4x4xi32>
    return %0, %1 : i32, i32
  }
}
```

最后只剩一个 subview、在窗口上的 fill，以及两次读取。原数组 `[1,2]` 变成 99，窗口外的 `[0,0]` 保持原内容。这里没有把整个 `4×4` 数组复制一遍。

这个最终结果是几个 passes 合作形成的：One-Shot 选择共享存储并改写操作，CSE/canonicalize 清理冗余。固定版本的原始 One-Shot 输出仍可能包含等价 subview 之间的 copy，不能将所有清理效果都归到分析器自身。

## 7. 操作接口与改写阶段

分析器不能从名字猜出 `my.some_op` 是否会读取 destination、是否完整覆盖结果、返回值是否与输入关联。标准操作通过 `BufferizableOpInterface` 提供这些事实及具体改写方法。

对于当前例子，消费关系可以理解成：

```text
操作接口：说明读写、结果与输入的存储关系
       ↓
One-Shot 分析：沿 use-def 与控制流检查冲突，决定原地与否
       ↓
按决定准备 operand 的 buffer，必要时分配和复制
       ↓
操作的 bufferize 实现：产生 store、load、subview 或 buffer 形式计算
```

因此，`tensor.insert` 的最终改写函数可以很短：取得已经准备好的 destination buffer，创建 store，再把原结果替换为这个 buffer。它无需在这个局部函数里重新分析整段程序。反之，仅实现一个“总是 store 输入”的 Pattern，缺少前面的契约与分析，就不能保证 Tensor 语义。

许多标准操作的实现使用外部 Interface 模型，位于方言的 Transforms 库中。自己构建工具时，除了注册方言，还要注册需要的模型；否则“工具能解析这个操作”并不意味着 One-Shot 能处理它。遇到不支持的 tensor 操作，默认转换会失败；`allow-unknown-ops` 可以保留混合边界，但剩余张量仍需安排后续处理，不能直接视为完整 bufferization 已完成。

本章建立的是消费者视角。实现自定义接口时再沿一个具体操作补齐这些方法，不需要现在背诵整套接口重载。

## 8. 决策能力的边界

One-Shot 按启发式顺序分析，增量建立 alias/equivalence 关系，并作出局部的原地决定。它追求减少分配和复制，但不承诺全局最优的内存计划。操作接口提供得不够精确、数据流有复杂分支，或者布局与函数边界约束变化，都可能带来保守结果或额外搬运。

本章的调试顺序可以迁移到更大程序：先找到那次额外分配或复制，再确认来源是否可写、哪个 operand 被判为非原地、哪条读取构成冲突，最后检查能否通过合法的数据流或 destination 调整改善。不要先删掉 copy 再猜它是否多余。

即便所有复用决定都正确，临时分配仍要释放。下一章[函数边界、所有权与释放](./ownership)继续处理这一部分。

## 阅读自查

1. fill 和 insert 都不能复用旧 buffer 时，为什么只有后者需要复制初始内容？
2. 一个 tensor 有两个 use 是否足以证明存在冲突？给出形状查询或提前取值的例子。
3. 如果在 tile 更新之后还读取原张量窗口中的旧元素，刚才的 subview 原地实现为什么需要重新评估？
4. `Equivalent` 为什么可以用于数值不同的旧张量与新张量？

## 资料与实践

输入、分析标记与实际输出按 LLVM 20.1.8 验证。实现依据见[One-Shot 分析](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp)、[Tensor 的外部接口模型](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp)以及[接口契约](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td)。

可选实践：[冲突、不可写来源、完整覆盖与窗口更新](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/10-memory-bufferization)。自定义接口工程仍作为本模块按需扩展。
