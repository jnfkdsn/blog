---
order: 40
title: 自定义操作接入 Bufferization
updated: 2026-10-04
---

# 自定义操作接入 Bufferization

前面用标准操作解释了 One-Shot Bufferization 怎样决定复用或分配。现在换成自己定义的计算：输入一维 f32 Tensor，每项加一。目标是让同一个通用 Pass 处理它，并继续保留 Tensor 的新旧值语义。

这项工作可以分成两段：操作先向分析器说明将来怎样访问存储；分析器作出决定后，操作再把自己改写成真正的 load、add 和 store。`BufferizableOpInterface` 把这两段连接起来。

## 1. 新值与候选存储

用 `my.add_one` 表达计算：

```mlir
%r = my.add_one %input outs(%init) : tensor<4xf32>
```

本章定义的语义是 `r[i] = input[i] + 1`。input 和 init 必须具有相同的一维形状，结果形状与它们相同。init 的元素不参与计算，但它提供结果存储的候选。对于动态类型，两个 `tensor<?xf32>` 类型相同并不证明运行时长度相等；相等长度是本操作的调用前提。

这里的 `outs` 是教学操作的文本语法。工程手写 Bufferizable 接口的对应关系，没有仅凭这个词就自动获得 `DestinationStyleOpInterface`。若还希望参加依赖 DPS 接口的其他变换，需要另外实现那个协议。

取 input=[10,20,30,40]，结果应为 [11,21,31,41]。若 init 与 input 是同一个 Tensor 值，仍须先考虑谁还需要旧的 [10,20,30,40]。

## 2. 同一操作的两种上下文

下面是工程中保留旧值的完整函数。`restrict` 提供相应别名承诺，`writable` 表示可写；它们不能被当成绕过真实别名关系的优化开关。

```mlir
func.func @preserve(%buffer: memref<4xf32>) -> (f32, f32) {
  %t = bufferization.to_tensor %buffer restrict writable
       : memref<4xf32> to tensor<4xf32>
  %r = my.add_one %t outs(%t) : tensor<4xf32>
  %c0 = arith.constant 0 : index
  %old = tensor.extract %t[%c0] : tensor<4xf32>
  %new = tensor.extract %r[%c0] : tensor<4xf32>
  return %old, %new : f32, f32
}
```

返回值必须是 (10,11)。如果直接把第一项写成 11，后面的旧值读取也会变成 11，程序就错了。正确实现为新结果准备另一块存储。

若删除 old 的读取和对应返回，只保留 new，原来的数据完成加一后不再需要，便可以复用。两种情况里操作定义完全相同，差别来自 use-def 关系。接口提供局部事实，分析器结合程序上下文作决定。

## 3. 读写声明描述哪一层行为

分析器遇到 add_one，需要知道两个 operand 在 **bufferized 实现** 中扮演什么角色：

| Operand | 读取其内容 | 写入对应目标 | 与结果的关系 |
|---|---|---|---|
| input | 是，计算 input[i]+1 | 否 | 不承诺结果一定沿用它的 buffer |
| init | 否，不用初始元素 | 是，作为结果写入目标 | 结果与最终选择的目标 equivalent |

input 和 init 在例子中恰好引用同一 Value，不会让这两行变成同一项声明。operand 是操作中的使用位置，分析器需要区分“作为输入读取”和“作为目标写入”。

这也不与 Tensor 操作的 `Pure` 冲突：原始操作产生新的 Tensor 值，没有承诺原地修改某个可观察的 MemRef。Bufferizable 接口描述的是将来的内存实现。只有分析证明旧值语义得以保留，才允许选择原地实现。

工程的核心回答为：

```cpp
bool bufferizesToMemoryRead(Operation *, OpOperand &operand,
                            const AnalysisState &) const {
  return operand.getOperandNumber() == 0;
}
bool bufferizesToMemoryWrite(Operation *, OpOperand &operand,
                             const AnalysisState &) const {
  return operand.getOperandNumber() == 1;
}
AliasingValueList getAliasingValues(Operation *op, OpOperand &operand,
                                   const AnalysisState &) const {
  if (operand.getOperandNumber() == 1)
    return {{op->getResult(0), BufferRelation::Equivalent}};
  return {};
}
```

Equivalent 表示完整对应的 buffer 关系，比“可能共享某部分存储”更强。它不是授权分析器覆盖仍被需要的旧数据：若目标必须 out-of-place，框架先为目标选择新存储，再让结果对应这份新存储。

## 4. 同一位置先读后写

即使旧 Tensor 在操作之后没有其他 use，也要确认操作自己的读写不会冲突。加一的循环在每个位置先读取，再写入同一位置：

```text
读取 x[0] → 写入 y[0]
读取 x[1] → 写入 y[1]
……
```

当 x 和 y 对应等价 buffer 时，这个顺序不破坏后续需要的输入。因此工程实现 `bufferizesToElementwiseAccess`，允许分析利用这一事实。

对照一个邻域计算 `y[i]=x[i-1]+x[i]`：写完 y[0] 后，下一次可能还需要原来的 x[0]。它不能沿用本章的声明。这个接口要求相关 operands 同形状，具体实现满足逐元素读写顺序、没有相应循环携带依赖；仅仅数学上“输出也是一个 Tensor”远远不够。

固定版本实现还区分 equivalent 与仅 aliasing 的输入。两个错位窗口可能共享分配，却不具有相同的元素对应关系。保守返回 false 会错过复用机会，错误返回 true 则可能破坏正确性。

## 5. 从分析决定到实际循环

完成分析后，改写方法使用 `getBuffer` 取得框架为两个 operand 安排的存储。它不自己重新猜测 `%init` 能否原地更新：

```cpp
FailureOr<Value> input = getBuffer(rewriter, add.getInput(), options);
FailureOr<Value> output = getBuffer(rewriter, add.getInit(), options);
if (failed(input) || failed(output))
  return failure();
```

接下来构造等价于以下 IR 的循环，最后用 `replaceOpWithBufferizedValues` 将 Tensor 结果接到 output 对应的 buffer。代码为变量名整理后的结构摘录：

```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %old = memref.load %input[%i] : memref<4xf32>
  %next = arith.addf %old, %one : f32
  memref.store %next, %output[%i] : memref<4xf32>
}
```

复用案例中 input 和 output 指向同一 buffer；保留旧值案例中，output 来自新 `memref.alloc`。两者使用同一个改写方法。新增分配也没有前置 `memref.copy`，因为该操作从不读取 init 的原始内容，而且会覆盖整个结果。

“需要新分配”和“需要复制初值”因此是两个决定。若将操作改成只更新一部分元素，其余位置沿用 init，本章的读声明和实现都要修改。

## 6. 外部模型与工具接入

工程采用外部模型，把 Bufferization 实现放在变换侧。工具注册 My dialect 后，再注册这个扩展；加载该 dialect 时把模型附到 AddOneOp。

```cpp
registry.addExtension(+[](MLIRContext *ctx, my::MyDialect *) {
  ctx->getOrLoadDialect<arith::ArithDialect>();
  ctx->getOrLoadDialect<memref::MemRefDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  my::AddOneOp::attachInterface<AddOneModel>(*ctx);
});
```

这里也加载改写会创建的目标方言。能够解析 my.add_one 与能够运行它的 Bufferization 是不同的能力；注册了操作，不代表所有变换接口都自动存在。

没有注册模型的对照工具仍然能读取这份 IR。正常 One-Shot 会报告操作未被 bufferize；允许未知操作后，转换可以保留 Tensor 边界，my.add_one 本身仍在。这个选项为分阶段接入提供边界，不会凭空生成加一的内存实现。

## 7. 释放与数值证据

One-Shot 输出新分配后，还需要所有权/释放阶段。工程在 `buffer-deallocation-pipeline` 后观察到 preserve 的一次释放，reuse 没有新分配或释放。

再沿已学的 LLVM lowering 和 C wrapper 调用，可观察：

```text
输入首项 10：reuse 返回 11，原 buffer 变成 [11,21,31,41]
输入首项 10：preserve 返回 old=10、new=11，原 buffer 保持不变
```

C 检查还覆盖负值、零与分数，并检查四个输入元素的最终内容。这里的数值证据针对固定形状主例；动态类型的形状接口在[类型与形状推导](../ir_definition/type_shape_inference)继续验证，不能据此声称所有动态别名和控制流均已覆盖。

现在可以自己尝试一种有限修改：把“加一”改为“乘二”，先预测接口事实是否需要变化，再修改计算和数值预期。若改为前缀和，则应重新分析逐元素声明，不能只替换循环中的算术。

## 依据与实践

接口契约见 LLVM 20.1.8 的 [BufferizableOpInterface 定义](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td)及 [One-Shot 分析](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp)。[官方 Bufferization 文档](https://mlir.llvm.org/docs/Bufferization/)提供总体分类，当前网页可能使用更新 API。

[19-extension-interfaces 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/19-extension-interfaces)保留模型、正反工具、IR 和 C 驱动。正文所需的复用判断、代码与结果均已给出；实践用于核对自己的预测，不要求先阅读整套框架模板。
