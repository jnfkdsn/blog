---
order: 120
title: Store 与 InsertSlice：冗余搬运的源头预防
updated: 2026-10-16
tags: [ai-compiler, npu, mlir, tensor, bufferization]
---

# Store 与 InsertSlice：冗余搬运的源头预防

一个算子分块计算输出时，每次可能只算好完整结果中的一小块。计算结果需要写入目标位置，IR 也需要表示“这一块已经更新后的完整张量”。在 AscendNPU-IR 的一类 Tensor IR 中，这两件事分别由 `hivm.hir.store` 和 `tensor.insert_slice` 表达。

问题出现在编译器安排搬运时：数据本来已经准备写到正确位置，却因为后面还有一条 `insert_slice`，又被安排读回片上，再写回原处。这篇文章沿同一块数据解释冗余如何产生，以及怎样在插入 Load 之前识别它。

本文整理的是[该参赛团队的方案](https://gitlab.eduxiji.net/educg-group-49265-3230617/T2026101412010750-1832)，对应其设计报告 **3.4「基于张量语义的冗余搬运源头预防」**。方案与实现归原团队所有；这里补充面向 MLIR 学习的推演，源码版本和核验范围列在文末。

## 1. 从局部结果到完整张量

把输出想成 16 张大小为 4×8 的小图，类型为 `tensor<16x4x8xf32>`。某一轮计算得到第 `%offset` 张小图的新内容，类型为 `tensor<4x8xf32>`。为了继续传递或返回完整结果，需要把这块新内容与其他 15 张小图组成一个新的张量值。

下面是团队回归测试中的实际输入，省略测试注释，保留完整函数：

```mlir
func.func @store_backed_identity_insert_static(
    %computed: tensor<4x8xf32>, %output: tensor<16x4x8xf32>,
    %offset: index) -> tensor<16x4x8xf32> {
  %slice = tensor.extract_slice %output[%offset, 0, 0] [1, 4, 8] [1, 1, 1]
      : tensor<16x4x8xf32> to tensor<4x8xf32>
  %stored = hivm.hir.store ins(%computed : tensor<4x8xf32>)
      outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
  %updated = tensor.insert_slice %stored into %output[%offset, 0, 0]
      [1, 4, 8] [1, 1, 1]
      : tensor<4x8xf32> into tensor<16x4x8xf32>
  return %updated : tensor<16x4x8xf32>
}
```

先只追踪四个值：

| 值 | 表示什么 | 形状 |
|---|---|---|
| `%output` | 本轮更新前的完整输出 | 16×4×8 |
| `%slice` | 从完整输出中选出的目标小图 | 4×8 |
| `%stored` | Store 产生的更新后局部结果 | 4×8 |
| `%updated` | 把局部结果放回后得到的完整输出 | 16×4×8 |

`extract_slice` 的 sizes 是 `[1,4,8]`，表示从第一维取一个位置，后两维全部取出。结果类型省略了长度为 1 的第一维，这就是降秩切片。`insert_slice` 使用相同的坐标和大小，把二维局部结果放回三维张量。

假设 `%offset = 3`，更新前 `%output[3,0,0] = 2`，而 `%computed[0,0] = 99`。最终 `%updated[3,0,0] = 99`，第 3 张小图中的其他位置也来自 `%computed`，其余小图的值来自旧 `%output`。用元素关系表示就是：

```text
updated[k,i,j] = computed[i,j]    当 k = offset
updated[k,i,j] = output[k,i,j]    当 k ≠ offset
```

这里能够直接回答一个常见疑问：**`insert_slice` 确实让局部结果重新组成了函数所需的完整形状，但它还决定更新哪些元素、保留哪些元素。** 只说“为了让类型相同”会漏掉这部分语义。把同一块数据插到第 4 张小图，返回类型完全一样，计算结果却变了。

这也不是所有 Store 后面都必须出现的结构。如果函数只需要返回局部结果，就没有这个完整张量的组装需求；如果已经处于 MemRef IR，写入目标内存也不需要靠 Tensor SSA 的结果来表达更新。[Tensor 官方定义](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert_slice-tensorinsertsliceop)描述的是子集更新，而不是一种通用的类型转换。

## 2. 张量值更新与实际内存写入

上面这段 IR 容易引起另一个疑问：既然 Store 已经写了，为什么还要 InsertSlice？

需要把两个层次分开。**在 Tensor SSA 层，`%output` 和 `%updated` 是更新前后的两个值。** 后面的操作如果需要新内容，就应使用 `%updated`。不能因为看到 Store，就把所有使用旧 `%output` 的地方理解成自动读到新内容。

到了存储实现层，编译器才要决定这些值用哪些 buffer 表示。若旧值不再需要保留，并且其他条件允许，更新前后可以复用同一块存储。Bufferization 的一项任务就是把这种张量值关系落实为存储、别名与复制关系。

在本案例允许原地实现的情况下，目标关系可以画成：

```text
完整输出的 buffer：GM_output
目标切片：          GM_output[offset, :, :]
局部计算结果：      UB_computed

UB_computed ── Store ──> GM_output[offset, :, :]
```

GM 是片外全局存储，UB 是用于这类向量计算的片上 Unified Buffer。上图是该 NPU 场景下的存储实现示意；`tensor<4x8xf32>` 这种类型本身没有写出 GM 或 UB，不能只看类型就确定存储位置。

如果 Store 的目标已经是完整输出的这个切片，那么完成第一次写入之后，完整输出所需的内容就已经到位。`insert_slice` 仍然有表达 `%updated` 的作用，但后续 lowering 有机会消去它对应的额外复制。

因此，团队所称的“恒等 InsertSlice”，是指**在这些条件下，它不再要求另一次物理数据搬运**。它并不表示 `%updated` 和旧 `%output` 的元素完全相等。上面的 2 已经变成了 99。

## 3. Store 与切片操作的生成来源

前面展示的是已经形成的 IR。现在回到它形成之前：程序原本可能只表达一个完整张量计算，编译器为什么会把它变成 Extract → 局部计算 → Store → Insert？

需要区分两个顺序。最终 IR 的数据流顺序，是先得到局部结果，再写回并组装完整输出；编译器的改写顺序却可能是先为完整计算建立 Store，再把计算和 Store 一起分块。操作出现在计算之后，不意味着编译器必须在生成分块计算之后才创建它。

下面沿“完整输出写回 → 分块 → 方言转换 → 搬运插入”解释一条来源路径。前述回归测试直接给出了 Store/InsertSlice，不能据此还原某个真实算子的唯一生成历史；这里结合固定基线的生成入口和 LLVM 分块实现解释机制，示意 IR 不作为实际 pipeline dump。

### 3.1 输出写回与 CacheIO

继续用 16 张 4×8 小图作为完整输出。假设上层计算先写成下面的伪 IR，`compute` 表示生成这组输出的逐元素计算：

```text
result = compute(inputs, outs=output)
return result
```

在基线的自动调度代码中，可以找到这条调用链：

```text
SchedulerBase::cacheIO
    → cacheFuncIO
        → cacheWriteFuncReturn
            → createCacheWrite
```

`cacheWriteFuncReturn` 遍历函数的返回操作数，跳过已标记为 cached 的结果，追踪相关 reshape 等关系，再尝试建立 cache write。`createCacheWrite` 只处理其支持的计算结果，并不是看到任意 `return` 都会成功创建 Store。

对于其中“将 Linalg 结果写回原输出目标”的分支，它把计算的目标改成新的临时张量，并把原目标交给 Store：

```text
temporary = tensor.empty(完整输出的形状)
result = compute(inputs, outs=temporary)
stored = hfusion.store(result, outs=output)
return stored
```

此时计算结果和输出写回已经成为两个操作，但它们仍可以具有完整的 16×4×8 形状。这里的 temporary 是 Tensor IR 中的目标值，尚不能仅凭 `tensor.empty` 就认定已经分配了一整块 UB。

在 [createCacheWrite 的实现](https://github.com/Ascend/AscendNPU-IR/blob/428ab8fdab46/bishengir/lib/Dialect/HFusion/Utils/Utils.cpp)中，真正创建 Store 的代码是：

```cpp
cachedOp = rewriter.create<hfusion::StoreOp>(loc, ValueRange{result},
                                           ValueRange{initOperand});
```

`result` 是计算产生的值，`initOperand` 是原来的输出目标。之后函数还会调整相应的使用关系，使需要写回结果的地方引用 Store 的结果。入口可从 [CacheIO.cpp](https://github.com/Ascend/AscendNPU-IR/blob/428ab8fdab46/bishengir/lib/Dialect/HFusion/Transforms/CacheIO.cpp) 的 `cacheWriteFuncReturn` 追到这里。

### 3.2 分块操作数与结果组装

调度选择一次处理一张 4×8 小图之后，局部计算需要输入区域，局部 Store 需要输出目标区域，循环又需要传递更新后的完整输出。分块与融合后的数据流可以示意为：

```text
current = output
for k in 0..15:
    input_tile  = extract_slice(inputs, 本轮输入区域)
    output_tile = extract_slice(current, [k,0,0], [1,4,8], [1,1,1])
    computed = compute(input_tile)
    stored = hfusion.store(computed, outs=output_tile)
    next = insert_slice(stored, into=current,
                        [k,0,0], [1,4,8], [1,1,1])
    current = next
return current
```

这是描述张量值传递的伪代码。实际 `scf.for` 会用 `iter_args` 和 `scf.yield` 表达 current 到 next 的更新，不是给同一个 SSA 值重新赋值。示意还省略了输入的具体访问映射和可能的中间切片简化；是否保留长度为 1 的维度由具体变换决定。

`input_tile` 和 `output_tile` 的作用不同。前者提供本轮计算需要的输入元素，后者指定本轮输出对应的目标子集。尤其是 **对输出做 ExtractSlice 不等于把旧输出从 GM 读进 UB**。是否需要读取旧内容，还取决于计算是否使用它以及后续存储实现。

在固定基线的 [HFusion 分块接口](https://github.com/Ascend/AscendNPU-IR/blob/428ab8fdab46/bishengir/lib/Dialect/HFusion/Transforms/TilingInterfaceImpl.cpp)中，`getTiledImplementation` 调用 `makeTiledShapes` 准备分块操作数，再用局部类型和操作数克隆操作。调用它的分块框架还需要把这些局部结果接回完整输出。

这一组装动作可以从 LLVM `llvmorg-20.1.8` 的 [generateLoopNestUsingForOp](https://github.com/llvm/llvm-project/blob/87f0227cb60147a26a1eeb4fb06e3b505e9c7261/mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp#L396)直接看到：

```cpp
auto insertSlice = rewriter.create<tensor::InsertSliceOp>(
    loc, tiledValue, destinationTensor, resultOffset, resultSize,
    resultStride);
yieldedValues.push_back(insertSlice);
```

这里 `tiledValue` 是局部操作的结果，`destinationTensor` 是循环当前持有的完整目标。框架创建 InsertSlice 后，再创建 `scf.yield` 传递更新后的完整张量。这个 LLVM 版本用于展示已核对的通用机制，不声称它就是本案例基线所固定的 LLVM 子模块版本。

因此，ExtractSlice/InsertSlice 来自需要分块和组装结果的变换逻辑；它们不是 parser 看到 Store 就自动添加的，也不是 verifier 为了补齐类型而生成的。

### 3.3 从 HFusion Store 到 HIVM Store

上面的写回操作是 `hfusion.store`。后续 [HFusionStoreOpToHIVMStoreOp](https://github.com/Ascend/AscendNPU-IR/blob/428ab8fdab46/bishengir/lib/Conversion/HFusionToHIVM/HFusionToHIVM.cpp) 将它替换为 HIVM Store，核心创建语句为：

```cpp
auto newStoreOp =
    rewriter.create<hivm::StoreOp>(op.getLoc(), TypeRange(res), src, dst);
```

转换保留输入、目标和结果类型的对应关系，并处理原子属性。到这一层，才出现前文实际测试里的 `hivm.hir.store` 名称。

另一类 Store 也可能由 HIVM 层的搬运插入规则直接创建，因此不能仅凭操作名称确定其来源。具体算子的来源应对照相邻 Pass 的 IR：在哪一步首次出现 Store，在哪一步变成局部形状，在哪一步出现 InsertSlice。

### 3.4 生成职责与优化位置

沿这条路径，各部分分工如下：

| 处理环节 | 创建或改变什么 | 所需信息 |
|---|---|---|
| CacheIO 输出写回 | 建立 `hfusion.store`，调整计算目标与使用关系 | 返回值、计算结果及原输出目标 |
| 分块与结果组装 | 准备切片、局部操作和完整张量更新 | tile 的范围、操作的访问关系与结果子集 |
| HFusion → HIVM | 将 HFusion Store 转为 HIVM Store | 输入、目标、类型与操作属性 |
| MixCV 搬运插入 | 为相应消费边补充 Load/Store | 生产者、消费者与计算/存储位置要求 |

这次团队优化作用于最后一行：结构已经存在时，判断 InsertSlice 的某次使用是否真的需要额外 Load。它不负责决定 tile 大小，也不负责最初创建 Store 或 InsertSlice。

对于当前逐元素例子，可以按“每轮算好一块、写回一块”理解。对于沿归约维度分块的 Matmul 等计算，可能先累计多轮再写回；不能把这里的循环示意推广成每个局部计算之后都必须 Store。

## 4. 基线如何引入冗余 Load

AscendNPU-IR 需要为不同计算与存储位置之间的数据流安排搬运。例如某个结果已经写到 GM，后面的向量计算又需要它，通常就需要先 Load 到 UB。

基线的 `InsertLoadStoreForMixCV` Pass 中有一类通用规则：检查消费者的操作数，若它能追溯到 Store 或 Fixpipe，就为相应消费位置安排 Load。Fixpipe 是这里另一类目标相关的数据输出操作；本案例的优化证明只处理普通 Store。

问题是，`tensor.insert_slice` 也进入了这套消费者规则。基线看到的是：

```text
Store 产生 %stored
    ↓
InsertSlice 使用 %stored
    ↓
为这次使用补一个 Load
```

对真正需要把数据读到 UB 才能计算的消费者，这个决定有意义。但对于上一节的同位置组装，数据已经在最终目标处，补 Load 就把一个逻辑上的张量更新变成了新的搬运需求。

用主例表示，额外插入的关键部分如下。这是省略其他 IR 的结构示意：

```mlir
%stored = hivm.hir.store ins(%computed : tensor<4x8xf32>)
    outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>

%tmp = tensor.empty() : tensor<4x8xf32>
%reloaded = hivm.hir.load ins(%stored : tensor<4x8xf32>)
    outs(%tmp : tensor<4x8xf32>) -> tensor<4x8xf32>

%updated = tensor.insert_slice %reloaded into %output[%offset, 0, 0]
    [1, 4, 8] [1, 1, 1]
    : tensor<4x8xf32> into tensor<16x4x8xf32>
```

此时 InsertSlice 的 source 已经改成了 `%reloaded`。它对应的内容被读进临时 UB，后续把这个临时结果插回 GM 输出，又可能物化为一次写回。报告分析的冗余路径是：

```text
UB_computed ──首次 Store──> GM_output 的目标切片
                           │
                           └──额外 Load──> UB_tmp
                                            │
GM_output 的同一目标切片 <──重复写回──────────┘
```

第一次 Store 是有用的。后面的读回与写回没有产生新的数值，只让同一块内容绕了一圈，还引入临时 UB 和相关调度开销。

这条链来自该项目的搬运插入与后续 lowering 方式，不是 MLIR 规定每个 `return` 都要加 Store、每个 Store 前后都要加 Load。分析时应定位具体 Pass 的决定，而不是从操作名字推出普遍规则。

## 5. 在搬运生成前证明它不必要

团队把优化放在原来插入 Load 的位置。处理 InsertSlice 时，先判断它是否只是在表达那个 Store 已经覆盖的同一个目标子集。若能证明，就让它直接使用 `%stored`，不创建 `%tmp` 和 `%reloaded`；证明失败则继续采用基线处理。

最重要的关系只有三条：

```text
E = 从 output 中提取目标切片
S = 将 computed 存入 E，产生 stored
I = 将 stored 放回同一个 output 的同一切片
```

但“看起来是这三条”还不够。接下来逐步补足证明。

### 5.1 相同的目标与坐标

首先，Store 的目标必须直接来自 ExtractSlice，InsertSlice 的 destination 必须正是 ExtractSlice 的源张量，InsertSlice 的 source 必须直接是该 Store 的唯一结果。

其次，Extract 和 Insert 的 offsets、sizes、strides 以及降秩方式必须一致。如果从第 3 张小图取目标，却插入第 4 张，那么第一次 Store 没有完成最终要求的更新，不能跳过相应处理。

实现对动态参数采用结构化比较：动态 offset、size 等必须是同一个 SSA 值。即使两个独立计算的值在数学上相等，这个局部判定也不额外做代数证明。它宁可少覆盖一些机会，也不把无法确认相同的位置当成相同。

类型检查也很明确：Store 输入、Store 目标、Extract 结果与 Insert source 的切片类型必须完全一致；Extract 源与 Insert destination 的完整张量类型也要一致。这里检查的是两组对应关系，**不要求 4×8 的切片类型等于 16×4×8 的完整输出类型**。

### 5.2 旧张量是否还需要保留

即使位置相同，旧值的使用也会影响存储复用。例如在主例中增加一次对旧 `%output` 的读取，并要求得到更新前的 2，就不能随意把承载旧值的存储覆盖成 99。

Bufferization 可能因此为 Store 的结果选择另一块存储。这样一来，InsertSlice 就确实需要把那块存储里的内容送回完整输出，不能再把它当成只有 SSA 组装意义的操作。

团队用较强的使用关系约束限制这种情况：

- `%slice` 只有 Store 这一个使用者。
- 旧 `%output` 的使用者只允许是当前配对的 ExtractSlice 和 InsertSlice。

这是针对简单数据流的保守判定，不是任意别名图上的通用分析。即使额外读取发生在不影响正确性的位置，这条规则也可能放弃优化。

### 5.3 普通写入与局部顺序

实现还要求 Store 使用纯 Tensor 形式、不是 atomic Store，也没有末轴隐式转置。这里的“纯 Tensor”说明参与的是 Tensor 操作数和结果，不能理解成 Store 没有内存效果。

普通覆写、原子累加、隐式转置具有不同语义，不能共用“原样写回同一块数据”的推理。Extract、Store、Insert 还必须在同一 Block 中按这个顺序出现，Store 与 Insert 之间的操作必须通过 `isMemoryEffectFree` 检查。

最后一项也很保守：它要求中间操作没有内存效果，而不是进一步证明“虽然读写了内存，但恰好与本切片无关”。这些条件共同把优化限定在容易核验的局部范围。

## 6. 专用 Pattern 只改变一条消费边

上述判断集中在 `isStoreBackedIdentityInsertSlice`。理解了证明后，再看 Pattern 的核心就很直接了。下面摘录专用 Pattern 的判定与操作数收集部分：

```cpp
auto sourceStore = insertOp.getSource().getDefiningOp<hivm::StoreOp>();
bool skipSourceLoad =
    sourceStore && isStoreBackedIdentityInsertSlice(insertOp, sourceStore);

llvm::SmallVector<OpOperand *> consumerOperands;
for (OpOperand &operand : insertOp->getOpOperands()) {
  if (skipSourceLoad && operand.getOperandNumber() == 0) {
    continue;
  }
  if (traceDefOp<hivm::FixpipeOp>(operand.get(), false, true).has_value() ||
      traceDefOp<hivm::StoreOp>(operand.get(), false, true).has_value()) {
    consumerOperands.push_back(&operand);
  }
}
```

`consumerOperands` 收集需要补 Load 的使用位置。编号为 0 的操作数是 InsertSlice 的 source；证明成功时只把这一项排除，其他操作数仍按原来的 Store/Fixpipe 追踪规则处理。收集完成后，原有 `insertLoadStoreOp` 辅助函数继续负责插入必要的 Load。

所以优化完成后的主例仍是第 1 节的 Extract → Store → Insert。这个 Pattern 不删除第一次 Store，不把 InsertSlice 替换为旧 `%output`，也不靠给整个操作添加一个新豁免标记来绕过所有检查。后续 Bufferization 再处理存储复用与可能的复制消除。

实现还替换了 Pattern 的注册方式：原来一次性为 InsertSlice 注册六类通用规则，现在保留其中五类，只把 Store-like 消费者规则换成这个专用版本。否则旧规则仍有机会把刚刚避免的 Load 加回来。已有 `elide_after_bufferize` 标记的处理也保持原行为。

### 多消费者的处理

假设 `%stored` 除了用于组装完整输出，还要参与一次向量加法：

```text
                         ┌──> InsertSlice：同位置组装
Store 的结果 %stored ────┤
                         └──> VAdd：继续计算
```

优化后，第一条分支可以直接使用 `%stored`；第二条分支仍可能需要先 Load 到 UB，才能提供向量计算的输入：

```text
                         ┌──────────> InsertSlice
Store 的结果 %stored ────┤
                         └──> Load ──> VAdd
```

这就是只修改一条“消费边”的意义。是否需要搬运取决于这次使用做什么，不能只根据生产者是某个 Store，就删除它的所有下游 Load。

## 7. 回归测试体现的边界

团队的 `insert-load-store-for-mix-cv.mlir` 中可以直接找到下面五个相关测试。表内是测试规定的预期行为，本篇未在本机执行该分支的测试。

| 测试情形 | 预期处理 | 对应原因 |
|---|---|---|
| 静态形状的降秩切片，Extract/Insert 对应 | Store 后直接是 InsertSlice | 同目标、同子集的组装 |
| 动态切片使用相同 SSA offset/size | 不为 source 新增 Load | 动态性本身不阻止结构化证明 |
| Extract/Insert 使用不同 SSA offset | 保留 Load | 无法证明同一子集 |
| atomic add Store | 保留 Load | 不属于普通覆写 |
| Store 同时供 InsertSlice 和 VAdd 使用 | InsertSlice 直接用 Store 结果，VAdd 使用 Load 结果 | 豁免只作用于已证明的那次使用 |

前面的数据流解释可以进一步形成两层验证：第一层检查这个 Pass 是否只避免了指定 Load；第二层检查后续 Bufferization/lowering 是否也避免了对应临时存储与重复写回。只有第一层的 IR 检查，不能直接等同于最终设备访存已经减少。

报告还讨论了旧值读取、重叠切片、Bufferization 后复制等验证场景。本篇独立核对的是上表五项测试定义以及对应源代码条件，不把报告中更广的测试陈述当成本机实测结果，也不据此给出性能加速数字。

## 8. 从这个案例理解优化时机

在较低层 IR 中看到多余的 Load，再分析它能否删除，是一条可行路线。但到了那时，往往需要从地址、视图和别名关系重新还原“这是不是同一个切片”，还要维护已有的搬运与同步关系。

这项方案利用了较高层 IR 仍然保留的信息：同一个完整张量、明确的 Extract/Insert 子集参数、可直接检查的 SSA 使用关系。它在决定是否生成搬运时就排除了已证明不必要的一次使用。

这也说明 Tensor 阶段并非只是在等待分配内存。它保留的值与子集关系，能够直接影响后端的数据搬运决策。不过该实现主动放弃复杂别名、跨 Block 和无法确认的动态等价情况；这些情况仍需要更深入的分析，不能通过放宽几条判断就默认安全。

读完可以用一个变化检查理解：如果 Extract 和 Insert 的类型都没变，只把 Insert 的 offset 换成另一个 SSA 值，为什么就不能按原样优化？答案应落在“第一次 Store 是否完成了最终要求的子集更新”上，而不只是类型是否匹配。

## 源码与阅读衔接

本篇对照 AscendNPU-IR 基线 `release/v1.1.x` 的提交 `428ab8fdab46`，团队实现固定为 `a4abee867832021a1714987ed048c87a08be7234`，核对日期为 2026-10-16：

- [团队设计报告](https://gitlab.eduxiji.net/educg-group-49265-3230617/T2026101412010750-1832/-/blob/a4abee867832021a1714987ed048c87a08be7234/%E8%AE%BE%E8%AE%A1%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.pdf)：3.4 节，印刷页 56–61，PDF 第 64–69 页。
- [基线的搬运插入规则](https://github.com/Ascend/AscendNPU-IR/blob/428ab8fdab46/bishengir/lib/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV.cpp)：定位 `InsertLoadOpBetweenStoreLikeAndVectorOrCube` 与 InsertSlice 注册位置。
- [团队实现](https://gitlab.eduxiji.net/educg-group-49265-3230617/T2026101412010750-1832/-/blob/a4abee867832021a1714987ed048c87a08be7234/bishengir/lib/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV.cpp)：定位 `isStoreBackedIdentityInsertSlice`、`InsertLoadForStoreBackedInsertSlice` 与 `populateInsertLoadStorePattern`。
- [团队回归测试](https://gitlab.eduxiji.net/educg-group-49265-3230617/T2026101412010750-1832/-/blob/a4abee867832021a1714987ed048c87a08be7234/bishengir/test/Dialect/HIVM/insert-load-store-for-mix-cv.mlir)：从 `store_backed_identity_insert_static` 开始阅读。

本文是报告、基线与团队源码的对照解读，未构建该团队编译器，未进行 NPU 执行或性能复现。正文中实际测试输入、源码摘录与用于解释的搬运示意已分别标明。

基础语义可回看 [Tensor 切片与局部更新](/notes/compile/mlir/dialects/tensor/slices)。进一步理解旧值使用为什么阻止原地复用，可继续阅读 [Bufferization](/notes/compile/mlir/compiler/memory/bufferization)。
