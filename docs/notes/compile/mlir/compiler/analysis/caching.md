---
order: 3
title: 分析结果的缓存与失效
updated: 2026-10-04
---

# 分析结果的缓存与失效

上一章的分析给了优化一个判断依据，但编译器还会继续修改 IR。删掉一条分支后，哪些 Block 支配哪些 Block 可能改变；增加一次使用后，某个值原先的“最后使用位置”也可能改变。分析结果因此总是关于某一份 IR 状态的结论。

如果每个 Pass 都重新计算全部事实，会浪费时间；如果一直保留旧结论，又可能让优化依据失效。本章用一个容易核对的统计过程说明 MLIR 的处理方式：需要时计算，允许时复用，修改后按有效性约定丢弃并重算。

## 1. 连续查询同一个函数

先把复杂的支配树换成一个简单分析：统计函数本身及其内部所有操作的数量。对[前文分支函数](./dominance_liveness)，这个数是 9：函数一个、常量一个、三条算术操作、if 一个、两个 yield、return 一个。

第一次统计遍历整个函数，得到 9。紧接着另一个只读 Pass 也需要这个数，若 IR 完全没变，它可以复用同一份结果。此时分析对象可理解为：

```text
当前函数 IR
    ↓ 第一次查询
OperationCount { count = 9 }
    ↓ 后续查询
读取已有 count，不再遍历
```

MLIR 的分析通常是独立 C++ 类，由 AnalysisManager 针对所分析的 Operation 按需构造并缓存。Pass 通过 `getAnalysis<OperationCount>()` 请求它。分析本身读取 IR，不承担改写职责。

最小的数据部分如下；计数函数 walk 时包含传入的根操作：

```cpp
struct OperationCount {
  explicit OperationCount(Operation *op) : count(countOperations(op)) {}
  unsigned count;
};
```

构造函数中的 `count` 是计算结果，不是以后会自动跟着 IR 增减的视图。理解这一点，就能理解缓存为什么可能过期。

## 2. 修改 IR 后的重新计算

现在安排四个 Pass：两个报告 Pass、一个在 return 前增加死常量的修改 Pass、再一个报告 Pass。

```text
report → report → insert constant → report
```

报告 Pass 只读取分析，并说明全部分析保持有效；修改 Pass 增加一个操作，没有承诺保留 OperationCount。固定版本的实际轨迹是：

<!-- analysis-output: cache -->
```text
build OperationCount=9
report cached=9 fresh=9 stale=0
report cached=9 fresh=9 stale=0
build OperationCount=10
report cached=10 fresh=10 stale=0
```

为观察过程，实验在构造分析时打印 `build`，报告时同时打印缓存值 `cached` 和临时重新遍历得到的 `fresh`。正式消费者通常只需要缓存结果；额外遍历是这里的对照手段。

两次初始 report 之间没有第二条 build，说明结果确实被复用。插入常量以后，第三次 report 重新触发构造，结果变为 10。

这不是 AnalysisManager 检测到“恰好增加了一条操作”并自动修正了 count。它根据 Pass 的保留声明，判定原来的分析需要失效；直到下一次请求时才重新计算。

## 3. 保留声明与正确性责任

报告 Pass 只读 IR，因此结束时可以调用：

```cpp
markAllAnalysesPreserved();
```

修改 Pass 默认不应做这项承诺。如果它只改变某种与特定分析无关的内容，可以精确保留已经证明仍然有效的分析：

```cpp
markAnalysesPreserved<SomeAnalysis>();
```

保留声明描述的是“现有分析对象还能正确回答它负责的查询”。不能仅依据“这次改动很小”判断。例如在一个 Block 中增加常量，没有增加 CFG 边，但操作计数和某些基于操作集合的摘要已经变了。是否能保留 DominanceInfo，还要看具体修改与该分析的实现；不要由一个分析可保留推出所有分析都可保留。

把插入常量的 Pass 故意改成保留 OperationCount，实际输出变为：

<!-- analysis-output: stale -->
```text
build OperationCount=9
report cached=9 fresh=9 stale=0
report cached=9 fresh=9 stale=0
report cached=9 fresh=10 stale=1
```

这次没有重建，最后报告还拿着 9，但当前函数已有 10 个操作。IR 本身仍能通过 verifier，因为 verifier 检查的是 IR 契约，而不是这个分析对象私有整数的准确性。

真实错误可能更隐蔽：一个旧支配结论允许替换本来不可用的值，或者一个旧别名结论允许删除实际必要的读取。错误并不一定在产生缓存的那个 Pass 暴露。

## 4. 单个 Pass 内部的过期结果

Pass 之间的失效管理没有解决一个 Pass 自己内部的所有问题：

```cpp
auto &info = getAnalysis<SomeAnalysis>();
// 利用 info 判断候选。
// 修改 IR。
// 再利用 info 判断下一个候选：此时必须确认它仍然有效。
```

修改 IR 之后再次调用 `getAnalysis`，也不能把它当作强制刷新命令。当前缓存如果仍在，就可能返回同一个对象；框架不会根据每条 builder 调用猜出所有分析的变化。

一种简单策略是先基于未变的 IR 收集候选，再按已证明相互兼容的条件实施修改。另一种策略是修改后显式维护所用分析，或在合适的位置清理并重新计算。选择取决于分析提供的更新能力、成本，以及后续判断是否依赖修改后的状态。

关键是界定结论的有效期。例如删掉一个使用者以后，旧的“此值仍活跃”可能只是保守；新增一个使用者以后，旧的“此值已经死亡”可能会导致错误。保守与错误的方向也要按具体分析判断，不能把所有旧结果都当作同一种问题。

## 5. 分析之间的依赖

复杂分析可能基于其他分析构造。比如一个摘要先查询调用图，再汇总被调用函数的行为。如果调用图失效，摘要仅仅“自己没有被修改”并不足以继续有效。

MLIR 允许分析构造时接收 `AnalysisManager &`，查询依赖；也允许通过 `isInvalidated` 钩子明确失效条件。下面是说明依赖关系的示意结构，省略实际摘要计算：

```cpp
struct SummaryAnalysis {
  SummaryAnalysis(Operation *op, AnalysisManager &am) {
    auto &calls = am.getAnalysis<CallGraph>();
    // 基于 calls 和 op 构造摘要。
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<SummaryAnalysis>() ||
           !pa.isPreserved<CallGraph>();
  }
};
```

这段表达两个必要条件：摘要自身被允许保留，所依赖的调用关系也被允许保留。它的意义是把正确性依据写入管理协议，而不是让使用者记住某个隐含顺序。

具体项目可能有更精细的条件，例如某类属性修改不影响分析。应从分析实际使用的信息推导保留规则，先保证正确，再考虑减少重算。

## 6. 分析范围与 Pass 层次

Module 级的调用图和单个函数的活跃性具有不同分析范围。函数 Pass 不能为了方便，就在并行处理其他函数时随意读取或修改兄弟函数。父级缓存也有专门查询接口，不能把“我能拿到父 Operation 指针”当成可以现场计算所有父级事实。

工程上可以由较外层的 Pass 建立跨函数事实，再让较内层的消费者在协议允许的范围内使用；若跨函数变换改变这些事实，则由合适层次安排失效和重算。[Pass 组织](../transforms/passes)中的嵌套关系在这里影响了分析可见性与并发安全。

因此，写优化时除了确定 `runOnOperation` 的 IR 范围，还要确定依赖事实的范围、生命周期和修改责任。分析缓存属于编译器正确性的一部分。

## 阅读自查

1. 最后一次 report 返回 9 时，为什么 verifier 没有报错？
2. 同一个 Pass 先取得分析，再修改 IR，再 `getAnalysis`，为什么不保证拿到新结论？
3. 一个分析基于 CallGraph 计算摘要，只保留摘要自己是否充分？

下一篇[数据流传播与收敛](./dataflow)会进入分析内部，解释某些事实如何通过传播计算出来；本篇解决的则是这些事实被计算以后如何保持可用。

## 实现依据与实践

主例在 LLVM/MLIR 20.1.8 上实际运行，正确与错误保留模式都保存在[分析实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/12-analysis)。失效机制依据 [AnalysisManager.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Pass/AnalysisManager.h)；官方 [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/#analysis-management)提供查询、保留和依赖的接口说明。实验关闭多线程以使日志顺序稳定，没有用共享可变全局计数器冒充真实分析状态。
