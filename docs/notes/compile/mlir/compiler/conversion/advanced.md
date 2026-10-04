---
order: 5
title: 复杂转换路径与框架行为
updated: 2026-10-02
---

# 复杂转换路径与框架行为

前面的主线已经足以实现一个带类型变化的转换。进入更大工程后，新的困难通常来自三个方面：不是所有区域都归当前阶段处理，一个源值可能展开为多个目标值，局部规则之间还可能有多条转换路径。

本篇是按需参考。每个主题都从已经熟悉的转换提出一个具体变化，不要求在开始 Tensor/Linalg 前记住全部细节。analysis conversion 与递归合法性有可运行对照；本篇的复数 1:N 仍为表示设计推演；[pair 的 1:N 章节](./one_to_many)另提供已验证的完整工程。

## 1. 不改变输入的可转换性检查

假设希望先了解“当前规则能处理哪些操作”，再决定是否启用某条 pipeline。直接运行转换后再观察，会失去原来的输入表示。

`applyAnalysisConversion` 可以分析可合法化的操作，并在结束后保留原 IR。框架仍会尝试转换过程，但不会把这些改写作为最终输入修改提交。配套程序使用：

<!-- range-source: analysis -->
```cpp
DenseSet<Operation *> legalizable;
ConversionConfig config;
config.legalizableOps = &legalizable;
LogicalResult result = applyAnalysisConversion(module, target, std::move(patterns), config);
module.walk([&](my::LimitOp op) {
  llvm::errs() << "limit_legalizable=" << legalizable.contains(op.getOperation()) << "\n";
});
return result;
```

对上一章的 limit 输入，报告为：

```text
limit_legalizable=1
```

同时打印出的函数仍包含 `my.limit` 和 range 类型。一个输出说明规则可用于该操作，另一个输出说明输入未被正式改写。两者需要一起观察。

可合法化集合记录的是原有操作。analysis 的成功状态不能当成 full conversion 的成功承诺：某个操作可转换，不代表整个模块都能满足目标，集合中的对象也不是未来目标 IR 的枚举。例如同时加入当前规则不处理的 `my.clamp`，可以观察到 limit 可合法化而 full conversion 仍失败。

这种检查适合诊断规则覆盖，不提供程序语义等价证明，也不替代完整 pipeline 的执行测试。

## 2. 递归合法性与编译边界

第一章将 func 本身声明为合法时，仍然继续处理函数体。如果当前阶段明确把整个函数当作另一组件负责的区域，可以进一步设置递归合法性：

```cpp
target.addLegalOp<func::FuncOp>();
target.markOpRecursivelyLegal<func::FuncOp>();
```

此时函数及其内部都被当前目标视为可接受，即使内部包含原本标为非法的 limit。配套 recursive 模式会成功，并保留原始计算。

这不是更强的验证，而是作者主动划定一个“不在此处展开”的范围。它适合明确的编译边界；如果为了消除报错就给所有函数加上递归合法性，很可能只是把本该完成的工作排除在检查之外。

动态递归合法性还可以仅为某些实例设置这种边界。使用前应回答：被保留的子树下一步由谁处理，它需要遵守什么接口？没有这个后续责任，局部成功并没有构成一条完整编译链。

## 3. 一个源值展开为多个目标值

前面的 range→i32 是 1:1 映射。考虑另一种表示设计：一个复数值用实部与虚部两个标量保存。以下只是映射示意，不是新教学 Dialect 的声明：

```text
源值 z : complex<f32>
          ↓
目标值组 [real : f32, imag : f32]
```

类型层面产生两个类型，还只是第一步。沿使用链看，变化包括：

| 位置 | 需要维护的对应 |
|---|---|
| 生产者 | 一个旧结果对应两个新结果 |
| 消费者 | 原来的一个 operand 现在对应一个值组，需知道 real/imag 的顺序 |
| Block 参数 | 一个参数可能拆成两个，后续参数的位置也会变化 |
| 函数与调用 | 参数、返回值与每个调用点按同一规则展开 |
| 旧边界 | 若仍需要复数表示，必须有合法的重建或桥接方式 |

假设原操作有两个 operand，分别是复数 z 和标量 s，转换后的值组长度分别为 2 和 1。把它们直接压成三个 Value 后，仍需要知道前两个属于 z，第三个属于 s；不能继续假设原第 i 个 operand 就是新列表第 i 项。

固定版本的 `OpConversionPattern` 提供 `OneToNOpAdaptor`，用于保留每个原 operand 对应的 ValueRange。标准调用转换代码也会记录每个原结果展开的数量，再建立替换映射。编写自定义规则时，应先画出这种分组与顺序，再选择匹配的 API。

1:0 也需要解释：某个类型若不再产生运行时值，它的静态信息或语义责任必须已经转移到其他地方。不能仅返回空类型列表，就假定所有使用都自然合理。

本模块没有提供复数的完整 1:N 工程。需要实现这种任务时，应同时设计生产者、消费者和边界，而不是只给 TypeConverter 加一条返回多个类型的回调。

## 4. 转换路径与新操作的合法化

一条规则可以先生成中间操作，再由另一条规则继续处理：

```text
源操作 A → 中间表示 B → 当前目标 C
```

框架会考虑规则产生的操作能否继续合法化。第一章“漏掉 constant 的许可”就是最小例子：源 clamp 的规则看似已经完成替换，但新常量没有通向合法目标的路径，整个候选转换仍不能成立。

如果还有其他规则可以处理 B，路径可能继续；如果都不能处理，必须回到目标与规则之间查找缺口。Pattern benefit 可以影响规则选择，却不能把不合法的最终状态变合法，也不是保证全局性能最优的代价模型。

实际定位时建议区分三种失败位置：

1. **匹配前提不成立**：形状、属性或支持范围不满足规则要求。
2. **新操作无法合法化**：规则成功创建了中间表示，但没有后续路径。
3. **类型边界无法落实**：新旧使用关系要求 materialization，却缺少实现。

对应的处理分别是检查输入与匹配条件、补齐目标或后续规则、追踪仍使用旧表示的对象。它们可能最终表现为同一个源操作“无法合法化”，不能只看操作名推断原因。

## 5. 改写可见性、回退与版本

ConversionPatternRewriter 需要管理转换中的尝试。创建、替换、删除以及区域参数变化，并不都必须像普通手写 IR 修改那样立刻以最终形式出现在对象树上。

在本系列固定版本的转换实现中，一些替换关系先保存在转换映射中，失败路径需要能够恢复。于是，在规则执行到一半打印 IR，可能看到旧操作与新操作同时存在。判断新消费者应使用哪个值，应遵守 adaptor 和 rewriter 的协议，而不是仅根据当前打印猜测旧结果已被全面 RAUW。

这并不意味着 conversion 会替作者撤销所有 C++ 副作用。写文件、修改外部容器或绕过 rewriter 操纵对象，不会因为规则回退就自动恢复。因此，不要在匹配成功前执行无法回退的外部动作，也不要依赖未提交 IR 的中间状态作为最终结果。

当前上游文档还介绍了可配置的 rollback/no-rollback 行为，但 LLVM 20.1.8 的公开 ConversionConfig 不包含新版同名开关。阅读实现和移植代码时，需要核对所用版本，不能把最新配置片段直接粘贴到旧工具。

## 6. 进入下一模块的边界

学完主线后，近期应能解释并完成一个有限转换：选择新表示，转换生产者与使用者，处理必要边界，区分源验证、合法化与语义正确性。analysis、递归合法性和 1:N 用于解决更具体的工作，不增加为所有读者的即时结业任务。

下一步可以阅读[后续章节目录](../../chapter_plan)，进入 Tensor/Linalg 与编译流程导读。那时同一套判断会用于更有算法含义的对象：形状、迭代空间、张量结果与存储。

## 实现与深入入口

[配套工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/08-type-conversion)提供 analysis、recursive 与主线失败对照。1:N 示例在本篇仅作表示推演，未宣称已构建完整复数转换。

固定版本可查 [DialectConversion.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/DialectConversion.h) 的 OneToNOpAdaptor、ConversionConfig 和签名转换入口，以及 [FuncConversions.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp) 的调用结果展开。driver 的规则路径与恢复逻辑位于 [DialectConversion.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Utils/DialectConversion.cpp)；先围绕一个失败案例进入，不从文件第一行开始通读。
