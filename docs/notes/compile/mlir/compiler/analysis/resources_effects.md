---
order: 60
title: 效果的对象、资源与阶段
updated: 2026-10-04
---

# 效果的对象、资源与阶段

[效果与别名](./alias_effects)已经解释：不能只根据结果没人使用，就删除一个写内存的操作。但“这条操作会写”仍可能过于粗糙。分析还需要知道它写哪个对象、作用于哪类资源、是否覆盖整个对象，以及多个效果之间提供了怎样的顺序信息。

本章从memref.copy读取src、写入dst的查询结果出发，再区分Resource与stage的作用。它们描述编译器可使用的事实，不是额外生成的运行时锁或事件。

## 1. Copy 的两个效果实例

```text
memref.copy %src, %dst : memref<4xi32> to memref<4xi32>
```

固定接口查询得到：

```text
read  resource=<Default> stage=0 full=1
write resource=<Default> stage=0 full=1
```

Read实例关联src operand，Write实例关联dst operand；默认Resource表示通用内存资源。full=1说明作用范围覆盖各自对象的完整区域，而不是只描述一个未确定的部分访问。

这里“完整区域”属于效果范围描述，不是说操作拥有一个MLIR Region。词相同但对象层次不同：memref.copy本身没有body Region。

若分析要判断copy与另一条store能否交换，还需要查询src/dst与store目标是否alias。效果接口提供访问对象，别名/访问分析继续判断它们是否可能相互影响。

## 2. Resource 区分哪类状态

同一函数中还有memref.alloca。固定查询得到Allocate，Resource为AutomaticAllocationScope，stage=0，full=1。这个资源描述分配发生在自动分配作用域内，与普通读写的默认资源不同。

Resource是编译器建模中的状态类别。项目也可以定义例如统计计数器、特定外部状态等资源，让效果消费者知道操作影响的不一定是某个显式memref值。固定探针构造了名为TeachingCounter的Resource及其Write实例，展示这种协议对象。

这只是效果模型；给Resource起一个新名字，不会创建硬件内存空间，也不会自动证明它与所有其他状态互不影响。任何利用资源区分来放宽排序的分析，都必须遵守生产者与消费者共同的语义约定。

## 3. Stage 表达操作内部的效果顺序

设某个领域操作的契约是先完整读取输入，再写输出，可以用stage0的Read与stage1的Write表达先后。探针单独构造这两个EffectInstance，并读回0和1，说明接口如何携带这项顺序。

但这不是上面memref.copy声明的阶段：固定copy的两个效果都是stage0，没有借此提供跨stage先后信息。不能看到源代码习惯上“先读后写”就擅自把实际查询解释成0→1。

stage也不是GPU pipeline编号、同步事件ID或Pass编号。它属于一个操作的效果描述；不同操作都写stage0，不意味着它们在目标程序中并行，也不意味着已经同步。

## 4. Full Effect 与优化结论的距离

假设一个操作完整覆盖dst，之前写入dst的内容可能不再需要。但要删除旧写入，还必须确认期间没有可观察读取、没有其他alias逃逸路径，并且新操作在相关执行路径上确实发生。

同理，完整Read不意味着结果一定依赖每个字节；效果是访问契约，具体数值依赖可能需要更细分析。不要让一个字段承担它没有表达的证明。

One-Shot Bufferization还有自己的BufferizableOpInterface，描述tensor operands的读写与结果alias关系。MemoryEffectOpInterface的事实有相近含义，但不能简单替代那套协议；应使用当前消费者实际查询的接口。

## 5. 查询事实与实现优化分别验证

本章作者工程实际查询了memref.copy/alloca的效果，并构造自定义资源和阶段实例；没有实现一个依据这些字段进行重排的Pass。这样可以确认协议和当前生产者的输出，而不把“能读字段”写成“已经证明一般调度安全”。

当项目确实需要这项优化时，应先选择一个受限场景，把效果、alias、支配与控制条件合在一起，给出允许和拒绝的输入；再通过语义检查与必要执行证据验证。沿这个任务深入，比先背完整接口参数更容易理解。

## 依据与实践

固定来源：[效果实例与Resource](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/SideEffectInterfaces.h)、[ODS效果描述](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Interfaces/SideEffectInterfaceBase.td)、[MemRef实际声明](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)。

[26 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/26-ir-maintenance)把真实操作查询与synthetic protocol输出分开。有限练习：对一个只读单元素的操作查询效果，比较它与copy的对象/覆盖信息，再说明还缺哪项事实才能决定是否跨store移动。
