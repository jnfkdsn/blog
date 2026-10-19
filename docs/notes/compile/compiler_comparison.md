---
order: 80
title: TVM 与 PyTorch Compiler 的系统对照
updated: 2026-10-17
---

# TVM 与 PyTorch Compiler 的系统对照

两套教程使用相同的数学模型 `relu(x @ weight + bias)`，最终小矩阵结果都是 `[[0.5,7.5],[1.5,0]]`。它们在途中却采用不同表示、不同调用组织与实现选择。

比较的目的，是找出可迁移的系统问题：输入怎样进入编译器，哪些信息支撑优化，何时选择算子实现，生成物怎样被运行。不是把两套组件按名字逐一对号入座，更不是将它们连成一条实际不存在的强制编译链。

## 1. 两条独立的实际路径

本系列TVM实验采用：

```text
简单PyTorch模型
  → FX symbolic tracing
  → TVM importer
  → Relax
  → call_tir与TensorIR实现
  → 图分组、函数组合和调度
  → LLVM数值代码 + Relax VM执行组织
  → NDArray输入与结果
```

PyTorch Compiler实验采用：

```text
Python函数 + 运行输入条件
  → Dynamo捕获FX图与适用条件
  → 图规范化 / AOT相关处理
  → Inductor计算、索引、布局和调度
  → 外部addmm + 生成的CPU C++内核 + wrapper
  → Tensor结果
```

TVM实验使用PyTorch来取得源图，只是一种导入方式。它不意味着默认 `torch.compile` 自动调用TVM。类似地，这两条CPU实验都没有通过MLIR；已有MLIR知识帮助理解IR与转换问题，但不是所有AI编译器都必须使用的实现框架。

## 2. 捕获与导入的区别

Dynamo面对的是Python执行语义，必须判断哪些行为可以进入图、哪些依赖需要保护，以及断图后如何继续执行。TVM importer面对的通常是已经取得的源图或模型表示，需要翻译操作、参数和结构信息。

两者都可以因为不支持某种行为失败，但失败位置不同。若Python数据相关分支无法得到预期图，应先检查捕获；若源图已存在，而某个节点没有Relax转换规则，应检查导入。

这个区别有助于实际开发：为一个新算子补源框架捕获支持，和为TVM补importer/lowering，并不是同一项修改。

## 3. 高层关系与低层执行信息

| 系统问题 | TVM主例 | PyTorch Compiler主例 |
|---|---|---|
| 模型计算关系 | Relax绑定、调用、dataflow | FX节点和引用关系 |
| 结构信息 | StructInfo、符号shape及推导 | 节点元数据、FakeTensor、符号关系 |
| 数值程序 | TensorIR block、循环、buffer | Inductor索引函数、计算对象、buffer/外部实现 |
| 执行选择 | Schedule、规则/搜索、库/BYOC | 调度、融合、模板/库/生成代码选择 |
| 调用组织 | Relax VM及runtime模块 | wrapper及PyTorch运行时集成 |

这张表对应职责，不表示某一行里的对象具有完全相同语义。例如TensorIR显式暴露可操作的循环和block；Inductor也有循环范围和索引表示，但其对象组织和延迟计算/realization机制有自己的设计。

真正可迁移的问题是：一个中间量现在只是逻辑值，还是已经要求存储？下一次变换还能利用哪些结构？哪些事实只是调用假设，哪些已经写进IR契约？

## 4. 相同模型的不同实现

TVM固定例子在FuseTIR后得到一个包含matmul、add、relu的PrimFunc，函数内仍有中间buffer。后续low-level优化可以继续处理它们。

PyTorch固定CPU例子将mm+add选择为外部addmm，ReLU由生成的C++内核完成，并复用输出存储。

这两个结果说明“融合”至少要回答：在哪一层融合？是否越过库边界？循环和临时存储是否改变？不能仅比较一个图里有几个节点或一个函数名里有没有fused。

也不能根据这些结构直接排名性能。数值库、目标代码、输入大小和调用成本都不同。要比较性能，需要固定同一目标、输入、精度、线程、预热和计时范围，并解释成本来自哪里。本系列没有从这两个小例子推导一般性能优劣。

## 5. 动态形状与接口条件

TVM的手写符号Relax实验让一次编译后的程序处理batch1、2、5；PyTorch的动态捕获实验观察到batch2、5、2复用一份图实现。

两者都在利用符号维度关系，但外围责任不同。TVM导入/构造的模型已具有明确计算描述，runtime据接口处理shape与调用；Dynamo还需要保护Python捕获时的条件，在不同调用下决定是否使用已有版本。

PyTorch export进一步把输入集合写成显式范围，如 `2≤B≤8`，再交给AOT部署。它与JIT的按需特化提供了不同的使用边界。

当某个模型面对大量变化时，应先问变化属于形状、数据相关控制流、状态还是布局，再判断需要符号化、多个版本、运行时检查或不同算法。只添加“dynamic”选项通常不足以解释全部行为。

## 6. 从图编译到算子编译

当进入Triton/Triton-Ascend时，可以接住本轮已经明确的边界：图层已经决定哪些计算由某个kernel完成，kernel编译器继续决定如何表达和映射这段计算。

后续可沿一段Softmax或Matmul追踪：

```text
输入语义、shape、dtype、stride、mask
  → kernel程序与高层IR
  → 布局和并行分工
  → 数据搬运、存储、同步和流水
  → 目标指令与产物
  → launch、数值与性能
```

GPU与Ascend NPU需要分别跟踪真实后端，不把GPU的线程/共享存储方案机械换名套过去。硬件执行单元、存储层次和同步机制会影响合法性与代价。

全链路能力仍然有用：图融合决定kernel边界，shape与layout影响特化，runtime决定如何调用、缓存和交换数据。若之后希望修改图层或部署系统，这一轮也保留了相应入口。

## 7. 用实践选择深入方向

当前不需要提前固定职业方向。可以依次完成三类有限工作：

- 在FX图中实现有语义前提的小改写，检查数据关系和输出别名。
- 给TensorIR矩阵函数实现一份schedule，解释尾部、归约和性能变化。
- 修改一个输入或部署条件，定位失败发生在捕获、图契约、代码生成还是运行时。

这些任务分别体验图层、算子实现和系统集成。完成后记录自己能独立解释与修改的范围，再决定下一轮在哪一层深入。专家能力仍需持续解决真实项目问题；一套教程和参考实验提供基础，不自动等同于熟练掌握所有层次。

## 8. 阅读与实践入口

建议先读 [TVM总览](/notes/compile/tvm/01_execution) 与 [PyTorch总览](/notes/compile/pytorch-compiler/01_execution)，建立两张地图；接着分别完成 [TVM主线](/notes/compile/tvm/) 和 [PyTorch主线](/notes/compile/pytorch-compiler/)，最后回到本篇比较。

本轮交付范围和证据归属见 [课程覆盖表](/notes/compile/compiler_courses_coverage)。配套实践分别位于 [TVM labs](https://github.com/jnfkdsn/aicompiler/tree/main/tvm-e2e) 和 [PyTorch labs](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。后续Triton课程将继续使用已建立的索引、归约、布局和调用边界，不要求重新从头学习整个系统。
