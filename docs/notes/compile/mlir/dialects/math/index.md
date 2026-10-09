---
order: 130
title: Math、Index 与 Complex
updated: 2026-10-04
---

# Math、Index 与 Complex

张量算法既需要指数等数学计算，也需要尺寸和边界计算。这些操作都可以很小，却会影响整条编译链：指数怎样选择目标实现，索引怎样适应目标位宽，都需要在降低时保留正确含义。

| 阅读顺序 | 章节 | 要解释的过程 |
|---|---|---|
| 1 | [标量数学、指数与数值实现](./scalar_math) | 从 Softmax 权重到 exp 操作、数学库/LLVM intrinsic，再用极小 expm1 比较实际精度 |
| 2 | [Index 运算与尺寸计算](./index_arithmetic) | 从 10 个元素分为 3 块到 ceildiv、余块、32/64 位折叠与实际访问保护 |
| 3 | [复数语义与标量展开](./complex) | (1+2i)(3+4i)→实虚部算术，再解释绝对值的缩放算法 |

Math/Index有正文、转换与本机观察；复数篇核对计算展开与常量折叠。先理解 Arith 与基本循环即可阅读；它们随后进入[归约与 Softmax](../../tutorials/kernels/reduction_softmax)。无需先记住数学操作目录，遇到具体算法再按契约查询。

Math 的数学含义、lowering 路径和误差约束分层说明；Index 的算术正确性与内存访问边界分别检查。实际目标的数值、性能和特殊值策略仍需随实现核验。
