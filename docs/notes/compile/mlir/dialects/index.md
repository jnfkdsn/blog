---
order: 20
title: 方言语义
updated: 2026-10-04
excludeFromSidebar: true
---

# 方言语义

Dialect 组织一组相关操作、类型、属性和扩展行为；它不是固定的一层 IR。不同 Dialect 可以共同组成一个程序，操作自身的契约决定它如何解释输入、结果和内部区域。

| 当前章节 | 抽象边界 |
|---|---|
| [Builtin](./builtin) | 模块与公共类型/属性基础 |
| [Func](./func) | 函数定义、声明、调用与返回 |
| [Arith](./arith) | 标量常量、运算、比较、选择与数值边界 |
| [CF](./cf) | 显式 CFG、分支与沿边传值 |
| [SCF](./scf/) | 结构化条件和循环；if/for/while 的区域协议 |

[Tensor](./tensor/)、[Linalg](./linalg/) 与 [MemRef](./memref/) 各三篇正文已提供，连接张量值、形状、访问关系、归约、DPS，以及存储布局与视图别名。[LLVM 三篇](./llvm/)继续解释低层操作、数据布局和 translation。[Math/Index/Complex](./math/)补充数学实现和尺寸计算。其他模块按下表继续展开：

| 目录 | 主要内容 |
|---|---|
| [Affine](./affine/) | 三篇已有：映射、整数域、循环依赖与适用边界 |
| [Vector](./vector/) | 三篇：向量值、transfer/mask 尾块、归约/收缩与目标表示 |
| [GPU](./gpu/) | 设备执行表示与边界 |
| [Math / Index / Complex](./math/) | 数学实现、索引位宽、复数计算展开 |
| [领域表示与扩展](./extensions/) | Quant、Sparse、Shape、IRDL及StableHLO/TOSA定位 |

具体可检查项在[覆盖表 D01—D18](../coverage)。方言语义和使用它们的变换分开归档，以案例链接；不将方言目录顺序视为固定编译流水线。

每个操作章节按“契约 → 执行/传值规则 → 完整例子 → 变体与约束 → 反例 → 相邻表示 → 源码与检查”展开。初读无需背全方言操作名，但不能跳过会改变含义的边界。
