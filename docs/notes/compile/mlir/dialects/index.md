---
order: 20
title: 方言语义
updated: 2026-09-06
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

后续按依赖展开 tensor、memref、linalg、affine、vector、gpu/async、llvm 和目标方言；math/index 等在相应算法需要时深化。具体可检查项在[覆盖表 D01—D18](../coverage)，不通过先生成空页面代替内容。

每个操作章节按“契约 → 执行/传值规则 → 完整例子 → 变体与约束 → 反例 → 相邻表示 → 源码与检查”展开。初读无需背全方言操作名，但不能跳过会改变含义的边界。
