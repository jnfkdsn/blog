---
order: 120
title: LLVM Dialect：低层表示
updated: 2026-10-04
---

# LLVM Dialect：低层表示

同一个数组访问在高层写成 MemRef load，在低层需要变成指针运算和标量读取。本模块先读懂这些低层操作，再解释访问元数据的表示，最后连接 LLVM 自己的 IR 对象。

## 阅读前置

Tensor/Linalg、MemRef 与 Bufferization；需要时回查 CF 和 Dialect Conversion。

## 阅读顺序

| 顺序 | 章节 | 贯穿过程 |
|---|---|---|
| 1 | [类型、指针与低层操作](./operations) | 以两个整数求和理解 GEP、load、数值和指针的区别。 |
| 2 | [DataLayout 与 MemRef 描述符](./data_layout) | 把窗口的基址、offset、sizes/strides 落实为低层字段，比较 index 位宽。 |
| 3 | [LLVM Dialect 与 LLVM IR](./translation) | 对照指令与 PHI 的实际翻译，区分解析、lowering 和 translation。 |

## 完成边界与衔接

三篇主线已有正文。随后接到[CPU Lowering](../../compiler/lowering/)的调用边界与执行；可先读第一篇，再按 lowering 案例需要回查描述符与 translation。

示例以 LLVM 20.1.8 和当前 64 位 CPU 为验证目标，目标布局与 ABI 需要随实际项目核对；设备路径归入后续目标模块。整个课程见[章节目录](../../chapter_plan)和[覆盖表](../../coverage)。
