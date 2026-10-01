---
order: 30
title: 自定义 Dialect 入门
updated: 2026-10-01
excludeFromSidebar: true
---

# 自定义 Dialect 入门

[自定义操作与 Dialect 接入](./01_minimal_dialect)用 my.add 建立一条完整认识：操作定义如何生成 C++ 能力，注册如何连接框架，这些能力又如何用于读取和验证具体 IR。

这是一篇贯通导读。后续机制统一在[定义 IR 抽象](../../compiler/ir_definition/)展开，不在这里再用 my.add 复制一套 Type、Attribute、Interface 与 Region 系列。

## 阅读衔接

1. 首次接触定义与注册时，先读 [my.add 导读](./01_minimal_dialect)。
2. 理解接入过程后，进入[操作定义：语义、字段与验证](../../compiler/ir_definition/op_definition)。它用 clamp 说明更丰富的字段设计，并给出实际 Pattern 展开。
3. 再按需求学习接口、类型/属性、区域与解析打印。

my.add 保留为最小练习工程，目前实现定义、读取、验证和打印。其普通 Pattern 展开未实现；机制学习不以等待该阶段为前置。

## 可选实践与前置参考

[Lab 入口](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)提供完整源码、构建说明与字段改名任务。先预测一个变化，再观察它对生成接口和 IR 的影响；无需先完成实验才能阅读机制章节。

若阅读具体代码时遇到障碍，可按需查阅 C++ 的[编译链接](/notes/cpp/source_to_program)、[宏与生成代码](/notes/cpp/generated_cpp)和[模板调用](/notes/cpp/template_calls)。它们不是整个模块的统一前置关卡。
