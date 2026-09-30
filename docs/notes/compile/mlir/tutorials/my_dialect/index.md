---
order: 30
title: 从 my.add 开始开发 Dialect
updated: 2026-10-01
excludeFromSidebar: true
---

# 从 my.add 开始开发 Dialect

这组教程围绕一项完整工作展开：定义一种操作，让工具能够读取和检查它，再用变换把它转成后续阶段支持的表示。`my.add` 使用熟悉的整数加法，便于把注意力放在各机制怎样配合上。

[01：让自己的工具认识 my.add](./01_minimal_dialect)从操作的结构与语义出发，解释 ODS 怎样生成 C++ 能力、注册怎样连接框架，以及这些能力如何在读取具体 IR 时发挥作用。代码、关键生成结果和工具输出直接在文中展示，可以先独立阅读，再选择实践。

| 阶段 | 要理解的过程 | 内容范围 |
|---|---|---|
| 01：认识新操作 | 定义 → 生成代码 → 接入工具 → 读取、验证与打印 | 已有独立讲解和最小参考工程 |
| 02：接回变换 | 读取新操作的输入，创建 arith.addi 并转接使用 | 后续章节；第一篇先解释连接关系 |
| 后续深入 | 操作的额外约束、通用消费者及更丰富的表示 | 随具体需求展开 Trait/Interface、Type/Attr、Region 等机制 |

[定义 IR 抽象](../../compiler/ir_definition/)用于回查具体机制，不要求先读完全部章节才能进入这里。C++ 的[编译链接](/notes/cpp/source_to_program)、[宏和生成代码](/notes/cpp/generated_cpp)、[模板调用](/notes/cpp/template_calls)也可按需要选读。

需要实际修改时，从 [Lab 入口](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)取得工程与任务。先预测一个变化，再运行观察并解释结果；构建步骤和练习要求在该入口集中维护。
