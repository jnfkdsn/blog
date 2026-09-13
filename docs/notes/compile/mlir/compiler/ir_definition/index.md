---
order: 20
title: 定义 IR 抽象
updated: 2026-09-13
excludeFromSidebar: true
---

# 定义 IR 抽象

[IR 变换基础](../transforms/)解释怎样处理已有操作。本模块转向定义这些操作：编译器需要知道它表达什么、保存哪些信息、哪些形式合法，以及哪些能力可以交给通用基础设施使用。

## 从一项计算连接到操作定义

以 `lab.clamp` 为例，先规定“将一个有符号整数限制在固定闭区间内”，再决定输入用 SSA operand 表达，上下界用静态属性保存。ODS 把这些字段与约束接入生成的 C++ API，verifier 检查对象是否满足契约，注册与 parser/printer 使工具能够读写这种操作。

```text
语义约定
  → 输入、结果、属性与类型约束
  → ODS 和生成的 C++ API
  → 构造、完整验证、注册与解析打印
  → 向分析与变换提供可使用的操作
```

定义与变换由此相接：前者确定 IR 表达的含义，后者在遵守这些含义的前提下改变表示。编写 ODS 不会自动实现目标机器码执行；后续展开和 lowering 仍有各自工作。

## 当前章节与后续范围

| 内容 | 作用 | 状态 |
|---|---|---|
| [一个操作是怎样被定义出来的](./op_definition) | 用 lab.clamp 连接语义、ODS、生成 API、builder、verifier、注册、打印与 Pass 展开 | 正文和配套工程已验证 |
| Trait / Interface | 描述共用约束及可查询能力，让通用变换理解不同操作 | 待编写；当前操作章使用 Pure 并说明语义依据 |
| 自定义 Type / Attribute | 为领域信息建立类型和静态数据表示 | 后续按需展开 |
| 较复杂的操作结构 | Region、可变数量输入/结果，以及更复杂的解析和验证 | 后续按需展开 |

第一章先走通一个操作，不把表中所有范围同时作为前置。Trait / Interface 将沿同一操作解释通用消费者怎样利用语义信息；之后再进入 Dialect Conversion，讨论目标合法性与类型变化的边界。

配套工程为 `aicompiler-labs/llvm-mlir/05-op-definition/`，阅读后可以观察自定义/通用打印、builder、验证失败和 Pass 展开。通用注册与 Pass 测试的完整工程另见[小 Pass 教程](../../tutorials/first_pass)。

范围与缺口见[覆盖表 M02—M03](../../coverage)，阅读阶段见[学习路径](../../learning_path)。本目录按知识职责归档，不用目录顺序替代学习依赖。
