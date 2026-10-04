---
order: 20
title: 定义 IR 抽象
updated: 2026-10-04
excludeFromSidebar: true
---

# 定义 IR 抽象

IR 变换使用已有操作的定义，读取输入、判断条件，再修改表示。本模块从定义者的角度解释这些能力的来源：怎样规定操作含义、设计字段、验证合法性，并把所需事实提供给通用代码。

本模块统一收录入门导读与各项定义机制。[my.add 导读](./dialect_basics)负责建立“定义 → 生成代码 → 注册 → 处理具体 IR”的整体认识；读过导读后直接进入下面的章节，无需再等待一套用 my.add 重讲全部机制的课程。

## 1. 内容组织与案例选择

用标量整数 clamp 作为主要计算：将输入限制到静态区间 `[-4,7]`。它能够自然引出属性、跨字段验证、范围查询和类型保证，比为了复用 my.add 而人为添加这些能力更合适。

```text
一项计算的含义
  → 操作字段与合法性约束
  → 可被构造、验证和变换的 IR
      ├─ 向通用消费者提供语义事实：Trait / Interface
      ├─ 组织静态参数与值的契约：Attribute / Type
      ├─ 包含内部程序并规定传值：Region
      └─ 与可读文本相互转换：parser / printer
```

这几项是操作设计的不同维度，不是每个真实操作都必须依次添加的功能。操作定义章已经用普通 Pattern 将 clamp 展开成 arith；不需要先完成自定义类型和区域，才能看到定义与变换接通。

## 2. 章节与阅读顺序

| 顺序 | 章节 | 本章完整解释的过程 |
|---|---|---|
| 入门，可按需跳过 | [自定义操作与 Dialect 接入](./dialect_basics) | 描述一种操作，生成 C++，接入工具，再处理一个具体实例 |
| 1 | [操作定义：语义、字段与验证](./op_definition) | 将 clamp 语义落实为输入和属性，验证约束，构造对象并展开成标准操作 |
| 2 | [通用语义：Trait 与 Interface](./traits_interfaces) | 操作提供范围或效果事实，消费者据此查询或作优化决定 |
| 3 | [自定义 Attribute 与 Type：参数与结果契约](./types_attributes) | 区间作为计算参数和结果保证，分别进入属性与类型，并接受分层检查 |
| 4 | [Region 操作：执行协议、传值与验证](./regions_assembly) | 外部输入绑定到内部参数，区域计算通过出口形成父结果，再检查边界 |
| 5 | [解析与打印：文本和 IR 对象的对应](./assembly_format) | 读取字段和引用，构造对象，再打印并检查信息是否保留 |

建议按表顺序读，具体任务也可以选择分支：Region 章先使用普通整数，不依赖自定义类型的存储实现；解析与打印章只需要理解操作字段和验证，可以在第一章后直接阅读。外部接口模型、uniquing 和复杂参数存储属于对应章节的深入部分。

### 接入通用消费者

完成基础定义、分析或 Bufferization 主线后，可回访两篇扩展：

- [结果类型推导与运行时形状](./type_shape_inference)：同一 add_one 的 Type 推导与尺寸查询怎样交给不同消费者。
- [区域、循环与调用接口](./region_interfaces)：为 scope 提供入口/出口映射，让 SCCP 推出 7，再连接标准循环与内联。

存储或字段复杂度增加后，再接续：

- [可变参数分组与操作属性存储](./complex_ods)：加权求和的多组/嵌套 variadic、修改分段与 Properties。
- [参数存储、类型身份与接口消费](./storage_interfaces)：数组所有权、Type/Attr Interface 的规划消费者，以及递归类型的受限可变存储。

## 3. 导读、机制章节与实验的分工

| 内容 | 主要职责 | 避免重复的方式 |
|---|---|---|
| my.add 导读 | 建立框架接入与运行的整体模型 | 后续章节简短回顾，不再次展开四份生成文件与注册外壳 |
| 本模块五章 | 深入解释操作设计及扩展机制 | 每章推进一个具体问题，保留关键代码与推演 |
| 可选 lab | 完整构建、观察、修改与测试 | 命令和源码可复用，但不决定教程叙述顺序 |

所有自定义例子统一属于 `my` 方言。不同操作承担不同的解释任务：

| 操作或类型 | 在主线中的用途 |
|---|---|
| `my.add` | 用熟悉的加法说明操作怎样接入框架 |
| `my.clamp` | 引入静态上下界、验证与实际改写，再增加范围接口 |
| `my.opaque_clamp` | 保留相同计算语义，刻意缺少效果声明；对照外部接口接入与 DCE |
| `my.limit`、`#my.bounds`、`!my.range` | 将区间分别表达为静态参数和结果类型保证 |
| `my.scope`、`my.yield` | 表达区域输入、内部计算与结果返回 |
| `my.identity` | 用简单计算集中解释手写解析与打印 |

`arith`、`func` 等标准方言仍按原名使用。统一教学方言后，每次扩展只需关注新增的表示或契约。

## 4. 当前实现与后续边界

本模块提供 Op 定义与验证、builder、Trait/Op Interface、自定义 Type/Attribute、Region 传值及解析打印的核心解释和编译器侧验证。其中 clamp 的普通 Pattern 展开已有实现。

以下内容仍需在相应任务中继续深入：

| 后续范围 | 当前与之相接的认识 |
|---|---|
| [Dialect Conversion](../conversion/)、TypeConverter、目标合法性 | 自定义类型无法只改一个结果而忽略其使用者和函数/区域边界 |
| [类型/形状推导](./type_shape_inference)（已有）；[Type/Attribute Interface](./storage_interfaces)（已有） | 生成 builder 与实际 dim 改写已有；字节数/对齐规划消费者、失败边界与存储身份已有 |
| [区域接口](./region_interfaces)（已有） | 自定义 scope 的 RegionBranch 与 SCCP；Function/Call/Loop/RegionKind 和内联边界 |
| [多组 variadic](./complex_ods)、[参数与可变存储](./storage_interfaces)（已有） | 分组修改、嵌套空组、Context 复制与名义递归身份已有作者实现 |
| 字节码、兼容与方言版本迁移 | 在序列化与工具主题展开 |

已理解本模块后，可以直接进入[操作转换与目标合法性](../conversion/operation_conversion)，沿已有 clamp 展开理解下一阶段的要求。Conversion 在并列模块讲解，不继续堆入定义机制。

具体知识范围与目标深度见[覆盖表](../../coverage)。学习者的阅读和实践进度另行记录，不由正文或作者工程通过情况推定。

## 实践入口

[最小接入工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/my-dialect)对应 my.add；[操作定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/05-op-definition)对应基础 clamp、builder 与展开；[IR 定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/06-ir-definition)对应其余机制。正文已经给出理解所需的代码和结果，实践用于检验预测及完成自己的修改。

这些工程保存 `my` 方言的不同教学阶段，分别构建，不同时链接多份定义。正文按知识依赖连续阅读，实验按当前章节选择；详细构建与工具路径见各工程 README。
