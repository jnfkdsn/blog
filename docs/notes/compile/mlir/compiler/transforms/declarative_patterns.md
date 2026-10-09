---
order: 60
title: 声明式改写：DRR、PDL 与 PDLL
updated: 2026-10-04
---

# 声明式改写：DRR、PDL 与 PDLL

前面用 C++ Pattern 实现过 `x+0 → x`：匹配根操作，确认右侧为零，再替换所有结果使用。对于许多类似局部规则，手写代码反复出现相同的取 operand、检查和替换结构。声明式改写希望把规则的形状直接写出来，让工具生成或解释这些工作。

本章仍使用同一条规则，分别走通 DRR 和 PDLL 路径，比较它们交给 driver 的是什么。它们改变规则的表达方式，不会自动证明数学等价，也不会替代 Pass 的组织职责。

## 1. 先固定规则的计算含义

教学 `my.add` 只接受两个 i32 并返回 i32，采用整数加法语义，没有额外溢出标志，也没有副作用。它刻意没有 fold 或 canonicalization 实现，以便观察本章规则到底做了什么。

```text
%z = arith.constant 0 : i32
%a = my.add %x, %z : i32
%b = my.add %a, %z : i32
return %b : i32
```

应用规则后可以直接返回 x。相邻变体用于检验匹配条件：`my.add(0,x)` 和 `my.add(x,3)` 在本章都保留，因为只写了右侧为零的规则。

这项限定很有价值。若同一份输出把所有加法都删掉，就应检查是否有其他 fold、CSE 或规则参与，而不能只凭最终 IR 猜本规则生效。

## 2. DRR 用树形描述生成 C++ Pattern

DRR 基于 TableGen 的 Pattern 记录。源 DAG 描述匹配形状，目标 DAG 描述替代结果，额外 constraint 表达不能只靠结构确认的条件。

本例规则为：

```tablegen
def IsZero : Constraint<
  CPred<"::mlir::matchPattern($0, ::mlir::m_Zero())">>;
def RemoveRightZero : Pat<
  (AddOp $x, $zero),
  (replaceWithValue $x),
  [(IsZero $zero)]>;
```

AddOp 来自本工程的 ODS 定义。x 和 zero 绑定到两个 operand；变量名 zero 本身不证明它为零，真正判断来自 IsZero。

`mlir-tblgen -gen-rewriters` 生成 C++ `RemoveRightZero` Pattern。生成代码先读取两个 operand，调用 `matchPattern(...,m_Zero())`；失败时报告未匹配，成功时调用 rewriter.replaceOp，用 x 替换结果。

因此 DRR 到运行之间有一段明确的构建过程：

```text
TableGen 规则 → 生成 C++ Pattern → 编译进工具
                                      ↓
                               加入 PatternSet
                                      ↓
                                  driver 应用
```

普通的“操作定义生成”与“改写规则生成”是两个 TableGen 后端任务；定义了 AddOp，不会自动得到这条删除规则。

## 3. 生成 Pattern 仍遵循普通驱动协议

Pass 创建 RewritePatternSet，加入生成规则，再调用 greedy driver。生成的 Pattern 通过同一个 rewriter 修改 IR，所以通知、工作列表和失败约定仍然适用。

在链式例子里，一条加法被替换后，另一条的 operand 相应更新。driver 继续处理候选，最终两条都消失。规则没有自己遍历整个函数，也没有替我们定义一条完整优化 pipeline。

增加 benefit 可以改变某些候选间的优先关系，但不是全程序最优代价模型。两条规则互相展开回去仍可能不收敛；源码换成 TableGen 不会消除这些设计问题。

## 4. PDLL 把规则写成独立语言

同一个右零规则可以写为 PDLL：

```text
#include "MyOps.td"
Constraint IsZero(v: Value);
Pattern RemoveRightZero {
  let x: Value;
  let z: Value;
  IsZero(z);
  let root = op<my.add>(x, z);
  replace root with x;
}
```

x、z 是匹配到的 IR Value，root 是待改写的 Operation。`replace` 描述成功匹配后的变换，而不是在编写 PDLL 文件时执行目标程序的加法。

本例声明的 IsZero 是原生约束：它的签名让 PDLL 知道怎样调用，但实现需要工具注册。这样的接口允许把适合用语言表达的匹配结构与特定 C++ 判断结合起来。

## 5. PDL 是描述改写的 IR

`mlir-pdll -x=mlir` 把上述文件转成 PDL。省略位置标记后，关键结构为：

```text
pdl.pattern @RemoveRightZero : benefit(0) {
  %x = operand
  %z = operand
  apply_native_constraint "IsZero"(%z : !pdl.value)
  %types = types
  %root = operation "my.add"(%x, %z : !pdl.value, !pdl.value)
          -> (%types : !pdl.range<type>)
  rewrite %root {
    replace %root with(%x : !pdl.value)
  }
}
```

这里有两层 IR：被优化的程序中存在 my.add；描述规则的 PDL 中存在“匹配名为 my.add 的操作”。PDL 的 `%x` 是匹配过程中绑定的实体，不是目标程序中某个 i32 数字。

这使规则本身也能作为 IR 被处理。固定版本的 Pattern 基础设施可把 PDL 降到 PDLInterp，再编译成用于匹配/改写的 bytecode，由 driver 配合执行。这里的 bytecode 属于改写引擎，不能与“序列化 MLIR 模块的 bytecode 文件”混为一谈。

PDLL 还支持生成 C++ 封装等输出形式；本章选择 PDL 路径，便于直接观察中间的规则 IR。

## 6. 将原生约束接入实际消费者

工具读取 PDL 模块，构造 PDLPatternModule，并注册同名约束：

```cpp
pdl.registerConstraintFunction("IsZero",
    [](PatternRewriter &, Value v) {
      return success(matchPattern(v, m_Zero()));
    });
patterns.add(std::move(pdl));
```

随后仍使用 greedy driver。名字与签名必须匹配，单独解析成功的 PDL 文件并不保证运行工具具备所有原生实现。

DRR 中的 C++ constraint 编译进生成 Pattern；本路径中的原生约束按名字注册到 PDLPatternModule。两者在本例调用相同的零匹配逻辑，但构建和接入方式不同。

当规则需要复杂支配判断、跨 Region 更新、目标资源分析或很难声明表达的构造时，保留 C++ Pattern 通常更直接。选择声明式表达应让规则更清楚，而不是把大量 C++ 字符串隐藏进另一种语法。

## 7. 用相同输入比较两条路径

作者实际执行 DRR 和 PDLL 两个 Pass，得到完全相同的结果：right 函数直接返回 x；left 与 nonzero 函数各保留一条 my.add。

只运行 canonicalize 的对照中，四条 my.add 都保留。它们的结果仍被使用，而本操作没有自带 fold；这项对照帮助把变化归因于本章的规则。无法据此推断真实项目中的 canonicalize 总是没有类似能力。

两个声明式机制都没有扩大规则的数学适用范围。将 i32 改成浮点数后，零、NaN、signed zero 和 fast-math 约定需要重新分析；不能照抄整数规则。把替代值类型改错，同样不会因为生成代码来自工具就自动合法。

## 8. 在已有编译器中的位置

这几项能力可按输入和输出区分：

| 机制 | 输入 | 交给下一层的产物 |
|---|---|---|
| DRR | TableGen Pattern | 生成的 C++ Pattern |
| PDLL | 规则语言文本 | 本例生成 PDL IR |
| PDL / PDLInterp | 描述匹配与改写的 IR | 可供改写驱动使用的实现 |
| PatternRewriter / driver | 规则与待处理 IR | 合法修改及应用过程 |
| Pass / pipeline | 变换任务、作用域与顺序 | 某一编译阶段的输出 |

Dialect Conversion 还需要目标合法性、TypeConverter 与边界处理。声明式局部规则不能单独替代这些完成条件；结构简单的规则可以成为其中的一部分，具体接入能力要按版本和消费者核对。

可做的练习是新增左零规则，让 left 也直接返回 x，同时保持 nonzero 不变。先写出匹配条件，再选择 DRR 或 PDLL 完成，不需要同时掌握两种全部语法。

## 依据与实践

固定来源：[DRR 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DeclarativeRewrites.md)、[PDLL 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/PDLL.md)、[PDL bytecode 消费者示例](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/test/lib/Rewrite/TestPDLByteCode.cpp)、[改写引擎构建与实现](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/lib/Rewrite)。

[21 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/21-conversion-patterns)包含两种规则、生成物、原生约束、三类输入及实际对照输出。它与[1:N 转换](../conversion/one_to_many)共用一个小工具，但两项实验彼此独立，分别解释规则表达和类型/值边界。
