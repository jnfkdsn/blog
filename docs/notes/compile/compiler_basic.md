---
order: 1
title: compiler base
status: draft
---

## 1.
一个经典编译器可以粗略分为：

```text
前端：lexer / parser / AST / semantic analysis
中端：IR / CFG / SSA / analysis / optimization
后端：lowering / instruction selection / register allocation / codegen
runtime：memory / call convention / execution support
```

### lexer(词法分析器)和 token
将源代码转换为一系列的token，token是编程语言的基本元素，如关键字、标识符、运算符等。
比如：

```c
x = 1 + 2 * y;
```

Lexer 看到的是字符：

```text
'x' ' ' '=' ' ' '1' ' ' '+' ' ' '2' ' ' '*' ' ' 'y' ';'
```

Token 流是：

```text
IDENT(x) ASSIGN INT(1) PLUS INT(2) STAR IDENT(y) SEMI
```

token 不只是字符串，还可以带类别，行列位置等信息。

### parser(语法分析器) 和 AST(抽象语法树)
丢掉很多语法细节，如括号分号不作为核心节点保留，只保留后续程序需要的信息：
```text
函数
参数
代码块
变量声明
return 语句
赋值
二元表达式
函数调用
整数常量
变量引用
```
构建一个树形结构来表示程序的语法结构。


例如：
```text
IDENT(x) ASSIGN INT(1) PLUS INT(2) STAR IDENT(y) SEMI
```
Parser 再根据语法优先级得到 AST：

```text
Assign
  target: Name(x)
  value:
    Binary(+)
      Int(1)
      Binary(*)
        Int(2)
        Name(y)
```

语法分析器parser：负责将token序列转换为AST
#### 表达式优先级
表达式优先级是指在编程语言中，不同的运算符具有不同的优先级，决定了在没有括号明确指定的情况下，运算的顺序。例如，在表达式 `1 + 2 * 3` 中，乘法 `*` 的优先级高于加法 `+`，parser需要按照语法规则创建ast节点

ai compiler中，parser的输出AST会被TorchDynamo捕获并转换为FX Graph，FX Graph是一种中间表示，适合进行后续的优化和代码生成。
```text
Python bytecode / frame
  -> TorchDynamo capture
  -> FX Graph
```

### semantic analysis(语义分析)
语义分析器负责检查AST的语义正确性，确保程序符合语言的语义规则,语法正确不一定语义合理，比如：

```c
int main() {
  x = 1;
  return x;
}
```

语法正确，`x` 没有声明，语义不合法。
它主要检查：

```text
函数是否重复定义
变量是否重复定义
变量使用前是否声明
函数调用是否存在
函数调用参数数量是否匹配
return 是否符合函数返回类型
void 变量是否合法
```
语义分析仍属于前端。它把“语法正确的 AST”变成“语义正确的 typed AST”。typed AST 的价值是：后续 IR builder 不需要反复猜类型。每个表达式节点都可以带上类型

AI Compiler 里的语义分析不一定是 type checker，但类似问题一直存在：

- Tensor dtype 是否匹配。
- Tensor shape 是否可广播。
- 某个 op 是否支持 dynamic shape。
- 某个 op 是否有副作用。
- 某个 graph node 是否能被 lower 到目标后端。

比如 FX Graph 中每个 node 可以带 metadata：

```text
node.meta["tensor_meta"] = shape / dtype / stride
```

这和 typed AST 很像。没有这些 metadata，后续 fusion、layout rewrite、codegen 都会很困难。

### IR
AST 很接近源语言结构，适合做语义检查，但不太适合做优化。

比如 AST 会保留：

```text
IfStmt
WhileStmt
Binary
Call
Assign
```

这些结构对于人来说直观，但对于优化 pass 来说还不够规则。

IR 更像一种小型汇编：
三地址码
```text
x = copy #0
%t0 = call add(#1, #2)
x = copy %t0
return x
```
它的特点：

```text
每条指令更简单
中间结果用临时变量保存
控制流可以拆成 basic block
适合后续构建 CFG、SSA、优化 pass
```

#### SSA、Use-Def、Phi

SSA 是 Static Single Assignment：每个 SSA value 只被定义一次。

SSA 的好处是 use-def 关系非常清楚。每个 use 都能追到唯一 def。

分支合流时需要 phi：

```c
if (c) {
  x = 1;
} else {
  x = 2;
}
return x;
```

SSA：

```text
entry:
  br c, then0, else0

then0:
  x1 = const 1
  jump merge0

else0:
  x2 = const 2
  jump merge0

merge0:
  x3 = phi [then0: x1], [else0: x2]
  ret x3
```

phi 的含义不是运行时调用函数，而是根据控制流来源选择对应 value。
SSA 是现代编译器中端优化的核心表示之一。
它让很多分析变简单：
- constant propagation 更容易追踪定义。
- DCE 更容易判断结果有没有 use。
- CSE 更容易比较表达式。
- register allocation 前可以从 SSA 结构得到 live range。
但 SSA 对 memory 不那么简单。数组、指针、load/store 会引入 alias 问题。

### Dataflow Analysis

Dataflow analysis 用来回答“程序某个点上一定/可能知道什么信息”。

典型问题：

- Reaching definitions：哪些定义可能到达当前点？
- Liveness：某个变量之后是否还会被使用？
- Available expressions：某个表达式是否已经计算过并且仍然有效？
- Constant propagation：某个 value 是否一定是某个常量？

Dataflow analysis 通常由几个元素组成：

```text
方向：forward 或 backward
格：信息集合及其合并方式
transfer function：一个 block 如何改变信息
meet/join：多个前驱或后继的信息如何合并
fixed point：不断迭代直到信息不再变化
```

例如 liveness 是 backward analysis：

```text
live_in[B] = use[B] union (live_out[B] - def[B])
live_out[B] = union live_in[S] for S in succ[B]
```

它从程序后面往前传播，因为一个变量是否活跃取决于未来是否会用。


Dataflow analysis 是优化 pass 的基础，很多 pass 需要根据分析结果改写。

比如：

- DCE 需要知道某个定义结果有没有后续 use。
- LICM 需要知道循环中哪些定义不变。
- Register allocation 需要 liveness。
- Constant propagation 需要在 CFG 上传播常量状态。

### Optimization Pass Pipeline



### Interpreter(解释器)
解释器直接执行AST或IR，不生成机器码，适合快速开发和调试

### 不同阶段
| 阶段 | 输入 | 输出 | 主要问题 |
| --- | --- | --- | --- |
| Lexer | 源码字符串 | token 列表 | 每个字符片段属于什么类别 |
| Parser | token 列表 | AST | 这些 token 组成什么语法结构 |
| Sema | AST | TypedProgram | 变量、函数、类型是否合法 |
| Lowering | TypedProgram | IRModule | 怎样把高层结构变成规则指令 |
| Interpreter | AST 或 IR | 执行结果 | 程序语义是什么 | 