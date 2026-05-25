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

### lexer(词法分析器)
将源代码转换为一系列的token，token是编程语言的基本元素，如关键字、标识符、运算符等。
比如：

```c
return 1 + 2;
```

可以被切成：

```text
return
1
+
2
;
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

语法分析器parser：负责将token序列转换为AST
#### 表达式优先级
表达式优先级是指在编程语言中，不同的运算符具有不同的优先级，决定了在没有括号明确指定的情况下，运算的顺序。例如，在表达式 `1 + 2 * 3` 中，乘法 `*` 的优先级高于加法 `+`，parser需要按照语法规则创建ast节点
