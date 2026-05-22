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
