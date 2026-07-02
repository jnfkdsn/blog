---
order: 2
title: 前端：Lexer、Parser、AST、语义分析
updated: 2026-07-01
tags: [compiler, frontend, lexer, parser, ast, sema]
status: draft
---

# 前端：Lexer、Parser、AST、语义分析

相关入口：[传统编译器](/notes/compile/traditional/) / [编译器基础](/notes/compile/traditional/compiler_basic)

前端把源码字符串变成后续阶段能处理的结构化程序。它的输出通常不是最终可优化的 IR，而是带语义信息的 AST 或 typed AST。

```text
source text
  -> lexer: token stream
  -> parser: AST
  -> semantic analysis: typed AST / symbol-resolved AST
```

## Lexer

Lexer 负责把字符流切成 token。token 不只是字符串，还包含类别、文本、位置，有些还会带解析后的值。

```c
x = 1 + 2 * y;
```

token 流：

```text
IDENT(x) ASSIGN INT(1) PLUS INT(2) STAR IDENT(y) SEMI
```

一个 token 通常包含：

```text
kind: IDENT / INT / PLUS / IF / WHILE ...
text: 原始文本
value: 整数、字符串等字面量解析结果
line/column: 报错位置
```

常见的 lexer 写法：

```text
while not eof:
  skip whitespace and comments
  if current is letter: scan identifier or keyword
  if current is digit: scan number
  if current starts operator: scan operator
  else: report unexpected character
```

关键点：

- 关键字通常先按 identifier 扫描，再查 keyword table。
- `==`、`<=`、`>=`、`!=` 这类多字符运算符要优先匹配。
- 注释和空白一般不进入 parser，除非语言语法依赖缩进。
- 位置记录用于后续 parser/sema 报错，不然后面只知道“错了”，不知道错在哪里。

## Parser

Parser 把 token 流变成 AST。AST 丢掉大多数标点细节，保留语义结构。

```text
IDENT(x) ASSIGN INT(1) PLUS INT(2) STAR IDENT(y) SEMI
```

对应 AST：

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

Parser 需要解决两个问题：

- 这些 token 能否组成合法语法。
- 表达式如何按优先级和结合性组合。

## 表达式优先级

表达式 `1 + 2 * 3` 不能解析成 `(1 + 2) * 3`，因为 `*` 优先级高于 `+`。

一种常见实现是 recursive descent 按优先级分层：优先级越低的运算放在越外层函数，优先级越高的运算放在越内层函数。

```text
parseExpr      -> parseAdd
parseAdd       -> parseMul (('+' | '-') parseMul)*
parseMul       -> parseUnary (('*' | '/') parseUnary)*
parseUnary     -> ('-' | '!') parseUnary | parsePrimary
parsePrimary   -> INT | IDENT | '(' parseExpr ')'
```

对于：

```text
1 + 2 * y
```

解析过程：

```text
parseAdd
  lhs = parseMul -> Int(1)
  sees '+'
  rhs = parseMul
    lhs = parseUnary -> Int(2)
    sees '*'
    rhs = parseUnary -> Name(y)
    return Binary(*, Int(2), Name(y))
  return Binary(+, Int(1), Binary(*, Int(2), Name(y)))
```

Pratt parser 也是表达式解析常用方法，特点是用 binding power 表示优先级：给每个运算符一个“绑定力” binding power，绑定力越强，越先结合。

```text
parse_expr(min_bp):
  lhs = parse_prefix()
  while current token is infix op with bp >= min_bp:
    op = consume()
    rhs = parse_expr(op.right_bp)
    lhs = Binary(op, lhs, rhs)
  return lhs
```

Pratt parser 对新增运算符比较友好；recursive descent 对语法结构更直观。基础阶段先掌握“低优先级函数调用高优先级函数”这个机制即可。

## AST 设计

AST 节点一般按语法结构拆分：

```text
Program
FunctionDecl
BlockStmt
VarDecl
IfStmt
WhileStmt
ReturnStmt
AssignStmt
BinaryExpr
UnaryExpr
CallExpr
NameExpr
IntLiteral
```

AST 示例：

```text
Function main() -> int
  Block
    VarDecl x: int = Int(1)
    VarDecl y: int = Binary(+, Name(x), Int(2))
    Return Name(y)
```

AST 设计需要注意：

- 源码位置要保留在节点上，方便报错。
- 节点不一定一开始就带类型，类型可以由 sema 阶段补上。
- AST 表示的是源语言结构，不适合直接做大量优化。
- parser 阶段尽量少做语义判断，避免 parser 和 sema 边界混乱。

## Symbol Table

语义分析需要知道每个名字指向哪个声明。Symbol table 负责维护作用域。

```c
int x = 1;
{
  int x = 2;
  return x;
}
```

作用域栈：

```text
global scope:
  x -> global variable

block scope:
  x -> local variable
```

查找规则通常是从内到外：

```text
lookup(name):
  for scope in reversed(scope_stack):
    if name in scope:
      return scope[name]
  report undefined symbol
```

常见检查：

- 同一作用域内变量是否重复定义。
- 变量是否在定义前使用。
- 函数调用是否存在。
- 函数调用参数数量和类型是否匹配。
- `break/continue` 是否位于循环内部。
- `return` 类型是否符合函数返回类型。

## Type Checking

Type checker 给表达式补类型，并检查操作是否合法。

```c
int x = 1;
bool b = x + true;
```

`x + true` 不合法，因为 `+` 期望两个数值类型。

表达式类型推导例子：

```text
IntLiteral(1): int
Name(x): int
Binary(+, int, int): int
Binary(>, int, int): bool
If(cond): cond must be bool
Return(expr): expr.type must match current_function.return_type
```

typed AST 可以长这样：

```text
Binary(+): int
  Name(x): int -> symbol local x
  Int(2): int
```

typed AST 的价值：后续 lowering 不再需要猜类型，也不需要反复解析名字绑定。

## 前端和 AI Compiler

AI Compiler 不一定有传统 lexer/parser。PyTorch 2.x 的入口更像：

```text
Python bytecode / frame
  -> TorchDynamo capture
  -> FX Graph
```

但语义分析仍然存在，只是检查对象变成 tensor program：

- Tensor dtype 是否匹配。
- Tensor shape 是否可广播。
- Tensor stride/layout 是否满足后端要求。
- 某个 op 是否支持 dynamic shape。
- 某个 op 是否有副作用。
- 某个 graph node 是否能 lower 到目标后端。

FX node 上的 metadata 类似 typed AST：

```text
node.meta["tensor_meta"] = shape / dtype / stride / memory_format
```

没有这些 metadata，fusion、layout rewrite、kernel selection、codegen 都会困难。
