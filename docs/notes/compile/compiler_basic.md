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