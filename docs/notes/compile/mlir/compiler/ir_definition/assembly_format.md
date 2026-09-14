---
order: 5
title: 解析与打印：在文本和 IR 对象之间往返
updated: 2026-09-14
---

# 解析与打印：在文本和 IR 对象之间往返

前几章已经规定操作的字段与合法形式。本章只追踪一件事：一段友好语法怎样填入 OperationState，经过验证成为 IR，再怎样打印回来。为了把注意力放在字段对应上，例子使用输入输出均为 i32 的 identity 操作，语义是原样返回输入。

## 1. 先确定文本中哪些东西要变成字段

<!-- irdef-example: assembly-input -->
```text
module {
  func.func @same(%x: i32) -> i32 {
    %r = lesson.identity same %x {tag = "demo"} : i32
    return %r : i32
  }
}
```

`same` 是固定关键字；`%x` 指向一个 SSA Value；`tag` 是不参与计算的额外属性；`i32` 同时说明输入与结果类型。构造出的对象需要保存引用、属性和类型，而不必保存 `same` 这个词。

```text
lesson.identity same %x {tag = "demo"} : i32
                  │         │            │
                  ↓         ↓            ↓
               operand    attribute   operand/result type
```

若只考虑这种简单语法，ODS `assemblyFormat` 就足以描述它。已有 yield 的可选参数组也采用这种方式。手写 parser/printer 适合声明式格式难以处理的语法；这里用同一个简单操作展示底层过程，方便逐项核对。

## 2. 从声明式格式走到手写入口

工程在 ODS 中保留 I32 输入/结果约束，并请求手写格式入口：

<!-- source-example: manual-op -->
```text
def IdentityOp : Op<Lesson_Dialect, "identity", [Pure]> {
 let arguments = (ins I32:$input);
 let results = (outs I32:$result);
 let hasCustomAssemblyFormat = 1;
}
```

这会生成 parse/print 声明。我们要实现的工作与生成 parser 一样：消费约定的文本，向操作状态填入对应字段。下面是完整的两项方法：

<!-- source-example: manual-format -->
```cpp
ParseResult IdentityOp::parse(OpAsmParser &parser, OperationState &result) {
 OpAsmParser::UnresolvedOperand input;
 Type type;
 if (parser.parseKeyword("same") || parser.parseOperand(input) ||
     parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type) ||
     parser.resolveOperand(input, type, result.operands))
   return failure();
 result.addTypes(type);
 return success();
}
void IdentityOp::print(OpAsmPrinter &printer) {
 printer << " same " << getInput();
 printer.printOptionalAttrDict((*this)->getAttrs());
 printer << " : " << getInput().getType();
}
```

## 3. 沿 parser 追踪 OperationState 的变化

按主例逐步走一次：

| 动作 | 此时获得的信息 | 对状态的影响 |
|---|---|---|
| 读取 `same` | 确认关键字符合语法 | 不增加 IR 字段 |
| 读取 `%x` | 获得尚待解析的 operand 引用 | 暂存在 UnresolvedOperand 中 |
| 读取属性字典 | `tag = "demo"` | 保存到 result.attributes |
| 读取冒号后的类型 | i32 | 暂存在 type 中 |
| resolveOperand | 按引用及类型找到函数参数 Value | 将它加入 result.operands |
| addTypes | 指定一个 i32 结果 | 将类型加入结果类型列表 |

UnresolvedOperand 还不是可以参与计算的 Value。先读取引用，再取得类型，最后进行解析，才能把文本中的名字与已有 SSA 定义连接起来。输入引用和新操作的结果也不能混同：前者已经存在，后者由操作构造产生。

如果把 `same` 改成其他词，第一步就会失败。如果把函数及 identity 的 i32 都改成 i64，parser 可以读完并填入状态，但随后 ODS 的 I32 约束会拒绝该操作。语法识别与对象验证因而是先后发生、职责不同的过程。

## 4. 沿 printer 追踪字段怎样写回文本

printer 先写固定关键字，再打印 operand 引用、额外属性字典和类型。它使用的是当前 IR 对象，所以参数经过重命名后，打印结果可能使用 `%arg0`，无需恢复最初的 `%x` 拼写。

通用格式进一步省去专用关键字，直接暴露同一操作的字段：

```text
%0 = "lesson.identity"(%arg0) {tag = "demo"} : (i32) -> i32
```

读取带引号的通用格式时，框架按通用语法填字段，不调用本例识别 `same` 的 parser；正常打印时仍可调用自定义 printer，重新输出友好格式。两条解析路径最终构造的是同一种操作，接受相同的操作验证。

## 5. 用一次往返检查是否遗漏信息

现在可以给 parser/printer 一个明确的检查目标：正常解析并打印主例；将同一对象通用打印；再解析通用文本并正常打印。最后两次正常输出应保留相同的对象关系：

- identity operand 仍引用函数参数。
- identity 结果仍交给 return。
- 输入、结果类型及 `tag` 属性保持一致。

如果 printer 漏掉属性字典，输出仍可能是语法合法的 IR，但 tag 会在下一次解析后丢失。只有检查字段与引用，才能发现这个问题；单纯确认工具退出成功不够。工作区验证会比较规范化输出，并核对主例中的属性和引用关系。

## 阅读后的一个小推演

在输入中增加另一个额外属性，预测通用格式和正常格式会把它放在哪里。再只把固定关键字写错，和把整段程序类型改成 i64 对照：两次失败分别发生在哪一步？

至此，对象定义、合法性检查和文本往返已经衔接起来。下一模块 **Dialect Conversion** 处理另一种变化：计算含义要保留，但操作与类型要换成目标支持的表示。

## 配套观察与依据

运行 `06-ir-definition/observe.py --section assembly` 可观察上述往返；`docs/validate_ir_definition.py` 核对正文源码、属性/引用保留及语法、ODS 验证两类错误。示例只实现这项简单格式，未覆盖复杂语法和错误恢复，也不执行目标机器码。

固定版本 LLVM `llvmorg-20.1.8`：

- [Operations.md](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)：声明式格式、custom parser/printer 与验证。
- [OpImplementation.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/OpImplementation.h)：解析及打印接口。
- 配套 `Lesson.cpp` 的 `IdentityOp::parse/print`：本例实现。
