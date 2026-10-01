---
order: 5
title: 解析与打印：文本和 IR 对象的对应
updated: 2026-10-01
---

# 解析与打印：文本和 IR 对象的对应

前面已经定义了操作的字段与约束。无论它表示标量计算还是区域，进入编译器后都需要成为可访问的 IR 对象；打印时又要将这些对象写成可读文本。

本章沿一个简单的 identity 操作追踪这个往返。它按语义原样返回输入，因此不需要再考虑范围计算。要理解的过程是：文本提供哪些信息，parser 怎样把信息填入构造状态，printer 怎样从当前对象恢复一份等价表示。

## 1. 文本语法与对象字段

先看完整输入：

<!-- irdef-example: assembly-input -->
```text
module {
  func.func @same(%x: i32) -> i32 {
    %r = lesson.identity same %x {tag = "demo"} : i32
    return %r : i32
  }
}
```

这条操作的语法包含一个固定关键字 same、一个输入引用、一个额外属性，以及类型。可以把文本与对象的对应写成：

```text
lesson.identity same %x {tag = "demo"} : i32
                  │   │       │           │
             固定语法 输入引用 静态属性    类型信息
                       ↓       ↓           ↓
                   operand  属性字典    输入/结果类型
```

same 只用于识别这项简洁写法，不需要在对象中再保存一个同名字段。`%x` 也不是对象里长期保存的字符串：解析后它对应函数参数的 Value，operand 引用这个 Value。

结果 `%r` 由新操作定义，后面的 return 使用它。解析器不会为了得到这个结果去执行一次 identity；这里建立的是表示与引用关系。

## 2. 声明式格式与手写入口

这类简单语法通常可以用 ODS `assemblyFormat` 表达。例如对于一个 I32 输入和一个 I32 结果，规则可以写为：

<!-- irdef-format: declarative-identity -->
```text
let assemblyFormat = "`same` $input attr-dict `:` type($input)";
```

生成器据此生成解析和打印方法，固定结果类型可由 I32 约束确定。实际开发中，声明式格式能清楚表达语法时，通常优先使用它。

为了看清底层工作，本教学实现选择手写同一种简单语法。操作仍声明输入和结果约束，但请求由作者提供格式方法：

<!-- source-example: manual-op -->
```text
def IdentityOp : Op<Lesson_Dialect, "identity", [Pure]> {
 let arguments = (ins I32:$input);
 let results = (outs I32:$result);
 let hasCustomAssemblyFormat = 1;
}
```

`hasCustomAssemblyFormat` 生成 parse/print 声明；作者负责实现它们。本章用手写方式展示过程，并不表示 identity 本身需要如此复杂的扩展。

无论代码来自生成器还是作者，任务相同：读入约定的语法，把语义需要的字段填入待构造状态。

## 3. Parser 的状态构造过程

输入中最需要解释的是 `%x`。刚读到这几个字符时，parser 得到的是一个尚待解析的引用；它还要结合所在作用域和所需类型，找到对应的 Value。找到后才能将其加入操作的 operand 列表。

沿第一节输入逐步处理：

| 动作 | 取得的信息 | 对待构造状态的影响 |
|---|---|---|
| 读取 same | 关键字符合语法 | 不增加字段 |
| 读取 `%x` | 尚待解析的输入引用 | 暂存引用信息 |
| 读取属性字典 | `tag = "demo"` | 保存到属性列表 |
| 读取冒号与类型 | i32 | 暂存类型 |
| 解析输入引用 | 找到函数参数 Value，确认类型对应 | 加入 operand 列表 |
| 记录结果类型 | 一个 i32 结果 | 加入结果类型列表 |

这份待构造状态在 C++ 中叫 `OperationState`。它尚不是一个已经插入 IR 的操作，也不是操作的 SSA 结果。解析方法准备好这些字段后，框架再据此创建 Operation 和它的结果。

反方向打印时，顺序类似：写出关键字，从对象中取得输入引用，再写出额外属性和类型。因为读的是当前对象，printer 无需记住原来 `%x` 的拼写，使用 `%arg0` 也能保持相同引用关系。

## 4. C++ 解析与打印实现

下面两项方法实现刚才的过程：

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

parse 中的局部变量 `input` 保存尚未解析的引用，`type` 保存读到的类型，参数 `result` 是上节的 OperationState。

那一串 `||` 按从左到右的顺序调用读取方法，任何一步失败就短路并返回 failure。例如关键字不是 same，就不再把后面的内容当成有效输入继续构造。`resolveOperand` 在读到类型之后执行，将文本引用连接到已有 Value；最后 `addTypes` 记录本操作新结果的类型。

print 中的 `getInput()` 取得对象已经保存的 operand。`printOptionalAttrDict` 保留额外属性，冒号后的类型则从输入 Value 取得。对于合法 identity，输入和结果都受 I32 约束，因此这份文本足以恢复本例的结果类型。

代码可以压缩成十几行，是因为状态已经由框架组织好了；理解时仍应按上一节的动作分开追踪，而不是只记住几个 parse 方法的名字。

## 5. 语法错误与验证错误

两种修改会在不同位置失败：

| 修改 | 发生什么 | 拒绝位置 |
|---|---|---|
| 将 same 改成 different | 不符合本操作的简洁语法 | parser 读取关键字时 |
| 将函数参数、结果与操作中的 i32 全部改成 i64 | 引用和类型能被读取，但操作不允许 i64 | 后续 ODS 字段验证 |

第二种修改中特意让各处类型一致，是为了排除“引用解析时类型不对应”的干扰。这样能够直接观察语法读取成功与操作合法性之间的区别。

Parser 也可以检查自己读取的语法要求，但结构与语义约束仍需要 verifier。原因是操作还可能由 C++ builder 创建，也可能由通用语法读入；这些入口不会执行同一段手写简洁语法 parser。

## 6. 通用格式与信息往返

在已构建工具的目录中，将第一节输入保存为 `assembly.mlir`，运行：

```bash
./lesson-opt assembly.mlir --mlir-print-op-generic
```

其中操作行变为：

```text
%0 = "lesson.identity"(%arg0) {tag = "demo"} : (i32) -> i32
```

通用语法直接写操作名、operand、属性和输入输出类型，不需要 same 关键字。读取这种写法时，框架使用通用 parser，不进入本例读取 same 的方法；形成对象之后仍接受 identity 的操作约束检查。

因此，两条路径的关系为：

```text
简洁文本 → 专用 parser ──┐
                        ├→ 同一种 Operation → 同一套操作验证
通用文本 → 通用 parser ──┘
```

可以将通用输出再次输入工具：

```bash
./lesson-opt assembly.mlir --mlir-print-op-generic > generic.mlir
./lesson-opt generic.mlir
```

正常打印会重新得到带 same 的格式，参数名可能规范化。判断往返是否正确，应看输入引用、结果使用、类型和 tag 是否保留，而不要求临时名字完全恢复。

例如若 printer 忘记输出属性字典，生成的文本仍可能是合法程序，但再次解析后 tag 已丢失。只检查工具退出成功看不出这项错误；必须检查需要保留的信息。

这个原则也适用于更复杂的操作：声明式或手写格式都要能恢复语义需要的字段，额外属性也不能无意丢弃。为了文本简洁而省略的字段，必须能够从剩余信息可靠恢复。

## 理解检查

在属性字典中再增加一个字符串属性，预测简洁与通用打印会如何保留它。随后考虑将 printer 改为完全不打印属性：第一次打印是否一定报错？第二次解析后对象与原对象有什么不同？

本章补齐了定义模块中的文本入口与出口。至此，字段、合法性、语义查询、类型/属性对象和区域协议都能够放回一套 IR 对象模型；后续转换模块处理的是如何在保持计算含义的前提下更换表示。

## 实现与依据

本例的手写实现、属性保留、引用关系以及两类失败由 LLVM `llvmorg-20.1.8` 下的[IR 定义工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/06-ir-definition)验证。它展示编译器侧的往返，不执行 identity 的目标代码，也未覆盖复杂语法和错误恢复。

固定版本 [ODS 文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/Operations.md)用于确认声明式和手写格式；[OpImplementation.h](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/OpImplementation.h)用于查阅解析与打印入口。
