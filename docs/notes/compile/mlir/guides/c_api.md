---
order: 50
title: C API 与工具中的 IR 所有权
updated: 2026-10-04
---

# C API 与工具中的 IR 所有权

前面的实验把 Pass 编译进一个 C++ 工具。若现在要把 MLIR 接入另一种语言的分析程序，就会遇到一个新边界：调用者未必能直接使用 C++ 模板和类，但仍需要解析模块、执行 pipeline、取得结果并修改 IR。

C API 为这些动作提供普通 C 函数与不透明句柄。本章用同一个小模块解释它怎样接入编译器，以及对象由谁保存、什么时候失效。任务是先把 `20+22` 折叠为 `42`，再把函数的返回值改为 `7`。这里操作的是编译器中的程序表示；调用生成的目标函数是[CPU ABI](../compiler/lowering/abi)讨论的另一项工作。

## 1. 从 C 程序进入编译过程

输入为：

```text
module {
  func.func @answer() -> i32 {
    %a = arith.constant 20 : i32
    %b = arith.constant 22 : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}
```

C 程序先建立 Context，注册 arith、func 及需要的 Pass，然后解析文本：

```c
MlirContext ctx = mlirContextCreate();
mlirDialectHandleRegisterDialect(mlirGetDialectHandle__arith__(), ctx);
mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), ctx);
mlirRegisterTransformsPasses();
MlirModule module = mlirModuleCreateParse(ctx, source);
```

这里 `source` 是 `MlirStringRef`，包含文本地址和长度。注册让 Context 在解析时能够找到方言定义；解析生成与 C++ 工具相同种类的 IR 对象。`MlirModule` 只是访问对象的句柄，复制这个小结构不会复制整份模块。

解析可能失败，调用者必须检查 `mlirModuleIsNull`。不能把空句柄继续传入一般的操作查询函数，指望每个函数都替自己处理失败。

## 2. 执行 Pipeline 并取得第一次结果

PassManager 持有要执行的编译步骤：

```c
MlirPassManager pm = mlirPassManagerCreate(ctx);
MlirOpPassManager opm = mlirPassManagerGetAsOpPassManager(pm);
MlirLogicalResult parsed = mlirParsePassPipeline(
    opm, pipeline, printChunk, stderr);
MlirLogicalResult ran = mlirPassManagerRunOnOp(
    pm, mlirModuleGetOperation(module));
```

`pipeline` 的内容是 `builtin.module(canonicalize)`。实际程序分别检查 parsed 和 ran；前者失败可能是 Pass 名称或语法错误，后者失败表示 pipeline 已建立但执行未成功。例中二者都成功，打印结果为：

```text
module {
  func.func @answer() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    return %c42_i32 : i32
  }
}
```

此时还没有运行 answer。编译器根据常量计算得出 42，并把这个事实写入 IR。

## 3. 拥有关系与借用句柄

接下来取得函数、入口 Block 和 return：

```c
MlirOperation function =
    mlirBlockGetFirstOperation(mlirModuleGetBody(module));
MlirBlock block =
    mlirRegionGetFirstBlock(mlirOperationGetRegion(function, 0));
MlirOperation ret = mlirBlockGetTerminator(block);
```

这段访问依赖主例只有一个已知结构的函数。在通用工具里，应先检查操作种类、Region/Block 数量和目标函数名称，不能对任意输入照搬固定位置。

三个句柄没有获得底层对象的所有权。模块仍然拥有函数，函数拥有 Region 和 Block，Block 拥有 return。调用者只借用了这些对象的访问入口：

```text
调用者拥有 module
  └─ module 拥有 function
       └─ Region / Block 拥有 ret

function、block、ret 变量只是借用句柄
```

因此，不应单独销毁 ret 再让父模块继续持有它。销毁 module 后，这些借用句柄全部失效。Type 和 Attribute 则由 Context 管理，Context 必须晚于依赖它的 IR 销毁。

## 4. 构造与插入一个新常量

为了让函数返回 7，需要先有一个定义在 return 之前的 i32 常量。C API 的通用构造方式把操作字段放入 OperationState：

```c
MlirType i32 = mlirIntegerTypeGet(ctx, 32);
MlirNamedAttribute value = mlirNamedAttributeGet(
    mlirIdentifierGet(ctx, str("value")), mlirIntegerAttrGet(i32, 7));
MlirOperationState state = mlirOperationStateGet(
    str("arith.constant"), mlirLocationUnknownGet(ctx));
mlirOperationStateAddResults(&state, 1, &i32);
mlirOperationStateAddAttributes(&state, 1, &value);
MlirOperation seven = mlirOperationCreate(&state);
```

这里的 `str` 表示把字符串转为 MlirStringRef 的小包装。字段仍对应熟悉的 IR：操作名是 arith.constant，一个 i32 结果，value 属性保存整数 7。创建返回的操作尚未属于任何 Block，调用者负责它的生命周期；实际程序也检查创建结果。

把它交给 Block 后，拥有关系改变：

```c
mlirBlockInsertOwnedOperationBefore(block, ret, seven);
mlirOperationSetOperand(ret, 0, mlirOperationGetResult(seven, 0));
```

名字中的 Owned 表示接收方取得所有权。此后 Block 会负责销毁 seven，调用者不能再次独立 destroy。第二行让 return 的 operand 从常量 42 的结果改为常量 7 的结果，它不会自动删除原常量。

现在的中间状态是：

```text
%c42 = arith.constant 42 : i32   // 已无使用者
%c7  = arith.constant 7 : i32
return %c7 : i32
```

验证成功后再次运行 canonicalize，未使用的 42 被删掉，最终只剩常量 7 与 return。C API 没有改变 SSA 和支配约束：若把 seven 放在 return 之后，仍会破坏合法使用关系。

## 5. 输出回调与清理顺序

C API 的打印回调可能被多次调用，每次收到一个字符串片段。片段不保证以零字符结束，因此示例使用长度：

```c
static void printChunk(MlirStringRef text, void *data) {
  fwrite(text.data, 1, text.length, (FILE *)data);
}
```

若要长期保存输出，应在回调中复制片段，而不是保存一个马上可能失效的指针。普通 `printf("%s", text.data)` 也不能替代带长度处理。

工作完成后，先销毁 PassManager 和模块，再销毁 Context。真实工具需要把错误分支也汇入清理路径，避免解析成功、pipeline 失败时漏掉释放。

运行任意变换后，先前借用的操作可能已被删除或替换。例如第一次 constant 42 在第二次 canonicalize 后就不再存在。应从仍然存活的拥有者重新定位需要的对象，不能因为 C 变量里还有一个地址就继续访问它。

## 6. 接口边界与适用范围

C API 适合作为语言绑定或嵌入工具的接入层，但它并不把每个 dialect 的语义都变成高级接口。通用操作 API 主要暴露 operands、results、attributes 等结构；某个字段是否匹配要求，仍由方言定义和 verifier 决定。

固定版本文档也明确说明 C API 尚无稳定性保证。它减少跨语言处理 C++ 的困难，不应据此假定任意 MLIR 版本的二进制都兼容。实际接入应固定头文件、库和工具版本，并保存构建配置。

在完成这个例子后，可以解释两项关键变化：创建 seven 时由调用者拥有；插入后由 Block 拥有。下一章会看到 [Python 绑定](./python_bindings)用包装对象和上下文管理器简化写法，但底层拥有关系仍然存在。

## 依据与实践

固定来源：[C API 设计与拥有约定](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/CAPI.md)、[IR C API 声明](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir-c/IR.h)、[Pass C API](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir-c/Pass.h)。正文按实际头文件核对回调签名。

[23 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/23-bindings)包含可编译的 C 程序、清理路径和两次真实输出。这里验证编译器对象操作，不包含目标函数调用或外部语言运行时的完整集成。
