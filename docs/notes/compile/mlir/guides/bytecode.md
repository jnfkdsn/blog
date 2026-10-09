---
order: 70
title: Bytecode 与方言版本升级
updated: 2026-10-04
---

# Bytecode 与方言版本升级

一个编译阶段输出 MLIR，另一个工具稍后读取它。保存文本便于查看，保存 Bytecode 则提供紧凑的结构化编码。但无论采用哪种容器，接收工具仍要理解操作的含义。

本章考虑一个具体变化：旧版 `my.shift` 用 bias 保存整数偏移，新版把字段改为 offset。计算始终是输入加偏移，按 i32 模运算；我们希望新工具能够读取旧文件，仍把 `37+5` 编译为 42。这个小例子区分三件容易混淆的事情：文件编码版本、方言定义版本和目标机器码。

## 1. 保存的是程序表示

旧版输入为：

```text
func.func @answer() -> i32 {
  %x = arith.constant 37 : i32
  %y = my.shift %x {bias = 5 : i64} : i32
  return %y : i32
}
```

`my-opt-v1 input.mlir --emit-bytecode -o old.mlirbc` 把这份 IR 编码成文件。它没有执行 shift，也没有生成 CPU 指令。Bytecode 重新读取后仍然形成 Operation、Region、Value 和 Attribute，之后才交给 Pass。

文件开头的 magic bytes 是 `4d 4c ef 52`。其后包含容器版本、producer 信息和不同 section，用来记录字符串、方言、类型/属性、操作与资源。相同字符串和类型可通过表中引用复用，无需像文本那样每次完整写出。[格式规范](https://mlir.llvm.org/docs/BytecodeFormat/)描述了这些编码关系。

这里无需逐字节实现一个 decoder；当前任务依赖的是“读取后恢复相同 IR 对象关系”。若工具不认识 my 方言，仅有 Bytecode 容器并不能补出它的 verifier、接口或 lowering。

## 2. 定义变化造成的断点

新版要求：

```text
%y = my.shift %x {offset = 5 : i64} : i32
```

如果只改新版 verifier，让它要求 offset，旧文件中的 bias 就不能直接通过检查。容器正确并不意味着其中的操作符合今天的定义。

因此两个工具采用不同方言版本：v1 写入版本1，只接受 bias；v2 写入版本2，只接受 offset。教学 ODS 暂时保留两个可选字段，让新版 decoder 能读出旧数据，再由升级逻辑将它转换为当前形式。两个字段并不同时合法，最终 verifier 仍严格要求当前版本的那一个。

本例把字段放在普通属性字典中，刻意不同时引入 Properties 编码变化。如果真实项目删除了旧属性/类型的编码器，旧数据可能连初步解码都无法完成，不能指望最后的升级钩子挽救所有不兼容。

## 3. 写入与读取 Dialect Version

方言通过 BytecodeDialectInterface 保存自身版本：

```cpp
void writeVersion(DialectBytecodeWriter &writer) const override {
  writer.writeVarInt(MY_DIALECT_VERSION);
}
```

对应 readVersion 读回整数，放入一个派生自 DialectVersion 的对象。这段整数的含义由 my 方言自己约定；它不是 MLIR 容器的版本，也不是 LLVM 的发行版本号。

读取器完成解析后，如果文件带有该方言版本信息，就调用 upgradeFromVersion。此时方言可以访问整份已解析的 IR，把旧构造转换到当前规则：

```cpp
if (oldVersion == 1) {
  top->walk([&](my::ShiftOp op) {
    auto bias = op.getBiasAttr();
    // 完整实现先拒绝缺 bias 或同时含 offset 的旧输入。
    op->setAttr("offset", bias);
    op->removeAttr("bias");
  });
}
```

重要的顺序是：保留足够兼容的解码能力 → 按记录的旧版本升级 → 用当前契约验证。这里不是猜测“有 bias 大概就是旧版”，而是依据文件中的明确版本。

## 4. 实际升级结果

v1 写入 old.mlirbc，v2 读取后打印：

```text
%0 = my.shift %c37_i32 {offset = 5 : i64} : i32
```

接着运行教学 lowering：取 offset，构造 i32 常量，再生成 arith.addi。canonicalize 得到：

```text
func.func @answer() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  return %c42_i32 : i32
}
```

作者同时让旧工具直接 lower 旧文件，得到相同输出。这检查了字段迁移没有丢失5，且当前消费者读取的是 offset。它是具体 IR 及常量折叠证据，没有把目标机器函数执行混入版本测试。

反过来，v1 读取 v2 文件会明确拒绝：`unsupported my dialect version 2 by reader 1`。旧工具没有新版含义时，拒绝比默默按旧规则解释更可靠。

## 5. 容器版本与方言版本分别控制

同一个 v2 工具可以用 `--emit-bytecode-version=0` 写旧容器格式。本例随后仍读到 offset，说明“容器为0”没有把 my 方言降为版本1。

若请求容器版本999，固定工具拒绝不支持的版本。能够向旧格式写出，还取决于当前 IR 特性是否能用那个格式表达；不能从本例成功推广到所有新特性。

要向真正的旧方言部署，通常还需要明确的降级/合法化策略：哪些新操作可以展开，哪些语义不能表示，以及失败时如何报告。自动保存为旧容器并不提供这些语义转换。[固定格式说明](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/BytecodeFormat.md)将容器兼容与方言演化分开讨论。

## 6. 文本、资源与持久化契约

把旧文本直接交给 v2，本例不会自动升级，因为普通文本没有本工程写入的 Bytecode 方言版本记录。它会因缺 offset 被当前 verifier 拒绝。若项目也要支持旧文本，需要另外约定文本升级工具或可识别的版本标记。

大型常量资源可以进入资源 section，并由相应资源管理协议恢复。资源的保存、外部引用以及是否允许省略内容都属于持久化契约；只保存省略过的调试打印，未必能重新构造原来的常量数据。

同样，Bytecode 不是一个包含所有编译器实现的独立可执行包。读取工具仍需方言注册、相关自定义编码与一致的外部约定。下一章会说明如何通过[插件与动态定义](./plugins)把这些能力接入现有工具。

## 依据与实践

固定实现入口：[BytecodeDialectInterface](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Bytecode/BytecodeImplementation.h)、[上游方言升级测试](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/test/Bytecode/versioning)。

[24 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/24-bytecode-plugins)构建同一教学方言的两个版本，实际验证读写、升级、拒绝未来版本、旧文本边界和容器版本。有限练习是增加一个负偏移输入，说明版本转换保持了哪个值及哪种整数语义。
