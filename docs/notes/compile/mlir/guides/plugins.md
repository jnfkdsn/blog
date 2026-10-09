---
order: 80
title: 插件注册与动态方言的边界
updated: 2026-10-04
---

# 插件注册与动态方言的边界

[版本升级](./bytecode)的两个工具都把 my 方言编译进可执行文件。如果希望继续使用已有 mlir-opt，同时加载自己开发的方言和 Pass，就需要把能力交给宿主工具，而不是为每个组合重新维护一个 main。

本章先让同一份 `my.shift` 在标准工具中可读，再加载一个可观察的 Pass。之后区分原生插件与 IRDL 动态定义：两者都能扩展工具，但提供能力的方式不同。

## 1. 加载前缺少什么

标准 mlir-opt 读取：

```text
%y = my.shift %x {offset = 5 : i64} : i32
```

会报告没有找到 my 方言。文本中的前缀只能告诉 parser 应寻找哪个方言，不能自动创建操作定义。

将同一份 MyDialect/ShiftOp 的 C++ 实现编译成共享模块 MyPlugin.so，并向工具提供 Dialect 插件入口，就可以在解析前填入 Registry：

```cpp
extern "C" DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MyDialect", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<my::MyDialect>();
          }};
}
```

加载器寻找约定的 C 符号，读取版本与回调，调用回调完成注册。它不是通过扫描所有 C++ 类名推断该注册什么。

## 2. 从注册到解析与验证

加载选项为：

```text
mlir-opt input.mlir --load-dialect-plugin=MyPlugin.so
```

实际工具随后能打印 my.shift，并运行它的 verifier。读取上一章的旧 Bytecode 时，还会调用同一方言附带的版本接口，将 bias 升级为 offset。

这条过程说明插件提供的是已有 C++ 方言实现：操作定义、接口和版本逻辑都来自共享模块。普通操作没有因为“动态加载”而绕开验证。

Registry 记录可加载能力，Context 在需要时实例化方言。加载共享库、注册方言和解析具体操作是三个相关但不同的动作。

## 3. Pass 插件注册另一类能力

本工程另有一个 Module Pass：运行后添加 `my.plugin_seen` 单元属性。它不改变目标计算，只让插件的实际执行可见。

Pass 插件入口的回调注册这个 Pass：

```cpp
extern "C" PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MyPass", LLVM_VERSION_STRING,
          []() { PassRegistration<Mark>(); }};
}
```

加载两个入口并运行 `builtin.module(my-plugin-mark)` 后，输出为：

```text
module attributes {my.plugin_seen} {
  func.func @answer() -> i32 {
    %c37_i32 = arith.constant 37 : i32
    %0 = my.shift %c37_i32 {offset = 5 : i64} : i32
    return %0 : i32
  }
}
```

注册使 pipeline parser 认识 Pass 名称；只有 pipeline 实际包含它，属性才会出现。Dialect 插件也可以主动注册 Pass，但本例将两条回调分开，便于观察能力从哪里进入宿主。

## 4. 动态库仍受构建契约约束

插件加载器检查 MLIR_PLUGIN_API_VERSION。作者构造了一个刻意使用错误入口版本的插件，用独立 loader 探针确认它被拒绝，并正确消费返回的 Error。

这个小版本号只约束插件入口结构，不等于全部 MLIR C++ ABI 的稳定版本。方言、Pass、LLVM 容器、RTTI/assertions 和链接配置仍需匹配宿主。上一章工具版本与本章插件都来自同一个固定 LLVM 构建。

当前 Linux 实验让共享模块的 MLIR 符号由导出符号的宿主提供，避免把另一套独立基础设施装进同一进程。其他平台和共享/静态库配置应采用对应的 LLVM CMake 集成方式。不能仅凭 `.so` 能生成就认定目标工具一定能加载。

## 5. IRDL 动态定义的不同路径

如果需求只是动态声明一个操作的结构与约束，可以使用 IRDL，把定义本身表示成 IR。例如：

```text
irdl.dialect @my {
  irdl.operation @add {
    %i = irdl.is i32
    irdl.operands(lhs: %i, rhs: %i)
    irdl.results(result: %i)
  }
}
```

这里 `%i` 是一个约束值，表示“必须是 i32”，不是被编译程序的整数。加载定义后，工具可以识别 generic 形式的 `"my.add"`，并检查两个输入和一个结果的类型。

这条路径无需为该结构先生成 C++ 类，但它不会仅凭名字 add 自动获得加法语义、Pure 效果、fold 或 lowering。要让优化器安全使用这些能力，仍需相应的定义、接口和消费者。`--allow-unregistered-dialect` 更只是允许未知结构进入，并不等同于加载了 IRDL 约束。

[领域扩展中的 IRDL](../dialects/extensions/irdl)会用实际动态加载及错误类型对照进一步说明。IRDL 与原生插件可以服务不同阶段：前者便于动态定义和约束原型，后者可提供完整 C++ 方言、Pass 及接口实现。

## 6. 选择扩展方式

若要部署一个包含自定义分析、lowering 和接口的成熟方言，可选择静态集成或匹配宿主构建的原生插件。若要探索操作 schema、按数据加载约束，IRDL 更直接。

选择时应先确定要交付的能力：仅能读取结构，还是还需要验证、优化事实、版本升级和执行链。把这些契约写清楚，比把所有方式都称为“动态方言”更有助于排查缺失环节。

## 依据与实践

固定入口：[DialectPlugin](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Tools/Plugins/DialectPlugin.h)、[PassPlugin](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Tools/Plugins/PassPlugin.h)、[官方 standalone 插件](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/examples/standalone/standalone-plugin)。

[24 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/24-bytecode-plugins)保留实际加载、标记输出、版本升级与错误入口版本检查。这里验证同版本本机 Linux 插件，不承诺跨版本或跨平台二进制兼容。
