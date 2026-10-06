---
order: 3
title: Translation、JIT 与 AOT
updated: 2026-10-04
---

# Translation、JIT 与 AOT

LLVM Dialect 函数和 C 调用约定已经准备好，下一项任务是让 CPU 执行它。这里有两条常见路径：提前生成目标文件并链接进程序，或在进程运行时生成机器码再调用。

两条路径都需要解决相同的基本问题：目标指令怎样生成，外部函数地址怎样找到，输入结果怎样按 ABI 传递。差异主要在于这些工作何时完成、产物如何装载与复用。

## 1. 从 MLIR 到目标文件

对已经降成可翻译形式的输入，一条 AOT 路径是：

```text
LLVM Dialect
    ↓ mlir-translate：构造 LLVM IR
LLVM IR
    ↓ llc：目标代码生成
目标文件 .o
    ↓ 链接器：合并对象并解析符号
可执行程序 / 动态库
```

AOT 即 ahead-of-time，强调机器码在调用前提前生成。目标文件包含机器码和必要的符号、重定位信息，仍可能引用尚未确定地址的函数，例如 malloc 或某个数学库函数。

链接器把这些引用与定义连接起来。生成动态库时，一部分符号解析也可能延后到装载或运行阶段；具体规则取决于平台和链接选项。

## 2. 直接观察一条 AOT 命令链

下面假设 `lowered.mlir` 已经完成必要的 LLVM lowering，`driver.c` 使用与生成函数匹配的 C 声明：

```bash
mlir-translate --mlir-to-llvmir lowered.mlir -o program.ll
llc -filetype=obj -relocation-model=pic -O0 program.ll -o program.o
clang driver.c program.o -o run
./run
```

这里每个工具只负责明确的一段：translate 不替你把任意 Tensor 操作展开；llc 不知道 Python 中张量的形状约定；clang 在这条命令中编译 C 驱动并完成链接，不负责重新读取 MLIR。

使用系统中较老的 clang 链接新 LLVM 生成的目标文件，与让它直接解析新 LLVM IR，是不同的兼容性问题。本文实验采用同版本 LLVM 的 translate 和 llc，系统 clang 只处理 C 与目标文件，并在本机验证这条组合。

`-O0` 让示例较容易观察，不表示没有任何折叠发生，更不表示最终性能。若要比较优化效果，需要固定目标、输入、优化配置与计时范围，后续验证指南再讨论。

## 3. 运行时函数与未解析符号

低层程序可能仍包含外部调用：

```text
llvm.func @malloc(i64) -> !llvm.ptr
```

这是一份声明，说明调用方式，不是 malloc 的实现。Translation 和目标代码生成都可以保留这类未解析引用，因为它们允许下一阶段提供实现。

链接可执行程序时，如果引用的函数没有任何定义，就会失败。同理，如果使用 MemRef 的某些通用复制路径，可能需要链接 `mlir_c_runner_utils` 等支持库。实际依赖应检查生成代码与未定义符号，而不是对所有程序无差别添加一长串库。

程序成功链接后，依旧可能因错误描述符、无效地址或数值语义变化得到错误结果。链接通过不是完整正确性证明，只是跨过了符号连接这一关。

## 4. JIT 的执行过程

JIT 即 just-in-time。它将后端编译和代码装载放到正在运行的进程中，常见过程为：

```text
准备可翻译的 MLIR 模块
    ↓
Translation 与 LLVM 编译
    ↓
在进程中为机器码分配并装载内存
    ↓
解析外部符号，查找入口地址
    ↓
按约定调用，获得结果
```

MLIR 的 ExecutionEngine 提供相关基础设施，`mlir-runner` 是可以直接观察这条路径的工具。它不会因为采用 JIT 就自动理解任意自定义高层操作；输入仍要满足工具所支持的转换与翻译条件。

用一个最小标量例子，将工具接入与矩阵 ABI 暂时分开：

<!-- cpu-example: jit -->
```text
module {
  func.func @main() -> i32 {
    %a = arith.constant 20 : i32
    %b = arith.constant 22 : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}
```

将其降低到 LLVM Dialect 后，执行：

```bash
mlir-runner jit-llvm.mlir -e main -entry-point-result=i32
```

工具实际打印 `42`。这表示查找并调用的 `main` 返回了 i32 值 42；它与 C 可执行程序将 main 返回值作为进程退出状态的用法不要混淆。

本例在转换中已经折叠为常量返回，所以它验证的是 JIT 接入与返回值通路，不是运行时加法性能。矩阵的真实数据读取和计算由同批 C 驱动验证。

## 5. 调用包装与符号解析

ExecutionEngine 的通用调用能力可以使用参数打包包装，将参数地址组成列表交给统一入口；这是引擎调用 API 的一层适配。前一章的 `_mlir_ciface_` 则是 Func-to-LLVM 为 C 兼容描述符边界生成的 wrapper。二者的用途和命名层次不同，不应看到“wrapper”就认为属于同一种 ABI。

如果 JIT 程序需要外部库，必须让引擎能发现其符号，例如通过加载支持库或注册运行时地址。错误的函数签名不会因为符号查找成功而被自动修正；解析出地址以后，仍要按一致的参数与返回约定调用。

JIT 引擎与生成代码本身也有生命周期。保存下来的函数地址通常依赖引擎所管理的代码内存继续存在；销毁引擎之后再调用旧地址，不能作为合法的复用方式。

## 6. 按阶段定位失败

| 失败位置 | 应首先查看的对象 | 本系列的具体例子 |
|---|---|---|
| Parser / verifier | 当前操作结构、类型和引用 | load 类型或 Block 参数不一致 |
| Lowering | 尚未完成的转换、目标合法性 | 仍存在不支持的高层操作 |
| Translation | 剩余操作及翻译接口 | LLVM 函数中仍有 arith 操作 |
| 链接 / JIT 符号解析 | 未定义符号与实际库 | 声明了外部函数却未提供实现 |
| 执行结果 | ABI、地址、初始化、算法语义 | offset/stride 错误或归约初值错误 |

这张表的作用是确定下一条证据从哪里取得。例如链接失败时，重新审查 Linalg indexing map 通常不是最快入口；数值错但 IR 合法时，继续只运行 verifier 也不能回答问题。

接下来在[Linalg 到 CPU 的最小执行链](../../tutorials/pipelines/cpu)里，把矩阵计算、描述符、AOT 链接和结果核对放在同一个完整案例中。

## 阅读自查

1. Translation 成功后，为什么仍可能在链接时找不到函数？
2. AOT 和 JIT 是否都需要遵守同样明确的调用约定？
3. `mlir-runner` 打印 42 证明了什么？为什么不能由此评价加法性能？
4. JIT 入口地址为什么不能脱离管理其代码内存的引擎生命周期使用？

## 资料与实践

LLVM 20.1.8 下 AOT 与最小 JIT 路径均已实际执行。源码入口见[ExecutionEngine.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/ExecutionEngine/ExecutionEngine.cpp)、[JitRunner.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/ExecutionEngine/JitRunner.cpp)与[mlir-runner 工具](https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8/mlir/tools/mlir-runner)。

可选实践：[直接命令、AOT 数值与 JIT 观察](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/11-cpu-lowering)。
