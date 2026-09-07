---
order: 1
title: 阅读、验证 IR 与定位源码
updated: 2026-09-06
---

# 阅读、验证 IR 与定位源码

前置：完成当前阶段的核心概念与贯通教程。本文说明如何把阅读中的判断变成可检查证据，以及如何按问题边界读源码。命令可以先理解；正式实践仍在阅读和讨论后安排。

## 1. 先区分工具检查了什么

| 检查层 | 能说明什么 | 不能据此证明什么 |
|---|---|---|
| parser | 文本能被已加载的语法解析，名字和结构可构建 | 所有 IR 不变量成立 |
| verifier | IR 满足通用结构、类型/支配及相关 Op 的验证约束 | 算法符合意图，所有动态前提成立 |
| generic print → parse | 当前内容能够通过通用文本重新构造并验证 | bytecode/跨版本兼容，算法数值正确 |
| transformation | 指定 pipeline 能处理输入并产生可验证输出 | 任意输入上的语义等价，完整后端已可运行 |
| 数值执行 | 选定输入上的运行结果满足判据 | 全输入正确，性能更优 |
| 性能实验 | 固定环境和方法下的测量 | 换硬件/版本/形状仍有相同收益 |

`mlir-opt` 正常流程包含解析和验证；不要看到它返回 0 就把结果叫“算子运行成功”。它可以处理含外部函数声明的模块，而当前系统可能根本没有相应实现可供链接。

## 2. 查看版本与两种打印格式

下面命令从 workspace 根目录执行，`example.mlir` 指某个已保存的完整模块。变量只为缩短路径：

```bash
MLIR_OPT="$PWD/artifacts/builds/mlir-20.1.8/bin/mlir-opt"
"$MLIR_OPT" --version
"$MLIR_OPT" example.mlir
"$MLIR_OPT" example.mlir --mlir-print-op-generic
"$MLIR_OPT" example.mlir --mlir-print-op-generic --mlir-print-debuginfo
```

普通输出通常使用已注册操作的自定义 printer；generic 输出显式显示 operands/results 类型、Properties 与 Region。看到函数定义末尾 `: () -> ()` 时，记得函数调用签名在 `function_type` 中。

Debug info 可以显示 Location，适合把变换后的操作追溯到输入位置。SSA 名字、Block 标签、属性顺序和部分省略形式可能改变；比较语义时跟踪对象关系，不只逐字比较打印文本。

未注册方言在特殊工具配置中可以按 opaque/通用形式处理，但这种模式缺少相应操作语义的正常验证。当前示例使用已注册方言，不通过 `--allow-unregistered-dialect` 掩盖拼写或版本错误。

## 3. 从一个 Pass 观察结构变化

直接转换 SCF 到 CF：

```bash
"$MLIR_OPT" example.mlir --convert-scf-to-cf
```

观察一个小 pipeline 中的中间 IR：

```bash
"$MLIR_OPT" example.mlir \
  --pass-pipeline='builtin.module(func.func(canonicalize,cse))' \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all
```

`builtin.module(func.func(...))` 表达 PassManager 的嵌套调度位置，不是 IR 文本本身的语法。某个 Pass 可以放在哪个层级，要看其定义和约束，不能把任意 Pass 都机械地塞进函数层。

IR dump 通常输出到 stderr，最终 IR 输出到 stdout；存日志时应分别保存。canonicalization 是基于 fold 和已注册模式的简化过程，不保证得到全局唯一或最优 IR。CSE 要遵守等价性、支配和效果约束；当前只观察结果，具体算法在编译器机制章节展开。

## 4. 阅读错误诊断

按顺序判断错误归属：

1. 文本语法/未注册操作：工具是否认识这种语法和方言，版本是否一致。
2. 本地类型/数量：operand/result 的类型与个数是否符合定义。
3. CFG 与 Region：Block 参数、successor、terminator、支配和作用域是否成立。
4. 符号与接口约束：callee 是否能解析，调用签名是否匹配，隔离是否被破坏。
5. 转换契约：输入是否满足该 Pass 的前提，转换后是否还留有非法操作或未处理类型。

例如 `does not dominate this use` 应回到所有进入使用点的路径，`operand type mismatch` 的函数调用错误还应核对 callee 签名。不要只改报错行的类型文本，因为 Value 定义和控制/符号关系可能在其他位置。

本系列的故意失败模块都标明预期错误类别，并由脚本检查诊断关键内容；不会把崩溃或任意非零返回当成“正确地拒绝了目标错误”。

## 5. 读源码应沿着一个可闭合的问题

假设要解释“for 的 yield 为什么必须与 iter_args 类型一致”：

| 层次 | 去哪里 | 要找的答案 |
|---|---|---|
| 操作契约 | `mlir/include/mlir/Dialect/SCF/IR/SCFOps.td` 的 ForOp | init、body 参数、yield、result 的对应 |
| 声明的约束/接口 | 同一定义的 traits、regions、arguments/results | 哪些条件已声明，哪些需 C++ 检查 |
| 验证实现 | `mlir/lib/Dialect/SCF/IR/SCF.cpp` 的 ForOp verifier | 数量/类型怎样检查，报错在哪产生 |
| 现有测试 | `mlir/test/Dialect/SCF/invalid.mlir` | 有哪些故意不合法的输入 |
| 转换使用 | `mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp` | 这条对应怎样被转为 header 和 branch 参数 |

工作区中可从这些精确符号开始搜索：

```bash
rg -n 'def ForOp|def YieldOp' upstream/llvm-project/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td
rg -n 'ForOp::verify|YieldOp::verify' upstream/llvm-project/mlir/lib/Dialect/SCF/IR/SCF.cpp
rg -n 'ForLowering|IfLowering|WhileLowering' upstream/llvm-project/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp
```

停止条件是能把本次问题的契约、实现入口和反例对应起来。遇到生成的 `.inc` 时，先回到 `.td` 和生成规则；只有理解具体 API 展开需要时才读生成代码。完整顺读 LLVM 仓库不是前置关卡。

## 6. 固定版本的使用方式

本系列固定 `llvmorg-20.1.8` / `87f0227cb60147a26a1eeb4fb06e3b505e9c7261`。文末源码链接指向这个 tag，本地源码在 `upstream/llvm-project/`，构建在 `artifacts/builds/mlir-20.1.8/`。

遇到新版官网与本地工具不同，应分别记录：网页说明的版本、操作定义的版本、实际二进制版本。不要只为让单个示例通过而升级全部源码/工具链，也不要把最新版网页上的可选语法不加核对地写入固定版本教材。

GPU/NPU 项目有自己的依赖组合。这里的基础工具可用于理解 MLIR 通用机制；不能由此推断目标项目能够直接替换为同一个 LLVM 版本。

## 文档示例的复现

已提供维护脚本 `aicompiler-labs/llvm-mlir/docs/validate_examples.py`。从 workspace 根目录运行：

```bash
python3 aicompiler-labs/llvm-mlir/docs/validate_examples.py
```

脚本直接提取正文中标记的完整模块，不复制另一套长期维护的源例子。合法模块检查解析/verifier 与通用打印往返；含 SCF 的模块额外转换为 CF、重新验证并检查没有剩余 SCF；失败模块检查对应诊断。

还做三项有明确预期的核对：零次 for 经 canonicalization 直接返回初值；同一 i8 位模式的 signed/unsigned 比较折叠成 true/false；贯通教程展示的 CF 代码与工具实际输出一致。它们是具体结构结果检查，不是一般等价性证明。

默认输出目录为 `artifacts/logs/mlir-docs/<日期>-reorganized/`，包含 `manifest.json`、每个例子的输入、通用输出、转换输出和诊断。manifest 记录工具版本、源页/行号、输入哈希与每项检查结果。用 `--output` 可另存某次复现。

2026-09-06 验证记录：36 个合法模块通过相应解析/往返检查，7 个错误模块被按预期拒绝；包含 SCF 的模块通过上述转换检查，三项指定结构核对也通过。对应记录位于 `artifacts/logs/mlir-docs/2026-09-06-reorganized/manifest.json`。

本次文档的验证没有执行机器码、设备算子或 benchmark，也没有宣称通过完整 LLVM 测试套件。博客另做 VitePress 构建和链接检查。这些工作属于教材质量检查，用户的正式实践仍按学习节奏安排。

## 继续学习的接口

当前能独立解释完整 IR 与错误边界后，下一阶段先学习已有 Pass 的 pipeline 与 IR dump，再实现小范围变换；lit/FileCheck、verify-diagnostics、单元测试和语义测试届时分别展开。详见[学习路径](../learning_path)和[编译器机制地图](../compiler/)。

依据：[mlir-opt 教程](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Tutorials/MlirOpt.md)、[Pass Management](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/PassManagement.md)、[Canonicalization](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Canonicalization.md)、[Diagnostics](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Diagnostics.md)。本地命令和示例结果以 manifest 为准。
