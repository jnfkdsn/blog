---
order: 20
title: 从失败 Pipeline 到最小复现
updated: 2026-10-04
---

# 从失败 Pipeline 到最小复现

一个包含许多函数的模块，在一条长 pipeline 中失败。第一步通常不是通读每个 Pass 的源码，而是建立可重复的事实：哪一阶段开始失败，当时输入是什么，删掉哪些内容后仍是同一问题。

本章用一个人为设置的教学限制贯穿全过程：`my-reject-marked-mul` 拒绝带 `my.trigger` 的 `arith.muli`。这不是上游编译器 bug，而是一个稳定、可控制的失败点，用来练习真实的诊断、重放和缩减工具。

## 1. 先确定失败属于哪一层

主例的关键函数为：

```text
func.func @hit(%x:i32, %y:i32) -> i32 {
  %z = arith.constant 0 : i32
  %dead = arith.addi %x, %z : i32
  %m = arith.muli %x, %y {my.trigger} : i32
  return %m : i32
}
```

模块里另外还有两个无关函数。这份 IR 能正常解析并通过 verifier；运行 canonicalize 后，死计算消失；再运行教学 Pass，得到“marked multiply is unsupported”。

这三个事实排除了“输入文本本身有错”，把问题定位到具体编译阶段的支持范围。与此不同，解析错误应先检查语法/注册；verifier 错误应追违反的结构或语义约束；转换失败要检查目标与缺失规则；执行结果错误则还要核对数值、内存和 ABI。

仅凭进程返回非零，无法区分这些原因。

## 2. 观察失败前后的 IR

对小输入，可以先打印每条 Pass 之前及失败后的 IR。这里的直接命令展示的是诊断选择，而不是依赖一键脚本才能理解：

```bash
my-opt input.mlir \
  --pass-pipeline='builtin.module(canonicalize,my-reject-marked-mul)' \
  --mlir-disable-threading \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-failure
```

输出表明 canonicalize 去掉了常量零和死加法，带标记的乘法仍在。教学 Pass 没有修改乘法，只是在看到它时报告不支持。失败后的 IR 不总是完整合法结果；有的 Pass 失败前已经做了一部分修改，因此更可靠的复现起点通常是失败 Pass 的输入。

长 pipeline 不宜长期打印全部 IR。先定位阶段，再用 `--mlir-print-ir-before=pass-name` 或 after 选取少数边界；`--mlir-print-ir-after-change` 可以减少重复输出。多线程会影响日志交错，模块级打印和本地 reproducer 也有相应要求，调试时可先固定单线程。

## 3. 保存输入与 Pipeline 配置

只保存一份 IR，仍可能丢掉触发问题所需的 Pass 顺序和选项。MLIR 的 pipeline reproducer 会把输入与配置保存在一起。

本例打开：

```text
--mlir-pass-pipeline-crash-reproducer=reproducer.mlir
--mlir-pass-pipeline-local-reproducer
```

虽然选项名含 crash，它也可为本例的 Pass 失败生成复现文件。local 模式在禁用多线程的条件下，保存失败 Pass 前的局部状态，而非要求重新运行前面的所有变换。

实际文件附带的资源配置包括：

```text
pipeline: "builtin.module(my-reject-marked-mul)"
disable_threading: true
verify_each: true
```

因此重放只需使用具备同一 Pass 注册的工具读取文件并加 `--run-reproducer`。作者实际重放得到了相同的教学诊断。reproducer 没有把自定义 Pass 的 C++ 实现打包进去；工具、版本和必要插件仍需匹配。

## 4. 缩减必须保留同一个问题

接下来删除两个无关函数和死计算。`mlir-reduce` 会尝试删除或简化内容，并反复询问一个判定脚本：新的输入是否仍然值得保留？

这个“interestingness”必须描述真正关心的现象。若脚本仅检查进程失败，缺少方言注册、语法错误甚至工具路径写错，都可能被误认为成功复现。

本例判断为：

```python
result = subprocess.run([tool, candidate, "--my-reject-marked-mul"],
                        capture_output=True, text=True, timeout=10)
interesting = (result.returncode == 1 and
    "teaching restriction: marked multiply is unsupported" in result.stderr)
```

固定 `mlir-reduce` 约定 **退出码 1 表示 interesting，0 表示不保留**。这与许多测试工具用零表示成功的习惯不同，应按具体 reducer 核对。脚本对超时、无法启动和无关诊断返回不保留。

本例还检查了两种反例：删除标记后不再 interesting；把文件改成无效语法也不再 interesting。这样才能确信缩减没有换成另一个错误。

## 5. 实际缩减结果与最小性边界

固定版本运行后得到：

```text
module {
  func.func @hit(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.muli %arg0, %arg1 {my.trigger} : i32
    return %0 : i32
  }
}
```

无关函数和死计算均消失，输入仍能通过 verifier，教学 Pass 仍拒绝同一条乘法。现在可以直接研究“标记乘法为何不被支持”，不用在三个函数里来回寻找。

这是选定 reducer 策略得到的更小复现，不是数学意义上全局最短输入的证明。复杂错误还可能依赖符号、属性、设备 target 或某条分析事实；缩减后应重新确认这些触发条件。

自定义语法或方言需要 reducer 认识相应定义。必要时构建注册了该 dialect 的 `mlir-reduce`，不要通过粗暴允许未知操作来假装保留了原来的验证契约。

## 6. Instrumentation、统计与时间

打印整个 IR 之外，PassInstrumentation 可以记录更紧凑的事件。本例观察器统计操作数并记录开始、成功和失败，实际顺序为：

```text
before canonicalize operations=12
after canonicalize
before my-reject-marked-mul operations=10
failed my-reject-marked-mul
```

这里的计数包含容器和终结操作，不能把 12→10 解释为执行指令数下降。失败回调与成功回调分开，有助于确定哪一边界没有正常完成。

`--mlir-pass-statistics` 同时记录检查了一条乘法；`--mlir-timing` 记录解析与编译阶段耗时。这些都是编译器运行的观察，不是目标 kernel 的执行时间。想比较优化收益，必须另外测量生成程序，并固定计时边界。

## 7. 工具崩溃与调试器入口

如果程序在读取 IR 前就崩溃，应先确认工具构建和 ABI。作者实现本实验时遇到一个真实问题：上游 LLVM 开启 assertions，而外部 Release 工程定义了 NDEBUG。`llvm::Statistic` 在两种配置下选择不同类型，导致包含统计字段的 Pass 布局不一致。

修复是让该外部工程匹配 `LLVM_ENABLE_ASSERTIONS`，随后统计、重放和缩减均通过。这说明“同一版本号”之外，RTTI、assertions、链接方式等配置也可能影响 C++ 集成；不能把此类崩溃归咎于输入乘法。

确实需要进入调试器时，优先对最小复现设置断点：自定义 `runOnOperation`、某条 matchAndRewrite，或最先报告失败的诊断位置。先观察操作名、operands/types 与当前插入点，再追 driver 内部。没有调试信息的优化构建可能无法逐行单步，应使用相应调试配置。这里给出使用路径，本机未进行 GDB 会话。

LSP 则服务于编辑阶段：定义跳转、引用、补全和诊断可缩短定位时间。作者用协议握手确认固定 `mlir-lsp-server` 可启动并提供相关能力，未把这项检查写成完整 IDE 集成验证。自定义方言的编辑支持仍需服务器具备对应注册。

## 依据与实践

固定来源：[Pass 打印与 reproducer](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/PassManagement.md)、[mlir-reduce 规范](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Tools/mlir-reduce.md)、[Statistic 配置](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/include/llvm/ADT/Statistic.h)。[在线 reducer 文档](https://mlir.llvm.org/docs/Tools/mlir-reduce/)可用于确认工具入口。

[22 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/22-debug-testing)提供原始输入、判定脚本、局部重放、Instrumentation、lit 测试与 LSP 握手。下一步把这份小复现变成[可长期维护的回归测试](./regression_tests)。
