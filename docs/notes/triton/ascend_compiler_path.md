---
order: 31
title: Triton-Ascend 的张量块与 NPU 编译路径
updated: 2026-10-04
---

# Triton-Ascend 的张量块与 NPU 编译路径

同一个 Triton 加法程序可以描述一块元素的地址、有效范围和加法计算，目标机器却可能采用不同的执行方式。[NVIDIA 路径](./compiler_path)把这些元素分给线程和寄存器；本章追踪 Triton-Ascend v3.2.1 怎样把计算交给 NPU 编译器。重点是同一块数据在两个编译器边界上保留什么信息，以及谁最终负责存储和同步。

## 1. 逻辑块与设备工作的边界

沿用长度 10、块大小 4 的加法。逻辑 program 依次处理 [0,4)、[4,8)、[8,12)，最后两项被 mask 排除：

```python
i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
valid = i < n
a = tl.load(x + i, valid, other=0.0)
b = tl.load(y + i, valid, other=0.0)
tl.store(out + i, a + b, valid)
```

程序约定的是有效元素及计算结果。`BLOCK=4` 本身没有说明用了四个 NPU 核、四条搬运指令，还是多大的 UB 空间。为了执行，编译器还要组织块的分工、地址和内存访问，再为目标安排计算、搬运与同步。

这一点连接了已有 MLIR 知识：方言可以保留逻辑计算，较后的转换再补充目标约束。相同的 Python 表达式不保证不同后端拥有相同的中间表示。

## 2. 后端实际注册的阶段

固定版本 `backend/compiler.py` 中，常规 NPU 路径注册的是：

```text
前端产生的 TTIR
  ↓ make_ttir
ttir 阶段产物
  ↓ ttir_to_linalg
ttadapter 阶段产物
  ↓ 所选架构的 linalg_to_bin 编译入口
npubin：设备对象，以及配套 metadata
```

`make_ttir` 优化前端输入的 TTIR，`ttir_to_linalg` 产生 ttadapter，最后的编译函数产生 npubin。这里没有 NVIDIA 后端的 `ttgir → llir → ptx → cubin` 阶段表。

还有一条明确的分支：设置 `force_simt_only` 后，TTIR 直接进入 `ttir_to_npubin`，不注册上述 ttadapter 阶段。因此不能把本章的常规路径当作所有模式的唯一流水线。架构选项 `compile_on_910_95` 也会选择另一套二进制编译入口；下面关注 A2/A3 分支。

## 3. 地址块转成可消费的计算

`ttir_to_linalg` 这个名字容易让人以为“一条 Pass 把所有 Triton 操作替换成 Linalg”。实际代码先做 auto blockify，按选项组织调度相关 Pass，再经过 structure、mask 处理、annotation、unstructure、HIVM/HFusion/LLVM 相关转换，最终加入 TritonToLinalg。

这些阶段围绕同一需求工作：从一块元素地址和 mask 中恢复适合目标消费的结构。例如最后一个 program 的四个逻辑位置，只有两个允许访问全局数组。若把它变成局部数据块，编译器必须同时处理两件事：

1. 只从有效的全局范围读取。
2. 让后续计算看到无效位置对应的 `other=0.0`。

固定实现的 `LoadStoreConverter.cpp` 包含不同访问路径。一个相关辅助函数在有效 mask 范围小于局部块尺寸时，用 `linalg.fill` 填入 other；另一个辅助函数把局部 MemRef 接成 Tensor，再替换原来的 load 结果。源码中的连接形式为：

```cpp
Value loadedTensor = rewriter.create<bufferization::ToTensorOp>(
    loc, tensorType, localMem, true, true);
rewriter.replaceOp(op, loadedTensor);
```

这段摘录省略了属性传播等处理。它展示的是某条转换路径中“内存中的块怎样成为后续张量计算的值”，不是声称所有 `tt.load` 都具有同一降低结果。离散索引、间接访问、block pointer、隐式转置等条件会选择不同处理。

现在可以解释为什么 ttadapter 未必是纯 Linalg：Tensor 表达值，MemRef 表达存储，Arith 表达标量运算，SCF 表达条件和循环，Annotation/HIVM 等携带项目约定。它们共同构成这个边界，不能排成必须逐个消失的方言名单。

## 4. 转换完成的判据

`TritonToLinalgPass.cpp` 注册函数接口的类型转换、各类操作规则，并调用 `applyPartialConversion`。前面学过的 ConversionTarget 和 adaptor 在此分别承担“哪些操作必须转换”和“规则读取哪个阶段的 operand”两项工作。

对于加法，不能只看到 `arith.addf` 仍在就判定转换失败。它可能已经是目标接受的计算。真正需要核对的是：Triton 的地址/访问及边界操作是否被处理，类型是否与消费者匹配，剩余项目操作是否有后续实现。

另一方面，partial conversion 成功也不等于已经有可执行 NPU 程序。它只满足这一阶段配置的合法性要求；片上分配、同步、目标代码生成和外部工具依赖仍属于后续职责。

## 5. 外部编译器接过哪些信息

A2/A3 编译入口先从 ttadapter 解析 kernel 名称、tensor kinds、并行方式等 metadata，然后写出临时 MLIR 文件，调用 `_get_npucompiler_path()` 返回的编译器。

选项会影响局部存储和调度，例如 multibuffer、UB 节省、预取、同步及链接 bitcode。若工具是 `bishengir-compile`，代码还加入 HFusion、HIVM 与 Triton kernel 编译选项。具体 HIVM 路径由工具能力检查选择 reg-based 或另一套启用选项，不能仅凭函数名中的 A2/A3 推定所有版本都使用完全相同内部路径。

对于加法，后续可能需要完成：把全局输入搬入计算可用的存储，安排向量计算，再把结果写回；多个任务重叠时补充依赖。其作用在 [NPU 表示与执行约束](/notes/compile/mlir/compiler/targets/npu_constraints)中已有逐步推演。本章关心的是交接点：Triton adapter 把语义和结构交给下游，设备编译器再消费这些信息。

## 6. 编译对象与运行时启动

编译结束还需要把二进制、kernel 名称和参数信息交给运行时。`pack_metadata` 会整理启动需要的数据；`driver.py` 根据签名生成 launcher，获取设备指针并打包参数，最终进入对应运行时调用。

固定源码包含 `rtKernelLaunch` 与带配置的 `rtKernelLaunchWithFlagV2` 分支。选择何者由配置和后端条件决定。它们不是 MLIR 的通用函数，也不是 NVIDIA 的 CUDA 启动接口。

由此得到完整职责链：

```text
Python 块程序 → TTIR 的计算与访问
             → adapter 的结构化表示与 metadata
             → NPU 编译器的设备对象
             → launcher 的参数打包与运行时启动
```

编译成功只覆盖前三段的一部分证据。设备数值、同步正确性与性能，需要在兼容环境中真正运行后另行验证。

## 7. 版本证据与有限阅读任务

本章依据公开镜像的 v3.2.1，提交 `2badfc89e70a9b7a5e88463a116c2feddce4b101`。其 LLVM hash 为 `b5cc222d7429fe6f18c787f633d5262fac2e676f`，AscendNPU-IR 子模块指向 `47a0229060e37f92a49cfb82d81c756628e6c7ae`。此前 [HIVM 源码篇](/notes/compile/ai-compiler/npu-ir/source_path)使用另一份独立固定快照；二者可帮助比较机制，不构成已验证的组合环境。

当前证据是源码检查，未构建 Triton-Ascend，也没有 NPU 执行结果。可先沿 [编译阶段](https://github.com/triton-lang/triton-ascend/blob/v3.2.1/third_party/ascend/backend/compiler.py)、[转换组织](https://github.com/triton-lang/triton-ascend/blob/v3.2.1/third_party/ascend/lib/TritonToLinalg/TritonToLinalgPass.cpp)、[Load/Store 转换](https://github.com/triton-lang/triton-ascend/blob/v3.2.1/third_party/ascend/lib/TritonToLinalg/LoadStoreConverter.cpp)、[启动器](https://github.com/triton-lang/triton-ascend/blob/v3.2.1/third_party/ascend/backend/driver.py)核对四个消费边界。

阅读时完成一个有限任务：在源码里找到最后一个块的 mask 处理，说明它怎样影响数据准备；再找到后端把 ttadapter 交给外部工具的位置。暂时不用阅读所有选项或整个 AscendNPU-IR。若开启 `force_simt_only`，应重新确认阶段表，而不能复用常规路径的结论。
