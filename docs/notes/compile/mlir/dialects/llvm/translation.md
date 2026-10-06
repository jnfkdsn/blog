---
order: 3
title: LLVM Dialect 与 LLVM IR
updated: 2026-10-04
---

# LLVM Dialect 与 LLVM IR

LLVM Dialect 已经出现 load、GEP 和低层函数，为什么还要“翻译成 LLVM IR”？因为相近的语义仍可能采用不同的对象模型：一边是 MLIR 的 Operation/Region/Block，另一边是 LLVM 自己的 Module/Function/Instruction。

Translation 负责在这两种表示之间建立对应。它接收已经准备好进入 LLVM 的操作，把它们构造成 LLVM IR 对象；后续 LLVM 优化和目标代码生成才继续处理这些对象。

## 1. 同一函数的两种表示

第一篇的整数求和函数使用以下核心操作：

```text
%q = llvm.getelementptr %p[1] : (!llvm.ptr) -> !llvm.ptr, i32
%x = llvm.load %p : !llvm.ptr -> i32
%y = llvm.load %q : !llvm.ptr -> i32
%s = llvm.add %x, %y : i32
llvm.return %s : i32
```

对完整模块执行 `mlir-translate --mlir-to-llvmir`，实际得到：

<!-- cpu-llvmir: pointer -->
```text
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @sum_pair(ptr %0) {
  %2 = getelementptr i32, ptr %0, i32 1
  %3 = load i32, ptr %0, align 4
  %4 = load i32, ptr %2, align 4
  %5 = add i32 %3, %4
  ret i32 %5
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

GEP、load、add 与 return 的对应很直接。这种“接近目标 IR 的语义”是 LLVM Dialect 的设计目的之一：需要作的复杂计算变换尽量已经在 MLIR lowering 中完成，translation 不再临时猜测一个矩阵乘法应该如何实现。

输出使用 `.ll` 文本格式时只是 LLVM IR 的可读形式；翻译过程构造的对象属于 LLVM。无论保存为文本还是 bitcode，此时仍没有必然生成机器码。

## 2. Block 参数如何成为 PHI

两边也有对象结构上的差异。MLIR 用 Block 参数在边上传值；LLVM IR 使用 PHI 节点选择来自不同前驱的值。

下面的函数按条件在 8 和 13 等两个输入中选择一个：

<!-- cpu-example: choose -->
```text
module {
  llvm.func @choose(%cond: i1, %a: i32, %b: i32) -> i32 {
    llvm.cond_br %cond, ^left, ^right
  ^left:
    llvm.br ^join(%a : i32)
  ^right:
    llvm.br ^join(%b : i32)
  ^join(%x: i32):
    llvm.return %x : i32
  }
}
```

进入 `^join` 时，左前驱传 `%a`，右前驱传 `%b`。LLVM IR 中的实际结果为：

<!-- cpu-llvmir: choose -->
```text
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @choose(i1 %0, i32 %1, i32 %2) {
  br i1 %0, label %4, label %5

4:                                                ; preds = %3
  br label %6

5:                                                ; preds = %3
  br label %6

6:                                                ; preds = %4, %5
  %7 = phi i32 [ %2, %5 ], [ %1, %4 ]
  ret i32 %7
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

PHI 的每一项都把一个值与其来源前驱配对。它保留的是前面的控制流传值协议，不是无条件同时计算两个值后再做普通内存读写。列表打印顺序也不决定执行路径；真正决定选择的是本次从哪个前驱到达。

这个例子说明 translation 不只是机械删掉每行的 `llvm.` 前缀。它需要将对象关系转换成目标 IR 所需的形式。

## 3. 常量与模块结构

MLIR 中常量通常由操作产生 SSA Value，例如 `llvm.mlir.constant`；LLVM IR 可以直接把常量写进指令 operand。类似的辅助操作用于适配两边结构，不表示目标机器必然单独执行一条“创建常量对象”的指令。

模块还包含函数声明、全局对象、属性与调试信息。translation 会按支持的规则将它们映射到 LLVM 模块中。最终打印出来的 metadata 可能与目标计算无关，但它们仍属于模块信息的一部分。

## 4. 可解析不等于可翻译

MLIR 允许多个方言共同组成一个模块。下面这种局部结构在 MLIR 中可以是合法的：

```text
llvm.func @mixed() -> i32 {
  %v = arith.constant 7 : i32
  llvm.return %v : i32
}
```

用注册了 Arith 的 `mlir-opt` 可以解析并验证它，但本系列使用的 LLVM translation 路径不能直接翻译仍未降低的 `arith.constant`。还要区分工具入口：固定版本 `mlir-translate --mlir-to-llvmir` 没有注册 Arith 的自定义文本解析，直接读取上述文本会先在解析处失败；改成 generic 文本并允许未注册方言后，才会观察到缺少翻译接口的失败。两种错误处于不同层次，都没有替代缺失的 lowering。

解析和 verifier 只回答“当前形式是否满足其约束”，不回答“后端是否已经支持全部操作”。

先将 arith 降为对应 LLVM 操作，再 translation，才跨过这一边界。相同问题也可能来自残留的 tensor、memref、自定义操作或未能消解的 conversion cast。

因此，排查 translation 失败时应先找实际残留了什么：是不是缺少一个 lowering 阶段？是不是转换目标允许某种操作暂时保留？是不是所需翻译接口未注册？反复运行打印工具不会消除这些缺口。

## 5. 翻译完成后的工作

完成 translation，只说明已经得到后端能接着处理的 LLVM IR。后面仍可能需要：

```text
LLVM IR
   ↓ 目标相关优化、指令选择、寄存器分配等
机器码 / 目标文件
   ↓ 符号解析、链接、加载
在约定的 ABI 下调用
```

LLVM Dialect lowering、translation 与机器码执行各有输入输出和失败条件。把它们分开观察，能区分“转换没有完成”“外部符号没找到”和“调用参数不符合 ABI”等不同问题。

后续[Translation、JIT 与 AOT](../../compiler/lowering/execution)将继续把这条链连接到实际运行。[LLVM Target 官方说明](https://mlir.llvm.org/docs/TargetLLVMIR/)也将转换到可翻译方言与 translation 作为两个阶段区分。

## 阅读自查

1. `llvm.load` 已经很像 LLVM 指令，为什么它仍不是 LLVM 自己的 Instruction 对象？
2. `^join(%x)` 的两个前驱分别传不同值时，PHI 必须保留什么对应关系？
3. 一个 verifier 通过、但含 arith 操作的模块，为什么可能 translation 失败？

## 资料与实践

实际文本按 LLVM 20.1.8 验证。对象映射入口见[ModuleTranslation.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Target/LLVMIR/ModuleTranslation.cpp)与[LLVM Dialect 翻译实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.cpp)。

可选实践：[翻译、PHI 与失败边界](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/11-cpu-lowering)。
