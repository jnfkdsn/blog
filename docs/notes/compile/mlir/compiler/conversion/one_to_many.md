---
order: 60
title: 一个值到多个值的转换
updated: 2026-10-04
---

# 一个值到多个值的转换

之前的类型转换把一个 range 值改为一个 i32。现在源表示用一个 `tuple<i32,i32>` 保存一对整数，目标阶段希望直接传递两个 i32。计算没有变化，但 Value 的数量变了：一个操作结果、一个函数参数或一个 call operand，都可能变成两个。

本章沿“构造一对值 → 调用求和函数 → 返回结果”的完整路径实现转换，重点是保存各值的对应关系，而不是只改变类型列表。

## 1. 一对整数的源表示

教学操作 `my.pack` 把两个 i32 组成 tuple，`my.first/second` 取相应分量。它们的语义非常有限，不涉及堆分配、动态列表或对象身份。

```text
func.func private @pair(%x:i32, %y:i32) -> tuple<i32,i32> {
  %p = my.pack %x, %y : tuple<i32,i32>
  return %p : tuple<i32,i32>
}
func.func private @sum(%p:tuple<i32,i32>) -> i32 {
  %x = my.first %p : tuple<i32,i32>
  %y = my.second %p : tuple<i32,i32>
  %r = arith.addi %x, %y : i32
  return %r : i32
}
```

调用者传入 20、22，最终应得到 42。转换前 p 是一个 Value，具有 tuple Type；两个分量不是两个隐藏的 OpResult，只能通过定义好的操作读取。

目标表示则把 pair 函数改成返回两个结果，把 sum 函数改成接收两个参数。tuple 的包装与提取操作可以消失。

## 2. 类型映射与值映射

TypeConverter 给出类型层面的规则：

```text
i32             → [i32]
tuple<i32,i32>  → [i32,i32]
```

结果用列表表达，才有能力容纳一个类型对应多个目标类型。工程先加入普通类型的 identity 转换，再加入 tuple 的专门规则；固定实现按相应回调优先级应用后注册的专门处理。

```cpp
converter.addConversion([](TupleType type,
                            SmallVectorImpl<Type> &result) -> LogicalResult {
  if (type.size() != 2 || !type.getType(0).isInteger(32) ||
      !type.getType(1).isInteger(32))
    return failure();
  llvm::append_range(result, type.getTypes());
  return success();
});
```

本例只支持这一种 pair，其他 tuple 应失败，不能假定一般嵌套 tuple 已处理完毕。

类型映射还没有说明具体的 p 由哪两个 Value 表示。实际转换 my.pack(x,y) 时，才建立 `p → [x,y]`；转换函数入口时，又建立 `旧参数 p → [新参数 p0,p1]`。这是相同类型规则在不同 Value 上的具体应用。

## 3. 一次替换对应一组新值

普通 `replaceOp(op, values)` 按旧结果数量逐个替换。原操作只有一个 tuple 结果，不能直接把两个 Value 塞进它的一个 operand 槽里。

1:N 转换框架维护“每个旧 Value 对应一个 ValueRange”的映射。my.pack 规则使用 `OneToNOpAdaptor` 读取已经转换的两个 scalar operands，再以一组替代值替换旧结果：

```cpp
SmallVector<Value> values{
    adaptor.getFirst().front(), adaptor.getSecond().front()};
SmallVector<ValueRange> replacements{ValueRange(values)};
rewriter.replaceOpWithMultiple(op, replacements);
```

外层列表长度为一，因为旧操作有一个结果；该结果对应的内层列表长度为二。若旧操作有多个结果，每个旧结果必须分别给出自己的目标范围，不能丢掉组边界。

框架把这个对应交给后续消费者。它不是一次不顾类型的 RAUW，也不是让最终 IR 的一条引用同时指向两个值。

## 4. 消费者怎样读取拆开的 operand

转换 `my.first %p` 时，OneToN adaptor 中的 input 已是 `[p0,p1]`。规则选择第一项替换自己的 i32 结果；second 选择第二项：

```cpp
ValueRange pair = adaptor.getInput();
if (pair.size() != 2)
  return failure();
rewriter.replaceOp(op, pair[index]);
```

其中 index 对 First 为 0，对 Second 为 1。提取操作的一个结果仍然只对应一个 scalar，所以这里可用普通 replaceOp。

如果改用旧 `op.getInput()`，拿到的仍是源 tuple Value，不是转换后的两项。1:N 使 adaptor 的意义更加直观：规则必须消费当前转换阶段的新表示。

## 5. 函数、调用与 Block 边界

实际工程还让 sum 的 pair 参数经过一次 `cf.br`，传到另一个 Block。这样可以检查转换没有只覆盖函数表面签名。

关键输出整理如下：

```text
func.func private @pair(%x:i32, %y:i32) -> (i32,i32) {
  return %x, %y : i32,i32
}
func.func private @sum(%p0:i32, %p1:i32) -> i32 {
  cf.br ^next(%p0, %p1 : i32,i32)
^next(%q0:i32, %q1:i32):
  %r = arith.addi %q0, %q1 : i32
  return %r : i32
}
```

调用端也必须同步：

```text
%p:2 = call @pair(%a, %b) : (i32,i32) -> (i32,i32)
%r = call @sum(%p#0, %p#1) : (i32,i32) -> i32
```

工程复用固定版本提供的函数、call 和 return 类型转换规则。call 规则还保留每个旧结果对应的新结果数量，再把新的 call 结果按组交回框架。自定义 `cf.br` 规则则平铺 adaptor 中每个原参数的目标范围，并让它们与已转换的目标 Block 参数一致。

这里特意实现了所需的无条件分支规则。不能据此声称任意 `cf.cond_br`、switch、SCF Region 或外部 ABI 已自动覆盖；增加新的边界，要检查其类型和传值协议。

## 6. 合法性要求闭合整条路径

转换目标要求 My 方言操作消失，函数签名与 body 参数符合 TypeConverter，call/return/branch 的 operands/results 也符合目标表示。

如果只转换 callee 而缺少 call 规则，源 caller 仍试图得到一个 tuple。实验的对照 Pass 会失败：

```text
failed to legalize operation 'func.call'
```

这比“最后输出里好像没有 my.pack”更强：目标契约需要每一条跨边界引用都闭合。

若确实需要新旧表示暂时共存，还必须设计相应 materialization，例如怎样把两项重新组合成源 tuple；那是一段有语义的桥接，不是随手增加一个 cast 就完成。相关原则见[混合表示与边界衔接](./materialization)。本实验采用完整转换，不保留这种混合边界。

外部调用者也必须遵守新的签名。若函数已经是公开 ABI，拆开返回值可能要求 wrapper 或新的调用约定；不能只转换内部 IR 就认定已有二进制调用者仍兼容。

## 7. 从转换结果到执行

转换后只剩标准标量、Func 和 CF 操作，再沿 LLVM lowering 生成 CPU object，并使用 C wrapper 调用 entry。实际结果包括：

```text
20 + 22 = 42
0 + 0 = 0
-3 + 5 = 2
100 + -100 = 0
```

附加 difference 函数使用相同的 pair 构造/提取但执行减法，20、22 实际得到 -2，用于识别加法交换性可能遮住的顺序错误。这些结果核对了本例的分量顺序、跨函数传递和基本数值，没有验证一般聚合类型的所有 ABI 或递归转换。

一个合适的练习是让 sum 计算 first-second：若某处错误交换了 pair 顺序，20、22 应得到 -2 而不是 2。再增加一个携带 pair 的新控制流边界，先画出每个旧参数对应的目标列表，再写规则和负例。

## 依据与实践

固定版本为 LLVM 20.1.8，采用当前 Dialect Conversion 中的 `OneToNOpAdaptor` 与 `replaceOpWithMultiple`。不要把其他版本的独立实验性 1:N 框架示例直接混入此工程。

源码依据：[转换框架接口](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/DialectConversion.h)、[Func call/return 的 1:N 实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp)、[上游多结果替换测试](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/test/lib/Dialect/Test/TestPatterns.cpp)。[21 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/21-conversion-patterns)提供完整转换、缺规则反例与 C 调用。
