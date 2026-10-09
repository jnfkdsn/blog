---
order: 40
title: 多 Block 区域与多路选择
updated: 2026-10-04
---

# 多 Block 区域与多路选择

scf.if和scf.for通过固定的Region协议保留结构，但它们的body通常只允许一个Block。如果某段局部计算需要显式CFG，又希望把它作为一个有结果的操作放进外层结构，可以用scf.execute_region封装这段区域。若需求本身是按整数标签多路选择，则scf.index_switch保留更明确的选择结构。

本章先让一个多出口区域计算x±1，再把多路选择的结果接到共同使用者。两者都通过yield交出值，但它们怎样进入区域、选择路径和降低到CF并不相同。

## 1. ExecuteRegion 执行一次区域

```text
func.func @adjust(%flag: i1, %x: i32) -> i32 {
  %one = arith.constant 1 : i32
  %r = scf.execute_region -> i32 {
    cf.cond_br %flag, ^yes, ^no
  ^yes:
    %a = arith.addi %x, %one : i32
    scf.yield %a : i32
  ^no:
    %b = arith.subi %x, %one : i32
    scf.yield %b : i32
  }
  return %r : i32
}
```

execute_region执行所持Region一次，本身没有operand，入口Block没有参数。它可以捕获支配父操作的外层值，因此这里直接使用flag、x和one。

flag为true时沿yes执行，yield x+1；否则yield x-1。r是父操作结果，供外部return使用。区域内的a、b不能越过作用域直接被外部引用。

“执行一次”指一次进入Region，不代表其中每条操作都执行一次；如果内部CFG有分支或回边，其执行仍由内部控制流决定。

## 2. 多个出口汇入共同 Block

转换到CF时，execute_region内部Block被接入外围CFG，多个yield变成到同一continuation Block的分支：

```text
^yes:
  %a = arith.addi %x, %one : i32
  cf.br ^continue(%a : i32)
^no:
  %b = arith.subi %x, %one : i32
  cf.br ^continue(%b : i32)
^continue(%r: i32):
  return %r : i32
```

父操作结果原先表达的合流关系，变成Block参数和分支实参。出口数量可以大于一，但每条出口传出的数量和类型必须符合父操作结果。

这个转换与普通“复制区域文本”不同：需要连接入口、重写出口、映射结果使用者，并保留CFG和SSA关系。固定实验对true/false及三个x值实际执行，结果均符合x±1。

## 3. 合法 IR 与特定 Pass 支持范围

上面的多出口execute_region通过自身verifier，也能转CF，但固定20.1.8的One-Shot Bufferization在这份结构上报告：

```text
op without unique scf.yield is not supported
```

这不说明execute_region语法错误，而是该接口实现要求唯一yield。可以根据pipeline需要先规范化出口、改变相关阶段顺序，或使用支持该结构的实现；不能为了消除诊断便删除一个合法控制路径。

本工程将标量区域转换和含tensor的forall bufferization分别验证，避免把不支持的混合输入写成可直接运行的通用管线。这个例子也是[阶段契约](../../tutorials/pipelines/overview)的一项具体检查。

## 4. IndexSwitch 保留多路选择

```text
%r = scf.index_switch %tag -> i32
case 0 {
  scf.yield %x : i32
}
case 2 {
  %twice = arith.addi %x, %x : i32
  scf.yield %twice : i32
}
default {
  %one = arith.constant 1 : i32
  %v = arith.subi %x, %one : i32
  scf.yield %v : i32
}
```

tag为0返回x，为2返回2x，其余返回x-1。每个case与default分别有区域，选中的路径通过yield产生r；这里没有C语言switch的fallthrough。

它适合保留多路选择关系，让后续变换知道这些区域属于同一个选择操作。转换成CF后，各区域出口与前例一样汇入一个带参数的Block，选择本身由cf.switch表达。

## 5. 固定版本的 Index 位宽限制

这里必须核对具体lowering。固定20.1.8的IndexSwitchLowering把selector从index cast为i32，并使用i32 case值。因此本章正确性用例将tag限制在有符号32位范围，测试-1、0、1、2、3。

对于本机64位index，额外反例tag=4294967296在源语义中应走default，但这条lowering截断后变为0。作者实际观察到x=7时返回7，而源default应返回6。

这是一项记录下来的版本实现限制，不是index_switch应当具有的通用语义。处理宽标签的项目需要限制/验证范围，或采用能保留宽度的合法化路径，例如在完整宽度上比较并组织分支；不能因为verifier与编译器都成功就宣称语义保持。升级版本时也应重新核对这条路径。

## 依据与实践

固定定义：[SCFOps](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)、[SCFToControlFlow](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp)、[SCF Bufferization接口](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp)。

[26 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/26-ir-maintenance)保存完整输入、CF与CPU结果、bufferization失败和宽标签反例。有限练习：把两个yield改为先传入区域内部的合流Block，再由唯一yield退出，解释它与原语义的关系及为何可能满足更多消费者。
