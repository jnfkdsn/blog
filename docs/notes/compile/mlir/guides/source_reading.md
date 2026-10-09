---
order: 90
title: 沿一个变化阅读 MLIR 源码
updated: 2026-10-04
---

# 沿一个变化阅读 MLIR 源码

读懂概念以后，源码阅读最有价值的目标是解释一个已经观察到的变化。若直接从Dialect目录第一行读起，会同时遇到生成代码、注册、接口和许多无关规则，很难判断何时已经读够。

本章选择一个有限任务：解释为什么canonicalize能把`arith.addi %x, %zero`变为直接返回%x，再说明未知加法怎样进入LLVM。沿这项变化追定义、生成接口、事实生产者、调用者与消费者，就能形成完整链条，而不需要先通读整个Arith。

## 1. 先固定输入和可观察结果

```text
func.func @zero(%x: i32) -> i32 {
  %z = arith.constant 0 : i32
  %r = arith.addi %x, %z : i32
  return %r : i32
}
```

canonicalize后只剩return %x。这个结果提出两个具体问题：哪段代码判断右边是零，谁负责把return的引用改掉并删除add？

不要一开始就假定是某个RewritePattern。Greedy driver还可能执行fold和死代码删除；相同终态可以来自不同机制。本例的直接解释来自AddIOp的fold。

## 2. ODS 定义约束和实现入口

固定源码ArithOps.td中，Arith_AddIOp继承整数二元操作基类，包含Commutative和hasFolder等声明。先核对本例依赖的契约：两个输入和结果类型一致，i32加法按位宽计算；未声明额外overflow承诺。

hasFolder说明该操作提供fold入口，并不包含“右零变左值”的C++实现。ODS和生成代码负责让框架知道如何调用这个入口。

此时不必展开同文件所有除法、比较和扩展运算。读者已经知道要寻找AddIOp::fold，继续沿这个名字向下即可。[固定ODS](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)

## 3. 生成代码连接声明与普通 C++

构建目录中的ArithOps.h.inc包含AddIOp类、访问器和fold声明；ArithOps.cpp.inc包含生成实现。它们由mlir-tblgen依据同版本ODS产生，再被手写头文件和源文件include。

这里值得确认的是：getLhs/getRhs对应哪两个operand，FoldAdaptor提供什么信息，以及fold签名返回什么。无需逐行背完全部模板展开，也不要修改.inc来实现新规则；生成文件会被下一次构建覆盖。

“生成C++”这一阶段构造的是编译器实现，运行canonicalize时调用的才是这些编译器代码；两者都不是执行目标函数zero。

## 4. Fold 生产替换事实

ArithOps.cpp中的关键代码为：

```cpp
OpFoldResult arith::AddIOp::fold(FoldAdaptor adaptor) {
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  // 其余情况还有减法关系和常量计算。
}
```

adaptor中的operand信息可以携带已知常量属性；本例右侧被识别为零。返回getLhs()表示“本操作的结果可由现有左值替代”。这段返回并没有亲自遍历return的operand，也没有手动erase当前add。

改变条件就能确认边界：若右侧为未知参数y，第一条规则不能给出替换。fold可能继续尝试别的已支持关系，但不能凭两个值都是i32就消去加法。[固定实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Arith/IR/ArithOps.cpp)

## 5. Pass 与 Driver 消费事实

Canonicalizer::runOnOperation调用applyPatternsGreedily。该driver处理一个操作时，除了尝试Patterns，还可调用op->fold。获得非空fold结果后，若其中是已有Value，就收集为替换值；若是Attribute，则可能需要物化成常量操作。

在本例中，替换值是%x。因此driver通过rewriter更新结果使用者并移除add。常量零随后无人使用，也可以被死代码清理删除。

现在观察到的终态被拆成有依据的过程：

```text
AddIOp::fold：右零成立 → 给出%x
Greedy driver：用%x替换add结果 → 删除add
死代码清理：零常量无人使用 → 删除常量
```

这也解释了为什么不能把所有变化都归因于用户提供的Pattern。源码阅读应同时找到事实生产和消费的位置。[Canonicalizer](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Canonicalizer.cpp)、[Greedy driver](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp)

## 6. 未知加法进入 Conversion

把主例改为`addi %x,%y`，加法必须保留。ArithToLLVM.cpp将AddIOpLowering关联到LLVM::AddOp，并使用overflow属性转换器。

固定实验运行Arith/Func到LLVM转换后，未知加法变为llvm.add。这里解决的是目标表示，不是证明计算可以消失。若源操作带nuw/nsw等承诺，相应信息也需要正确转移，不能只检查操作名变了。

这条分支连接之前的两类机制：fold利用已有事实简化，conversion保持计算含义并改变表示。它们可能在同一pipeline中先后出现，但角色不同。[ArithToLLVM](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Conversion/ArithToLLVM/ArithToLLVM.cpp)

## 7. 用测试界定阅读终点

最后回到Arith的canonicalize测试，查看输入、运行选项与预期关系。测试帮助确认该行为属于维护范围，但单个文本匹配仍不是所有输入等价性的证明。

这一轮读够的标准是：能够画出上述有限链，指出右零事实在哪里产生、在哪里消费，说明未知y的变体为何不同，并用固定工具观察两种结果。生成器内部的模板算法、worklist全部实现或所有Arith转换规则可以暂缓，直到具体修改任务需要它们。

迁移到真实项目时也采用同样方法：选一项操作或事实，追定义→生成接口→生产者→消费者→测试。若研究的是Triton布局或HIVM同步，先明确要解释哪项布局/依赖变化，再进入相应源码链，避免把整个仓库当成必须读完的课本。

## 实践

[26 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/26-ir-maintenance)提供本章输入、固定源码/生成文件的七个有限入口与真实转换输出。源码观察器记录commit、文件哈希和小段上下文；它帮助定位，不代替正文的因果解释，也不将源码查阅记作项目构建或设备运行。
