---
order: 50
title: Transform Dialect 与显式调度
updated: 2026-10-04
---

# Transform Dialect 与显式调度

同一个 Linalg 计算可以选择不同的 tile size、遍历次序和向量宽度。前面用 C++ Pass 表达“怎样修改 IR”；现在希望把“对哪部分计算，按什么顺序应用哪些变换”单独写出来，让它可以阅读、修改与复现。

Transform Dialect 用 MLIR 表达这份调度。它处理的对象仍然是原来的计算 IR，但它自己的值代表待变换的对象集合。把这两层分清，后面的匹配、句柄与失效就有了共同依据。

## 1. 算法与调度的两层 IR

沿用上一章的计算：

```text
算法 IR：linalg.generic 表达 y[i] = 2 × (x[i] + 1)
调度 IR：匹配 generic → 分块 4 → 对块向量化 4
```

算法 IR 常称 payload IR。它将来参与生成目标程序：里面的 `%x` 是程序的数组，`%a` 是计算中的数值。

调度 IR 则在编译时由 Transform interpreter 执行。里面的 `%target` 关联到某些 Operation，不是数组中的元素，也不是目标程序运行时要传递的指针。完成调度后，算法 IR 已经改变；目标程序不需要运行这段 Transform 程序。

这层分离允许同一份算法配不同调度，但并不会免除合法性条件。若变换需要无依赖、特定迭代类型或尺寸保证，调度仍要满足这些条件。

## 2. 句柄与对象集合

上一章的匹配是：

```text
%target = transform.structured.match ops{["linalg.generic"]} in %root
  : (!transform.any_op) -> !transform.any_op
```

`%root` 关联本次解释器处理的根操作；在本例里是模块。匹配在这个范围内寻找指定名字的操作，把找到的对象关联给 `%target`。

一个 handle 可以关联零个、一个或多个对象；其类型不是元素数量的保证。我们的输入只有一个 generic，所以此时集合里只有它。若以后输入变成三个 generic，匹配可能得到三个候选：后续变换会怎样处理这组对象，要看相应操作的契约，不能默认“只选第一项”。

需要更精确地定位时，可以按操作名、属性、父子关系等进一步匹配或筛选。接口和扩展提供更多定位方式；最终目的始终是让“想优化的那段计算”与“拿到的对象集合”一致。

## 3. 变换结果与句柄失效

分块操作接受旧对象，返回修改后对象及新循环：

```text
%tiled, %loop = transform.structured.tile_using_for %target tile_sizes [4]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
transform.structured.vectorize %tiled vector_sizes [4] : !transform.any_op
```

可以把执行时的关联画成：

```text
修改前：%target ──→ 原 generic

分块后：%target ──× 不能继续使用
        %loop  ──→ 新 scf.for
        %tiled ──→ 循环内部的小块 generic
```

分块消费了旧 handle。这里的“消费”是 Transform 对映射生命周期的约定，不是在运行时销毁一个 Tensor。变换可能替换原操作、改变其嵌套关系，继续把旧对象当作可靠目标就会产生歧义，所以调用方应使用变换返回的 handle。

如果把向量化的 `%tiled` 错改成 `%target`，实际解释器报错：

```text
error: op uses a handle invalidated by a previously executed transform op
note: invalidated by this transform op that consumes its operand #0 ...
```

错误定位到前一次分块，说明问题在调度的对象生命周期，而不是浮点加法不正确。

## 4. 别名句柄与嵌套对象

失效不只影响某个变量名。若两个 handle 指向同一个 payload 操作，消费其中一个后，另一个也不能假装指向一个独立副本。类似地，消费一个容器对象会影响关联其内部对象的旧 handle。

因此“我重新给它起一个 SSA 名字”不能修复失效。正确方式通常是使用变换返回的对象，或从仍有效的上层范围重新匹配当前 IR。并非每种变换都消费目标；Transform 通过效果描述区分查询、生成映射、修改 payload 和释放映射。

`transform.readonly` 在 named sequence 参数上描述的是这个 handle 的使用/消费约定。它不是说整个 sequence 不能修改该范围内的计算：本例 `%root` 标记 readonly，仍然可以匹配其子对象并分块。真正被消费的是相应子对象的 handle。

解释器的 expensive checks 可以帮助捕获这些错误，但关闭检查并不会让失效句柄变合法。它是调度程序需要遵守的协议。

## 5. 失败传播与抑制

变换可能不适用于某个对象。例如 `structured.vectorize` 接受 Linalg 操作，把 `scf.for` 交给它就不满足条件。这种“不适用”可能以 silenceable failure 返回；协议损坏或无法继续的错误则是 definite failure。

二者不等于“警告”和“普通错误”：silenceable failure 只有被合适的外层操作显式处理时，才可以抑制。未处理而传播到最外层，仍会让解释失败。

实验专门按下面顺序执行：

```text
在 transform.sequence failures(suppress) 中：
  1. 匹配 generic
  2. 分块 4                  ← 已修改 payload
  3. 尝试向量化 %loop        ← scf.for 不是 Linalg，silenceable failure
  4. transform.print %scope  ← 仍会执行
```

实际打印结果仍包含新 `scf.for` 和内部 generic，没有恢复成分块之前的程序。固定版本的 sequence 在 suppress 模式下抑制本次可抑制失败，然后继续执行后面的操作。

这个例子说明两点。第一，**抑制不是事务回滚**，之前成功的修改会保留。第二，**抑制不修复失效映射**：第 3 步仍涉及消费 handle，后续不能随意使用指向循环内部的旧 handle。Definite failure 也不会因 suppress 而被忽略。

要实现“试一种调度，不适用就恢复并试另一种”，需要明确选择具有所需隔离/恢复语义的机制，或在可恢复的副本上尝试；不能仅把 sequence 模式换成 suppress。

## 6. 与 Pass、Pattern 和搜索的关系

现在可以把已经学习的几层重新连接起来：

| 层次 | 在本例里承担的工作 |
|---|---|
| Pattern / 变换实现 | 真正创建循环、切片或向量操作，并维护 IR |
| Pass / pipeline | 在工具里组织阶段、分析、验证及后续 lowering |
| Transform 程序 | 定位目标，显式组合分块、向量化等动作与参数 |
| 搜索或调优程序 | 如果需要，生成多份参数选择，调用编译与测量比较 |

Transform interpreter 本身通常由 Pass 调用。已有变换通过 Transform 操作暴露出来；需要新动作时也可以写扩展，但仅使用已有动作不必先开发一套新框架。

把 tile size 写成 4，只是明确记录了一种选择，不意味着 interpreter 已经自动搜索最优值。下篇会看到同样合法的向量计算也可能更慢；调度可复现是性能研究的起点，成本评估仍需要目标和测量。

## 7. 阅读检查与衔接

1. `%target` 关联两个 generic 时，它与一个包含两个元素的 `vector` 有什么本质区别？
2. 分块后使用另一个曾经指向原 generic 的 handle，能否绕过消费规则？
3. sequence 抑制了第 3 步的失败，第 2 步已经生成的循环还在吗？第 4 步是否执行？

继续读[调度选择与性能解释](./performance)。本文已实际验证成功调度、失效句柄、抑制后继续执行、传播失败及保留中间修改。版本依据：[Transform 设计文档](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Dialects/Transform.md)、[sequence 实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Dialect/Transform/IR/TransformOps.cpp)。可选 [15 实验](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/15-vector-transform)会打印实际 IR 和诊断。
