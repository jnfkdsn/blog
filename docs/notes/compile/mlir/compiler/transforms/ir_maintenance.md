---
order: 25
title: CFG 克隆、移动与符号维护
updated: 2026-10-04
---

# CFG 克隆、移动与符号维护

[C++ IR API](./ir_api)介绍了创建、替换和删除。面对一个有分支、嵌套Region和函数调用的函数，操作还要维护不同种类的关系：SSA引用、CFG边、Region所有权以及符号解析。

本章围绕同一个helper函数完成两项工作：先重命名并复制它，再尝试移动其中的常量。这样可以直接看到哪些引用随克隆改变，哪些仍指向原符号，以及为什么“节点已经移过去”还不等于变换合法。

## 1. 一个函数里的几种连接

helper有四个Block：入口、yes、no、merge。入口常量one被两个分支使用；yes还包含一个scf.if，其内部捕获函数参数x；merge通过Block参数接收结果，再调用leaf。

简化结构为：

```text
helper(flag,x):
  one = 1
  branch flag -> yes / no
  yes: v = nested if using x and one; branch merge(v)
  no:  b = x - one; branch merge(b)
  merge(m): r = call @leaf(m); return r
```

调用者另外有call @helper。这时至少要区分：operand指向哪个Value，branch指向哪个Block，操作属于哪个Region，call通过哪个SymbolRef找到函数。这些关系不能用同一种“改名字”统一处理。

## 2. 重命名要维护符号使用者

仅修改helper的sym_name，会让call @helper仍使用旧名字。固定工程使用SymbolTable：

```cpp
SymbolTable table(module);
if (failed(table.rename(helper, "renamed")))
  return failure();
```

目标名称必须不存在；本例先选择确定无冲突的新名。实际输出中caller的调用变为call @renamed，符号表查询也以新名称找到同一函数。

SSA的RAUW不能替代这件事：函数符号不是caller中的某个SSA operand。相反，改一个SymbolRef也不会自动替换函数体中的SSA值。[SymbolTable接口](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/SymbolTable.h)提供专门的使用与重命名机制。

## 3. 克隆整个 CFG 的映射

接下来复制renamed：

```cpp
IRMapping mapping;
auto copy = cast<func::FuncOp>(helper->clone(mapping));
copy.setName("helper_copy");
table.insert(copy);
```

clone递归复制函数体，并建立旧Block/Value到新对象的映射。新函数的入口参数不是旧参数，branch successor必须指向新函数中的Block，merge参数和内部操作结果也属于新对象。

固定探针检查了四个Block的归属、BlockArgument身份、条件分支的两个目标，以及嵌套if里的add左值确实是新函数的x参数。输出全部确认映射成立。

这不表示所有引用都改成新实体。clone里的call @leaf仍然指向原模块中的leaf，因为SymbolRef属于符号解析关系，不在SSA/Block映射的同一层。若要克隆一个相互调用的函数集合，还需设计相应符号映射；递归调用也不能仅靠IRMapping自动改名。

## 4. 跨 Block 移动破坏了什么

原来的one在入口定义，支配yes和no。现在用moveBefore把它移到yes中的第一条操作之前：

```text
entry: branch flag -> yes / no
  yes: one = 1; ... use one ...
  no:  b = x - one
```

从入口直接走no时，没有经过one的新定义。因此对象列表虽然已经更新，verifier报告operand does not dominate this use。

固定工程只在编译器对象上构造这个反例，验证失败后把one移回入口terminator前，重新验证成功。它没有把非法IR编译执行。

moveBefore负责改变位置和拥有链，不负责替变换证明所有operand可用、所有uses受支配或区域捕获合法。若跨越IsolatedFromAbove边界，还需通过合法参数/操作结构传值，不能把原来的外部SSA引用直接带进去。

## 5. Verifier 通过之后仍要证明语义

即使移动保持支配，也可能改变内存读写顺序、条件执行次数或异常/未定义行为的触发条件。例如把可能除零的计算从分支提升到入口，结构可以合法，却可能在原本不执行它的路径上引入问题。

所以合法移动需要两层判断：先维持IR结构关系，再依据效果、依赖与可推测执行条件证明语义。相关事实见[优化合法性](../analysis/legality)。如果在Pattern driver内部实施修改，还应使用对应rewriter通知协议；本例是独立C++探针，没有正在运行的driver需要接收通知。

## 6. 未知符号作用域的保守策略

还有一类输入：某个未注册操作带Region，它可能定义新的符号表。工具不知道是否应该向内部搜索，也不能把“没查到use”当成确定没有引用。

固定工程在模块body上调用getSymbolUses，遇到这种未知区域返回无法完整确定的结果。随后由本工具自己的保护条件拒绝重命名，保留原符号h。

要注意查询范围：对SymbolTable操作本身查询，不等同于扫描其body；符号表边界会限制遍历。本例明确查询body Region。也不要把工具加的保护条件说成所有版本rename都会自动拒绝未知结构，应核对所用版本实现及调用前提。

在真实工程里，如果新操作本来就属于已知方言，应注册它并提供明确结构；保守拒绝适用于当前确实无法理解的输入边界。

## 依据与实践

固定来源：[Operation克隆与移动](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/Operation.h)、[IRMapping](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/IR/IRMapping.h)、[SymbolTable实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/IR/SymbolTable.cpp)。

[26 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/26-ir-maintenance)提供完整函数、映射身份检查、重命名结果、非法移动诊断及恢复。有限练习：增加一个只在yes使用的常量，比较它移入yes所需的条件，与当前one的区别。
