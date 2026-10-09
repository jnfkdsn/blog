---
order: 60
title: Python 绑定中的构造、变换与生命周期
updated: 2026-10-04
---

# Python 绑定中的构造、变换与生命周期

[C API](./c_api)展示了一个外部程序怎样管理模块并改变返回值。Python 绑定进一步把这些动作包装成可组合的对象接口，适合原型工具、测试输入生成和编译流程组织。它并没有另外实现一套 Python IR：脚本仍在操纵底层 MLIR 对象。

本章沿同一个 `20+22 → 42 → 改为7` 的过程展开。先完成一次变换，再解释看似简洁的 with、操作包装和插入点背后保留了哪些约束。

## 1. 解析与运行 Pipeline

下面的程序使用固定源码构建出的官方 mlir Python 包：

```python
from mlir.ir import Context, Location, Module
from mlir.passmanager import PassManager

source = """module {
  func.func @answer() -> i32 {
    %a = arith.constant 20 : i32
    %b = arith.constant 22 : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}"""
with Context() as ctx, Location.unknown():
    module = Module.parse(source)
    pm = PassManager.parse("builtin.module(canonicalize)")
    pm.run(module.operation)
    print(module)
```

输出仍然是一个返回常量 42 的函数。PassManager 调用的是 C++ 编译基础设施；Python 在这里组织流程，没有执行 answer 的目标机器码。

`with Context()` 把 ctx 设为当前线程的默认 Context，后续没有显式指定 context 的构造和解析可以使用它。Location.unknown 同样提供默认位置。它们解决“从哪里取得默认参数”的问题，不等于离开缩进后立即销毁一切相关对象。

## 2. 访问对象与定位修改位置

在同一个 with 中继续取得 return：

```python
function = module.body.operations[0]
block = function.regions[0].blocks[0]
ret = block.operations[len(block.operations) - 1]
```

module.body 是模块的 Block，function 是其中的操作包装；其第一个 Region 的第一个 Block 是函数体。本例结构固定，实际工具面对任意模块时仍需检查名称和形状。

Python 可能提供通用 Operation，也可能根据已注册的包装返回具体 OpView。两者访问同一个底层操作，`.operation` 可取得通用入口。不要把包装层的类对象与 IR 的第二份拷贝混淆。

这里显式用 `len-1`，因为固定版本该列表接口没有承诺与普通 Python list 相同的所有索引行为。使用绑定时应确认具体协议，而不是仅按 Python 习惯猜测。

## 3. 插入点与 SSA 修改

现在构造一个常量 7：

```python
from mlir.ir import InsertionPoint, IntegerType, IntegerAttr
from mlir.dialects import arith

with InsertionPoint(ret):
    i32 = IntegerType.get_signless(32)
    seven = arith.ConstantOp(i32, IntegerAttr.get(i32, 7))
ret.operands[0] = seven.result
assert module.operation.verify()
pm.run(module.operation)
print(module)
```

InsertionPoint(ret) 指定新操作插在 ret 之前。因此 seven 的定义支配 return 的使用。arith.ConstantOp 的 Python 构造包装把类型和属性交给底层操作构造；插入点负责让它进入正确的 Block。

`ret.operands[0] = seven.result` 改变的是引用关系。它没有计算整数返回值，也没有自动删掉常量 42。第二次 canonicalize 才清除死常量，输出为：

```text
module {
  func.func @answer() -> i32 {
    %c7_i32 = arith.constant 7 : i32
    return %c7_i32 : i32
  }
}
```

默认插入点减少了显式参数，但也意味着代码有一个环境依赖。较长函数或多个 Block 的构造中，应让插入点作用范围尽量清晰；否则一个正确操作可能被插进错误区域。必要时可显式传入位置/插入点参数。

## 4. 对象存活与默认作用域

Python 包装会保留相关拥有者的引用，让合法使用不必完全照搬 C 的手动释放。下面的模块在离开 with 后仍然可以验证：

```python
with Context():
    module = Module.parse("module { func.func @f() { return } }")
print(module.operation.verify())  # True
```

原因是 module 仍保留底层 Context 所需的生命周期依赖。with 结束撤销默认 Context 作用域，不是命令所有已创建对象立刻失效。

但这不表示操作永远存活。若显式 erase 一个操作，Python 变量仍可以存在，底层操作却已经删除。固定版本实验执行：

```python
victim.operation.erase()
victim.operation.verify()
```

第二行得到 RuntimeError：`the operation has been invalidated`。这项检查说明该显式删除路径会使包装失效；不能据此假设任何外部 C++ Pass 的所有删除路径都自动满足同样的检测保证。

因此运行会修改 IR 的 pipeline 后，应从存活模块重新取得需要的操作，尤其不要长期保存可能被 rewrite/erase 的叶操作包装。变量引用、包装生命周期和 IR 对象生命周期不是同一件事。

## 5. 错误报告与验证层次

文本不合法会在 parse 时失败；pipeline 名字错误会在 PassManager.parse 阶段失败；Pass 执行失败发生在 run；构造出的结构违反契约则由 verifier 报告。这些位置与 C++ 工具是一致的，只是调用者看到 Python 异常和诊断。

不要用一个宽泛 `except: pass` 吞掉所有错误后继续使用 module。若这是编译服务，应保留失败阶段、诊断和输入快照，并让上层知道没有得到有效编译结果。

本例 C 与 Python 两条路径产生相同的两份规范 IR，这比仅检查“import 成功”更有意义。但它仍不证明所有 dialect 的 Python 包装可用；某些扩展需要额外的生成绑定、注册和编译配置。

## 6. 从脚本原型到实际接入

Python 很适合生成边界输入、组合已有 Pass 和检查输出。例如把 answer 中两个常量参数化，得到一组可预测的常量折叠测试。但核心 rewrite 若需要与 driver、分析缓存及性能要求紧密合作，通常仍会放进 C++ Pass，再由 Python 调度。

安装环境也需要与代码一致：本系列使用 LLVM 20.1.8 源码的 MLIRPythonModules，以及同一次构建中的本地扩展库。不能从包名相同就推定任意 `pip install mlir` 得到的是相同实现。依赖、Python 版本和 package 路径在实验环境说明中固定。

可以做一个有限练习：把构造的 7 改为输入参数指定的整数，预测两次打印差异，再加入错误 pipeline 名称，说明错误发生在执行前还是执行中。无需先记住整套绑定 API。

## 依据与实践

固定来源：[Python 绑定及所有权](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/Bindings/Python.md)、[IR 包装实现](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/lib/Bindings/Python/IRCore.cpp)。当前在线[绑定文档](https://mlir.llvm.org/docs/Bindings/Python/)适合查询新版本，示例行为以固定构建为准。

[23 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/23-bindings)包含完整脚本、与 C 输出的对照、删除包装反例和默认 Context 作用域检查。后续若要把 IR 保存为紧凑文件，还需要区分[Bytecode 与方言版本](./bytecode)。
