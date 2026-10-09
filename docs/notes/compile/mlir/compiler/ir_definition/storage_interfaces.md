---
order: 90
title: 参数存储、类型身份与接口消费
updated: 2026-10-04
---

# 参数存储、类型身份与接口消费

假设一个简单存储规划器要放入一块连续的 2×3 f32 数据，要求起始地址按 16 字节对齐。已用空间到偏移 5，它应把新块放在 [16,40)，留下 [5,16) 的对齐空隙。

计算并不复杂，但编译器需要可靠地保存“2×3”和“16”，并让规划器取得所需事实。本章沿这次规划解释数组参数的所有权、Type/Attribute 的共享身份、相应接口，以及递归类型为什么有时需要受限的可变存储。

## 1. 数据契约与规划策略

用一个教学类型和一个属性分别承载信息：

```text
!my.tile<[2, 3]>
#my.alignment<16>
```

TileType 约定这是静态正形状、连续 f32 的逻辑块，因此 payload 为 `2×3×4=24` 字节。AlignmentAttr 提供本规划器采用的字节对齐要求。本例允许 1 到 4096 内的二次幂。

这不是通用 MLIR Tensor 的物理存储规则：实际布局、padding、压缩或目标 DataLayout 都可能改变字节数。这里先明确狭窄语义，才能检验后面的存储与接口机制。

类型和属性分开，也保留了两个设计维度：相同 Tile 可以在不同规划策略下要求不同对齐。若某个项目把布局作为值类型的不可分割契约，就应把相应信息纳入类型，而不能机械照搬本例分类。

## 2. 数组参数必须拥有有效存储

创建类型时，调用者可能传入一个临时 C++ 数组：

```cpp
SmallVector<int64_t> shape{2, 3};
auto tile = my::TileType::get(&context, shape);
shape[0] = 99;
shape.clear();
```

期望 tile 仍表示 [2,3]。`ArrayRef` 只是一段数据的视图；复制 ArrayRef 本身不会复制元素。如果类型存储仍指向调用者的缓冲区，后续修改、扩容或销毁都会破坏已创建的类型。

因此 ODS 参数使用带分配约定的类型：

```tablegen
let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
```

生成存储的关键步骤是：

```cpp
shape = allocator.copyInto(shape);
return new (allocator.allocate<TileTypeStorage>())
    TileTypeStorage(std::move(shape));
```

数组元素进入 Context 管理的存储，类型句柄中的 shape 视图指向这份副本。作者程序修改原数组后，打印仍为 `!my.tile<[2, 3]>`，payload 仍为 24。

这与保存目标程序的张量元素不同：当前复制的是编译器描述类型所需的两个尺寸整数，既没有分配目标设备 Tensor，也没有复制 24 字节的运行时 payload。

## 3. 参数决定共享身份

在同一个 Context 再创建 `[2,3]`，得到与先前相等的 TileType 句柄。底层通过参数 key、相等比较和 hash 查找唯一存储；不同的输入数组地址不改变类型身份。

由此产生两项约束。首先，key 必须完整描述语义身份。若忽略元素类型或布局参数，原本不同的类型可能错误地合并。其次，进入 key 的内容不能在创建后随意修改，否则哈希查找和已经建立的相等关系会失效。

本例的数组参数按元素比较并被稳定保存。若存储自定义结构，除了 allocator，还可能需要自定义 comparator 或 hash；“能放进一个 C++ struct”不等于已经具备正确的 uniquing 规则。

Type/Attribute 句柄没有独立拥有一份可修改副本。修改一个句柄所共享的对象，可能影响所有引用它的地方。Context 销毁后，其类型、属性及参数视图也不能继续使用。

## 4. 参数验证先建立可用前提

TileType 的 verifier 要求 shape 非空、每维大于零，并在乘法前检查字节数不会超出 int64。AlignmentAttr 则拒绝零、负数、非二次幂和大于 4096 的值。

这些是教学抽象自己的限制；普通 Tensor 可以有零维长度或 rank=0，不能把这里的约束推广给所有 MLIR 类型。

解析或面对不可信参数时使用 checked 构造路径，失败时产生诊断。普通 `get` 适合调用者已保证前提的路径，不能靠 Release 构建中的断言替代输入验证。

类型检查成功后，`getPayloadBytes()` 才能按既定范围执行乘法。它仍只给出 payload 大小；对齐空隙以及给定容量是否足够，是规划器的另一个判断。

## 5. 消费者查询接口而非具体类

如果规划器只接受 TileType 和 AlignmentAttr，可以直接调用它们的访问器。引入 Type/Attribute Interface 的价值，是让不同具体表示都能回答同一规划问题。

本例定义两个狭窄协议：

```tablegen
def TileInfo : TypeInterface<"TileInfo"> {
  let cppNamespace = "::mlir::my";
  let methods = [InterfaceMethod<"Payload bytes",
      "int64_t", "getPayloadBytes">];
}
def AlignmentPolicy : AttrInterface<"AlignmentPolicy"> {
  let cppNamespace = "::mlir::my";
  let methods = [InterfaceMethod<"Required alignment",
      "int64_t", "getRequiredAlignment">];
}
```

具体 Type/Attribute 声明实现相应接口，方法分别根据 shape 计算 24、根据属性返回 16。消费者接收的是普通 Type 和 Attribute，再查询：

```cpp
auto info = dyn_cast<my::TileInfo>(type);
auto alignment = dyn_cast<my::AlignmentPolicy>(policy);
if (!info || !alignment)
  return std::nullopt;
```

这是编译器中的协议调用，不是目标程序运行时的虚函数调用。它与 Op Interface 的思想相同，但被询问的对象是 Type 或 Attribute；无需为了查询静态形状先找到产生该值的某个操作。

一个普通 i32 类型没有实现本教学协议，规划器就返回“不支持”。不能把接口查询失败当成 payload 为零。未来可以为另一种类型直接实现或注册外部模型，但新答案仍必须满足同一个字节数契约。

## 6. 从接口答案完成规划

现在规划器已知 payload=24、alignment=16、start=5。计算对齐填充：

```text
padding = (alignment - start % alignment) % alignment = 11
begin   = start + padding = 16
end     = begin + payload = 40
```

若容量为 64，这块范围可容纳；容量只有 32 则拒绝。实现先比较剩余容量，确认 padding 和 payload 放得下，再执行相加，避免通过整数溢出误判范围。

实际探针得到：

```text
tile=!my.tile<[2, 3]> same=1 bytes=24 alignment=16 start=16 end=40
planner capacity64=1 bounds_and_protocol_rejected=1
```

最后一项包含容量不足和类型不支持两个反例。这个程序执行了真实 C++ 查询与范围计算，但没有调用 memref.alloc，也没有保证某个运行时 allocator 返回的地址满足 16 对齐。规划结果与最终分配实现仍要在后续边界连接。

## 7. 递归类型的受限可变存储

普通参数化类型通过不可变 key 创建，通常已经足够。递归结构有不同的构造需要：先得到名为 Node 的类型身份，再把它的 body 定义为“一个 i32 和一个 Node”。如果必须在获得 Node 前先构造完整 body，就形成循环依赖。

一种解决方式是把稳定名字作为 key，把 body 作为受限的可变部分：

```text
get("Node") → 得到尚无 body 的身份
    ↓
构造 tuple<i32, Node> 作为 body
    ↓
setBody(body) → 所有引用同一 Node 的句柄看到这个 body
```

工程用手写 `NodeStorage` 展示这一过程。key 的字符串仍需复制到 allocator；相等比较始终依据名字。`mutate` 允许首次设置，重复设置相同 body 成功，尝试改成不同 body 则失败。

实际观察为 `same=1、first=1、repeat=1、conflict_rejected=1、shared_body=1`。这说明变化作用于共享身份，并非只改变局部句柄。

这个分支的目的是解决分阶段构造，不是为普通 Tensor shape 提供任意修改通道。应该在消费者分析之前完成定义，并维护“一致 body”的项目协议；不能更改已经进入 uniquing key 的数组内容。并发访问、递归遍历终止、文本/bytecode 表示和 ABI lowering 还需要各自设计。

本例 NodeType 仅用于 C++ 构造与身份观察，没有提供文本语法或 lowering，也不在 MLIR fixture 中使用。它依据固定上游 TestRecursiveType 的原则改写；读者进入真实递归类型项目时再扩展这些边界。

## 检查与依据

可先把 shape 改为 [2,4]，预测 payload、对齐起点和终点。再试图为同名 Node 设置不同 body，解释为什么返回失败更符合稳定身份。最后思考：若 Tile 实际带有行 padding，应该修改哪条协议，才能避免消费者继续按 24 字节规划？

固定依据为 [Type/Attribute 定义规范](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/docs/DefiningDialects/AttributesAndTypes.md)、[上游递归类型实例](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/test/lib/Dialect/Test/TestTypes.h)、[接口定义实例](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/test/lib/Dialect/Test/TestInterfaces.td)。[在线文档](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)可查参数分类，生成代码按 20.1.8 核对。

[20 工程](https://github.com/jnfkdsn/aicompiler/tree/main/llvm-mlir/20-ods-storage)保留 generated storage、真实身份与接口查询、递归 body 冲突及参数诊断。至此，定义机制已经覆盖从操作字段、语义协议到共享类型存储的主要工作；下一步应围绕实际转换或目标需求使用这些能力，而不必预先背下全部生成 API。
