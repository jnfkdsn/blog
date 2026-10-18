---
order: 7
title: Export 契约与 AOTInductor 部署
updated: 2026-10-17
---

# Export 契约与 AOTInductor 部署

`torch.compile` 可以在第一次调用时准备执行程序，适合与原来的 Python 应用一起使用。部署时却可能希望把图捕获和编译提前完成，在服务启动时直接加载产物，并明确允许哪些输入。

这时需要先把模型变成具有明确输入契约的导出程序，再为目标环境编译。本章继续使用 `relu(x @ weight + bias)`，将 batch 的允许范围写出来，观察导出、编译、加载三步分别得到什么。

## 1. 从一次调用到可交付的程序

JIT 路径收到新输入以后，可以通过 guards 判断是否需要新版本。部署产物则需要更明确的边界：输入形状如何变化，参数怎样提供，哪些计算被包含，运行环境要具备什么能力。

例如，本例约定：

```text
x:      [B,7]，float32
weight: [7,3]，float32
bias:   [3]，float32
B 的允许范围：2 到 8
输出：  [B,3]
```

这份契约把“支持动态 batch”从模糊承诺变成了可检查条件。后端可以利用内积维 7 和输出维 3，同时保留 batch 为符号量。

## 2. ExportedProgram 保存什么

使用 `torch.export` 导出模型：

```python
class AffineRelu(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.relu(x @ weight + bias)

batch = torch.export.Dim("batch", min=2, max=8)
example = (torch.randn(2, 7), torch.randn(7, 3), torch.randn(3))
ep = torch.export.export(
    AffineRelu(), example,
    dynamic_shapes=({0: batch}, None, None),
)
```

`dynamic_shapes` 中第一项说明 x 的第 0 维使用 batch 符号，后两项没有声明动态维度。导出结果包含计算图、输入输出签名、状态和形状约束等信息，而不是只有一段打印出来的 FX 代码。

本例实际导出的图核心为：

```python
matmul = torch.ops.aten.matmul.default(x, weight)
add = torch.ops.aten.add.Tensor(matmul, bias)
relu = torch.ops.aten.relu.default(add)
return (relu,)
```

范围约束打印为 `{s0: VR[2, 8]}`。内部符号名可以不同，但它与输入维度和允许范围的关系才是后续消费者需要的信息。

本例故意将 weight 和 bias 作为显式输入。如果模型把它们注册为参数，导出签名还要区分参数、buffer 和用户输入，并保存相应状态。看到图里多个 placeholder 时，应结合签名判断角色，不能都当成外部 API 的位置参数。

## 3. 导出约束与失败诊断

导出希望得到能够完整表达目标计算的程序，不能像一般 `torch.compile` 那样依靠任意 graph break 返回 Python 继续执行。

如果模型包含由 batch 决定的分支，而声明范围跨过分支边界，捕获过程必须证明导出的程序适用于整个范围，或给出约束冲突。不能只观察 batch=2 的路径，就默默把它用于 2 到 8 的所有输入。

解决方式取决于意图：缩小输入范围、显式表达可支持的控制流，或者调整模型。导出的失败有时在提示“声明的输入集合比这份图的有效集合更大”，并非单纯的 API 使用错误。

配套实验把 ExportedProgram 保存后重新加载，使用 batch 2、5、8 执行，并与 eager 比较。随后把 batch 改为 9，`loaded.module()` 的输入约束检查拒绝该调用。

这里的负例验证发生在 ExportedProgram 的调用包装层。不能据此推断任意裸底层函数都自动进行完全相同的检查；部署入口必须明确承担契约检查的责任。

## 4. AOTInductor 的编译产物

ExportedProgram 仍然是待编译的程序描述。为了提前准备目标实现，再交给 AOTInductor：

```python
package = torch._inductor.aoti_compile_and_package(
    ep, package_path="compiled.pt2"
)
loaded_model = torch._inductor.aoti_load_package(package)
y = loaded_model(*example)
```

这是本系列固定 PyTorch 2.7.1 实际验证的接口。下划线命名空间中的 API 需要按目标版本确认，理解主线时不必记忆所有参数。

AOTInductor 使用相应后端生成内核和调用包装，完成编译，并组织可加载的产物。加载后的调用执行已准备的实现，不要求重新用 Dynamo 分析原模型的 Python 函数。

在本机 CPU 验证中，编译包成功生成和加载，batch 2、5、8 均与 eager 一致。GPU 产物的设备架构、运行库和代码兼容性需要另行验证，不能由 CPU 结果代替。

## 5. 两种保存结果的区别

本实验同时产生两个 `.pt2` 文件，但它们的角色不同：

| 产物 | 生成方式 | 下一步消费者 |
|---|---|---|
| `exported.pt2` | `torch.export.save` | 恢复 ExportedProgram，检查、变换或继续编译 |
| `compiled.pt2` | AOTInductor 编译与打包 | 加载目标实现并调用 |

相同扩展名不意味着内部内容或加载 API 相同。前者保留较高层的程序与契约，后者包含目标相关编译结果及包装信息。

同样，保存一个导出图不等于已经完成部署验证；生成一个编译包也不等于任何机器都可以加载。至少要确认目标设备、操作系统/架构、运行时和依赖库与产物相容，并在预期输入范围内核验数值。

本轮使用 Python 加载编译包，已经验证产物保存和加载执行链。C++ 服务集成可以沿 AOTI loader/runtime 继续推进，但不把未实现的 C++ 服务端当作本例已有结果。

## 6. 训练 AOT 与部署 AOT

前文出现的 AOTAutograd 用于组织前向/反向图及其编译边界；它可以在普通 JIT 编译过程中被调用。这里的 AOTInductor 则将导出程序提前编译成部署产物。

二者可能共享底层组件，但回答不同问题：一个关注自动微分后的图怎样准备和编译，一个关注怎样在运行前准备可部署实现。

不要仅凭名称中的 AOT 把它们画成同一个可替换阶段。下一章会从梯度计算的实际需求说明 AOTAutograd 的工作。

## 7. 推演与实践入口

1. 已经有 ExportedProgram，为什么还需要一次目标编译？
2. 如果只在 batch=2 捕获，为什么要显式声明并验证 2 到 8 的范围？
3. 把 weight 改成模块参数后，为什么需要同时看 graph signature？
4. “导出包装拒绝 batch=9”与“裸机器码自动检查全部约束”差别在哪里？

实验见 [labs 的 export](https://github.com/jnfkdsn/aicompiler/tree/main/pytorch-compiler)。对应机制可核对 [torch.export](https://docs.pytorch.org/docs/2.7/export.html)、[AOTInductor 官方说明](https://docs.pytorch.org/docs/2.7/torch.compiler_aot_inductor.html) 和 [固定版本打包入口](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/__init__.py)。
