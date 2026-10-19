---
excludeFromSidebar: true
order: 6
title: TVM 与 PyTorch Compiler 课程覆盖表
updated: 2026-10-17
---

# TVM 与 PyTorch Compiler 课程覆盖表

本表检查当前课程承诺的主题和证据。正文与参考实验已提供，不将作者运行成功记录成读者已掌握。更深的硬件优化和生产后端开发是后续任务，仍保留在长期能力范围内。

## TVM

| 主题 | 主要正文 | 作者证据 | 深入边界 |
|---|---|---|---|
| 完整编译链 | [01](/notes/compile/tvm/01_execution) | Relax→CPU实际数值 | 非大型模型性能评估 |
| 模型导入与表示 | [02](/notes/compile/tvm/02_relax) | 静态FX导入、手写符号Relax三种batch | 未验证所有导入器/数据相关shape |
| 图优化与实现转换 | [03](/notes/compile/tvm/03_graph_optimization) | 五阶段IR、函数分组和组合、最终数值 | 复杂多分支融合按任务深入 |
| TensorIR | [04](/notes/compile/tvm/04_tensor_ir) | 32×32函数编译与执行 | 复杂intrinsic另学 |
| Schedule | [05](/notes/compile/tvm/05_schedule) | tile4/8/7、尾部、非法归约并行、有限CPU计时 | GPU/NPU映射是机制说明，未设备运行 |
| 自动调度与外部实现 | [06](/notes/compile/tvm/06_implementation_selection) | 8次实测、数据库复用、PackedFunc填充输出 | BYOC机制查源码，未开发生产后端 |
| Runtime与部署 | [07](/notes/compile/tvm/07_runtime) | VM/分配IR、.so导出加载、shape/dtype负例 | 未实现C++服务或异构部署 |
| 综合修改与调查 | [08](/notes/compile/tvm/08_investigation) | 练习骨架、参考解和6组调度数值 | 学习者完成状态待本人反馈 |

## PyTorch Compiler

| 主题 | 主要正文 | 作者证据 | 深入边界 |
|---|---|---|---|
| compile与执行 | [01](/notes/compile/pytorch-compiler/01_execution) | Inductor CPU数值与实际wrapper/C++ | 小例子不作通用加速承诺 |
| Dynamo/FX | [02](/notes/compile/pytorch-compiler/02_capture) | backend收到FX、第二次复用 | 观察后端返回FX Python，单独标注 |
| guards/动态shape/断图 | [03](/notes/compile/pytorch-compiler/03_specialization) | 静态/动态计数、显式断图、Inductor动态执行 | 数据相关shape和完整控制流另学 |
| 规范化/分解 | [04](/notes/compile/pytorch-compiler/04_normalization) | add_→add、输入修改补copy_、前后状态验证 | 独立functionalize观察，不冒充所有AOT实际顺序 |
| Inductor IR/layout | [05](/notes/compile/pytorch-compiler/05_inductor_ir) | 连续/非连续输入、调度表示 | 复杂布局与目标机制按任务深入 |
| 融合/调度/codegen | [06](/notes/compile/pytorch-compiler/06_codegen) | addmm+ReLU、Softmax内部循环和临时存储 | Triton路径说明职责，未GPU执行 |
| export/AOT | [07](/notes/compile/pytorch-compiler/07_export_aot) | 导出约束、包编译加载、三种batch与wrapper越界拒绝 | 未验证裸ABI完整检查或C++服务 |
| 训练机制 | [08](/notes/compile/pytorch-compiler/08_training) | AOT前后向图、真实Inductor loss与三项梯度 | 分布式/优化器编译/重算性能另学 |

## 交叉能力

[系统对照](/notes/compile/compiler_comparison)连接输入、IR、shape、实现选择、产物和运行边界；两套系统保持各自真实路径。练习分别体验图改写与算子调度，为方向选择提供实际依据。

复现与证据记录见 [课程实验环境](https://github.com/jnfkdsn/aicompiler/blob/main/environments/ai-compiler-course.md)。目录生成、源码阅读、实际运行和学习者独立完成分别记录，不能互相替代。
