---
order: 30
title: Triton 加法算子的编译路径
updated: 2026-10-04
---

# Triton 加法算子的编译路径

Triton 让程序直接描述一块元素的计算。理解它与 MLIR 的关系，适合从一个加法 kernel 沿地址、mask 和元素分布追到设备代码，而不是先阅读全部方言定义。

本章固定 Triton v3.2.0 的 NVIDIA 后端。相同项目还可以有其他目标，下面的 warp、PTX 和 cubin 路径只属于所选后端，不能直接套到 Triton-Ascend。

## 1. 一个 Program 处理哪些元素

kernel 的核心为：

```python
@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    a = tl.load(x + i, valid, other=0.0)
    b = tl.load(y + i, valid, other=0.0)
    tl.store(out + i, a + b, valid)
```

取 n=10、BLOCK=4，三个 program 负责 [0,4)、[4,8)、[8,12)。最后两个位置不读写数组。`i` 是一块索引，a/b 是一块值；Python 源码没有手写每个 GPU thread 负责哪个位置。

这里 BLOCK 是 program 的逻辑块大小，不等于线程数。选择 1024 个元素也不意味着一定启动 1024 个 thread。下游还需决定一块值怎样分给线程和每线程寄存器。

## 2. TTIR 保留的地址与数据流

前端生成的 TTIR 使用 `tt.get_program_id`、`tt.make_range`、指针广播与 `tt.addptr` 等操作表达索引和地址，再由 `tt.load/store` 读写。中间加法可以仍使用 Arith 操作，因此 TTIR 也是多个方言共同组成的表示。

把核心结构压缩表示为：

```text
program_id × BLOCK + range(0,BLOCK)
              ↓
一块指针 + 一块有效性谓词
              ↓ tt.load
一块 f32 值 → arith.addf → tt.store
```

一块指针与“指向一个完整 tensor 的指针”是不同的表示。固定 LoadOp 定义能覆盖相应变体，而 NVIDIA 的后期 LoadOpConversion 要求 tensor pointer 形式已被较早阶段改写。这个前置条件说明 Pass 顺序为何影响规则能否成立。

在 `TritonOps.td` 中，load 的 ptr、可选 mask、可选 other 具有对应的 shape/encoding 约束和类型推导。对本例，mask=false 时不访问该元素的地址，并由 other 提供后续计算使用的值；store 的 mask 再阻止无效结果写回。

## 3. TTGIR 增加元素分布

进入 TritonGPU 后，Tensor encoding 描述逻辑元素分给哪些 lane、warp 和线程内位置。一个一维 blocked encoding 的示意是：

```text
#triton_gpu.blocked<{
  sizePerThread = [1], threadsPerWarp = [32],
  warpsPerCTA = [4], order = [0]
}>
```

它的基本分布块含 1×32×4=128 个元素。若逻辑张量有 1024 个元素，需要按布局继续覆盖多个这样的块；不是仅表示前 128 个元素。对于这一简单一维分布，可将线程 t 的元素理解为 t、t+128、t+256……。

改变 sizePerThread 或 order 会改变相邻逻辑元素由谁持有。数学张量仍可保持相同 shape，分布却已经不同。这正对应 [MLIR 设备布局章节](/notes/compile/mlir/compiler/targets/layout_transfer)中“存储地址与线程分工是两层关系”的区别。

这里的值 tensor 不等于必须分配一份同样大小的 global buffer。后续可以把每个线程负责的部分拆成标量/向量寄存器表示；通信需求由布局转换和具体消费者决定。

## 4. Coalesce 怎样消费布局信息

假设相邻 lane 读连续地址，后端更有机会使用适合目标的合并访问；若相邻 lane 的地址跨越大步长，则可能需要改变分布或访问组织。

真实的 coalesce Pass 会分析访问关系并选择布局。上游转置测试明确检查：load 侧与 store 侧选取不同 blocked layout，指针、mask、other/value 随之转换。这说明布局变化必须同时作用于同一访问的相关值，不能只改结果 Tensor 的属性。

`convert_layout` 也不意味着总要通过 shared memory。某些变化只是线程内重新排列，另一些需要线程间通信；最终如何实现由两端布局和目标决定。需要进一步看 lowering，不能仅根据高层操作计数判断成本。

## 5. LoadOp 到目标读取

NVIDIA 后端的 `LoadOpConversion` 取得 adaptor 中已转换的指针、mask 和 other，将它们拆成当前线程负责的元素。接着决定一次指令可处理多少元素。

这项决定不只看张量总大小：实现会参考地址连续/对齐信息，并用 mask 的对齐信息限制向量宽度。若相邻元素的 mask 不能作为同一组处理，就不能任意把它们合成一条带单一谓词的访问。

随后规则构造带 predicate 的 PTX load，把结果重新组织成下游值。这个版本用 PTXBuilder 生成相应指令路径，因此最终 LLVM IR 中可以包含目标 inline assembly，而不必让每条读取都表现为普通 LLVM load。

到这里，最初的 `tl.load(x+i,valid)` 已落实为：某个线程负责的地址、目标可用的访问宽度、实际谓词和返回值。前面学过的 ODS、adaptor、TypeConverter、分析与 Pattern，分别服务于这条具体变化。

## 6. 真实编译阶段与公共 MLIR 机制

`third_party/nvidia/backend/compiler.py` 的 `add_stages` 注册：

```text
ttir → ttgir → llir → ptx → cubin
```

`make_ttir` 组织 inlining、组合、canonicalize、CSE、LICM 与展开等。`make_ttgir` 先转 TritonGPU，再做 coalesce、布局处理、matmul/流水等目标优化；具体项目能力和选项会选择不同分支，并非加法都会发生所有变换。

`make_llir` 内部先把 TritonGPU 及控制流等转成 LLVM 兼容的 MLIR，再 translation 到 LLVM IR，附加目标布局、链接所需外部库并优化。因此阶段名 llir 隐藏了“LLVM dialect 与 LLVM IR”之间的一次表示转换。

最后由 NVPTX 生成 PTX，再调用 ptxas 形成 cubin。编译器还保存 kernel 名称、共享存储等 metadata，运行时据此加载和启动；设备对象只是完整调用路径的一部分。

## 7. 源码证据与复现边界

本章固定提交 `9641643da6c52000c807b5eeed05edaec4402a67`，仓库记录的 LLVM hash 为 `86b69c31642e98f8357df62c09d118ad1da4e16a`。与标准 MLIR 20.1.8 实验共用概念，不意味着工具二进制或语法可以混用。

当前核对了源码、布局定义与上游 FileCheck 期望，未构建/运行这个 Triton 包，也未将标准 GPU 实验生成的 PTX 冒充 Triton 产物。部署后的 IR、编译缓存、设备数值与性能应在兼容环境中另行记录。

有限阅读顺序：[加法前端示例](https://github.com/triton-lang/triton/blob/v3.2.0/python/tutorials/01-vector-add.py) → [LoadOp 定义](https://github.com/triton-lang/triton/blob/v3.2.0/include/triton/Dialect/Triton/IR/TritonOps.td) → [blocked encoding](https://github.com/triton-lang/triton/blob/v3.2.0/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td) → [coalesce 测试](https://github.com/triton-lang/triton/blob/v3.2.0/test/TritonGPU/coalesce.mlir) → [LoadOp lowering](https://github.com/triton-lang/triton/blob/v3.2.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp) → [后端 pipeline](https://github.com/triton-lang/triton/blob/v3.2.0/third_party/nvidia/backend/compiler.py)。

读完后可以用自己的话解释三项决定：BLOCK 决定哪组逻辑元素；encoding 决定谁持有元素；load lowering 决定它们怎样成为实际读取。随后再选 Matmul 的 dot 或共享布局转换深入，比立即通读整个编译器更容易形成可检查的理解。
