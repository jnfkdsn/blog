---
order: 4
title: Roofline 分析
updated: 2026-05-18
tags: [cuda, performance, roofline]
status: draft
---
# Roofline 分析

相关路线：[GPU 编程与算子优化知识地图](/notes/gpu-programming)  
相关实践：[Softmax 算子实现与优化](/posts/softmax) / [CUDA 矩阵乘法](/posts/GEMM)

### Roofline Model
Roofline Model 是判断一个 kernel **瓶颈在计算还是在访存**的核心工具。它用一张图回答一个问题：**我的 kernel 跑到了硬件极限的百分之多少？瓶颈是什么？**

```
             性能 (FLOPS)
                 │
  Peak Compute ──┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──────────────
                 │                                          /
                 │                                        /
                 │                                      /  ← Roofline
                 │                                    /
                 │                   ┌───────┐      /
                 │                   │ GEMM  │    /    compute-bound 区域
                 │                   └───────┘  /
                 │                            /
                 │                          /
                 │           ┌────────┐   /
                 │           │  Conv  │ /
                 │           └────────/
                 │    ┌──────────┐  /
                 │    │ Softmax  │/
                 │    └────────/─┘  memory-bound 区域
                 │           /
                 │         / ← 斜率 = Peak Memory Bandwidth
                 │       /
                 │     /
                 │   /
                 │ /
                 └──────────────────────────────────────── 算术强度
                              ↑                           (FLOPs/Byte)
                          Ridge Point
```

### 核心概念

**1. 算术强度 (Arithmetic Intensity, AI)**

$$\text{AI} = \frac{\text{FLOPs（总浮点运算次数）}}{\text{Bytes（总内存传输量）}}$$

单位是 FLOPs/Byte。"每搬运一字节数据，做了多少次运算"。

**2. 硬件上限 (Roofline 的两条线)**

GPU 有两个独立的硬件上限：
- **计算天花板**（水平线）：Peak Compute，如 RTX 3090 FP32 = 35.6 TFLOPS
- **带宽天花板**（斜线）：Peak BW × AI，如 RTX 3090 = 936 GB/s × AI

理论性能上限是两者的较小值：

$$\text{Attainable Performance} = \min\left(\text{Peak Compute}, \quad \text{Peak BW} \times \text{AI}\right)$$

**3. Ridge Point（脊点/拐点）**

两条线的交点，分界 memory-bound 和 compute-bound：

$$\text{Ridge Point} = \frac{\text{Peak Compute}}{\text{Peak BW}}$$

- AI < Ridge Point → **memory-bound**（计算单元在等数据）→ 优化访存
- AI > Ridge Point → **compute-bound**（带宽有余量）→ 优化计算

### 完整示例 Softmax

**Step 1：查硬件参数**

```
RTX 3090:
  FP32 Peak Compute:       35.6 TFLOPS
  GDDR6X Bandwidth:        936 GB/s
  L2 Cache:                6 MB

Ridge Point = 35.6 TFLOPS / 936 GB/s ≈ 38 FLOPs/Byte
```


**Step 2：算算术强度**

以 Softmax V3（[寄存器缓存版本](../../posts/softmax.md)），M=1024, N=4096 为例：

```
数据传输量 (Bytes):
  读 input:   M × N × 4 = 1024 × 4096 × 4 = 16 MB
  写 output:  M × N × 4 = 16 MB
  总计:       32 MB = 3.2 × 10^7 Bytes

计算量 (FLOPs):
  对每个元素:
    fmaxf 比较:   1 FLOP     (Step 1: 求 max)
    减法:         1 FLOP     (x[j] - max)
    exp:          ~1 FLOP    (特殊函数, SFU 执行, 按 1 FLOP 近似)
    加法:         1 FLOP     (累加到 sum)
    乘法:         1 FLOP     (× inv_sum)
  总计:  5 × M × N = 5 × 1024 × 4096 ≈ 2.10 × 10^7 FLOPs

算术强度:
  AI = 2.10 × 10^7 / 3.2 × 10^7 ≈ 0.66 FLOPs/Byte
```

这里把 `exp` 近似成 1 FLOP 只是为了做粗略 Roofline 定位。`exp`、`sin`、`cos` 这类特殊函数通常由 SFU 执行，真实延迟和吞吐不一定能被 FLOPs 完整表达。遇到这类 kernel，Roofline 只能先判断大方向，最终还要结合 Nsight Compute 的指令统计和 stall 原因。

**Step 3：画 Roofline 并定位**

```
  AI = 0.66,  Ridge Point = 38

  0.66 << 38 → 极度 memory-bound

  理论性能上限 = 936 GB/s × 0.66 = 617 GFLOPS
  理论最短时间 = 32 MB / 936 GB/s = 0.034 ms
  实测时间 = 0.04ms, 计算效率85%
```

> "给定一个 kernel，如何优化？"

```
1. 算 AI = FLOPs / Bytes
2. 查 Ridge Point = Peak Compute / Peak BW
3. 判断:
   AI < Ridge → memory-bound → 减少访存、fusion、向量化、缓存
   AI > Ridge → compute-bound → Tensor Core、tiling、减少冗余计算
4. 实测有效带宽或算力利用率 → 看达到理论上限的百分之多少
5. ncu profile → 找具体瓶颈 (bank conflict、occupancy、指令延迟)
```

### 用 ncu 自动画 Roofline

```bash
# ncu 加 --set roofline 会自动测量 kernel 并画 Roofline 图
ncu --set roofline -o softmax_roofline ./softmax_test
# 用 Nsight Compute GUI 打开 .ncu-rep 文件
```
