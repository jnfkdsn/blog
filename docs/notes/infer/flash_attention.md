---
order: 2
---

# flash_attention 原理与实现

## attention的缺陷
传统实现如下
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$​（$N$ = 序列长度，$d$ = head dimension，通常 64 或 128）。


### 问题一：O(N^2)显存占用
pytorch 标准 eager 实现：
```python
def attention(Q,K,V):
    S = Q @ K.T / math.sqrt(d)        # S: [N, N]  ← 写入 HBM
    P = torch.softmax(S, dim=-1)      # P: [N, N]  ← 从 HBM 读 S，写入 HBM
    O = P @ V                          # O: [N, d]  ← 从 HBM 读 P，写入 HBM
    return O
```
需要先缓存分数矩阵S
矩阵 $S$ 和 $P$ 的形状是 $[N, N]$，其中 $N$ 是序列长度。
```
N = 2048:   S 大小 = 2048² × 2 bytes (FP16) = 8 MB     
N = 8192:   S 大小 = 8192² × 2 bytes = 128 MB            
N = 32768:  S 大小 = 32768² × 2 bytes = 2 GB              
N = 131072: S 大小 = 131072² × 2 bytes = 32 GB            
```
一个 32-head 的模型，一层 Attention 的 S 矩阵需要 `heads × N² × 2` 字节。

### 问题二：4次HBM读写

S写入
P读写
O写入

### 问题三：softmax瓶颈
Softmax 需要看到**整行**数据才能计算。标准实现中，必须先算完整个 $S = QK^T$，才能做 Softmax， $S$ 必须完整存在于显存中。

### roofline分析
```
RTX 3090:
  FP32 Peak Compute:       35.6 TFLOPS
  HBM Bandwidth:           936 GB/s     
  L2 Cache:                32 MB

Ridge Point = 35.6 TFLOPS / 936 GB/s ≈ 38 FLOPs/Byte
```
以N=2048,head_dim=64为例
```
数据传输量 (Bytes):
     Q+K+V+4*S = 33MB

计算量 (FLOPs):
    1. S=QK^T: FLOPs = 2 N^2 * d  
    2. P=SOFTMAX(S): FLOPs = 5N^2
    3. O=PV: FLOPs = 2 N^2 * d 
    FLOPs = 1GFLOPs
  

算术强度:
  AI = 1G / 33MB = 30 FLOPS/Byte
memory-bound
```

## flash_attention v1

flash attention核心思想是避免物化整个注意力矩阵，用 tiling + Online Softmax 在片上完成所有计算

### online softmax
核心思想：在遍历过程中动态维护(max,sum)对，当发现新的max时修正之前的sum
设：
- 处理到第 $j$ 个元素时，当前的最大值是 $m_j$，修正后的指数和是 $d_j$
- 来了第 $j+1$ 个元素 $x_{j+1}$

更新规则：
$$m_{j+1} = \max(m_j, x_{j+1})$$
$$d_{j+1} = d_j \cdot e^{m_j - m_{j+1}} + e^{x_{j+1} - m_{j+1}}$$
当max被超过时，之前的指数和需要通过乘e^{m_j - m_{j+1}}来修正

- 对于分块的softmax：

设向量 $X$ 被分为两块
$$X^{(1)} = [x_1, \dots, x_k], \quad X^{(2)} = [x_{k+1}, \dots, x_n]$$
分别记两块的统计量为：
$$m_1, d_1 \quad \text{和} \quad m_2, d_2$$
则合并后的统计量为：
$$m_{\text{new}} = \max(m_1, m_2)$$
$$d_{\text{new}} = d_1 \cdot e^{m_1 - m_{\text{new}}} + d_2 \cdot e^{m_2 - m_{\text{new}}}$$

```
FlashAttention：
1. 把 K, V 切成 tile
2. 逐 tile 计算局部 S_tile = Q_tile @ K_tile^T
3. 用 Online Softmax 更新 (max, sum, 局部 O)
4. 循环结束后得到最终 O

S 矩阵从未完整存在，每个 tile 的 S 只在 SRAM 中短暂存在
```

### 算法推导

设 Q 按行分成 $T_q$ 个 tile，K 和 V 被分成 $T_{kv}$ 个 tile。

对于 Q 的第 $i$ 个 tile $Q_i$（形状 $[B_r, d]$），我们需要计算：

$$O_i = \text{softmax}(Q_i K^T / \sqrt{d}) \cdot V$$

把 K 和 V 沿序列维度分 tile：$K_1, K_2, \ldots, K_{T_{kv}}$ 和 $V_1, V_2, \ldots, V_{T_{kv}}$。

**初始化**：

$$m_i^{(0)} = -\infty, \quad \ell_i^{(0)} = 0, \quad O_i^{(0)} = 0$$

**对每个 KV tile $j = 1, 2, \ldots, T_{kv}$**：

$$
S_{ij} = Q_i K_j^T / \sqrt{d}
\quad \text{（局部 score，形状 }[B_r, B_c]\text{）}
$$

$$\tilde{m}_{ij} = \text{rowmax}(S_{ij}) \quad \text{（当前 tile 的行最大值）}$$

$$m_i^{(j)} = \max(m_i^{(j-1)}, \tilde{m}_{ij}) \quad \text{（全局 max 更新）}$$

$$\tilde{P}_{ij} = \exp(S_{ij} - m_i^{(j)}) \quad \text{（当前 tile 的 exp）}$$

$$\ell_i^{(j)} = \ell_i^{(j-1)} \cdot \exp(m_i^{(j-1)} - m_i^{(j)}) + \text{rowsum}(\tilde{P}_{ij}) \quad \text{（全局 sum 更新）}$$

$$O_i^{(j)} = O_i^{(j-1)} \cdot \frac{\ell_i^{(j-1)} \cdot \exp(m_i^{(j-1)} - m_i^{(j)})}{\ell_i^{(j)}} + \frac{\tilde{P}_{ij}}{\ell_i^{(j)}} \cdot V_j \quad \text{（O 修正）}$$

**循环结束后**，$O_i^{(T_{kv})}$ 就是最终结果。

### 内存分析

```
标准 Attention:
  额外显存: O(N²) — 存 S 和 P
  HBM 读写: O(N² d) — 多次读写 S, P

FlashAttention:
  额外显存: O(N) — 只存 m 和 ℓ（每行一个 max 和 sum）
  HBM 读写: O(N² d² / SRAM_size) — Q/K/V 各被读 O(N d / SRAM_size) 次

  当 SRAM 足够大时，每个 Q/K/V 只被读一次或少数几次
  S 矩阵不需要完整写入 HBM
```