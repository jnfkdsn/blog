---
order: 3
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

先固定某一行 \(q\)。设它对所有 key 的打分是 \(s_t\)，对应 value 是 \(v_t\)。

处理到第 \(j\) 个 tile 后，定义：

$$
m^{(j)}=\max_{t\le j\text{ tiles}} s_t
$$

$$
\ell^{(j)}=\sum_{t\le j\text{ tiles}} e^{s_t-m^{(j)}}
$$

$$
o^{(j)}=\frac{1}{\ell^{(j)}}\sum_{t\le j\text{ tiles}} e^{s_t-m^{(j)}}v_t
$$

这就是“当前已经处理部分”的 softmax 输出。


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

其中 O 修正项可由“旧贡献 + 新贡献”直接得到：

$$\sum_{\text{old}} e^{s-m_i^{(j)}}V = e^{m_i^{(j-1)}-m_i^{(j)}} \sum_{\text{old}} e^{s-m_i^{(j-1)}}V = e^{m_i^{(j-1)}-m_i^{(j)}}\,\ell_i^{(j-1)}\,O_i^{(j-1)}$$

$$\sum_{\text{new}} e^{s-m_i^{(j)}}V = \tilde{P}_{ij}V_j$$

$$O_i^{(j)} = \frac{\sum_{\text{old}} e^{s-m_i^{(j)}}V + \sum_{\text{new}} e^{s-m_i^{(j)}}V}{\ell_i^{(j)}}$$

也就是说：旧块先按 $e^{m_i^{(j-1)}-m_i^{(j)}}$ 缩放到新坐标系，再和当前 tile 贡献合并，最后除以新的 $\ell_i^{(j)}$。
**循环结束后**，$O_i^{(T_{kv})}$ 就是最终结果。


### 内存分析

```
标准 Attention:
  额外显存: O(N²) — 存 S 和 P
  HBM 读写: O(N² d) — 多次读写 S, P

FlashAttention:
  额外显存: O(N) — 只存 m 和 ℓ（每行一个 max 和 sum）
  HBM 读写: O(N² d² / SRAM_size) — Q/K/V 各被读 O(N d / SRAM_size) 次

时间复杂度： O(N^2 d)

空间复杂度：
sram需要存储：Q,K,V,O,score，max和sum向量
HBM只需要存储输出O,O(Nd)
```


## flash attention v2

v2主要在v1的基础上做了三个改进
### 1.减少非计算FLOPs

定义未归一化的累积量
$$
U_i^{(j)}=\sum_{t\le j} e^{s_t-m_i^{(j)}}V_t
$$
并递推
$$
U_i^{(j)}=U_i^{(j-1)}e^{m_i^{(j-1)}-m_i^{(j)}}+\tilde P_{ij}V_j
$$
$$
\ell_i^{(j)}=\ell_i^{(j-1)}e^{m_i^{(j-1)}-m_i^{(j)}}+\mathrm{rowsum}(\tilde P_{ij})
$$
最后再做
$$
O_i^{(T_{kv})}= \frac{U_i^{(T_{kv})}}{\ell_i^{(T_{kv})}}.
$$

可以减少计算量
```python
# V1: 每步都做归一化修正
O_i = O_i * (l_old * alpha / l_new) + P_ij @ V_j / l_new
```
```python
# V2: 循环内不归一化，只做乘加
O_i = O_i * alpha[:, None] + P_ij @ V_j  # 没有除以 l_new
# 循环结束后
O_i = O_i / l_i[:, None]                 # 最后统一归一化
```

### 2.交换循环顺序
v1的循环结构是：**外层遍历Q tile，内层遍历 KV tile**

v2分析了两种循环在不同场景下的优劣：
- 外层Q,内层KV:每个 program 独立处理一个 Q tile，不需要跨 program 通信。但 K/V 被重复从 HBM 读取
- **外层 KV，内层 Q**：每个 KV tile 只读一次，Q tile 在循环中流过。但 O 的更新需要跨 program 通信（写冲突）。

前向过程适合外层Q，反向过程适合内层Q：
前向计算时，每个q_i对应一整行attention，一个 Q 块在扫描 K/V 的过程中，只需要维护自己的m，l，o，这些状态是按Q行独立的，因此如果外层固定 Q 块，内层扫描 K/V 块，就可以把这个 Q 块的中间状态一直留在 SRAM/寄存器里，直到它的输出完成。
反向过程已知dO,要求dQ,dK,dV,
```
前向：
O = PV
P = softmax(S)
S = QK^T/sqrt(d)

反向：
dV = P^TdO
dP = dO V^T
逐元素写
dV_j = sum_i p_i,j dO_i
dP_i,j = dO_i · v_j
对每一行
p_i = softmax(s_i)
softmax的反向为：
dS_i,j = p_i,j * (dP_i,j - sum_t p_i,t dP_i,t)

sum_t p_i,t dP_i,t
= sum_t p_i,t (dO_i · v_t)
= dO_i · sum_t p_i,t v_t
= dO_i · O_i

令D_i = dO_i · O_i
则dS_i,j = p_i,j * (dO_i · v_j - D_i)

有S = QK^T / sqrt(d)
dQ_i = sum_j dS_i,j k_j / sqrt(d)
dK_j = sum_i dS_i,j q_i / sqrt(d)
dV_j = sum_i p_i,j dO_i

```
dK和dV都是固定一个 K/V 位置 j，对所有 Q 行 i 求和，所以适合内层Q

### 3.warp分工优化
V1 中每个 warp 都参与所有计算（GEMM 和 softmax）。V2 让不同 warp 做不同的工作：

```
V1: 所有 4 个 warp 一起做 QK^T， softmax，PV
    → warp 间需要大量同步

V2: 将 BLOCK_M 分配给不同 warp，每个 warp 独立处理几行
    → 减少 warp 间的同步和通信

    Warp 0: 处理 Q 的第 0-15 行
    Warp 1: 处理 Q 的第 16-31 行
    Warp 2: 处理 Q 的第 32-47 行
    Warp 3: 处理 Q 的第 48-63 行
```

这在 Triton 中通过调整 `BLOCK_M` 和 `num_warps` 的比例来实现。



## 附录：反向传播公式推导
设最终 loss 为 $L$，上游梯度为：

$$
G = \frac{\partial L}{\partial O} = dO
$$

其中：

$$
Q,K,V,O \in \mathbb{R}^{N \times d}, \quad S,P \in \mathbb{R}^{N \times N}
$$

前向过程为：

$$
S = \frac{QK^T}{\sqrt d}
$$

$$
P = \mathrm{softmax}(S)
$$

$$
O = PV
$$

反向需要依次求：

$$
dV,\ dP,\ dS,\ dQ,\ dK
$$



### 1. $O = PV$ 的反向

对 $O = PV$ 求微分：

$$
dO = dP \cdot V + P \cdot dV
$$

根据标量 loss 的微分定义：

$$
dL = \mathrm{tr}(G^T dO)
$$

代入 $dO$：

$$
dL
= \mathrm{tr}(G^T dP V) + \mathrm{tr}(G^T P dV)
$$

先看第一项：

$$
\mathrm{tr}(G^T dP V)
= \mathrm{tr}(V G^T dP)
= \mathrm{tr}((G V^T)^T dP)
$$

所以：

$$
dP = G V^T
$$

再看第二项：

$$
\mathrm{tr}(G^T P dV)
= \mathrm{tr}((P^T G)^T dV)
$$

所以：

$$
dV = P^T G
$$

也就是：

$$
dP = dO \cdot V^T
$$

$$
dV = P^T \cdot dO
$$

逐元素写为：

$$
dP_{ij} = dO_i \cdot v_j
$$

$$
dV_j = \sum_i P_{ij} dO_i
$$

其中 $dO_i$ 和 $v_j$ 都是长度为 $d$ 的向量。

### 2. $P = \mathrm{softmax}(S)$ 的反向

softmax 是逐行计算的，所以对第 $i$ 行单独推导。

设：

$$
p_i = \mathrm{softmax}(s_i)
$$

其中：

$$
p_{ij} = \frac{e^{s_{ij}}}{\sum_t e^{s_{it}}}
$$

softmax 的 Jacobian 为：

$$
\frac{\partial p_{ij}}{\partial s_{ik}}
= p_{ij}(\delta_{jk} - p_{ik})
$$

其中 $\delta_{jk}$ 是 Kronecker delta。当 $j=k$ 时为 1，否则为 0。

对第 $i$ 行，有：

$$
dS_{ik}
= \sum_j dP_{ij} \frac{\partial p_{ij}}{\partial s_{ik}}
$$

代入 softmax Jacobian：

$$
dS_{ik}
= \sum_j dP_{ij} p_{ij}(\delta_{jk} - p_{ik})
$$

拆开两项：

$$
dS_{ik}
= dP_{ik}p_{ik} - p_{ik}\sum_j dP_{ij}p_{ij}
$$

提取 $p_{ik}$：

$$
dS_{ik}
= p_{ik}\left(dP_{ik} - \sum_j p_{ij}dP_{ij}\right)
$$

换回常用下标 $j$：

$$
dS_{ij}
= P_{ij}\left(dP_{ij} - \sum_t P_{it}dP_{it}\right)
$$

写成矩阵形式：

$$
dS = P \odot \left(dP - D\right)
$$

其中 $\odot$ 表示逐元素乘法，$D$ 是把每一行的内积广播到整行的矩阵：

$$
D_i = \sum_t P_{it}dP_{it}
$$

由于：

$$
dP_{it} = dO_i \cdot v_t
$$

所以：

$$
D_i
= \sum_t P_{it}(dO_i \cdot v_t)
$$

把 $dO_i$ 提出来：

$$
D_i
= dO_i \cdot \sum_t P_{it}v_t
$$

而前向输出：

$$
O_i = \sum_t P_{it}v_t
$$

因此：

$$
D_i = dO_i \cdot O_i
$$

所以 softmax 反向最终可以写成：

$$
dS_{ij}
= P_{ij}(dO_i \cdot v_j - dO_i \cdot O_i)
$$

也就是：

$$
dS = P \odot (dO V^T - D)
$$

其中：

$$
D = \mathrm{rowsum}(dO \odot O)
$$

并且 $D$ 在列方向广播。

### 3. $S = QK^T / \sqrt d$ 的反向

对：

$$
S = \frac{QK^T}{\sqrt d}
$$

求微分：

$$
dS = \frac{dQK^T + QdK^T}{\sqrt d}
$$

根据：

$$
dL = \mathrm{tr}(dS_{\mathrm{grad}}^T dS)
$$

这里把上一步得到的梯度记为 $dS_{\mathrm{grad}}$，为了避免和微分符号混淆。

代入：

$$
dL
= \mathrm{tr}\left(dS_{\mathrm{grad}}^T \frac{dQK^T + QdK^T}{\sqrt d}\right)
$$

拆成两项：

$$
dL
= \frac{1}{\sqrt d}\mathrm{tr}(dS_{\mathrm{grad}}^T dQK^T)
+ \frac{1}{\sqrt d}\mathrm{tr}(dS_{\mathrm{grad}}^T QdK^T)
$$

第一项：

$$
\mathrm{tr}(dS_{\mathrm{grad}}^T dQK^T)
= \mathrm{tr}(K^T dS_{\mathrm{grad}}^T dQ)
= \mathrm{tr}((dS_{\mathrm{grad}}K)^T dQ)
$$

所以：

$$
dQ = \frac{dS_{\mathrm{grad}}K}{\sqrt d}
$$

第二项：

$$
\mathrm{tr}(dS_{\mathrm{grad}}^T QdK^T)
= \mathrm{tr}(dK^T dS_{\mathrm{grad}}^T Q)
= \mathrm{tr}((dS_{\mathrm{grad}}^T Q)^T dK)
$$

所以：

$$
dK = \frac{dS_{\mathrm{grad}}^T Q}{\sqrt d}
$$

最终反向公式为：

$$
dV = P^T dO
$$

$$
dP = dO V^T
$$

$$
D_i = dO_i \cdot O_i
$$

$$
dS_{ij} = P_{ij}(dP_{ij} - D_i)
$$

$$
dQ = \frac{dS K}{\sqrt d}
$$

$$
dK = \frac{dS^T Q}{\sqrt d}
$$

逐元素形式为：

$$
dQ_i = \sum_j dS_{ij}k_j / \sqrt d
$$

$$
dK_j = \sum_i dS_{ij}q_i / \sqrt d
$$

$$
dV_j = \sum_i P_{ij}dO_i
$$

这三个式子正好解释了循环顺序：

- $dQ_i$ 固定 query 行 $i$，对所有 key/value 位置 $j$ 求和。
- $dK_j$ 固定 key 行 $j$，对所有 query 位置 $i$ 求和。
- $dV_j$ 固定 value 行 $j$，对所有 query 位置 $i$ 求和。

所以反向中如果要高效累加 $dK$ 和 $dV$，更自然的方式是固定一个 KV tile，把 Q tile 放在内层扫过来累加。


