---
order: 1
---

## RMSNorm
$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot g
$$

代码如下：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

## RoPE
$$\text{RoPE}(x, pos) = \begin{pmatrix} x_0 \cos\theta_0 - x_1 \sin\theta_0 \\ x_0 \sin\theta_0 + x_1 \cos\theta_0 \\ x_2 \cos\theta_1 - x_3 \sin\theta_1 \\ x_2 \sin\theta_1 + x_3 \cos\theta_1 \\ \vdots \end{pmatrix}$$

每两个相邻维度视为一个二维平面，按角度 $\theta_i$ 旋转。$\theta_i$ 由位置 $pos$ 和维度 $i$ 决定：

$$\theta_i = pos \cdot \frac{1}{10000^{2i/d}}$$

代码如下：
```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device=None, dtype=None):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    return q, k
```

## kv cache
每个attettion层都有自己的kvcache，为什么第 2 层的输入不是会受新 token 影响吗，所以旧 token 的第 2 层表示会不会变？
设某一层的输入是
```math
X = [x_1, x_2, \dots, x_T]^\top \in \mathbb{R}^{T \times d}
```
先做线性投影得到
```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```
其中第 `i` 个位置对应
```math
q_i = x_i W_Q,\quad k_i = x_i W_K,\quad v_i = x_i W_V
```
---

**没有 mask 的 attention**

第 `i` 个位置对第 `j` 个位置的打分是：

```math
s_{ij} = \frac{q_i k_j^\top}{\sqrt{d_k}}
```

然后对第 `i` 行做 softmax：

```math
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{m=1}^T \exp(s_{im})}
```

最后输出：

```math
o_i = \sum_{j=1}^T \alpha_{ij} v_j
```

这时 `i` 可以看到所有 `j=1...T`，包括未来位置。

---

**加入 causal mask**

causal mask 定义成一个矩阵 `M`：

```math
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
```

也就是：

- 当前位置 `i` 可以看自己和左边
- 不能看右边未来 token

于是 attention 分数变成：

```math
\tilde{s}_{ij} = \frac{q_i k_j^\top}{\sqrt{d_k}} + M_{ij}
```

再做 softmax：

```math
\alpha_{ij} = \frac{\exp(\tilde{s}_{ij})}{\sum_{m=1}^T \exp(\tilde{s}_{im})}
```

因为当 `j > i` 时，`M_{ij} = -\infty`，所以：

```math
\exp(\tilde{s}_{ij}) = 0
```

于是：

```math
\alpha_{ij} = 0,\quad \forall j > i
```

所以输出就变成：

```math
o_i = \sum_{j=1}^i \alpha_{ij} v_j
```

**第 `i` 个位置的输出只依赖 `1...i`，不依赖 `i+1...T`。**


- 新 token 不会影响旧位置第 1 层的输出
- 第 1 层输出不变，则第 2 层输入不变
- 第 2 层也有同样的 causal mask，所以第 2 层旧位置输出也不变
- 层层递推，所有层都成立

所以旧 token 在所有层的 `K/V` 都可以缓存。
整体写成矩阵就是：

```math
\text{Attention}(Q,K,V) =
\text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
```
