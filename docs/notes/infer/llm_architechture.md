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
