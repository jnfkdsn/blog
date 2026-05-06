---
order: 2
---

## 量化推理

### 1.weight-only quantization
只量化权重，激活值保持FP16
```
推理时计算流程：
  INT8/INT4 权重 (存在 HBM)
       ↓ load
  Dequant: ŵ = scale × (w_int - zero_point)    ← 在 SRAM/寄存器中做
       ↓
  FP16 GEMM:  output = activation_fp16 × ŵ_fp16
       ↓
  FP16 输出
```
- Decode 阶段的 GEMM 形状：$[1, d] \times [d, d]$（batch=1 时）
- 是严重的 **memory-bound**：算术强度 $\approx 2$，远低于 GPU 的 Roofline 拐点
- 瓶颈在于**读取权重的带宽**，INT4 权重 = 读取量减少 4 倍 → 速度接近 4 倍

#### W8A16：INT8权重+FP16激活
量化过程
[-128, +127] (INT8) 
```python
import torch

def quantize_w8a16(weight_fp16: torch.Tensor):
    """Per-channel 对称 INT8 量化"""
    # weight_fp16: [out_features, in_features]
    # 每行独立计算 scale
    amax = weight_fp16.abs().amax(dim=1, keepdim=True)  # [out, 1]
    scale = amax / 127.0                                # [out, 1] 
    # 量化
    w_int8 = torch.round(weight_fp16 / scale).clamp(-128, 127).to(torch.int8)
    # scale 存为 FP16，后续 dequant 
    return w_int8, scale.half().squeeze()  # w_int8: [out, in], scale: [out]
```

推理时：triton dequant+GEMM Fused kernel
优化：不要先dequant整个矩阵再做GEMM,而是在GEMM 的 K 循环内每加载一个 tile 的 INT8 权重，立即转换为 FP16 并参与计算。

#### GPTQ       