---
order: 5
---
# nano-vllm

## 整体目录结构
```
nanovllm/
├── __init__.py          # 入口，暴露 LLM 和 SamplingParams
├── config.py            # 全局配置数据类
├── sampling_params.py   # 采样参数定义
├── llm.py               # LLM 入口类（继承自 LLMEngine）
├── engine/              # 推理引擎核心模块
│   ├── llm_engine.py    # LLMEngine：整个系统的总指挥
│   ├── scheduler.py     # Scheduler：请求调度与资源分配
│   ├── sequence.py      # Sequence：单个推理请求的数据结构
│   ├── block_manager.py # BlockManager：PagedAttention KV 缓存管理
│   └── model_runner.py  # ModelRunner：GPU 模型执行引擎
├── models/              # 模型架构实现
│   ├── __init__.py
│   ├── model_registry.py# 模型注册系统
│   ├── qwen3.py         # Qwen3 模型实现
│   └── cpm4.py          # MiniCPM/CPM4 模型实现
├── layers/              # Transformer 核心层实现
│   ├── attention.py     # 注意力层（FlashAttention + GQA）
│   ├── linear.py        # 张量并行线性层
│   ├── rotary_embedding.py # 旋转位置编码
│   ├── layernorm.py     # RMSNorm 层（支持残差融合）
│   ├── embed_head.py    # 词嵌入与 LM 头
│   ├── sampler.py       # Token 采样器
│   └── activation.py    # SiLU 激活函数
└── utils/               # 工具函数
    ├── context.py       # 全局执行上下文管理
    └── loader.py        # 模型权重加载器
```