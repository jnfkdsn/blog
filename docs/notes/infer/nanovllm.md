---
order: 5
---
# nano-vllm

## 1. 整体目录结构
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

## 2. 入口层(根目录文件)
config.py：定义 Config 数据类，包含所有推理配置参数（模型路径、张量并行大小、KV 缓存块大小等），并进行参数校验
sampling_params.py：定义 SamplingParams 类，包含温度、top_p、top_k、max_tokens 等采样相关参数
llm.py：继承自 LLMEngine

## 3. engine/
### 3.1 llm_engine
- 初始化所有子组件：sequence，modelrunner，scheduler，tokenizer
- add_request：接收用户请求，创建 Sequence 对象并加入等待队列
- step：
    从调度器里取出这一轮要处理的序列seqs，调用模型执行前向推理并返回结果
- generate:
    调度 → 执行 → 采样 → 更新序列状态
    处理生成完成的请求，返回结果

### 3.2 Scheduler
维护两条队列 waiting 和 running，每次 step() 决定本轮送哪些 Sequence 给模型、是跑 prefill 还是 decode，并配合 BlockManager 管理 KV cache block。

- schedule:
    - 先尝试 prefill，如果本轮有任何 prefill，就直接返回；只有没有 waiting 可跑时才进入 decode
    - prefill阶段：处理waiting[0]请求，分配KVCache block，chunked prefill，
    - decode阶段，确保kvcache有空追加一个新的token，如果没有空间则抢占其他的running seq，被抢占的序列会释放 KV cache，然后放回 waiting 队首，之后重新 prefill。最后把本轮 decode 的 seq 放回 running 队首，下轮可以继续生成。
    