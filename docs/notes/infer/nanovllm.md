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

## 3. Engine 
### 3.1 engine核心组件
`LLMEngine` 是推理主循环的总入口，负责接收 prompts、创建 `Sequence`、循环调用 `step()`，并在请求完成后收集输出。

`Sequence` 表示一条请求的运行时状态：
- `token_ids`：prompt token + 已生成 token
- `num_prompt_tokens`：原始 prompt 长度
- `num_cached_tokens`：已经写入 KV cache、不需要重新 prefill 的 token 数
- `num_scheduled_tokens`：本轮计划送进模型的 token 数
- `block_table`：这条请求的逻辑 block 到物理 KV block 的映射
- `status`：`WAITING` / `RUNNING` / `FINISHED`

`Scheduler` 决定每个 `step()` 跑哪些 `Sequence`，以及本轮是 `prefill` 还是 `decode`。它维护两条队列：
- `waiting`：还没完成 prefill，或者被抢占后需要重新 prefill 的请求
- `running`：已经完成 prefill，正在逐 token decode 的请求

`BlockManager` 负责 KV cache 的 block 级资源管理：
- prefill 前分配 `block_table`
- 用链式 hash 支持 prefix cache
- decode 时必要地追加新 block
- 请求结束或被抢占时回收 block

`ModelRunner` 负责把 `Sequence` 状态组织成 GPU 输入，设置 attention 执行上下文，然后执行模型前向和采样。

### 3.2 Generate 主循环

一个 `step()` 只表示“一次模型 batch 调用”，不表示处理完所有请求。每一轮都会重新调度，所以这是 iteration-level scheduling。**并且batch是做成一维的表示，再用 cu_seqlens_q / cu_seqlens_k 记录每条 seq 在这个扁平数组里的边界**。varlen版本的flash attention支持同一个batch里不同seq长度不同，且不需要padding

### 3.3 Scheduler 调度规则
`schedule()` 先尝试构造 prefill batch。如果本轮有任何 `waiting` 请求能被调度，就直接返回 `(scheduled_seqs, True)`，不会再混入 decode。

prefill 阶段从 `waiting[0]` 开始，受两个预算限制：
- `max_num_seqs`：本轮最多调度多少条请求
- `max_num_batched_tokens`：本轮 prefill 最多处理多少 token

对每个被调度的 seq：
1. 计算还需要 prefill 的 token 数：`seq.num_tokens - seq.num_cached_tokens`
2. 如果还没有 `block_table`，调用 `BlockManager.allocate(seq)` 分配 KV blocks
3. 设置 `seq.num_scheduled_tokens`
4. 如果本轮已经把该 seq 的 prefill 全部跑完，就把它从 `waiting` 移到 `running`
5. 如果是 chunked prefill 未完成，该 seq 留在 `waiting` 队首，下轮继续处理

只有当本轮没有任何 prefill 可做时，才进入 decode 阶段。decode 阶段从 `running` 队列中取若干条 seq，**每条 seq 本轮只处理 1 个 token**：

```python
seq.num_scheduled_tokens = 1
```

如果 KV cache 空间不足，scheduler 会调用 `preempt()` 抢占某条 running seq：释放它的 KV blocks，把它放回 `waiting` 队首，之后通过重新 prefill 恢复 KV cache，被抢占的seq，scheduler虽然会优先调度waiting队列，但是必须要等`can_allocate == False`时才会被调度到。

### 3.4 ModelRunner 执行规则
prefill 和 decode 的并行都不是 Python 线程并行，而是 GPU batch 并行。

prefill 时，`prepare_prefill(seqs)` 会把多个 seq 本轮要处理的 token 展平成一个 `input_ids`，并通过 `cu_seqlens_q/cu_seqlens_k` 告诉 FlashAttention 每条序列的边界。`slot_mapping` 告诉 attention：本轮新算出的 K/V 应该写入 KV cache 的哪个物理 slot。prefill阶段也是多个seq按batch并行，但是只有waiting队首的seq可以分chunk，其他的seq必须完整进入batch或等下一轮，这样主要是实现比较容易，它主要是为“队首这条 seq 太长，一轮根本放不下”这种情况兜底，但是token budget可能没有被充分利用。


decode 时，`prepare_decode(seqs)` 每条 seq 只取 `last_token`，把多条请求的最后一个 token 拼成 batch。attention 通过 `block_tables` 读取每条 seq 的历史 KV cache，再为当前 token 计算 K/V 并写入 cache，最后输出下一 token 的 logits。

需要注意：chunked prefill 未完成时，模型仍会 forward，也会算 logits 和采样，但 `postprocess()` 会丢弃这个采样结果，因为此时真正需要的是把当前 chunk 的 K/V 写入 cache。只有完整 prefill 完成后，采样 token 才会被 append 到 `seq.token_ids`。

### 3.5 Postprocess 状态更新
`postprocess()` 根据 `is_prefill` 和 seq 状态更新请求：

- prefill chunk 未完成：只更新 `num_cached_tokens`，不 append 新 token
- 抢占后的 re-prefill：重建 KV cache，不 append 新 token
- 正常 prefill 完成：append 第一个生成 token
- decode 完成：append 本轮生成 token
- 遇到 EOS 或达到 `max_tokens`：标记 `FINISHED`，释放 KV blocks

### 3.6 完整流程图
```mermaid
flowchart TD
    A[example.py 创建 LLM 和 SamplingParams] --> B[LLM.generate(prompts, sampling_params)]
    B --> C[LLMEngine.add_request]
    C --> D[Tokenizer 编码 prompt]
    D --> E[创建 Sequence]
    E --> F[加入 Scheduler.waiting 队列]

    F --> G{所有请求是否结束}
    G -- 否 --> H[LLMEngine.step]
    G -- 是 --> Z[收集 completion_token_ids 并 decode 输出]

    H --> I[Scheduler.schedule]
    I --> J{waiting 队列是否有可调度 seq}

    J -- 是 --> K[构造 prefill batch]
    K --> K1[取 waiting 队首 seq]
    K1 --> K2[计算未缓存 token 数]
    K2 --> K3{是否已有 block_table}
    K3 -- 否 --> K4[BlockManager.can_allocate / allocate]
    K3 -- 是 --> K5[复用已有 block_table]
    K4 --> K6[设置 num_scheduled_tokens]
    K5 --> K6
    K6 --> K7{本 seq prefill 是否完成}
    K7 -- 是 --> K8[seq: WAITING -> RUNNING，移入 running]
    K7 -- 否 --> K9[chunked prefill，留在 waiting 队首]
    K8 --> M[ModelRunner.run is_prefill=True]
    K9 --> M

    J -- 否 --> L[构造 decode batch]
    L --> L1[从 running 左侧取 seq]
    L1 --> L2{BlockManager.can_append}
    L2 -- 否 --> L3[preempt 其他 running seq]
    L3 --> L4[释放被抢占 seq 的 KV blocks]
    L4 --> L5[被抢占 seq 放回 waiting 队首]
    L5 --> L2
    L2 -- 是 --> L6[BlockManager.may_append]
    L6 --> L7[设置 num_scheduled_tokens = 1]
    L7 --> N[ModelRunner.run is_prefill=False]

    M --> M1[prepare_prefill]
    M1 --> M2[拼接 input_ids / positions]
    M2 --> M3[构造 cu_seqlens 和 slot_mapping]
    M3 --> M4{是否需要 prefix cache}
    M4 -- 是 --> M5[准备 block_tables]
    M4 -- 否 --> M6[set_context is_prefill=True]
    M5 --> M6
    M6 --> O[模型 forward]

    N --> N1[prepare_decode]
    N1 --> N2[每个 seq 取 last_token]
    N2 --> N3[构造 context_lens / slot_mapping / block_tables]
    N3 --> N4[set_context is_prefill=False]
    N4 --> O

    O --> P[Attention 读取 context]
    P --> P1[store_kvcache 写入当前 K/V]
    P1 --> P2{prefill or decode}
    P2 -- prefill --> P3[flash_attn_varlen_func]
    P2 -- decode --> P4[flash_attn_with_kvcache]
    P3 --> Q[compute_logits]
    P4 --> Q
    Q --> R[Sampler 采样 token_ids]
    R --> S[Scheduler.postprocess]

    S --> T{is_prefill}
    T -- 是 --> U[更新 num_cached_tokens]
    U --> U1{chunked prefill 未完成 或 re-prefill}
    U1 -- 是 --> U2[丢弃采样 token，清空 num_scheduled_tokens]
    U1 -- 否 --> V[append_token]
    T -- 否 --> V

    V --> W[更新 num_cached_tokens 和 last_token]
    W --> X{EOS 或达到 max_tokens}
    X -- 是 --> Y[标记 FINISHED; BlockManager.deallocate; 从 running 移除]
    X -- 否 --> Y1[保持 RUNNING; 等待下一轮 step]

    U2 --> G
    Y --> G
    Y1 --> G
```
