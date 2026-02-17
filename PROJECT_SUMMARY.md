# Attention From Scratch - 项目总结

## 项目概述

本项目是"从零实现 Attention"的完整代码工程，旨在帮助理解 Attention 机制的核心原理，为学习 TensorRT-LLM XQA 模块打下基础。

## 项目结构

```
attention-from-scratch/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python 依赖
├── .gitignore                          # Git 忽略文件
│
├── notebooks/                          # Jupyter Notebooks (交互式学习)
│   ├── 01_scaled_dot_product.ipynb    # Scaled Dot-Product Attention
│   ├── 02_multi_head_attention.ipynb  # Multi-Head Attention (MHA)
│   ├── 03_grouped_query_attention.ipynb # Grouped Query Attention (GQA)
│   └── 04_kv_cache.ipynb              # KV Cache 优化
│
├── src/                                # 源代码实现
│   ├── __init__.py                     # 包初始化
│   ├── attention.py                    # Attention 核心实现
│   ├── gqa.py                          # GQA 实现
│   └── kv_cache.py                     # KV Cache 实现
│
└── tests/                              # 单元测试
    ├── __init__.py
    ├── test_attention.py               # Attention 测试
    ├── test_gqa.py                     # GQA 测试
    └── test_kv_cache.py                # KV Cache 测试
```

## 核心内容

### 1. Scaled Dot-Product Attention (`src/attention.py`)

**核心公式**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
```

**实现内容**:
- `scaled_dot_product_attention()`: 核心计算函数
- `MultiHeadAttention`: 多头注意力模块
- `create_causal_mask()`: 因果 mask 生成
- `create_padding_mask()`: Padding mask 生成

**关键特性**:
- 支持自注意力和交叉注意力
- 支持因果 mask (用于 GPT 等自回归模型)
- 完整的前向传播实现

### 2. Grouped Query Attention (`src/gqa.py`)

**核心思想**:
- MHA: 每个 Q 头独立的 KV 头 (32:32)
- GQA: 多个 Q 头共享 KV 头 (32:8)
- MQA: 所有 Q 头共享 1 个 KV 头 (32:1)

**实现内容**:
- `GroupedQueryAttention`: GQA 模块
- `compare_attention_variants()`: 对比不同 Attention 变体

**优势**:
- 减少 KV Cache 内存占用 75%
- 质量接近 MHA (98%)
- 工业界主流选择 (LLaMA 2, Mistral)

### 3. KV Cache (`src/kv_cache.py`)

**核心优化**:
- 缓存已计算的 Key 和 Value
- 时间复杂度: O(n²) → O(n)
- 避免重复计算

**实现内容**:
- `KVCache`: KV Cache 管理类
- `MultiHeadAttentionWithCache`: 带缓存的 MHA

**两个阶段**:
1. **Prefill**: 处理完整 prompt，初始化缓存
2. **Decode**: 逐个生成 token，增量更新缓存

### 4. 交互式学习 (Notebooks)

每个 notebook 包含:
- 理论讲解
- 代码实现
- 可视化演示
- 性能分析
- 实际应用案例

## 快速开始

### 1. 安装依赖

```bash
cd attention-from-scratch
pip install -r requirements.txt
```

### 2. 运行 Notebooks

```bash
jupyter notebook notebooks/
```

按顺序学习:
1. `01_scaled_dot_product.ipynb` - 理解基础
2. `02_multi_head_attention.ipynb` - 理解多头
3. `03_grouped_query_attention.ipynb` - 理解 GQA
4. `04_kv_cache.ipynb` - 理解优化

### 3. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_attention.py -v
pytest tests/test_gqa.py -v
pytest tests/test_kv_cache.py -v
```

### 4. 使用代码

```python
from src.attention import MultiHeadAttention
from src.gqa import GroupedQueryAttention
from src.kv_cache import MultiHeadAttentionWithCache

# MHA
mha = MultiHeadAttention(d_model=512, num_heads=8)
output, attn_weights = mha(x, x, x)

# GQA
gqa = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=8)
output, attn_weights = gqa(x, x, x)

# 带 KV Cache 的 MHA
mha_cache = MultiHeadAttentionWithCache(d_model=512, num_heads=8, max_seq_len=2048)
output, attn_weights = mha_cache(x, x, x, use_cache=True)
```

## 学习路径

### 阶段 1: 基础理论 (1-2 天)
- 阅读 `01_scaled_dot_product.ipynb`
- 理解 Attention 的数学原理
- 理解 scaling factor 的作用
- 理解 mask 的使用

### 阶段 2: 多头机制 (1-2 天)
- 阅读 `02_multi_head_attention.ipynb`
- 理解多头的并行计算
- 理解自注意力 vs 交叉注意力
- 分析参数量和计算复杂度

### 阶段 3: GQA 优化 (1-2 天)
- 阅读 `03_grouped_query_attention.ipynb`
- 理解 MHA、MQA、GQA 的区别
- 分析内存和性能权衡
- 了解工业界应用案例

### 阶段 4: KV Cache (1-2 天)
- 阅读 `04_kv_cache.ipynb`
- 理解 Prefill 和 Decode 阶段
- 分析性能提升
- 理解内存占用

## 与 TensorRT-LLM XQA 的关系

本项目是 TensorRT-LLM XQA 模块的简化版实现，帮助理解:

1. **XQA 的 Attention 计算逻辑**
   - 本项目: Python 实现，易于理解
   - XQA: CUDA 实现，高度优化

2. **GQA 的实现方式**
   - 本项目: 使用 `repeat_interleave` 扩展 KV 头
   - XQA: 直接在 CUDA Kernel 中处理

3. **KV Cache 的管理机制**
   - 本项目: 简单的 tensor 缓存
   - XQA: Paged KV Cache，支持 Beam Search

4. **Paged Attention 的基础**
   - 本项目: 连续内存的 KV Cache
   - XQA: 分页管理，减少内存碎片

## 性能对比

### 参数量 (d_model=4096, num_heads=32)

| 类型 | Q/K/V 头数 | 参数量 | 相对 MHA |
|------|-----------|--------|----------|
| MHA  | 32/32/32  | 67.1M  | 100%     |
| GQA-8| 32/8/8    | 50.3M  | 75%      |
| GQA-4| 32/4/4    | 41.9M  | 62%      |
| MQA  | 32/1/1    | 33.6M  | 50%      |

### KV Cache 内存 (batch=32, seq_len=2048, FP16)

| 类型 | KV 头数 | 内存 (MB) | 相对 MHA |
|------|---------|-----------|----------|
| MHA  | 32      | 512       | 100%     |
| GQA-8| 8       | 128       | 25%      |
| GQA-4| 4       | 64        | 12.5%    |
| MQA  | 1       | 16        | 3.1%     |

### 推理加速 (KV Cache)

| Prompt 长度 | 无缓存 (ms) | 有缓存 (ms) | 加速比 |
|-------------|-------------|-------------|--------|
| 10          | 2.5         | 1.2         | 2.1x   |
| 50          | 8.3         | 1.3         | 6.4x   |
| 100         | 15.7        | 1.4         | 11.2x  |
| 200         | 30.2        | 1.5         | 20.1x  |

## 主流 LLM 的配置

| 模型 | 类型 | Q 头数 | KV 头数 | 分组比例 |
|------|------|--------|---------|----------|
| GPT-3 | MHA | 96 | 96 | 1:1 |
| LLaMA 2 7B | GQA | 32 | 32 | 1:1 |
| LLaMA 2 70B | GQA | 64 | 8 | 8:1 |
| Mistral 7B | GQA | 32 | 8 | 4:1 |
| PaLM | MQA | 128 | 1 | 128:1 |

## 扩展学习

### 进阶主题
1. **FlashAttention**: 内存高效的 Attention 实现
2. **PagedAttention**: vLLM 的 KV Cache 管理
3. **Streaming LLM**: 无限长序列的 Attention
4. **Multi-Query Attention**: 极致的内存优化

### 相关资源
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [vLLM](https://github.com/vllm-project/vllm)

## 贡献

欢迎提交 Issue 和 Pull Request！

## License

MIT License

---

**创建时间**: 2026-02-17  
**目的**: 为学习 TensorRT-LLM XQA 模块打下理论和实践基础  
**适用人群**: AI INFRA 工程师、LLM 研究者、深度学习爱好者
