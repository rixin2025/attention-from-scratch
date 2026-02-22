# Attention From Scratch

从零实现 Attention 机制的完整教程，涵盖从基础到进阶的所有核心技术，包括 Scaled Dot-Product Attention、Multi-Head Attention、Grouped Query Attention、KV Cache、Flash Attention 和 Paged KV Cache。

*** github地址：https://github.com/rixin2025/attention-from-scratch/tree/main
## 目录

- [项目结构](#项目结构)
- [学习路径](#学习路径)
  - [基础篇](#基础篇)
  - [进阶篇](#进阶篇)
- [快速开始](#快速开始)
- [学习目标](#学习目标)
- [进阶学习资源](#进阶学习资源)
- [与 TensorRT-LLM XQA 的关系](#与-tensorrt-llm-xqa-的关系)
- [参考资料](#参考资料)
- [项目特色](#项目特色)
- [适合人群](#适合人群)

## 项目结构

```
attention-from-scratch/
├── README.md                           # 项目说明
├── requirements.txt                    # Python 依赖
├── notebooks/                          # Jupyter Notebooks（渐进式学习）
│   ├── 01_scaled_dot_product.ipynb    # Scaled Dot-Product Attention
│   ├── 02_multi_head_attention.ipynb  # Multi-Head Attention (MHA)
│   ├── 03_grouped_query_attention.ipynb # Grouped Query Attention (GQA)
│   ├── 04_kv_cache.ipynb              # KV Cache 实现
│   ├── 05_flash_attention.ipynb       # Flash Attention（内存优化）
│   └── 06_paged_kv_cache.ipynb        # Paged KV Cache（分页管理）
├── src/                                # 源代码实现
│   ├── __init__.py
│   ├── attention.py                    # Attention 核心实现
│   ├── gqa.py                          # GQA 实现
│   ├── kv_cache.py                     # KV Cache 实现
│   ├── flash_attention.py              # Flash Attention 实现
│   └── paged_kv_cache.py               # Paged KV Cache 实现
├── tests/                              # 单元测试
│   ├── __init__.py
│   ├── test_attention.py               # Attention 测试
│   ├── test_gqa.py                     # GQA 测试
│   └── test_kv_cache.py                # KV Cache 测试
└── docs/                               # 文档
    ├── ADVANCED_IMPLEMENTATION_ROADMAP.md  # 进阶实现路线图
    ├── ADVANCED_SUMMARY.md                 # 进阶实现总结
    ├── XQA_MAPPING.md                      # XQA 映射指南
    ├── BLOG_ARTICLE.md                     # 博客文章
    └── PROJECT_SUMMARY.md                  # 项目总结
```

## 学习路径

本项目采用渐进式学习方式，从基础到进阶，逐步深入理解 Attention 机制。

### 基础篇

#### 1. Scaled Dot-Product Attention
- 理解 Attention 的基本计算公式：`Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`
- 实现标准的 Attention 计算
- 理解 scaling factor 的作用和数值稳定性

#### 2. Multi-Head Attention (MHA)
- 理解多头注意力的并行计算机制
- 实现 Q、K、V 的线性投影和多头分割
- 实现多头的拼接和输出投影
- 分析多头的优势和计算复杂度

#### 3. Grouped Query Attention (GQA)
- 理解 MHA、MQA、GQA 的区别和演进
- 实现 KV 头的分组共享机制
- 分析计算量和内存占用的权衡
- 理解 GQA 在大模型中的应用

#### 4. KV Cache
- 理解 KV Cache 在自回归推理中的作用
- 实现增量式 KV Cache 更新
- 分析性能提升和内存占用
- 理解 prefill 和 decode 阶段的区别

### 进阶篇

#### 5. Flash Attention
- **核心问题**: 标准 Attention 的内存瓶颈（O(N²) 内存复杂度）
- **解决方案**: 
  - Tiling（分块计算）：避免存储完整的注意力矩阵
  - Online Softmax：增量计算 softmax，避免两次遍历
  - Recomputation：反向传播时重新计算，节省内存
- **实现要点**:
  - 分块计算 Q·K^T
  - 在线 softmax 合并算法
  - 数值稳定性优化
- **性能提升**: 内存从 O(N²) 降低到 O(N)

#### 6. Paged KV Cache
- **核心问题**: 连续 KV Cache 导致的内存碎片和低利用率
- **解决方案**:
  - 分页管理：将 KV Cache 分成固定大小的页面
  - 页面表：记录每个序列的页面映射
  - 内存池：全局页面池，支持动态分配和回收
- **实现要点**:
  - 页面分配和回收机制
  - 页面表管理
  - 跨页面的 Attention 计算
- **优势**: 提高内存利用率，减少内存碎片

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：
- `torch>=2.0.0` - PyTorch 深度学习框架
- `numpy>=1.21.0` - 数值计算
- `jupyter>=1.0.0` - Jupyter Notebook
- `matplotlib>=3.5.0` - 可视化
- `pytest>=7.0.0` - 单元测试
- `einops>=0.6.0` - 张量操作

### 2. 运行 Notebooks（推荐学习方式）

按顺序学习 Notebooks，每个 notebook 都包含详细的原理讲解、代码实现和可视化：

```bash
jupyter notebook notebooks/
```

建议学习顺序：
1. `01_scaled_dot_product.ipynb` - 基础 Attention
2. `02_multi_head_attention.ipynb` - 多头 Attention
3. `03_grouped_query_attention.ipynb` - 分组查询 Attention
4. `04_kv_cache.ipynb` - KV 缓存
5. `05_flash_attention.ipynb` - Flash Attention（进阶）
6. `06_paged_kv_cache.ipynb` - 分页 KV 缓存（进阶）

### 3. 使用源代码

```python
# 基础 Attention
from src.attention import ScaledDotProductAttention, MultiHeadAttention
attn = MultiHeadAttention(d_model=512, num_heads=8)
output = attn(x)

# Grouped Query Attention
from src.gqa import GroupedQueryAttention
gqa = GroupedQueryAttention(d_model=512, num_q_heads=8, num_kv_heads=2)
output = gqa(x)

# KV Cache
from src.kv_cache import KVCache
cache = KVCache(num_heads=8, head_dim=64, max_seq_len=2048)
output = cache.forward_with_cache(q, k, v, start_pos=0)

# Flash Attention
from src.flash_attention import FlashAttention
flash_attn = FlashAttention(d_model=512, num_heads=8, block_size=64)
output, lse = flash_attn(x)

# Paged KV Cache
from src.paged_kv_cache import PagedKVCache
paged_cache = PagedKVCache(num_heads=8, head_dim=64, page_size=16)
```

### 4. 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_attention.py
pytest tests/test_gqa.py
pytest tests/test_kv_cache.py
```

## 总结博客论文
待补充微信公众号 文章链接---------

## 参考资料

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - GQA 论文
- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Flash Attention 论文
- [Flash Attention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) - Flash Attention v2

### 博客和教程
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化 Transformer
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 带注释的 Transformer
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/) - vLLM 博客

### 开源项目
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 高性能推理引擎
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention 实现
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - 官方实现

## 项目特色

- 📚 **渐进式学习**: 从基础到进阶，循序渐进
- 💡 **原理讲解**: 每个 notebook 都包含详细的数学推导和可视化
- 🔬 **完整实现**: 提供可运行的 Python 实现和单元测试
- 🎯 **工程对照**: 与 TensorRT-LLM XQA 的实现对应
- 📊 **性能分析**: 分析内存占用和计算复杂度
- 🚀 **生产级技术**: 涵盖 Flash Attention、Paged KV Cache 等前沿技术

## 适合人群

- 🎓 想要深入理解 Attention 机制的学习者
- 💻 需要实现自定义 Attention 的开发者
- 🔧 希望优化 LLM 推理性能的工程师
- 📖 准备阅读 TensorRT-LLM 源码的研究者

## 贡献

欢迎提交 Issue 和 Pull Request！

## 后续计划

进一步梳理 TensorRT-LLM XQA 工程级代码，欢迎star！

## License

MIT License
