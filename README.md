# Attention From Scratch

从零实现 Attention 机制，包括 Scaled Dot-Product Attention、Multi-Head Attention、Grouped Query Attention 和 KV Cache。

## 项目结构

```
attention-from-scratch/
├── README.md                           # 项目说明
├── requirements.txt                    # Python 依赖
├── notebooks/                          # Jupyter Notebooks
│   ├── 01_scaled_dot_product.ipynb    # Scaled Dot-Product Attention
│   ├── 02_multi_head_attention.ipynb  # Multi-Head Attention (MHA)
│   ├── 03_grouped_query_attention.ipynb # Grouped Query Attention (GQA)
│   └── 04_kv_cache.ipynb              # KV Cache 实现
├── src/                                # 源代码
│   ├── __init__.py
│   ├── attention.py                    # Attention 核心实现
│   ├── gqa.py                          # GQA 实现
│   └── kv_cache.py                     # KV Cache 实现
└── tests/                              # 单元测试
    ├── __init__.py
    ├── test_attention.py               # Attention 测试
    ├── test_gqa.py                     # GQA 测试
    └── test_kv_cache.py                # KV Cache 测试
```

## 学习路径

### 1. Scaled Dot-Product Attention
- 理解 Attention 的基本计算公式
- 实现 `softmax(Q @ K^T / sqrt(d_k)) @ V`
- 理解 scaling factor 的作用

### 2. Multi-Head Attention (MHA)
- 理解多头注意力的并行计算
- 实现 Q、K、V 的线性投影
- 实现多头的拼接和输出投影

### 3. Grouped Query Attention (GQA)
- 理解 MHA、MQA、GQA 的区别
- 实现 KV 头的分组共享
- 分析计算量和内存占用

### 4. KV Cache
- 理解 KV Cache 在推理中的作用
- 实现增量式 KV Cache 更新
- 分析性能提升

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行 Notebooks

```bash
jupyter notebook notebooks/
```

### 运行测试

```bash
pytest tests/
```

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## 学习目标

完成本项目后，你将：
- ✅ 深入理解 Attention 机制的数学原理
- ✅ 掌握 MHA、MQA、GQA 的实现细节
- ✅ 理解 KV Cache 的优化原理
- ✅ 能够从零实现完整的 Attention 模块

## 与 TensorRT-LLM XQA 的关系

本项目是 TensorRT-LLM XQA 模块的简化版实现，帮助理解：
- XQA 中的 Attention 计算逻辑
- GQA 的实现方式
- KV Cache 的管理机制
- Paged Attention 的基础

## 作者

jensen.li(后续将基于 TensorRT-LLM XQA 模块更加深入分析，添加更多优化手段和工程化技巧介绍；欢迎star和讨论)

## License

MIT License
