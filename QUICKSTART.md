# 快速开始指南

## 5 分钟快速体验

### 1. 测试基础 Attention

```bash
cd attention-from-scratch
python -c "
from src.attention import MultiHeadAttention
import torch

# 创建 MHA
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)

# 前向传播
output, attn_weights = mha(x, x, x)
print(f'输入形状: {x.shape}')
print(f'输出形状: {output.shape}')
print(f'注意力权重形状: {attn_weights.shape}')
print('✓ Multi-Head Attention 工作正常！')
"
```

### 2. 测试 GQA

```bash
python -c "
from src.gqa import GroupedQueryAttention
import torch

# 创建 GQA (32 Q 头, 8 KV 头)
gqa = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=8)
x = torch.randn(2, 10, 512)

output, attn_weights = gqa(x, x, x)
print(f'GQA 配置: {gqa}')
print(f'输出形状: {output.shape}')
print('✓ Grouped Query Attention 工作正常！')
"
```

### 3. 测试 KV Cache

```bash
python -c "
from src.kv_cache import MultiHeadAttentionWithCache
import torch

# 创建带缓存的 MHA
mha = MultiHeadAttentionWithCache(d_model=512, num_heads=8, max_seq_len=100)

# Prefill
prompt = torch.randn(1, 10, 512)
output, _ = mha(prompt, prompt, prompt, use_cache=True, start_pos=0)
print(f'Prefill 完成，缓存长度: {mha.kv_cache.cache_len}')

# Decode
for i in range(5):
    new_token = torch.randn(1, 1, 512)
    output, _ = mha(new_token, new_token, new_token, use_cache=True)
    print(f'Decode step {i+1}, 缓存长度: {mha.kv_cache.cache_len}')

print('✓ KV Cache 工作正常！')
"
```

### 4. 运行测试

```bash
# 安装测试依赖
pip install pytest

# 运行所有测试
pytest tests/ -v

# 应该看到类似输出:
# tests/test_attention.py::TestScaledDotProductAttention::test_output_shape PASSED
# tests/test_gqa.py::TestGroupedQueryAttention::test_output_shape PASSED
# tests/test_kv_cache.py::TestKVCache::test_initialization PASSED
# ...
```

### 5. 启动 Jupyter Notebook

```bash
# 安装 jupyter
pip install jupyter matplotlib seaborn

# 启动 notebook
jupyter notebook notebooks/

# 在浏览器中打开:
# - 01_scaled_dot_product.ipynb (从这里开始)
# - 02_multi_head_attention.ipynb
# - 03_grouped_query_attention.ipynb
# - 04_kv_cache.ipynb
```

## 学习路径建议

### 路径 1: 理论优先 (推荐初学者)
1. 阅读 `README.md` 了解项目概况
2. 按顺序学习 4 个 Notebooks
3. 阅读源码 `src/attention.py` → `src/gqa.py` → `src/kv_cache.py`
4. 运行测试验证理解

### 路径 2: 代码优先 (推荐有基础者)
1. 直接阅读 `src/attention.py`
2. 运行测试 `pytest tests/test_attention.py -v`
3. 查看 Notebook 中的可视化
4. 依次学习 GQA 和 KV Cache

### 路径 3: 实践优先 (推荐动手派)
1. 运行快速测试脚本
2. 修改参数观察变化
3. 阅读 Notebook 理解原理
4. 尝试实现自己的变体

## 常见问题

### Q1: 如何修改模型配置？

```python
# 修改头数
mha = MultiHeadAttention(d_model=512, num_heads=16)  # 从 8 改为 16

# 修改 GQA 分组
gqa = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=4)  # 8:1 分组

# 修改最大序列长度
mha_cache = MultiHeadAttentionWithCache(d_model=512, num_heads=8, max_seq_len=4096)
```

### Q2: 如何可视化注意力权重？

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 获取注意力权重
output, attn_weights = mha(x, x, x)

# 可视化第一个样本的第一个头
plt.figure(figsize=(8, 6))
sns.heatmap(attn_weights[0, 0].detach().numpy(), cmap='YlOrRd')
plt.title('Attention Weights')
plt.show()
```

### Q3: 如何测试性能？

```python
import time
import torch

model = MultiHeadAttention(d_model=512, num_heads=8)
model.eval()

x = torch.randn(1, 100, 512)

# 预热
for _ in range(10):
    with torch.no_grad():
        _ = model(x, x, x)

# 测试
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(x, x, x)
end = time.time()

print(f"平均时间: {(end - start) / 100 * 1000:.2f} ms")
```

### Q4: 如何与 PyTorch 的实现对比？

```python
import torch.nn as nn

# 我们的实现
our_mha = MultiHeadAttention(d_model=512, num_heads=8)

# PyTorch 的实现
torch_mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

# 对比输出形状
x = torch.randn(2, 10, 512)
our_output, _ = our_mha(x, x, x)
torch_output, _ = torch_mha(x, x, x)

print(f"我们的输出: {our_output.shape}")
print(f"PyTorch 输出: {torch_output.shape}")
# 应该都是 torch.Size([2, 10, 512])
```

## 下一步

完成本项目后，你可以：

1. **深入 TensorRT-LLM XQA**
   - 阅读 `cpp/kernels/xqa/test/refAttention.h`
   - 理解 CUDA Kernel 实现
   - 学习性能优化技巧

2. **学习 FlashAttention**
   - 理解 IO-aware 的优化
   - 学习 tiling 技术
   - 实现简化版

3. **研究 PagedAttention**
   - 学习 vLLM 的实现
   - 理解内存管理
   - 优化 batch 推理

4. **实现自己的变体**
   - 尝试不同的分组策略
   - 实现 Sliding Window Attention
   - 优化长序列处理

## 获取帮助

- 查看 `PROJECT_SUMMARY.md` 了解完整项目信息
- 阅读 Notebooks 中的详细说明
- 运行测试验证你的理解
- 参考 TensorRT-LLM XQA 的实现

祝学习顺利！🚀
