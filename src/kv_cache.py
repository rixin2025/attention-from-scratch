"""
KV Cache 实现

在 LLM 推理中，KV Cache 用于缓存已计算的 Key 和 Value，避免重复计算
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# 导入 create_causal_mask
try:
    from .attention import create_causal_mask
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from attention import create_causal_mask


class KVCache:
    """
    KV Cache 用于存储和管理 Key 和 Value 的缓存
    
    在自回归生成中，每次只生成一个 token，但需要 attend 到所有历史 token。
    通过缓存历史的 K 和 V，可以避免重复计算。
    """
    
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        """
        Args:
            batch_size: 批次大小
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            head_dim: 每个头的维度
            dtype: 数据类型
            device: 设备
        """
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # 初始化缓存
        # 形状: [batch_size, num_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
        
        # 当前缓存的长度
        self.cache_len = 0
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新 KV Cache
        
        Args:
            key: 新的 Key，形状 [batch_size, num_heads, seq_len, head_dim]
            value: 新的 Value，形状 [batch_size, num_heads, seq_len, head_dim]
            start_pos: 起始位置，如果为 None 则使用 self.cache_len
        
        Returns:
            k_cache: 完整的 Key 缓存
            v_cache: 完整的 Value 缓存
        """
        if start_pos is None:
            start_pos = self.cache_len
        
        seq_len = key.size(2)
        end_pos = start_pos + seq_len
        
        # 更新缓存
        self.k_cache[:, :, start_pos:end_pos, :] = key
        self.v_cache[:, :, start_pos:end_pos, :] = value
        
        # 更新缓存长度
        self.cache_len = end_pos
        
        # 返回当前有效的缓存
        return (
            self.k_cache[:, :, :end_pos, :],
            self.v_cache[:, :, :end_pos, :]
        )
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前的 KV Cache
        
        Returns:
            k_cache: Key 缓存
            v_cache: Value 缓存
        """
        return (
            self.k_cache[:, :, :self.cache_len, :],
            self.v_cache[:, :, :self.cache_len, :]
        )
    
    def reset(self):
        """重置缓存"""
        self.cache_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
    
    def __repr__(self) -> str:
        return (f"KVCache(batch_size={self.batch_size}, num_heads={self.num_heads}, "
                f"max_seq_len={self.max_seq_len}, head_dim={self.head_dim}, "
                f"cache_len={self.cache_len})")


class MultiHeadAttentionWithCache(nn.Module):
    """
    带 KV Cache 的 Multi-Head Attention
    
    支持两种模式：
    1. Prefill: 处理完整的 prompt，初始化 KV Cache
    2. Decode: 每次生成一个 token，使用 KV Cache
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: Dropout 概率
            bias: 是否使用 bias
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Q, K, V 的线性投影
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # KV Cache (延迟初始化)
        self.kv_cache: Optional[KVCache] = None
    
    def _init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """初始化 KV Cache"""
        self.kv_cache = KVCache(
            batch_size=batch_size,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len,
            head_dim=self.d_k,
            dtype=dtype,
            device=device,
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_cache: bool = False,
        start_pos: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query 张量，形状 [batch_size, seq_len_q, d_model]
            key: Key 张量，形状 [batch_size, seq_len_k, d_model]
            value: Value 张量，形状 [batch_size, seq_len_k, d_model]
            use_cache: 是否使用 KV Cache
            start_pos: KV Cache 的起始位置
            mask: 可选的 mask 张量
            causal: 是否使用因果 mask（自回归场景）
        
        Returns:
            output: 输出张量，形状 [batch_size, seq_len_q, d_model]
            attention_weights: 注意力权重
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        
        # 1. 线性投影
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 使用 KV Cache (如果启用)
        if use_cache:
            if self.kv_cache is None:
                self._init_cache(batch_size, query.device, query.dtype)
            
            # 更新缓存并获取完整的 K, V
            K, V = self.kv_cache.update(K, V, start_pos)
            seq_len_k = K.size(2)
        else:
            seq_len_k = K.size(2)
        
        # 3. 计算 Scaled Dot-Product Attention
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
        
        # 4. 应用 mask
        if mask is not None:
            # 用户提供的 mask 优先
            scores = scores.masked_fill(mask == 0, float('-inf'))
        elif causal:
            # 需要因果 mask 的场景：
            # 1. Prefill 阶段（use_cache=True, start_pos=0）
            # 2. 不使用缓存（use_cache=False）
            # 不需要 mask 的场景：
            # - Decode 阶段（use_cache=True, start_pos=None）：每次只有一个新 token
            if (use_cache and start_pos == 0) or not use_cache:
                causal_mask = create_causal_mask(seq_len_q, device=query.device)
                scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        if self.dropout > 0.0:
            attention_weights = torch.dropout(attention_weights, self.dropout, self.training)
        
        attn_output = torch.matmul(attention_weights, V)
        
        # 5. 拼接多个头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # 6. 输出投影
        output = self.W_o(attn_output)
        
        return output, attention_weights
    
    def reset_cache(self):
        """重置 KV Cache"""
        if self.kv_cache is not None:
            self.kv_cache.reset()


def compare_with_without_cache():
    """
    比较使用和不使用 KV Cache 的性能差异
    """
    import time
    
    batch_size = 1
    d_model = 512
    num_heads = 8
    max_seq_len = 2048
    prompt_len = 100
    gen_len = 100
    
    print("=" * 60)
    print("KV Cache 性能对比")
    print("=" * 60)
    print(f"配置: d_model={d_model}, num_heads={num_heads}")
    print(f"Prompt 长度: {prompt_len}, 生成长度: {gen_len}")
    print("=" * 60)
    
    # 创建模型
    model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
    model.eval()
    
    # 1. 不使用 KV Cache
    print("\n不使用 KV Cache:")
    tokens = torch.randn(batch_size, prompt_len, d_model)
    
    start_time = time.time()
    for i in range(gen_len):
        # 每次都要处理完整的序列
        with torch.no_grad():
            output, _ = model(tokens, tokens, tokens, use_cache=False)
        # 模拟添加新 token
        new_token = torch.randn(batch_size, 1, d_model)
        tokens = torch.cat([tokens, new_token], dim=1)
    
    time_without_cache = time.time() - start_time
    print(f"  总时间: {time_without_cache:.2f}s")
    print(f"  平均每个 token: {time_without_cache / gen_len * 1000:.2f}ms")
    
    # 2. 使用 KV Cache
    print("\n使用 KV Cache:")
    model.reset_cache()
    
    # Prefill: 处理 prompt
    prompt = torch.randn(batch_size, prompt_len, d_model)
    with torch.no_grad():
        output, _ = model(prompt, prompt, prompt, use_cache=True, start_pos=0)
    
    # Decode: 逐个生成 token
    start_time = time.time()
    for i in range(gen_len):
        new_token = torch.randn(batch_size, 1, d_model)
        with torch.no_grad():
            output, _ = model(new_token, new_token, new_token, use_cache=True)
    
    time_with_cache = time.time() - start_time
    print(f"  总时间: {time_with_cache:.2f}s")
    print(f"  平均每个 token: {time_with_cache / gen_len * 1000:.2f}ms")
    
    # 加速比
    speedup = time_without_cache / time_with_cache
    print(f"\n加速比: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    # 测试 KV Cache
    batch_size = 2
    num_heads = 8
    max_seq_len = 100
    head_dim = 64
    
    # 创建 KV Cache
    cache = KVCache(batch_size, num_heads, max_seq_len, head_dim)
    print(cache)
    
    # 模拟 prefill
    prompt_len = 10
    k_prompt = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    v_prompt = torch.randn(batch_size, num_heads, prompt_len, head_dim)
    k_cached, v_cached = cache.update(k_prompt, v_prompt)
    print(f"\nPrefill 后缓存长度: {cache.cache_len}")
    print(f"K 缓存形状: {k_cached.shape}")
    
    # 模拟 decode
    for i in range(5):
        k_new = torch.randn(batch_size, num_heads, 1, head_dim)
        v_new = torch.randn(batch_size, num_heads, 1, head_dim)
        k_cached, v_cached = cache.update(k_new, v_new)
        print(f"Decode step {i+1}, 缓存长度: {cache.cache_len}")
    
    # 性能对比
    print("\n")
    compare_with_without_cache()
