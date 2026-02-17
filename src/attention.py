"""
Attention 核心实现

实现 Scaled Dot-Product Attention 和 Multi-Head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention
    
    公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Query 张量，形状 [batch_size, num_heads, seq_len_q, d_k]
        key: Key 张量，形状 [batch_size, num_heads, seq_len_k, d_k]
        value: Value 张量，形状 [batch_size, num_heads, seq_len_k, d_v]
        mask: 可选的 mask 张量，形状 [batch_size, 1, seq_len_q, seq_len_k]
        dropout: Dropout 概率
    
    Returns:
        output: 注意力输出，形状 [batch_size, num_heads, seq_len_q, d_v]
        attention_weights: 注意力权重，形状 [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    # 获取 d_k (key 的维度)
    d_k = query.size(-1)
    
    # 计算注意力分数: Q @ K^T / sqrt(d_k)
    # [batch_size, num_heads, seq_len_q, d_k] @ [batch_size, num_heads, d_k, seq_len_k]
    # -> [batch_size, num_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用 mask (如果提供)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 应用 softmax 得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 应用 dropout (如果需要)
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    
    # 计算加权和: attention_weights @ V
    # [batch_size, num_heads, seq_len_q, seq_len_k] @ [batch_size, num_heads, seq_len_k, d_v]
    # -> [batch_size, num_heads, seq_len_q, d_v]
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA)
    
    将输入投影到多个头，每个头独立计算 attention，最后拼接输出
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 概率
            bias: 是否使用 bias
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.dropout = dropout
        
        # Q, K, V 的线性投影
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query 张量，形状 [batch_size, seq_len_q, d_model]
            key: Key 张量，形状 [batch_size, seq_len_k, d_model]
            value: Value 张量，形状 [batch_size, seq_len_k, d_model]
            mask: 可选的 mask 张量
        
        Returns:
            output: 输出张量，形状 [batch_size, seq_len_q, d_model]
            attention_weights: 注意力权重
        """
        batch_size = query.size(0)
        
        # 1. 线性投影并分割成多个头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        # -> [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算 Scaled Dot-Product Attention
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # 3. 拼接多个头
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_heads, d_k]
        # -> [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 输出投影
        output = self.W_o(attn_output)
        
        return output, attention_weights
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}'


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果 mask (下三角矩阵)
    
    用于 decoder 的自注意力，防止看到未来的信息
    
    Args:
        seq_len: 序列长度
        device: 设备
    
    Returns:
        mask: 形状 [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def create_padding_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    创建 padding mask
    
    Args:
        seq_lens: 每个序列的实际长度，形状 [batch_size]
        max_len: 最大序列长度
    
    Returns:
        mask: 形状 [batch_size, 1, 1, max_len]
    """
    batch_size = seq_lens.size(0)
    mask = torch.arange(max_len, device=seq_lens.device).expand(batch_size, max_len)
    mask = (mask < seq_lens.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    return mask


if __name__ == "__main__":
    # 简单测试
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建 MHA 模块
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 前向传播
    output, attn_weights = mha(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试因果 mask
    causal_mask = create_causal_mask(seq_len)
    output_masked, _ = mha(x, x, x, mask=causal_mask)
    print(f"带 mask 的输出形状: {output_masked.shape}")
