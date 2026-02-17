"""
Grouped Query Attention (GQA) 实现

GQA 是 MHA 和 MQA 的折中方案：
- MHA (Multi-Head Attention): 每个 Q 头对应一个 K 头和一个 V 头
- MQA (Multi-Query Attention): 所有 Q 头共享一个 K 头和一个 V 头
- GQA (Grouped Query Attention): 多个 Q 头共享一组 K 头和 V 头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .attention import scaled_dot_product_attention


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    
    将 Q 头分组，每组共享一个 K 头和 V 头
    
    例如：
    - num_q_heads = 32, num_kv_heads = 8 -> 每 4 个 Q 头共享 1 个 KV 头
    - num_q_heads = 32, num_kv_heads = 1 -> MQA (所有 Q 头共享 1 个 KV 头)
    - num_q_heads = 32, num_kv_heads = 32 -> MHA (每个 Q 头独立 KV 头)
    """
    
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            d_model: 模型维度
            num_q_heads: Query 头数
            num_kv_heads: Key/Value 头数
            dropout: Dropout 概率
            bias: 是否使用 bias
        """
        super().__init__()
        
        assert d_model % num_q_heads == 0, "d_model 必须能被 num_q_heads 整除"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须能被 num_kv_heads 整除"
        
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads  # 每组的 Q 头数
        self.d_k = d_model // num_q_heads  # 每个头的维度
        self.dropout = dropout
        
        # Q 投影: d_model -> num_q_heads * d_k
        self.W_q = nn.Linear(d_model, num_q_heads * self.d_k, bias=bias)
        
        # K, V 投影: d_model -> num_kv_heads * d_k
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        
        # 输出投影
        self.W_o = nn.Linear(num_q_heads * self.d_k, d_model, bias=bias)
        
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
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 1. 线性投影
        # Q: [batch_size, seq_len_q, num_q_heads * d_k]
        Q = self.W_q(query)
        # K, V: [batch_size, seq_len_k, num_kv_heads * d_k]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 重塑并转置
        # Q: [batch_size, seq_len_q, num_q_heads, d_k] -> [batch_size, num_q_heads, seq_len_q, d_k]
        Q = Q.view(batch_size, seq_len_q, self.num_q_heads, self.d_k).transpose(1, 2)
        
        # K, V: [batch_size, seq_len_k, num_kv_heads, d_k] -> [batch_size, num_kv_heads, seq_len_k, d_k]
        K = K.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 3. 扩展 K, V 以匹配 Q 的头数
        # 方法：repeat_interleave 每个 KV 头复制 num_groups 次
        # [batch_size, num_kv_heads, seq_len_k, d_k] -> [batch_size, num_q_heads, seq_len_k, d_k]
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # 4. 计算 Scaled Dot-Product Attention
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # 5. 拼接多个头
        # [batch_size, num_q_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_q_heads * d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.num_q_heads * self.d_k
        )
        
        # 6. 输出投影
        output = self.W_o(attn_output)
        
        return output, attention_weights
    
    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, num_q_heads={self.num_q_heads}, '
                f'num_kv_heads={self.num_kv_heads}, num_groups={self.num_groups}, d_k={self.d_k}')


def compare_attention_variants(d_model: int, num_heads: int, seq_len: int):
    """
    比较 MHA、MQA、GQA 的参数量和计算量
    
    Args:
        d_model: 模型维度
        num_heads: 头数
        seq_len: 序列长度
    """
    d_k = d_model // num_heads
    
    print("=" * 60)
    print(f"配置: d_model={d_model}, num_heads={num_heads}, seq_len={seq_len}")
    print("=" * 60)
    
    # MHA: 所有头独立
    mha_qkv_params = 3 * d_model * d_model  # W_q, W_k, W_v
    mha_o_params = d_model * d_model  # W_o
    mha_total_params = mha_qkv_params + mha_o_params
    mha_flops = 4 * num_heads * seq_len * seq_len * d_k  # QK^T + softmax + V
    
    print(f"\nMHA (Multi-Head Attention):")
    print(f"  - Q/K/V 头数: {num_heads}/{num_heads}/{num_heads}")
    print(f"  - 参数量: {mha_total_params:,} ({mha_total_params / 1e6:.2f}M)")
    print(f"  - FLOPs: {mha_flops:,} ({mha_flops / 1e9:.2f}G)")
    
    # MQA: 所有 Q 头共享 1 个 KV 头
    mqa_q_params = d_model * d_model
    mqa_kv_params = 2 * d_model * d_k  # 只有 1 个 KV 头
    mqa_o_params = d_model * d_model
    mqa_total_params = mqa_q_params + mqa_kv_params + mqa_o_params
    mqa_flops = 4 * num_heads * seq_len * seq_len * d_k
    
    print(f"\nMQA (Multi-Query Attention):")
    print(f"  - Q/K/V 头数: {num_heads}/1/1")
    print(f"  - 参数量: {mqa_total_params:,} ({mqa_total_params / 1e6:.2f}M)")
    print(f"  - 参数减少: {(1 - mqa_total_params / mha_total_params) * 100:.1f}%")
    print(f"  - FLOPs: {mqa_flops:,} ({mqa_flops / 1e9:.2f}G)")
    
    # GQA: 每 4 个 Q 头共享 1 个 KV 头
    num_kv_heads = num_heads // 4
    gqa_q_params = d_model * d_model
    gqa_kv_params = 2 * d_model * (num_kv_heads * d_k)
    gqa_o_params = d_model * d_model
    gqa_total_params = gqa_q_params + gqa_kv_params + gqa_o_params
    gqa_flops = 4 * num_heads * seq_len * seq_len * d_k
    
    print(f"\nGQA (Grouped Query Attention, 4:1):")
    print(f"  - Q/K/V 头数: {num_heads}/{num_kv_heads}/{num_kv_heads}")
    print(f"  - 参数量: {gqa_total_params:,} ({gqa_total_params / 1e6:.2f}M)")
    print(f"  - 参数减少: {(1 - gqa_total_params / mha_total_params) * 100:.1f}%")
    print(f"  - FLOPs: {gqa_flops:,} ({gqa_flops / 1e9:.2f}G)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 测试 GQA
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_q_heads = 32
    num_kv_heads = 8  # 每 4 个 Q 头共享 1 个 KV 头
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建 GQA 模块
    gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
    
    # 前向传播
    output, attn_weights = gqa(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"\n{gqa}")
    
    # 比较不同 Attention 变体
    print("\n")
    compare_attention_variants(d_model=4096, num_heads=32, seq_len=2048)
