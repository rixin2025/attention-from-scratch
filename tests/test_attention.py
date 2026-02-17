"""
测试 Attention 实现
"""

import pytest
import torch
from src.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask,
)


class TestScaledDotProductAttention:
    """测试 Scaled Dot-Product Attention"""
    
    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 2
        num_heads = 4
        seq_len_q = 10
        seq_len_k = 15
        d_k = 64
        
        Q = torch.randn(batch_size, num_heads, seq_len_q, d_k)
        K = torch.randn(batch_size, num_heads, seq_len_k, d_k)
        V = torch.randn(batch_size, num_heads, seq_len_k, d_k)
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, num_heads, seq_len_q, d_k)
        assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    
    def test_attention_weights_sum_to_one(self):
        """测试注意力权重是否归一化"""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = scaled_dot_product_attention(Q, K, V)
        
        # 每一行的权重应该和为 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    def test_with_mask(self):
        """测试带 mask 的情况"""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        # 创建因果 mask
        mask = create_causal_mask(seq_len)
        
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # 检查 mask 后的权重
        # 上三角部分应该为 0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(
                    attn_weights[:, :, i, j],
                    torch.zeros_like(attn_weights[:, :, i, j]),
                    atol=1e-6
                )


class TestMultiHeadAttention:
    """测试 Multi-Head Attention"""
    
    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        
        x = torch.randn(batch_size, seq_len, d_model)
        mha = MultiHeadAttention(d_model, num_heads)
        
        output, attn_weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_self_attention(self):
        """测试自注意力"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        
        x = torch.randn(batch_size, seq_len, d_model)
        mha = MultiHeadAttention(d_model, num_heads)
        
        output, _ = mha(x, x, x)
        
        # 输出应该是有限的数值
        assert torch.isfinite(output).all()
    
    def test_cross_attention(self):
        """测试交叉注意力"""
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 15
        d_model = 512
        num_heads = 8
        
        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_k, d_model)
        value = torch.randn(batch_size, seq_len_k, d_model)
        
        mha = MultiHeadAttention(d_model, num_heads)
        output, attn_weights = mha(query, key, value)
        
        assert output.shape == (batch_size, seq_len_q, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    
    def test_with_causal_mask(self):
        """测试因果 mask"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        
        x = torch.randn(batch_size, seq_len, d_model)
        mha = MultiHeadAttention(d_model, num_heads)
        
        mask = create_causal_mask(seq_len)
        output, attn_weights = mha(x, x, x, mask=mask)
        
        # 检查因果性
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(
                    attn_weights[:, :, i, j],
                    torch.zeros_like(attn_weights[:, :, i, j]),
                    atol=1e-6
                )


class TestMaskFunctions:
    """测试 mask 函数"""
    
    def test_causal_mask_shape(self):
        """测试因果 mask 形状"""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        assert mask.shape == (1, 1, seq_len, seq_len)
    
    def test_causal_mask_values(self):
        """测试因果 mask 的值"""
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        # 下三角应该为 1
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask[0, 0, i, j] == 1
        
        # 上三角应该为 0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[0, 0, i, j] == 0
    
    def test_padding_mask_shape(self):
        """测试 padding mask 形状"""
        batch_size = 4
        max_len = 10
        seq_lens = torch.tensor([5, 7, 10, 3])
        
        mask = create_padding_mask(seq_lens, max_len)
        assert mask.shape == (batch_size, 1, 1, max_len)
    
    def test_padding_mask_values(self):
        """测试 padding mask 的值"""
        batch_size = 2
        max_len = 5
        seq_lens = torch.tensor([3, 5])
        
        mask = create_padding_mask(seq_lens, max_len)
        
        # 第一个序列前 3 个位置应该为 True
        assert mask[0, 0, 0, :3].all()
        assert not mask[0, 0, 0, 3:].any()
        
        # 第二个序列所有位置应该为 True
        assert mask[1, 0, 0, :].all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
