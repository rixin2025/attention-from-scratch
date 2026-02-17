"""
测试 Grouped Query Attention (GQA)
"""

import pytest
import torch
from src.gqa import GroupedQueryAttention, compare_attention_variants


class TestGroupedQueryAttention:
    """测试 Grouped Query Attention"""
    
    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_q_heads = 32
        num_kv_heads = 8
        
        x = torch.randn(batch_size, seq_len, d_model)
        gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        
        output, attn_weights = gqa(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_q_heads, seq_len, seq_len)
    
    def test_mha_mode(self):
        """测试 MHA 模式 (num_q_heads == num_kv_heads)"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        
        x = torch.randn(batch_size, seq_len, d_model)
        gqa = GroupedQueryAttention(d_model, num_heads, num_heads)
        
        output, _ = gqa(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()
    
    def test_mqa_mode(self):
        """测试 MQA 模式 (num_kv_heads == 1)"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_q_heads = 32
        num_kv_heads = 1
        
        x = torch.randn(batch_size, seq_len, d_model)
        gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        
        output, _ = gqa(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()
    
    def test_gqa_mode(self):
        """测试 GQA 模式 (1 < num_kv_heads < num_q_heads)"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_q_heads = 32
        num_kv_heads = 8
        
        x = torch.randn(batch_size, seq_len, d_model)
        gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        
        output, _ = gqa(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()
    
    def test_cross_attention(self):
        """测试交叉注意力"""
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 15
        d_model = 512
        num_q_heads = 32
        num_kv_heads = 8
        
        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_k, d_model)
        value = torch.randn(batch_size, seq_len_k, d_model)
        
        gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        output, attn_weights = gqa(query, key, value)
        
        assert output.shape == (batch_size, seq_len_q, d_model)
        assert attn_weights.shape == (batch_size, num_q_heads, seq_len_q, seq_len_k)
    
    def test_parameter_count(self):
        """测试参数量"""
        d_model = 512
        num_q_heads = 32
        num_kv_heads = 8
        d_k = d_model // num_q_heads
        
        gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        
        # 计算预期的参数量
        # W_q: d_model * (num_q_heads * d_k)
        # W_k, W_v: d_model * (num_kv_heads * d_k) * 2
        # W_o: (num_q_heads * d_k) * d_model
        expected_params = (
            d_model * num_q_heads * d_k +  # W_q
            d_model * num_kv_heads * d_k * 2 +  # W_k, W_v
            num_q_heads * d_k * d_model  # W_o
        )
        
        actual_params = sum(p.numel() for p in gqa.parameters())
        
        # 注意：这里没有考虑 bias，如果有 bias 会多一些参数
        assert actual_params >= expected_params
    
    def test_invalid_config(self):
        """测试无效配置"""
        d_model = 512
        num_q_heads = 32
        num_kv_heads = 7  # 不能整除
        
        with pytest.raises(AssertionError):
            GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)


class TestCompareAttentionVariants:
    """测试 Attention 变体比较函数"""
    
    def test_compare_function(self):
        """测试比较函数能正常运行"""
        # 这个测试主要确保函数不会崩溃
        compare_attention_variants(d_model=512, num_heads=8, seq_len=1024)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
