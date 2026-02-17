"""
测试 KV Cache
"""

import pytest
import torch
from src.kv_cache import KVCache, MultiHeadAttentionWithCache


class TestKVCache:
    """测试 KV Cache"""
    
    def test_initialization(self):
        """测试初始化"""
        batch_size = 2
        num_heads = 8
        max_seq_len = 100
        head_dim = 64
        
        cache = KVCache(batch_size, num_heads, max_seq_len, head_dim)
        
        assert cache.batch_size == batch_size
        assert cache.num_heads == num_heads
        assert cache.max_seq_len == max_seq_len
        assert cache.head_dim == head_dim
        assert cache.cache_len == 0
    
    def test_update(self):
        """测试更新缓存"""
        batch_size = 2
        num_heads = 8
        max_seq_len = 100
        head_dim = 64
        
        cache = KVCache(batch_size, num_heads, max_seq_len, head_dim)
        
        # 第一次更新
        seq_len = 10
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        k_cached, v_cached = cache.update(k, v)
        
        assert cache.cache_len == seq_len
        assert k_cached.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v_cached.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_incremental_update(self):
        """测试增量更新"""
        batch_size = 2
        num_heads = 8
        max_seq_len = 100
        head_dim = 64
        
        cache = KVCache(batch_size, num_heads, max_seq_len, head_dim)
        
        # Prefill
        prompt_len = 10
        k_prompt = torch.randn(batch_size, num_heads, prompt_len, head_dim)
        v_prompt = torch.randn(batch_size, num_heads, prompt_len, head_dim)
        cache.update(k_prompt, v_prompt)
        
        assert cache.cache_len == prompt_len
        
        # Decode
        for i in range(5):
            k_new = torch.randn(batch_size, num_heads, 1, head_dim)
            v_new = torch.randn(batch_size, num_heads, 1, head_dim)
            k_cached, v_cached = cache.update(k_new, v_new)
            
            expected_len = prompt_len + i + 1
            assert cache.cache_len == expected_len
            assert k_cached.shape == (batch_size, num_heads, expected_len, head_dim)
    
    def test_get(self):
        """测试获取缓存"""
        batch_size = 2
        num_heads = 8
        max_seq_len = 100
        head_dim = 64
        
        cache = KVCache(batch_size, num_heads, max_seq_len, head_dim)
        
        seq_len = 10
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(k, v)
        
        k_cached, v_cached = cache.get()
        
        assert k_cached.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v_cached.shape == (batch_size, num_heads, seq_len, head_dim)
    
    def test_reset(self):
        """测试重置缓存"""
        batch_size = 2
        num_heads = 8
        max_seq_len = 100
        head_dim = 64
        
        cache = KVCache(batch_size, num_heads, max_seq_len, head_dim)
        
        # 更新缓存
        seq_len = 10
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(k, v)
        
        assert cache.cache_len == seq_len
        
        # 重置
        cache.reset()
        
        assert cache.cache_len == 0


class TestMultiHeadAttentionWithCache:
    """测试带 KV Cache 的 Multi-Head Attention"""
    
    def test_without_cache(self):
        """测试不使用缓存"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        max_seq_len = 100
        
        x = torch.randn(batch_size, seq_len, d_model)
        model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
        
        output, _ = model(x, x, x, use_cache=False)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert model.kv_cache is None
    
    def test_with_cache_prefill(self):
        """测试使用缓存 - Prefill"""
        batch_size = 2
        prompt_len = 10
        d_model = 512
        num_heads = 8
        max_seq_len = 100
        
        prompt = torch.randn(batch_size, prompt_len, d_model)
        model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
        
        output, _ = model(prompt, prompt, prompt, use_cache=True, start_pos=0)
        
        assert output.shape == (batch_size, prompt_len, d_model)
        assert model.kv_cache is not None
        assert model.kv_cache.cache_len == prompt_len
    
    def test_with_cache_decode(self):
        """测试使用缓存 - Decode"""
        batch_size = 2
        prompt_len = 10
        d_model = 512
        num_heads = 8
        max_seq_len = 100
        gen_len = 5
        
        model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
        
        # Prefill
        prompt = torch.randn(batch_size, prompt_len, d_model)
        model(prompt, prompt, prompt, use_cache=True, start_pos=0)
        
        # Decode
        for i in range(gen_len):
            new_token = torch.randn(batch_size, 1, d_model)
            output, _ = model(new_token, new_token, new_token, use_cache=True)
            
            assert output.shape == (batch_size, 1, d_model)
            assert model.kv_cache.cache_len == prompt_len + i + 1
    
    def test_reset_cache(self):
        """测试重置缓存"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        max_seq_len = 100
        
        x = torch.randn(batch_size, seq_len, d_model)
        model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
        
        # 使用缓存
        model(x, x, x, use_cache=True)
        assert model.kv_cache.cache_len == seq_len
        
        # 重置
        model.reset_cache()
        assert model.kv_cache.cache_len == 0
    
    def test_cache_consistency(self):
        """测试缓存一致性"""
        batch_size = 1
        prompt_len = 5
        d_model = 64
        num_heads = 4
        max_seq_len = 100
        
        model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
        model.eval()
        
        # 生成完整序列
        full_seq = torch.randn(batch_size, prompt_len + 3, d_model)
        
        with torch.no_grad():
            # 方法 1: 不使用缓存，一次性处理
            output_no_cache, _ = model(full_seq, full_seq, full_seq, use_cache=False)
        
        # 方法 2: 使用缓存，分步处理
        model.reset_cache()
        
        with torch.no_grad():
            # Prefill
            prompt = full_seq[:, :prompt_len, :]
            output_prefill, _ = model(prompt, prompt, prompt, use_cache=True, start_pos=0)
            
            # Decode
            outputs = [output_prefill]
            for i in range(3):
                new_token = full_seq[:, prompt_len + i:prompt_len + i + 1, :]
                output_decode, _ = model(new_token, new_token, new_token, use_cache=True)
                outputs.append(output_decode)
            
            output_with_cache = torch.cat(outputs, dim=1)
        
        # 两种方法的输出应该接近
        assert torch.allclose(output_no_cache, output_with_cache, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
