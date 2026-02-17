#!/usr/bin/env python3
"""
Attention From Scratch - 演示脚本

展示 MHA、GQA 和 KV Cache 的基本使用
"""

import torch
from src.attention import MultiHeadAttention, create_causal_mask
from src.gqa import GroupedQueryAttention, compare_attention_variants
from src.kv_cache import MultiHeadAttentionWithCache


def demo_multi_head_attention():
    """演示 Multi-Head Attention"""
    print("=" * 60)
    print("1. Multi-Head Attention (MHA) 演示")
    print("=" * 60)
    
    # 配置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建模型
    mha = MultiHeadAttention(d_model, num_heads)
    print(f"\n模型配置: {mha}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output, attn_weights = mha(x, x, x)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 带因果 mask
    mask = create_causal_mask(seq_len)
    output_masked, _ = mha(x, x, x, mask=mask)
    print(f"\n带因果 mask 的输出形状: {output_masked.shape}")
    
    print("\n✓ Multi-Head Attention 演示完成！\n")


def demo_grouped_query_attention():
    """演示 Grouped Query Attention"""
    print("=" * 60)
    print("2. Grouped Query Attention (GQA) 演示")
    print("=" * 60)
    
    # 配置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_q_heads = 32
    num_kv_heads = 8  # 4:1 分组
    
    # 创建模型
    gqa = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
    print(f"\n模型配置: {gqa}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, attn_weights = gqa(x, x, x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 参数量对比
    print("\n参数量和内存对比:")
    compare_attention_variants(d_model=4096, num_heads=32, seq_len=2048)
    
    print("\n✓ Grouped Query Attention 演示完成！\n")


def demo_kv_cache():
    """演示 KV Cache"""
    print("=" * 60)
    print("3. KV Cache 演示")
    print("=" * 60)
    
    # 配置
    batch_size = 1
    d_model = 512
    num_heads = 8
    max_seq_len = 100
    prompt_len = 10
    gen_len = 5
    
    # 创建模型
    model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
    print(f"\n模型配置:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  max_seq_len: {max_seq_len}")
    
    # Prefill 阶段
    print(f"\n--- Prefill 阶段 ---")
    prompt = torch.randn(batch_size, prompt_len, d_model)
    print(f"Prompt 形状: {prompt.shape}")
    
    output, _ = model(prompt, prompt, prompt, use_cache=True, start_pos=0)
    print(f"输出形状: {output.shape}")
    print(f"缓存长度: {model.kv_cache.cache_len}")
    
    # Decode 阶段
    print(f"\n--- Decode 阶段 ---")
    for i in range(gen_len):
        new_token = torch.randn(batch_size, 1, d_model)
        output, _ = model(new_token, new_token, new_token, use_cache=True)
        print(f"Step {i+1}: 生成 1 token, 缓存长度 = {model.kv_cache.cache_len}")
    
    print(f"\n最终缓存长度: {model.kv_cache.cache_len} (prompt {prompt_len} + 生成 {gen_len})")
    
    print("\n✓ KV Cache 演示完成！\n")


def demo_performance_comparison():
    """演示性能对比"""
    print("=" * 60)
    print("4. 性能对比")
    print("=" * 60)
    
    import time
    
    # 配置
    d_model = 512
    num_heads = 8
    max_seq_len = 2048
    prompt_len = 50
    gen_len = 20
    
    model = MultiHeadAttentionWithCache(d_model, num_heads, max_seq_len)
    model.eval()
    
    print(f"\n配置: prompt_len={prompt_len}, gen_len={gen_len}")
    
    # 不使用 KV Cache
    print("\n--- 不使用 KV Cache ---")
    tokens = torch.randn(1, prompt_len, d_model)
    
    start = time.time()
    for i in range(gen_len):
        with torch.no_grad():
            output, _ = model(tokens, tokens, tokens, use_cache=False)
        new_token = torch.randn(1, 1, d_model)
        tokens = torch.cat([tokens, new_token], dim=1)
    time_no_cache = time.time() - start
    
    print(f"总时间: {time_no_cache:.3f}s")
    print(f"平均每个 token: {time_no_cache / gen_len * 1000:.2f}ms")
    
    # 使用 KV Cache
    print("\n--- 使用 KV Cache ---")
    model.reset_cache()
    
    # Prefill
    prompt = torch.randn(1, prompt_len, d_model)
    with torch.no_grad():
        output, _ = model(prompt, prompt, prompt, use_cache=True, start_pos=0)
    
    # Decode
    start = time.time()
    for i in range(gen_len):
        new_token = torch.randn(1, 1, d_model)
        with torch.no_grad():
            output, _ = model(new_token, new_token, new_token, use_cache=True)
    time_with_cache = time.time() - start
    
    print(f"总时间: {time_with_cache:.3f}s")
    print(f"平均每个 token: {time_with_cache / gen_len * 1000:.2f}ms")
    
    # 加速比
    speedup = time_no_cache / time_with_cache
    print(f"\n加速比: {speedup:.2f}x")
    print(f"时间节省: {(1 - 1/speedup) * 100:.1f}%")
    
    print("\n✓ 性能对比完成！\n")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Attention From Scratch - 完整演示")
    print("=" * 60 + "\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行所有演示
    demo_multi_head_attention()
    demo_grouped_query_attention()
    demo_kv_cache()
    demo_performance_comparison()
    
    print("=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行测试: pytest tests/ -v")
    print("2. 学习 Notebooks: jupyter notebook notebooks/")
    print("3. 阅读源码: src/attention.py, src/gqa.py, src/kv_cache.py")
    print("4. 查看文档: README.md, PROJECT_SUMMARY.md")
    print("\n祝学习顺利！🚀\n")


if __name__ == "__main__":
    main()
