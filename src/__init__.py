"""
Attention From Scratch - 从零实现 Attention 机制

基础实现:
- attention: Scaled Dot-Product Attention, Multi-Head Attention
- gqa: Grouped Query Attention
- kv_cache: 基础 KV Cache

进阶实现:
- flash_attention: Flash Attention 算法实现
- paged_kv_cache: Paged KV Cache 实现
"""

from .attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
)
from .gqa import GroupedQueryAttention
from .kv_cache import KVCache

# 进阶实现（可选导入）
try:
    from .flash_attention import (
        flash_attention_forward,
        FlashAttention,
        compare_attention_implementations,
    )
    from .paged_kv_cache import (
        PagedKVCache,
        PagedAttentionWithCache,
    )
    __all__ = [
        # 基础实现
        'scaled_dot_product_attention',
        'MultiHeadAttention',
        'GroupedQueryAttention',
        'KVCache',
        # 进阶实现
        'flash_attention_forward',
        'FlashAttention',
        'compare_attention_implementations',
        'PagedKVCache',
        'PagedAttentionWithCache',
    ]
except ImportError:
    # 如果进阶实现有依赖问题，只导出基础实现
    __all__ = [
        'scaled_dot_product_attention',
        'MultiHeadAttention',
        'GroupedQueryAttention',
        'KVCache',
    ]
