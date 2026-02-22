"""
Paged KV Cache 实现

核心思想:
- 将 KV Cache 分成固定大小的页面（pages）
- 使用页面表（page table）管理每个序列的页面映射
- 支持动态分配和回收，减少内存碎片

参考:
- vLLM PagedAttention: https://github.com/vllm-project/vllm
- XQA Paged KV Cache: cpp/kernels/xqa/defines.h (TOKENS_PER_PAGE)
"""

import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class PagedKVCache:
    """
    Paged KV Cache 实现
    
    内存布局:
    - 全局页面池: [num_pages, num_heads, page_size, head_dim]
    - 页面表: {seq_id: [page_id1, page_id2, ...]}
    - 空闲页面列表: [page_id1, page_id2, ...]
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        num_pages: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: str = 'cpu'  # 默认使用 CPU（避免 CUDA 兼容性问题）
    ):
        """
        Args:
            num_heads: KV 头数量
            head_dim: 每个头的维度
            page_size: 每个页面存储的 token 数（通常 16, 32, 64）
            num_pages: 总页面数
            dtype: 数据类型
            device: 设备（默认 'cpu'）
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = num_pages
        self.dtype = dtype
        self.device = device
        
        # 全局页面池
        # K Cache: [num_pages, num_heads, page_size, head_dim]
        self.k_cache = torch.zeros(
            num_pages, num_heads, page_size, head_dim,
            dtype=dtype, device=device
        )
        
        # V Cache: [num_pages, num_heads, page_size, head_dim]
        self.v_cache = torch.zeros(
            num_pages, num_heads, page_size, head_dim,
            dtype=dtype, device=device
        )
        
        # 页面表: {seq_id: [page_id1, page_id2, ...]}
        self.page_table: Dict[int, List[int]] = {}
        
        # 空闲页面列表
        self.free_pages = list(range(num_pages))
        
        # 序列长度记录: {seq_id: current_length}
        self.seq_lengths: Dict[int, int] = {}
    
    def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        为序列分配页面
        
        Args:
            seq_id: 序列 ID
            num_tokens: 需要的 token 数量
        
        Returns:
            allocated_pages: 分配的页面 ID 列表
        """
        num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages_needed:
            raise RuntimeError(
                f"Not enough free pages. "
                f"Need {num_pages_needed}, available {len(self.free_pages)}"
            )
        
        # 分配页面
        allocated_pages = self.free_pages[:num_pages_needed]
        self.free_pages = self.free_pages[num_pages_needed:]
        
        # 更新页面表
        self.page_table[seq_id] = allocated_pages
        self.seq_lengths[seq_id] = 0
        
        return allocated_pages
    
    def free(self, seq_id: int):
        """
        释放序列占用的页面
        
        Args:
            seq_id: 序列 ID
        """
        if seq_id not in self.page_table:
            return
        
        # 回收页面
        freed_pages = self.page_table[seq_id]
        self.free_pages.extend(freed_pages)
        
        # 清理页面表
        del self.page_table[seq_id]
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]
    
    def update(
        self,
        seq_id: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int
    ):
        """
        更新 KV Cache
        
        Args:
            seq_id: 序列 ID
            k_new: 新的 K [num_heads, num_tokens, head_dim]
            v_new: 新的 V [num_heads, num_tokens, head_dim]
            start_pos: 起始位置
        """
        if seq_id not in self.page_table:
            raise RuntimeError(f"Sequence {seq_id} not allocated")
        
        num_tokens = k_new.shape[1]
        page_ids = self.page_table[seq_id]
        
        # 更新每个 token
        for i in range(num_tokens):
            pos = start_pos + i
            page_idx = pos // self.page_size
            token_in_page = pos % self.page_size
            
            if page_idx >= len(page_ids):
                # 需要分配新页面
                if len(self.free_pages) == 0:
                    raise RuntimeError("No free pages available for expansion")
                new_page_id = self.free_pages.pop(0)
                page_ids.append(new_page_id)
                self.page_table[seq_id] = page_ids
            
            page_id = page_ids[page_idx]
            
            # 写入 K, V
            self.k_cache[page_id, :, token_in_page, :] = k_new[:, i, :]
            self.v_cache[page_id, :, token_in_page, :] = v_new[:, i, :]
        
        # 更新序列长度
        self.seq_lengths[seq_id] = max(
            self.seq_lengths.get(seq_id, 0),
            start_pos + num_tokens
        )
    
    def get_kv(
        self,
        seq_id: int,
        start_pos: int,
        end_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定范围的 KV
        
        Args:
            seq_id: 序列 ID
            start_pos: 起始位置
            end_pos: 结束位置（不包含）
        
        Returns:
            k: [num_heads, end_pos - start_pos, head_dim]
            v: [num_heads, end_pos - start_pos, head_dim]
        """
        if seq_id not in self.page_table:
            raise RuntimeError(f"Sequence {seq_id} not allocated")
        
        page_ids = self.page_table[seq_id]
        num_tokens = end_pos - start_pos
        
        # 收集 KV
        k_list = []
        v_list = []
        
        for i in range(num_tokens):
            pos = start_pos + i
            page_idx = pos // self.page_size
            token_in_page = pos % self.page_size
            
            if page_idx >= len(page_ids):
                # 超出范围，返回零
                k_list.append(torch.zeros(
                    self.num_heads, 1, self.head_dim,
                    dtype=self.dtype, device=self.device
                ))
                v_list.append(torch.zeros(
                    self.num_heads, 1, self.head_dim,
                    dtype=self.dtype, device=self.device
                ))
            else:
                page_id = page_ids[page_idx]
                k_list.append(self.k_cache[page_id, :, token_in_page:token_in_page+1, :])
                v_list.append(self.v_cache[page_id, :, token_in_page:token_in_page+1, :])
        
        k = torch.cat(k_list, dim=1)  # [num_heads, num_tokens, head_dim]
        v = torch.cat(v_list, dim=1)  # [num_heads, num_tokens, head_dim]
        
        return k, v
    
    def get_statistics(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            stats: 统计信息字典
        """
        total_pages = self.num_pages
        used_pages = sum(len(pages) for pages in self.page_table.values())
        free_pages = len(self.free_pages)
        
        total_tokens = sum(self.seq_lengths.values())
        
        return {
            'total_pages': total_pages,
            'used_pages': used_pages,
            'free_pages': free_pages,
            'usage_ratio': used_pages / total_pages if total_pages > 0 else 0,
            'total_tokens': total_tokens,
            'num_sequences': len(self.page_table),
            'avg_tokens_per_seq': total_tokens / len(self.page_table) if len(self.page_table) > 0 else 0
        }


class PagedAttentionWithCache:
    """
    使用 Paged KV Cache 的 Attention 模块
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        page_size: int = 16,
        num_pages: int = 1024,
        dtype: torch.dtype = torch.float16
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            head_dim: 每个头的维度
            page_size: 页面大小
            num_pages: 总页面数
            dtype: 数据类型
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or (d_model // num_heads)
        self.page_size = page_size
        
        # 创建 Paged KV Cache（默认使用 CPU）
        self.kv_cache = PagedKVCache(
            num_heads=num_heads,
            head_dim=self.head_dim,
            page_size=page_size,
            num_pages=num_pages,
            dtype=dtype,
            device='cpu'  # 强制使用 CPU
        )
        
        # Q, K, V 投影（简化版，实际应该使用 nn.Linear）
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(
        self,
        seq_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Args:
            seq_id: 序列 ID
            q: Query [batch, seq_len, num_heads, head_dim]
            k: Key [batch, seq_len, num_heads, head_dim]
            v: Value [batch, seq_len, num_heads, head_dim]
            start_pos: 起始位置
            use_cache: 是否使用缓存
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        batch, seq_len, num_heads, head_dim = q.shape
        
        # 转置为 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if use_cache and start_pos > 0:
            # 从缓存中获取历史的 K, V
            k_cached, v_cached = self.kv_cache.get_kv(seq_id, 0, start_pos)
            # k_cached: [num_heads, cached_len, head_dim]
            # v_cached: [num_heads, cached_len, head_dim]
            
            # 拼接: [num_heads, cached_len + seq_len, head_dim]
            k = torch.cat([k_cached.unsqueeze(0), k], dim=2)
            v = torch.cat([v_cached.unsqueeze(0), v], dim=2)
        
        # 计算注意力
        # Q: [batch, num_heads, seq_len, head_dim]
        # K: [batch, num_heads, total_len, head_dim]
        # S = Q @ K^T: [batch, num_heads, seq_len, total_len]
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Attention @ V: [batch, num_heads, seq_len, head_dim]
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # 更新缓存
        if use_cache:
            # k, v: [batch, num_heads, seq_len, head_dim]
            # 需要转换为 [num_heads, seq_len, head_dim]
            # 🔧 修复：取第一个 batch 的数据
            k_to_cache = k[0, :, -seq_len:, :]  # [num_heads, seq_len, head_dim]
            v_to_cache = v[0, :, -seq_len:, :]  # [num_heads, seq_len, head_dim]
            
            self.kv_cache.update(seq_id, k_to_cache, v_to_cache, start_pos)
        
        # 转回 [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)
        
        return output


def demo_paged_kv_cache():
    """演示 Paged KV Cache 的使用"""
    print("=" * 60)
    print("Paged KV Cache 演示")
    print("=" * 60)
    
    num_heads = 8
    head_dim = 64
    page_size = 16
    num_pages = 64
    
    # 创建缓存（使用 CPU）
    cache = PagedKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_pages=num_pages,
        device='cpu'
    )
    
    print(f"配置: num_heads={num_heads}, head_dim={head_dim}")
    print(f"      page_size={page_size}, num_pages={num_pages}")
    print()
    
    # 分配序列
    seq_id = 0
    num_tokens = 50
    cache.allocate(seq_id, num_tokens)
    print(f"序列 {seq_id}: 分配 {num_tokens} tokens")
    
    # 更新缓存
    k_new = torch.randn(num_heads, 10, head_dim)
    v_new = torch.randn(num_heads, 10, head_dim)
    cache.update(seq_id, k_new, v_new, start_pos=0)
    print(f"更新: 写入 10 tokens (位置 0-9)")
    
    # 获取 KV
    k, v = cache.get_kv(seq_id, 0, 10)
    print(f"读取: 获取 10 tokens (位置 0-9)")
    print(f"      K shape: {k.shape}, V shape: {v.shape}")
    
    # 统计信息
    stats = cache.get_statistics()
    print()
    print("统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 释放序列
    cache.free(seq_id)
    print()
    print(f"释放序列 {seq_id}")
    
    stats_after = cache.get_statistics()
    print(f"释放后空闲页面: {stats_after['free_pages']}")


if __name__ == "__main__":
    demo_paged_kv_cache()
