"""
Flash Attention 实现

核心优化技术:
1. Tiling: 分块计算 Q·K^T，避免存储完整的注意力矩阵
2. Online Softmax: 增量计算 softmax，避免两次遍历
3. Recomputation: 反向传播时重新计算，节省内存

参考:
- Flash Attention 论文: https://arxiv.org/abs/2205.14135
- XQA 实现: cpp/kernels/xqa/mha.cu
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def online_softmax_merge(
    old_max: torch.Tensor,
    old_sum: torch.Tensor,
    new_max: torch.Tensor,
    new_sum: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    合并两个 online softmax 的结果
    
    Args:
        old_max: 旧的最大值 [..., N]
        old_sum: 旧的求和 [..., N]
        new_max: 新的最大值 [..., N]
        new_sum: 新的求和 [..., N]
    
    Returns:
        merged_max: 合并后的最大值
        merged_sum: 合并后的求和
    """
    # 计算全局最大值（数值稳定性）
    merged_max = torch.maximum(old_max, new_max)
    
    # 归一化并合并
    old_sum_corrected = old_sum * torch.exp(old_max - merged_max)
    new_sum_corrected = new_sum * torch.exp(new_max - merged_max)
    merged_sum = old_sum_corrected + new_sum_corrected
    
    return merged_max, merged_sum


def flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    causal_mask: bool = False,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash Attention 前向传播
    
    基于论文算法实现，使用 online softmax 进行增量计算
    
    Args:
        Q: Query tensor [batch, num_heads, seq_len_q, head_dim]
        K: Key tensor [batch, num_heads, seq_len_k, head_dim]
        V: Value tensor [batch, num_heads, seq_len_k, head_dim]
        block_size: Tile 大小（建议 64-128）
        causal_mask: 是否使用因果 mask
        scale: 缩放因子，默认 1/sqrt(head_dim)
    
    Returns:
        output: [batch, num_heads, seq_len_q, head_dim]
        softmax_lse: softmax 的 log-sum-exp [batch, num_heads, seq_len_q]
    """
    batch, num_heads, seq_len_q, head_dim = Q.shape
    seq_len_k = K.shape[2]
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # 初始化输出和统计量
    output = torch.zeros_like(Q)
    m = torch.full(
        (batch, num_heads, seq_len_q),
        float('-inf'),
        device=Q.device,
        dtype=Q.dtype
    )
    l = torch.zeros(
        (batch, num_heads, seq_len_q),
        device=Q.device,
        dtype=Q.dtype
    )
    
    # 分块处理 Q
    for i in range(0, seq_len_q, block_size):
        q_block_end = min(i + block_size, seq_len_q)
        q_block = Q[:, :, i:q_block_end, :]  # [B, H, Br, D]
        
        # 初始化当前 Q block 的累积值
        o_i = torch.zeros_like(q_block)  # [B, H, Br, D]
        m_i = torch.full(
            (batch, num_heads, q_block_end - i),
            float('-inf'),
            device=Q.device,
            dtype=Q.dtype
        )
        l_i = torch.zeros(
            (batch, num_heads, q_block_end - i),
            device=Q.device,
            dtype=Q.dtype
        )
        
        # 分块处理 K, V
        for j in range(0, seq_len_k, block_size):
            k_block_end = min(j + block_size, seq_len_k)
            k_block = K[:, :, j:k_block_end, :]  # [B, H, Bc, D]
            v_block = V[:, :, j:k_block_end, :]  # [B, H, Bc, D]
            
            # 计算注意力分数: S = Q @ K^T * scale
            S = torch.einsum('bhqd,bhkd->bhqk', q_block, k_block) * scale  # [B, H, Br, Bc]
            
            # 应用因果 mask（如果需要）
            if causal_mask:
                q_indices = torch.arange(i, q_block_end, device=Q.device)
                k_indices = torch.arange(j, k_block_end, device=K.device)
                mask = (q_indices[:, None] >= k_indices[None, :])
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Br, Bc]
                S = S.masked_fill(~mask, float('-inf'))
            
            # 计算当前 block 的 max
            m_ij = S.max(dim=-1)[0]  # [B, H, Br]
            
            # 计算新的 max
            m_i_new = torch.maximum(m_i, m_ij)  # [B, H, Br]
            
            # 计算 exp(S - m_i_new)
            p = torch.exp(S - m_i_new.unsqueeze(-1))  # [B, H, Br, Bc]
            
            # 计算当前 block 的 sum
            l_ij = p.sum(dim=-1)  # [B, H, Br]
            
            # 更新累积的 sum: l_i_new = exp(m_i - m_i_new) * l_i + l_ij
            l_i_new = torch.exp(m_i - m_i_new) * l_i + l_ij  # [B, H, Br]
            
            # 更新输出: o_i = exp(m_i - m_i_new) * o_i + p @ v
            # 注意：这里不除以 l_i_new，最后统一归一化
            alpha = torch.exp(m_i - m_i_new).unsqueeze(-1)  # [B, H, Br, 1]
            o_i = alpha * o_i + torch.einsum('bhqk,bhkd->bhqd', p, v_block)  # [B, H, Br, D]
            
            # 更新统计量
            m_i = m_i_new
            l_i = l_i_new
        
        # 归一化输出
        o_i = o_i / l_i.unsqueeze(-1)  # [B, H, Br, D]
        
        # 保存结果
        output[:, :, i:q_block_end, :] = o_i
        m[:, :, i:q_block_end] = m_i
        l[:, :, i:q_block_end] = l_i
    
    # 计算 log-sum-exp: lse = log(l) + m
    lse = torch.log(l) + m
    
    return output, lse


class FlashAttention(nn.Module):
    """
    Flash Attention 模块
    
    相比标准 Attention 的优势:
    1. 内存复杂度: O(N²) -> O(N) (不存储完整注意力矩阵)
    2. 计算效率: 更好的内存访问模式
    3. 数值稳定性: 使用 online softmax
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        block_size: int = 64,
        causal: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            head_dim: 每个头的维度（默认 d_model // num_heads）
            block_size: Flash Attention 的 tile 大小
            causal: 是否使用因果 mask
            dropout: Dropout 比率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or (d_model // num_heads)
        self.block_size = block_size
        self.causal = causal
        self.dropout = dropout
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_heads * self.head_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model)
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 [batch, seq_len, d_model]
            mask: 可选的 mask [batch, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, d_model]
            lse: softmax log-sum-exp [batch, num_heads, seq_len]
        """
        batch, seq_len, _ = x.shape
        
        # 投影到 Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Flash Attention 计算
        output, lse = flash_attention_forward(
            Q, K, V,
            block_size=self.block_size,
            causal_mask=self.causal
        )
        
        # 转回 [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.num_heads * self.head_dim)
        
        # 输出投影
        output = self.out_proj(output)
        output = self.dropout_layer(output)
        
        return output, lse


def compare_attention_implementations(
    batch_size: int = 2,
    seq_len: int = 1024,
    d_model: int = 512,
    num_heads: int = 8,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    对比标准 Attention 和 Flash Attention 的内存占用和速度
    """
    from src.attention import MultiHeadAttention
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 标准 Attention
    standard_attn = MultiHeadAttention(d_model, num_heads).to(device)
    
    # Flash Attention
    flash_attn = FlashAttention(d_model, num_heads, block_size=64, causal=False).to(device)
    
    # ⚠️ 复制权重，确保两个模型使用相同的参数
    # 这样才能公平对比两种实现的数值一致性
    flash_attn.q_proj.weight.data = standard_attn.W_q.weight.data.clone()
    flash_attn.k_proj.weight.data = standard_attn.W_k.weight.data.clone()
    flash_attn.v_proj.weight.data = standard_attn.W_v.weight.data.clone()
    flash_attn.out_proj.weight.data = standard_attn.W_o.weight.data.clone()
    
    flash_attn.q_proj.bias.data = standard_attn.W_q.bias.data.clone()
    flash_attn.k_proj.bias.data = standard_attn.W_k.bias.data.clone()
    flash_attn.v_proj.bias.data = standard_attn.W_v.bias.data.clone()
    flash_attn.out_proj.bias.data = standard_attn.W_o.bias.data.clone()
    
    print("=" * 60)
    print("Attention 实现对比")
    print("=" * 60)
    print(f"配置: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
    print()
    
    # 标准 Attention
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    import time
    start = time.time()
    with torch.no_grad():
        out_standard, _ = standard_attn(x, x, x)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_standard = time.time() - start
    
    if device == 'cuda':
        mem_standard = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Flash Attention
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        out_flash, _ = flash_attn(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_flash = time.time() - start
    
    if device == 'cuda':
        mem_flash = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # 验证输出相似性
    max_diff = (out_standard - out_flash).abs().max().item()
    mean_diff = (out_standard - out_flash).abs().mean().item()
    
    print("标准 Attention:")
    print(f"  时间: {time_standard*1000:.2f} ms")
    if device == 'cuda':
        print(f"  峰值内存: {mem_standard:.2f} MB")
    print()
    
    print("Flash Attention:")
    print(f"  时间: {time_flash*1000:.2f} ms")
    if device == 'cuda':
        print(f"  峰值内存: {mem_flash:.2f} MB")
        print(f"  内存节省: {(1 - mem_flash/mem_standard)*100:.1f}%")
    print(f"  加速比: {time_standard/time_flash:.2f}x")
    print()
    
    print("数值差异:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print()
    
    if max_diff < 1e-3:
        print("✓ 两种实现输出基本一致")
    else:
        print("⚠ 输出存在较大差异，需要检查实现")


if __name__ == "__main__":
    # 简单测试
    if torch.cuda.is_available():
        compare_attention_implementations(
            batch_size=2,
            seq_len=512,
            d_model=512,
            num_heads=8,
            device='cuda'
        )
    else:
        print("需要 CUDA 支持才能进行完整测试")
