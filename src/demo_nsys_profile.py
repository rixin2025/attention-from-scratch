"""
CS336 Chapter 2 — Nsight Systems 性能剖析 Demo
================================================
用 NVTX 标记三种 Attention 实现的前向过程，在 nsys timeline 中直观对比。

要求: Nsight Systems >= 2025.2 (Blackwell 支持), 推荐 >= 2026.3.1 (cuda-hw 正式版)

用法:
  # 命令行 profiling（推荐；2026.3.1+ 默认使用硬件级 CUDA trace）
  nsys profile -o attention_nsys --force-overwrite true \
    --trace=cuda,nvtx,osrt \
    python chapter2/demo_nsys_profile.py

  # 回退到软件级 CUDA trace（如果硬件 trace 异常）
  nsys profile -o attention_nsys --force-overwrite true \
    --trace=cuda-sw,nvtx,osrt \
    python chapter2/demo_nsys_profile.py

  # 仅 warmup 不 profile（验证脚本能跑通）
  python chapter2/demo_nsys_profile.py

  # 用 NVTX push/pop 精确控制采集范围（减少 profile 文件体积）
  python chapter2/demo_nsys_profile.py --nvtx-gate

在 WSL 中运行:
  cd ~/cs336_demo && nsys profile -o attention_nsys --force-overwrite true \
    uv run python /mnt/c/workspace/github-code/cs336_note_and_hw/chapter2/demo_nsys_profile.py

注意:
  - Windows WDDM 下，kernel 耗时是 CPU 端 API 估算值，非 GPU 硬件时间戳
  - 如需精确 kernel 级性能分析（SM 占用率等），需 Nsight Compute + Linux/TCC 模式
"""

import torch
import torch.nn.functional as F
import time
import argparse

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# 优先用 GPU 1（显存更空闲），否则回退到 GPU 0
if torch.cuda.device_count() > 1:
    DEVICE = "cuda:1"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Triton Flash Attention Kernel
# ══════════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        i = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(i * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
            offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
            offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
            offsets=(i * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES, 1), strides=(stride_lq, 1),
            offsets=(i * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, 1), order=(1, 0),
        )

        Q_i = tl.load(Q_block_ptr)
        O_i_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        L_i_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
        M_i_acc = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype=tl.float32)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            K_j = tl.load(K_block_ptr)
            V_j = tl.load(V_block_ptr)
            S_ij = tl.dot(Q_i, K_j.T) * scale
            if is_causal:
                q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
                k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
                S_ij = tl.where(q_idx >= k_idx, S_ij, -1e6)
            M_ij = tl.max(S_ij, axis=1, keep_dims=True)
            M_i_new = tl.maximum(M_i_acc, M_ij)
            P_ij = tl.exp(S_ij - M_ij)
            L_i_new = (tl.exp(M_i_acc - M_i_new) * L_i_acc
                     + tl.exp(M_ij - M_i_new) * tl.sum(P_ij, axis=1, keep_dims=True))
            P_ij_cast = P_ij.to(V_j.dtype)
            O_i_new = (tl.exp(M_i_acc - M_i_new) * O_i_acc
                     + tl.exp(M_ij - M_i_new) * tl.dot(P_ij_cast, V_j))
            M_i_acc, O_i_acc, L_i_acc = M_i_new, O_i_new, L_i_new
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        O_i = O_i_acc / L_i_acc
        tl.store(O_block_ptr, O_i.to(O_block_ptr.dtype.element_ty))
        tl.store(L_block_ptr, M_i_acc + tl.log(L_i_acc))

    def triton_flash_attention(q, k, v, is_causal=True):
        batch_size, Nq, d = q.shape
        Nk = k.shape[1]

        # Triton 要求 contiguous tensor 才能获取有效 CUDA 指针
        # nsys profiling 时 CUDA API interposition 可能加剧此问题
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # shared_mem ≈ d*(6*Bq + 4*Bk). HW limit ≈ 99KB. 留 10% 余量 ~90KB.
        if d <= 64:       Bq, Bk = 64, 64    # d=64 →  64*160=40,960 B ✓
        elif d <= 128:    Bq, Bk = 32, 32    # d=128→ 128*80 =10,240 B ✓
        elif d <= 512:    Bq, Bk = 16, 16    # d=512→ 512*160=81,920 B ✓
        else:             Bq, Bk =  8,  8    # d=1024→1024*80=81,920 B ✓

        Nq_padded = ((Nq + Bq - 1) // Bq) * Bq
        if Nq != Nq_padded:
            q = F.pad(q, (0, 0, 0, Nq_padded - Nq)).contiguous()
            k = F.pad(k, (0, 0, 0, Nq_padded - Nq)).contiguous()
            v = F.pad(v, (0, 0, 0, Nq_padded - Nq)).contiguous()

        scale = 1.0 / d**0.5
        O = torch.zeros_like(q)
        L = torch.zeros(batch_size, Nq_padded, device=q.device)
        grid = (Nq_padded // Bq, batch_size)
        flash_fwd_kernel[grid](
            q, k, v, O, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )
        return O[:, :Nq, :]


def naive_attention(q, k, v, is_causal=True):
    d = q.shape[-1]
    S = torch.matmul(q, k.transpose(-2, -1)) / d**0.5
    if is_causal:
        Nq, Nk = q.shape[-2], k.shape[-2]
        S = S.masked_fill(
            ~torch.tril(torch.ones(Nq, Nk, device=q.device, dtype=torch.bool)),
            float('-inf'))
    return torch.matmul(F.softmax(S, dim=-1), v)


def pytorch_sdpa(q, k, v, is_causal=True):
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    v = v.unsqueeze(1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=is_causal,
        scale=1.0 / q.shape[-1]**0.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# NVTX 辅助
# ══════════════════════════════════════════════════════════════════════════════

def nvtx_range(name, color=0, enabled=True):
    """创建一个 NVTX range context manager，enabled=False 时退化为 no-op"""
    if not enabled or not DEVICE.startswith("cuda"):
        from contextlib import nullcontext
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def nvtx_push(name):
    """手动 push NVTX range（无法用 with 语句时）"""
    if DEVICE.startswith("cuda"):
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if DEVICE.startswith("cuda"):
        torch.cuda.nvtx.range_pop()


def cuda_sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def profiled_step(q, k, v, is_causal=True):
   with nvtx_range("01_Naive_Attention"):
        _ = naive_attention(q, k, v)
   cuda_sync()

    # ── 2. PyTorch SDPA ───────────────────────────────────────────────
   with nvtx_range("02_PyTorch_SDPA"):
        _ = pytorch_sdpa(q, k, v)
   cuda_sync()

    # ── 3. Triton Flash Attention ────────────────────────────────────
   if HAS_TRITON:
        with nvtx_range("03_Triton_FA"):
            _ = triton_flash_attention(q, k, v)
        cuda_sync()

def run_profile(args):
    # Triton 需要在正确的 CUDA 上下文中访问 tensor 指针，否则会报
    # "Pointer argument cannot be accessed from Triton (cpu tensor?)"
    if DEVICE.startswith("cuda"):
        torch.cuda.set_device(DEVICE)

    print(f"设备: {DEVICE}  |  Triton: {'可用' if HAS_TRITON else '不可用'}")
    print(f"配置: B={args.batch}, N={args.seq_len}, d={args.d_model}")
    print(f"NVTX 精确控制: {'开启' if args.nvtx_gate else '关闭'}")
    print()

    B, N, d = args.batch, args.seq_len, args.d_model

    # 准备数据（fp16 更接近真实推理场景）
    dtype = torch.float16 if args.fp16 else torch.float32
    q = torch.randn(B, N, d, device=DEVICE, dtype=dtype)
    k = torch.randn(B, N, d, device=DEVICE, dtype=dtype)
    v = torch.randn(B, N, d, device=DEVICE, dtype=dtype)

    # 确保 N 能被 Triton tile size 整除
    tile = 64 if d <= 64 else 32
    if N % tile != 0:
        print(f"⚠️  N={N} 不能被 tile={tile} 整除, Triton 会内部 padding")

    print("Warming up...")
    for _ in range(3):
        _ = naive_attention(q, k, v)
        _ = pytorch_sdpa(q, k, v)
        if HAS_TRITON:
            _ = triton_flash_attention(q, k, v)
    cuda_sync()

    # ── 精确 NVTX gate（可选）─────────────────────────────────────────
    if args.nvtx_gate:
        print("启动 CUDA profiler gate，nsys 将只采集 gate 内的区间...")
        torch.cuda.cudart().cudaProfilerStart()

    reps = args.repeat

    for i in range(reps):
        with nvtx_range(f"step_{i}"):
            profiled_step(q, k, v)

    if args.nvtx_gate:
        torch.cuda.cudart().cudaProfilerStop()

    # # ── 4. 正确性检查 ─────────────────────────────────────────────────
    # print("\n正确性检查 (以 SDPA 为基准):")
    # ref = o_sdpa.float()
    # diff_naive = (o_naive.float() - ref).abs().max().item()
    # print(f"  Naive  vs SDPA  max diff: {diff_naive:.6e}  {'[OK]' if diff_naive < 1e-2 else '[FAIL]'}")

    # if HAS_TRITON:
    #     diff_triton = (o_triton.float() - ref).abs().max().item()
    #     print(f"  Triton vs SDPA  max diff: {diff_triton:.6e}  {'[OK]' if diff_triton < 5e-3 else '[FAIL]'}")

    print("\nDone. 用 Nsight Systems 打开 .nsys-rep 文件查看 GPU timeline。")


def main():
    parser = argparse.ArgumentParser(description="Nsight Systems Attention Profiling Demo")
    parser.add_argument("--batch",       type=int,   default=2)
    parser.add_argument("--seq-len",     type=int,   default=2048)
    parser.add_argument("--d-model",     type=int,   default=256)
    parser.add_argument("--repeat",      type=int,   default=6)
    parser.add_argument("--fp16",        action="store_true", default=True)
    parser.add_argument("--fp32",        action="store_true",
                        help="使用 float32 (默认 float16)")
    parser.add_argument("--nvtx-gate",   action="store_true",
                        help="用 cudaProfilerStart/Stop 精确控制采集范围，缩小 profile 文件")
    args = parser.parse_args()

    if args.fp32:
        args.fp16 = False

    run_profile(args)


if __name__ == "__main__":
    main()
