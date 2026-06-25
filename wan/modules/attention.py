# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import os
from matplotlib import pyplot as plt
from typing import List
try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]
def _pool_by_1560(a_qk: torch.Tensor, block: List=[1560, 1560]) -> torch.Tensor:
    # a_qk: [Q, K]
    Q, K = a_qk.shape
    Qb, Kb = Q // block[0], K // block[1]
    if Qb <= 0 or Kb <= 0:
        return a_qk
    a_qk = a_qk[:Qb * block[0], :Kb * block[1]]
    return a_qk.view(Qb, block[0], Kb, block[1]).mean(dim=(1, 3))  # [Qb, Kb]


@torch.no_grad()
def _save_attn_png(attn_bh_qk: torch.Tensor, save_path: str, block: List=[1560, 1560], debug_dict: dict = None):
    # 1. 数据准备
    a_qk = attn_bh_qk.float().mean(dim=(0, 1))
    a_pool = _pool_by_1560(a_qk, block).cpu().numpy()
    is_visual_cross = debug_dict is not None and debug_dict.get("visual_cross_attn", False)
    
    if is_visual_cross:
        a_pool = a_pool[:, :len(debug_dict["decode_tokens"])]

    # 2. 计算尺寸并创建画布 (必须在 xticks 之前)
    Q, K = a_pool.shape
    base = 6.0
    aspect = K / max(Q, 1)
    height = base
    width = base * aspect
    width = min(max(width, 6), 24)
    height = min(max(height, 4), 16)
    
    if is_visual_cross:
        width = width * 4
    plt.figure(figsize=(width, height)) # <--- 先创建画布
    plt.imshow(a_pool, aspect="auto")

    # 4. 设置标签 (必须在 figure 之后)
    if is_visual_cross:
        plt.xticks(
            ticks=range(len(debug_dict["decode_tokens"])),
            labels=[t for t in debug_dict["decode_tokens"]],
            fontsize=6,
            rotation=45 # 建议增加旋转，防止 token 太长重叠
        )
    plt.xlabel(f"Key/Value Tokens (History) - Pooled x{block}")
    plt.ylabel(f"Query Tokens (All Blocks) - Pooled x{block}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[Debug] Saving attention map to {save_path}, shape after pooling: {a_pool.shape}")
    plt.close()

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    debug_dict=None,
):
    # if debug_dict is not None and debug_dict["block_index"] == 0:
        # print(f"[Attention DEBUG] q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, q_lens={q_lens}, k_lens={k_lens}")
    if (debug_dict==None or not debug_dict['visualize_attention'])and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        # q,k,v: [B, S, H, D] -> SDPA expects [B, H, S, D]
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        if debug_dict is not None and debug_dict["block_index"] ==0 and debug_dict['visualize_attention']:

            save_path = debug_dict.get("save_path", f"./visual-output/self-attn-{debug_dict['current_chunk']}-{debug_dict['current_step']}.npy")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

            # scale：默认 1/sqrt(D)
            D = q.shape[-1]
            scale = softmax_scale if softmax_scale is not None else (1.0 / (D ** 0.5))
            if q_scale is not None:
                scale = scale * q_scale

            # attn = softmax(qk^T)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)  # [B,H,S_Q,D] x [B,H,D,S_K] = [B,H,S_Q,S_K]
            attn = torch.softmax(logits, dim=-1)

            import numpy as np
            np.save(save_path, attn.cpu().float().numpy())
            print(f"[Debug] Saved attention map to {save_path}, shape={attn.shape}")
            
            # 保存attention output (out)
            chunk_idx = debug_dict.get('current_chunk', 0)
            step_idx = debug_dict.get('current_step', 0)
            out_save_path = os.path.join(os.path.dirname(save_path) or ".", 
                                         f'self-attn-out-{chunk_idx}-{step_idx}.npy')
            np.save(out_save_path, out.cpu().float().numpy())
            print(f"[Debug] Saved attention output to {out_save_path}, shape={out.shape}")
        return out
