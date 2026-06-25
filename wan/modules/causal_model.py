from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import os

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_pruned(x, grid_sizes, freqs, pruning_info, start_frame=0):
    """
    对pruned tokens应用RoPE，使用kept_indices的原始位置信息
    现在每个frame保留相同数量的tokens，可以正常reshape
    
    Args:
        x: [B, F*N_kept, num_heads, head_dim] pruned tokens（每个frame N_kept个）
        grid_sizes: [B, 3] 原始的(F, H, W)
        freqs: RoPE频率参数
        pruning_info: dict 包含kept_indices等信息
        start_frame: 起始帧索引
    
    Returns:
        [B, F*N_kept, num_heads, head_dim] 应用RoPE后的tokens
    """
    B, total_len, n, head_dim = x.shape
    c = head_dim // 2
    
    kept_indices_list = pruning_info['kept_indices']
    F = pruning_info['num_frames']
    tokens_per_frame_kept = pruning_info['tokens_per_frame_kept']
    tokens_per_frame_original = pruning_info['tokens_per_frame_original']
    
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        assert f == F and h * w == tokens_per_frame_original
        
        kept_idx = kept_indices_list[i]  # [F*tokens_per_frame_kept]
        
        # 将kept_idx转换为3D坐标(frame, height, width)
        frame_indices = kept_idx // tokens_per_frame_original
        spatial_idx = kept_idx % tokens_per_frame_original
        height_indices = spatial_idx // w
        width_indices = spatial_idx % w
        
        # 获取对应位置的频率
        freqs_f = freqs[0][start_frame + frame_indices]  # [F*tokens_per_frame_kept, c1]
        freqs_h = freqs[1][height_indices]  # [F*tokens_per_frame_kept, c2]
        freqs_w = freqs[2][width_indices]  # [F*tokens_per_frame_kept, c3]
        freqs_i = torch.cat([freqs_f, freqs_h, freqs_w], dim=1)  # [F*tokens_per_frame_kept, c]
        
        # 应用RoPE
        num_tokens = len(kept_idx)  # F*tokens_per_frame_kept
        x_i = x[i].to(torch.float64)  # [num_tokens, n, head_dim]
        x_i = torch.view_as_complex(x_i.reshape(num_tokens, n, c, 2))
        freqs_i = freqs_i.unsqueeze(1)  # [num_tokens, 1, c]
        
        # 应用旋转
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        
        output.append(x_i)
    
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 block_id=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.block_id = block_id

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        debug_dict=None,
        pruning_info=None  # 🆕 接收pruning信息
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                # 🎯 根据pruning_info选择RoPE函数
                if pruning_info and pruning_info.get('pruned', False):
                    # 使用pruned RoPE，保持原始位置信息
                    roped_query = causal_rope_apply_pruned(
                        q, grid_sizes, freqs, pruning_info, start_frame=0
                    ).type_as(v)
                    roped_key = causal_rope_apply_pruned(
                        k, grid_sizes, freqs, pruning_info, start_frame=0
                    ).type_as(v)
                else:
                    # 正常的RoPE
                    roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                    roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            
            # 🎯 根据pruning_info选择RoPE函数（kv_cache分支）
            if pruning_info and pruning_info.get('pruned', False):
                roped_query = causal_rope_apply_pruned(
                    q, grid_sizes, freqs, pruning_info,
                    start_frame=current_start_frame
                ).type_as(v)
                roped_key = causal_rope_apply_pruned(
                    k, grid_sizes, freqs, pruning_info,
                    start_frame=current_start_frame
                ).type_as(v)
            else:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                # if debug_dict["block_index"] == 0 :
                #     print(f"[KVCache DEBUG] {local_start_index} to {local_end_index}, {sink_tokens=}")
                #     print(f"[KVCache DEBUG] {roped_key.shape=}")
                #     print(f"[KVCache DEBUG] {v.shape=}")
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                # if debug_dict is not None and debug_dict["block_index"] == 0 :
                #     print(f"[KVCache DEBUG] {local_start_index} to {local_end_index}, {sink_tokens=}")
                #     print(f"[KVCache DEBUG] {roped_key.shape=}")
                #     print(f"[KVCache DEBUG] {v.shape=}")
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            # print(f"{max(0, local_end_index - self.max_attention_size)} to {local_end_index}")
            # 
            x = attention( # 对应wan/modules/attention.py中的attention函数
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                debug_dict=debug_dict
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, block_id)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        
        self.ffn_hidden_dim = ffn_dim

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.block_id = block_id

    def _block_internal_prune(self, x, kept_indices_per_frame, num_frames):
        """
        Block内部pruning：在计算delta时对输入进行prune
        
        Args:
            x: [B, F*H*W, C] 完整的输入tokens
            kept_indices_per_frame: List[Tensor], 每个frame的kept indices
            num_frames: int, 帧数
        
        Returns:
            x_pruned: [B, total_kept, C] pruned后的tokens
            flat_kept_indices: [total_kept] 扁平化的全局indices
        """
        B, L, C = x.shape
        tokens_per_frame = L // num_frames
        
        # 收集所有kept indices（转换为全局索引）
        flat_kept_indices = []
        for f_idx in range(num_frames):
            frame_kept = kept_indices_per_frame[f_idx]  # [kept_count]
            global_indices = frame_kept + f_idx * tokens_per_frame
            flat_kept_indices.append(global_indices)
        
        flat_kept_indices = torch.cat(flat_kept_indices, dim=0)  # [total_kept]
        
        # Gather pruned tokens
        x_pruned = x[:, flat_kept_indices, :]  # [B, total_kept, C]
        
        return x_pruned, flat_kept_indices
    
    def _block_internal_restore(self, delta_pruned, x_full, flat_kept_indices, block_idx=None, 
                               first_chunk_cache=None, cache_item="x", fill_strategy="prev_step",
                               num_frames=None, frame_seqlen=None, prev_step_deltas=None, source_latent=None):
        """
        Block内部restore：将pruned delta恢复到完整维度
        
        Args:
            delta_pruned: [B, total_kept, C] pruned计算的delta
            x_full: [B, F*H*W, C] 完整的x（用于获取shape）
            flat_kept_indices: [total_kept] 全局kept indices
            block_idx: int, 当前block的索引
            first_chunk_cache: dict, first chunk的缓存
            cache_item: str, 缓存内容类型（当fill_strategy="first_chunk"时使用）
                - "x": 缓存block输入x
                - "delta": 缓存delta（y_full）
            fill_strategy: str, 未pruning位置的填充策略
                - "identity": delta=0，保持当前x不变
                - "prev_step": 使用前一个全量计算step的delta（推荐）
                - "first_chunk": 使用first chunk的值（根据cache_item）
                - "source_latent": 使用source_latent (y_input)，delta = source - current_x
                - "avg_delta": 使用保留token的平均delta
                - "interpolate": 从周围保留token插值
                - "zero": 填充0（等同identity）
            num_frames: int, 帧数（插值时需要）
            frame_seqlen: int, 每帧token数（插值时需要）
            prev_step_deltas: dict, 上一个step的delta缓存 {block_idx: delta}
            source_latent: Tensor, 源latent (y_input)，用于source_latent策略
        
        Returns:
            delta_full: [B, F*H*W, C] 完整维度的delta
                - 保留位置：使用计算得到的delta
                - 未保留位置：根据fill_strategy填充
        """
        B, L, C = x_full.shape
        delta_full = torch.zeros_like(x_full)  # [B, L, C]
        
        # Scatter pruned delta到保留位置
        delta_full[:, flat_kept_indices, :] = delta_pruned
        
        # 创建未保留位置的mask
        all_indices = torch.arange(L, device=x_full.device)
        kept_mask = torch.zeros(L, dtype=torch.bool, device=x_full.device)
        kept_mask[flat_kept_indices] = True
        unkept_indices = all_indices[~kept_mask]
        
        if len(unkept_indices) == 0:
            # 没有未保留位置，直接返回
            return delta_full
        
        # 🎨 根据fill_strategy填充未保留位置
        if fill_strategy == "zero" or fill_strategy == "identity":
            # 策略1&2: delta=0，保持当前x不变（已经是0了，不需要操作）
            pass
        
        elif fill_strategy == "prev_step":
            # 策略: 使用前一个全量计算step的delta（⭐推荐）
            # if block_idx == 0:
            #     print(f"[Debug Fill] prev_step_deltas is not None: {prev_step_deltas is not None}")
            #     print(f"[Debug Fill] block_idx is not None: {block_idx is not None}")
            #     if prev_step_deltas is not None:
            #         print(f"[Debug Fill] block_idx in prev_step_deltas: {block_idx in prev_step_deltas}")
            #         print(f"[Debug Fill] prev_step_deltas.keys(): {list(prev_step_deltas.keys())}")
            #     print(f"[Debug Fill] not self.training: {not self.training}")
            
            if (prev_step_deltas is not None and block_idx is not None and \
                block_idx in prev_step_deltas and not self.training):
                prev_delta = prev_step_deltas[block_idx]
                # 使用上一个step的delta填充未保留位置
                delta_full[:, unkept_indices, :] = prev_delta[:, unkept_indices, :]
                # if block_idx == 0:
                #     print(f"[Debug Fill] Successfully filled {len(unkept_indices)} unkept positions from prev_step")
                
        elif fill_strategy == "debug":
            # 策略: 使用前一个全量计算step的delta 直接覆盖
            if (prev_step_deltas is not None and block_idx is not None and 
                block_idx in prev_step_deltas and not self.training):
                prev_delta = prev_step_deltas[block_idx]
                # 使用上一个step的delta填充未保留位置
                delta_full = prev_delta

        elif fill_strategy == "first_chunk":
            # 策略: 使用first chunk的值
            if (first_chunk_cache is not None and block_idx is not None and 
                block_idx in first_chunk_cache and not self.training):
                
                if cache_item == "x":
                    # 缓存的是x：delta = first_chunk_x - current_x
                    first_chunk_x = first_chunk_cache[block_idx]
                    delta_full[:, unkept_indices, :] = first_chunk_x[:, unkept_indices, :] - x_full[:, unkept_indices, :]
                elif cache_item == "delta":
                    # 缓存的是delta：直接使用
                    first_chunk_delta = first_chunk_cache[block_idx]
                    delta_full[:, unkept_indices, :] = first_chunk_delta[:, unkept_indices, :]
        
        elif fill_strategy == "source_latent":
            # 策略: 使用source_latent (y_input)
            # delta = source - current_x
            if source_latent is not None:
                delta_full[:, unkept_indices, :] = source_latent[:, unkept_indices, :] - x_full[:, unkept_indices, :]
        
        elif fill_strategy == "avg_delta":
            # 策略: 使用保留token的平均delta
            if len(flat_kept_indices) > 0:
                avg_delta = delta_pruned.mean(dim=1, keepdim=True)  # [B, 1, C]
                delta_full[:, unkept_indices, :] = avg_delta.expand(B, len(unkept_indices), C)
        
        elif fill_strategy == "interpolate":
            # 策略: 从周围保留token插值（2D空间插值）
            if num_frames is not None and frame_seqlen is not None:
                # 需要2D grid信息进行插值
                # 这里实现一个简单的最近邻插值
                H = int(frame_seqlen ** 0.5)
                W = frame_seqlen // H
                
                for b in range(B):
                    for f_idx in range(num_frames):
                        # 当前帧的范围
                        frame_start = f_idx * frame_seqlen
                        frame_end = (f_idx + 1) * frame_seqlen
                        
                        # 当前帧的kept和unkept indices（转为局部索引）
                        frame_kept_global = flat_kept_indices[(flat_kept_indices >= frame_start) & (flat_kept_indices < frame_end)]
                        frame_kept_local = frame_kept_global - frame_start
                        
                        frame_unkept_global = unkept_indices[(unkept_indices >= frame_start) & (unkept_indices < frame_end)]
                        frame_unkept_local = frame_unkept_global - frame_start
                        
                        if len(frame_kept_local) == 0 or len(frame_unkept_local) == 0:
                            continue
                        
                        # 转换为2D坐标
                        kept_y = frame_kept_local // W
                        kept_x = frame_kept_local % W
                        unkept_y = frame_unkept_local // W
                        unkept_x = frame_unkept_local % W
                        
                        # 对每个unkept位置，找最近的kept位置
                        for i, unk_idx in enumerate(frame_unkept_local):
                            uy, ux = unkept_y[i].item(), unkept_x[i].item()
                            
                            # 计算距离
                            dist = (kept_y - uy) ** 2 + (kept_x - ux) ** 2
                            nearest_idx = dist.argmin()
                            
                            # 使用最近邻的delta
                            nearest_global_idx = frame_kept_global[nearest_idx]
                            delta_full[b, frame_start + unk_idx] = delta_full[b, nearest_global_idx]
        
        return delta_full

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        debug_dict=None,
        pruning_info=None,
        block_idx=None  # 🆕 用于保存/读取first chunk缓存
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] - 保持完整维度
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            pruning_info(dict): {'internal_pruning': bool, 'kept_indices_per_frame': List[Tensor]}
        """
        num_frames = e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        
        # ✨ Block内部Pruning模式判断
        use_block_internal_pruning = (
            pruning_info is not None and 
            pruning_info.get('internal_pruning', False) and
            'kept_indices_per_frame' in pruning_info and
            pruning_info.get('use_pruning', False)  # 🆕 新增：检查是否启用pruning
        )
        
        # 🆕 层级控制：检查是否在指定层进行pruning
        pruning_layers = pruning_info.get('pruning_layers', set()) if pruning_info else set()
        use_self_attn_pruning = use_block_internal_pruning and "self_attn" in pruning_layers
        use_ffn_pruning = use_block_internal_pruning and "ffn" in pruning_layers
        
        # 🆕 提前提取first_chunk_cache、cache_item、fill_strategy和prev_step_deltas（供所有分支使用）
        first_chunk_cache = pruning_info.get('first_chunk_block_inputs', {}) if pruning_info else {}
        cache_item = pruning_info.get('cache_item', 'x') if pruning_info else 'x'
        fill_strategy = pruning_info.get('fill_strategy', 'prev_step') if pruning_info else 'prev_step'
        use_mean_alignment = pruning_info.get('use_mean_alignment', True) if pruning_info else True  # 🆕 均值对齐开关
        source_latent = pruning_info.get('source_latent', None) if pruning_info else None  # 🆕 源latent (y_input)
        
        # ⚠️ 关键：直接使用pruning_info中的引用，如果不存在则用空字典（会导致保存失败）
        # 这样修改局部变量会直接影响pruning_info中的字典
        prev_step_self_attn_deltas = pruning_info.get('prev_step_self_attn_deltas', {}) if pruning_info else {}
        prev_step_ffn_deltas = pruning_info.get('prev_step_ffn_deltas', {}) if pruning_info else {}
        
        # 🐛 Debug: 打印引用检查
        # if block_idx == 0 and pruning_info and 'prev_step_self_attn_deltas' in pruning_info:
        #     print(f"[Block Debug] prev_step_self_attn_deltas is pruning_info ref: {prev_step_self_attn_deltas is pruning_info['prev_step_self_attn_deltas']}")
        #     print(f"[Block Debug] prev_step_self_attn_deltas id: {id(prev_step_self_attn_deltas)}, pruning_info id: {id(pruning_info['prev_step_self_attn_deltas'])}")
        
        # 原始frame_seqlen（x始终保持完整）
        frame_seqlen = x.shape[1] // num_frames
        
        # ========== Self-Attention with Block Internal Pruning ==========
        if use_self_attn_pruning:
            # 💾 保存first chunk（根据cache_item决定保存x还是delta）
            should_cache = (block_idx is not None and block_idx not in first_chunk_cache and not self.training)
            if should_cache and cache_item == "x":
                # 缓存x：在计算前保存
                first_chunk_cache[block_idx] = x.detach().clone()
            
            # 1. Norm + Modulation (完整维度)
            x_norm = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)
            
            # 2. Prune输入
            x_pruned, flat_kept_indices = self._block_internal_prune(
                x_norm, 
                pruning_info['kept_indices_per_frame'], 
                num_frames
            )

            # 3. Self-Attn计算pruned delta
            # 🔑 构造兼容旧格式的pruning_info（用于RoPE）
            pruned_info = {
                'pruned': True,
                'tokens_per_frame_kept': len(pruning_info['kept_indices_per_frame'][0]),
                'tokens_per_frame_original': frame_seqlen,
                'num_frames': num_frames,
                'kept_indices': [flat_kept_indices]  # List of [total_kept] for batch
            }

            y_pruned = self.self_attn(
                x_pruned, seq_lens, grid_sizes, freqs, block_mask, 
                kv_cache, current_start, cache_start, debug_dict, pruned_info
            )
            
            # 🔧 对齐y_pruned与prev_step delta的均值（如果使用prev_step填充策略且启用均值对齐）
            # print("DEBUG: use_mean_alignment", use_mean_alignment, "fill_strategy", fill_strategy, "block_idx", block_idx)
            if use_mean_alignment and (fill_strategy in ["prev_step", "debug"]) and block_idx in prev_step_self_attn_deltas:
                prev_delta = prev_step_self_attn_deltas[block_idx]  # [B, L, D]
                # 计算y_pruned的均值
                y_pruned_mean = y_pruned.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
                # 计算kept位置对应的prev_delta均值
                prev_delta_kept = prev_delta[:, flat_kept_indices, :]  # [B, kept, D]
                prev_delta_kept_mean = prev_delta_kept.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
                # 对齐：y_pruned -= (y_pruned_mean - prev_delta_kept_mean)
                y_pruned = y_pruned - (y_pruned_mean - prev_delta_kept_mean)
                if block_idx == 0:
                    print(f"[Mean Align] Self-Attn: y_pruned_mean={y_pruned_mean[0,0,:3].tolist()}, "
                          f"prev_kept_mean={prev_delta_kept_mean[0,0,:3].tolist()}")
            
            # 4. Restore delta到完整维度（使用first chunk填充未保留位置）
            y_full = self._block_internal_restore(
                y_pruned, x, flat_kept_indices, block_idx, first_chunk_cache, cache_item, fill_strategy,
                num_frames, frame_seqlen, prev_step_self_attn_deltas, source_latent  # ✅ 传递source_latent
            )
            
            # if block_idx == 0:
            #     print("[Reuse] Self Attn delta in forward pass")

            # 💾 保存当前step的self-attn delta（用于下一个pruning step填充）
            # 注意：保存未经modulation的delta，使用时需要应用当前step的modulation
            # if block_idx is not None and not self.training:
            #     # prev_step_self_attn_deltas[block_idx] = y_full.detach().clone()
            #     if block_idx == 0:
            #         # 🔍 打印填充后的完整delta统计（包含保留+填充部分）
            #         filled_mean = y_full.mean().item()
            #         filled_std = y_full.std().item()
            #         # 🔍 打印pruned delta统计（仅保留部分，真实计算的delta）
            #         pruned_mean = y_pruned.mean().item()
            #         pruned_std = y_pruned.std().item()
            #         print(f"[Reuse] Filled delta: mean={filled_mean:.6f}, std={filled_std:.6f}")
            #         print(f"[Reuse] Pruned delta (computed): mean={pruned_mean:.6f}, std={pruned_std:.6f}")
            
            # 5. 残差连接（x保持完整）
            x = x + (y_full.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        else:
            # 原始流程（无pruning，全量计算）
            y = self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start, debug_dict, None
            )
            # print(f"[Shapes] self-attn {y.shape=}")
            
            # 💾 全量计算时也保存self-attn delta（用于下一个pruning step填充）
            if block_idx is not None and not self.training:
                # y是完整的delta，直接保存
                prev_step_self_attn_deltas[block_idx]=y
                # if block_idx == 0:
                #     delta_mean = y.mean().item()
                #     delta_std = y.std().item()
                #     print(f"[Cache] Saving Self-Attn delta: mean={delta_mean:.6f}, std={delta_std:.6f} in block {block_idx}")
            
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        
        # ========== Cross-Attention & FFN ==========
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None, debug_dict=None, 
                          block_idx=None, first_chunk_cache=None, use_ffn_pruning=False, cache_item="x", 
                          fill_strategy="prev_step", prev_step_ffn_deltas=None):
            # Cross-Attention (不prune，因为涉及外部context)
            x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                    crossattn_cache=crossattn_cache, debug_dict=debug_dict)
            
            # FFN with Block Internal Pruning
            if use_ffn_pruning:
                # 💾 保存first chunk（如果cache_item="x"且还没缓存）
                # 注意：如果self-attention已经缓存了，这里不重复缓存
                should_cache_ffn = (block_idx is not None and block_idx not in first_chunk_cache and not self.training)
                if should_cache_ffn and cache_item == "x":
                    first_chunk_cache[block_idx] = x.detach().clone()
                    # print(f"[Block {block_idx} FFN] 💾 Saved first chunk x: {x.shape}")
                
                # 1. Norm + Modulation
                x_norm = (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
                
                # 2. Prune输入
                x_pruned, flat_kept_indices = self._block_internal_prune(
                    x_norm, 
                    pruning_info['kept_indices_per_frame'], 
                    num_frames
                )
                
                # 3. FFN计算pruned delta
                x_norm_for_ffn = x_pruned
                
                # 先正常计算
                y_pruned = self.ffn(x_norm_for_ffn)
                
                # 🔍 保存FFN中间激活（如果启用debug）
                if debug_dict is not None and debug_dict.get('visualize_ffn', False) and block_idx == 0:
                    with torch.no_grad():
                        ffn_hidden = self.ffn[0](x_norm_for_ffn)
                        ffn_act = self.ffn[1](ffn_hidden)
                        
                        import numpy as np
                        save_dir = debug_dict.get("save_path", "./visual-output").replace(os.path.basename(debug_dict.get("save_path", "")), "")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        chunk_idx = debug_dict.get('current_chunk', 0)
                        step_idx = debug_dict.get('current_step', 0)
                        
                        np.save(os.path.join(save_dir, f'ffn_hidden_{chunk_idx}_{step_idx}.npy'), 
                               ffn_hidden.cpu().float().numpy())
                        np.save(os.path.join(save_dir, f'ffn_act_{chunk_idx}_{step_idx}.npy'), 
                               ffn_act.cpu().float().numpy())
                        np.save(os.path.join(save_dir, f'ffn_output_{chunk_idx}_{step_idx}.npy'), 
                               y_pruned.cpu().float().numpy())
                        print(f"[Debug] Saved FFN activations: hidden={ffn_hidden.shape}, act={ffn_act.shape}, output={y_pruned.shape}")
                
                # 🔧 对齐y_pruned与prev_step delta的均值（如果使用prev_step填充策略且启用均值对齐）
                if use_mean_alignment and (fill_strategy in ["prev_step", "debug"]) and block_idx in prev_step_ffn_deltas:
                    prev_delta = prev_step_ffn_deltas[block_idx]  # [B, L, D]
                    # 计算y_pruned的均值
                    y_pruned_mean = y_pruned.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
                    # 计算kept位置对应的prev_delta均值
                    prev_delta_kept = prev_delta[:, flat_kept_indices, :]  # [B, kept, D]
                    prev_delta_kept_mean = prev_delta_kept.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
                    # 对齐：y_pruned -= (y_pruned_mean - prev_delta_kept_mean)
                    y_pruned = y_pruned - (y_pruned_mean - prev_delta_kept_mean)
                    if block_idx == 0:
                        print(f"[Mean Align] FFN: y_pruned_mean={y_pruned_mean[0,0,:3].tolist()}, "
                              f"prev_kept_mean={prev_delta_kept_mean[0,0,:3].tolist()}")
                
                # 4. Restore delta（使用first chunk填充未保留位置）
                y_full = self._block_internal_restore(
                    y_pruned, x, flat_kept_indices, block_idx, first_chunk_cache, cache_item, fill_strategy,
                    num_frames, frame_seqlen, prev_step_ffn_deltas, source_latent  # ✅ 传递source_latent
                )
                
                # if block_idx == 0:
                #     print("[Reuse] FFN delta in cross_attn_ffn") 
                
                # 5. 残差连接
                x = x + (y_full.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            else:
                # 原始流程（无pruning，全量计算）
                x_norm_input = (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
                
                # 先正常计算
                y = self.ffn(x_norm_input)
                
                # 🔍 保存FFN中间激活（如果启用debug）
                if debug_dict is not None and debug_dict.get('visualize_ffn', False) and block_idx == 0:
                    with torch.no_grad():
                        ffn_hidden = self.ffn[0](x_norm_input)
                        ffn_act = self.ffn[1](ffn_hidden)
                        
                        import numpy as np
                        save_dir = debug_dict.get("save_path", "./visual-output").replace(os.path.basename(debug_dict.get("save_path", "")), "")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        chunk_idx = debug_dict.get('current_chunk', 0)
                        step_idx = debug_dict.get('current_step', 0)
                        
                        np.save(os.path.join(save_dir, f'ffn_hidden_{chunk_idx}_{step_idx}.npy'), 
                               ffn_hidden.cpu().float().numpy())
                        np.save(os.path.join(save_dir, f'ffn_act_{chunk_idx}_{step_idx}.npy'), 
                               ffn_act.cpu().float().numpy())
                        np.save(os.path.join(save_dir, f'ffn_output_{chunk_idx}_{step_idx}.npy'), 
                               y.cpu().float().numpy())
                        print(f"[Debug] Saved FFN activations: hidden={ffn_hidden.shape}, act={ffn_act.shape}, output={y.shape}")
                # print(f"[Shapes] ffn {y.shape=}")
                
                # 💾 全量计算时也保存FFN delta（用于下一个pruning step填充）
                if block_idx is not None and not self.training:
                    prev_step_ffn_deltas[block_idx] = y #.detach().clone()
                    # if block_idx == 0:
                    #     print("[Cache] Saving FFN delta in cross_attn_ffn")
                
                x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache, debug_dict,
                          block_idx, first_chunk_cache, use_ffn_pruning, cache_item, fill_strategy, prev_step_ffn_deltas)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.source_patch_embedding = None
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps, block_id)
            for block_id in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

        # Token Pruning配置（在patch_embedding后进行）
        self.enable_internal_pruning = False
        self.internal_pruning_layers = set()  # 🆕 控制哪些层进行pruning: {"self_attn", "ffn"}
        self.first_chunk_cache_item = "x"
        self.unpruned_fill_strategy = "first_chunk"
        self.use_mean_alignment = False
        
        self.first_chunk_block_inputs = {}
        
        self.prev_step_self_attn_deltas = {}
        for i in range(self.num_layers):
            self.prev_step_self_attn_deltas[i] = torch.zeros([1, 4680, 1536])
        self.prev_step_ffn_deltas = {}  # {block_idx: [B, F*H*W, D]}
        for i in range(self.num_layers):
            self.prev_step_ffn_deltas[i] = torch.zeros([1, 4680, 1536])

    def _initialize_source_patch_embedding(self, model_path=None):
        """
        初始化source_patch_embedding（只在unpruned_fill_strategy="source_latent"时调用）
        从预训练模型加载patch_embedding权重
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
        """
        if self.source_patch_embedding is None:
            # 获取主patch_embedding的device和dtype
            device = self.patch_embedding.weight.device
            dtype = self.patch_embedding.weight.dtype
            
            self.source_patch_embedding = nn.Conv3d(
                self.in_dim, self.dim, 
                kernel_size=self.patch_size, 
                stride=self.patch_size
            ).to(device=device, dtype=dtype)
            
            # 如果没有指定路径，使用默认路径
            if model_path is None:
                model_path = "wan_models/Wan2.1-T2V-1.3B/"
            
            # 从预训练模型加载patch_embedding权重
            try:
                # 尝试加载safetensors格式
                from safetensors.torch import load_file
                checkpoint_path = f"{model_path}/diffusion_pytorch_model.safetensors"
                state_dict = load_file(checkpoint_path)
                print(f"[Source Patch Embedding] Loading from safetensors: {checkpoint_path}")
            except Exception as e:
                try:
                    # 如果safetensors失败，尝试.pth格式
                    checkpoint_path = f"{model_path}/diffusion_pytorch_model.pth"
                    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    print(f"[Source Patch Embedding] Loading from pth: {checkpoint_path}")
                except Exception as e2:
                    # 如果都失败，则复制当前模型的权重
                    print(f"[Source Patch Embedding] Warning: Failed to load checkpoint, copying from current model")
                    print(f"  Safetensors error: {e}")
                    print(f"  PTH error: {e2}")
                    with torch.no_grad():
                        self.source_patch_embedding.weight.copy_(self.patch_embedding.weight)
                        if self.patch_embedding.bias is not None:
                            self.source_patch_embedding.bias.copy_(self.patch_embedding.bias)
                    
                    # 设置为eval模式
                    self.source_patch_embedding.eval()
                    self.source_patch_embedding.requires_grad_(False)
                    print(f"[Source Patch Embedding] Initialized by copying from current model")
                    return
            
            # 提取patch_embedding的权重
            patch_weight_key = 'patch_embedding.weight'
            patch_bias_key = 'patch_embedding.bias'
            
            if patch_weight_key in state_dict:
                with torch.no_grad():
                    # 转换到正确的dtype和device
                    weight = state_dict[patch_weight_key].to(device=device, dtype=dtype)
                    self.source_patch_embedding.weight.copy_(weight)
                    
                    if patch_bias_key in state_dict and self.source_patch_embedding.bias is not None:
                        bias = state_dict[patch_bias_key].to(device=device, dtype=dtype)
                        self.source_patch_embedding.bias.copy_(bias)
                
                print(f"[Source Patch Embedding] Loaded weights: {self.source_patch_embedding.weight.shape}, dtype={dtype}")
            else:
                # 如果找不到，则复制当前模型的权重
                print(f"[Source Patch Embedding] Warning: 'patch_embedding' not found in checkpoint, copying from current model")
                with torch.no_grad():
                    self.source_patch_embedding.weight.copy_(self.patch_embedding.weight)
                    if self.patch_embedding.bias is not None:
                        self.source_patch_embedding.bias.copy_(self.patch_embedding.bias)
            
            # 设置为eval模式，不参与训练
            self.source_patch_embedding.eval()
            self.source_patch_embedding.requires_grad_(False)
            
            print(f"[Source Patch Embedding] Initialized successfully")

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask
    
    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=None,
        cache_start=None,
        debug_dict=None,
        y=None,  # 🆕 source latent for unpruned filling
        kept_indices_per_frame=None,  # 🆕 Block Internal Pruning使用
        importance_mask=None,  # Old Per-frame Pruning使用（废弃）
        use_pruning=False,  # 🆕 是否在当前step应用pruning
    ):
        # 🐛 Debug: 检查进入forward时的状态
        # if hasattr(self, 'prev_step_self_attn_deltas') and not self.training:
        #     none_count = sum(1 for v in self.prev_step_self_attn_deltas.values() if v is None)
        #     total_count = len(self.prev_step_self_attn_deltas)
        #     print(f"[Forward Entry] prev_step_self_attn_deltas: {none_count}/{total_count} blocks are None, id={id(self.prev_step_self_attn_deltas)}")
        
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # 🆕 如果有y，并且使用source_latent填充策略，单独处理source
        source_latent_embedded = None
        if y is not None:
            # 原有训练逻辑：concat后一起patch_embedding
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            
            # 只在使用source_latent填充策略时才处理source
            if self.unpruned_fill_strategy == "source_latent":
                # 初始化source_patch_embedding（如果还没初始化）
                if self.source_patch_embedding is None:
                    self._initialize_source_patch_embedding()
                
                # 对source单独做patch_embedding
                with torch.no_grad():
                    source_embed = [self.source_patch_embedding(v.unsqueeze(0)) for v in y]
                    source_latent_embedded = [u.flatten(2).transpose(1, 2) for u in source_embed]
                    source_latent_embedded = torch.cat(source_latent_embedded)  # [B, L, C]

        # 💾 保存latent空间的grid_sizes（patch_embedding前）
        # x: List[Tensor], 每个shape=[C, F, H, W]（如果有y则C=32，否则C=16）
        latent_grid_sizes = torch.stack(
            [torch.tensor(u.shape[1:], dtype=torch.long) for u in x])  # [B, 3] (F, H_latent, W_latent)
        
        # embeddings (保持原有训练逻辑)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # [B, 3] (F, H_patch, W_patch)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        
        pruning_info = {'pruned': False}
        original_seq_lens = seq_lens
        original_grid_sizes = grid_sizes
        
        if kept_indices_per_frame is not None:
            # print("🎯 Block Internal Pruning (新实现)")
            # Pipeline传来的kept_indices是latent空间的(60×104)
            # 需要转换到patch_embedding后空间的indices (30×52)
            
            # 1. 使用latent空间的grid_sizes重建mask
            B = x.shape[0]
            F, H_latent, W_latent = latent_grid_sizes[0].tolist()  # ✅ 使用latent空间尺寸
            patch_t, patch_h, patch_w = self.patch_size
            H_patch = H_latent // patch_h
            W_patch = W_latent // patch_w
            tokens_per_frame_latent = H_latent * W_latent
            tokens_per_frame_patch = H_patch * W_patch
            
            # 重建latent空间mask
            latent_mask = torch.zeros(F, H_latent, W_latent, dtype=torch.bool, device=x.device)
            for f_idx in range(F):
                kept_idx_latent = kept_indices_per_frame[f_idx]
                # 🔧 添加边界检查和裁剪
                valid_mask = (kept_idx_latent >= 0) & (kept_idx_latent < H_latent * W_latent)
                kept_idx_latent = kept_idx_latent[valid_mask]
                
                h_coords = kept_idx_latent // W_latent
                w_coords = kept_idx_latent % W_latent
                
                # 再次确保索引在范围内
                h_coords = torch.clamp(h_coords, 0, H_latent - 1)
                w_coords = torch.clamp(w_coords, 0, W_latent - 1)
                
                latent_mask[f_idx, h_coords, w_coords] = True
            
            # 2. 下采样到patch空间（average pooling + threshold）
            # [F, H_latent, W_latent] → [F, H_patch, W_patch]
            latent_mask_float = latent_mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, F, H, W]
            latent_mask_2d = latent_mask_float.squeeze(1)  # [1, F, H, W]
            
            # 对每个frame单独average pooling
            patch_masks = []
            for f_idx in range(F):
                frame_mask = latent_mask_2d[:, f_idx:f_idx+1]  # [1, 1, H, W]
                # Average pooling: 计算每个patch内True的比例
                pooled = torch.nn.functional.avg_pool2d(
                    frame_mask,
                    kernel_size=(patch_h, patch_w),
                    stride=(patch_h, patch_w)
                )
                # 阈值：patch内>50%的像素为True → 该patch为True
                patch_mask_bool = (pooled > 0.5).squeeze()  # [H_patch, W_patch]
                patch_masks.append(patch_mask_bool)
            
            patch_mask = torch.stack(patch_masks)  # [F, H_patch, W_patch]
            
            # 3. 从patch mask提取indices
            kept_indices_per_frame_patch = []
            for f_idx in range(F):
                frame_mask_flat = patch_mask[f_idx].flatten()  # [H_patch * W_patch]
                kept_idx = torch.nonzero(frame_mask_flat, as_tuple=True)[0]
                kept_indices_per_frame_patch.append(kept_idx)
            
            # 4. Per-frame统一长度
            min_kept = min(len(idx) for idx in kept_indices_per_frame_patch)
            kept_indices_per_frame_patch = [idx[:min_kept] for idx in kept_indices_per_frame_patch]
            
            # x保持完整，pruning在每个Block内部进行
            pruning_info = {
                'internal_pruning': True,
                'use_pruning': use_pruning,  # 🆕 是否在当前step应用pruning
                'kept_indices_per_frame': kept_indices_per_frame_patch,  # ✅ 使用patch空间的indices
                'first_chunk_block_inputs': self.first_chunk_block_inputs,  # 🆕 传递缓存引用
                'pruning_layers': self.internal_pruning_layers,  # 🆕 传递层级配置
                'cache_item': self.first_chunk_cache_item,  # 🆕 传递缓存内容配置: "x" or "delta"
                'fill_strategy': self.unpruned_fill_strategy,  # 🆕 传递填充策略
                'prev_step_self_attn_deltas': self.prev_step_self_attn_deltas,  # 🆕 self-attn的delta缓存
                'prev_step_ffn_deltas': self.prev_step_ffn_deltas,  # 🆕 FFN的delta缓存
                'use_mean_alignment': self.use_mean_alignment,  # 🆕 均值对齐开关
                'source_latent': source_latent_embedded  # 🆕 源latent (patch_embedding后的y)
            }
            
            total_kept = F * min_kept
            reduction = (1 - total_kept / (F * tokens_per_frame_patch)) * 100
            
            # 🆕 显示层级配置
            layers_enabled = []
            if "self_attn" in self.internal_pruning_layers:
                layers_enabled.append("Self-Attn")
            if "ffn" in self.internal_pruning_layers:
                layers_enabled.append("FFN")
            layers_str = " + ".join(layers_enabled) if layers_enabled else "None"

            importance_mask = None
        
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            pruning_info=pruning_info  # 🆕 传递pruning信息给attention层
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                if debug_dict is not None:
                    debug_dict["block_index"] = block_index
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "debug_dict": debug_dict,
                        "block_idx": block_index  # 🆕 传递block索引
                    }
                )
                x = block(x, **kwargs) # 对应CausalWanAttentionBlock

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # print(f"{[u.shape for u in x]=} {t.shape=} {[u.shape for u in context]=} {seq_len=}")
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:  # channel-wise
            x = [torch.cat([u, v[:, -u.shape[1]:] if u.shape[1] != v.shape[1] else v], dim=0)
                for u, v in zip(x, y)]
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print(f"after patch embedding, {[u.shape for u in x]=}")

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        # print(f"after flatten, {[u.shape for u in x]=}")
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        # print(f"{x.shape=} {self.freq_dim=} {t.flatten().shape=}")
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print(f"{x.shape=} {e.shape=} {e0.shape=} {seq_lens=} {grid_sizes=}")

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            # print(f"in training forward, after block {block}, {x.shape=}")

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
