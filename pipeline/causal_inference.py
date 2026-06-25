from typing import List, Optional
import torch
import os

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            local_attn_size=-1,
            sink_size=0,
            expand_patch_embedding=False
    ):
        super().__init__()
        self.device = device
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True, local_attn_size=local_attn_size, sink_size=sink_size) if generator is None else generator
        if expand_patch_embedding:
            self._expand_input_layer(args)
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        # self.scheduler = self.generator.get_scheduler()
        # self.denoising_step_list = torch.tensor(
        #     args.denoising_step_list, dtype=torch.long)
        # if args.warp_denoising_step:
        #     timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        #     self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.scheduler = self.generator.get_scheduler()
        if hasattr(args, "denoising_step_list"):    
            N = len(args.denoising_step_list)  # 你想跑多少步就多少
            self.denoising_step_list = torch.tensor(
             args.denoising_step_list, dtype=torch.long)
        else:
            if hasattr(args, "inference_num_steps"):
                N = args.inference_num_steps
            else :
                N = 50
            self.scheduler.set_timesteps(num_inference_steps=N, training=False)
            timesteps = self.scheduler.timesteps.to(device)
            self.denoising_step_list = torch.tensor(
                timesteps, dtype=torch.long)        

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        
        self.use_internal_pruning = getattr(args, "use_internal_pruning", False)
        self.internal_pruning_layers = getattr(args, "internal_pruning_layers", ["self_attn", "ffn"])
        self.first_chunk_cache_item = getattr(args, "first_chunk_cache_item", "x")  # "x" or "delta"
        self.internal_pruning_steps = getattr(args, "internal_pruning_steps", None)  # None or list of step indices
        self.unpruned_fill_strategy = getattr(args, "unpruned_fill_strategy", "first_chunk")  # 填充策略
        self.use_mean_alignment = getattr(args, "use_mean_alignment", True)  # 🆕 均值对齐
        self.adaptive_patch_ratio = getattr(args, "adaptive_patch_ratio", None)
        
        self.use_history_guided_pruning = getattr(args, "use_history_guided_pruning", False)
        self.prev_chunk_mask = None
        self.prev_chunk_generated_latent = None
        self.prev_chunk_importance_mask_1d = None  # 用于internal pruning的1D mask
        
        # Mask可视化
        self.save_mask = getattr(args, "save_mask", False)
        self.mask_output_folder = getattr(args, "mask_output_folder", "videos/masks")
        self.collected_masks = []  # 收集所有chunk的mask用于保存
        
        print(f"KV inference with {self.num_frame_per_block} frames per block")
        
        if self.use_internal_pruning:
            layers_str = ", ".join(self.internal_pruning_layers)
            print(f"🔧 Internal Token Pruning enabled (patch_embedding后): "
                  f"pruning in high-dim feature space (1536-D)")
            print(f"   Pruning layers: {layers_str}")
            if self.internal_pruning_steps is not None:
                print(f"   Pruning steps: {self.internal_pruning_steps} (out of {len(self.denoising_step_list)} total steps)")
            else:
                print(f"   Pruning steps: all steps")
            if self.use_history_guided_pruning:
                print(f"🔥 History-Guided Pruning enabled: using previous chunk's mask")
            self.generator.model.enable_internal_pruning = True
            self.generator.model.internal_pruning_layers = set(self.internal_pruning_layers)
            self.generator.model.first_chunk_cache_item = self.first_chunk_cache_item
            self.generator.model.unpruned_fill_strategy = self.unpruned_fill_strategy
            self.generator.model.use_mean_alignment = self.use_mean_alignment

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _expand_input_layer(self, args):
        """
        将 Wan 模型的 patch_embedding 从 16 通道扩展到 32 通道
        """
        # 获取原始的 patch_embedding (nn.Conv3d)
        old_proj = self.generator.model.patch_embedding
        
        # 检查是否已经是 32 通道，避免重复修改（例如在 resume 训练时）
        if old_proj.in_channels == 32:
            return

        print(f"Expanding patch_embedding input channels: {old_proj.in_channels} -> 32")

        # 创建新的卷积层
        # Wan 的参数通常是: in_channels=16, out_channels=1536, kernel=(1,2,2), stride=(1,2,2)
        new_proj = torch.nn.Conv3d(
            in_channels=32,
            out_channels=old_proj.out_channels,
            # out_channels=old_proj.out_channels*2,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None)
        ).to(device=self.device, dtype=old_proj.weight.dtype)

        # 权重初始化策略：
        # 1. 前 16 通道拷贝原有的预训练权重（保证模型还记得怎么处理噪声视频）
        # 2. 后 16 通道（Source 视频输入）初始化为 0
        #    这样在训练初期，Source 视频不会对预测产生干扰，模型可以平滑地开始学习编辑逻辑
        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[:, :16].copy_(old_proj.weight)
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        # 替换原有的层
        self.generator.model.patch_embedding = new_proj

    def _compute_mask_from_importance(self, importance_map):
        """
        根据importance map计算low_importance_mask
        
        使用比例（adaptive_patch_ratio）来决定保留多少token
        
        Args:
            importance_map: [B, F, H, W] 重要性分数（越高越重要）
        
        Returns:
            low_importance_mask: [B, F, H, W] bool tensor，True表示低重要性
        """
        if self.adaptive_patch_ratio is None:
            # 如果没有设置比例，默认保留全部
            return torch.zeros_like(importance_map, dtype=torch.bool)
        
        # 使用比例：保留top-k%的重要区域
        B, F, H, W = importance_map.shape
        num_tokens = H * W
        keep_num = int(num_tokens * self.adaptive_patch_ratio)
        keep_num = max(1, keep_num)  # 至少保留1个token
        
        # 展平spatial维度
        flat_importance = importance_map.reshape(B, F, -1)  # [B, F, H*W]
        
        # 计算阈值：每个frame独立计算top-k
        low_importance_mask_flat = torch.ones_like(flat_importance, dtype=torch.bool)
        
        for b in range(B):
            for f in range(F):
                importance_frame = flat_importance[b, f]  # [H*W]
                
                # 🔧 修复：当keep_num >= num_tokens时，保留所有token
                if keep_num >= num_tokens:
                    # 所有token都是高重要性
                    low_importance_mask_flat[b, f] = False
                else:
                    # 找到第k大的值作为阈值
                    threshold_value = torch.kthvalue(importance_frame, num_tokens - keep_num + 1).values
                    
                    # 保留 >= threshold的区域（高重要性）
                    high_importance_frame = (importance_frame >= threshold_value)
                    low_importance_mask_flat[b, f] = ~high_importance_frame
        
        # 恢复shape
        low_importance_mask = low_importance_mask_flat.reshape(B, F, H, W)
        
        # actual_keep_ratio = (~low_importance_mask).float().mean()
        # print(f"[Mask from Ratio] Target ratio={self.adaptive_patch_ratio:.1%}, "
        #       f"Actual keep ratio={actual_keep_ratio:.1%}")
        
        return low_importance_mask
    
    def _compute_importance_for_internal_pruning(self, noisy_input, y_input, chunk_idx=None):
        """
        为internal pruning计算importance mask
        
        Args:
            noisy_input: [B, F, C, H, W] 当前的noisy latent
            y_input: [B, F, C, H, W] source latent (可选)
            chunk_idx: int, 当前chunk的索引（用于判断是否为第一个chunk）
        
        Returns:
            importance_mask_1d: [B, F*H*W] bool tensor，True表示高重要性
        """
        if chunk_idx == 0:
            B, F, C, H, W = noisy_input.shape
            return torch.ones(B, F*H*W, dtype=torch.bool, device=noisy_input.device)
        
        if self.prev_chunk_importance_mask_1d is not None:
            return self.prev_chunk_importance_mask_1d
        
        print(f"[Internal Pruning] WARNING: No prev_chunk_mask found for chunk {chunk_idx}!")
        print(f"[Internal Pruning] Falling back to full tokens")
        B, F, C, H, W = noisy_input.shape
        return torch.ones(B, F*H*W, dtype=torch.bool, device=noisy_input.device)
    
    def _compute_mask_from_generated(self, generated_latent, source_latent):
        """
        基于已生成的latent计算下一个chunk的mask
        
        Args:
            generated_latent: [B, F, C, H, W] 刚生成的chunk
            source_latent: [B, F, C, H, W] 对应的source chunk
        
        Returns:
            low_importance_mask: [B, F, H, W] 用于下一个chunk的mask
        """
        B, F, C, H, W = generated_latent.shape
        
        diff = (generated_latent - source_latent).pow(2).mean(dim=2)
        importance = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        
        low_importance_mask = self._compute_mask_from_importance(importance)
        
        return low_importance_mask
    
    def save_mask_video(self, output_path, prompt=""):
        """
        保存收集的mask为黑白视频
        
        Args:
            output_path: 输出视频路径
            prompt: 提示词（用于文件名）
        """
        if len(self.collected_masks) == 0:
            print("[Warning] No masks collected, skip saving")
            return
        
        import os
        from torchvision.io import write_video
        import torch.nn.functional as F
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        all_masks = torch.cat(self.collected_masks, dim=1)
        
        B, num_frames, H, W = all_masks.shape
        
        target_H, target_W = H * 8, W * 8
        
        upsampled_frames = []
        for f_idx in range(num_frames):
            frame_mask = all_masks[:, f_idx]
            upsampled_frame = F.interpolate(
                frame_mask.float().unsqueeze(1),
                size=(target_H, target_W),
                mode='nearest'
            ).squeeze(1)
            upsampled_frames.append(upsampled_frame)
        
        upsampled_masks = torch.stack(upsampled_frames, dim=1)
        
        mask_video = upsampled_masks.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        mask_video = (mask_video * 255).byte().cpu()
        
        for b in range(B):
            if prompt:
                safe_prompt = prompt[:50].replace(' ', '_').replace('/', '_')
                file_path = output_path.replace('.mp4', f'_{safe_prompt}_batch{b}.mp4')
            else:
                file_path = output_path.replace('.mp4', f'_batch{b}.mp4')
            
            write_video(file_path, mask_video[b], fps=16)
            print(f"[Mask Video] Saved to '{file_path}' ({num_frames} frames, {target_H}x{target_W})")
        
        # 清空已保存的mask
        self.collected_masks = []

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        y = None,
        wo_scale=False,
        return_generation_time: bool = False  # 🆕 是否返回生成时长
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        # print(f"[CausalInferencePipeline] {y is not None=}")
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        
        # 🔥 添加source_latent到conditional_dict（用于历史引导）
        if y is not None:
            conditional_dict['source_latent'] = y
            # print(f"[CausalInferencePipeline] Added source_latent to conditional_dict, shape: {y.shape}")

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        debug_dict = {}
        debug_dict['visualize_attention'] = False
        debug_dict['visualize_ffn'] = False
        debug_dict['pooled_attn_rows'] = []
        debug_dict["block_index"] = -1
        debug_dict["input_ids"] = conditional_dict["input_ids"]
        debug_dict["decode_tokens"] = conditional_dict["decode_tokens"]
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        debug_dict["all_chunks"] = len(all_num_frames)
        
        # accumulate_denoised_preds = []
        # 🆕 记录总生成时间
        import time
        generation_start_time = time.time()
        for frame_idx, current_num_frames in enumerate(all_num_frames):
            debug_dict["current_chunk"] = frame_idx
            # print(f"[DEBUG] Processing chunk {frame_idx} with {current_num_frames} frames...")
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            y_input = None
            if y is not None:
                y_input = y[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            

            # print(f"[CausalInferencePipeline] from {current_start_frame - num_input_frames} to {current_start_frame + current_num_frames - num_input_frames}")
            # 🔧 Chunk级别：获取当前chunk的importance mask（只计算一次）
            # - 第一个chunk: 全保留
            # - 后续chunk: 使用上一个chunk计算好的prev_chunk_importance_mask_1d
            chunk_importance_mask_1d = None
            chunk_kept_indices_per_frame = None
            
            if self.use_internal_pruning:
                # 获取importance mask（内部会检查是否使用历史mask）
                chunk_importance_mask_1d = self._compute_importance_for_internal_pruning(
                    noisy_input, y_input, chunk_idx=frame_idx
                )
                
                # 🎯 从importance_mask计算kept_indices_per_frame
                # importance_mask_1d: [B, F*H*W] → per-frame indices
                B, total_tokens = chunk_importance_mask_1d.shape
                F = current_num_frames
                H, W = noisy_input.shape[-2:]  # latent空间的H,W
                tokens_per_frame = H * W
                
                importance_mask_2d = chunk_importance_mask_1d.reshape(B, F, tokens_per_frame)
                
                chunk_kept_indices_per_frame = []
                for f_idx in range(F):
                    frame_mask = importance_mask_2d[0, f_idx]
                    frame_kept_indices = torch.nonzero(frame_mask, as_tuple=True)[0]
                    chunk_kept_indices_per_frame.append(frame_kept_indices)
                
                min_kept = min(len(indices) for indices in chunk_kept_indices_per_frame)
                
                if self.adaptive_patch_ratio is not None and self.adaptive_patch_ratio >= 1.0:
                    if min_kept < tokens_per_frame:
                        print(f"⚠️ WARNING: ratio={self.adaptive_patch_ratio:.1%} but min_kept={min_kept} < {tokens_per_frame}!")
                        print(f"   Per-frame kept: {[len(idx) for idx in chunk_kept_indices_per_frame]}")
                
                chunk_kept_indices_per_frame = [indices[:min_kept] for indices in chunk_kept_indices_per_frame]
                
                total_kept = sum(len(indices) for indices in chunk_kept_indices_per_frame)
                reduction_ratio = (1 - total_kept / (F * tokens_per_frame)) * 100
                print(f"[Chunk {frame_idx}] Computed importance mask: {tokens_per_frame} → {min_kept} tokens/frame, "
                      f"Reduction: {reduction_ratio:.1f}%")
                
                if self.save_mask:
                    importance_mask_spatial = chunk_importance_mask_1d.reshape(B, F, H, W)
                    self.collected_masks.append(importance_mask_spatial.cpu())
            
            debug_dict["all_steps"] = len(self.denoising_step_list)
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                debug_dict['current_step'] = index
                debug_dict['current_timestep'] = current_timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                
                importance_mask_1d = None
                kept_indices_per_frame = None
                use_pruning_this_step = False
                
                if self.use_internal_pruning:
                    importance_mask_1d = chunk_importance_mask_1d
                    kept_indices_per_frame = chunk_kept_indices_per_frame
                    
                    use_pruning_this_step = True
                    if self.internal_pruning_steps is not None:
                        use_pruning_this_step = index in self.internal_pruning_steps
                    
                    if use_pruning_this_step:
                        B, F = chunk_importance_mask_1d.shape[0], current_num_frames
                        H, W = noisy_input.shape[-2:]
                        tokens_per_frame = H * W
                        min_kept = len(kept_indices_per_frame[0]) if kept_indices_per_frame else tokens_per_frame
                        total_kept = sum(len(indices) for indices in kept_indices_per_frame)
                        reduction_ratio = (1 - total_kept / (F * tokens_per_frame)) * 100
                        
                        print(f"[Step {index}] Applying Block Internal Pruning: {tokens_per_frame} → {min_kept} tokens/frame, "
                              f"Reduction: {reduction_ratio:.1f}%")
                    else:
                        print(f"[Step {index}] Skipping pruning (not in pruning_steps), using full tokens")

                
                if index < len(self.denoising_step_list) - 1:
                    # todo: 提取注意力图
                    _, denoised_pred = self.generator( 
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        # current_start=0,
                        current_start=current_start_frame * self.frame_seq_length,
                        debug_dict=debug_dict,
                        y=y_input,
                        importance_mask=importance_mask_1d,
                        kept_indices_per_frame=kept_indices_per_frame,
                        use_pruning=use_pruning_this_step,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        debug_dict=debug_dict,
                        y=y_input,
                        importance_mask=importance_mask_1d,
                        kept_indices_per_frame=kept_indices_per_frame,
                        use_pruning=use_pruning_this_step
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            # print(f"context_timestep: {context_timestep}")
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                y=y_input,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            if self.use_internal_pruning and y_input is not None:
                self.prev_chunk_mask = self._compute_mask_from_generated(
                    generated_latent=denoised_pred,
                    source_latent=y_input
                )
                
                B, F, H, W = self.prev_chunk_mask.shape
                high_importance_mask_2d = ~self.prev_chunk_mask
                self.prev_chunk_importance_mask_1d = high_importance_mask_2d.reshape(B, -1)
            elif self.use_history_guided_pruning and y_input is not None:
                self.prev_chunk_mask = self._compute_mask_from_generated(
                    generated_latent=denoised_pred,
                    source_latent=y_input
                )
            
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        if wo_scale:
            video = (video).clamp(0, 1)
        else:
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        generation_end_time = time.time()
        generation_time_seconds = generation_end_time - generation_start_time
        print(f"\n[Generation Time] Total: {generation_time_seconds:.2f}s ({generation_time_seconds/60:.2f}min)")
        
        if return_generation_time:
            if return_latents:
                return video, output, generation_time_seconds
            else:
                return video, generation_time_seconds
        else:
            if return_latents:
                return video, output
            else:
                return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
            print(f"[DEBUG] {self.local_attn_size=} {self.frame_seq_length=}")
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
