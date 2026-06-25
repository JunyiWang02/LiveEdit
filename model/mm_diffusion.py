from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class MMDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        print("Initialized WanDiffusionWrapper for MMDiffusion is_causal =", getattr(args, "is_causal", True))
        self._expand_input_layer(args, device)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _expand_input_layer(self, args, device):
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
        ).to(device=device, dtype=old_proj.weight.dtype)

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

    def generator_loss(
        self,
        conditional_dict: dict,
        target_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(target_latent)
        image_or_video_shape = target_latent.shape
        batch_size, num_frame = image_or_video_shape[:2]

        source_latent = conditional_dict["source_latent"]

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            target_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(target_latent, noise, timestep)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                target_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = target_latent
            timestep_clean_aug = None

        # Compute loss
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            y=source_latent,              # [B,T,C,H,W]
        )
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        log_dict = {
            "x0": target_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, log_dict
