import torch.nn.functional as F
from typing import Tuple, Dict
import torch
from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

class MMRegression(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        # 初始化模型 (保持 Wan 系列包装)：在BaseModel里调用了_initialize_models
        # 加载权重逻辑保持不变
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")['generator']
            self.generator.load_state_dict(state_dict, strict=True)
            
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        print("Initialized WanDiffusionWrapper for MMRegression is_causal =", getattr(args, "is_causal", True))
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
        )-> Tuple[torch.Tensor, dict]:
        """
        target_latent: [B, T, C, H, W]  (x0)
        conditional_dict must contain "source_latent": [B, T, C, H, W]
        """
        B, T, C, H, W = target_latent.shape
        device = target_latent.device

        source_latent = conditional_dict["source_latent"]

        # 1) sample a timestep (match inference step space)
        #    推理里 denoising_step_list 是 long/int，所以这里也用 long 更一致
        t_b = torch.randint(1, 1000, (B,), device=device, dtype=torch.long)  # [B]
        t = t_b[:, None].repeat(1, T)                                        # [B,T] causal

        # 2) construct xt using the SAME scheduler.add_noise as inference
        noise = torch.randn_like(target_latent)                               # [B,T,C,H,W]
        xt = self.generator.scheduler.add_noise(
            target_latent.flatten(0, 1),                                      # [B*T,C,H,W]
            noise.flatten(0, 1),                                              # [B*T,C,H,W]
            t.flatten(0, 1),                                                  # [B*T]
        ).unflatten(0, (B, T))                                                # [B,T,C,H,W]

        # 3) predict x0
        flow_pred, pred_x0 = self.generator(
            noisy_image_or_video=xt,
            conditional_dict=conditional_dict,
            timestep=t,                   # [B,T]
            y=source_latent,              # [B,T,C,H,W]
        )

        # 4) x0 regression loss (aligned with inference usage of pred_x0)
        loss = F.mse_loss(pred_x0.float(), target_latent.float(), reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_x0, target_latent, reduction="none").mean(dim=[1,2,3,4]).detach(),
            "timestep": t_b.detach(),          # [B] 用于统计
            "output": pred_x0.detach(),        # 可视化/调试
        }
        return loss, log_dict