import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    MMInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset, TextVideoPairDataset
from utils.misc import set_seed

# from demo_utils.memory import device, get_cuda_free_memory_gb, DynamicSwapInstaller

def remove_fsdp_wrapped_module(state_dict):
    """
    去掉state_dict中与 FSDP 相关的包装字段（如 '_fsdp_wrapped_module'），
    返回一个没有这些字段的state_dict。
    """
    cleaned_state_dict = {}
    
    for key, value in state_dict.items():
        # 去掉任何包含 '_fsdp_wrapped_module' 的字段
        if '_fsdp_wrapped_module' in key:
            new_key = 'model.' + key.split('._fsdp_wrapped_module.')[-1]  # 去掉 FSDP 包装部分
            cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict[key] = value
    
    return cleaned_state_dict

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "v2v", "i2v"], help="Type of generation task: text-to-video (t2v), video-to-video (v2v), or image-to-video (i2v)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--inference_num_steps", type=int, default=50, help="Number of inference steps")
parser.add_argument("--prefix", type=str, default="", help="Prefix for output files")
parser.add_argument("--return_generation_time", action="store_true", help="Return generation time")
parser.add_argument("--profile", action="store_true", help="Enable profiling")
args = parser.parse_args()

# Initialize distributed inference
# os.environ["LOCAL_RANK"] = '0'
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

# print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
# low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
config.inference_num_steps = args.inference_num_steps

local_attn_size = -1
if args.num_output_frames > 21:
    local_attn_size = 21

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device, local_attn_size=local_attn_size, expand_patch_embedding=True)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device, expand_patch_embedding=True)
    # pipeline = CausalInferencePipeline(config, device=device, local_attn_size=local_attn_size, expand_patch_embedding=True)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    clean_state_dict = remove_fsdp_wrapped_module(state_dict['generator' if not args.use_ema else 'generator_ema'])
    pipeline.generator.load_state_dict(clean_state_dict)

pipeline = pipeline.to(dtype=torch.bfloat16)
# if low_memory:
    # DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
# else:
pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
if args.task=="v2v":
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])
    # dataset = TextVideoPairDataset(args.data_path)
    dataset = TextVideoPairDataset(args.data_path, transform=transform, num_frames=int(args.num_output_frames*4-3))
elif args.task=="i2v":
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)

num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    # Check if output already exists
    prompt = batch['prompts'][0]
    model = "regular" if not args.use_ema else "ema"
    
    # Check generated video for all tasks
    if args.save_with_index:
        file_name = f'{idx}-0_{model}.mp4'
    else:
        file_name = f'{prompt[:100]}-frame={args.num_output_frames}-0-step-{args.inference_num_steps}.mp4'
    file_name = args.prefix + file_name
    output_path = os.path.join(args.output_folder, file_name.replace(' ', '_'))
    
    if os.path.exists(output_path):
        print(f"[SKIP] Output already exists for idx={idx}: {output_path}")
        continue

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.task=="v2v":
        # For video-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples
        # Encode the input image as the first latent
        source_pixel = batch["source_video"].to(device=device).permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16)  # [B, C, T, H, W]
        target_pixel = batch["edited_video"].to(device=device).permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16)  # [B, C, T, H, W]
        # print(f"[DEBUG] {source_pixel.shape=}")
        source_latent = pipeline.vae.encode_to_latent(source_pixel).to(dtype=torch.bfloat16) # [B, T, C', H', W']
        target_latent = pipeline.vae.encode_to_latent(target_pixel).to(dtype=torch.bfloat16) # [B, T, C', H', W']
        # print(f"[DEBUG] {source_latent.shape=}")
        initial_latent = None
        noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        sampled_noise = noise

        # t_b = torch.randint(1, 1000, (1,), device=device, dtype=torch.long)  # [B]
        # t = t_b[:, None].repeat(1, args.num_output_frames)                                        # [B,T] causal
        # print(f"[DEBUG] {t=}")
        # sampled_noise = pipeline.generator.scheduler.add_noise(
        #     target_latent.flatten(0, 1),                                      # [B*T,C,H,W]
        #     noise.flatten(0, 1),                                              # [B*T,C,H,W]
        #     t.flatten(0, 1),                                                  # [B*T]
        # ).unflatten(0, (1, args.num_output_frames))                           # [B,T,C,H,W]

        # conditional_dict = pipeline.text_encoder(prompts)
        # print(f"[DEBUG] {conditional_dict.keys()=}")

        # flow_pred, pred_x0 = pipeline.generator(
        #     noisy_image_or_video=sampled_noise,
        #     conditional_dict=conditional_dict,
        #     timestep=t,                   # [B,T]
        #     y=source_latent,              # [B,T,C,H,W]
        # )

        # video = pipeline.vae.decode_to_pixel(pred_x0, use_cache=False)

        video = pipeline.vae.decode_to_pixel(source_latent, use_cache=False)
        video = (video).clamp(0, 1)
        # save video
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        video = 255.0 * torch.cat(all_video, dim=1)
        # Clear VAE cache
        pipeline.vae.model.clear_cache()
        # Save the video if the current prompt is not a dummy prompt
        if idx < num_prompts:
            model = "regular" if not args.use_ema else "ema"
            for seed_idx in range(args.num_samples):
                # All processes save their videos
                if args.save_with_index:
                    file_name=f'{idx}-{seed_idx}_{model}.mp4'
                else:
                    file_name=f'source-video-{prompt[:100]}-frame={args.num_output_frames}-{seed_idx}.mp4'
                output_path = os.path.join(args.output_folder, file_name.replace(' ', '_'))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_video(output_path, video[seed_idx], fps=16)
                print(f"Saved video to '{output_path}'")
        all_video = []
    
        video = pipeline.vae.decode_to_pixel(target_latent, use_cache=False)
        video = (video).clamp(0, 1)
        # save video
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        video = 255.0 * torch.cat(all_video, dim=1)
        # Clear VAE cache
        pipeline.vae.model.clear_cache()
        # Save the video if the current prompt is not a dummy prompt
        if idx < num_prompts:
            model = "regular" if not args.use_ema else "ema"
            for seed_idx in range(args.num_samples):
                # All processes save their videos
                if args.save_with_index:
                    file_name=f'{idx}-{seed_idx}_{model}.mp4'
                else:
                    file_name=f'target-video-{prompt[:100]}-frame={args.num_output_frames}-{seed_idx}.mp4'
                output_path = os.path.join(args.output_folder, file_name.replace(' ', '_'))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_video(output_path, video[seed_idx], fps=16)
                print(f"Saved video to '{output_path}'")
        all_video = []


    elif args.task=="i2v":
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    # print(f"[DEBUG] {sampled_noise.shape=}")
    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        y=source_latent,
        wo_scale=True,
        # low_memory=low_memory,
    )

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]
    # print(f"[DEBUG] {num_generated_frames=}")
    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                file_name=f'{idx}-{seed_idx}_{model}.mp4'
            else:
                file_name=f'{prompt[:100]}-frame={args.num_output_frames}-{seed_idx}-step-{args.inference_num_steps}.mp4'
            file_name = args.prefix + file_name
            output_path = os.path.join(args.output_folder, file_name.replace(' ', '_'))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(output_path, video[seed_idx], fps=16)
            print(f"Saved video to '{output_path}'")
