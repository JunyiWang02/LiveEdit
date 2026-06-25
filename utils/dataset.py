from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import decord

class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }
class TextVideoPairDataset(Dataset):
    def __init__(
        self,
        json_path,
        root_dir="",
        transform=None,
        num_frames=16,
        sample_rate=1,
        eval_first_n=-1
    ):
        """
        Args:
            json_path (str): 包含 source_path, edited_path, instruction 的 JSON 文件路径
            root_dir (str): 视频文件的根目录（如果 JSON 里的路径是相对路径）
            transform (callable, optional): 同时应用于 source 和 edited 视频的变换
            num_frames (int): 抽取的总帧数
            sample_rate (int): 抽帧间隔
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_frames = num_frames
        self.sample_rate = sample_rate

        with open(json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        if eval_first_n != -1:
            self.metadata = self.metadata[:eval_first_n]

    def __len__(self):
        return len(self.metadata)

    def _get_frames(self, video_path):
        """使用 decord 载入视频并均匀/步进采样帧"""
        try:
            full_path = self.root_dir / video_path
            vr = decord.VideoReader(str(full_path), ctx=decord.cpu(0))
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # 返回全黑帧作为 fallback（或者抛出错误）
            return [Image.new('RGB', (224, 224), (0, 0, 0))] * self.num_frames

        total_frames = len(vr)
        # 采样逻辑：根据 sample_rate 计算索引
        # 如果视频长度不足，则进行简单处理
        required_len = self.num_frames * self.sample_rate
        
        if total_frames >= required_len:
            # 正常采样
            indices = np.arange(0, required_len, self.sample_rate)
        else:
            # 视频太短时的策略：重复最后一帧或等间距取帧
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = vr.get_batch(indices).asnumpy() # (T, H, W, C)
        # 为了适应vae的输入要求 (B, C, T, H, W)
        return [Image.fromarray(f) for f in frames]

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 1. 加载视频帧。推理时可以只提供 source_path；成对评估/训练仍可提供 edited_path。
        source_frames = self._get_frames(item['source_path'])
        edited_frames = self._get_frames(item['edited_path']) if 'edited_path' in item else None

        # 2. 应用变换
        if self.transform:
            source_video = torch.stack([self.transform(f) for f in source_frames]).permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
            if edited_frames is not None:
                edited_video = torch.stack([self.transform(f) for f in edited_frames]).permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

        else:
            # [W, H] * T
            source_video = torch.stack([torch.tensor(np.array(f)) for f in source_frames]) # [T, H, W, C]
            if edited_frames is not None:
                edited_video = torch.stack([torch.tensor(np.array(f)) for f in edited_frames]) # [T, H, W, C]

        # 3. 整理输出
        batch = {
            "source_video": source_video,    # (T, H, W, C)
            "prompts": item['instruction'],  # 这里的指令对应之前的 prompts
            "idx": idx
        }
        if edited_frames is not None:
            batch["edited_video"] = edited_video    # (T, H, W, C)
        return batch

def cycle(dl):
    while True:
        for data in dl:
            yield data
