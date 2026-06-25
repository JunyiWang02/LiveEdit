#!/bin/bash

# AR-oriented Mask Cache inference.
# This uses the token-pruning config to reuse computation in unchanged regions.

# Activate your environment before running, for example:
# conda activate liveedit

CKPT_PATH="checkpoints/liveedit/ar-forcing_002000.pt"
DATA_PATH="test_cases/test.json"

CUDA_VISIBLE_DEVICES=0 python inference-mm.py \
    --config_path configs/wan_mm-token-pruning.yaml \
    --output_folder "videos/mask-cache-test" \
    --checkpoint_path "${CKPT_PATH}" \
    --data_path "${DATA_PATH}" \
    --num_output_frames 21 \
    --prefix "mask_cache_" \
    --task v2v \
    --save_mask
