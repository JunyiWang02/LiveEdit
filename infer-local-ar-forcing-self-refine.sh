# Seed-matched self-refining ablation for infer-local-ar-forcing.sh.
# Run the original script for the baseline, then this script for P&P.

CKPT_PATH="checkpoints/liveedit/ar-forcing_002000.pt"

CUDA_VISIBLE_DEVICES=0 python inference-mm.py \
    --config_path configs/wan_mm-ar-forcing-self-refine.yaml \
    --output_folder "videos/self-refine-test" \
    --checkpoint_path "${CKPT_PATH}" \
    --data_path "./test_cases/test.json" \
    --num_output_frames 21 \
    --task v2v \
    --seed 0
