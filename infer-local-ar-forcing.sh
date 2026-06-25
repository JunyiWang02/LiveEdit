# mkdir -p checkpoints/liveedit
# huggingface-cli download cp-cp/LiveEdit ar-forcing_002000.pt \
#   --local-dir checkpoints/liveedit

CKPT_PATH="checkpoints/liveedit/ar-forcing_002000.pt"

CUDA_VISIBLE_DEVICES=0 python inference-mm.py \
    --config_path configs/wan_mm-ar-forcing-local.yaml \
    --output_folder "videos/test" \
    --checkpoint_path "${CKPT_PATH}" \
    --data_path "./test_cases/test.json" \
    --num_output_frames 21 \
    --task v2v