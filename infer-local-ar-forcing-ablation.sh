#!/usr/bin/env bash
# Seed-matched inference entry point for LiveEdit baseline vs. self-refining.
#
# Usage:
#   bash infer-local-ar-forcing-ablation.sh baseline
#   bash infer-local-ar-forcing-ablation.sh refine
#
# Optional overrides, for example:
#   GPU=1 SEED=42 OUTPUT_FOLDER=videos/my-refine \
#     bash infer-local-ar-forcing-ablation.sh refine

set -euo pipefail

MODE="${1:-baseline}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SEED="${SEED:-0}"
CKPT_PATH="${CKPT_PATH:-checkpoints/liveedit/ar-forcing_002000.pt}"
DATA_PATH="${DATA_PATH:-./test_cases/test.json}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"

case "${MODE}" in
    baseline)
        CONFIG_PATH="configs/wan_mm-ar-forcing-local.yaml"
        DEFAULT_OUTPUT_FOLDER="videos/ablation/baseline-seed${SEED}"
        ;;
    refine)
        CONFIG_PATH="configs/wan_mm-ar-forcing-self-refine.yaml"
        DEFAULT_OUTPUT_FOLDER="videos/ablation/refine-seed${SEED}"
        ;;
    *)
        echo "Unknown mode: ${MODE}. Use 'baseline' or 'refine'." >&2
        exit 2
        ;;
esac

OUTPUT_FOLDER="${OUTPUT_FOLDER:-${DEFAULT_OUTPUT_FOLDER}}"

echo "[Ablation] mode=${MODE}, seed=${SEED}, gpu=${GPU}"
echo "[Ablation] config=${CONFIG_PATH}"
echo "[Ablation] output=${OUTPUT_FOLDER}"

CUDA_VISIBLE_DEVICES="${GPU}" python inference-mm.py \
    --config_path "${CONFIG_PATH}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --checkpoint_path "${CKPT_PATH}" \
    --data_path "${DATA_PATH}" \
    --num_output_frames "${NUM_OUTPUT_FRAMES}" \
    --task v2v \
    --seed "${SEED}"
