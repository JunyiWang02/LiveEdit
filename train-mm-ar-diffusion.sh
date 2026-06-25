export MASTER_ADDR=<YOUR_MASTER_ADDR>
export MASTER_PORT=<YOUR_MASTER_PORT>

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path configs/wan_mm-ar-diffusion-local.yaml \
  --logdir ./logs/wan_mm-ar-diffusion-local \
  --no_visualize 