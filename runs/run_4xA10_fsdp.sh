#!/bin/bash
# FSDP training script for 4×A10 GPUs (24GB each)
#
# FSDP shards model parameters across GPUs, allowing larger models to fit.
# Compared to the default DDP mode (full model replica on each GPU):
#   - DDP:  Each GPU holds all ~0.9B params (~1.8GB in fp32) → d20 fits
#   - FSDP: Each GPU holds ~0.22B params (~0.45GB in fp32) → d26+ fits
#
# Usage:
#   bash runs/run_4xA10_fsdp.sh              # Pretrain d20 with FSDP
#   DEPTH=26 bash runs/run_4xA10_fsdp.sh     # Pretrain d26 (won't fit with DDP)

set -e

# Configuration
DEPTH=${DEPTH:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
RUN_NAME=${RUN_NAME:-"fsdp_d${DEPTH}"}

echo "========================================"
echo "NanoChat FSDP Training on 4×A10"
echo "========================================"
echo "Depth:         ${DEPTH}"
echo "Batch size:    ${BATCH_SIZE} per device"
echo "Parallelism:   FSDP (full sharding)"
echo "Run name:      ${RUN_NAME}"
echo "========================================"

# Step 1: Download data (if not already done)
echo "[Step 1/5] Downloading data..."
python -m nanochat.dataset -n 170

# Step 2: Train tokenizer (if not already done)
echo "[Step 2/5] Training tokenizer..."
python -m scripts.tok_train

# Step 3: Pretrain with FSDP
echo "[Step 3/5] Pretraining base model with FSDP..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=${DEPTH} \
    --device-batch-size=${BATCH_SIZE} \
    --parallelism=fsdp \
    --fsdp-sharding=full \
    --run=${RUN_NAME}

# Step 4: Evaluate the model
echo "[Step 4/5] Evaluating base model..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
    --device-batch-size=${BATCH_SIZE}

# Step 5: SFT with FSDP
echo "[Step 5/5] Running SFT with FSDP..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    --device-batch-size=${BATCH_SIZE} \
    --parallelism=fsdp \
    --fsdp-sharding=full \
    --run=${RUN_NAME}_sft

echo "Done! Checkpoints saved to ~/.cache/nanochat/"
