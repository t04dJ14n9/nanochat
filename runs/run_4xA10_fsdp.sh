#!/bin/bash
# FSDP training script for 4×A10 GPUs (24GB each)
#
# FSDP shards model parameters across GPUs, allowing larger models to fit.
# Compared to the default DDP mode (full model replica on each GPU):
#   - DDP:  Each GPU holds all ~0.9B params (~1.8GB in fp32) → d20 fits
#   - FSDP: Each GPU holds ~0.22B params (~0.45GB in fp32) → d26+ fits
#
# Memory guidelines for 4×A10 (24GB each):
#   d20 (~0.9B):  BATCH_SIZE=8, window-pattern=SSSL (fits easily)
#   d24 (~1.5B):  BATCH_SIZE=4, window-pattern=L
#   d26 (~2.1B):  BATCH_SIZE=4, window-pattern=L  (may need BATCH_SIZE=2)
#
# Usage:
#   bash runs/run_4xA10_fsdp.sh                          # Pretrain d20 with FSDP
#   DEPTH=26 bash runs/run_4xA10_fsdp.sh                  # Pretrain d26 (auto-adjusts batch)
#   DEPTH=26 BATCH_SIZE=2 bash runs/run_4xA10_fsdp.sh     # Pretrain d26 with smaller batch

set -e

# Configuration
DEPTH=${DEPTH:-20}
# Auto-adjust batch size based on depth for A10 24GB VRAM
if [ -z "$BATCH_SIZE" ]; then
    if [ "$DEPTH" -ge 26 ]; then
        BATCH_SIZE=4
    elif [ "$DEPTH" -ge 24 ]; then
        BATCH_SIZE=4
    else
        BATCH_SIZE=8
    fi
fi
# Use local attention for d24+ (SSSL is inefficient without Flash Attention 3)
if [ -z "$WINDOW_PATTERN" ]; then
    if [ "$DEPTH" -ge 24 ]; then
        WINDOW_PATTERN="L"
    else
        WINDOW_PATTERN="SSSL"
    fi
fi
RUN_NAME=${RUN_NAME:-"fsdp_d${DEPTH}"}

echo "========================================"
echo "NanoChat FSDP Training on 4×A10"
echo "========================================"
echo "Depth:         ${DEPTH}"
echo "Batch size:    ${BATCH_SIZE} per device"
echo "Window pattern:${WINDOW_PATTERN}"
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
    --window-pattern=${WINDOW_PATTERN} \
    --parallelism=fsdp \
    --fsdp-sharding=full \
    --no-compile \
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
    --no-compile \
    --run=${RUN_NAME}_sft

echo "Done! Checkpoints saved to ~/.cache/nanochat/"
