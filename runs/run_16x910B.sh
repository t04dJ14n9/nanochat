#!/bin/bash

# This script is configured to train a GPT-2 grade LLM on 16×Huawei Ascend 910B NPUs
# Each NPU has 64GB HBM, total 1TB aggregate memory
#
# Prerequisites:
#   1. Install CANN toolkit (e.g. 8.0.T13) and source env:
#      source /usr/local/Ascend/ascend-toolkit/set_env.sh
#      (Add to ~/.bashrc to auto-load on every session)
#   2. Check your CANN version and update environment_npu.yml accordingly:
#      cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
#      CANN 8.0.RCx → torch==2.1.0 + torch-npu==2.1.0.post10
#      CANN 8.1.RCx → torch==2.3.1 + torch-npu==2.3.1.postX
#   3. Create and activate conda environment:
#      conda create -n nanochat-npu python=3.9 -y
#      conda activate nanochat-npu
#      pip install -r requirements_npu.txt
#   4. Verify NPU visibility:
#      npu-smi info
#   5. Verify NPU + PyTorch integration:
#      python -c "import torch; import torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
#
# Usage:
#   # Simplest launch:
#   bash runs/run_16x910B.sh
#
#   # Recommended: use screen to persist across SSH disconnects
#   screen -L -Logfile runs/run_16x910B.log -S nanochat bash runs/run_16x910B.sh
#
set -e

# Ensure CANN environment is sourced (skip if already loaded)
if [ -z "$ASCEND_HOME_PATH" ] && [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "Sourced CANN environment from /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Disable wandb (not included in NPU environment)
export WANDB_MODE=disabled

# Skip torch.compile on NPU by default (set to 1 to enable, experimental)
export NANOCHAT_COMPILE=0

# Force BF16 on NPU (910B supports BF16 natively)
export NANOCHAT_DTYPE=bfloat16

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model on 16×910B NPUs
# NPU-specific modifications:
# - No --fp8 flag (910B does not support FP8 training)
# - --device-batch-size=8 (61GB usable HBM per NPU; batch_size=16 causes OOM)
# - --window-pattern=L (NPU SDPA doesn't support sliding window, use full context)
# - Uses hccl backend for distributed training (handled automatically by compute_init)
# - torch.compile is disabled by default on NPU (set NANOCHAT_COMPILE=1 to enable)
torchrun --standalone --nproc_per_node=16 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=8 --window-pattern=L --run=910b

# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=16 -m scripts.base_eval -- --device-batch-size=8

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=16 -m scripts.chat_sft -- --device-batch-size=8 --run=910b_sft
torchrun --standalone --nproc_per_node=16 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
