"""
FSDP (Fully Sharded Data Parallel) utilities for nanochat.

This module provides FSDP-based model parallelism as an alternative to the
default DistMuonAdamW approach. With FSDP, model parameters, gradients, and
optimizer states are sharded across GPUs, reducing per-GPU memory from O(N_params)
to O(N_params / world_size).

Key differences from the default DistMuonAdamW approach:
- Default (ddp): Full model replica on each GPU, custom ZeRO-2 optimizer sharding
- FSDP:        Model parameters sharded across GPUs, PyTorch-managed communication

When to use FSDP:
- When model doesn't fit on a single GPU (e.g. d26+ on 4×A10)
- When you want to train larger models with limited VRAM
- When you want automatic memory management without manual tuning

Usage:
    torchrun --nproc_per_node=4 -m scripts.base_train --parallelism fsdp

Architecture:
- FSDP wraps each Transformer Block as a sharding unit (good granularity)
- use_orig_params=True for compatibility with our custom MuonAdamW optimizer
- MixedPrecision policy handles bf16 compute with fp32 master weights
- Checkpoint save/load uses FSDP's full_state_dict API for portability
"""

import os
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from nanochat.gpt import GPT, Block
from nanochat.common import COMPUTE_DTYPE, print0


def get_fsdp_wrap_policy():
    """
    Get the auto-wrap policy for FSDP.

    We wrap each Transformer Block as a sharding unit. This gives good
    granularity: each block's parameters are sharded together, and
    all-gather happens once per block during forward/backward.

    transformer_auto_wrap_policy is a callable that FSDP invokes as:
        policy_fn(module, recurse, nonwrapped_numel) -> bool
    We use functools.partial to pre-bind the transformer_layer_cls argument.
    """
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})


def get_fsdp_mixed_precision():
    """
    Configure FSDP mixed precision to match nanochat's existing precision strategy.

    nanochat's approach: master weights in fp32, activations in COMPUTE_DTYPE (bf16/fp16).
    The custom Linear layer already casts weights to activation dtype in forward().
    FSDP's MixedPrecision should be configured to:
    - Keep parameters in fp32 for optimizer precision (param_dtype)
    - Compute in bf16 for Tensor Core efficiency (compute_dtype)
    - Reduce gradients in bf16 to save communication bandwidth (reduce_dtype)
    """
    if COMPUTE_DTYPE == torch.bfloat16:
        return MixedPrecision(
            param_dtype=torch.float32,      # master weights stay fp32
            reduce_dtype=torch.bfloat16,    # gradient all-reduce in bf16
            buffer_dtype=torch.bfloat16,    # buffers (rotary emb) in bf16
        )
    elif COMPUTE_DTYPE == torch.float16:
        return MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:  # float32
        return None  # no mixed precision


def wrap_model_fsdp(model, device, sharding_strategy="full"):
    """
    Wrap a GPT model with FSDP for distributed training.

    Args:
        model: The GPT model (already on device, weights initialized)
        device: The CUDA device for this rank
        sharding_strategy: "full" (ZeRO-3), "shard_grad_op" (ZeRO-2), or "no_shard" (DDP)

    Returns:
        The FSDP-wrapped model

    Note:
        - use_orig_params=True is required for our custom MuonAdamW optimizer
          to access original parameter objects for grouping by shape/kind.
        - sync_module_states=True ensures all ranks start from the same model
          state (important when only rank 0 loads from checkpoint).
        - device_id is set for efficient NCCL communication.
    """
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,           # ZeRO-3: shard params + grads + optim
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2: shard grads + optim only
        "no_shard": ShardingStrategy.NO_SHARD,         # DDP equivalent: replicate everything
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    mixed_precision = get_fsdp_mixed_precision()
    wrap_policy = get_fsdp_wrap_policy()

    # When using FSDP, we need to make sure the model is on the correct device
    # before wrapping. The wrapping process will shard parameters.
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=strategy,
        use_orig_params=True,       # Required for custom optimizer param grouping
        sync_module_states=True,    # Sync model state across ranks during init
        device_id=device,
    )

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in fsdp_model.parameters())
        sharded_params = sum(p.numel() for p in fsdp_model.parameters())
        print(f"FSDP: Total params across all ranks: {total_params:,}")
        print(f"FSDP: Sharding strategy: {sharding_strategy}")

    return fsdp_model


def save_fsdp_checkpoint(fsdp_model, optimizer, checkpoint_dir, step, meta_data, rank=0):
    """
    Save a checkpoint from an FSDP-wrapped model.

    Uses FSDP's full_state_dict API to gather the complete model state on rank 0,
    then saves it in the same format as non-FSDP checkpoints for compatibility.

    Optimizer state is saved per-rank (sharded), matching the existing convention.

    Args:
        fsdp_model: The FSDP-wrapped model
        optimizer: The optimizer (single-GPU MuonAdamW when using FSDP)
        checkpoint_dir: Directory to save checkpoints
        step: Current training step
        meta_data: Metadata dict to save as JSON
        rank: This rank's ID
    """
    import json
    from nanochat.checkpoint_manager import save_checkpoint

    # Gather full model state dict on rank 0
    # This is expensive (all-gather) but ensures compatibility with
    # non-FSDP checkpoint loading
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = fsdp_model.state_dict()

    # Remove torch.compile prefix if present
    model_state_dict = {k.removeprefix("_orig_mod."): v for k, v in model_state_dict.items()}

    # Save using existing checkpoint utility (rank 0 saves model, all ranks save optimizer shard)
    save_checkpoint(
        checkpoint_dir,
        step,
        model_state_dict,
        optimizer.state_dict() if optimizer is not None else None,
        meta_data,
        rank=rank,
    )


def load_fsdp_model_for_inference(source, device, model_tag=None, step=None):
    """
    Load an FSDP-trained model for inference (single GPU).

    Since we save checkpoints in full state dict format (compatible with non-FSDP),
    we can just use the regular load_model function.

    Args:
        source: "base", "sft", or "rl"
        device: Target device
        model_tag: Optional model tag override
        step: Optional step override

    Returns:
        model, tokenizer, meta_data
    """
    from nanochat.checkpoint_manager import load_model
    return load_model(source, device, phase="eval", model_tag=model_tag, step=step)


def get_fsdp_optim_param_groups(model):
    """
    Build parameter groups for MuonAdamW optimizer from an FSDP-wrapped model.

    With use_orig_params=True, FSDP preserves the original parameter objects,
    so we can use the model's setup_optimizer() method directly. This function
    is a convenience wrapper.

    Note: When using FSDP, always use the single-GPU MuonAdamW (not DistMuonAdamW),
    because FSDP handles gradient synchronization itself.
    """
    return model.setup_optimizer()
