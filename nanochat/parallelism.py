"""Distributed model-state sharding helpers.

The default nanochat training path keeps a full model replica on every rank and
uses MuonAdamW's custom collectives. The FSDP2 path in this module instead
shards parameters, gradients, and optimizer state across ranks so models that
do not fit on a single GPU can be trained.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nanochat.common import COMPUTE_DTYPE, print0


def apply_fsdp2(model, world_size):
    """Shard a GPT bottom-up with PyTorch FSDP2 and return its device mesh.

    Large standalone tables and every Transformer block receive their own FSDP
    group. The GPT root is sharded last, as required by the composable FSDP2
    API, and owns only the remaining small parameters.
    """
    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires an initialized torch.distributed process group")
    if world_size < 2:
        raise ValueError("FSDP2 model sharding requires at least two processes")
    if not torch.cuda.is_available():
        raise RuntimeError("nanochat FSDP2 currently requires CUDA")

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=COMPUTE_DTYPE,
        reduce_dtype=torch.float32,
    )
    shard_kwargs = {"mesh": mesh, "mp_policy": mp_policy}

    # Apply FSDP2 bottom-up. Keeping the large embeddings and output head in
    # separate groups avoids gathering all of them at once in the root group.
    fully_shard(model.transformer.wte, **shard_kwargs)
    for value_embedding in model.value_embeds.values():
        fully_shard(value_embedding, **shard_kwargs)
    for block in model.transformer.h:
        fully_shard(block, **shard_kwargs)
    fully_shard(model.lm_head, **shard_kwargs)
    fully_shard(model, **shard_kwargs)

    print0(
        f"FSDP2 enabled: parameters, gradients, and optimizer state are sharded "
        f"across {world_size} GPUs"
    )
    return mesh


def setup_fsdp_adamw(model, lr=3e-4, weight_decay=0.1):
    """Create a DTensor-compatible AdamW optimizer after FSDP2 sharding.

    MuonAdamW implements its own ZeRO-style communication and assumes complete
    matrix gradients, so it cannot be composed with FSDP2's gradient shards.
    Native AdamW operates directly on FSDP2 DTensor parameters.
    """
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay_params if param.ndim >= 2 else no_decay_params).append(param)

    param_groups = []
    if decay_params:
        param_groups.append({
            "kind": "adamw",
            "params": decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    if no_decay_params:
        param_groups.append({
            "kind": "adamw",
            "params": no_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
        })
    if not param_groups:
        raise ValueError("Cannot construct an optimizer for a model with no trainable parameters")

    # foreach=True materializes optimizer-wide temporary tensor lists (notably
    # sqrt(exp_avg_sq)) and can consume tens of GiB for very large models.
    # Single-tensor updates keep the transient bounded by one parameter shard.
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    print0(
        f"FSDP2 optimizer: memory-efficient AdamW (lr={lr:g}, weight_decay={weight_decay:g}); "
        "Muon is disabled because it requires complete matrix gradients"
    )
    return optimizer


def parameter_numel(model, local=False):
    """Count global or rank-local parameter elements for sharding diagnostics."""
    total = 0
    for param in model.parameters():
        if local and isinstance(param, DTensor):
            total += param.to_local().numel()
        else:
            total += param.numel()
    return total
