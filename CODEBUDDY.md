# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Common Commands

### Environment Setup
```bash
# Install dependencies with uv (CUDA version)
uv sync --extra gpu
source .venv/bin/activate

# CPU/MPS version
uv sync --extra cpu
source .venv/bin/activate
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_engine.py -v

# Run tests matching a pattern
python -m pytest tests/ -v -k "test_name"
```

### Training Pipeline

**Complete GPT-2 training pipeline (8×H100):**
```bash
bash runs/speedrun.sh
```

**Custom training for different GPU setups:**
```bash
# Tokenizer training (first time only)
python -m nanochat.dataset -n 8                    # Download initial data shards
python -m nanochat.dataset -n 170 &                # Download remaining in background
python -m scripts.tok_train                        # Train BPE tokenizer
python -m scripts.tok_eval                         # Evaluate tokenizer

# Base model pretraining
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=8 \
    --run=experiment_name

# Model evaluation
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
    --device-batch-size=8

# Supervised fine-tuning (SFT)
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft \
    --device-batch-size=8 \
    --run=sft_experiment

# Chat with the model
python -m scripts.chat_cli -p "Why is the sky blue?"
python -m scripts.chat_web  # Web UI at http://localhost:8000
```

**Quick experiments (smaller models):**
```bash
# GPT-1 size model for rapid iteration (~1 hour on 4×A10)
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --run=d12_test
```

### Development Commands
```bash
# Check linting (if configured)
python -m pytest tests/ -v

# Run specific script module
python -m scripts.base_train --help
python -m scripts.chat_sft --help
```

## Architecture Overview

### Single Complexity Dial Philosophy

The entire system is designed around **one parameter**: `--depth` (number of Transformer layers). All other hyperparameters are automatically calculated for compute-optimal configuration based on this single value:

- Model dimensions: `model_dim = depth * 64`
- Number of heads: `num_heads = model_dim // 128`
- Training iterations: Calculated from `--target-param-data-ratio` (default 12)
- Learning rates, weight decay, etc.: All derived from depth

This design principle means you should never manually tune individual hyperparameters unless you have a specific research reason. The system is optimized end-to-end for the depth parameter.

### Three-Stage Training Pipeline

**Stage 1: Tokenizer (scripts/tok_train.py)**
- Trains BPE tokenizer with vocab size 32,768
- Uses first ~2B characters from dataset (8 shards)
- Outputs to `~/.cache/nanochat/tokenizer/`

**Stage 2: Pretraining (scripts/base_train.py)**
- The most critical and compute-intensive stage
- Trains on ~42.5B tokens (170 shards of ~250M chars each)
- Key technologies:
  - **Muon optimizer** for matrix parameters (orthogonalized gradients)
  - **AdamW optimizer** for embeddings/scalars
  - **Mixed precision**: BF16 on A100/H100, FP32 on older GPUs
  - **FP8 training**: H100+ only (A10 does not support)
  - **Flash Attention 3**: Hopper GPUs, with SDPA fallback for others
- Outputs checkpoints to `~/.cache/nanochat/base_checkpoints/d{depth}/`
- Evaluation metrics:
  - **BPB** (Bits Per Byte): Validation loss
  - **CORE metric**: DCLM benchmark score (GPT-2 baseline = 0.2565)

**Stage 3: Supervised Fine-Tuning (scripts/chat_sft.py)**
- Teaches conversation abilities, tool use, multiple choice
- Uses task mixture: GSM8K (math), MMLU (knowledge), SmolTalk (chat), etc.
- Downloads identity conversation data for personality
- Outputs to `~/.cache/nanochat/chatsft_checkpoints/`

### Core Module Architecture

**Model Layer (nanochat/gpt.py)**
- GPT Transformer with modern architecture:
  - Rotary embeddings (relative positional encoding)
  - QK normalization (attention stabilization)
  - Untied embeddings (separate input/output embeddings)
  - ReLU² activation in MLP
  - Group-Query Attention (GQA) for efficient inference
  - Value Embedding (ResFormer-style residual connections)
  - Sliding window attention patterns (configurable per layer)
- Custom `Linear` layer handles mixed precision automatically:
  - Master weights in FP32 for optimizer precision
  - Casts to BF16 during forward pass
  - No torch.amp.autocast needed

**Optimizer Layer (nanochat/optim.py)**
- Hybrid MuonAdamW optimizer:
  - Matrix parameters → Muon (Newton-Schulz orthogonalization + variance normalization)
  - Embeddings/scalars → AdamW (standard Adam with weight decay)
- Distributed version (`DistMuonAdamW`) for multi-GPU training
- Fused kernels for performance (torch.compile)

**Inference Layer (nanochat/engine.py)**
- Efficient inference engine with KV cache:
  - Prefill phase: Process prompt tokens in parallel
  - Generation phase: Autoregressive token-by-token generation
  - Multi-sample generation with independent sampling
  - Tool use integration (calculator for GSM8K)
- KVCache class manages key-value cache across layers

**Data Layer**
- `nanochat/dataloader.py`: BOS-aligned best-fit dataloader
  - Every sequence starts with BOS token
  - Best-fit algorithm minimizes cropping
  - 100% utilization (no padding), ~35% tokens cropped
- `nanochat/dataset.py`: Dataset download/management
  - Downloads parquet shards from S3
  - Train/val split (last shard is validation)
- `nanochat/tokenizer.py`: BPE tokenizer wrapper

**Checkpoint System (nanochat/checkpoint_manager.py)**
- Save/load model and optimizer state
- Automatic backward compatibility patches for old checkpoints
- Metadata storage (model config, training hyperparameters)
- Resume training from checkpoint

### Task System (tasks/)

Base class `Task` defines interface for training/evaluation datasets:
- `get_example(index)`: Returns conversation dict
- `evaluate(problem, completion)`: Evaluates model output
- `eval_type`: Either 'generative' or 'categorical'

**TaskMixture** combines multiple tasks for SFT training:
- Deterministically shuffled (seed=42)
- Oversample tasks by passing them multiple times

Key tasks:
- **GSM8K**: Grade school math (tool use training)
- **MMLU**: Multiple choice knowledge questions
- **HumanEval**: Simple Python coding tasks
- **SmolTalk**: HuggingFace conversation dataset
- **SpellingBee**: Letter counting (teaches character-level reasoning)

### Evaluation System (nanochat/core_eval.py)

Implements DCLM CORE metric evaluation:
- Multiple choice tasks: Compare likelihood of answer choices
- Language modeling tasks: Compare perplexity of continuations
- Schema tasks: Evaluate structured outputs
- Few-shot prompting support

### Precision Management

The system uses explicit precision control instead of torch.amp.autocast:

| Hardware | Default dtype | Reason |
|----------|---------------|--------|
| H100/A100 (SM 80+) | `bfloat16` | Native BF16 Tensor Cores |
| A10/T4/V100 (SM < 80) | `float32` | No BF16 support |
| CPU/MPS | `float32` | No Tensor Cores |

Override with environment variable: `NANOCHAT_DTYPE=bfloat16`

FP16 training available with `NANOCHAT_DTYPE=float16` (uses GradScaler for stability).

### Distributed Training

Multi-GPU training uses PyTorch DDP:
- `torchrun --nproc_per_node=N` launches N processes
- Gradient accumulation automatically calculated when batch size doesn't divide evenly
- Optimizer state is sharded across ranks (each GPU saves its own optimizer checkpoint)
- DataLoader shards data across ranks by row group index

### Key Design Principles

1. **Minimal configuration**: One dial (`--depth`) controls everything
2. **Compute-optimal defaults**: All hyperparameters calculated from depth
3. **No abstraction layers**: Direct, readable PyTorch code
4. **Explicit over implicit**: No hidden magic or config files
5. **Batteries included**: Complete pipeline from data to chat UI

### Common Patterns

**Adding a new task:**
1. Inherit from `Task` base class in `tasks/`
2. Implement `get_example()`, `evaluate()`, `num_examples()`
3. Add to SFT mixture in `scripts/chat_sft.py` if needed for training

**Modifying the model:**
1. Edit `nanochat/gpt.py` for architecture changes
2. Ensure changes work across all depths (d12 to d26+)
3. Test with d12 for fast iteration

**Debugging training:**
1. Start with d12 model (completes in ~1 hour on 4×A10)
2. Monitor VRAM with `nvidia-smi`
3. Reduce `--device-batch-size` if OOM
4. Check `~/.cache/nanochat/base_checkpoints/` for saved models

**Hardware-specific adjustments:**
- **A10/T4 (24GB VRAM)**: Use `--device-batch-size=8`, no FP8
- **V100 (16GB VRAM)**: Use `--device-batch-size=4`, FP32 or FP16
- **H100/A100 (80GB VRAM)**: Full configuration with FP8 and batch_size=16+

### Cache Directory Structure

All outputs stored in `~/.cache/nanochat/`:
```
~/.cache/nanochat/
├── base_data_climbmix/       # Downloaded parquet shards
├── tokenizer/                 # BPE tokenizer files
├── base_checkpoints/          # Pretrained model checkpoints
│   └── d{depth}/
│       ├── model_{step}.pt
│       ├── optim_{step}_rank{r}.pt
│       └── meta_{step}.json
├── chatsft_checkpoints/       # SFT model checkpoints
├── identity_conversations.jsonl  # Personality data
└── report/                    # Training reports
```

### Performance Metrics

Monitor these during training:
- **val_bpb**: Validation bits per byte (lower is better)
- **core_metric**: DCLM benchmark score (target > 0.2565)
- **bf16_mfu**: Model FLOPs utilization (efficiency metric)
- **tok/sec**: Training throughput
- **VRAM usage**: Should fit within GPU memory

GPT-2 baseline: CORE = 0.256525, achieved with d24-d26 models in ~2-3 hours on 8×H100.
