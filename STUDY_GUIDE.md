# NanoChat Study Guide (4×A10)

> A comprehensive guide for understanding the NanoChat codebase, designed for hands-on learning on 4× NVIDIA A10 GPUs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Source Code Walkthrough](#3-source-code-walkthrough)
4. [Training Pipeline](#4-training-pipeline)
5. [Your 4×A10 Setup: What Works & What Doesn't](#5-your-4xa10-setup)
6. [Model Parallelism Analysis: Can We Split Weights Evenly?](#6-model-parallelism-analysis)
7. [Study Path & Exercises](#7-study-path--exercises)
8. [Reference Materials](#8-reference-materials)

---

## 1. Project Overview

NanoChat is a from-scratch GPT training system by Andrej Karpathy. It implements the complete pipeline from data to chat model with one design philosophy: **a single complexity dial (`--depth`) controls everything**.

### Key Innovation: Depth = Everything

```
depth=12 → model_dim=768,  heads=6,  ~100M params   (GPT-1 scale)
depth=20 → model_dim=1280, heads=10, ~900M params   (your current model)
depth=24 → model_dim=1536, heads=12, ~1.6B params   (GPT-2 scale)
depth=26 → model_dim=1664, heads=13, ~2.1B params   (matches GPT-2)
```

All hyperparameters (LR, batch size, weight decay, training steps) are **automatically derived** from depth using scaling laws.

### Modern Architecture Features

| Feature | NanoChat | GPT-2 (original) | Why it matters |
|---------|----------|-------------------|----------------|
| Positional encoding | Rotary (RoPE) | Learned absolute | Generalizes to longer sequences |
| Normalization | RMSNorm (no params) | LayerNorm | Simpler, faster |
| Activation | ReLU² | GELU | Simpler, competitive |
| Attention | QK-norm + GQA + sliding window | Standard MHA | More stable, efficient |
| Optimizer | Muon (matrices) + AdamW (rest) | Adam | Better convergence |
| Embeddings | Untied (separate wte/lm_head) | Tied | More parameters, better performance |
| Value Embeddings | ResFormer-style | None | Skip connections from input |
| Logit handling | Softcap (tanh squash) | Raw | Prevents logit explosion |

---

## 2. Architecture Deep Dive

### Data Flow Diagram

```
Input Token IDs (B, T)
       │
       ▼
┌──────────────┐
│  wte (Embed) │  → token embeddings (B, T, n_embd)
│  + RMSNorm   │  → normalized embeddings
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Smear Gate  │  → mix prev token embedding into current (cheap bigram info)
└──────┬───────┘
       │  x0 = x (save for residual blending)
       │
       ▼
┌──────────────────────────────────────────────────┐
│  For each Transformer Block i (0..n_layer-1):    │
│                                                    │
│  x = resid_lambdas[i] * x + x0_lambdas[i] * x0   │ ← residual scaling + input blending
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │  CausalSelfAttention                     │     │
│  │  1. Project Q, K, V                     │     │
│  │  2. Mix Value Embedding (alternating)    │     │ ← ResFormer-style skip from input
│  │  3. Apply Rotary Embeddings             │     │
│  │  4. QK Normalization + scale             │     │
│  │  5. Flash Attention (FA3 or SDPA)       │     │
│  │  6. Output projection                   │     │
│  └──────────────────────────────────────────┘     │
│  x = x + attn(norm(x))     ← Pre-norm residual   │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │  MLP                                      │     │
│  │  1. c_fc: (n_embd → 4*n_embd)           │     │
│  │  2. ReLU² activation                     │     │
│  │  3. c_proj: (4*n_embd → n_embd)          │     │
│  └──────────────────────────────────────────┘     │
│  x = x + mlp(norm(x))      ← Pre-norm residual   │
│                                                    │
│  If i == n_layer//2: cache x as x_backout         │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│  Backout         │  → x = x - backout_lambda * x_backout
│  (subtract mid-  │     Remove low-level features before logit projection
│   layer residual) │
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  RMSNorm     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  lm_head     │  → logits (B, T, vocab_size)
│  + Softcap   │  → logits = 15 * tanh(logits / 15)
└──────┬───────┘
       │
       ▼
  Cross-Entropy Loss
```

### Your d20 Model Anatomy

```
Parameter counts (d20):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
wte (embedding)         :  41,943,040   (4.7%)  ← AdamW optimized
value_embeds            : 419,430,400  (46.8%)  ← AdamW optimized (alternating layers)
lm_head (unembedding)   :  41,943,040   (4.7%)  ← AdamW optimized
transformer_matrices    : 393,217,200  (43.8%)  ← Muon optimized
scalars                 :          66   (0.0%)  ← AdamW optimized
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                   : 896,533,746  (~0.9B)

Memory breakdown (BF16 training):
- Model weights (BF16):     ~1.7 GB
- Optimizer state (FP32):   ~3.4 GB  (2× model for momentum + variance)
- Gradients (FP32):         ~3.4 GB
- Activations + temp:       ~8-10 GB
- Total per GPU (DDP):      ~16-18 GB  ← fits in A10 24GB
```

### Sliding Window Attention Pattern (SSSL)

```
Layer 0: S (short=512)    │░░░░░░░│████████│
Layer 1: S (short=512)    │░░░░░░░│████████│
Layer 2: S (short=512)    │░░░░░░░│████████│
Layer 3: L (full=2048)    │████████████████│
Layer 4: S (short=512)    │░░░░░░░│████████│
...                        (pattern tiles)
Layer 19: L (full=2048)   │████████████████│  ← last layer always full

S = attend to 1/4 of context (cheaper)
L = attend to full context (expensive but needed)
```

**A10 issue**: SDPA does not efficiently support sliding windows. The `SSSL` pattern
still computes full attention but masks out tokens — no actual speedup. Use `--window-pattern L`
for equal speed but better quality on A10.

---

## 3. Source Code Walkthrough

### File Map

```
nanochat/
├── gpt.py              ← THE model (GPT, Block, CausalSelfAttention, MLP)
├── optim.py            ← MuonAdamW / DistMuonAdamW optimizer
├── engine.py           ← Inference engine with KV cache + tool use
├── common.py           ← Distributed init, dtype detection, GPU peak flops
├── dataloader.py       ← BOS-aligned best-fit dataloader
├── dataset.py          ← Parquet shard download/management
├── tokenizer.py        ← BPE tokenizer wrapper
├── checkpoint_manager.py ← Save/load model + optimizer + metadata
├── flash_attention.py  ← FA3/SDPA abstraction layer
├── fp8.py              ← FP8 training (H100+ only)
├── loss_eval.py        ← Validation BPB evaluation
├── core_eval.py        ← DCLM CORE benchmark evaluation
└── report.py           ← Training report generation

scripts/
├── base_train.py       ← Stage 1: Pretrain
├── base_eval.py        ← Evaluate base model
├── chat_sft.py         ← Stage 2: SFT
├── chat_rl.py          ← Stage 3: RL (GRPO on GSM8K)
├── chat_eval.py        ← Chat evaluation
├── chat_cli.py         ← CLI chat interface
├── chat_web.py         ← Web chat UI
└── tok_train.py        ← Tokenizer training

tasks/
├── common.py           ← Task base class + TaskMixture
├── gsm8k.py            ← Math problem task (with tool use)
├── mmlu.py             ← Multiple choice knowledge
├── smoltalk.py         ← General conversations
├── spellingbee.py      ← Character counting
├── customjson.py       ← Custom JSON conversations
└── humaneval.py        ← Python coding task
```

### Key Code Insights

#### 1. `nanochat/gpt.py` — The Model

**Meta Device Initialization Pattern** (lines 155-161):
```python
# Step 1: Build on meta device (shapes only, no data)
with torch.device("meta"):
    model = GPT(config)
# Step 2: Allocate storage on target device
model.to_empty(device=device)
# Step 3: Initialize all weights
model.init_weights()
```
This 3-step pattern avoids ever allocating the full model on CPU (which would double memory usage).

**Custom Linear Layer** (lines 45-50):
```python
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```
Instead of `torch.autocast`, the weight is cast to the activation dtype in forward.
Master weights stay FP32 for optimizer precision, but matmuls run in BF16.

**Value Embedding (ResFormer)** (lines 53-55, 91-95):
Alternating layers get a skip connection from the input token embedding:
```python
# has_ve: True for alternating layers, always True for last layer
if ve is not None:
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))  # (B, T, n_kv_head)
    v = v + gate.unsqueeze(-1) * ve  # blend value embedding
```
This lets deep layers "look back" at the original input — a form of learnable skip connection.

**Smear** (lines 427-444):
A cheap way to give each token bigram information from its neighbor:
```python
gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]])
```

**Backout** (lines 456-459):
Subtract a mid-layer residual before the final projection:
```python
x = x - backout_lambda * x_backout  # remove low-level features
```

#### 2. `nanochat/optim.py` — The Optimizer

**Two Optimizers in One**:
- **AdamW**: For embeddings (wte, value_embeds, lm_head) and scalars
- **Muon**: For all 2D matrix parameters (attention Q/K/V/O, MLP fc/proj)

**Muon Algorithm** (simplified):
```
1. Nesterov momentum: g = grad + momentum * (momentum_buffer - grad)
2. Polar Express orthogonalization: make g near-orthogonal via 5 iterations
3. Variance reduction (NorMuon): normalize column/row scales
4. Cautious update: only apply weight decay where sign(grad) == sign(weight)
```

**Distributed Optimizer** (`DistMuonAdamW`):
- 3-phase async communication to overlap compute and networking
- AdamW: small params → all_reduce; large params → reduce_scatter + all_gather
- Muon: stack same-shaped params, divide across ranks, reduce_scatter + all_gather
- Optimizer state is **sharded** across ranks (ZeRO-2 style)

#### 3. `nanochat/engine.py` — Inference

**Two-Phase Generation**:
1. **Prefill**: Process all prompt tokens at once (batch=1)
2. **Decode**: Clone KV cache to `num_samples` copies, then autoregressive decode

**Tool Use (Calculator)**:
The engine detects `<|python_start|>...<|python_end|>` in generated text,
evaluates the expression, and injects `<|output_start|>result<|output_end|>` tokens.

**KVCache** (lines 82-137):
- Layout: `(n_layers, B, T, H, D)` — FA3's native format (not BHTD)
- Position tracked via `cache_seqlens` tensor
- Supports prefill from another cache (for batch=1 → batch=N expansion)

#### 4. `nanochat/dataloader.py` — Data Loading

**BOS-Aligned Best-Fit Algorithm**:
```
For each row of T+1 tokens:
1. Find largest document that fits entirely → place it
2. Repeat until no document fits
3. Crop the shortest document to fill remaining space
Result: 100% utilization (no padding), ~35% tokens cropped
```

**DDP Sharding**: Each rank reads every N-th row group from parquet files.

---

## 4. Training Pipeline

### Stage 1: Pretraining (`scripts/base_train.py`)

```
Data: ~42.5B tokens (170 parquet shards)
Optimizer: MuonAdamW (hybrid)
Schedule: Linear warmup (40 steps) → constant → linear warmdown (65% of training)
Total steps: depth * data_ratio / batch_size  (3,320 for d20)
```

**Scaling Law Derivations** (all from depth):
```
1. num_scaling_params = transformer_matrices + lm_head
2. target_tokens = target_param_data_ratio * num_scaling_params  (8:1)
3. batch_size = B_ref * (target_tokens/D_ref)^0.383  (Power Lines paper)
4. LR_scale = sqrt(batch_size/B_ref)  (AdamW sqrt scaling)
5. weight_decay = wd_ref * sqrt(B/B_ref) * (D_ref/D)  (T_epoch framework)
```

### Stage 2: SFT (`scripts/chat_sft.py`)

```
Data mixture: SmolTalk + MMLU×3 + GSM8K×4 + SpellingBee + Identity
LR: 80% of pretraining LR (init_lr_frac=0.8)
Schedule: No warmup → constant → 50% warmdown to 0
Stops: After 1 epoch over the training mixture
Loss mask: Only train on assistant responses (mask=1), ignore prompts (mask=0)
```

### Stage 3: RL (`scripts/chat_rl.py`)

```
Task: GSM8K only (math word problems)
Algorithm: Simplified GRPO ≈ REINFORCE with mean-subtracted advantages
Samples: 16 per question, temperature=1.0
Reward: 1.0 if correct answer, 0.0 otherwise
LR: 5% of base LR (very conservative for stability)
```

---

## 5. Your 4×A10 Setup

### What Works

| Feature | Status | Notes |
|---------|--------|-------|
| d20 pretraining | ✅ | ~14.5 hours, fits in 24GB |
| d12 pretraining | ✅ | ~1 hour, great for experiments |
| SFT | ✅ | Same memory as pretraining |
| RL | ✅ | Actually lighter (no grad accumulation) |
| BF16 training | ✅ | A10 SM 86 supports BF16 |

### What Doesn't Work

| Feature | Status | Workaround |
|---------|--------|------------|
| Flash Attention 3 | ❌ | SDPA fallback (slower) |
| FP8 training | ❌ | Not supported on A10 |
| MFU metric | ❌ | A10 not in peak_flops table |
| Sliding window speedup | ❌ | SDPA can't skip computation |
| d24+ models | ❌ | OOM at batch_size=4 |

### A10-Specific Recommendations

```bash
# Use full context attention (SSSL wastes compute on SDPA)
--window-pattern L

# Batch size 8 works for d20
--device-batch-size 8

# For d24, try batch size 4 or reduce sequence length
--depth=24 --device-batch-size=4 --max-seq-len=1024
```

### Adding A10 to Peak Flops Table

Edit `nanochat/common.py` and add after the A40 line:
```python
(["a10"], 125e12),   # A10 BF16 peak: 125 TFLOPS
```

---

## 6. Model Parallelism Analysis

### Can We Split Model Weights Evenly on 4 GPUs?

**Yes! FSDP is now implemented on the `feature/fsdp-model-parallelism` branch.**

#### Current Approach: Data Parallelism (DDP, default)

```
GPU 0: Full model copy (1.7GB weights) + Full optimizer (3.4GB) + Full gradients (3.4GB)
GPU 1: Full model copy (1.7GB weights) + Full optimizer (3.4GB) + Full gradients (3.4GB)
GPU 2: Full model copy (1.7GB weights) + Full optimizer (3.4GB) + Full gradients (3.4GB)
GPU 3: Full model copy (1.7GB weights) + Full optimizer (3.4GB) + Full gradients (3.4GB)

Total per GPU: ~16-18 GB
```

Every GPU holds a **complete copy** of the model. The only sharding is:
- **Data**: Different batches on each GPU
- **Optimizer state**: Already partially sharded (DistMuonAdamW does ZeRO-2)

#### NEW: FSDP (Fully Sharded Data Parallel) — Now Implemented

FSDP shards model parameters, gradients, AND optimizer state across GPUs:

```
GPU 0: 1/4 of model weights (0.4GB) + 1/4 optimizer + 1/4 gradients
GPU 1: 1/4 of model weights (0.4GB) + 1/4 optimizer + 1/4 gradients
GPU 2: 1/4 of model weights (0.4GB) + 1/4 optimizer + 1/4 gradients
GPU 3: 1/4 of model weights (0.4GB) + 1/4 optimizer + 1/4 gradients

Total per GPU: ~4-6 GB  (before all-gather for forward/backward)
Peak per GPU: ~10-12 GB (during forward when params are gathered)
```

**Usage:**
```bash
# Train d20 with FSDP (same model, less memory per GPU)
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --parallelism=fsdp --run=dummy --device-batch-size=8

# Train d26 (~2.1B params) — won't fit with DDP on A10!
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
    --depth=26 --parallelism=fsdp --run=dummy --device-batch-size=8

# Or use the convenience script:
bash runs/run_4xA10_fsdp.sh
DEPTH=26 bash runs/run_4xA10_fsdp.sh  # train d26
```

**Key implementation details:**

| Aspect | DDP (default) | FSDP |
|--------|---------------|------|
| Optimizer | `DistMuonAdamW` (ZeRO-2 built-in) | `MuonAdamW` (single-GPU, FSDP handles sync) |
| Gradient sync | Manual reduce_scatter in optimizer | FSDP automatic |
| Model params | Full replica per GPU | Sharded (1/N per GPU) |
| torch.compile | After model init | After FSDP wrapping |
| Checkpoint save | `orig_model.state_dict()` | FSDP `full_state_dict` API |
| Checkpoint load | Standard `load_state_dict()` | Same (compatible format) |

**How it works:**

1. **Wrap each Block as FSDP unit**: `transformer_auto_wrap_policy(transformer_layer_cls={Block})` — each Transformer block is a sharding boundary
2. **use_orig_params=True**: FSDP preserves original parameter objects, so MuonAdamW can still group params by shape for stacked orthogonalization
3. **MixedPrecision**: FSDP handles bf16 compute with fp32 master weights, matching nanochat's existing approach
4. **sync_module_states=True**: All ranks start from the same model state during init
5. **Backward compatible**: Checkpoints saved by FSDP are in the same format as DDP checkpoints

**FSDP sharding strategies:**

| Strategy | What's sharded | Memory savings | Communication |
|----------|---------------|----------------|---------------|
| `--fsdp-sharding=full` (ZeRO-3) | Params + Grads + Optim | 4x on 4 GPUs | all-gather fwd/bwd |
| `--fsdp-sharding=shard_grad_op` (ZeRO-2) | Grads + Optim | ~2x on 4 GPUs | all-reduce grads only |

**Current limitations:**
- CORE metric evaluation is skipped in FSDP mode (requires variable-shape inputs across forward passes)
- Sampling during training loads a separate model for inference
- FSDP + torch.compile may have recompilation overhead on first few steps

#### Memory Comparison: d20 on 4×A10

| Mode | Weights | Optimizer | Gradients | Activations | Total |
|------|---------|-----------|-----------|-------------|-------|
| DDP | 1.7 GB | 3.4 GB | 1.7 GB | ~6 GB | ~13 GB |
| FSDP (full) | 0.4 GB | 0.9 GB | 0.4 GB | ~6 GB | ~8 GB |
| FSDP (shard_grad_op) | 1.7 GB | 0.9 GB | 0.4 GB | ~6 GB | ~9 GB |

With FSDP `full` sharding, you gain ~5GB of headroom per GPU, enabling:
- **d26 model** (~2.1B params) which would OOM with DDP
- **Larger batch sizes** (device-batch-size=16 instead of 8)
- **Longer sequences** (4096 instead of 2048)

#### Option B: Tensor Parallelism (TP) — Not implemented

Split individual weight matrices across GPUs. More complex, requires rewriting the entire model forward pass. Better for inference latency but not needed for training on 4 GPUs.

#### Option C: Pipeline Parallelism (PP) — Not implemented

Split layers across GPUs. Simple conceptually but pipeline bubbles waste ~25% compute. Not recommended for 4 GPUs.

#### Practical Recommendation for 4×A10

| Goal | Recommended Approach |
|------|---------------------|
| Train d20 faster | Current DDP is fine (already works) |
| Train d24 (1.6B) | `--parallelism=fsdp` with `--fsdp-sharding=full` |
| Train d26+ (2.1B+) | `--parallelism=fsdp` + larger batch size |
| Inference only | Simple pipeline parallel (split layers) |
| Production scaling | FSDP + TP hybrid (like Llama training) |

---

## 7. Study Path & Exercises

### Week 1: Understanding the Codebase

**Day 1-2: Model Architecture**
- Read `nanochat/gpt.py` with the architecture diagram above
- Exercise: Trace the forward pass of a single token through d20 model on paper
- Exercise: Calculate how many FLOPs one forward pass takes (use `estimate_flops()`)

**Day 3-4: Optimizer**
- Read `nanochat/optim.py`
- Study: What is Newton-Schulz iteration? Why does orthogonalization help?
- Exercise: Compare AdamW vs Muon on a small matrix. Run:
  ```python
  import torch
  W = torch.randn(128, 256, requires_grad=True)
  for _ in range(100):
      loss = (W ** 2).sum()
      loss.backward()
      # try different optimizers
  ```

**Day 5-7: Training Loop**
- Read `scripts/base_train.py` end-to-end
- Exercise: Run a d4 model on CPU for 20 steps:
  ```bash
  python -m scripts.base_train --depth=4 --max-seq-len=512 \
      --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 \
      --total-batch-size=512 --num-iterations=20
  ```

### Week 2: Hands-On Experiments

**Day 1-2: Scaling Laws**
- Read `runs/scaling_laws.sh`
- Exercise: Train d4, d8, d12 models and plot val_bpb vs params
- Study: Chinchilla scaling laws paper (see references)

**Day 3-4: SFT & RL**
- Run SFT and RL on your d20 model
- Exercise: Add a new task to the SFT mixture (e.g., a custom JSON dataset)
- Exercise: Modify RL temperature and observe GSM8K pass@k changes

**Day 5-7: Architecture Modifications**
- Exercise: Remove value embeddings and compare training curves
- Exercise: Change ReLU² to GELU and observe the difference
- Exercise: Implement gradient checkpointing to reduce memory

### Week 3: Advanced Topics

**Day 1-3: Distributed Training**
- Study the 3-phase async communication in `DistMuonAdamW`
- Read `nanochat/fsdp_utils.py` — understand how FSDP wraps the model
- Exercise: Compare DDP vs FSDP memory usage on d20:
  ```bash
  # DDP mode
  torchrun --nproc_per_node=4 -m scripts.base_train --depth=20 --device-batch-size=8 --num-iterations=5 --run=dummy
  
  # FSDP mode
  torchrun --nproc_per_node=4 -m scripts.base_train --depth=20 --device-batch-size=8 --num-iterations=5 --parallelism=fsdp --run=dummy
  ```
- Exercise: Train d26 model with FSDP (won't fit with DDP on A10!)

**Day 4-5: Inference Optimization**
- Study the KV cache implementation in `engine.py`
- Exercise: Benchmark prefill vs decode latency
- Exercise: Implement speculative decoding

**Day 6-7: Capstone Project**
Pick one:
- Implement FSDP + gradient checkpointing for d32+ on 4×A10
- Add LoRA/QLoRA fine-tuning support
- Implement batched inference with continuous batching
- Add a new evaluation benchmark

---

## 8. Reference Materials

### Foundational Papers

| Paper | Topic | Why Read It |
|-------|-------|-------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer architecture | The original — understand what changed in NanoChat |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan scaling laws | How depth controls everything |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | Compute-optimal training | Why data:param ratio = 8-20 |
| [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) | Rotary embeddings | Relative position encoding |
| [Muon optimizer](https://kellerjordan.github.io/posts/muon/) | Muon | Momentum orthogonalized by Newton-Schulz |
| [Polar Express](https://arxiv.org/abs/2505.16932) | Orthogonalization | The algorithm used in NanoChat's Muon |
| [NorMuon](https://arxiv.org/abs/2510.05491) | Variance reduction | Per-neuron adaptive LR after orthogonalization |
| [Power Lines](https://arxiv.org/abs/2505.13738) | Batch size scaling | Why batch_size ∝ D^0.383 |

### Architecture References

| Topic | Resource |
|-------|----------|
| QK Normalization | [Scaling Vision Transformers to 22B Params](https://arxiv.org/abs/2302.05442) |
| GQA (Group-Query Attention) | [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) |
| ReLU² activation | [Prime](), used in modded-nanogpt |
| Sliding Window Attention | [Mistral 7B](https://arxiv.org/abs/2310.06825) |
| Value Embeddings (ResFormer) | [ResFormer](https://arxiv.org/abs/2310.13452) |
| Softcap logits | [Gemma 2](https://arxiv.org/abs/2408.00118) |
| Flash Attention 3 | [FA3 paper](https://arxiv.org/abs/2407.08608) |

### Training & Infrastructure

| Topic | Resource |
|-------|----------|
| FSDP Tutorial | [PyTorch FSDP Guide](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) |
| ZeRO Stages | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) |
| Tensor Parallelism | [Megatron-LM](https://arxiv.org/abs/1909.08053) |
| Pipeline Parallelism | [GPipe](https://arxiv.org/abs/1811.06965) |
| Mixed Precision Training | [Mixed Precision Paper](https://arxiv.org/abs/1710.03740) |
| torch.compile | [PyTorch 2.0 Blog](https://pytorch.org/get-started/pytorch-2.0/) |

### Video Lectures

| Topic | Resource |
|-------|----------|
| GPT from scratch | [Let's build GPT: by Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| LLM Training | [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) |
| Distributed Training | [DeepLearning.AI MLOps Course](https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops-specialty/) |

### Codebases to Study

| Project | What to Learn |
|---------|---------------|
| [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Original Muon optimizer, speedrun benchmark |
| [llama](https://github.com/meta-llama/llama) | Production-grade training code |
| [litgpt](https://github.com/Lightning-AI/litgpt) | Clean PyTorch Lightning implementation |
| [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch's reference distributed training |

---

## Quick Reference: Your d20 Model Stats

```
Model:           GPT (d20)
Parameters:      896,533,746 (~0.9B)
Layers:          20
Model dim:       1280
Heads:           10 (query), 10 (kv, no GQA)
Head dim:        128
MLP ratio:       4x (1280 → 5120 → 1280)
Vocab size:      32,768
Context length:  2048
Window pattern:  SSSL (3 short + 1 long)
Training tokens: 3,481,272,320 (~3.5B)
Training FLOPs:  ~1.13e+19
Val BPB:         ~0.75 (at end of training)
```
