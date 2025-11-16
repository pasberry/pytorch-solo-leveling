# Lab 1: FSDP Advanced - Training Billion-Parameter Models at Scale 🚀

> **Time:** 3-4 hours
> **Difficulty:** Expert
> **Goal:** Master Fully Sharded Data Parallel (FSDP) to train LLaMA-scale models across multiple GPUs and nodes

---

## 📖 Why This Lab Matters

You've trained models on a single GPU. You've used DataParallel for multi-GPU training. But when you try to load a 7B parameter model, you hit the **memory wall**:

```
RuntimeError: CUDA out of memory.
Tried to allocate 26.25 GiB (GPU 0 has 40.00 GiB total)
```

**The reality of modern ML:**
- **GPT-3:** 175 billion parameters (~700GB in FP32)
- **LLaMA-70B:** 70 billion parameters (~280GB in FP32)
- **LLaMA-7B:** 7 billion parameters (~28GB in FP32)

A single A100 GPU has only 40-80GB memory. **You physically cannot fit these models.**

This lab teaches you **FSDP (Fully Sharded Data Parallel)** - the technology that Meta used to train LLaMA, OPT, and FAIR's cutting-edge models. It's the same approach that powers:
- **Meta's LLaMA training** (65B, 70B models)
- **OpenAI's GPT-3/4** (distributed training infrastructure)
- **DeepMind's Chinchilla** (70B parameters)
- **Anthropic's Claude** (constitutional AI at scale)

**Master FSDP, and you can train models that were impossible just years ago.**

---

## 🧠 The Big Picture: Why Single-GPU Training Fails

### The Memory Wall

Training a neural network requires memory for:

1. **Model parameters** (weights, biases)
2. **Gradients** (same size as parameters)
3. **Optimizer states** (Adam: 2x parameters for momentum + variance)
4. **Activations** (forward pass intermediate values)

**For a 7B parameter LLaMA model:**

```
Parameters:         7B × 4 bytes (FP32) = 28 GB
Gradients:          7B × 4 bytes        = 28 GB
Optimizer (Adam):   7B × 8 bytes        = 56 GB
Activations:        ~10-20 GB (depends on batch size)
─────────────────────────────────────────────────
TOTAL:              ~122-132 GB
```

**You need 132GB for a single GPU, but A100 only has 80GB!**

### The Evolution of Distributed Training

**Stage 1: Data Parallelism (2012-2018)**
```
GPU 0: [Full Model Copy] → processes batch 0
GPU 1: [Full Model Copy] → processes batch 1
GPU 2: [Full Model Copy] → processes batch 2
GPU 3: [Full Model Copy] → processes batch 3
         ↓ Sync gradients (AllReduce)
     Update all copies
```

**Problem:** Each GPU stores the ENTIRE model. Doesn't solve memory problem.

**Stage 2: Model Parallelism (2018-2020)**
```
GPU 0: [Layer 0-8]   → Sequential pipeline
GPU 1: [Layer 9-16]  → Pass activations forward
GPU 2: [Layer 17-24] → Pass gradients backward
GPU 3: [Layer 25-32]
```

**Problem:** Sequential execution. GPU utilization low (~25%). Communication overhead.

**Stage 3: ZeRO and FSDP (2020-Present)**
```
GPU 0: [Shard 0 of parameters/gradients/optimizer]
GPU 1: [Shard 1 of parameters/gradients/optimizer]
GPU 2: [Shard 2 of parameters/gradients/optimizer]
GPU 3: [Shard 3 of parameters/gradients/optimizer]
         ↓ Gather when needed, discard after use
     8x memory reduction + data parallelism efficiency!
```

**Breakthrough:** Split memory, keep parallelism!

---

## 🔬 Deep Dive: FSDP and ZeRO Optimization

### ZeRO: Zero Redundancy Optimizer

Developed by Microsoft (DeepSpeed), ZeRO has 3 stages:

**ZeRO-1: Shard Optimizer States**
```
Before (8 GPUs):
  Each GPU: Parameters (7B × 4B) + Gradients (7B × 4B) + Adam states (7B × 8B)
  Total per GPU: 112 GB

After ZeRO-1:
  Each GPU: Parameters (7B × 4B) + Gradients (7B × 4B) + Adam shard (7B/8 × 8B)
  Total per GPU: 63 GB
  Memory reduction: 44%
```

**ZeRO-2: Shard Gradients**
```
Each GPU: Parameters (7B × 4B) + Gradient shard (7B/8 × 4B) + Adam shard (7B/8 × 8B)
Total per GPU: 38.5 GB
Memory reduction: 66%
```

**ZeRO-3: Shard Parameters**
```
Each GPU: Parameter shard (7B/8 × 4B) + Gradient shard (7B/8 × 4B) + Adam shard (7B/8 × 8B)
Total per GPU: 14 GB
Memory reduction: 87.5%!
```

**The key insight:** Each GPU only keeps 1/N of everything, gathers on-demand.

### FSDP Architecture

PyTorch's FSDP is inspired by ZeRO-3 but with PyTorch-native implementation:

```python
# Conceptual FSDP workflow
class FSDP_Layer:
    def __init__(self, module, world_size):
        # Shard parameters across GPUs
        self.param_shard = shard_params(module.parameters(), world_size)

    def forward(self, x):
        # 1. Gather full parameters from all GPUs
        full_params = all_gather(self.param_shard)

        # 2. Run forward pass
        output = compute(x, full_params)

        # 3. Discard full parameters (save memory!)
        del full_params

        return output

    def backward(self, grad):
        # 1. Gather parameters again
        full_params = all_gather(self.param_shard)

        # 2. Compute gradients
        param_grads = compute_backward(grad, full_params)

        # 3. Reduce-scatter gradients (each GPU keeps its shard)
        self.grad_shard = reduce_scatter(param_grads)

        # 4. Discard full parameters
        del full_params
```

**Memory flow per layer:**
```
Idle:         14 GB (parameter shard)
Forward:      28 GB (gathered parameters) → then freed
Backward:     28 GB (gathered parameters) → then freed
Optimizer:    14 GB (gradient shard + optimizer shard)
```

**Peak memory per layer: 28 GB (during gather), not 112 GB!**

---

## 🎯 FSDP Sharding Strategies

### 1. FULL_SHARD (ZeRO-3)

**Configuration:**
```python
from torch.distributed.fsdp import ShardingStrategy

fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

**Behavior:**
- Shards parameters, gradients, AND optimizer states
- Maximum memory savings (87.5% reduction)
- All-gather on forward and backward
- **Use for:** Largest models (7B+ parameters)

**Communication pattern:**
```
Forward:  all_gather(params) → compute → free params
Backward: all_gather(params) → compute gradients → reduce_scatter(grads) → free params
```

**Tradeoff:**
- ✅ Minimal memory (fits massive models)
- ❌ More communication (2x all-gather per layer)

### 2. SHARD_GRAD_OP (ZeRO-2)

**Configuration:**
```python
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
)
```

**Behavior:**
- Shards gradients and optimizer states only
- Parameters replicated on all GPUs
- **Use for:** Medium models (1B-7B) when you have memory

**Tradeoff:**
- ✅ Less communication (no parameter gathering)
- ❌ More memory (parameters replicated)

### 3. NO_SHARD (DDP)

**Configuration:**
```python
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.NO_SHARD
)
```

**Behavior:**
- Equivalent to DistributedDataParallel
- Everything replicated
- **Use for:** Small models (<1B) or baseline comparison

**Tradeoff:**
- ✅ Minimal communication
- ❌ Maximum memory

---

## 🔧 FSDP Advanced Features

### 1. Mixed Precision Training

**Problem:** FP32 training uses 4 bytes per parameter. FP16 uses 2 bytes.

**FSDP Mixed Precision:**
```python
from torch.distributed.fsdp import MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # Parameters stored as BF16
    reduce_dtype=torch.float32,       # Gradient reduction in FP32 (accuracy)
    buffer_dtype=torch.bfloat16       # Buffers (BatchNorm, etc) in BF16
)

fsdp_model = FSDP(
    model,
    mixed_precision=mp_policy
)
```

**Memory savings:**
```
FP32: 7B × 4 bytes = 28 GB
BF16: 7B × 2 bytes = 14 GB
Memory reduction: 50%!
```

**Why BF16 > FP16:**
- BF16 has same exponent range as FP32 (avoids overflow/underflow)
- FP16 can hit numerical issues with large models
- BF16 is the standard for LLaMA, GPT-3, PaLM

### 2. CPU Offloading

**Idea:** Store parameters on CPU RAM (TBs available), move to GPU only when needed.

```python
from torch.distributed.fsdp import CPUOffload

fsdp_model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
```

**Memory tradeoff:**
- GPU memory: 14 GB → 2-3 GB (just activations!)
- CPU memory: 28 GB (parameter shards on CPU)
- Latency: +20-30% (PCIe transfer overhead)

**Use case:** Training 70B models on 8x A100 (40GB) instead of A100 (80GB)

**Production example (Meta LLaMA-65B):**
```python
# Without CPU offload: Requires 8x A100 80GB = $$$$$
# With CPU offload: Runs on 8x A100 40GB = $$$ (50% cost savings!)
```

### 3. Activation Checkpointing

**Problem:** Activations grow with batch size and sequence length.

**For LLaMA-7B with sequence length 2048, batch size 4:**
```
Activations per layer: ~500 MB
32 layers: 16 GB!
```

**Solution:** Recompute activations during backward instead of storing.

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

# Apply to transformer blocks
def check_fn(module):
    return isinstance(module, TransformerBlock)

apply_activation_checkpointing(
    fsdp_model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn
)
```

**Tradeoff:**
- Memory: 16 GB → 2 GB (8x reduction!)
- Compute: +33% (one extra forward pass per backward)
- Overall: Worth it for large models

---

## 🌍 Multi-Node Training

### Single-Node vs Multi-Node

**Single-node (8 GPUs):**
```
Server 0: [GPU 0, GPU 1, GPU 2, GPU 3, GPU 4, GPU 5, GPU 6, GPU 7]
          └─ NVLink: 600 GB/s bandwidth
```

**Multi-node (4 servers × 8 GPUs = 32 GPUs):**
```
Server 0: [GPU 0-7]  ─┐
Server 1: [GPU 8-15]  ├─ InfiniBand: 100-200 GB/s
Server 2: [GPU 16-23] │  (10x slower than NVLink!)
Server 3: [GPU 24-31] ─┘
```

**Challenge:** Network is the bottleneck.

### FSDP Multi-Node Setup

```python
import torch.distributed as dist

# Initialize distributed backend
dist.init_process_group(
    backend='nccl',                    # NVIDIA Collective Communications Library
    init_method='env://',              # Read from environment variables
    world_size=32,                     # Total GPUs
    rank=int(os.environ['RANK'])       # This GPU's global rank
)

# Wrap model with FSDP
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device()
)
```

**Launch script (SLURM):**
```bash
#!/bin/bash
#SBATCH --nodes=4                  # 4 servers
#SBATCH --gpus-per-node=8          # 8 GPUs per server
#SBATCH --ntasks-per-node=8        # 8 processes per node

# Launch distributed training
srun python train_fsdp.py \
    --model llama-7b \
    --batch-size 4 \
    --gradient-accumulation 4
```

**Environment variables:**
```bash
MASTER_ADDR=server0.cluster.com    # Coordinator node
MASTER_PORT=29500                   # Communication port
WORLD_SIZE=32                       # Total processes (4 nodes × 8 GPUs)
RANK=0-31                           # Unique ID per process
LOCAL_RANK=0-7                      # GPU ID within node
```

---

## 📊 Mathematical Foundations

### Memory Calculation Mathematics

**Total training memory for a model with P parameters:**

```
Memory_total = Memory_params + Memory_grads + Memory_optimizer + Memory_activations

Where:
  Memory_params     = P × bytes_per_param
  Memory_grads      = P × bytes_per_param  (same as parameters)
  Memory_optimizer  = P × optimizer_factor
  Memory_activations= B × L × H × S²      (batch × layers × hidden × sequence²)

For Adam optimizer:
  optimizer_factor = 8 bytes (2× for momentum, 2× for variance)

For FP32:
  bytes_per_param = 4 bytes

For BF16:
  bytes_per_param = 2 bytes
```

**Example: LLaMA-7B in FP32 with Adam**
```
P = 7 billion parameters
Memory_params     = 7B × 4B = 28 GB
Memory_grads      = 7B × 4B = 28 GB
Memory_optimizer  = 7B × 8B = 56 GB (Adam: momentum + variance)
Memory_activations≈ 10-20 GB (depends on batch size and sequence length)
────────────────────────────────────
Total            ≈ 122-132 GB
```

**With FULL_SHARD across N GPUs:**
```
Memory_per_GPU = (Memory_params + Memory_grads + Memory_optimizer) / N + Memory_activations

For 8 GPUs:
  Memory_per_GPU = (28 + 28 + 56) / 8 + 15 = 14 + 15 = 29 GB

Reduction: 132 GB → 29 GB (78% savings!)
```

**With BF16 mixed precision:**
```
Memory_params     = 7B × 2B = 14 GB  (BF16)
Memory_grads      = 7B × 2B = 14 GB  (BF16)
Memory_optimizer  = 7B × 8B = 56 GB  (still FP32 for stability)
Total            = 84 GB

With FULL_SHARD (8 GPUs):
  Memory_per_GPU = 84 / 8 + 15 = 10.5 + 15 = 25.5 GB
```

### Communication Complexity Analysis

**FULL_SHARD communication per iteration:**

```
Forward pass (per layer):
  Operation: all_gather(param_shard)
  Data transferred: P_layer × bytes_per_param × (N-1)/N

  For 8 GPUs, BF16:
    Data = P_layer × 2 × 7/8

Backward pass (per layer):
  Operation 1: all_gather(param_shard)
  Operation 2: reduce_scatter(grad_shard)
  Data transferred: 2 × P_layer × bytes_per_param × (N-1)/N

Total per layer: 3 × P_layer × bytes_per_param × (N-1)/N
```

**Example: LLaMA-7B with 32 layers**
```
Per layer: 7B/32 ≈ 220M parameters
Forward:   220M × 2 bytes × 7/8 = 385 MB per GPU
Backward:  2 × 385 MB = 770 MB per GPU
Total per iteration: 32 layers × 1.155 GB = 37 GB per GPU

With NVLink (600 GB/s): 37 GB / 600 GB/s = 62 ms communication overhead
With InfiniBand (100 GB/s): 37 GB / 100 GB/s = 370 ms communication overhead
```

**Scaling Law:**
```
Communication_time ∝ (Model_size × bytes_per_param × 3) / Bandwidth

Key insight: Communication is linear in model size, independent of #GPUs!
```

### FSDP Performance Analysis

**Communication Overhead Breakdown:**

```
Forward:  all_gather(params) → N × param_size
Backward: all_gather(params) + reduce_scatter(grads) → 2N × param_size

For LLaMA-7B with 32 layers:
  Per layer: 7B/32 ≈ 220M parameters
  Forward:   220M × 2 bytes (BF16) × 32 GPUs = 14 GB transferred
  Backward:  28 GB transferred
  Total:     42 GB per iteration per GPU
```

**With NVLink (600 GB/s):** 42 GB / 600 GB/s = **70ms overhead**

**With InfiniBand (100 GB/s):** 42 GB / 100 GB/s = **420ms overhead** (6x slower!)

**Optimization strategies:**
1. **Increase batch size** → Amortize communication over more compute
2. **Gradient accumulation** → Reduce communication frequency
3. **Mixed precision** → 2x less data transferred (FP16/BF16)
4. **Better interconnect** → NVLink switches, InfiniBand HDR (200-400 GB/s)

### Memory vs Speed Tradeoff

| Strategy | Memory per GPU | Communication | Speed | Use Case |
|----------|---------------|---------------|-------|----------|
| NO_SHARD (DDP) | 112 GB | Minimal | 1.0x | Small models |
| SHARD_GRAD_OP | 63 GB | Moderate | 0.9x | Medium models |
| FULL_SHARD | 14 GB | High | 0.7-0.8x | Large models |
| FULL_SHARD + CPU offload | 3 GB | Highest | 0.5-0.6x | Massive models |

**Rule of thumb:**
- If model fits in memory → Use DDP
- If model barely fits → Use SHARD_GRAD_OP
- If model doesn't fit → Use FULL_SHARD
- If still doesn't fit → Add CPU offload

### Activation Checkpointing Mathematics

**Without checkpointing:**
```
Forward: Store all L layer activations
Memory = L × activation_size_per_layer

For transformer with L=32, hidden=4096, seq_len=2048, batch=4:
  Per layer ≈ 4 × 4096 × 2048 × 4 bytes = 512 MB
  Total = 32 × 512 MB = 16 GB
```

**With full checkpointing:**
```
Forward: Store only checkpointed layers (every Nth layer)
Backward: Recompute intermediate activations on-the-fly

Memory = (L/N) × activation_size_per_layer
Compute = 1.33× (one extra forward pass)

With N=1 (checkpoint every layer):
  Memory = 32/32 × 512 MB ≈ 512 MB (97% reduction!)
  Compute = 1.33× original
```

**Selective checkpointing (optimal):**
```
Checkpoint expensive layers (attention) only
Memory reduction: ~80%
Compute overhead: ~15%

Best tradeoff for production!
```

---

## 🏭 Production: How Meta Trains LLaMA

### LLaMA-70B Training Configuration

**Hardware:**
- 2048 A100 80GB GPUs (256 nodes × 8 GPUs)
- NVLink + InfiniBand HDR (200 GB/s)
- ~$5M in hardware

**FSDP Configuration:**
```python
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16
    ),
    "cpu_offload": None,  # No CPU offload (80GB is enough with FSDP)
    "activation_checkpointing": True,
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,  # Overlap comm & compute
}
```

**Training stats:**
```
Model size:        70B parameters (140 GB in BF16)
Per-GPU memory:    ~75 GB (with FULL_SHARD + activation checkpointing)
Batch size:        4M tokens (2048 GPUs × 2048 tokens/GPU)
Training time:     ~21 days for 1.4T tokens
Cost:              ~$3-4M (GPU hours + power)
```

**Key optimizations:**
1. **Backward prefetch:** Start all-gathering next layer's parameters during current layer's backward
2. **Mixed precision:** BF16 for compute, FP32 for gradient reduction
3. **Selective activation checkpointing:** Only checkpoint expensive attention layers
4. **Gradient accumulation:** Accumulate 4-8 steps to reduce communication
5. **Flash Attention:** Custom CUDA kernel for 3x faster attention

---

## 🎯 Learning Objectives

By the end of this lab, you will:

**Theory:**
- [ ] Understand why single-GPU training fails for large models
- [ ] Explain ZeRO stages 1, 2, 3 and their memory tradeoffs
- [ ] Compare FSDP sharding strategies (FULL_SHARD vs SHARD_GRAD_OP vs NO_SHARD)
- [ ] Analyze communication overhead and bandwidth requirements
- [ ] Calculate memory requirements for billion-parameter models
- [ ] Design multi-node training configurations

**Implementation:**
- [ ] Wrap models with FSDP using different sharding strategies
- [ ] Apply mixed precision training (BF16) for memory efficiency
- [ ] Implement activation checkpointing to reduce peak memory
- [ ] Configure CPU offloading for extremely large models
- [ ] Set up multi-GPU and multi-node distributed training
- [ ] Benchmark memory usage and training throughput
- [ ] Profile communication overhead and identify bottlenecks

**Production Skills:**
- [ ] Train LLaMA-scale models (7B parameters) on multi-GPU setups
- [ ] Optimize for minimal peak memory usage
- [ ] Tune for maximum throughput (tokens/second)
- [ ] Debug FSDP-related OOM errors and hangs
- [ ] Monitor GPU utilization and communication efficiency

---

## 🔑 Key Concepts

### 1. Model Sharding

**Definition:** Splitting model parameters across multiple GPUs, gathering on-demand.

**Example:**
```python
# Without FSDP: 7B parameters × 4 bytes = 28 GB per GPU
model = LLaMA7B()  # OOM on A100 40GB!

# With FSDP: 7B/8 parameters × 4 bytes = 3.5 GB per GPU
fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

### 2. All-Gather and Reduce-Scatter

**All-Gather:** Each GPU gathers shards from all other GPUs.
```
Before:
  GPU 0: [shard_0]
  GPU 1: [shard_1]
  GPU 2: [shard_2]

After all-gather:
  GPU 0: [shard_0, shard_1, shard_2]
  GPU 1: [shard_0, shard_1, shard_2]
  GPU 2: [shard_0, shard_1, shard_2]
```

**Reduce-Scatter:** Sum gradients and distribute shards.
```
Before (gradients):
  GPU 0: [grad_0, grad_1, grad_2]
  GPU 1: [grad_0, grad_1, grad_2]
  GPU 2: [grad_0, grad_1, grad_2]

After reduce-scatter:
  GPU 0: [sum(grad_0)]
  GPU 1: [sum(grad_1)]
  GPU 2: [sum(grad_2)]
```

### 3. Gradient Accumulation

**Problem:** Communication overhead dominates with small batch sizes.

**Solution:** Accumulate gradients locally, sync less frequently.

```python
optimizer.zero_grad()
for micro_batch in range(gradient_accumulation_steps):
    loss = model(data[micro_batch])
    loss.backward()  # Accumulate gradients (no sync)

optimizer.step()  # Single communication here!
```

**Effect:**
- Effective batch size = micro_batch × accumulation_steps
- Communication: every N steps instead of every step
- Tradeoff: Slightly stale gradients (usually negligible)

### 4. Mixed Precision Training

**FP32 (32-bit floating point):**
```
Sign: 1 bit, Exponent: 8 bits, Mantissa: 23 bits
Range: ±1.4 × 10^-45 to ±3.4 × 10^38
Precision: ~7 decimal digits
```

**FP16 (16-bit floating point):**
```
Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
Range: ±6 × 10^-8 to ±6.5 × 10^4  ← Small range!
Precision: ~3 decimal digits
```

**BF16 (Brain Float 16):**
```
Sign: 1 bit, Exponent: 8 bits (same as FP32!), Mantissa: 7 bits
Range: ±1.4 × 10^-45 to ±3.4 × 10^38  ← Same as FP32!
Precision: ~2 decimal digits
```

**Why BF16 is best for deep learning:**
- Same range as FP32 → No overflow/underflow
- Half the memory (2 bytes vs 4 bytes)
- Hardware support on A100/H100 (TF32 Tensor Cores)
- Used by GPT-3, LLaMA, PaLM, Chinchilla

---

## 💻 Exercises

### Exercise 1: Train LLaMA-7B with FSDP (60 mins)

**What You'll Learn:**
- Setting up distributed training with FSDP
- Configuring FULL_SHARD for maximum memory savings
- Monitoring memory usage and GPU utilization
- Debugging distributed training issues

**Why It Matters:**
This is the fundamental skill for training modern LLMs. Every large model (GPT, LLaMA, Falcon) uses FSDP or equivalent sharding. Without this, you're limited to <1B parameter models. With FSDP, you can train:
- 7B models on 8x A100 40GB
- 70B models on 64x A100 80GB
- 175B+ models on hundreds of GPUs

**Starter code:**
```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

# Initialize distributed
torch.distributed.init_process_group(backend='nccl')

# Load model (7B parameters)
model = LLaMA7B()

# Wrap with FSDP
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)

# Training loop
for batch in dataloader:
    loss = fsdp_model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Expected results:**
- Memory usage: <20 GB per GPU (vs 132 GB without FSDP)
- Training completes without OOM
- Model trains correctly (loss decreases)
- GPU utilization >85%

### Exercise 2: Compare Sharding Strategies (45 mins)

**What You'll Learn:**
- Performance characteristics of each sharding strategy
- Memory-speed tradeoffs in distributed training
- How to profile and benchmark GPU training
- When to choose which strategy

**Why It Matters:**
Choosing the wrong sharding strategy wastes resources or causes OOM:
- NO_SHARD on 70B model → OOM crash
- FULL_SHARD on 1B model → Unnecessary communication overhead
- Understanding tradeoffs saves millions in compute costs at scale

**Metrics to measure:**
1. Peak memory usage (GB)
2. Throughput (samples/second)
3. Communication overhead (ms/iteration)

**Implementation:**
```python
strategies = [
    ShardingStrategy.NO_SHARD,
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.FULL_SHARD
]

for strategy in strategies:
    fsdp_model = FSDP(model, sharding_strategy=strategy)

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    train_one_epoch(fsdp_model)
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"{strategy}: {peak_memory:.2f} GB")
```

**Expected results:**
| Strategy | Memory | Speed | Best For |
|----------|--------|-------|----------|
| NO_SHARD | 112 GB | 100% | Small models |
| SHARD_GRAD_OP | 63 GB | 90% | Medium models |
| FULL_SHARD | 14 GB | 75% | Large models |

### Exercise 3: Activation Checkpointing (45 mins)

**What You'll Learn:**
- How activation checkpointing trades compute for memory
- Implementing selective checkpointing for transformers
- Measuring memory savings and compute overhead
- Optimal checkpointing strategies

**Why It Matters:**
Activations often consume more memory than parameters! For long sequences:
- LLaMA-7B with 4k context: ~30 GB activation memory
- Without checkpointing: OOM on most GPUs
- With checkpointing: Fits comfortably
This technique is used by every LLM trainer (OpenAI, Anthropic, Meta).

**Task:** Reduce activation memory by 8x using selective checkpointing.

**Implementation:**
```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing
)

# Checkpoint only transformer blocks
def check_fn(module):
    return isinstance(module, TransformerBlock)

apply_activation_checkpointing(
    fsdp_model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn
)
```

**Measure:**
- Memory before: ~30 GB
- Memory after: ~10 GB
- Slowdown: ~10-15%

### Exercise 4: Multi-Node Training (60 mins)

**What You'll Learn:**
- Setting up multi-node distributed training
- Configuring network communication (NCCL, InfiniBand)
- Debugging hangs and synchronization issues
- Monitoring distributed training metrics

**Why It Matters:**
Training frontier models requires hundreds or thousands of GPUs:
- GPT-3: 10,000+ GPUs across multiple nodes
- LLaMA-2: 2,048 GPUs (256 nodes)
- Can't fit on single node → Must master multi-node training
This is the path from research to production-scale AI.

**Task:** Train on 2 nodes × 8 GPUs (16 GPUs total).

**Launch script:**
```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_fsdp.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_fsdp.py
```

**Verify:**
- All 16 GPUs participate in training
- Gradients synchronized correctly
- Training throughput scales near-linearly (14-15x vs 1 GPU)

### Exercise 5: Production LLaMA Training (90 mins)

**What You'll Learn:**
- Full production training pipeline
- Combining all FSDP optimizations
- Monitoring and checkpointing at scale
- Achieving production-grade throughput and efficiency

**Why It Matters:**
This exercise simulates real production LLM training. You'll implement the same techniques Meta used for LLaMA:
- Mixed precision training for 2x memory savings
- Activation checkpointing for long sequences
- Gradient accumulation for stability
- Optimal hyperparameters for convergence
Master this, and you can train state-of-the-art models.

**Task:** Replicate Meta's LLaMA training setup (scaled down).

**Configuration:**
- Model: LLaMA-7B (7B parameters)
- Hardware: 8x A100 40GB
- Data: C4 dataset (1B tokens)
- Target: Train for 10K steps, achieve <2.5 perplexity

**Optimizations to apply:**
1. FULL_SHARD with BF16 mixed precision
2. Activation checkpointing on transformer blocks
3. Flash Attention for memory efficiency
4. Gradient accumulation (4 steps)
5. AdamW optimizer with weight decay
6. Cosine learning rate schedule

**Starter code:**
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch
)

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16
)

fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device()
)

# Apply activation checkpointing
apply_activation_checkpointing(fsdp_model, check_fn=lambda m: isinstance(m, TransformerBlock))

# Optimizer
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=3e-4, weight_decay=0.1)

# Training loop with gradient accumulation
for step, batch in enumerate(dataloader):
    loss = fsdp_model(batch) / gradient_accumulation_steps
    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Success criteria:**
- Peak memory <35 GB per GPU
- Throughput >10K tokens/second (total)
- Perplexity <2.5 on validation set
- Training completes in <24 hours

---

## ⚠️ Common Pitfalls at Scale

### 1. OOM During Backward Pass

**Symptom:**
```
RuntimeError: CUDA out of memory during backward
```

**Cause:** Activation memory accumulates during forward pass.

**Solution:**
```python
# Apply activation checkpointing
apply_activation_checkpointing(fsdp_model, check_fn=check_fn)
```

### 2. Slow Multi-Node Training

**Symptom:** 32 GPUs only 5x faster than 8 GPUs (expected: 4x).

**Cause:** Network bandwidth bottleneck (InfiniBand saturated).

**Solutions:**
1. Increase gradient accumulation steps (reduce communication frequency)
2. Use larger batch sizes (amortize communication)
3. Enable `backward_prefetch=BACKWARD_PRE` (overlap communication)
4. Upgrade to InfiniBand HDR (200 GB/s) or switch fabric

### 3. Divergent Training Across Nodes

**Symptom:** Loss diverges, NaN gradients on some nodes.

**Cause:** Random seeds not synchronized, data loading inconsistent.

**Solution:**
```python
# Set deterministic seed per rank
seed = base_seed + rank
torch.manual_seed(seed)
np.random.seed(seed)

# Use DistributedSampler for data loading
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler)
```

### 4. Process Hangs on Initialization

**Symptom:** Training script hangs at `init_process_group()`.

**Cause:** Firewall blocking communication, wrong MASTER_ADDR/PORT.

**Debug:**
```bash
# Test connectivity
nc -zv $MASTER_ADDR $MASTER_PORT

# Check environment variables
echo $MASTER_ADDR $MASTER_PORT $WORLD_SIZE $RANK

# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
```

### 5. Uneven GPU Utilization

**Symptom:** GPU 0 at 95%, GPU 7 at 60%.

**Cause:** Imbalanced sharding, pipeline bubbles, data loading bottleneck.

**Solutions:**
1. Use `ShardingStrategy.FULL_SHARD` (balanced sharding)
2. Increase `num_workers` in DataLoader (faster data loading)
3. Enable `pin_memory=True` for faster CPU-GPU transfer
4. Profile with `torch.profiler` to find bottlenecks

---

## 🏆 Expert Checklist for Mastery

**Foundations:**
- [ ] Can explain why DDP doesn't solve memory for large models
- [ ] Understand ZeRO-1, ZeRO-2, ZeRO-3 memory partitioning
- [ ] Know when to use FULL_SHARD vs SHARD_GRAD_OP vs NO_SHARD
- [ ] Can calculate memory requirements for billion-parameter models

**Implementation:**
- [ ] Trained a 7B parameter model using FSDP
- [ ] Applied mixed precision (BF16) for 2x memory savings
- [ ] Implemented activation checkpointing for 8x memory reduction
- [ ] Configured multi-node training (2+ nodes)
- [ ] Benchmarked memory and throughput across configurations

**Optimization:**
- [ ] Tuned gradient accumulation for optimal throughput
- [ ] Enabled backward prefetching to overlap communication
- [ ] Applied Flash Attention or custom CUDA kernels
- [ ] Optimized data loading to saturate GPUs
- [ ] Achieved >80% GPU utilization on multi-node setup

**Production:**
- [ ] Debugged OOM errors using memory profiling
- [ ] Resolved multi-node hangs (network, firewall, NCCL)
- [ ] Monitored training with W&B or TensorBoard
- [ ] Implemented checkpointing and resumable training
- [ ] Trained a LLaMA-scale model from scratch

**Advanced:**
- [ ] Understand FSDP vs DeepSpeed ZeRO tradeoffs
- [ ] Can design training infrastructure for 100B+ parameter models
- [ ] Know cost-performance tradeoffs (A100 40GB vs 80GB vs H100)
- [ ] Familiar with latest optimizations (Flash Attention 2, FSDP2)
- [ ] Can estimate training time and cost for large-scale runs

---

## 🚀 Next Steps

**After mastering FSDP, you can:**

1. **Train LLaMA from Scratch**
   - Use Hugging Face Transformers + FSDP
   - Train on C4 or RedPajama dataset
   - Achieve state-of-the-art perplexity

2. **Explore DeepSpeed ZeRO**
   - Compare PyTorch FSDP vs Microsoft DeepSpeed
   - Try ZeRO-Infinity (NVMe offloading)
   - Experiment with ZeRO++

3. **Optimize for Production**
   - Add checkpointing and fault tolerance
   - Implement dynamic loss scaling (AMP)
   - Monitor with NCCL flight recorder

4. **Scale to 100B+ Parameters**
   - Try tensor parallelism (Megatron-LM)
   - Combine FSDP + tensor parallelism + pipeline parallelism
   - Train on 100+ nodes

---

## 📚 References

**Papers:**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Microsoft, 2020)](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (Meta, 2023)](https://arxiv.org/abs/2304.11277)
- [LLaMA: Open and Efficient Foundation Language Models (Meta, 2023)](https://arxiv.org/abs/2302.13971)

**Documentation:**
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [FSDP API Reference](https://pytorch.org/docs/stable/fsdp.html)
- [Meta's LLaMA Training Code](https://github.com/facebookresearch/llama)

**Production Examples:**
- Meta LLaMA: 70B parameters on 2048 A100s
- OpenAI GPT-3: 175B parameters on 10,000+ V100s
- DeepMind Chinchilla: 70B parameters with optimal compute scaling

---

## 🎯 Solution

Complete implementation: `solution/fsdp_training.py`

**What you'll build:**
- FSDP trainer for LLaMA-7B
- Support for all sharding strategies
- Mixed precision (BF16) and activation checkpointing
- Multi-node distributed training
- Memory and throughput benchmarking
- Production-ready checkpointing and logging

**Next: Lab 2 - Vector Databases for billion-scale semantic search!**
