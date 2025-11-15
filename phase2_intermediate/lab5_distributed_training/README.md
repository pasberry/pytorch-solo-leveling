# Lab 5: Distributed Training - Scale to Multiple GPUs 🚀

> **Time:** 4-5 hours
> **Difficulty:** Advanced
> **Goal:** Master distributed training for large-scale deep learning

---

## 📖 Why This Lab Matters

Modern AI requires distributed training across multiple GPUs or even multiple machines:

- **GPT-3** (OpenAI) - Trained on 10,000+ GPUs (96 days)
- **LLAMA-2** (Meta) - Trained on 2000 GPUs (several weeks)
- **Stable Diffusion** - Multi-GPU training for diffusion models
- **BERT** (Google) - Trained on 64 TPUs (4 days)
- **ImageNet training** - Hours on single GPU → minutes on multiple GPUs

**Why distributed training:**
- **Model too large** - Doesn't fit on single GPU (LLAMA-70B: 140GB)
- **Training too slow** - Months on single GPU → days on cluster
- **Data too large** - Process more data simultaneously
- **Faster iteration** - Rapid experimentation

**This lab teaches you to scale from 1 GPU to hundreds.**

---

## 🧠 The Big Picture: Why Single GPU Isn't Enough

### The Problem: Scale Wall

**Modern AI hits three walls:**

```
Model Size Wall:
  GPT-3 (175B params): 700GB in FP32
  Single A100 GPU: 80GB memory
  Problem: Model doesn't fit!

Training Time Wall:
  ImageNet on single GPU: 30 days
  Business need: Results in hours
  Problem: Too slow!

Data Size Wall:
  LLAMA-2 training data: 2 trillion tokens
  Single GPU throughput: 100K tokens/sec
  Time: 230+ days!
  Problem: Impractical!
```

### The Solution: Distributed Training

**Three strategies:**

1. **Data Parallelism (DP/DDP):** Same model, different data on each GPU
2. **Model Parallelism:** Different parts of model on different GPUs
3. **Pipeline Parallelism:** Different layers on different GPUs

---

## 🔬 Deep Dive: Distributed Training Strategies

### 1. Data Parallelism (DDP): The Standard Approach

**Idea:** Replicate model on each GPU, split data across GPUs.

```
GPU 0: Model Copy + Batch 0-31
GPU 1: Model Copy + Batch 32-63
GPU 2: Model Copy + Batch 64-95
GPU 3: Model Copy + Batch 96-127

Forward → Compute Loss → Backward → AllReduce Gradients → Update
```

**Mathematics:**

```
Single GPU:
  gradient = ∇L(batch)
  θ ← θ - α × gradient

Data Parallel (N GPUs):
  gradient_i = ∇L(batch_i)  for GPU i
  gradient_avg = (1/N) Σᵢ gradient_i  ← AllReduce!
  θ ← θ - α × gradient_avg

Effective batch size: N × batch_per_gpu
```

**AllReduce Operation:**

```
GPU 0: [g₀₀, g₀₁, g₀₂, g₀₃]
GPU 1: [g₁₀, g₁₁, g₁₂, g₁₃]
GPU 2: [g₂₀, g₂₁, g₂₂, g₂₃]
GPU 3: [g₃₀, g₃₁, g₃₂, g₃₃]
   ↓ AllReduce (average)
GPU 0: [ḡ₀, ḡ₁, ḡ₂, ḡ₃]  ← All GPUs have same averaged gradient
GPU 1: [ḡ₀, ḡ₁, ḡ₂, ḡ₃]
GPU 2: [ḡ₀, ḡ₁, ḡ₂, ḡ₃]
GPU 3: [ḡ₀, ḡ₁, ḡ₂, ḡ₃]

Where: ḡⱼ = (g₀ⱼ + g₁ⱼ + g₂ⱼ + g₃ⱼ) / 4
```

**Implementation (PyTorch DDP):**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            loss = model(batch)

            # Backward pass (gradients automatically synchronized!)
            loss.backward()

            # Update (all GPUs update with same gradients)
            optimizer.step()
            optimizer.zero_grad()

    cleanup()

# Launch
torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

**Benefits:**
- Linear speedup (4 GPUs → 4x faster*)
- Easy to implement
- Works for most models

**Limitations:**
- Model must fit on single GPU
- Communication overhead for large models
- Synchronization bottleneck

---

### 2. DistributedDataParallel (DDP) vs DataParallel (DP)

**DataParallel (DP) - Don't use!**

```python
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

**Problems:**
- Single-process (Python GIL bottleneck)
- Uneven GPU memory (GPU 0 is master)
- Gradient gather inefficient
- 50-70% GPU utilization typical

**DistributedDataParallel (DDP) - Use this!**

```python
model = DDP(model, device_ids=[rank])
```

**Advantages:**
- Multi-process (no GIL)
- Balanced GPU memory
- Efficient AllReduce (ring algorithm)
- 90-95% GPU utilization

**Performance comparison:**

| GPUs | DP Speed | DDP Speed | DDP Speedup |
|------|----------|-----------|-------------|
| 1    | 100 img/s | 100 img/s | 1.0x |
| 2    | 150 img/s | 190 img/s | 1.3x |
| 4    | 250 img/s | 380 img/s | 1.5x |
| 8    | 350 img/s | 750 img/s | 2.1x |

---

### 3. Gradient Accumulation: Virtual Large Batches

**Problem:** GPU memory limits batch size.

```
Desired batch size: 128
GPU memory: Fits batch 32
Problem: Can't train with large batch!
```

**Solution:** Accumulate gradients over multiple steps.

```python
# Effective batch size = 32 × 4 = 128
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward pass
    loss = model(batch) / accumulation_steps  # Scale loss!

    # Backward pass (gradients accumulate)
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        # Update every N steps
        optimizer.step()
        optimizer.zero_grad()
```

**Why it works:**

```
Gradients add linearly:
  ∇L(batch1) + ∇L(batch2) = ∇L(batch1 ∪ batch2)

So accumulating over 4 batches of 32 = 1 batch of 128!
```

**Combined with DDP:**

```
4 GPUs × 32 batch × 4 accumulation = 512 effective batch size!
```

---

### 4. Fully Sharded Data Parallel (FSDP)

**Paper:** "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (Meta, 2023)

**Problem:** DDP replicates entire model on each GPU.

```
LLAMA-70B model: 140GB
4 GPUs with DDP: 140GB × 4 = 560GB total memory!
Wasteful: Most memory for model replication
```

**Solution:** Shard model parameters, gradients, and optimizer states.

```
FSDP Strategy:
┌─────────────────┐  ┌─────────────────┐
│ GPU 0           │  │ GPU 1           │
│ Params: 1-25%   │  │ Params: 26-50%  │
│ Grads: 1-25%    │  │ Grads: 26-50%   │
│ Optim: 1-25%    │  │ Optim: 26-50%   │
└─────────────────┘  └─────────────────┘
┌─────────────────┐  ┌─────────────────┐
│ GPU 2           │  │ GPU 3           │
│ Params: 51-75%  │  │ Params: 76-100% │
│ Grads: 51-75%   │  │ Grads: 76-100%  │
│ Optim: 51-75%   │  │ Optim: 76-100%  │
└─────────────────┘  └─────────────────┘

Forward pass: Gather needed params, compute, discard
Backward pass: Gather needed params, compute grads, shard
```

**Memory savings:**

```
DDP (4 GPUs):
  Parameters: 140GB × 4 = 560GB
  Gradients: 140GB × 4 = 560GB
  Optimizer: 280GB × 4 = 1120GB (AdamW)
  Total: 2240GB

FSDP (4 GPUs):
  Parameters: 140GB / 4 = 35GB per GPU
  Gradients: 140GB / 4 = 35GB per GPU
  Optimizer: 280GB / 4 = 70GB per GPU
  Total: 560GB (4x reduction!)
```

**Implementation:**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyLargeModel()
model = FSDP(model,
             sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard everything
             mixed_precision=bf16_policy,  # Use BF16
             device_id=rank)

# Training works same as DDP!
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Used by:**
- Meta for LLAMA-2 training
- OpenAI for GPT models
- Any model > 10B parameters

---

### 5. Mixed Precision Training

**Idea:** Use FP16/BF16 for most computations, FP32 for critical parts.

**Memory savings:**

```
FP32 model: 4 bytes/param
  LLAMA-70B: 70B × 4 = 280GB

FP16 model: 2 bytes/param
  LLAMA-70B: 70B × 2 = 140GB (50% savings!)

BF16 model: 2 bytes/param (better range than FP16)
  LLAMA-70B: 70B × 2 = 140GB + better numerical stability
```

**Speed improvements:**

```
Modern GPUs have specialized FP16/BF16 hardware:
  A100: 312 TFLOPS (FP16) vs 156 TFLOPS (FP32)
  → 2x faster compute!
```

**Implementation (AMP - Automatic Mixed Precision):**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # Prevents underflow

for batch in dataloader:
    optimizer.zero_grad()

    # Forward in FP16
    with autocast():
        output = model(batch)
        loss = criterion(output, labels)

    # Backward with gradient scaling
    scaler.scale(loss).backward()

    # Unscale before optimizer step
    scaler.step(optimizer)
    scaler.update()
```

**Why gradient scaling:**

```
FP16 range: [6e-5, 65504]
Small gradients (< 6e-5) → underflow → become zero!

Solution: Scale gradients up before backward, down after:
  gradient = 0.0001  ← Would underflow in FP16!
  scaled_gradient = 0.0001 × 1024 = 0.1024  ← Safe!
  After backward: gradient = 0.1024 / 1024 = 0.0001 ✓
```

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Data parallelism vs model parallelism
- AllReduce algorithm and communication
- Gradient synchronization mathematics
- DDP vs DP trade-offs
- FSDP memory sharding strategy
- Mixed precision training and gradient scaling
- Communication overhead and bandwidth
- Scaling laws and efficiency

**Practical Skills:**
- Setup distributed training environment
- Implement DDP from scratch
- Use gradient accumulation for large batches
- Apply FSDP for large models
- Enable mixed precision training
- Debug distributed training issues
- Measure and optimize GPU utilization
- Scale training to multiple nodes

---

## 🔑 Key Concepts

### 1. Communication Patterns

**AllReduce (Ring Algorithm):**

```
4 GPUs, data size D:
  Naive: Each GPU → broadcast to all
    Communication: 3 × D per GPU = 3D total

  Ring AllReduce:
    Scatter-reduce + All-gather
    Communication: 2 × (N-1)/N × D ≈ 2D
    Bandwidth optimal!
```

**Time breakdown:**

```
Single iteration:
  Compute (forward + backward): 90%
  Communication (AllReduce): 10%

If communication > 20% → bottleneck!
Solutions:
  - Larger batch per GPU
  - Gradient compression
  - Overlap compute and communication
```

### 2. Scaling Efficiency

**Linear scaling (ideal):**

```
1 GPU: 100 img/s
2 GPUs: 200 img/s (2.0x)
4 GPUs: 400 img/s (4.0x)
8 GPUs: 800 img/s (8.0x)
```

**Real-world scaling:**

```
1 GPU: 100 img/s
2 GPUs: 190 img/s (1.9x) ← 95% efficient
4 GPUs: 360 img/s (3.6x) ← 90% efficient
8 GPUs: 680 img/s (6.8x) ← 85% efficient

Efficiency = actual speedup / ideal speedup
```

**Scaling bottlenecks:**
- Communication overhead (network bandwidth)
- Synchronization barriers (stragglers)
- Load imbalance (uneven batches)
- I/O bottleneck (data loading)

### 3. Batch Size and Learning Rate Scaling

**Linear Scaling Rule (Goyal et al., 2017):**

```
If you multiply batch size by N, multiply learning rate by N.

Example:
  1 GPU, batch 32, LR 0.001
  ↓
  4 GPUs, batch 128, LR 0.004

Why: Larger batch → less noise → can take larger steps
```

**Warmup:**

```
Large LR can destabilize early training.
Solution: Gradually increase LR at start.

for epoch in range(warmup_epochs):
    lr = target_lr * (epoch / warmup_epochs)
    set_lr(optimizer, lr)
```

---

## 🧪 Exercises

### Exercise 1: Basic DDP Setup (60 mins)

**What You'll Learn:**
- Distributed process group initialization
- DDP model wrapping
- Distributed sampler for data loading
- Multi-GPU training from scratch

**Why It Matters:**
DDP is the standard for distributed training. Every major AI lab uses it for multi-GPU training.

**Tasks:**
1. Setup distributed environment (NCCL backend)
2. Wrap model with DDP
3. Create DistributedSampler for data
4. Implement training loop
5. Launch with torchrun/torch.distributed.launch
6. Verify gradient synchronization

**Starter code:**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    # Create model
    model = ResNet50().to(rank)
    model = DDP(model, device_ids=[rank])

    # Create distributed sampler
    train_sampler = DistributedSampler(dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch!

        for batch in train_loader:
            # ... training code ...
            pass

    dist.destroy_process_group()
```

**Launch:**
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Or with torch.distributed.launch (older)
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

---

### Exercise 2: Gradient Accumulation (30 mins)

**What You'll Learn:**
- Implementing gradient accumulation
- Effective batch size calculation
- Combining with DDP
- Memory-efficient large-batch training

**Why It Matters:**
Train with large batches without large GPUs. Critical for reproducing research results with limited hardware.

**Tasks:**
1. Modify training loop for gradient accumulation
2. Properly scale loss
3. Update every N steps
4. Combine with DDP (distributed gradient accumulation)
5. Verify effective batch size
6. Compare convergence with different accumulation steps

---

### Exercise 3: FSDP for Large Models (90 mins)

**What You'll Learn:**
- FSDP configuration and sharding strategies
- Memory profiling and optimization
- Mixed precision with FSDP
- Training models that don't fit on single GPU

**Why It Matters:**
FSDP is how Meta trains LLAMA, how researchers train large models. Essential for frontier AI.

**Tasks:**
1. Setup FSDP with full sharding
2. Configure mixed precision policy
3. Train model > GPU memory
4. Profile memory usage (FSDP vs DDP)
5. Measure throughput and scaling
6. Compare different sharding strategies

**Implementation:**
```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)

# Mixed precision policy
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16
)

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16_policy,
    device_id=rank,
    limit_all_gathers=True  # Save memory
)

# Training same as DDP!
```

---

### Exercise 4: Mixed Precision Training (45 mins)

**What You'll Learn:**
- Automatic Mixed Precision (AMP)
- Gradient scaling
- Loss scaling strategies
- FP16 vs BF16 trade-offs

**Why It Matters:**
2x speedup with minimal code changes. Standard in production training.

**Tasks:**
1. Implement AMP training loop
2. Configure GradScaler
3. Measure memory savings
4. Benchmark FP16 vs FP32 speed
5. Handle numerical instability
6. Test BF16 (if available)

---

### Exercise 5: Multi-Node Training (120 mins)

**What You'll Learn:**
- Multi-node setup and configuration
- Network communication patterns
- Fault tolerance and checkpointing
- Real distributed training at scale

**Why It Matters:**
This is how production AI is trained. Understand multi-node = understand frontier AI infrastructure.

**Tasks:**
1. Setup multi-node environment (2+ machines)
2. Configure network (TCP/NCCL)
3. Launch distributed training across nodes
4. Implement fault-tolerant checkpointing
5. Measure cross-node bandwidth
6. Optimize for network efficiency

**Multi-node launch:**
```bash
# Master node (rank 0)
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 --master_addr=192.168.1.100 \
         --master_port=29500 train.py

# Worker node (rank 1)
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=1 --master_addr=192.168.1.100 \
         --master_port=29500 train.py
```

---

### Exercise 6: Performance Optimization (90 mins)

**What You'll Learn:**
- Profiling distributed training
- Identifying bottlenecks
- Communication-computation overlap
- Maximum GPU utilization

**Why It Matters:**
The difference between 60% and 95% GPU utilization is 58% more training! Optimization is critical.

**Tasks:**
1. Profile training with PyTorch Profiler
2. Identify bottlenecks (compute vs communication)
3. Optimize data loading (prefetching, pinned memory)
4. Implement gradient bucketing
5. Measure and visualize GPU utilization
6. Achieve >90% efficiency

---

## 📝 Design Patterns

### Pattern 1: Standard DDP Training

```python
def main(rank, world_size):
    # Setup
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Model
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Training
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            loss = train_step(model, batch)

            if rank == 0:  # Only log from master
                print(f"Loss: {loss.item()}")

    # Cleanup
    dist.destroy_process_group()

# Launch
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
```

### Pattern 2: Gradient Accumulation with DDP

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(loader):
    # Forward
    with autocast():
        loss = model(batch) / accumulation_steps

    # Backward
    scaler.scale(loss).backward()

    # Update every N steps
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Pattern 3: FSDP with Checkpointing

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),  # Offload to CPU
    mixed_precision=bf16_policy,
    device_id=rank
)

# Save checkpoint
if rank == 0:
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state_dict = model.state_dict()
        torch.save(state_dict, "checkpoint.pt")
```

---

## ✅ Solutions

Complete implementations in `solution/` directory.

**Files:**
- `01_ddp_basic.py` - Basic DDP setup
- `02_gradient_accumulation.py` - Gradient accumulation
- `03_fsdp.py` - FSDP for large models
- `04_mixed_precision.py` - AMP training
- `05_multi_node.py` - Multi-node training
- `06_profiling.py` - Performance profiling

Run examples:
```bash
cd solution

# Single node DDP
torchrun --nproc_per_node=4 01_ddp_basic.py

# FSDP
torchrun --nproc_per_node=4 03_fsdp.py

# Multi-node (2 nodes, 4 GPUs each)
# Node 0:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
         --master_addr=NODE0_IP --master_port=29500 05_multi_node.py

# Node 1:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
         --master_addr=NODE0_IP --master_port=29500 05_multi_node.py
```

---

## 🎓 Key Takeaways

1. **DDP is standard** - Multi-process, efficient AllReduce
2. **Avoid DP** - Single-process, inefficient
3. **Gradient accumulation** - Large effective batch size
4. **FSDP for large models** - Shard parameters/gradients/optimizer
5. **Mixed precision** - 2x speedup, 50% memory savings
6. **Linear LR scaling** - Batch size × N → LR × N
7. **Communication overhead** - Minimize for scaling
8. **Efficiency matters** - 90%+ GPU utilization

**The Scaling Ladder:**
```
Single GPU → DDP (2-8 GPUs) → FSDP (8-64 GPUs) → Multi-node (64-1000s GPUs)
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**At Meta (LLAMA-2 Training):**
- 2048 A100 GPUs (256 nodes × 8 GPUs)
- FSDP with activation checkpointing
- 2 trillion tokens (several weeks)
- Cost: Millions of dollars

**At OpenAI (GPT-3):**
- 10,000+ V100 GPUs
- Model parallelism + data parallelism
- 300B tokens, 96 days
- Custom distributed training infrastructure

**At Google (BERT):**
- 64 TPU v3 chips (8×8 pod)
- Data parallelism across TPUs
- 4 days training
- Highly optimized for TPU hardware

**Cost implications:**
- A100 GPU: $2-4/hour
- 8-GPU node: $16-32/hour
- 1000-GPU cluster: $2000-4000/hour
- Week of training: $336K-672K!

**Efficiency savings:**
- 90% vs 60% utilization = 50% cost reduction
- $672K → $403K saved per training run

---

## 🚀 Next Steps

1. **Complete all exercises** - Build distributed training expertise
2. **Scale real model** - Train on multiple GPUs
3. **Read Meta FSDP paper** - Deep dive into sharding
4. **Move to Lab 6** - Model Export and Deployment

---

## 💪 Bonus Challenges

1. **ZeRO Optimizer**
   - Implement DeepSpeed ZeRO
   - Compare to FSDP
   - Benchmark memory and speed

2. **Gradient Compression**
   - Implement PowerSGD compression
   - Reduce communication volume
   - Measure impact on convergence

3. **Fault Tolerance**
   - Implement elastic training (handle failures)
   - Automatic checkpoint recovery
   - Test with node failures

4. **Pipeline Parallelism**
   - Split model across GPUs by layers
   - Implement GPipe-style pipelining
   - Compare to data parallelism

5. **Custom AllReduce**
   - Implement ring AllReduce from scratch
   - Benchmark against PyTorch
   - Visualize communication patterns

---

## 📚 Essential Resources

**Papers:**
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - Goyal et al., 2017 (Linear scaling rule)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) - Li et al., 2020 (DDP)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - Rajbhandari et al., 2019
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

**Tutorials:**
- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html)

**Tools:**
- [torchrun](https://pytorch.org/docs/stable/elastic/run.html) - Launch utility
- [DeepSpeed](https://www.deepspeed.ai/) - Microsoft's distributed training
- [Horovod](https://github.com/horovod/horovod) - Uber's distributed training

---

## 🤔 Common Pitfalls

### Pitfall 1: Forgetting to Set Epoch for Sampler

```python
# ❌ Same shuffling every epoch!
for epoch in range(num_epochs):
    for batch in loader:
        train(batch)

# ✓ Different shuffling each epoch
sampler = DistributedSampler(dataset)
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Critical!
    for batch in loader:
        train(batch)
```

### Pitfall 2: Not Scaling Loss with Gradient Accumulation

```python
# ❌ Wrong accumulated gradient!
for i in range(4):
    loss = model(batch)
    loss.backward()  # Accumulates 4x gradient!

# ✓ Scale loss properly
for i in range(4):
    loss = model(batch) / 4  # Divide by accumulation steps
    loss.backward()
```

### Pitfall 3: Synchronizing Too Often

```python
# ❌ Synchronization every iteration (slow!)
for batch in loader:
    loss = model(batch)
    if rank == 0:
        dist.all_reduce(loss)  # Unnecessary sync!
        print(loss)

# ✓ Minimize synchronization
for i, batch in enumerate(loader):
    loss = model(batch)
    if rank == 0 and i % 100 == 0:  # Log every 100 steps
        print(loss)  # No sync needed!
```

### Pitfall 4: Wrong Batch Size Scaling

```python
# ❌ Same LR for different batch sizes
# 1 GPU, batch 32, LR 0.001
# 4 GPUs, batch 128, LR 0.001  ← Wrong!

# ✓ Scale LR with batch size
# 1 GPU, batch 32, LR 0.001
# 4 GPUs, batch 128, LR 0.004  ← 4x LR for 4x batch
```

---

## 💡 Pro Tips

1. **Use torchrun** - Easier than torch.distributed.launch
2. **Start with DDP** - Simplest and most common
3. **Profile before optimizing** - Find real bottlenecks
4. **Gradient checkpointing** - Trade compute for memory
5. **NCCL for NVIDIA** - Fastest backend for CUDA
6. **Pin memory** - Faster CPU→GPU transfer
7. **Prefetch data** - Overlap I/O with compute
8. **Monitor GPU utilization** - Aim for 90%+

---

## ✨ You're Ready When...

- [ ] You understand data parallelism fundamentals
- [ ] You can setup and run DDP training
- [ ] You know DDP vs DP differences
- [ ] You've implemented gradient accumulation
- [ ] You understand AllReduce operation
- [ ] You can use FSDP for large models
- [ ] You've applied mixed precision training
- [ ] You understand linear LR scaling
- [ ] You can profile and optimize training
- [ ] You've achieved 90%+ GPU utilization

**Remember:** Distributed training is how modern AI is built. Master it, and you can train state-of-the-art models!
