# Lab 5: GPU Acceleration & Mixed Precision - Training at Scale ⚡

> **Time:** 2-3 hours
> **Difficulty:** Intermediate
> **Goal:** Master GPU training and memory optimization techniques used in production ML systems

---

## 📖 Why This Lab Matters

You've learned to build and train models. But there's a problem:

**Modern deep learning is computationally MASSIVE.**
- GPT-3: 355 GPU-years of compute
- Stable Diffusion: Thousands of GPU-hours
- Even a small ResNet on ImageNet: Days on CPU → Hours on GPU

This lab teaches you how to:
1. **Accelerate training** 10-100x using GPUs
2. **Reduce memory** 2x with mixed precision
3. **Scale to production** where compute = money

Understanding GPU optimization isn't just about speed - it's about making the impossible possible. Many models simply **cannot be trained without GPUs**.

---

## 🧠 The Big Picture: Why GPUs?

### The Problem: CPUs are Sequential

Modern neural networks require **billions of operations**:
- ResNet forward pass: ~4 billion FLOPs
- Training step: Forward + backward = ~12 billion ops
- Full training: Millions of steps

**CPUs execute sequentially:**
```
CPU (8 cores, 3 GHz):
- 8 operations in parallel
- Optimized for: Complex logic, branching, caching
- Bad at: Doing the same thing millions of times
```

**Training ResNet on CPU:** ~2 weeks 😱

### The Solution: Massive Parallelism

**GPUs execute in parallel:**
```
NVIDIA A100 GPU:
- 6,912 CUDA cores
- 19.5 TFLOPS (FP32)
- 312 TFLOPS (with Tensor Cores, FP16)
- Memory bandwidth: 1.5 TB/s

Same ResNet training: ~12 hours 🚀
```

### The Architecture Difference

**CPU Architecture:**
```
[Control] [Control] [Control] [Control]
   [ALU]     [ALU]     [ALU]     [ALU]
   [L1]      [L1]      [L1]      [L1]
        [Large L2/L3 Cache]

Design: Complex cores, large cache, branch prediction
Perfect for: Operating systems, databases, complex logic
```

**GPU Architecture:**
```
[Simple ALU] [Simple ALU] [Simple ALU] ... × 6,912
[Simple ALU] [Simple ALU] [Simple ALU]
[Simple ALU] [Simple ALU] [Simple ALU]
        [High Bandwidth Memory]

Design: Many simple cores, minimal cache, SIMD
Perfect for: Matrix multiplication, convolutions, parallel ops
```

### Real-World Impact

**Neural network operations are embarrassingly parallel:**

```python
# Matrix multiplication: C = A @ B
# CPU: Compute each element sequentially
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j]  # ~1M sequential ops

# GPU: Compute ALL elements in parallel
C = A @ B  # 1000x faster - all at once!
```

---

## 🔬 Deep Dive: CUDA and PyTorch

### CUDA: Programming the GPU

**CUDA (Compute Unified Device Architecture)** is NVIDIA's platform for GPU programming.

**Key Concepts:**

1. **Host (CPU) vs Device (GPU)**
```python
# Data lives on CPU by default
x_cpu = torch.randn(1000, 1000)  # Host memory

# Must explicitly move to GPU
x_gpu = x_cpu.to('cuda')  # Device memory
```

2. **Kernel Launches**
   - CPU code launches GPU kernels (functions)
   - Thousands of threads execute in parallel
   - PyTorch hides this complexity!

3. **Memory Transfers**
```
CPU → GPU transfer: ~10 GB/s (PCIe bottleneck)
GPU internal ops: ~1500 GB/s (HBM bandwidth)

Lesson: Minimize data movement between CPU/GPU!
```

### PyTorch Device Management

**Moving tensors to GPU:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Method 1: .to()
x = torch.randn(100, 100).to(device)

# Method 2: Direct creation
x = torch.randn(100, 100, device=device)

# Method 3: Model
model = MyModel().to(device)
```

**Critical Rule: All tensors in an operation must be on the same device!**
```python
# ❌ Error!
x_cpu = torch.randn(10)
x_gpu = torch.randn(10, device='cuda')
z = x_cpu + x_gpu  # RuntimeError: Expected all tensors on same device

# ✓ Correct
x_cpu = torch.randn(10)
x_gpu = x_cpu.to('cuda')
z = x_gpu + x_gpu  # Works!
```

---

## 🎯 Mixed Precision Training: The Memory Revolution

### The Problem: Memory Bottleneck

**GPUs have limited memory:**
- RTX 3090: 24 GB
- A100: 40-80 GB
- H100: 80 GB

**Modern models are HUGE:**
- GPT-3: 350 GB (just weights!)
- Stable Diffusion: ~5 GB
- ResNet-50: ~100 MB

**During training, memory holds:**
1. Model weights
2. Gradients (same size as weights)
3. Optimizer state (2x weights for Adam)
4. Activations (for backprop)

**Total: ~4x model size + activations!**

### The Solution: Mixed Precision (FP16 + FP32)

**Floating Point Formats:**

**FP32 (Full Precision):**
```
32 bits: 1 sign + 8 exponent + 23 mantissa
Range: ±3.4 × 10³⁸
Precision: ~7 decimal digits
Memory: 4 bytes per number
```

**FP16 (Half Precision):**
```
16 bits: 1 sign + 5 exponent + 10 mantissa
Range: ±65,504
Precision: ~3 decimal digits
Memory: 2 bytes per number
```

**Benefits of FP16:**
1. **2x memory reduction** → Larger batch sizes or models
2. **2-3x speed increase** → Tensor Cores accelerate FP16
3. **Same final accuracy** → With proper techniques

**Why not use FP16 everywhere?**

**Problem 1: Gradient Underflow**
```
Gradients are often TINY (e.g., 0.00001)
In FP16: Numbers < 6 × 10⁻⁸ become ZERO!
Result: Gradients vanish, training fails
```

**Problem 2: Loss of Precision**
```
Weight update: w ← w - 0.0001 × grad
If w = 1.234 and update = 0.0001
FP16 can't represent this precision!
```

### Automatic Mixed Precision (AMP): Best of Both Worlds

**Strategy:**
1. **Store weights in FP32** (master copy)
2. **Compute forward/backward in FP16** (fast)
3. **Update weights in FP32** (precise)
4. **Scale gradients** to prevent underflow

**The AMP Algorithm:**
```python
# Pseudo-code of what AMP does
with autocast():  # Enable FP16 for ops that can handle it
    output = model(input)  # FP16 computation
    loss = criterion(output, target)

# Gradient scaling prevents underflow
scaler.scale(loss).backward()  # loss *= scale_factor
scaler.step(optimizer)  # Unscale gradients, then update FP32 weights
scaler.update()  # Adjust scale_factor for next iteration
```

**Gradient Scaling:**
```
Problem: grad = 0.00001 → Underflows in FP16

Solution: Scale up BEFORE FP16 conversion
scaled_grad = 0.00001 × 65536 = 0.65536 ✓ (representable!)

After backward: Unscale before optimizer step
grad = scaled_grad / 65536 = 0.00001 ✓ (correct value!)
```

**Dynamic Loss Scaling:**
- Start with scale = 65536
- If gradients overflow (inf/nan): Reduce scale
- If gradients are stable: Increase scale
- Automatically adapts to your model!

### Which Operations Use FP16?

PyTorch autocast automatically chooses precision:

**FP16 (safe & fast):**
- Matrix multiplication (`@`)
- Convolutions
- Linear layers
- Element-wise ops

**FP32 (needs precision):**
- Reductions (sum, mean)
- Loss functions
- Normalization layers
- Softmax

---

## 📊 Mathematical Foundations: Memory Analysis

### Memory Breakdown for Training

**Given:**
- Model parameters: P
- Batch size: B
- Sequence length (for transformers): L
- Hidden dimension: H

**Memory = Weights + Gradients + Optimizer + Activations**

**1. Model Weights:**
```
FP32: 4P bytes
FP16: 2P bytes
```

**2. Gradients:**
```
Same size as weights: 4P or 2P bytes
```

**3. Optimizer State (Adam):**
```
First moment (m): 4P bytes
Second moment (v): 4P bytes
Total: 8P bytes (always FP32!)
```

**4. Activations (stored for backprop):**
```
Transformer layer: ~2BLH bytes per layer
CNN layer: ~B × C × H × W bytes per layer
Total: Depends on architecture depth
```

**Example: ResNet-50**
```
Parameters: 25M
Batch size: 32
Input: 224×224 RGB

FP32 Training Memory:
- Weights: 4 × 25M = 100 MB
- Gradients: 100 MB
- Optimizer: 8 × 25M = 200 MB
- Activations: ~2 GB
Total: ~2.4 GB

FP16 Training Memory:
- Weights (FP32 master): 100 MB
- Weights (FP16 copy): 50 MB
- Gradients (FP16): 50 MB
- Optimizer: 200 MB (still FP32)
- Activations (FP16): ~1 GB
Total: ~1.4 GB → 1.7x reduction
```

---

## 🎯 Learning Objectives

By the end of this lab, you'll understand:

**Theoretical Understanding:**
- Why GPUs are essential for deep learning (parallelism, memory bandwidth)
- CPU vs GPU architecture trade-offs
- How floating-point precision affects training
- Why mixed precision works (FP16 speed + FP32 accuracy)
- Memory optimization strategies

**Practical Skills:**
- Move models and data to GPU efficiently
- Implement mixed precision training with AMP
- Debug GPU memory errors (OOM)
- Optimize batch size for throughput
- Benchmark and profile GPU utilization
- Handle multi-GPU scenarios

---

## 🔑 Key Concepts

### 1. Device Placement Best Practices

**Anti-pattern: Unnecessary transfers**
```python
# ❌ Bad: Moves data every iteration
for batch in dataloader:
    batch = batch.to('cuda')  # Slow!
```

**Better: Move once**
```python
# ✓ Better: DataLoader on GPU
dataloader = DataLoader(dataset, pin_memory=True)
# pin_memory=True enables fast CPU→GPU transfer
```

### 2. Memory Management

**Check GPU memory:**
```python
torch.cuda.memory_allocated()  # Currently used
torch.cuda.memory_reserved()   # Reserved by PyTorch
torch.cuda.max_memory_allocated()  # Peak usage
```

**Clear cache:**
```python
torch.cuda.empty_cache()  # Free unused memory
# Use sparingly - PyTorch manages memory efficiently!
```

### 3. Gradient Accumulation

**When batch size doesn't fit in memory:**
```python
effective_batch = 128  # What you want
actual_batch = 32      # What fits in GPU
accumulation_steps = 4 # 32 × 4 = 128

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    output = model(batch.x)
    loss = criterion(output, batch.y) / accumulation_steps
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update after 4 batches
        optimizer.zero_grad()
```

### 4. Asynchronous Execution

**GPUs execute asynchronously:**
```python
# This returns IMMEDIATELY (doesn't wait for GPU)
output = model(input)

# To force synchronization:
torch.cuda.synchronize()  # Wait for all GPU ops to complete
```

**For timing:**
```python
# ❌ Wrong: Doesn't measure GPU time
start = time.time()
output = model(input)
end = time.time()  # Measures LAUNCH time, not execution!

# ✓ Correct: Synchronize before timing
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
end = time.time()  # Actual GPU time
```

---

## 🧪 Exercises

### Exercise 1: GPU Device Management
**File:** `01_gpu_basics.py`

**What You'll Learn:**
- Detect GPU availability
- Move tensors and models to GPU
- Verify device placement
- Handle CPU fallback gracefully

**Why It Matters:**
- Production code runs on different hardware (local CPU, cloud GPU)
- Must handle both gracefully
- Device mismatches are the #1 beginner bug

**Tasks:**
```python
# TODO: Check if CUDA is available
# TODO: Create tensor directly on GPU
# TODO: Move existing tensor to GPU
# TODO: Move model to GPU
# TODO: Verify all parameters are on correct device
# TODO: Handle device mismatch errors
```

**Expected Output:**
```
CUDA available: True
Device: cuda:0
GPU Name: NVIDIA A100
Model on GPU: ✓
All parameters on cuda: ✓
```

---

### Exercise 2: Mixed Precision Training
**File:** `02_amp_training.py`

**What You'll Learn:**
- Enable automatic mixed precision
- Use GradScaler for gradient scaling
- Monitor for numerical instability
- Compare FP32 vs FP16 speed and memory

**Why It Matters:**
Mixed precision is **standard practice** in production:
- Meta's LLaMA: Trained with mixed precision
- OpenAI's GPT models: Mixed precision
- Stable Diffusion: Mixed precision
- Every modern vision model: Mixed precision

**Not using AMP = wasting 2x memory and 2x time!**

**Tasks:**
```python
# TODO: Wrap forward pass with autocast
# TODO: Initialize GradScaler
# TODO: Scale loss before backward
# TODO: Unscale and update optimizer
# TODO: Update scaler state
# TODO: Compare memory usage: FP32 vs FP16
# TODO: Compare training speed: FP32 vs FP16
```

**Expected speedup:** 1.5-3x faster, 1.5-2x less memory

---

### Exercise 3: Memory Optimization
**File:** `03_memory_optimization.py`

**What You'll Learn:**
- Profile memory usage
- Optimize batch size for throughput
- Implement gradient accumulation
- Use gradient checkpointing
- Debug OOM (Out of Memory) errors

**Why It Matters:**
**GPU memory is expensive and limited.**
- A100 80GB: ~$15,000
- H100 80GB: ~$30,000

Efficient memory usage = train bigger models = better results!

**Tasks:**
```python
# TODO: Measure memory per batch size
# TODO: Find optimal batch size (max throughput without OOM)
# TODO: Implement gradient accumulation for larger effective batch
# TODO: Use torch.utils.checkpoint for memory-efficient backprop
# TODO: Compare memory: with/without checkpointing
```

**Common OOM scenarios:**
- Batch size too large
- Model too big for GPU
- Activations accumulating
- Memory leaks (detach tensors!)

---

### Exercise 4: End-to-End Accelerated Training
**File:** `04_accelerated_training.py`

**What You'll Learn:**
- Convert existing training loop to GPU + AMP
- Benchmark speedup vs CPU
- Monitor GPU utilization
- Production-ready training script

**Why It Matters:**
This is how you'll train **every model** going forward:
1. Start on CPU for debugging (small data)
2. Move to GPU for real training
3. Enable AMP for maximum efficiency

**Tasks:**
1. Take your MNIST classifier from Lab 4
2. Add GPU support (device placement)
3. Enable mixed precision training
4. Add memory profiling
5. Benchmark speedup: CPU vs GPU vs GPU+AMP
6. Achieve 3-5x speedup minimum

**Expected Results:**
```
CPU Training: 120 sec/epoch
GPU (FP32): 25 sec/epoch → 4.8x speedup
GPU (FP16): 15 sec/epoch → 8x speedup

Memory Usage:
FP32: 8.2 GB
FP16: 4.7 GB → 1.7x reduction
```

---

## 📝 Starter Code

See the `starter/` directory for templates with detailed comments and TODOs.

---

## ✅ Solutions

Full implementations in `solution/` directory.

**Before checking solutions:**
1. Read error messages carefully
2. Check device placement (print tensor.device)
3. Verify batch fits in memory (reduce batch size)
4. Test on CPU first (isolate GPU issues)

---

## 🎓 Key Takeaways

After completing this lab, you should understand:

1. **GPUs are designed for parallelism** - Perfect for neural networks (matrix ops)
2. **Device placement is critical** - All tensors in operation must match
3. **Mixed precision = 2x memory, 2-3x speed** - With minimal code changes
4. **Gradient scaling prevents underflow** - Makes FP16 training stable
5. **Memory is the bottleneck** - Batch size tuning is crucial
6. **Async execution** - GPU ops return immediately
7. **Production ML requires GPU efficiency** - Compute = money

**The Fundamental Insight:**
```
Modern deep learning is impossible without GPUs.
Understanding GPU optimization is understanding production ML.
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**Cost Savings:**
- Training LLaMA-2 70B: ~1M GPU-hours
- At $2/hour: $2M in compute costs!
- 2x speedup from AMP: Save $1M

**Faster Iteration:**
- Experiment takes 8 hours instead of 24 hours
- 3 experiments/day instead of 1
- Faster experimentation = better models

**Bigger Models:**
- 2x memory → 2x model size
- Bigger models often = better performance
- Enables research that wasn't possible before

**Environmental Impact:**
- Training GPT-3: ~500 tons CO₂
- Efficiency improvements = reduce carbon footprint

### Real-World Examples

**Meta's LLaMA:**
- Trained on 2048 A100 GPUs
- Mixed precision training
- Custom CUDA kernels for efficiency

**OpenAI's GPT-4:**
- Thousands of GPUs
- Advanced multi-GPU parallelism
- Every optimization matters at this scale

**Stable Diffusion:**
- Would be impossible to train without GPUs
- Mixed precision enables high-resolution training
- Community fine-tuning relies on accessible GPU training

---

## 🚀 Next Steps

Once you've completed all exercises:

1. **Profile your GPU usage** - Use `nvidia-smi` to check utilization
2. **Experiment with batch sizes** - Find the sweet spot
3. **Compare architectures** - How does memory scale with model size?
4. **Move to Lab 6** - Build CNNs with your new GPU skills!

---

## 💪 Bonus Challenges

1. **Multi-GPU Training Preview**
   - Use `DataParallel` for simple multi-GPU
   - Compare 1 GPU vs 2 GPUs vs 4 GPUs
   - Understand communication overhead

2. **Custom CUDA Kernels (Advanced)**
   - Write simple CUDA kernel
   - Call from PyTorch
   - Measure speedup vs PyTorch ops

3. **Memory Profiling Dashboard**
   - Track memory over time
   - Identify memory leaks
   - Visualize allocation patterns

4. **Gradient Checkpointing Deep Dive**
   - Implement manual checkpointing
   - Trade compute for memory
   - Find optimal checkpoint frequency

5. **Mixed Precision for Transformers**
   - Apply to attention mechanism
   - Handle numerical instability in softmax
   - Compare convergence: FP32 vs FP16

6. **TensorFloat-32 (TF32)**
   - Enable TF32 on Ampere GPUs
   - Compare: FP32 vs TF32 vs FP16
   - Understand precision/speed trade-offs

---

## 📚 Additional Resources

**Essential Reading:**
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)

**Papers:**
- [Mixed Precision Training (2018)](https://arxiv.org/abs/1710.03740) - Original AMP paper
- [Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - Large batch training

**Tools:**
- `nvidia-smi`: Monitor GPU usage, memory, utilization
- `torch.profiler`: Profile PyTorch operations
- [PyTorch Memory Profiler](https://pytorch.org/blog/understanding-gpu-memory-1/)

**Video Lectures:**
- [NVIDIA GTC Talks](https://www.nvidia.com/en-us/gtc/) - Latest GPU optimization techniques
- [Fast.ai: Deep Learning for Coders](https://course.fast.ai/) - Practical GPU training tips

---

## 🤔 Common Pitfalls & Solutions

### Pitfall 1: Device Mismatch
```python
# ❌ Error: tensors on different devices
model = model.to('cuda')
for batch in dataloader:
    output = model(batch.x)  # batch.x still on CPU!

# ✓ Fix: Move batch to GPU
for batch in dataloader:
    batch_gpu = batch.to('cuda')
    output = model(batch_gpu)
```

### Pitfall 2: Out of Memory (OOM)
```python
# ❌ Batch size too large
batch_size = 256  # OOM!

# ✓ Solutions:
# 1. Reduce batch size
batch_size = 64

# 2. Use gradient accumulation
accumulation_steps = 4  # Effective batch = 64 × 4 = 256

# 3. Enable mixed precision
with autocast():  # Uses less memory
    output = model(x)
```

### Pitfall 3: Forgetting to Synchronize
```python
# ❌ Wrong timing
start = time.time()
output = model(input)  # Returns immediately!
print(time.time() - start)  # Wrong!

# ✓ Correct timing
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
print(time.time() - start)  # Correct!
```

### Pitfall 4: Memory Leaks
```python
# ❌ Accumulating history
losses = []
for batch in dataloader:
    loss = criterion(output, target)
    losses.append(loss)  # Keeps graph in memory!

# ✓ Detach scalars
losses = []
for batch in dataloader:
    loss = criterion(output, target)
    losses.append(loss.item())  # Just the number, no graph
```

### Pitfall 5: Incorrect AMP Usage
```python
# ❌ Wrong: Scale before autocast
loss.backward()  # Loss not scaled!

# ✓ Correct: Scale inside autocast context
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()  # Scaled correctly
```

---

## 💡 Pro Tips

1. **Always use pin_memory=True** - Faster CPU→GPU transfer
2. **Start with small batch size** - Increase until OOM, then back off
3. **Use mixed precision by default** - Unless you have a specific reason not to
4. **Monitor GPU utilization** - Run `nvidia-smi` in another terminal
5. **Profile before optimizing** - Don't guess bottlenecks
6. **Batch size affects convergence** - Larger batch = adjust learning rate
7. **Clear cache only when needed** - PyTorch manages memory well
8. **Use DataLoader num_workers** - Parallelize data loading

**Golden Rule:**
```
If you're not using 80%+ GPU utilization,
you're leaving performance on the table.
```

---

## ✨ You're Ready When...

- [ ] You can explain why GPUs accelerate deep learning
- [ ] You understand CPU vs GPU architecture differences
- [ ] You can move models and data to GPU correctly
- [ ] You've implemented mixed precision training with AMP
- [ ] You understand gradient scaling and why it's needed
- [ ] You can debug OOM errors and optimize batch size
- [ ] You know how to profile GPU memory usage
- [ ] You can benchmark speedups from GPU + AMP
- [ ] You understand when to use FP16 vs FP32

**Critical Understanding:**
- GPUs aren't just "faster CPUs" - they're fundamentally different architectures
- Mixed precision isn't magic - it's carefully managing precision where it matters
- Memory optimization is as important as speed optimization
- Production ML requires mastering these techniques

**Next:** Lab 6 - You'll use GPU training to build a CNN classifier!

