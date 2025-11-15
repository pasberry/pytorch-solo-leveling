# Lab 4: Data Loading Pipeline - The Foundation of Efficient Training 📦

> **Time:** 2-3 hours
> **Difficulty:** Intermediate
> **Goal:** Build production-grade data pipelines that don't bottleneck GPU training

---

## 📖 Why This Lab Matters

**The harsh truth about deep learning:**

Your model can only learn from data it sees. But here's the problem:
- **GPUs are FAST** - Process millions of operations per second
- **Data loading is SLOW** - Reading from disk, preprocessing, augmentation
- **Result:** GPU sits idle waiting for data (wasteful and expensive!)

**At Meta scale:**
- Training on billions of images
- Data loading is often the bottleneck
- Inefficient pipelines = millions in wasted compute
- Proper data loading = 10-100x speedup

This lab teaches you to build data pipelines that keep GPUs fed and training fast.

---

## 🧠 The Big Picture: The Data Loading Bottleneck

### The Problem

**Naive approach:**
```python
for epoch in range(100):
    for i in range(len(dataset)):
        # Read image from disk (SLOW! ~10ms)
        image = load_image(i)
        # Preprocess (SLOW! ~5ms)
        image = preprocess(image)
        # Train (FAST! ~1ms on GPU)
        loss = model(image)
        loss.backward()
```

**Time per sample:** 10ms + 5ms + 1ms = **16ms**
**GPU utilization:** 1ms / 16ms = **6%** ← TERRIBLE!

### The Solution: Parallel Data Loading

```python
# PyTorch loads data in parallel workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 4 parallel processes loading data!
    pin_memory=True  # Fast CPU→GPU transfer
)

for batch in train_loader:
    # Data already loaded and preprocessed!
    loss = model(batch)
    loss.backward()
```

**GPU utilization:** ~95% ← EXCELLENT!

---

## 🔬 Deep Dive: PyTorch Data Loading Architecture

### The Dataset Abstraction

**Why we need it:**
- Thousands of image formats (JPEG, PNG, TIFF, ...)
- Different data sources (disk, S3, database, ...)
- Various preprocessing needs
- Need a **unified interface**

**The Dataset contract:**
```python
class Dataset:
    def __len__(self):
        """Return total number of samples"""
        return num_samples

    def __getitem__(self, idx):
        """Return sample at index idx"""
        return sample, label
```

**That's it!** PyTorch can now:
- Iterate over your data
- Shuffle it
- Batch it
- Load it in parallel

### DataLoader: The Parallel Loading Engine

**What DataLoader does:**

1. **Batching:** Groups samples into batches
   ```python
   # Instead of: [sample1], [sample2], [sample3], ...
   # You get: [[sample1, sample2, sample3, ...], ...]
   ```

2. **Shuffling:** Randomizes order each epoch
   ```python
   # Prevents model from learning order patterns
   # Critical for generalization
   ```

3. **Parallel loading:** Spawns worker processes
   ```python
   # Workers load data in background
   # GPU never waits for data!
   ```

4. **Pin memory:** Fast CPU→GPU transfer
   ```python
   # Pre-allocates page-locked memory
   # 2-3x faster transfers
   ```

### Multi-Process Data Loading

**How it works:**

```
Main Process (Training)
    ↓
    └─→ DataLoader
         ├─→ Worker 1 (loads batch 1)
         ├─→ Worker 2 (loads batch 2)
         ├─→ Worker 3 (loads batch 3)
         └─→ Worker 4 (loads batch 4)
              ↓
         Queue of ready batches
              ↓
         Training consumes batches
```

**Key insight:** While GPU trains on batch N, workers prepare batches N+1, N+2, N+3, N+4!

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Why data loading is often the bottleneck
- How multi-process data loading works
- When to use IterableDataset vs Dataset
- Memory vs computation tradeoffs
- Data augmentation and generalization

**Practical Skills:**
- Build custom Dataset classes for any data type
- Implement efficient data transformations
- Use DataLoader with optimal settings
- Handle imbalanced datasets
- Create data augmentation pipelines
- Debug data loading issues

---

## 📊 Mathematical Foundations

### Data Augmentation: Learning Invariances

**The problem:** Limited training data leads to overfitting

**The solution:** Create variations of existing data

**Common augmentations:**

**Geometric transformations:**
```
Original image → Rotate, flip, crop, scale
Effect: Model learns rotation/position invariance
Used in: All computer vision tasks
```

**Color transformations:**
```
Original image → Adjust brightness, contrast, saturation
Effect: Model learns lighting invariance
Used in: Outdoor scenes, varying conditions
```

**Why it works mathematically:**
```
Given: Limited dataset D = {(x₁, y₁), ..., (xₙ, yₙ)}
Augment: D' = {(T₁(x₁), y₁), (T₂(x₁), y₁), ...}
Where: Tᵢ are label-preserving transformations

Result: Effectively multiply dataset size
        Regularization (prevents overfitting)
```

### Sampling Strategies

**Uniform sampling:**
```python
# Each sample equally likely
prob(sample i) = 1/N
```

**Weighted sampling:**
```python
# Important samples more likely
prob(sample i) = weight_i / Σ weights
```

**Class-balanced sampling:**
```python
# Equal samples per class (handles imbalance)
samples_per_class = batch_size / num_classes
```

---

## 🔑 Key Concepts

### 1. Dataset Types

**Map-Style Dataset (most common):**
```python
class MapDataset(Dataset):
    def __getitem__(self, idx):
        return self.data[idx]

# Random access: can get any index
# Works with: Images, tabular data, pre-loaded data
```

**Iterable-Style Dataset (for streams):**
```python
class IterableDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield next_sample()

# Sequential access: can't jump to index
# Works with: Database streams, real-time data, huge files
```

### 2. Transforms and Augmentation

**Deterministic transforms (same every time):**
```python
transforms.Compose([
    transforms.Resize(224),      # Always resize to 224
    transforms.ToTensor(),       # Always convert to tensor
    transforms.Normalize(...)    # Always normalize
])
```

**Stochastic transforms (random each time):**
```python
transforms.Compose([
    transforms.RandomCrop(224),           # Different crop each time
    transforms.RandomHorizontalFlip(),    # 50% chance of flip
    transforms.ColorJitter(0.2, 0.2)     # Random color variation
])
```

### 3. Collate Functions

**Default collate:**
```python
# Stacks samples into batch
samples = [(img1, label1), (img2, label2), ...]
batch = (torch.stack([img1, img2, ...]),
         torch.tensor([label1, label2, ...]))
```

**Custom collate (for variable-length data):**
```python
def collate_fn(batch):
    # Handle sequences of different lengths
    # Pad to max length in batch
    # Return padded tensors + lengths
```

### 4. Memory vs Speed Tradeoffs

**Cache everything in RAM:**
```
+ Fastest possible (no disk I/O)
- Requires huge RAM
- Only works for small datasets
```

**Load on-the-fly:**
```
+ Works for any dataset size
- Slower (disk I/O every epoch)
+ Can use data augmentation
```

**Hybrid approach:**
```
+ Cache processed data
+ Random augmentation on-the-fly
= Good balance
```

---

## 🧪 Exercises

### Exercise 1: Custom Datasets
**File:** `01_custom_datasets.py`

**What You'll Learn:**
- Build Dataset for various data types
- Handle CSV, images, sequences
- Implement caching for speed
- Multi-task datasets
- Custom collate functions

**Why It Matters:**
Real-world data is messy:
- Multiple modalities (text + images)
- Variable-length sequences
- Different file formats
- Imbalanced classes

You need to handle all of this!

**Tasks:**
1. Implement `CSVDataset` for tabular data
2. Implement `SequenceDataset` with padding
3. Implement `CachedDataset` wrapper
4. Implement `MultiTaskDataset`
5. Write custom collate functions

---

### Exercise 2: Efficient Data Loading
**File:** `02_efficient_loading.py`

**What You'll Learn:**
- Optimal num_workers setting
- Pin memory benefits
- Prefetching strategies
- Persistent workers
- Benchmarking data loading

**Why It Matters:**
- Wrong num_workers → 10x slower training
- No pin_memory → Wasted GPU time
- Understanding tradeoffs → Optimal performance

**Tasks:**
1. Benchmark different num_workers (0, 2, 4, 8)
2. Measure pin_memory speedup
3. Compare transform strategies (on-the-fly vs precomputed)
4. Implement IterableDataset
5. Use persistent_workers for faster epochs

**Expected insights:**
```
num_workers=0:  20 batches/sec  ← Single process bottleneck
num_workers=4:  200 batches/sec ← 10x faster!
num_workers=16: 180 batches/sec ← Diminishing returns

pin_memory=False: 15ms/batch
pin_memory=True:  5ms/batch  ← 3x faster GPU transfer
```

---

## 📝 Best Practices

### 1. Choosing num_workers

```python
# Rule of thumb:
num_workers = min(4 * num_GPUs, num_CPU_cores - 2)

# Too few: GPU starves waiting for data
# Too many: Memory overhead, slower due to context switching

# Always benchmark your specific setup!
```

### 2. Data Augmentation Guidelines

**Training:**
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

**Validation/Test:**
```python
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Deterministic!
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 3. Handling Imbalanced Data

**Problem:**
```
Class 0: 10,000 samples
Class 1: 100 samples  ← Underrepresented
```

**Solutions:**

**Weighted sampling:**
```python
weights = [1.0 / class_counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, len(weights))
```

**Class-balanced batch:**
```python
# Ensure each batch has examples from all classes
```

**Loss weighting:**
```python
class_weights = torch.tensor([1.0, 100.0])  # Weight rare class more
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 🎓 Key Takeaways

1. **Data loading can bottleneck training** - Optimize it!
2. **Multi-process loading is essential** - Use num_workers > 0
3. **Dataset abstraction is powerful** - Unified interface for any data
4. **Augmentation prevents overfitting** - Essential for vision tasks
5. **Transforms should differ** - Train uses random, val uses deterministic
6. **Pin memory speeds GPU transfer** - Always use with CUDA
7. **Benchmark your pipeline** - Every setup is different

**The Core Insight:**
```
Efficient data loading is not optional for production ML.
At scale, it's the difference between:
- Training in 1 day vs 10 days
- Millions in compute costs vs thousands
- Feasible vs infeasible projects
```

---

## 🔗 Connections to Production ML

### Why This Matters at Meta Scale

**Real Numbers:**
- **ImageNet training:** 1.2M images, ~150GB
- **Instagram:** Billions of images, petabytes
- **Feed ranking:** Continuous data streams

**Without efficient loading:**
- GPUs idle 90% of time
- 10x longer training
- 10x higher costs

**With proper pipeline:**
- GPU utilization > 95%
- Faster iteration
- More experiments

**Production patterns:**
- Data stored in distributed systems (HDFS, S3)
- Preprocessing pipelines (Spark, Ray)
- Caching layers (Redis, Memcached)
- Streaming updates (Kafka)

---

## 🚀 Next Steps

1. Complete both exercises
2. Benchmark on your hardware
3. Experiment with different num_workers
4. Move to Lab 5 - GPU training

---

## 💪 Bonus Challenges

1. **Implement DataLoader from scratch**
   - Understand batching, shuffling
   - See why PyTorch's implementation is fast

2. **Build data pipeline for video**
   - Handle temporal dimension
   - Memory-efficient loading
   - Frame sampling strategies

3. **Implement mixup augmentation**
   - Mix two samples: `x = λx₁ + (1-λ)x₂`
   - Improves generalization
   - Used in modern training

4. **Distributed data loading**
   - Split dataset across multiple machines
   - Ensure no overlap
   - Handle worker failures

5. **Auto-tune num_workers**
   - Automatically find optimal setting
   - Adapt to system load
   - Profile and adjust

---

## 📚 Essential Resources

**Documentation:**
- [PyTorch Dataset & DataLoader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [TorchVision Transforms](https://pytorch.org/vision/stable/transforms.html)

**Papers:**
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [AutoAugment](https://arxiv.org/abs/1805.09501) - Learned augmentation

**Libraries:**
- [Albumentations](https://albumentations.ai/) - Fast augmentation library
- [FFCV](https://ffcv.io/) - Ultra-fast data loading

---

## 🤔 Common Pitfalls

### Pitfall 1: Too Many Workers
```python
# ❌ More workers = more context switching overhead
num_workers=32  # On 8-core CPU → slower!

# ✓ Sweet spot is usually 2-8
num_workers=4
```

### Pitfall 2: Transforms in __init__
```python
# ❌ Transform once, same augmentation every epoch!
def __init__(self):
    self.data = [transform(x) for x in raw_data]

# ✓ Transform in __getitem__, different each time
def __getitem__(self, idx):
    return self.transform(self.raw_data[idx])
```

### Pitfall 3: Not Using pin_memory
```python
# ❌ Slow CPU→GPU transfer
loader = DataLoader(dataset, pin_memory=False)

# ✓ Fast transfer with CUDA
loader = DataLoader(dataset, pin_memory=True)
```

### Pitfall 4: Inconsistent Train/Val Transforms
```python
# ❌ Both use random transforms → can't reproduce val results
train_transform = RandomCrop(224)
val_transform = RandomCrop(224)  # Different each time!

# ✓ Val uses deterministic transform
val_transform = CenterCrop(224)  # Same every time
```

---

## 💡 Pro Tips

1. **Always benchmark** - Your intuition about performance is probably wrong
2. **Profile with torchprof** - Find bottlenecks
3. **Cache small datasets** - RAM is faster than disk
4. **Use SSD for data** - 10x faster than HDD
5. **Prefetch to GPU** - Overlap data transfer with compute
6. **Monitor data loading time** - Should be < 10% of iteration time

---

## ✨ You're Ready When...

- [ ] You can build Dataset for any data type
- [ ] You understand the num_workers tradeoff
- [ ] You can implement data augmentation
- [ ] You know when to use pin_memory
- [ ] You can handle imbalanced datasets
- [ ] You can debug data loading bottlenecks

**Next up:** Lab 5 - GPU training and mixed precision for maximum speed!
