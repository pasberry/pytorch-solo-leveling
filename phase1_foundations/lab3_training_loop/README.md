# Lab 3: The Training Loop - From Theory to Production 🔄

> **Time:** 3-4 hours
> **Difficulty:** Intermediate
> **Goal:** Master the complete training workflow used in production ML systems

---

## 📖 Why This Lab Matters

You've learned to build models (Lab 2) and compute gradients (Lab 1). Now comes the critical question:

**How do we actually train these models to solve real problems?**

This lab teaches you the **training loop** - the heartbeat of every machine learning system. Whether you're training:
- GPT-4 (OpenAI)
- LLAMA (Meta)
- Stable Diffusion (Stability AI)
- Recommendation systems (Meta, Netflix, Amazon)

They all use variations of the same fundamental training loop you'll build here.

---

## 🧠 The Big Picture: What is Training?

### The Optimization Problem

Training a neural network is finding the best parameters θ* that minimize a loss function:

```
θ* = argmin L(θ; D)
         θ

Where:
- θ: model parameters (weights, biases)
- L: loss function (measures how bad predictions are)
- D: training dataset
```

### Why Gradient Descent Works

We can't search all possible θ (infinite space!), but we can follow the gradient downhill:

```
θ_{t+1} = θ_t - α ∇L(θ_t)

Intuition:
- ∇L(θ_t) points in direction of steepest INCREASE in loss
- We go the OPPOSITE direction (subtract gradient)
- α (learning rate) controls step size
```

**The training loop is just this simple update repeated thousands of times!**

### The Training Loop Anatomy

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. Forward pass: compute predictions
        predictions = model(batch.x)

        # 2. Compute loss: how wrong are we?
        loss = criterion(predictions, batch.y)

        # 3. Backward pass: compute gradients
        loss.backward()

        # 4. Update parameters: improve model
        optimizer.step()

        # 5. Zero gradients: ready for next batch
        optimizer.zero_grad()
```

**This 5-line loop trains GPT-4!** (with some additional engineering)

---

## 🔬 Deep Dive: Training Dynamics

### Loss Functions: Measuring Error

The loss function L defines what "good" means. Different problems need different losses:

**Classification (Cross-Entropy):**
```
L_CE = -Σ y_i log(ŷ_i)

Why: Encourages high probability for correct class
Used in: Image classification, NLP, speech recognition
```

**Regression (MSE):**
```
L_MSE = (1/N) Σ (y_i - ŷ_i)²

Why: Penalizes large errors more than small ones
Used in: Price prediction, forecasting, control
```

**Ranking (Pairwise Loss):**
```
L_rank = max(0, margin - score(pos) + score(neg))

Why: Ensures relevant items rank higher
Used in: Search, recommendation, feed ranking
```

### Optimizers: How to Update Parameters

**Gradient Descent (Vanilla):**
```
θ ← θ - α∇L(θ)

Pros: Simple, guaranteed convergence (convex)
Cons: Slow, same step size for all parameters
```

**SGD with Momentum:**
```
v ← βv + (1-β)∇L(θ)
θ ← θ - αv

Why: Accumulates gradients, smooths updates
Effect: Faster convergence, escapes local minima
```

**Adam (Adaptive Moment Estimation):**
```
m ← β₁m + (1-β₁)∇L(θ)        # First moment (mean)
v ← β₂v + (1-β₂)(∇L(θ))²     # Second moment (variance)
θ ← θ - α · m / (√v + ε)

Why: Adapts learning rate per parameter
Effect: Fast convergence, works well out-of-the-box
Used: Most popular optimizer (default choice)
```

**Which to use?**
- **Adam**: Default choice, works for most problems
- **SGD + Momentum**: Better generalization for vision tasks
- **AdamW**: Adam with proper weight decay (modern default)

### Learning Rate: The Most Important Hyperparameter

**Too small:**
```
θ ← θ - 0.0001 · ∇L(θ)
Problem: Training is VERY slow, may never converge
```

**Too large:**
```
θ ← θ - 10.0 · ∇L(θ)
Problem: Overshoots minimum, loss oscillates or diverges!
```

**Just right:**
```
θ ← θ - 0.001 · ∇L(θ)
Result: Smooth convergence in reasonable time
```

**How to find the right learning rate?**
1. Start with 0.001 (good default)
2. If loss doesn't decrease → increase LR
3. If loss explodes → decrease LR
4. Use learning rate finder (plot loss vs LR)

### Learning Rate Schedules: Adaptive Training

**Why schedule the learning rate?**
- Early training: Need large steps to escape bad initialization
- Late training: Need small steps to fine-tune solution

**Common schedules:**

**Step Decay:**
```
α(t) = α₀ · γ^(t/k)

Example: Halve LR every 30 epochs
Effect: Large steps → medium steps → small steps
```

**Cosine Annealing:**
```
α(t) = α_min + (α_max - α_min) · (1 + cos(πt/T)) / 2

Effect: Smooth decay with restarts
Used in: Vision models, BERT training
```

**ReduceLROnPlateau:**
```
If validation loss doesn't improve for N epochs:
    α ← α · factor

Effect: Automatic adaptation based on progress
Safest: No manual tuning needed
```

### Gradient Clipping: Preventing Exploding Gradients

**The Problem:**
In deep networks, gradients can explode:
```
∂L/∂θ₁ = 1000.0  ← Normal
∂L/∂θ₂ = 1e10    ← EXPLOSION!
```

**The Solution:**
```
if ||∇L|| > threshold:
    ∇L ← ∇L · threshold / ||∇L||

Effect: Caps gradient magnitude while preserving direction
Critical for: RNNs, Transformers, very deep networks
```

### Early Stopping: Preventing Overfitting

**The Problem:**
```
Train loss: 0.01 ↓  (getting better)
Val loss:   0.50 ↑  (getting worse!)

Diagnosis: OVERFITTING - memorizing training data
```

**The Solution:**
```python
if val_loss hasn't improved in N epochs:
    stop training
    restore best checkpoint
```

**Why it works:**
- Training loss always decreases (we're optimizing it!)
- Validation loss increases when we overfit
- Best model is where val loss is minimum

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Why gradient descent optimizes neural networks
- How different optimizers work (SGD, Adam, AdamW)
- The role of learning rate and schedules
- Why gradient clipping prevents training instability
- When to stop training (early stopping)
- Training vs evaluation mode behavior

**Practical Skills:**
- Implement complete training loop from scratch
- Add validation for monitoring generalization
- Use checkpointing for fault tolerance
- Implement early stopping automatically
- Apply learning rate scheduling
- Use gradient clipping for stability
- Log and visualize training metrics

---

## 📊 Mathematical Foundations

### The Training Objective

**Empirical Risk Minimization:**
```
min (1/N) Σᵢ L(f(xᵢ; θ), yᵢ) + λR(θ)
 θ

Where:
- L: loss function (data fit)
- R(θ): regularization (e.g., weight decay)
- λ: regularization strength
```

**Why regularization?**
```
Without: Model memorizes training data (overfits)
With: Model learns generalizable patterns
```

### Batch vs Stochastic vs Mini-Batch Gradient Descent

**Batch GD (use all data):**
```
∇L = (1/N) Σᵢ ∇L(xᵢ, yᵢ; θ)

Pros: Exact gradient, stable updates
Cons: Very slow for large datasets, poor generalization
```

**Stochastic GD (one sample):**
```
∇L ≈ ∇L(xᵢ, yᵢ; θ)

Pros: Fast updates, better generalization
Cons: Noisy gradients, unstable
```

**Mini-Batch GD (subset of data):**
```
∇L ≈ (1/B) Σᵢ ∇L(xᵢ, yᵢ; θ)  where B = batch size

Pros: Balance of speed and stability
Cons: Requires tuning batch size
Default: Batch size 32-256 (sweet spot)
```

**Why mini-batches are standard:**
- GPU parallelism (process 32-256 samples simultaneously)
- Noisy gradients help escape local minima
- Faster iteration than full batch
- More stable than single sample

---

## 🔑 Key Concepts

### 1. Train vs Eval Mode

```python
model.train()  # Training behavior
model.eval()   # Inference behavior
```

**What changes?**

**Dropout:**
```
train(): Randomly drop neurons (regularization)
eval():  Use all neurons (deterministic)
```

**BatchNorm:**
```
train(): Use batch statistics (μ_batch, σ²_batch)
eval():  Use running statistics (μ_running, σ²_running)
```

**Why this matters:**
- Training needs stochasticity for regularization
- Inference needs determinism for reproducibility
- Forgetting `.eval()` → wrong predictions!

### 2. Gradient Accumulation

**The Problem:**
```
GPU memory: 16 GB
Batch size 128: 20 GB needed ← Doesn't fit!
```

**The Solution:**
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()

# Effective batch size = 32 × 4 = 128
# Memory usage = 32 samples at a time
```

**Effect:** Train with large batches on small GPU!

### 3. Checkpointing

**Why save checkpoints?**
1. **Training crashes** → Resume from checkpoint
2. **Experiments** → Compare different runs
3. **Production** → Deploy best model
4. **Research** → Share with community

**What to save?**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': best_loss,
    'metrics': metrics
}
torch.save(checkpoint, 'checkpoint.pt')
```

**Best practices:**
- Save best model (lowest val loss)
- Save periodic checkpoints (every N epochs)
- Save final model
- Include all state for resumability

### 4. Metrics and Logging

**What to track?**

**Training metrics:**
- Loss (decreasing = good)
- Accuracy (increasing = good)
- Learning rate (shows schedule)
- Gradient norm (detects explosions)

**Validation metrics:**
- Same as training, but on held-out data
- Early indicator of overfitting

**System metrics:**
- Training time per epoch
- GPU memory usage
- Throughput (samples/second)

---

## 🧪 Exercises

### Exercise 1: Basic Training Loop
**File:** `01_basic_training.py`

**What You'll Learn:**
- The core training loop structure
- Forward pass, loss, backward pass, update
- Train/validation split
- Computing metrics (loss, accuracy)
- Checkpoint saving and loading

**Why It Matters:**
This is the foundation. Every ML system - from research to production - uses this loop. Understanding it deeply means you can:
- Debug training issues
- Implement custom training logic
- Understand research papers
- Build production systems

**Tasks:**
1. Implement `train_one_epoch()` function
2. Implement `validate()` function
3. Create training loop with both train and val
4. Compute and print loss/accuracy each epoch
5. Save best model checkpoint
6. Implement checkpoint loading for resumption

**Expected Behavior:**
```
Epoch [1/10]
  Train Loss: 0.8234 | Train Acc: 65.23%
  Val Loss:   0.7123 | Val Acc:   72.15%
  ✓ Best model saved!

Epoch [2/10]
  Train Loss: 0.6891 | Train Acc: 74.56%
  Val Loss:   0.6234 | Val Acc:   76.89%
  ✓ Best model saved!
```

---

### Exercise 2: Advanced Training Techniques
**File:** `02_advanced_training.py`

**What You'll Learn:**
- Learning rate scheduling (StepLR, CosineAnnealing, ReduceLROnPlateau)
- Gradient clipping for stability
- Early stopping to prevent overfitting
- Mixed precision training (AMP)
- Comprehensive metrics tracking

**Why It Matters:**
These techniques are what separate toy examples from production systems:
- **LR scheduling** → Faster convergence, better final performance
- **Gradient clipping** → Stable training for deep/recurrent models
- **Early stopping** → Automatic overfitting prevention
- **AMP** → 2-3x speedup with minimal code

**Tasks:**
1. Implement `EarlyStopping` class
2. Implement `GradientClipper` class
3. Add learning rate scheduling (all 3 types)
4. Implement mixed precision training
5. Track gradient norms
6. Comprehensive logging

**Key Insights:**
- Different LR schedules for different tasks
- Gradient clipping threshold depends on model depth
- Early stopping patience depends on dataset size
- AMP gives free 2x speedup!

---

## 📝 Design Patterns

### Pattern 1: The Standard Training Loop
```python
def train(model, train_loader, val_loader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        # Train
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = evaluate(model, val_loader)

        # Save if best
        if val_loss < best_loss:
            save_checkpoint(model, optimizer, epoch)
```

### Pattern 2: Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop?
```

### Pattern 3: Learning Rate Warmup + Decay
```python
# Warmup: gradually increase LR
for epoch in range(warmup_epochs):
    lr = base_lr * (epoch + 1) / warmup_epochs
    set_lr(optimizer, lr)

# Decay: gradually decrease LR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## ✅ Solutions

Full implementations in `solution/` directory.

**Learning approach:**
1. Try implementing from scratch
2. Test on simple problem (synthetic data)
3. Debug until it works
4. Compare with solution
5. Understand differences

**Don't just copy code** - the learning is in the debugging!

---

## 🎓 Key Takeaways

1. **Training is optimization** - Find θ that minimizes loss
2. **The loop is simple** - Forward, loss, backward, update
3. **But the details matter** - LR, clipping, early stopping
4. **Validation prevents overfitting** - Never trust training metrics alone
5. **Checkpointing is essential** - For resumability and reproducibility
6. **Logging enables debugging** - Track everything!
7. **These patterns scale** - Same loop trains GPT-4

**The Core Insight:**
```
The training loop is the same from linear regression to GPT-4.
What changes is:
- Model architecture (bigger)
- Dataset size (more data)
- Compute (more GPUs)
- Engineering (distributed training, mixed precision)

But the loop remains: forward, loss, backward, update.
```

---

## 🔗 Connections to Production ML

### Why This Matters at Meta Scale

**Training at Scale:**
- **Feed ranking**: Billions of examples, continuous training
- **Computer vision**: ImageNet-scale datasets (millions of images)
- **NLP**: Trillions of tokens for language models

**Production Requirements:**
- **Fault tolerance** → Checkpointing every hour
- **Monitoring** → Track metrics, detect issues
- **Reproducibility** → Log hyperparameters, random seeds
- **Cost efficiency** → Mixed precision, gradient accumulation

**Real Examples:**
- **LLAMA-2** (Meta): Trained on 2 trillion tokens, thousands of GPUs
- **Feed ranking**: Retrained daily on fresh data
- **Content understanding**: Continuously learning models

---

## 🚀 Next Steps

1. **Complete both exercises** - Basic and advanced
2. **Experiment** - Try different optimizers, schedules
3. **Debug** - Intentionally break things to understand
4. **Move to Lab 4** - Efficient data loading

---

## 💪 Bonus Challenges

1. **Implement Custom Optimizer**
   - Build SGD with momentum from scratch
   - Verify matches PyTorch's implementation

2. **Learning Rate Finder**
   - Implement Leslie Smith's LR range test
   - Plot loss vs learning rate
   - Find optimal LR automatically

3. **Gradient Flow Visualization**
   - Plot gradient magnitudes per layer
   - Detect vanishing/exploding gradients
   - Visualize throughout training

4. **Custom Training Callback System**
   - Build hooks for: epoch start/end, batch start/end
   - Enable modular training extensions
   - Similar to Keras callbacks

5. **Curriculum Learning**
   - Start with easy examples
   - Gradually increase difficulty
   - Observe faster convergence

---

## 📚 Essential Resources

**Papers:**
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Kingma & Ba, 2014
- [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) - Smith, 2015
- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187) - He et al., 2018

**Tutorials:**
- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Fast.ai: Training Loop](https://docs.fast.ai/callback.core.html)

**Tools:**
- [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) - Visualization
- [Weights & Biases](https://wandb.ai) - Experiment tracking

---

## 🤔 Common Pitfalls

### Pitfall 1: Not Zeroing Gradients
```python
# ❌ Gradients accumulate!
for epoch in range(10):
    loss.backward()
    optimizer.step()

# ✓ Zero before each backward
for epoch in range(10):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Pitfall 2: Evaluating in Training Mode
```python
# ❌ Dropout active during eval!
val_loss = validate(model, val_loader)

# ✓ Set eval mode
model.eval()
with torch.no_grad():
    val_loss = validate(model, val_loader)
model.train()
```

### Pitfall 3: Not Tracking Best Model
```python
# ❌ Only save final model (may have overfit!)
train(model, 100)
save(model)

# ✓ Save best validation model
best_loss = float('inf')
for epoch in range(100):
    val_loss = validate(model)
    if val_loss < best_loss:
        best_loss = val_loss
        save(model)
```

### Pitfall 4: Learning Rate Too High/Low
```python
# ❌ Loss explodes or doesn't decrease
optimizer = Adam(model.parameters(), lr=1.0)  # Too high!
optimizer = Adam(model.parameters(), lr=1e-10)  # Too low!

# ✓ Start with reasonable default
optimizer = Adam(model.parameters(), lr=1e-3)  # Good default
```

---

## 💡 Pro Tips

1. **Always validate** - Train metrics lie about generalization
2. **Plot learning curves** - Visualize train/val loss over time
3. **Monitor gradient norms** - Detect vanishing/exploding early
4. **Use warm restarts** - Escape local minima
5. **Save often** - Disk is cheap, lost training is expensive
6. **Log everything** - You never know what you'll need later
7. **Start simple** - Get basic loop working before adding complexity

---

## ✨ You're Ready When...

- [ ] You can implement training loop from scratch
- [ ] You understand when to use train() vs eval()
- [ ] You can explain why gradients must be zeroed
- [ ] You know how learning rate affects training
- [ ] You can implement early stopping
- [ ] You understand gradient clipping
- [ ] You can save and load checkpoints

**Remember:** The training loop is the heartbeat of machine learning. Master it, and you can train any model!
