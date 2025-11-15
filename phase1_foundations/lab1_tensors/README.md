# Lab 1: Tensors & Autograd - The Foundation of Deep Learning 🎯

> **Time:** 2-3 hours
> **Difficulty:** Beginner
> **Goal:** Master PyTorch tensors and automatic differentiation from first principles

---

## 📖 Why This Lab Matters

Before you can build neural networks, you must understand the two fundamental pillars of modern deep learning:

1. **Tensors** - The language of data in deep learning
2. **Automatic Differentiation** - The engine that makes learning possible

This lab teaches you the foundational concepts that power everything from GPT-4 to Stable Diffusion. Understanding these deeply will make everything else in the curriculum make sense.

---

## 🧠 The Big Picture: Why Tensors?

### The Problem
Neural networks process enormous amounts of data (images with millions of pixels, text with thousands of tokens, videos with billions of values). We need a data structure that:
- Can represent multi-dimensional data efficiently
- Runs blazingly fast on GPUs
- Tracks mathematical operations for learning
- Provides a unified interface for all data types

### The Solution: Tensors

A **tensor** is a generalization of matrices to N dimensions:
- **Scalar** (0D tensor): A single number `5.0`
- **Vector** (1D tensor): Array of numbers `[1, 2, 3]`
- **Matrix** (2D tensor): Table of numbers (like a spreadsheet)
- **3D Tensor**: Cube of numbers (e.g., RGB image: height × width × channels)
- **4D Tensor**: Batch of images (batch × channels × height × width)
- **5D Tensor**: Video (batch × time × channels × height × width)

### Real-World Example
Consider processing a batch of images for ImageNet classification:
```
Shape: (32, 3, 224, 224)
       ↑   ↑   ↑     ↑
       |   |   |     └─ Width: 224 pixels
       |   |   └─────── Height: 224 pixels
       |   └─────────── Channels: RGB (3)
       └─────────────── Batch size: 32 images
```

This single tensor contains **32 × 3 × 224 × 224 = 4,816,896 numbers**. Tensors let us manipulate all these values efficiently with simple operations.

---

## 🔬 Deep Dive: Automatic Differentiation

### The Central Problem of Deep Learning

**Training a neural network is an optimization problem:**
1. We have a model with parameters θ (weights and biases)
2. We have a loss function L(θ) that measures how bad the model is
3. We want to find θ* that minimizes L(θ)

**The solution is gradient descent:**
```
θ_new = θ_old - α * ∇L(θ)
                    ↑
                    gradient (direction of steepest increase)
```

### Why Automatic Differentiation is Revolutionary

Before autodiff, you had two options:

**Option 1: Manual Derivatives** ❌
- Write out the math by hand for every model
- Error-prone and tedious
- Doesn't scale to modern architectures

**Option 2: Numerical Differentiation** ❌
```python
# Approximate derivative using finite differences
f'(x) ≈ [f(x + h) - f(x)] / h
```
- Slow (requires multiple forward passes)
- Numerically unstable
- Doesn't scale to millions of parameters

**Option 3: Automatic Differentiation** ✅
- Exact gradients (not approximations)
- Fast (single backward pass)
- Automatic (works for any computation)
- Efficient (O(1) overhead per operation)

### How Autograd Works: The Computational Graph

PyTorch builds a **computational graph** dynamically as you run code:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2        # y = x²
z = 3 * y + 5     # z = 3y + 5 = 3x² + 5
z.backward()      # Compute ∂z/∂x
```

**Computational Graph Built:**
```
x (2.0) ──► power(2) ──► y (4.0) ──► multiply(3) ──► add(5) ──► z (17.0)
   ↑                        ↑                            ↑
   └────────────────────────┴────────────────────────────┘
          Backward pass computes: ∂z/∂y, ∂y/∂x
```

**Forward Pass:** Compute values
**Backward Pass:** Compute gradients using chain rule

### The Chain Rule: The Magic Behind Backpropagation

Given `z = f(g(x))`, the derivative is:
```
dz/dx = (dz/dg) × (dg/dx)
```

For our example:
```
z = 3x² + 5
dz/dx = d/dx(3x² + 5) = 6x

Using chain rule:
z = 3y + 5,  y = x²
dz/dy = 3
dy/dx = 2x
dz/dx = (dz/dy)(dy/dx) = 3 × 2x = 6x

At x=2: dz/dx = 12 ✓
```

PyTorch computes this automatically!

---

## 🎯 Learning Objectives

By the end of this lab, you'll understand:

**Theoretical Understanding:**
- Why tensors are the fundamental data structure for ML
- How automatic differentiation works mathematically
- The computational graph and chain rule
- Why gradient descent can train neural networks

**Practical Skills:**
- Create and manipulate tensors efficiently
- Use broadcasting to write vectorized code
- Track gradients with autograd
- Implement gradient descent from scratch
- Build linear regression without any `nn` module

---

## 📊 Mathematical Foundations

### Linear Regression: The Simplest Learning Problem

**Goal:** Fit a line through data points

**Model:** `y = wx + b`
- `w` = weight (slope)
- `b` = bias (y-intercept)
- `x` = input feature
- `y` = prediction

**Loss Function (Mean Squared Error):**
```
L(w, b) = (1/N) Σ (y_pred - y_true)²
        = (1/N) Σ (wx + b - y_true)²
```

**Gradients:**
```
∂L/∂w = (2/N) Σ (wx + b - y_true) × x
∂L/∂b = (2/N) Σ (wx + b - y_true)
```

**Update Rule:**
```
w ← w - α(∂L/∂w)
b ← b - α(∂L/∂b)
```

PyTorch computes these gradients automatically!

### Why Start with Linear Regression?

1. **Simplest non-trivial learning problem** - One weight, one bias
2. **Demonstrates core concepts** - Loss, gradients, optimization
3. **Same principles scale** - Deep networks use the same gradient descent
4. **Debuggable** - You can visualize and verify everything

Even GPT-4 is trained using the same fundamental algorithm (gradient descent on a loss function) - just with billions of parameters instead of two!

---

## 🔑 Key Concepts

### 1. Tensor Properties

Every tensor has:
- **Shape**: Dimensions `(batch, channels, height, width)`
- **Dtype**: Data type `torch.float32, torch.int64, torch.bool`
- **Device**: Where it lives `cpu, cuda:0, cuda:1`
- **Requires Grad**: Whether to track gradients `True/False`

### 2. Broadcasting Rules

Broadcasting allows operations on tensors of different shapes:

```python
x = torch.randn(3, 1)     # Shape: (3, 1)
y = torch.randn(1, 4)     # Shape: (1, 4)
z = x + y                 # Shape: (3, 4) ← broadcast!
```

**Rules:**
1. If ranks differ, prepend 1s to smaller tensor's shape
2. Dimensions are compatible if they're equal or one is 1
3. The dimension with size 1 is stretched

**Example:**
```
x: (3, 1)
y: (1, 4)
→ Result: (3, 4)

How: x's second dim (1) broadcasts to 4
     y's first dim (1) broadcasts to 3
```

### 3. Gradient Accumulation

**Critical Detail:** Gradients accumulate by default!

```python
loss1.backward()  # Gradients computed
loss2.backward()  # Gradients ADDED to existing ones!
```

**Why?** Useful for:
- Accumulating gradients over batches
- Computing higher-order derivatives

**Must remember:** Always zero gradients before each optimization step:
```python
optimizer.zero_grad()  # or w.grad.zero_()
```

### 4. Detaching from the Graph

Sometimes you don't want gradients:

```python
with torch.no_grad():
    # No computational graph built here
    # Used for inference/evaluation

x.detach()  # Remove x from computational graph
```

---

## 🧪 Exercises

### Exercise 1: Tensor Basics
**File:** `01_tensor_basics.py`

**What You'll Learn:**
- Tensor creation methods
- Understanding shapes and reshaping
- Device management (CPU vs GPU)
- Memory efficiency

**Why It Matters:**
Proper tensor manipulation is crucial for:
- Preparing data for neural networks
- Debugging shape mismatches
- Writing efficient GPU code
- Managing memory in large-scale training

```python
import torch

# TODO: Create a 3x3 tensor of zeros
# TODO: Create a 2x4 tensor of random values from normal distribution
# TODO: Create a tensor from a Python list
# TODO: Create a tensor with values from 0 to 9
# TODO: Reshape a tensor from (10,) to (2, 5)
# TODO: Move a tensor to GPU (if available)
```

---

### Exercise 2: Tensor Operations
**File:** `02_tensor_operations.py`

**What You'll Learn:**
- Element-wise vs matrix operations
- Broadcasting in practice
- Reduction operations
- In-place vs out-of-place operations

**Why It Matters:**
- Neural network forward passes are sequences of tensor operations
- Broadcasting eliminates loops (vectorization)
- Understanding operations helps debug model architectures

```python
# TODO: Element-wise addition, multiplication
# TODO: Matrix multiplication (@ operator vs * operator)
# TODO: Broadcasting example (add vector to matrix)
# TODO: Reduction operations (sum, mean, max along dimensions)
# TODO: In-place operations (add_ vs add)
```

**Key Insight:**
```python
# Different operations!
A * B      # Element-wise multiplication
A @ B      # Matrix multiplication (dot product)
```

---

### Exercise 3: Autograd Basics
**File:** `03_autograd_basics.py`

**What You'll Learn:**
- Creating tensors that track gradients
- The backward pass
- Accessing and interpreting gradients
- Gradient accumulation behavior

**Why It Matters:**
This is the engine of deep learning. Understanding autograd deeply means:
- You can debug gradient issues (vanishing/exploding gradients)
- You can implement custom training loops
- You understand what PyTorch does under the hood

```python
# TODO: Create tensors with requires_grad=True
# TODO: Perform operations and call .backward()
# TODO: Access gradients with .grad
# TODO: Understand gradient accumulation
# TODO: Use torch.no_grad() for inference
```

**Mental Model:**
```
Every operation builds a recipe for computing gradients
.backward() executes that recipe in reverse
```

---

### Exercise 4: Linear Regression from Scratch
**File:** `04_linear_regression.py`

**What You'll Learn:**
- The complete training loop
- Loss computation
- Gradient-based optimization
- Manual parameter updates
- Why we need optimizers

**Why It Matters:**
This is machine learning distilled to its essence:
1. Make predictions (forward pass)
2. Measure error (loss)
3. Compute how to improve (gradients)
4. Update parameters (optimization step)
5. Repeat

Every neural network, from ResNet to GPT-4, follows this same loop!

**Steps:**
1. Generate synthetic data: `y = 3x + 2 + noise`
2. Initialize parameters `w` and `b` randomly
3. Define forward pass: `y_pred = w * x + b`
4. Compute MSE loss: `L = mean((y_pred - y_true)²)`
5. Compute gradients with `.backward()`
6. Update parameters: `w -= learning_rate * w.grad`
7. Zero gradients: `w.grad.zero_()`
8. Repeat for multiple epochs
9. Visualize the fitted line

**Expected Outcome:**
Your model should discover `w ≈ 3` and `b ≈ 2` automatically!

---

## 📝 Starter Code

See the `starter/` directory for templates with detailed comments.

---

## ✅ Solutions

Full implementations are in `solution/` - but **try on your own first!**

The struggle is where the learning happens. Debugging why your gradients are wrong teaches more than reading correct code.

---

## 🎓 Key Takeaways

After completing this lab, you should understand:

1. **Tensors are the universal data structure** - From scalars to videos, everything is a tensor
2. **Autograd eliminates calculus** - You describe the computation, PyTorch computes derivatives
3. **Computational graphs enable backpropagation** - Dynamic graph construction is PyTorch's superpower
4. **Gradients accumulate** - Always zero them before each backward pass
5. **Training = loss + gradients + updates** - This loop scales from linear regression to GPT-4
6. **Broadcasting eliminates loops** - Write vectorized code for efficiency
7. **The chain rule connects everything** - Autograd applies it automatically

**The Fundamental Insight:**
```
Neural network training =
    Compute loss → Compute gradients → Update parameters → Repeat
```

This simple loop, applied billions of times, creates intelligence.

---

## 🔗 Connections to Production ML

### Why This Matters at Meta Scale

**Tensor Efficiency:**
- Facebook processes billions of images daily
- Efficient tensor operations = lower compute costs
- GPU utilization = millions in savings

**Autograd Enables:**
- Rapid experimentation (no manual derivatives)
- Complex custom architectures
- Research to production quickly

**Gradient Descent Powers:**
- Ranking models (feed, ads, search)
- Computer vision (face recognition, content moderation)
- NLP (translation, content understanding)
- Recommendation systems

---

## 🚀 Next Steps

Once you've completed all exercises:

1. **Review your code** - Can you explain every line?
2. **Try the bonus challenges** (below)
3. **Reflect on what you learned** - Write down 3 key insights
4. **Move to Lab 2** - Custom Neural Network Modules

---

## 💪 Bonus Challenges

1. **Polynomial Regression**
   - Fit `y = w2*x² + w1*x + b`
   - Requires computing gradients for 3 parameters
   - Visualize how the curve fits the data

2. **Visualize the Computational Graph**
   - Use `torchviz` to visualize the graph
   - See how operations connect
   - Understand the backward pass visually

3. **Gradient Descent with Momentum**
   - Implement: `v = β*v + (1-β)*grad; θ -= α*v`
   - Compare convergence speed to vanilla GD
   - See why momentum helps

4. **L2 Regularization**
   - Add penalty: `L = MSE + λ(w² + b²)`
   - Observe how it prevents overfitting
   - Tune λ and observe effect on solution

5. **Batch Gradient Descent**
   - Process multiple examples at once
   - Use broadcasting for efficiency
   - Compare batch sizes: 1, 32, 128

6. **Convergence Analysis**
   - Plot loss vs epochs
   - Try different learning rates: 0.001, 0.01, 0.1, 1.0
   - Observe divergence when α is too large

---

## 📚 Additional Resources

**Essential Reading:**
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/notes/autograd.html)
- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) by Andrej Karpathy
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

**Video Lectures:**
- [3Blue1Brown: Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) - Visual intuition
- [MIT 6.S191: Intro to Deep Learning](https://www.youtube.com/watch?v=njKP3FqW3Sk)

**Interactive:**
- [Tensorflow Playground](https://playground.tensorflow.org/) - Visualize gradient descent
- [Distill.pub: Momentum](https://distill.pub/2017/momentum/) - Interactive optimization

---

## 🤔 Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Zero Gradients
```python
# ❌ Wrong
for epoch in range(100):
    loss.backward()
    # Gradients accumulate!

# ✓ Correct
for epoch in range(100):
    optimizer.zero_grad()  # or w.grad.zero_()
    loss.backward()
```

### Pitfall 2: Mixing In-Place Operations with Autograd
```python
# ❌ Can break autograd
x.add_(1)  # In-place modification
loss.backward()  # May fail!

# ✓ Safe
x = x + 1  # Out-of-place
```

### Pitfall 3: Shape Mismatches
```python
# ❌ Common error
x = torch.randn(32, 10)
y = torch.randn(10)
z = x @ y  # Error! Dimension mismatch

# ✓ Correct
z = x @ y.unsqueeze(1)  # Now (32, 10) @ (10, 1) = (32, 1)
```

### Pitfall 4: Not Detaching During Inference
```python
# ❌ Wastes memory
predictions = model(x)  # Builds graph even during eval!

# ✓ Efficient
with torch.no_grad():
    predictions = model(x)  # No graph, saves memory
```

---

## 💡 Pro Tips

1. **Always check shapes** - Most bugs are shape mismatches
2. **Use `.item()` for scalars** - Extract Python number from tensor
3. **Prefer matrix ops over loops** - 10-100x faster on GPU
4. **Use `torch.no_grad()` for inference** - Saves memory
5. **Name your tensors** - Helps with debugging
6. **Start simple** - Test on small data first

---

## ✨ You're Ready When...

- [ ] You can explain what a tensor is and why it's used
- [ ] You understand how automatic differentiation works
- [ ] You can describe the chain rule intuitively
- [ ] You've implemented linear regression from scratch
- [ ] You can debug gradient-related errors
- [ ] You understand the training loop

**Remember:** This is the foundation everything else is built on. Take your time and truly understand these concepts!
