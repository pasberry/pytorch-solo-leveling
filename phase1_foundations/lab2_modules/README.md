# Lab 2: Custom Neural Network Modules - Building Blocks of Intelligence 🧱

> **Time:** 2-3 hours
> **Difficulty:** Beginner → Intermediate
> **Goal:** Master `nn.Module` and build reusable, composable neural network components

---

## 📖 Why This Lab Matters

In Lab 1, you learned to compute gradients on raw tensors. But imagine building GPT-4 by manually tracking 175 billion parameters and their gradients - impossible!

**The solution:** `nn.Module` - PyTorch's abstraction for building neural networks.

This lab teaches you:
- **How to build modular, reusable components** (like LEGO blocks)
- **How PyTorch manages parameters automatically**
- **How to compose simple modules into complex architectures**
- **Why good abstractions make research and production ML possible**

---

## 🧠 The Big Picture: Why Abstraction Matters

### The Problem: Manual Parameter Management Doesn't Scale

Imagine building a simple 3-layer network manually:
```python
# ❌ Terrible approach
w1 = torch.randn(784, 256, requires_grad=True)
b1 = torch.randn(256, requires_grad=True)
w2 = torch.randn(256, 128, requires_grad=True)
b2 = torch.randn(128, requires_grad=True)
w3 = torch.randn(128, 10, requires_grad=True)
b3 = torch.randn(10, requires_grad=True)

# Forward pass
h1 = x @ w1 + b1
h1 = torch.relu(h1)
h2 = h1 @ w2 + b2
h2 = torch.relu(h2)
y = h2 @ w3 + b3

# Now you need to manually:
# - Track all 6 parameters
# - Initialize them correctly
# - Save/load them
# - Move them to GPU
# - Apply weight decay
# - ...this is insane!
```

### The Solution: `nn.Module`

```python
# ✓ Beautiful, modular, maintainable
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
# PyTorch automatically:
# ✓ Tracks all parameters
# ✓ Initializes weights properly
# ✓ Moves to GPU with one call
# ✓ Saves/loads state
# ✓ Integrates with optimizers
```

**This is the power of abstraction!**

---

## 🔬 Deep Dive: Understanding `nn.Module`

### What is `nn.Module`?

`nn.Module` is the base class for all neural network components in PyTorch. It provides:

1. **Automatic parameter registration** - Any `nn.Parameter` assigned as an attribute is automatically tracked
2. **Hierarchical structure** - Modules can contain other modules (composability)
3. **Hooks** - Intercept forward/backward passes for debugging or custom logic
4. **State management** - Easily save/load models
5. **Device management** - Move entire models between CPU/GPU

### The Lifecycle of a Module

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()  # ← CRITICAL: Initializes parent class
        self.weight = nn.Parameter(torch.randn(10, 10))  # ← Registered!

    def forward(self, x):
        return x @ self.weight
```

**What happens internally:**
1. `__init__` is called → registers all `nn.Parameter` and child `nn.Module` objects
2. `forward()` defines the computation
3. When you call `model(x)`, PyTorch:
   - Calls `model.forward(x)`
   - Builds computational graph (if `requires_grad=True`)
   - Applies any registered hooks
   - Returns the result

### Parameters vs Buffers

**Parameters** (`nn.Parameter`):
- Trainable weights that need gradients
- Updated by optimizers
- Included in `model.parameters()`
- Examples: weights, biases

**Buffers** (`self.register_buffer()`):
- Non-trainable state that needs to be saved
- NOT updated by optimizers
- Examples: running mean/variance in BatchNorm, position encodings

```python
class CustomBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Trainable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Non-trainable buffers
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
```

**Why the distinction?**
- Parameters appear in `model.parameters()` for optimizer
- Buffers are included in `state_dict()` for saving, but not optimized
- Both move to GPU when you call `model.to('cuda')`

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Why abstraction is critical for building complex models
- How `nn.Module` manages parameters and state
- The module hierarchy and composability
- When to use Parameters vs Buffers

**Practical Skills:**
- Build custom layers from scratch (Linear, activations, normalization)
- Implement proper parameter initialization
- Use hooks for debugging and feature extraction
- Compose modules into larger architectures
- Manage model state (save/load, device management)

---

## 📊 Mathematical Foundations

### Linear Layer: The Building Block

**Purpose:** Transform input to new representation

**Mathematics:**
```
y = xW^T + b

Where:
- x: input (batch_size, in_features)
- W: weight matrix (out_features, in_features)
- b: bias vector (out_features,)
- y: output (batch_size, out_features)
```

**Why `W^T`?**
PyTorch stores weights as `(out_features, in_features)` for efficiency:
- Each row is the weight vector for one output neuron
- Easier to think about: "this row produces output neuron i"

**Initialization Matters:**
- **Too small:** Vanishing activations/gradients
- **Too large:** Exploding activations/gradients
- **Just right:** Xavier/Kaiming initialization

**Xavier/Glorot Initialization:**
```
W ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
```
Keeps variance stable across layers.

### Activation Functions: Introducing Nonlinearity

**Why we need them:**
- Without activations, deep networks collapse to single linear layer:
  ```
  y = W3(W2(W1x)) = (W3W2W1)x = W_combined x
  ```
- Activations introduce **nonlinearity**, enabling complex functions

**Common Activations:**

**ReLU** (Rectified Linear Unit):
```
ReLU(x) = max(0, x)

Pros: Simple, fast, no vanishing gradient for x > 0
Cons: Dead neurons (negative inputs always → 0)
```

**GELU** (Gaussian Error Linear Unit):
```
GELU(x) = x × Φ(x)
where Φ(x) is the CDF of standard normal distribution

Pros: Smooth, works well in transformers (GPT, BERT)
Cons: Slightly more expensive
```

**SiLU/Swish** (Sigmoid Linear Unit):
```
SiLU(x) = x × σ(x) = x / (1 + e^(-x))

Pros: Smooth, unbounded above, self-gating
Used in: EfficientNet, mobile models
```

### Normalization: Stabilizing Training

**Problem:** Internal covariate shift
- As we train deeper layers, input distributions change
- Makes learning unstable and slow

**BatchNorm Solution:**
```
y = γ × (x - μ_batch) / √(σ²_batch + ε) + β

Where:
- μ_batch, σ²_batch: mean and variance of current batch
- γ, β: learnable parameters (scale and shift)
- ε: small constant for numerical stability
```

**LayerNorm Solution:** (used in transformers)
```
Normalize across features (not batch)
Better for: Sequences, small batches, RNNs/Transformers
```

---

## 🔑 Key Concepts

### 1. The Module Hierarchy

Modules can contain modules:
```python
model = nn.Sequential(
    nn.Linear(784, 256),      # ← Module
    nn.ReLU(),                # ← Module
    nn.Linear(256, 10)        # ← Module
)
```

This creates a tree structure:
```
Model (nn.Sequential)
├── Linear (in=784, out=256)
│   ├── weight (Parameter)
│   └── bias (Parameter)
├── ReLU
└── Linear (in=256, out=10)
    ├── weight (Parameter)
    └── bias (Parameter)
```

### 2. Forward Hooks: Intercepting Computations

Hooks let you inspect or modify forward/backward passes:

```python
def hook_fn(module, input, output):
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")

model.fc1.register_forward_hook(hook_fn)
```

**Use cases:**
- Feature extraction (get intermediate layer outputs)
- Debugging (check shapes, values)
- Visualization (activation maps)
- Gradient analysis (detect vanishing/exploding)

### 3. State Dict: Model Persistence

```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model.load_state_dict(torch.load('model.pt'))
```

**What's in `state_dict()`?**
- All parameters (weights, biases)
- All buffers (running statistics, etc.)
- Structure info (useful with custom classes)

### 4. Training vs Eval Mode

```python
model.train()  # Enable dropout, batch norm training mode
model.eval()   # Disable dropout, use running stats in batch norm
```

**Critical for:**
- Dropout (needs to be disabled during inference)
- BatchNorm (use running stats, not batch stats during eval)
- Any custom behavior

---

## 🧪 Exercises

### Exercise 1: Build a Custom Linear Layer
**File:** `01_custom_linear.py`

**What You'll Learn:**
- How to implement `nn.Linear` from scratch
- Proper weight initialization (Kaiming Uniform)
- Parameter registration
- The difference between `__init__` and `forward`

**Why It Matters:**
- Understanding the simplest building block deeply
- Learn how PyTorch initializes weights
- Foundation for understanding complex layers (attention, convolutions)

**Tasks:**
1. Implement `CustomLinear` with weight and optional bias
2. Use proper Kaiming initialization
3. Implement `extra_repr()` for nice printing
4. Build an MLP using your custom layers
5. Verify it matches `nn.Linear` output

**Key Insights:**
- Weight shape is `(out_features, in_features)` - why?
- Initialization affects training convergence
- Forward pass is just matrix multiplication + bias

---

### Exercise 2: Custom Activation Functions & Normalization
**File:** `02_custom_activations.py`

**What You'll Learn:**
- Implement ReLU, GELU, SiLU from scratch
- Build custom LayerNorm and BatchNorm
- Understand the math behind normalization
- Training vs eval mode

**Why It Matters:**
- Activations introduce nonlinearity (without them, deep learning is just linear regression!)
- Normalization makes training deep networks possible
- Understanding internals helps debug issues

**Tasks:**
1. Implement `CustomReLU`, `CustomGELU`, `CustomSiLU`
2. Implement `CustomLayerNorm` with learnable parameters
3. Implement `CustomBatchNorm1d` with running statistics
4. Test training vs eval mode differences
5. Compare with PyTorch's built-in implementations

**Key Insights:**
- GELU is smoother than ReLU (better gradients)
- BatchNorm uses different behavior in train/eval mode
- LayerNorm normalizes per sample (better for variable-length sequences)

---

### Exercise 3: Build a Complete Custom Model
**File:** `03_complete_model.py`

**What You'll Learn:**
- Compose modules into architectures
- Implement residual connections
- Use forward hooks for feature extraction
- Custom `state_dict` with metadata
- Model configuration and reproducibility

**Why It Matters:**
This is how you build real models:
- ResNet, Transformers, etc. are all composed modules
- Hooks enable advanced use cases (feature extraction, debugging)
- Proper state management crucial for production

**Tasks:**
1. Build a `CustomBlock` with residual connections
2. Implement `CustomClassifier` with multiple blocks
3. Add forward hooks for activation extraction
4. Implement enhanced `state_dict()` with metadata
5. Save and load configurations
6. Benchmark different architectures

**Key Insights:**
- Residual connections help gradients flow (enables very deep networks)
- Hooks are powerful for introspection
- Good abstractions make experimentation fast

---

## 📝 Design Patterns

### Pattern 1: Residual Connection
```python
class ResBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity  # ← Skip connection
        return out
```
**Why:** Enables training of very deep networks (ResNet, Transformer)

### Pattern 2: Sequential Construction
```python
layers = []
for i in range(num_layers):
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    layers.append(nn.ReLU())
self.network = nn.Sequential(*layers)
```
**Why:** Dynamic architecture construction

### Pattern 3: Feature Extraction Hook
```python
activations = {}
def hook(name):
    def fn(module, input, output):
        activations[name] = output.detach()
    return fn

model.layer1.register_forward_hook(hook('layer1'))
```
**Why:** Extract intermediate features without modifying model

---

## ✅ Solutions

Full implementations in `solution/` - but try implementing yourself first!

**Learning progression:**
1. Try to implement from scratch
2. Get stuck (normal!)
3. Check solution for hints
4. Try again
5. Compare approaches

---

## 🎓 Key Takeaways

1. **`nn.Module` is the foundation** - All PyTorch models inherit from it
2. **Parameters are automatically tracked** - No manual management needed
3. **Modules compose hierarchically** - Build complex from simple
4. **Initialization matters** - Affects training stability
5. **Hooks enable advanced use cases** - Feature extraction, debugging
6. **State dict enables persistence** - Save/load models easily
7. **Train/eval mode is critical** - Affects dropout, batch norm behavior

**The Big Lesson:**
```
Good abstractions make impossible problems tractable.
nn.Module lets us build GPT-4, not just linear regression.
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**Modularity Enables:**
- **Rapid experimentation** - Swap components easily
- **Code reuse** - Share layers across projects
- **Team collaboration** - Clear interfaces
- **Model zoo** - Pre-trained components

**Real Examples:**
- **FAIR's detectron2** - Object detection built from modular components
- **HuggingFace Transformers** - Every model is composed modules
- **TorchVision** - ResNet, VGG, etc. all use `nn.Module` pattern

**At Meta Scale:**
- Feed ranking models: Thousands of features, dozens of layers
- Without modularity: Unmaintainable spaghetti code
- With `nn.Module`: Clean, testable, composable components

---

## 🚀 Next Steps

1. **Complete all exercises** - Don't skip any!
2. **Read the solutions** - Compare your approach
3. **Experiment** - Try different architectures
4. **Move to Lab 3** - Training loops and optimization

---

## 💪 Bonus Challenges

1. **Implement Dropout**
   - Random deactivation during training
   - Identity during eval
   - Understand why it prevents overfitting

2. **Build a ResNet Block**
   - Conv → BatchNorm → ReLU → Conv → BatchNorm → Add skip → ReLU
   - See why residual connections help

3. **Custom Initialization**
   - Implement different schemes (Xavier, Kaiming, Orthogonal)
   - Test which works best for different activations

4. **Attention Mechanism**
   - Build simplified self-attention
   - Foundation for transformers

5. **Model Surgery**
   - Load pre-trained model
   - Replace final layer
   - Fine-tune for new task

---

## 📚 Essential Resources

**Documentation:**
- [nn.Module API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Parameter Initialization](https://pytorch.org/docs/stable/nn.init.html)
- [Hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

**Papers:**
- [Batch Normalization](https://arxiv.org/abs/1502.03167) - Ioffe & Szegedy, 2015
- [Layer Normalization](https://arxiv.org/abs/1607.06450) - Ba et al., 2016
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) - He et al., 2015

**Tutorials:**
- [PyTorch: Defining New autograd Functions](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

---

## 🤔 Common Pitfalls

### Pitfall 1: Forgetting `super().__init__()`
```python
class MyModule(nn.Module):
    def __init__(self):
        # ❌ WRONG - will break parameter registration!
        self.fc = nn.Linear(10, 10)

    def __init__(self):
        super().__init__()  # ✓ CRITICAL!
        self.fc = nn.Linear(10, 10)
```

### Pitfall 2: Using Regular Tensors Instead of Parameters
```python
# ❌ Won't be tracked or trained!
self.weight = torch.randn(10, 10)

# ✓ Proper parameter
self.weight = nn.Parameter(torch.randn(10, 10))
```

### Pitfall 3: Modifying Parameters In-Place
```python
# ❌ Can break autograd
self.weight.data += 0.01

# ✓ Use optimizer to update
optimizer.step()
```

### Pitfall 4: Not Setting train() / eval()
```python
# ❌ Dropout active during inference!
predictions = model(test_data)

# ✓ Disable dropout
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

---

## 💡 Pro Tips

1. **Use `nn.ModuleList` for dynamic layers** - Not regular Python list!
2. **Print models to understand structure** - `print(model)` shows hierarchy
3. **Use `named_parameters()` for debugging** - See all parameters with names
4. **Leverage `nn.Sequential` when possible** - Cleaner for simple stacks
5. **Test modules independently** - Unit test each component
6. **Check `requires_grad`** - Verify trainable parameters

---

## ✨ You're Ready When...

- [ ] You understand why `nn.Module` exists
- [ ] You can build custom layers from scratch
- [ ] You know the difference between Parameters and Buffers
- [ ] You can use hooks for feature extraction
- [ ] You understand initialization strategies
- [ ] You can compose modules into architectures
- [ ] You know when to use train() vs eval()

**Next up:** Lab 3 - Training loops, where you'll put these modules to work!
