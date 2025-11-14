# Lab 1: Tensors & Autograd 🎯

> **Time:** 1-2 hours
> **Difficulty:** Beginner
> **Goal:** Master PyTorch tensors and automatic differentiation

---

## 📚 Theory Brief (15 mins)

### What are Tensors?

Tensors are the fundamental data structure in PyTorch - think of them as multi-dimensional arrays (like NumPy arrays) but with superpowers:
- Run on GPU for massive speedups
- Track gradients automatically for backpropagation
- Efficient memory management

### What is Autograd?

Autograd is PyTorch's automatic differentiation engine. It:
- Builds a computational graph of your operations
- Automatically computes gradients using the chain rule
- Powers all neural network training in PyTorch

**Key Concept:** When you perform operations on tensors with `requires_grad=True`, PyTorch records the operations in a directed acyclic graph (DAG). When you call `.backward()`, it traverses this graph to compute gradients.

---

## 🎯 Learning Objectives

By the end of this lab, you'll be able to:
- Create and manipulate tensors
- Understand tensor shapes, dtypes, and devices
- Use broadcasting and advanced indexing
- Compute gradients with autograd
- Implement linear regression without using `nn.Module`

---

## 🧪 Exercises

### Exercise 1: Tensor Basics

**File:** `01_tensor_basics.py`

Create tensors in different ways and understand their properties.

```python
import torch

# TODO: Create a 3x3 tensor of zeros
# TODO: Create a 2x4 tensor of random values from normal distribution
# TODO: Create a tensor from a Python list
# TODO: Create a tensor with values from 0 to 9
# TODO: Reshape a tensor from (10,) to (2, 5)
```

---

### Exercise 2: Tensor Operations

**File:** `02_tensor_operations.py`

Perform mathematical operations on tensors.

```python
# TODO: Element-wise addition, multiplication
# TODO: Matrix multiplication (@ operator)
# TODO: Broadcasting example
# TODO: Reduction operations (sum, mean, max)
# TODO: In-place operations
```

---

### Exercise 3: Autograd Basics

**File:** `03_autograd_basics.py`

Learn automatic differentiation.

```python
# TODO: Create tensors with requires_grad=True
# TODO: Perform operations and call .backward()
# TODO: Access gradients with .grad
# TODO: Understand gradient accumulation
# TODO: Use torch.no_grad() for inference
```

---

### Exercise 4: Linear Regression from Scratch

**File:** `04_linear_regression.py`

Implement linear regression using only tensors and autograd.

**Goal:** Fit a line `y = wx + b` to noisy data.

**Steps:**
1. Generate synthetic data: `y = 3x + 2 + noise`
2. Initialize parameters `w` and `b` randomly
3. Define forward pass (prediction)
4. Compute MSE loss
5. Compute gradients with `.backward()`
6. Update parameters manually
7. Repeat for multiple epochs
8. Visualize results

---

## 📝 Starter Code

See the `starter/` directory for templates.

---

## ✅ Solutions

Full implementations are in `solution/` - but try on your own first!

---

## 🎓 Key Takeaways

After completing this lab, you should understand:

1. **Tensors are the foundation** - Everything in PyTorch starts with tensors
2. **Autograd is automatic** - You don't manually compute derivatives
3. **Computational graphs are built dynamically** - PyTorch uses define-by-run
4. **Gradients accumulate** - Always zero them before backward pass
5. **Training is just optimization** - Compute loss, gradients, update parameters

---

## 🚀 Next Steps

Once you've completed all exercises:
1. Review your code
2. Try the bonus challenges below
3. Move to Lab 2: Custom Neural Network Modules

---

## 💪 Bonus Challenges

1. Implement polynomial regression (fit `y = w2*x^2 + w1*x + b`)
2. Visualize the computational graph using torchviz
3. Implement gradient descent with momentum
4. Add regularization (L2 penalty) to linear regression
5. Implement batch gradient descent (not just single examples)

---

## 📚 References

- [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
