# Phase 1: PyTorch Foundations 🎯

> **Goal:** Master PyTorch fundamentals and build your first neural networks

Welcome to Phase 1! This is where you'll build a rock-solid foundation in PyTorch. By the end of this phase, you'll be comfortable with tensors, automatic differentiation, building custom neural network modules, and training models on GPUs.

---

## 📚 What You'll Learn

- ✅ Tensors and their operations
- ✅ Autograd and computational graphs
- ✅ Building custom `nn.Module` layers
- ✅ Training loops from scratch
- ✅ DataLoaders and data pipelines
- ✅ GPU acceleration and mixed precision training
- ✅ Building CNNs for image classification

---

## 🧪 Labs

### Lab 1: Tensors & Autograd (1-2 hours)
**Location:** `lab1_tensors/`

Learn the foundation of PyTorch: tensors and automatic differentiation.

**Topics:**
- Creating and manipulating tensors
- Broadcasting and indexing
- Autograd and gradient computation
- Computational graph visualization
- Custom backward functions

**Deliverable:** Implement linear regression using only tensors and autograd (no `nn` module).

---

### Lab 2: Custom Neural Network Modules (1-2 hours)
**Location:** `lab2_modules/`

Master `nn.Module` and build custom layers.

**Topics:**
- Understanding `nn.Module`
- Building custom layers
- Parameters vs buffers
- Forward hooks and backward hooks
- Module composition

**Deliverable:** Build a custom MLP with flexible architecture.

---

### Lab 3: Training Loop from Scratch (1-2 hours)
**Location:** `lab3_training_loop/`

Learn how to train models properly.

**Topics:**
- Training vs evaluation mode
- Loss functions and optimizers
- Gradient accumulation
- Gradient clipping
- Learning rate scheduling
- Training metrics and logging

**Deliverable:** Complete training loop with validation, early stopping, and checkpointing.

---

### Lab 4: Data Loading Pipeline (1-2 hours)
**Location:** `lab4_data_loading/`

Build efficient data pipelines.

**Topics:**
- `Dataset` and `DataLoader`
- Data transformations and augmentation
- Custom datasets
- Efficient data loading with workers
- Handling imbalanced datasets
- Data preprocessing best practices

**Deliverable:** Custom dataset with augmentation for image classification.

---

### Lab 5: GPU Acceleration & Mixed Precision (1-2 hours)
**Location:** `lab5_gpu_mixed_precision/`

Speed up training with GPUs and mixed precision.

**Topics:**
- Moving models and data to GPU
- Mixed precision training with `torch.cuda.amp`
- Gradient scaling
- Memory optimization techniques
- Batch size tuning
- Introduction to FSDP

**Deliverable:** Accelerate training by 3-5x using GPU and mixed precision.

---

### Lab 6: CNN Image Classifier (2-3 hours)
**Location:** `lab6_cnn_classifier/`

Build a complete CNN for image classification.

**Topics:**
- Convolutional layers and pooling
- Batch normalization and dropout
- Residual connections
- Data augmentation for vision
- Transfer learning basics
- Model evaluation and confusion matrices

**Deliverable:** Train a CNN on CIFAR-10 with >85% accuracy.

---

## 🎯 Checkpoint Project: Complete Image Classification Pipeline

**Location:** `checkpoint_project/`

**Time:** 4-6 hours

Build a production-ready image classification system from scratch:

**Requirements:**
1. Custom CNN architecture (at least 3 conv layers)
2. Data loading with augmentation
3. Training loop with:
   - Validation split
   - Learning rate scheduling
   - Early stopping
   - Model checkpointing
   - TensorBoard logging
4. GPU training with mixed precision
5. Model evaluation with confusion matrix
6. Achieve >85% accuracy on CIFAR-10

**Stretch Goals:**
- Implement a residual network
- Add ensemble predictions
- Export model to ONNX
- Implement gradual unfreezing for transfer learning

**Meta-Scale Considerations:**
- How would you scale this to ImageNet (1000 classes, 1M images)?
- What changes for multi-GPU training?
- How would you handle class imbalance at scale?

---

## 📊 Progress Checklist

Track your progress through Phase 1:

- [ ] Lab 1: Tensors & Autograd
- [ ] Lab 2: Custom Modules
- [ ] Lab 3: Training Loop
- [ ] Lab 4: Data Loading
- [ ] Lab 5: GPU & Mixed Precision
- [ ] Lab 6: CNN Classifier
- [ ] Checkpoint Project
- [ ] Phase 1 Quiz

---

## ✅ Quiz & Assessment

Before moving to Phase 2, complete the quiz in `quiz.md` to verify your understanding.

**Topics Covered:**
- Tensor operations and broadcasting
- Autograd mechanics
- `nn.Module` internals
- Training loop components
- Data loading best practices
- GPU optimization

**Passing Score:** 80% (8/10 questions)

---

## 🚀 Next Steps

Once you've completed all labs, the checkpoint project, and passed the quiz:

1. Review your checkpoint project code
2. Get feedback on your implementation
3. Move to **Phase 2: Intermediate Deep Learning**

---

## 💡 Tips for Success

- **Don't skip labs:** Each builds on the previous one
- **Code along:** Don't just read, implement everything yourself
- **Experiment:** Try variations and break things to learn
- **Ask questions:** Understanding "why" is as important as "how"
- **Review regularly:** Come back to earlier labs to solidify knowledge

---

**Ready to start? Let's dive into Lab 1! 🔥**
