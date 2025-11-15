# Phase 2: Intermediate Deep Learning 🚀

> **Goal:** Master modern architectures and advanced training techniques

Welcome to Phase 2! You'll learn to build Transformers from scratch, implement advanced training techniques, and prepare for production deployment.

---

## 📚 What You'll Learn

- ✅ Transformers from scratch (self-attention, positional encoding)
- ✅ Advanced attention mechanisms (multi-head, cross-attention)
- ✅ Loss functions and optimization strategies
- ✅ Checkpointing and model serialization
- ✅ Distributed training (DDP, FSDP)
- ✅ Inference optimization (TorchScript, ONNX)
- ✅ Model quantization and pruning

---

## 🧪 Labs

### Lab 1: Self-Attention Mechanism (2-3 hours)
**Location:** `lab1_attention/`

Build the foundation of Transformers: self-attention.

**Topics:**
- Scaled dot-product attention
- Multi-head attention
- Causal masking for decoders
- Padding masks
- Cross-attention
- Attention visualization

**Deliverable:** Implement attention from scratch with visualizations.

---

### Lab 2: Transformer Encoder (3-4 hours)
**Location:** `lab2_transformer_encoder/`

Build a complete Transformer encoder.

**Topics:**
- Positional encoding (sinusoidal)
- Encoder layers with residual connections
- Layer normalization
- Feed-forward networks
- Complete encoder stack
- Classification head

**Deliverable:** Build a Transformer encoder for text classification.

---

### Lab 3: Full Transformer (4-5 hours)
**Location:** `lab3_transformer_full/`

Implement the complete Transformer (encoder-decoder).

**Topics:**
- Decoder with masked self-attention
- Cross-attention mechanism
- Full sequence-to-sequence model
- Greedy decoding
- Beam search (optional)
- Training loop for translation

**Deliverable:** Complete Transformer for machine translation.

---

### Lab 4: Custom Loss Functions (2 hours)
**Location:** `lab4_custom_losses/`

Implement advanced loss functions.

**Topics:**
- Label smoothing
- Focal loss
- Contrastive loss
- Triplet loss
- Custom loss design
- Loss weighting strategies

**Deliverable:** Implement and compare different loss functions.

---

### Lab 5: Distributed Training (3-4 hours)
**Location:** `lab5_distributed_training/`

Train models across multiple GPUs.

**Topics:**
- DataParallel (DP)
- DistributedDataParallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- Gradient accumulation
- Mixed precision training
- Multi-node setup

**Deliverable:** Train a model on multiple GPUs with DDP/FSDP.

---

### Lab 6: Model Export & Optimization (2-3 hours)
**Location:** `lab6_model_export/`

Optimize models for production.

**Topics:**
- TorchScript (tracing and scripting)
- ONNX export
- Model quantization (dynamic, static)
- Pruning
- Latency benchmarking
- Model comparison

**Deliverable:** Export and optimize a model for inference.

---

## 🎯 Checkpoint Project: Transformer Text Classifier

**Time:** 6-8 hours

Build a complete Transformer-based text classification system:

**Requirements:**
1. Transformer encoder from scratch
2. Positional encoding
3. Multi-head attention
4. Training with distributed setup (DDP)
5. Model export (ONNX or TorchScript)
6. Achieve >85% accuracy on IMDB sentiment analysis

**Stretch Goals:**
- Implement learning rate warmup
- Add gradient clipping
- Use mixed precision training
- Export optimized model with quantization

---

## 📊 Progress Checklist

- [ ] Lab 1: Self-Attention
- [ ] Lab 2: Transformer Encoder
- [ ] Lab 3: Full Transformer
- [ ] Lab 4: Custom Losses
- [ ] Lab 5: Distributed Training
- [ ] Lab 6: Model Export
- [ ] Checkpoint Project
- [ ] Phase 2 Quiz

---

## ✅ Success Criteria

You've successfully completed Phase 2 when you can:
- [ ] Implement self-attention from scratch
- [ ] Build a complete Transformer encoder
- [ ] Implement encoder-decoder architecture
- [ ] Train models on multiple GPUs
- [ ] Export models for production
- [ ] Optimize inference latency

---

## 🚀 Next Steps

After completing Phase 2:
1. Review checkpoint project
2. Get feedback on implementation
3. Move to **Phase 3: Applied Meta-Scale Projects**

---

**Ready to master Transformers? Let's dive in! 🔥**
