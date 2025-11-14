# Project 2: Reels & Video Understanding 🎬

> **Time:** 1.5-2 weeks
> **Difficulty:** Advanced
> **Goal:** Build a video classification/recommendation model for short-form video (like Reels/TikTok)

---

## 📚 Project Overview

Build a complete video understanding system that classifies videos and generates embeddings for recommendation. This simulates the core technology behind Instagram Reels, TikTok, and YouTube Shorts.

**Real-world application:** Power video recommendations, content moderation, and auto-captioning for billions of short videos.

---

## 🎯 Learning Objectives

By the end of this project, you'll understand:
- Video data representation and preprocessing
- Temporal modeling techniques (3D CNN, Transformers, RNN)
- Video classification and embedding generation
- Transfer learning for video
- Handling variable-length video sequences
- Production considerations for video ML

---

## 🎨 Architecture Options

You'll choose ONE of three architectures (or implement all three to compare):

### Option 1: 3D CNN (C3D / R(2+1)D)
**Best for:** Efficient spatiotemporal feature learning

```
Video (T x H x W x C)
    ↓
3D Convolutions (spatial + temporal)
    ↓
3D Pooling
    ↓
Fully Connected Layers
    ↓
Classification / Embeddings
```

**Pros:**
- Fast inference
- Good for short clips
- Mature architecture

**Cons:**
- Fixed temporal length
- Limited long-range modeling

---

### Option 2: TimeSformer (Video Transformer)
**Best for:** Long-range temporal dependencies

```
Video (T x H x W x C)
    ↓
Patch Extraction (per frame)
    ↓
Patch Embeddings + Positional Encoding
    ↓
Divided Space-Time Attention
    ↓
Transformer Layers
    ↓
Classification / Embeddings
```

**Pros:**
- Captures long-range dependencies
- State-of-the-art accuracy
- Flexible temporal length

**Cons:**
- Computationally expensive
- Requires more data

---

### Option 3: CNN + LSTM Hybrid
**Best for:** Balance between efficiency and performance

```
Video (T x H x W x C)
    ↓
Per-Frame CNN (ResNet/EfficientNet)
    ↓
Frame Embeddings (T x D)
    ↓
Bidirectional LSTM
    ↓
Temporal Pooling (mean/max)
    ↓
Classification / Embeddings
```

**Pros:**
- Good balance of speed and accuracy
- Handles variable length
- Easy to implement

**Cons:**
- Sequential processing (can't parallelize as easily)
- Limited compared to Transformers

---

## 📊 Dataset

### Primary Dataset: Kinetics-400 (Subset)

**Task:** Multi-label video classification
- **Classes:** 400 human action classes (subset: 50 classes)
- **Videos:** ~240K training, ~20K validation (we'll use 10K for training)
- **Duration:** 10 seconds per video
- **Resolution:** 256×256
- **FPS:** 25-30

**Sample Classes:**
- playing_basketball
- cooking
- dancing
- swimming
- playing_guitar
- etc.

### Alternative: Custom Synthetic Dataset
If Kinetics is too large, use our synthetic dataset:
- 10K short videos
- 20 action categories
- Pre-extracted features available

---

## 🚀 Milestones

### Milestone 1: Video Data Pipeline (Week 1, Days 1-3)
**Goal:** Build efficient video loading and preprocessing

**Tasks:**
- [ ] Download dataset (Kinetics subset or synthetic)
- [ ] Implement video decoder (decord or torchvision)
- [ ] Frame extraction and sampling strategies
- [ ] Video augmentation (temporal crop, spatial crop, flip, color jitter)
- [ ] Efficient DataLoader with prefetching
- [ ] Normalize frames (ImageNet stats)

**Deliverables:**
- `data.py` - Video dataset class
- `transforms.py` - Video augmentation
- Data exploration notebook

**Code Structure:**
```python
class VideoDataset(Dataset):
    def __init__(self, annotations, num_frames=16, transform=None):
        """
        Args:
            num_frames: Number of frames to sample
            transform: Video augmentation pipeline
        """
        pass

    def __getitem__(self, idx):
        # Load video
        # Sample frames (uniform, random, dense)
        # Apply transforms
        # Return: (T, C, H, W), label
        pass
```

**Frame Sampling Strategies:**
1. **Uniform:** Sample frames evenly across video
2. **Random:** Random start + consecutive frames
3. **Dense:** All frames (for short videos)

---

### Milestone 2: Model Architecture (Week 1, Days 4-5)
**Goal:** Implement video model

**Choose and implement ONE:**

#### Option A: 3D CNN (R(2+1)D)
```python
class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes):
        # Spatial convolution (1x3x3)
        # Temporal convolution (3x1x1)
        # Pooling and FC layers
        pass

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Apply 3D convs
        # Pool spatiotemporally
        # Classify
        return logits, embeddings
```

#### Option B: TimeSformer
```python
class TimeSformer(nn.Module):
    def __init__(self, num_frames=16, num_classes=50):
        # Patch embedding
        # Positional encoding (spatial + temporal)
        # Divided attention blocks
        # Classification head
        pass

    def forward(self, x):
        # x: (B, T, C, H, W)
        # Extract patches
        # Add positional encodings
        # Self-attention (space then time)
        # Classify
        return logits, embeddings
```

#### Option C: CNN-LSTM
```python
class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        # Pretrained CNN (ResNet-18)
        # BiLSTM for temporal modeling
        # Attention pooling (optional)
        # Classification head
        pass

    def forward(self, x):
        # x: (B, T, C, H, W)
        # Extract per-frame features with CNN
        # LSTM over time
        # Pool temporal features
        # Classify
        return logits, embeddings
```

**Deliverables:**
- `model.py` - Complete architecture
- Unit tests
- Model size and FLOPs analysis

---

### Milestone 3: Training (Week 2, Days 1-2)
**Goal:** Train video model

**Tasks:**
- [ ] Implement training loop
- [ ] Multi-label classification loss
- [ ] Learning rate scheduling (cosine annealing)
- [ ] Gradient clipping
- [ ] Mixed precision training
- [ ] Checkpointing
- [ ] TensorBoard/WandB logging

**Deliverables:**
- `train.py` - Training script
- Config file for hyperparameters
- Training logs

**Training Tips:**
- Start with pretrained weights (ImageNet for CNN backbone)
- Use aggressive data augmentation
- Gradient accumulation for larger effective batch size
- Learning rate warmup

---

### Milestone 4: Evaluation (Week 2, Days 3-4)
**Goal:** Evaluate video understanding

**Tasks:**
- [ ] Implement top-1 and top-5 accuracy
- [ ] Per-class accuracy analysis
- [ ] Confusion matrix
- [ ] Embedding quality (t-SNE visualization)
- [ ] Temporal robustness analysis
- [ ] Inference speed benchmarking

**Deliverables:**
- `evaluate.py` - Evaluation script
- Performance report
- Error analysis notebook

**Key Metrics:**
1. **Top-1 Accuracy:** % correct predictions
2. **Top-5 Accuracy:** % times correct label in top 5
3. **Per-class Accuracy:** Identify weak classes
4. **Inference Time:** FPS on CPU/GPU

---

### Milestone 5: Production Optimization (Week 2, Day 5)
**Goal:** Optimize for production deployment

**Tasks:**
- [ ] Model quantization (INT8)
- [ ] ONNX export
- [ ] TensorRT optimization (if GPU)
- [ ] Batch inference
- [ ] Feature caching
- [ ] Video embedding generation pipeline

**Deliverables:**
- Optimized model (quantized)
- Inference server (FastAPI)
- Benchmark report

---

## 🎯 Stretch Goals

### 1. Pretrain on Kinetics-400 (Full Dataset)
- Train on full Kinetics-400 (240K videos)
- Compare transfer learning to training from scratch
- Analyze what the model learns at each layer

### 2. Multi-Task Learning
Add auxiliary tasks:
- Action localization (temporal)
- Video captioning
- Audio classification (if audio available)

### 3. Self-Supervised Pretraining
- Frame order prediction
- Clip contrastive learning
- Masked frame prediction

### 4. Temporal Action Detection
Not just classification, but detect WHEN actions occur:
- Temporal proposals
- Action boundary detection

---

## 🏭 Meta-Scale Considerations

### 1. Data Scale
**Challenge:** Billions of videos uploaded daily

**Solutions:**
- Streaming data pipeline
- Distributed video decoding
- On-the-fly frame extraction
- Efficient storage (compressed frames)

### 2. Compute Cost
**Challenge:** Video models are expensive (3D convs, many frames)

**Solutions:**
- Two-stage approach: Cheap classifier → Expensive model (only if needed)
- Fewer frames (8 instead of 16)
- Lower resolution (112×112 instead of 224×224)
- Knowledge distillation (large model → small model)

### 3. Latency Requirements
**Challenge:** Real-time video understanding for moderation

**Solutions:**
- Model quantization
- Early exit networks (stop if confident)
- Cascade models (fast → slow)
- GPU batching

### 4. Video Recommendation
**Challenge:** Recommend from millions of videos

**Solutions:**
- Extract video embeddings offline
- Use approximate nearest neighbors (FAISS)
- Two-tower architecture (user tower, video tower)
- Real-time scoring of candidate videos

### 5. Content Moderation
**Challenge:** Detect violating content in real-time

**Solutions:**
- Prioritize sensitive categories (violence, nudity)
- Human-in-the-loop for edge cases
- Ensemble models for robustness
- Explain predictions (attention maps)

### 6. Multi-Modal Understanding
**Challenge:** Video = visual + audio + text (captions)

**Solutions:**
- Separate encoders for each modality
- Fusion strategies (early, late, hybrid)
- Contrastive learning across modalities

---

## 📊 Expected Results

After training, your model should achieve:
- **Top-1 Accuracy:** > 60% (Kinetics-50 subset)
- **Top-5 Accuracy:** > 85%
- **Inference Speed:** > 30 FPS (GPU), > 5 FPS (CPU)
- **Model Size:** < 100MB (after quantization)

**Architecture Comparison:**
| Model | Accuracy | Speed (FPS) | Params |
|-------|----------|-------------|---------|
| R(2+1)D | ~65% | 40 | 30M |
| TimeSformer | ~70% | 15 | 120M |
| CNN-LSTM | ~62% | 35 | 25M |

---

## 📚 Resources

**Papers:**
- [C3D: Learning Spatiotemporal Features (Facebook, 2014)](https://arxiv.org/abs/1412.0767)
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Facebook, 2018)](https://arxiv.org/abs/1711.11248)
- [TimeSformer: Is Space-Time Attention All You Need? (Facebook, 2021)](https://arxiv.org/abs/2102.05095)
- [SlowFast Networks (Facebook, 2019)](https://arxiv.org/abs/1812.03982)

**Code:**
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo)
- [MMAction2](https://github.com/open-mmlab/mmaction2)

---

## ✅ Success Criteria

You've successfully completed this project when you can:
- [ ] Load and preprocess video data efficiently
- [ ] Implement a video classification model
- [ ] Train with proper augmentation and optimization
- [ ] Achieve target accuracy metrics
- [ ] Generate video embeddings
- [ ] Optimize for production (quantization, export)
- [ ] Discuss Meta-scale video challenges

---

## 🤝 Getting Started

1. Choose your architecture (3D CNN, TimeSformer, or CNN-LSTM)
2. Set up data pipeline
3. Implement model from scratch
4. Train and iterate
5. Optimize for production
6. Compare with other architectures (optional)

**Ready to understand videos like Instagram Reels? Let's go! 🎬**
