# Project 1: Feed Ranking Model 🎯

> **Time:** 2-3 weeks
> **Difficulty:** Advanced
> **Goal:** Build a production-grade feed ranking system like Meta's News Feed

---

## 📚 Project Overview

Build a complete feed ranking system that predicts user engagement with posts. This project simulates how Meta ranks billions of posts daily for billions of users.

**Real-world application:** This is the core technology behind Facebook/Instagram feed ranking, recommending which posts users see first.

---

## 🎯 Learning Objectives

By the end of this project, you'll understand:
- Wide & Deep neural network architecture
- Embedding-based feature engineering
- Multi-task learning (CTR + dwell time)
- Negative sampling strategies
- Feature normalization and handling
- Large-scale ranking metrics (NDCG, MAP, AUC)
- Production considerations for billion-scale systems

---

## 📊 Architecture Overview

```
Feed Ranking Model (Wide & Deep)

Input Features:
├── User Features (ID, demographics, history)
├── Post Features (ID, type, author, content)
└── Context Features (time, device, location)

Model Architecture:
├── Deep Component:
│   ├── User Embedding (dim=128)
│   ├── Post Embedding (dim=128)
│   ├── Dense Features → MLP
│   └── Feature Interactions → MLP(512 → 256 → 128)
│
├── Wide Component:
│   └── Cross Features → Linear Layer
│
└── Combiner:
    └── [Deep, Wide] → Output Heads

Output Heads (Multi-task):
├── CTR Head: Binary classification (click or not)
└── Dwell Time Head: Regression (time spent)
```

---

## 🎓 Theory Brief (30 mins)

### 1. Wide & Deep Architecture

**Wide Component:** Memorizes specific feature combinations
- Linear model on cross-product features
- Good for memorization (e.g., "user123 always clicks posts from friend456")

**Deep Component:** Generalizes through embeddings
- Deep neural network with embeddings
- Good for generalization (e.g., "users who like sports tend to click sport posts")

**Why Both?**
- Wide: Handles specific user-item interactions (memorization)
- Deep: Captures complex patterns (generalization)
- Together: Best of both worlds

### 2. Multi-Task Learning

Train one model for multiple objectives:
- **Task 1:** CTR (Click-Through Rate) - Binary classification
- **Task 2:** Dwell Time - Regression (seconds spent)

**Benefits:**
- Shared representations improve both tasks
- More efficient than separate models
- Better feature learning

**Loss Function:**
```
L = α * L_CTR + β * L_dwell_time
```

### 3. Negative Sampling

**Problem:** Most posts are NOT clicked (99%+ negative examples)

**Solution:** Sample negatives intelligently
- Random sampling
- Hard negative mining (close but not clicked)
- In-batch negatives

### 4. Embeddings

Convert categorical features to dense vectors:
- User ID → 128-dim vector
- Post ID → 128-dim vector
- Author ID → 64-dim vector

**Why?**
- Capture similarity (similar users have similar embeddings)
- Reduce dimensionality (millions of users → 128 dims)
- Enable generalization

---

## 📁 Dataset

You'll use a synthetic dataset simulating Meta-scale feed ranking:

**Training Data:** 1M user-post interactions
**Validation Data:** 100K interactions
**Test Data:** 100K interactions

**Features:**
1. **User Features:**
   - user_id (categorical, 10K unique users)
   - age (numeric, 18-80)
   - gender (categorical)
   - num_friends (numeric)
   - user_engagement_history (vector, last 10 interactions)

2. **Post Features:**
   - post_id (categorical, 100K unique posts)
   - author_id (categorical, 50K unique authors)
   - post_type (photo/video/text/link)
   - num_likes (numeric)
   - num_comments (numeric)
   - post_age_hours (numeric)

3. **Context Features:**
   - hour_of_day (0-23)
   - day_of_week (0-6)
   - device_type (mobile/desktop)

4. **Labels:**
   - clicked (0/1) - for CTR task
   - dwell_time_seconds (0-300) - for dwell time task

---

## 🚀 Milestones

### Milestone 1: Data Pipeline (Week 1, Days 1-2)
**Goal:** Build efficient data loading

**Tasks:**
- [ ] Download and explore dataset
- [ ] Create custom `Dataset` class
- [ ] Implement feature preprocessing
- [ ] Build `DataLoader` with batching
- [ ] Handle categorical encoding
- [ ] Implement feature normalization

**Deliverables:**
- `data.py` - Dataset and DataLoader
- `features.py` - Feature engineering utilities
- Data exploration notebook

---

### Milestone 2: Model Architecture (Week 1, Days 3-5)
**Goal:** Implement Wide & Deep model

**Tasks:**
- [ ] Implement embedding layers for categorical features
- [ ] Build Deep component (MLP)
- [ ] Build Wide component (linear)
- [ ] Implement feature interaction layer
- [ ] Create multi-task output heads
- [ ] Combine Wide & Deep

**Deliverables:**
- `model.py` - Complete model architecture
- Unit tests for each component
- Architecture diagram

**Code Structure:**
```python
class WideAndDeepRanker(nn.Module):
    def __init__(self, config):
        self.user_embedding = nn.Embedding(...)
        self.post_embedding = nn.Embedding(...)
        self.deep_component = DeepComponent(...)
        self.wide_component = WideComponent(...)
        self.ctr_head = nn.Linear(...)
        self.dwell_time_head = nn.Linear(...)

    def forward(self, features):
        # Embed categorical features
        # Deep path
        # Wide path
        # Combine and predict
        return ctr_logits, dwell_time_pred
```

---

### Milestone 3: Training Loop (Week 2, Days 1-3)
**Goal:** Train model with multi-task learning

**Tasks:**
- [ ] Implement multi-task loss function
- [ ] Set up optimizer and learning rate scheduler
- [ ] Implement negative sampling
- [ ] Add logging (TensorBoard/WandB)
- [ ] Implement checkpointing
- [ ] Add early stopping

**Deliverables:**
- `train.py` - Training script
- Training loop with validation
- Hyperparameter config file

**Loss Implementation:**
```python
def compute_loss(ctr_logits, ctr_labels, dwell_pred, dwell_labels, alpha=1.0, beta=1.0):
    # CTR loss (binary cross-entropy)
    ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits, ctr_labels)

    # Dwell time loss (MSE, only on clicked examples)
    clicked_mask = ctr_labels == 1
    dwell_loss = F.mse_loss(dwell_pred[clicked_mask], dwell_labels[clicked_mask])

    return alpha * ctr_loss + beta * dwell_loss
```

---

### Milestone 4: Evaluation & Metrics (Week 2, Days 4-5)
**Goal:** Implement ranking metrics

**Tasks:**
- [ ] Implement AUC-ROC for CTR
- [ ] Implement NDCG (Normalized Discounted Cumulative Gain)
- [ ] Implement MAP (Mean Average Precision)
- [ ] Implement calibration metrics
- [ ] Create evaluation script
- [ ] Generate ranking quality report

**Deliverables:**
- `metrics.py` - Ranking metrics
- `evaluate.py` - Evaluation script
- Performance analysis notebook

**Key Metrics:**
1. **AUC-ROC:** Overall ranking quality
2. **NDCG@K:** Quality of top-K recommendations
3. **MAP:** Mean average precision
4. **Calibration:** Predicted probabilities vs actual rates

---

### Milestone 5: Advanced Features (Week 3)
**Goal:** Production-ready improvements

**Tasks:**
- [ ] Implement hard negative mining
- [ ] Add feature importance analysis
- [ ] Implement model serving code
- [ ] Add A/B testing simulation
- [ ] Optimize inference latency
- [ ] Create model export (ONNX)

**Deliverables:**
- Optimized model
- Serving infrastructure
- A/B test analysis

---

## 🎯 Stretch Goals

### 1. Two-Tower Retrieval + Re-ranking
Split into two stages:
- **Stage 1 (Retrieval):** Fast two-tower model (user tower, post tower) for candidate generation
- **Stage 2 (Ranking):** Wide & Deep model for precise ranking

### 2. Feature Interaction Techniques
- Implement DCN (Deep & Cross Network)
- Add self-attention for feature interactions
- Implement FM (Factorization Machines) layer

### 3. Advanced Sampling
- Implement importance sampling
- Batch hard negative mining
- Curriculum learning (easy → hard examples)

### 4. Online Learning
- Implement incremental learning
- Add concept drift detection
- Simulate real-time updates

---

## 🏭 Meta-Scale Considerations

### 1. Scalability
**Challenge:** Billions of users × millions of posts = trillions of combinations

**Solutions:**
- **Candidate Generation:** Reduce from millions to hundreds using efficient retrieval
- **Distributed Training:** Use FSDP to train on multiple GPUs
- **Embedding Tables:** Shard large embedding tables across machines
- **Batch Size:** Use gradient accumulation for large effective batch sizes

### 2. Latency Requirements
**Target:** < 50ms for ranking 500 posts

**Optimizations:**
- Model quantization (INT8)
- ONNX export for optimized inference
- Batch predictions
- Model distillation (large model → small model)
- Feature caching

### 3. Data Pipeline
**Challenge:** Streaming data, billions of examples/day

**Solutions:**
- Shuffle buffer for randomization
- Prefetching and parallel data loading
- Feature preprocessing at scale (Spark)
- Incremental dataset updates

### 4. Feature Engineering
**Challenge:** Thousands of features, many categorical with high cardinality

**Solutions:**
- Feature hashing for rare categories
- Embedding dimension tuning
- Feature selection based on importance
- Automatic feature interaction discovery

### 5. Online-Offline Consistency
**Challenge:** Model trained offline must work online

**Solutions:**
- Log-based training data
- Online metric monitoring
- Shadow mode deployment
- Gradual rollout with A/B testing

### 6. Fairness & Diversity
**Challenge:** Avoid filter bubbles, ensure diverse content

**Solutions:**
- Diversity penalties in ranking
- Exploration-exploitation tradeoffs
- Fairness metrics across demographics
- Debiasing techniques

---

## 📊 Expected Results

After training, your model should achieve:
- **AUC-ROC:** > 0.75
- **NDCG@10:** > 0.70
- **MAP:** > 0.65
- **Dwell Time MAE:** < 20 seconds
- **Inference latency:** < 10ms per example (CPU)

---

## 📚 Resources

**Papers:**
- [Wide & Deep Learning (Google, 2016)](https://arxiv.org/abs/1606.07792)
- [Deep & Cross Network (Google, 2017)](https://arxiv.org/abs/1708.05123)
- [DLRM: Deep Learning Recommendation Model (Meta, 2019)](https://arxiv.org/abs/1906.00091)

**Blog Posts:**
- [Meta: Powered by AI: Instagram's Explore recommender system](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/)
- [Tech at Meta: Recommender Systems](https://engineering.fb.com/category/ml-applications/recommendations/)

---

## ✅ Success Criteria

You've successfully completed this project when you can:
- [ ] Explain Wide & Deep architecture
- [ ] Implement multi-task learning
- [ ] Train model with negative sampling
- [ ] Evaluate with ranking metrics
- [ ] Discuss Meta-scale challenges
- [ ] Optimize for production latency
- [ ] Achieve target performance metrics

---

## 🤝 Getting Started

1. Read the theory brief above
2. Explore the synthetic dataset
3. Start with Milestone 1: Data Pipeline
4. Implement incrementally, testing each component
5. Get feedback after each milestone
6. Iterate and improve

**Ready to build your first production-scale ranking system? Let's go! 🚀**
