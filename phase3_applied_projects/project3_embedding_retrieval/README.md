# Project 3: Embedding-Based Retrieval (Two-Tower Model) 🔍

> **Time:** 1-2 weeks
> **Difficulty:** Advanced
> **Goal:** Build a two-tower retrieval system for candidate generation at scale

---

## 📚 Project Overview

Build a two-tower neural retrieval system that efficiently searches through millions of items to find the most relevant candidates for a user. This is the first stage in recommendation systems at Meta scale.

**Real-world application:** Power Facebook Marketplace search, Instagram Explore, and initial candidate generation for feed ranking.

---

## 🎯 Architecture

```
Two-Tower Retrieval System

User Tower:
User Features → Embedding Network → User Vector (dim=128)

Item Tower:
Item Features → Embedding Network → Item Vector (dim=128)

Similarity:
cosine_similarity(user_vector, item_vector)

Retrieval:
FAISS Approximate Nearest Neighbor Search
```

---

## 🎓 Theory Brief

### Why Two-Tower?
- **Separation:** Encode users and items independently
- **Scalability:** Pre-compute item embeddings offline
- **Speed:** ANN search is ~1ms for millions of items
- **Flexibility:** Update user/item towers separately

### Training Strategy
**In-Batch Negatives:**
- Batch size = 1024
- Each user's positive item = 1 positive
- All other items in batch = 1023 negatives
- Efficient! No need to sample negatives separately

**Loss Function:**
```python
# Contrastive loss (InfoNCE)
similarity_matrix = user_embeddings @ item_embeddings.T  # (B, B)
labels = torch.arange(batch_size)  # Diagonal is positive
loss = F.cross_entropy(similarity_matrix / temperature, labels)
```

---

## 🚀 Milestones

### Milestone 1: Data Pipeline & Feature Engineering
**Tasks:**
- [ ] Create user-item interaction dataset
- [ ] Extract user features (demographics, history)
- [ ] Extract item features (metadata, stats)
- [ ] Implement positive/negative sampling
- [ ] Build DataLoader with in-batch negatives

---

### Milestone 2: Two-Tower Model
**Tasks:**
- [ ] Implement User Tower (MLP with embeddings)
- [ ] Implement Item Tower (MLP with embeddings)
- [ ] Contrastive loss (InfoNCE)
- [ ] Temperature scaling
- [ ] Batch hard negative mining (optional)

**Code Structure:**
```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_features, item_features, embedding_dim=128):
        self.user_tower = MLPEncoder(user_features, embedding_dim)
        self.item_tower = MLPEncoder(item_features, embedding_dim)

    def forward(self, user_features, item_features):
        user_emb = F.normalize(self.user_tower(user_features), dim=-1)
        item_emb = F.normalize(self.item_tower(item_features), dim=-1)
        return user_emb, item_emb

    def compute_similarity(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.T)
```

---

### Milestone 3: Training with Hard Negatives
**Tasks:**
- [ ] Implement training loop
- [ ] In-batch negative sampling
- [ ] Hard negative mining (mine hardest negatives)
- [ ] Gradient accumulation
- [ ] Learning rate scheduling

**Hard Negative Mining:**
```python
# Find hard negatives (high similarity but not clicked)
similarities = user_emb @ all_item_embs.T
hard_negatives = similarities.topk(k=10, dim=1)  # Top-10 similar but not clicked
```

---

### Milestone 4: FAISS Integration & Retrieval
**Tasks:**
- [ ] Generate item embeddings for entire catalog
- [ ] Build FAISS index (Flat, IVF, HNSW)
- [ ] Implement KNN search
- [ ] Batch retrieval
- [ ] Evaluate Recall@K

**FAISS Example:**
```python
import faiss

# Build index
embedding_dim = 128
index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine if normalized)

# Add item embeddings
item_embeddings = ...  # (N, 128)
index.add(item_embeddings)

# Search
user_embedding = ...  # (1, 128)
k = 100
distances, indices = index.search(user_embedding, k)  # Top-100 items
```

---

### Milestone 5: Evaluation & Optimization
**Tasks:**
- [ ] Implement Recall@K (K=10, 50, 100)
- [ ] Implement MRR (Mean Reciprocal Rank)
- [ ] Implement NDCG@K
- [ ] Analyze embedding quality (t-SNE)
- [ ] Optimize FAISS index (quantization)
- [ ] Benchmark retrieval latency

---

## 🎯 Stretch Goals

### 1. Quantization & Compression
- Product Quantization (PQ) for item embeddings
- Reduce 128-dim float32 to 8-byte codes
- 16x compression with minimal accuracy loss

### 2. Multi-Tower Architecture
- Add context tower (time, location, device)
- Implement three-way similarity

### 3. Cross-Batch Negatives
- Use embeddings from previous batches as negatives
- Momentum encoder (like MoCo)

### 4. Diversity & Exploration
- Add diversity penalty
- Implement explore-exploit (ε-greedy)

---

## 🏭 Meta-Scale Considerations

### 1. Billion-Scale Catalog
**Challenge:** 1B+ items to search

**Solutions:**
- IVF (Inverted File) FAISS index
- Product Quantization
- Distributed FAISS
- Hierarchical search

### 2. Real-Time Updates
**Challenge:** New items added constantly

**Solutions:**
- Incremental index updates
- Dual indexes (old + new)
- Periodic full rebuild

### 3. Personalization
**Challenge:** User preferences change over time

**Solutions:**
- Online learning
- Contextual features (recent interactions)
- Session-based modeling

### 4. Cold Start
**Challenge:** New users/items with no history

**Solutions:**
- Content-based features
- Popularity bias
- Hybrid retrieval (embedding + rule-based)

---

## 📊 Expected Results

- **Recall@10:** > 40%
- **Recall@100:** > 70%
- **MRR:** > 0.30
- **Retrieval Latency:** < 5ms for 1M items
- **Index Size:** < 1GB after quantization

---

## 📚 Resources

**Papers:**
- [Sampling-Bias-Corrected Neural Modeling (Google, 2019)](https://research.google/pubs/pub48840/)
- [FAISS: A Library for Efficient Similarity Search (Meta, 2017)](https://arxiv.org/abs/1702.08734)
- [Real-time Personalization using Embeddings (Airbnb, 2018)](https://dl.acm.org/doi/10.1145/3219819.3219885)

**Code:**
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

---

## ✅ Success Criteria

- [ ] Implement two-tower architecture
- [ ] Train with in-batch negatives
- [ ] Integrate FAISS for fast retrieval
- [ ] Achieve target Recall@K metrics
- [ ] Optimize index with quantization
- [ ] Benchmark retrieval at scale

**Ready to build billion-scale search? Let's go! 🔍**
