# Project 3: Embedding-Based Retrieval (Two-Tower Model) 🔍

> **Time:** 1-2 weeks
> **Difficulty:** Advanced
> **Goal:** Build a two-tower retrieval system for billion-scale candidate generation

---

## Why This Project Matters

Every search query, every recommendation, every "People You May Know" suggestion starts with **retrieval**. Before complex ranking models can run, you need to narrow down from **millions or billions of items to hundreds of candidates**. This is the retrieval problem.

**Scale and Impact**:
- **Google Search**: 8.5 billion searches/day across trillions of web pages
- **Meta**: Billions of users, millions of posts/second
- **Amazon**: 350+ million products, personalized for each user
- **Spotify**: 100M+ songs, personalized for 500M+ users

**Business Critical**:
- **Search**: Find relevant products from billions ($400B+ e-commerce market)
- **Recommendations**: Surface content users will engage with (drives 80% of Netflix views)
- **Ads**: Match ads to users (Google Ads: $280B revenue)
- **Social**: Find friends, groups, events (Meta's entire business model)

**Technical Challenge**:
```
How do you search through 1 billion items in <10ms?

Naive approach: Score each item with complex model
Time: 1B items × 10ms/item = 10M seconds = 115 days! ❌

Production approach: Two-stage system
Stage 1 (Retrieval): Fast embedding search → 1000 candidates in ~5ms ✅
Stage 2 (Ranking): Complex model on 1000 candidates → Top 10 in ~30ms ✅
Total: ~35ms! 🎉
```

This project teaches you **Stage 1: Retrieval**.

---

## The Big Picture

### The Problem

**Input**: Query (e.g., user, search query, current item)
**Output**: Top-K most relevant items from massive corpus

**Example applications**:
- **Search**: Query = "red running shoes", Corpus = All products
- **Recommendation**: Query = User, Corpus = All posts/videos/products
- **Ads**: Query = User + content, Corpus = All ads
- **Similar items**: Query = Current product, Corpus = All other products

**Key Requirements**:
1. **Accuracy**: Retrieve truly relevant items (high recall)
2. **Speed**: Sub-10ms for real-time applications
3. **Scalability**: Handle billions of items
4. **Freshness**: Update with new items constantly

**Why not just use a complex model?**
```python
# Naive approach
for item in all_items:  # 1 billion iterations!
    score = complex_model(user, item)  # 10ms per call
# Total time: 10 billion ms = 115 days ❌
```

**Two-tower solution**:
```python
# Precompute item embeddings (offline)
item_embeddings = [item_tower(item) for item in all_items]
# Build index (one-time cost)
index = FAISSIndex(item_embeddings)

# Online inference
user_emb = user_tower(user)  # ~1ms
top_k_items = index.search(user_emb, k=1000)  # ~5ms via ANN
# Total: ~6ms for 1B items! ✅
```

---

## Deep Dive: Two-Tower Architecture Theory

### 1. The Two-Tower Paradigm

**Core Idea**: Encode queries and items **independently** into same embedding space, then use **similarity** for retrieval.

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE (INDEXING)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  All Items (1B items)                                   │
│       ↓                                                 │
│  Item Tower (Neural Network)                            │
│       ↓                                                 │
│  Item Embeddings (1B × 128-dim vectors)                 │
│       ↓                                                 │
│  FAISS Index (for fast similarity search)               │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE (SERVING)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query (user, search text, etc.)                        │
│       ↓                                                 │
│  Query Tower (Neural Network)                           │
│       ↓                                                 │
│  Query Embedding (128-dim vector)                       │
│       ↓                                                 │
│  FAISS Search (find top-K most similar items)           │
│       ↓                                                 │
│  Top 1000 candidate items                               │
│       ↓                                                 │
│  [Send to ranking model for precise scoring]            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Why "Two-Tower"?**
- **Query Tower**: Encodes query into embedding
- **Item Tower**: Encodes item into embedding
- **Independent**: Can precompute item embeddings
- **Same space**: Embeddings are comparable (cosine similarity, dot product)

**Mathematical formulation**:
```
Query embedding: q = f_query(query_features)
Item embedding:  d = f_item(item_features)

Similarity: s(q, d) = q · d (dot product)
           or s(q, d) = (q · d) / (||q|| ||d||) (cosine similarity)

Retrieval: top_k = argmax_{d in corpus} s(q, d)
```

---

### 2. Contrastive Learning: Training Two-Towers

**Challenge**: How do we train the two towers?

**Goal**:
- Similar query-item pairs → High similarity
- Dissimilar query-item pairs → Low similarity

**Contrastive Learning** (InfoNCE Loss):

Given a batch of (query, positive_item) pairs:
```
Batch: [(q₁, d₁⁺), (q₂, d₂⁺), ..., (qₙ, dₙ⁺)]

For each query qᵢ:
- Positive: dᵢ⁺ (the item user interacted with)
- Negatives: {d₁⁺, d₂⁺, ..., dₙ⁺} \ {dᵢ⁺} (other items in batch)
```

**Loss** (InfoNCE / NT-Xent):
```python
# Compute all similarities
similarities = query_embeddings @ item_embeddings.T  # (N, N) matrix

# Normalize by temperature
similarities = similarities / temperature

# Labels: diagonal is positive (qᵢ matched with dᵢ)
labels = torch.arange(batch_size)

# Cross-entropy loss
loss = F.cross_entropy(similarities, labels)
```

**Why this works**:
- For q₁, want: s(q₁, d₁⁺) > s(q₁, d₂⁺), s(q₁, d₃⁺), ...
- Cross-entropy encourages diagonal (positives) to be large
- Automatically pushes negatives away

**In-Batch Negatives**:
- **Key insight**: Use other examples in batch as negatives
- **Efficiency**: Get N-1 negatives for free (no extra sampling needed)
- **Batch size matters**: Larger batch = more negatives = better training
  ```
  Batch size 256 → 255 negatives per query
  Batch size 2048 → 2047 negatives (Google uses this)
  ```

**Temperature τ**:
```python
similarities = similarities / τ

τ = 0.07 (common value)
```
- **Low τ (e.g., 0.05)**: Sharpens distribution (harder negatives)
- **High τ (e.g., 0.1)**: Softens distribution (easier negatives)
- **Tuning**: Critical hyperparameter, usually 0.05-0.1

---

### 3. Hard Negative Mining

**Problem**: Random negatives are often too easy.
```
Query: "red shoes"
Positive: Red Nike sneakers
Easy negative: Dog food (obviously irrelevant)
Hard negative: Blue Nike sneakers (similar but not correct)
```

**Solution**: Mine hard negatives

**Strategies**:

#### 1. Batch Hard Negatives
```python
# For each query, find hardest negative in batch
similarities = q_emb @ d_emb.T  # (N, N)

# Mask diagonal (positives)
mask = torch.eye(N).bool()
similarities.masked_fill_(mask, float('-inf'))

# Hardest negative for each query
hard_neg_scores = similarities.max(dim=1).values
```

#### 2. Offline Mining
```python
# Periodically mine hard negatives
model.eval()
with torch.no_grad():
    # For each query, find top-k items
    top_k = index.search(query_emb, k=100)

    # Remove positives
    negatives = [item for item in top_k if item not in positives]

    # Use top hard negatives
    hard_negatives = negatives[:10]
```

#### 3. Mixed Sampling
```python
# Combine random and hard negatives
negatives = {
    'in_batch': in_batch_negatives,     # All others in batch
    'hard': top_10_hard_negatives,       # Mined offline
    'random': 5_random_negatives         # Random from corpus
}
```

**Best practice**: Start with in-batch, add hard mining after initial training.

---

### 4. Approximate Nearest Neighbor (ANN) Search

**Problem**: Exact nearest neighbor search is too slow for billions of items.
```
Brute force: Compare query to all N items
Time complexity: O(N × d) where d = embedding dimension
For N=1B, d=128: 128 billion operations ❌
```

**Solution**: Approximate Nearest Neighbor (ANN) algorithms

#### FAISS (Facebook AI Similarity Search)

**Key insight**: Trade accuracy for speed (95% recall is good enough)

**Index types**:

**1. Flat Index** (Exact, slow):
```python
import faiss

dim = 128
index = faiss.IndexFlatIP(dim)  # Inner Product (for cosine if normalized)
index.add(item_embeddings)      # Add all items

# Search
D, I = index.search(query_embedding, k=100)  # Top-100
# D: distances, I: indices

# Time: O(N × d) - slow for large N
```

**2. IVF (Inverted File)**: Cluster-based search
```python
# Create IVF index
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

# Train clustering
index.train(sample_embeddings)  # Learn clusters
index.add(item_embeddings)

# Search (only search nearby clusters)
index.nprobe = 10  # Search 10 nearest clusters
D, I = index.search(query_embedding, k=100)

# Time: O(nprobe × (N/nlist) × d) - Much faster!
```

**How IVF works**:
```
1. Offline: Cluster items into nlist groups (e.g., 100-1000 clusters)
2. Online:
   - Find nearest clusters to query (e.g., 10 clusters)
   - Search only items in those clusters
   - Total searched: 10 clusters × (1M items/cluster) = 10M items (vs 1B!)
```

**3. HNSW (Hierarchical Navigable Small World)**: Graph-based search
```python
# HNSW index (state-of-the-art)
M = 32  # Number of connections per node
index = faiss.IndexHNSWFlat(dim, M)
index.add(item_embeddings)

# Search
D, I = index.search(query_embedding, k=100)

# Pros: Fastest, highest accuracy
# Cons: Large memory (stores graph)
```

**How HNSW works**:
```
Build graph where:
- Each item is a node
- Edges connect similar items
- Multiple layers (hierarchical)

Search:
1. Start at top layer (sparse graph)
2. Greedily navigate to query's region
3. Go down layers (denser graphs)
4. Find k nearest neighbors efficiently

Time: O(log N) hops instead of O(N) comparisons!
```

**4. Product Quantization (PQ)**: Compression
```python
# PQ index (compressed embeddings)
m = 8  # Number of subvectors
nbits = 8  # Bits per subquantizer

index = faiss.IndexPQ(dim, m, nbits)
index.train(sample_embeddings)
index.add(item_embeddings)

# Memory: 1B items × 8 bytes vs 1B × 512 bytes (64x compression!)
```

**Comparison**:
| Index | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| Flat | Slow | 100% | High | <1M items, offline |
| IVF | Medium | 95% | Medium | 1M-100M items |
| HNSW | Fast | 98% | High | <100M items, high QPS |
| PQ | Medium | 90% | Low | >100M items, memory-constrained |
| IVF+PQ | Fast | 93% | Low | Billions of items |

**Meta/Google scale**:
```python
# Production setup for billions of items
# Use IVF + Product Quantization

nlist = 65536  # 64K clusters
m = 64         # 64 subvectors
nbits = 8      # 8 bits each

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

# Distributed: Shard across multiple machines
```

---

### 5. Retrieval Metrics

Unlike ranking, retrieval cares about **recall**: did we retrieve the relevant items?

#### Recall@K
```
Recall@K = (# relevant items in top-K) / (total # relevant items)
```

**Example**:
```
User has engaged with 10 posts
Retrieval returns 100 candidates
7 of the 100 are in the set of 10 relevant posts

Recall@100 = 7/10 = 0.7 (70%)
```

**Why it matters**: If relevant items aren't in candidates, ranking can't fix it!

#### Mean Reciprocal Rank (MRR)
```
MRR = Average of (1 / rank of first relevant item)
```

**Example**:
```
Query 1: First relevant item at rank 3 → RR = 1/3
Query 2: First relevant item at rank 1 → RR = 1/1
Query 3: First relevant item at rank 5 → RR = 1/5

MRR = (1/3 + 1/1 + 1/5) / 3 = 0.53
```

**Why it matters**: Measures how high the first relevant item appears.

#### NDCG@K (Normalized Discounted Cumulative Gain)
```
DCG@K = Σᵢ₌₁ᴷ (2^{relevance_i} - 1) / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K
```

**Why it matters**: Accounts for position (top results matter more).

#### Hit Rate@K
```
Hit Rate@K = (# queries with ≥1 relevant item in top-K) / (total queries)
```

**Example**:
```
100 queries
80 queries have at least 1 relevant item in top-10

Hit Rate@10 = 80/100 = 0.8
```

**Production targets**:
```
Recall@100: >70% (retrieve most relevant items)
Recall@1000: >90% (give ranking model good candidates)
MRR: >0.3 (relevant items appear early)
Latency: <10ms (real-time)
```

---

### 6. Production Deployment: Billion-Scale Challenges

#### Challenge 1: Index Size
```
1 billion items × 128 dimensions × 4 bytes (float32) = 512 GB!
```

**Solutions**:
1. **Quantization**: INT8 or Product Quantization (8-16x compression)
2. **Dimensionality reduction**: 128-dim → 64-dim
3. **Sharding**: Distribute across machines
4. **Hierarchical**: Coarse + fine retrieval

#### Challenge 2: Index Updates
```
Problem: New items added constantly (posts, products)
Can't rebuild index every time (takes hours for billions)
```

**Solutions**:
1. **Dual indexes**:
   ```python
   main_index = faiss.IndexIVFPQ(...)  # Bulk of items
   delta_index = faiss.IndexFlat(...)   # New items (small, exact)

   # Search both
   results_main = main_index.search(q, k)
   results_delta = delta_index.search(q, k)
   results = merge_and_resort(results_main, results_delta)
   ```

2. **Incremental updates**:
   ```python
   index.add_with_ids(new_embeddings, new_ids)  # Add new items
   ```

3. **Periodic rebuild**:
   ```
   - Build new index daily/weekly with all items
   - Swap atomically (blue-green deployment)
   ```

#### Challenge 3: Serving Latency
```
Target: <10ms for retrieval
Challenges:
- Index loading time
- Network latency
- CPU/GPU contention
```

**Solutions**:
1. **GPU acceleration**:
   ```python
   # Move index to GPU
   res = faiss.StandardGpuResources()
   gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

   # 10-100x faster searches!
   ```

2. **Batching**:
   ```python
   # Search multiple queries at once
   query_batch = np.vstack([q1, q2, q3, ...])  # (batch_size, dim)
   D, I = index.search(query_batch, k)          # Vectorized

   # Throughput: 1000 QPS vs 100 QPS for sequential
   ```

3. **Caching**:
   ```python
   # Cache popular queries
   cache = LRUCache(max_size=10000)

   if query in cache:
       return cache[query]
   else:
       results = index.search(query, k)
       cache[query] = results
       return results
   ```

4. **Serving infrastructure**:
   ```
   ┌─────────┐
   │ Client  │
   └────┬────┘
        │
        ▼
   ┌─────────┐   Sharded indexes
   │ Router  │───┐
   └─────────┘   │
        ┌────────┴───────┬────────┐
        ▼                ▼        ▼
   ┌─────────┐    ┌─────────┐  ┌─────────┐
   │ Index 1 │    │ Index 2 │  │ Index 3 │
   │ Items   │    │ Items   │  │ Items   │
   │ 0-333M  │    │ 333M-   │  │ 666M-1B │
   │         │    │ 666M    │  │         │
   └─────────┘    └─────────┘  └─────────┘

   - Route query to all shards in parallel
   - Merge results (top-K from each shard)
   - Return combined top-K
   ```

#### Challenge 4: Quality vs Speed Trade-off
```
Exact search: 100% recall, slow
ANN search: 95% recall, fast

Question: Is 5% recall loss acceptable?
```

**Answer depends on**:
- Application (search vs recommendation)
- Stage 2 ranking quality (can ranking fix mistakes?)
- Business metrics (revenue, engagement)

**Best practice**: A/B test different index configurations.

---

### 7. Applications & Variants

#### Application 1: Product Search (Amazon, Alibaba)
```
Query Tower: Text encoder (BERT) for search query
Item Tower: Multi-modal encoder (image + text) for product
Similarity: Dot product

Example:
Query: "red running shoes"
→ Text embedding
→ Find top-100 products with similar embeddings
→ Rank with complex model (price, reviews, personalization)
```

#### Application 2: Video Recommendation (YouTube, TikTok)
```
Query Tower: User history + context encoder
Item Tower: Video encoder (visual + audio + metadata)

Special consideration: Freshness
- New videos must appear quickly
- Use dual index (main + recent)
```

#### Application 3: Social Search (Facebook Friends, LinkedIn)
```
Query Tower: User profile + network
Item Tower: Candidate user profile + network
Symmetry: Bidirectional similarity (A similar to B ⇔ B similar to A)

Hard constraint: Exclude already-friends, blocked users
```

#### Application 4: Ad Targeting
```
Query Tower: User + page context
Item Tower: Ad creative + targeting criteria

Business logic:
- Retrieval: Find relevant ads
- Ranking: Maximize expected revenue (CTR × bid)
- Constraint: Diversity, frequency capping
```

---

## Implementation Guide

### Phase 1: Data Preparation (Days 1-2)

#### Synthetic Dataset
```python
import numpy as np
import pandas as pd

def generate_two_tower_data(
    num_users=10000,
    num_items=100000,
    interactions_per_user=20,
    seed=42
):
    """
    Generate synthetic retrieval dataset.

    Simulates users interacting with items (clicks, views, purchases).
    """
    np.random.seed(seed)

    # === USER FEATURES ===
    user_features = {
        'user_id': np.arange(num_users),
        'age': np.random.randint(18, 70, num_users),
        'gender': np.random.choice(['M', 'F', 'O'], num_users),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU'], num_users),
        # Latent user interests (will correlate with interactions)
        'interest_vector': np.random.randn(num_users, 10)
    }

    # === ITEM FEATURES ===
    item_features = {
        'item_id': np.arange(num_items),
        'category': np.random.randint(0, 20, num_items),
        'price': np.random.exponential(30, num_items),
        'rating': np.random.uniform(1, 5, num_items),
        # Latent item attributes (will correlate with user interests)
        'attribute_vector': np.random.randn(num_items, 10)
    }

    # === INTERACTIONS (USER-ITEM PAIRS) ===
    interactions = []

    for user_id in range(num_users):
        # User's interest vector
        user_interest = user_features['interest_vector'][user_id]

        # Compute affinity to all items (dot product + noise)
        affinities = item_features['attribute_vector'] @ user_interest
        affinities += np.random.randn(num_items) * 0.5  # Add noise

        # Sample top items (weighted by affinity)
        probs = np.exp(affinities) / np.exp(affinities).sum()
        interacted_items = np.random.choice(
            num_items,
            size=interactions_per_user,
            replace=False,
            p=probs
        )

        for item_id in interacted_items:
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'timestamp': np.random.randint(0, 1000000)
            })

    interactions_df = pd.DataFrame(interactions)

    return user_features, item_features, interactions_df


# Generate data
user_features, item_features, interactions = generate_two_tower_data()

print(f"Users: {len(user_features['user_id'])}")
print(f"Items: {len(item_features['item_id'])}")
print(f"Interactions: {len(interactions)}")
print(f"Sparsity: {len(interactions) / (len(user_features['user_id']) * len(item_features['item_id'])):.6f}")
```

#### Dataset Class
```python
class RetrievalDataset(Dataset):
    """
    Dataset for two-tower model training.

    Returns (user_features, positive_item, metadata)
    Negatives are sampled in-batch during training.
    """

    def __init__(self, interactions_df, user_features, item_features):
        self.interactions = interactions_df
        self.user_features = user_features
        self.item_features = item_features

        # Group interactions by user (for faster sampling)
        self.user_to_items = (
            interactions_df
            .groupby('user_id')['item_id']
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        item_id = row['item_id']

        # === USER FEATURES ===
        user_feats = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'age': torch.tensor(
                self.user_features['age'][user_id],
                dtype=torch.float32
            ),
            'gender': torch.tensor(
                {'M': 0, 'F': 1, 'O': 2}[self.user_features['gender'][user_id]],
                dtype=torch.long
            ),
            # ... more features
        }

        # === ITEM FEATURES ===
        item_feats = {
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'category': torch.tensor(
                self.item_features['category'][item_id],
                dtype=torch.long
            ),
            'price': torch.tensor(
                self.item_features['price'][item_id],
                dtype=torch.float32
            ),
            # ... more features
        }

        return user_feats, item_feats
```

---

### Phase 2: Model Architecture (Days 3-4)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TowerEncoder(nn.Module):
    """
    Generic tower encoder: Embeddings + MLP
    """

    def __init__(
        self,
        categorical_features,  # List of (name, cardinality, emb_dim)
        numerical_features,     # List of feature names
        hidden_dims=[512, 256],
        output_dim=128,
        dropout=0.1
    ):
        super().__init__()

        # === EMBEDDINGS ===
        self.embeddings = nn.ModuleDict()
        emb_total_dim = 0

        for name, cardinality, emb_dim in categorical_features:
            self.embeddings[name] = nn.Embedding(cardinality, emb_dim)
            emb_total_dim += emb_dim

        # === MLP ===
        self.numerical_features = numerical_features
        input_dim = emb_total_dim + len(numerical_features)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output projection
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        # === EMBED CATEGORICAL FEATURES ===
        embedded = []
        for name, embedding in self.embeddings.items():
            embedded.append(embedding(features[name]))

        # === CONCATENATE NUMERICAL FEATURES ===
        numerical = torch.stack(
            [features[name] for name in self.numerical_features],
            dim=1
        )

        # === COMBINE ===
        x = torch.cat(embedded + [numerical], dim=1)

        # === MLP ===
        x = self.mlp(x)

        # === L2 NORMALIZE (for cosine similarity) ===
        x = F.normalize(x, p=2, dim=1)

        return x


class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model.

    User tower and item tower encode independently.
    Training: Contrastive loss (in-batch negatives)
    Inference: Similarity search
    """

    def __init__(self, user_config, item_config, embedding_dim=128):
        super().__init__()

        self.user_tower = TowerEncoder(
            categorical_features=user_config['categorical'],
            numerical_features=user_config['numerical'],
            output_dim=embedding_dim
        )

        self.item_tower = TowerEncoder(
            categorical_features=item_config['categorical'],
            numerical_features=item_config['numerical'],
            output_dim=embedding_dim
        )

    def forward(self, user_features, item_features):
        """
        Encode user and item independently.

        Returns:
            user_emb: (batch_size, embedding_dim)
            item_emb: (batch_size, embedding_dim)
        """
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)

        return user_emb, item_emb

    def compute_similarity(self, user_emb, item_emb):
        """
        Compute similarity matrix.

        Args:
            user_emb: (batch_size, dim)
            item_emb: (batch_size, dim)

        Returns:
            similarities: (batch_size, batch_size)
        """
        # Dot product (cosine since embeddings are normalized)
        return torch.matmul(user_emb, item_emb.T)


def compute_contrastive_loss(user_emb, item_emb, temperature=0.07):
    """
    InfoNCE loss with in-batch negatives.

    Args:
        user_emb: (batch_size, dim)
        item_emb: (batch_size, dim)
        temperature: Temperature for scaling

    Returns:
        loss: Scalar
        metrics: Dict with accuracy, etc.
    """
    batch_size = user_emb.size(0)

    # Compute similarity matrix
    similarities = torch.matmul(user_emb, item_emb.T)  # (B, B)

    # Scale by temperature
    similarities = similarities / temperature

    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=user_emb.device)

    # Cross-entropy loss
    loss = F.cross_entropy(similarities, labels)

    # === METRICS ===
    with torch.no_grad():
        # Accuracy: % of examples where positive is highest
        predictions = similarities.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()

        # In-batch recall@K
        _, top_k_indices = similarities.topk(k=10, dim=1)
        recall_10 = (
            (top_k_indices == labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
        )

    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'recall@10': recall_10.item()
    }

    return loss, metrics
```

---

### Phase 3: Training (Days 5-6)

```python
def train_two_tower(model, train_loader, val_loader, config):
    """
    Train two-tower model with contrastive loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )

    best_recall = 0.0

    for epoch in range(config['num_epochs']):
        # === TRAINING ===
        model.train()
        train_metrics = {'loss': 0, 'accuracy': 0, 'recall@10': 0}

        for user_feats, item_feats in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move to device
            user_feats = {k: v.to(device) for k, v in user_feats.items()}
            item_feats = {k: v.to(device) for k, v in item_feats.items()}

            # Forward pass
            user_emb, item_emb = model(user_feats, item_feats)

            # Contrastive loss
            loss, metrics = compute_contrastive_loss(
                user_emb, item_emb,
                temperature=config['temperature']
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                train_metrics[k] += v

        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # === VALIDATION ===
        model.eval()
        val_metrics = {'loss': 0, 'accuracy': 0, 'recall@10': 0}

        with torch.no_grad():
            for user_feats, item_feats in val_loader:
                user_feats = {k: v.to(device) for k, v in user_feats.items()}
                item_feats = {k: v.to(device) for k, v in item_feats.items()}

                user_emb, item_emb = model(user_feats, item_feats)
                loss, metrics = compute_contrastive_loss(user_emb, item_emb)

                for k, v in metrics.items():
                    val_metrics[k] += v

        for k in val_metrics:
            val_metrics[k] /= len(val_loader)

        # Logging
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.3f}, "
              f"Recall@10: {train_metrics['recall@10']:.3f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.3f}, "
              f"Recall@10: {val_metrics['recall@10']:.3f}")

        # Save best model
        if val_metrics['recall@10'] > best_recall:
            best_recall = val_metrics['recall@10']
            torch.save(model.state_dict(), 'best_two_tower.pt')

        scheduler.step()

    return model
```

---

### Phase 4: FAISS Integration & Evaluation (Days 7-8)

```python
import faiss
import numpy as np

def build_item_index(model, all_items, device, use_gpu=False):
    """
    Build FAISS index from all item embeddings.

    Args:
        model: Trained two-tower model
        all_items: List of item feature dicts
        device: torch device
        use_gpu: Whether to use GPU for FAISS

    Returns:
        index: FAISS index
        item_ids: Corresponding item IDs
    """
    model.eval()
    embeddings = []
    item_ids = []

    with torch.no_grad():
        for batch in tqdm(all_items, desc="Encoding items"):
            # Move to device
            item_feats = {k: v.to(device) for k, v in batch.items()}

            # Encode
            _, item_emb = model(None, item_feats)  # Only need item tower

            embeddings.append(item_emb.cpu().numpy())
            item_ids.extend(batch['item_id'].cpu().numpy())

    # Concatenate all embeddings
    embeddings = np.vstack(embeddings).astype('float32')
    item_ids = np.array(item_ids)

    print(f"Total items: {len(embeddings)}")
    print(f"Embedding dim: {embeddings.shape[1]}")

    # === BUILD FAISS INDEX ===
    dim = embeddings.shape[1]

    # Choose index type based on corpus size
    if len(embeddings) < 100000:
        # Small corpus: Use exact search
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)

    elif len(embeddings) < 10000000:
        # Medium corpus: Use IVF
        nlist = 1000  # Number of clusters
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)

        # Train clustering
        print("Training IVF index...")
        index.train(embeddings)

        # Set search parameters
        index.nprobe = 50  # Search 50 nearest clusters

    else:
        # Large corpus: Use IVF + PQ
        nlist = 4096
        m = 64  # Subvectors
        nbits = 8

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        print("Training IVF+PQ index...")
        index.train(embeddings)
        index.nprobe = 100

    # Add items to index
    print("Adding items to index...")
    index.add(embeddings)

    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Moving index to GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index, item_ids


def retrieve_top_k(model, user_features, index, item_ids, k=100, device='cuda'):
    """
    Retrieve top-K items for a user.

    Args:
        model: Trained two-tower model
        user_features: User feature dict
        index: FAISS index
        item_ids: Item IDs corresponding to index
        k: Number of items to retrieve

    Returns:
        top_k_items: Top-K item IDs
        scores: Similarity scores
    """
    model.eval()

    with torch.no_grad():
        # Move user features to device
        user_feats = {k: v.to(device) for k, v in user_features.items()}

        # Encode user
        user_emb, _ = model(user_feats, None)  # Only need user tower

        # Convert to numpy
        user_emb_np = user_emb.cpu().numpy().astype('float32')

        # Search
        scores, indices = index.search(user_emb_np, k)

        # Map indices to item IDs
        top_k_items = item_ids[indices[0]]

    return top_k_items, scores[0]


def evaluate_retrieval(model, test_data, index, item_ids, k_values=[10, 50, 100]):
    """
    Evaluate retrieval quality.

    Metrics: Recall@K, MRR, NDCG@K
    """
    model.eval()

    recalls = {k: [] for k in k_values}
    mrrs = []
    ndcgs = {k: [] for k in k_values}

    for user_features, ground_truth_items in tqdm(test_data, desc="Evaluating"):
        # Retrieve top-K
        max_k = max(k_values)
        top_k_items, scores = retrieve_top_k(
            model, user_features, index, item_ids, k=max_k
        )

        # Convert to set for fast lookup
        ground_truth_set = set(ground_truth_items)

        # === RECALL@K ===
        for k in k_values:
            retrieved = set(top_k_items[:k])
            recall = len(retrieved & ground_truth_set) / len(ground_truth_set)
            recalls[k].append(recall)

        # === MRR (Mean Reciprocal Rank) ===
        for i, item_id in enumerate(top_k_items):
            if item_id in ground_truth_set:
                mrrs.append(1.0 / (i + 1))
                break
        else:
            mrrs.append(0.0)

        # === NDCG@K ===
        for k in k_values:
            dcg = 0.0
            for i, item_id in enumerate(top_k_items[:k]):
                if item_id in ground_truth_set:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

            # Ideal DCG (all relevant items at top)
            num_relevant = min(len(ground_truth_set), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs[k].append(ndcg)

    # Average metrics
    results = {
        f'recall@{k}': np.mean(recalls[k]) for k in k_values
    }
    results['mrr'] = np.mean(mrrs)
    results.update({
        f'ndcg@{k}': np.mean(ndcgs[k]) for k in k_values
    })

    return results
```

---

## Expected Results

| Metric | Target | Excellent |
|--------|--------|-----------|
| Recall@10 | >30% | >50% |
| Recall@100 | >60% | >80% |
| Recall@1000 | >85% | >95% |
| MRR | >0.20 | >0.35 |
| NDCG@100 | >0.40 | >0.60 |
| Latency (1M items) | <10ms | <5ms |
| Index size | <10GB | <5GB |

---

## Success Criteria

**Theory**:
- [ ] Explain two-tower architecture and why towers are independent
- [ ] Describe contrastive learning and in-batch negatives
- [ ] Explain ANN algorithms (IVF, HNSW, PQ) and trade-offs
- [ ] Calculate and interpret retrieval metrics (Recall@K, MRR, NDCG)
- [ ] Discuss billion-scale deployment challenges

**Implementation**:
- [ ] Build two-tower model with proper normalization
- [ ] Train with contrastive loss and achieve Recall@100 >60%
- [ ] Integrate FAISS for fast similarity search
- [ ] Implement hard negative mining
- [ ] Optimize index with quantization

**Production**:
- [ ] Build FAISS index for 100K+ items
- [ ] Achieve <10ms retrieval latency
- [ ] Design index update strategy (dual index or incremental)
- [ ] Benchmark different index types (Flat, IVF, HNSW, PQ)

---

## Resources

**Papers**:
- [Sampling-Bias-Corrected Neural Modeling (Google, 2019)](https://research.google/pubs/pub48840/)
- [FAISS: Efficient Similarity Search (Meta, 2017)](https://arxiv.org/abs/1702.08734)
- [HNSW: Efficient and robust ANN using hierarchical navigable small world graphs (2016)](https://arxiv.org/abs/1603.09320)
- [Embedding-based Retrieval (Airbnb, 2018)](https://dl.acm.org/doi/10.1145/3219819.3219885)

**Code**:
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

**Ready to build billion-scale search? Let's go! 🔍**
