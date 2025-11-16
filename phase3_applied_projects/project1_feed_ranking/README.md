# Project 1: Feed Ranking Model 🎯

> **Time:** 2-3 weeks
> **Difficulty:** Advanced
> **Goal:** Build a production-grade feed ranking system like Meta's News Feed

---

## Why This Project Matters

Every day, billions of people scroll through social media feeds. Behind every swipe lies one of the most sophisticated ML systems in the world: **feed ranking**. This project teaches you to build the core technology that:

- **Powers engagement**: Facebook's News Feed drives 2+ billion daily active users
- **Generates revenue**: Feed ranking directly impacts ad placement worth $100B+ annually
- **Scales massively**: Ranks trillions of posts per day with sub-100ms latency
- **Drives business**: 10% improvement in ranking = billions in revenue

**Real-world impact:**
- Meta's News Feed and Instagram Feed use Wide & Deep variants
- LinkedIn Feed, Twitter Timeline, TikTok For You Page all use similar architectures
- E-commerce product ranking (Amazon, Alibaba)
- Content recommendation (YouTube, Netflix)

Feed ranking is a **two-stage system**:
1. **Candidate Generation** (Retrieval): Narrow millions of posts to hundreds (~1000)
2. **Ranking** (This project): Precisely rank those hundreds to show top ~50

This project focuses on **Stage 2: Ranking**, where you predict how much a user will engage with each candidate post.

---

## The Big Picture

### The Problem

**Input**: User U sees candidate posts P₁, P₂, ..., Pₙ
**Output**: Ranked list showing posts U will engage with most

**Challenges:**
- **Billions of users** × **millions of posts** = impossible to score all combinations
- **Multiple objectives**: Clicks, likes, comments, shares, dwell time, hiding posts
- **Real-time constraints**: Must rank in <50ms
- **Data sparsity**: Most user-post pairs never interact (99.9%+ are negatives)
- **Diversity**: Can't show all posts from one friend or topic
- **Fairness**: Balance engagement with content quality, misinformation, etc.

### The Solution: Wide & Deep Learning

Google introduced **Wide & Deep** in 2016 for app recommendations. Meta adapted it for feed ranking. The key insight:

**Wide Component (Memorization)**:
- Linear model on cross-product features
- Learns specific rules: "User123 always clicks Friend456's posts"
- Handles exceptions and specific patterns

**Deep Component (Generalization)**:
- Neural network with embeddings
- Learns general patterns: "Users who like sports engage with sports content"
- Generalizes to new users/posts

**Together**: Best of both worlds—handles both specific memorization and general patterns.

---

## Deep Dive: Feed Ranking Theory

### 1. Learning to Rank (LTR) Fundamentals

Feed ranking is a **Learning to Rank** problem. There are three main approaches:

#### Pointwise Approach
Treat ranking as **regression or classification** on individual items.

**Method**: Predict engagement score for each post independently.
```
P(engage | user, post) = σ(f(user_features, post_features))
```

**Loss**: Binary cross-entropy or MSE
```python
loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

**Pros**: Simple, easy to implement
**Cons**: Ignores relative ordering between posts

**When to use**: Initial baseline, when you have strong engagement signals

#### Pairwise Approach
Learn to predict **relative order** between pairs of items.

**Method**: For each user, compare pairs of posts (one clicked, one not)
```
P(post_i > post_j) = σ(f(user, post_i) - f(user, post_j))
```

**Loss**: Pairwise hinge loss or pairwise logistic loss
```python
loss = max(0, margin - (score_pos - score_neg))
```

**Pros**: Directly optimizes ranking order
**Cons**: Requires O(n²) pairs, can be slow

**When to use**: When you have clear positive/negative examples

#### Listwise Approach
Optimize **entire ranking** as a list.

**Method**: Model the probability of a permutation or use metrics like NDCG
```
P(ranking) = softmax over all possible rankings
```

**Loss**: ListNet, ListMLE, or direct NDCG optimization
```python
loss = -log(∏ᵢ P(item_i | items_ranked_higher))
```

**Pros**: Directly optimizes ranking metrics
**Cons**: Complex, computationally expensive

**When to use**: When ranking quality is critical, have computational resources

**In practice**: Most production systems use **pointwise** for simplicity and speed, but incorporate pairwise ideas through negative sampling.

---

### 2. Wide & Deep Architecture in Detail

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT FEATURES                          │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ User Features│ Post Features│Context Feats │Cross Features │
│  - user_id   │  - post_id   │  - hour      │ user×post_type│
│  - age       │  - author_id │  - device    │ user×author   │
│  - #friends  │  - post_type │  - day       │    ...        │
└──────────────┴──────────────┴──────────────┴───────────────┘
        │              │              │              │
        └──────┬───────┴──────┬───────┘              │
               ▼              ▼                       ▼
        ┌─────────────────────────┐          ┌──────────────┐
        │   DEEP COMPONENT        │          │WIDE COMPONENT│
        ├─────────────────────────┤          ├──────────────┤
        │ Embedding Layers        │          │Cross Product │
        │  user_id → 128-dim      │          │Features      │
        │  post_id → 128-dim      │          │     ↓        │
        │  author_id → 64-dim     │          │Linear Layer  │
        ├─────────────────────────┤          └──────────────┘
        │ Concatenate:            │                  │
        │  [embeddings,           │                  │
        │   dense_features]       │                  │
        ├─────────────────────────┤                  │
        │ MLP Layers:             │                  │
        │  512 → ReLU → Dropout   │                  │
        │  256 → ReLU → Dropout   │                  │
        │  128 → ReLU             │                  │
        └─────────────┬───────────┘                  │
                      │                              │
                      └──────────┬───────────────────┘
                                 ▼
                        ┌────────────────┐
                        │  COMBINATION   │
                        │  [deep, wide]  │
                        └────────┬───────┘
                                 │
                    ┌────────────┴───────────┐
                    ▼                        ▼
            ┌──────────────┐        ┌──────────────┐
            │  CTR HEAD    │        │DWELL TIME    │
            │  (sigmoid)   │        │HEAD (ReLU)   │
            │ P(click)     │        │ seconds      │
            └──────────────┘        └──────────────┘
```

#### Deep Component Details

**Why Embeddings?**
- **Dimensionality reduction**: 10M users → 128-dim vectors
- **Similarity learning**: Similar users get similar embeddings
- **Feature sharing**: Same user embedding used across all posts

**Embedding dimensions** (rule of thumb):
```python
embedding_dim = min(50, cardinality**0.25 * 1.6)

# Examples:
# 10K users → 128-dim
# 1M posts → 178-dim
# 100K authors → 106-dim
```

**MLP Design**:
- Start wide (512), gradually narrow (256 → 128)
- ReLU activation for non-linearity
- Dropout (0.1-0.3) for regularization
- Batch normalization for stable training

**Feature interactions**:
The deep network automatically learns feature interactions through hidden layers:
- Layer 1: Pairwise interactions (user × post type)
- Layer 2: Three-way interactions (user × post × time)
- Layer 3+: Higher-order interactions

#### Wide Component Details

**Cross Features** (manually engineered):
```python
cross_features = [
    'user_id × post_type',           # User preference for content type
    'user_id × author_id',            # User-creator affinity
    'user_age_group × post_category', # Demographic preferences
    'device × hour_of_day',           # Usage patterns
]
```

**Why keep Wide?**
- **Exception handling**: "User X never clicks video posts" even if deep model predicts high
- **Quick adaptation**: Linear model updates faster than deep network
- **Interpretability**: Easy to see which cross-features matter
- **Memorization**: Remembers specific user-item combinations

**Implementation**:
```python
# Wide component is just a linear layer on cross features
wide_logits = W @ cross_features + b
```

---

### 3. Multi-Task Learning (MTL)

Feed ranking optimizes **multiple objectives** simultaneously:

| Task | Type | Why It Matters |
|------|------|----------------|
| Click (CTR) | Binary | Basic engagement signal |
| Like | Binary | Stronger positive signal |
| Comment | Binary | Highest engagement |
| Share | Binary | Viral potential |
| Dwell Time | Regression | Time spent = quality |
| Hide/Report | Binary | Negative feedback |

**Multi-Task Loss**:
```
L_total = α₁·L_CTR + α₂·L_like + α₃·L_comment + α₄·L_share + α₅·L_dwell + α₆·L_hide
```

**Why Multi-Task?**

1. **Shared representations**: Features useful for predicting clicks also help predict likes
2. **Data efficiency**: Limited labels for some tasks (shares), abundant for others (clicks)
3. **Regularization**: Learning multiple tasks prevents overfitting to one
4. **Holistic optimization**: Balance different engagement types

**Task Weighting Strategies**:

**Fixed weights**:
```python
alpha_ctr = 1.0
alpha_dwell = 0.1  # Scale down to match CTR loss magnitude
```

**Gradient-based balancing** (GradNorm):
Automatically balance task weights based on gradient magnitudes

**Uncertainty weighting** (Kendall et al.):
```python
# Learn task uncertainties
L = L₁/(2σ₁²) + L₂/(2σ₂²) + log(σ₁σ₂)
```

**In this project**: We'll use two tasks (CTR + dwell time) with fixed weights.

---

### 4. Negative Sampling Strategies

**The Imbalance Problem**:
- **Positive rate**: ~3-5% of shown posts are clicked
- **95%+ are negatives**: Model can achieve 95% accuracy by predicting "no click" always
- **Training inefficiency**: Most negatives are easy examples

**Solution**: Sample negatives intelligently

#### Random Sampling
```python
# Sample k negatives for each positive
positives = data[data['clicked'] == 1]
negatives = data[data['clicked'] == 0].sample(n=len(positives) * k)
```

**Pros**: Simple, unbiased
**Cons**: Wastes time on easy negatives

#### Hard Negative Mining
```python
# Sample negatives with high predicted scores (but not clicked)
model_scores = model(user, all_posts)
hard_negatives = not_clicked_posts.sort_by(model_scores).top_k(k)
```

**Pros**: Focuses on hard examples, faster learning
**Cons**: Can be unstable, requires periodic re-mining

#### In-Batch Negatives
```python
# Use other examples in batch as negatives
batch = [(user₁, post₁), (user₂, post₂), ..., (userₙ, postₙ)]
# For user₁, post₁ is positive, {post₂, ..., postₙ} are negatives
```

**Pros**: Efficient, no extra sampling needed
**Cons**: Assumes batch items are random negatives

**Best practice**: Combine strategies
- Start with random sampling
- Add hard negative mining after initial training
- Use in-batch negatives for efficiency

---

### 5. Feature Engineering for Ranking

Feed ranking features fall into four categories:

#### User Features
```python
user_features = {
    # Identity
    'user_id': 'categorical',           # Embedding
    'age': 'numeric',                   # Normalize
    'gender': 'categorical',

    # Historical behavior (aggregated)
    'num_friends': 'numeric',
    'avg_posts_per_day': 'numeric',
    'engagement_rate': 'numeric',       # clicks / impressions

    # Recent activity (temporal)
    'posts_last_hour': 'numeric',
    'session_length': 'numeric',

    # Interests (learned)
    'top_categories': 'multi-hot',      # [sports, tech, fashion]
    'interest_embeddings': 'dense',     # From other models
}
```

#### Post Features
```python
post_features = {
    # Identity
    'post_id': 'categorical',
    'author_id': 'categorical',
    'post_type': 'categorical',         # photo/video/text/link

    # Content
    'num_hashtags': 'numeric',
    'num_mentions': 'numeric',
    'has_location': 'binary',
    'text_length': 'numeric',

    # Popularity (causal!)
    'num_likes': 'numeric',             # Be careful: can cause feedback loops
    'num_comments': 'numeric',
    'engagement_rate': 'numeric',

    # Recency
    'post_age_hours': 'numeric',        # Freshness
    'author_last_post_hours': 'numeric',
}
```

#### Context Features
```python
context_features = {
    # Time
    'hour_of_day': 'categorical',       # Peak usage times
    'day_of_week': 'categorical',       # Weekend vs weekday
    'is_holiday': 'binary',

    # Device
    'device_type': 'categorical',       # mobile/desktop/tablet
    'connection_speed': 'categorical',  # wifi/4g/3g

    # Location
    'country': 'categorical',
    'timezone': 'categorical',
}
```

#### Cross Features (for Wide component)
```python
cross_features = [
    ('user_id', 'post_type'),           # User content preferences
    ('user_id', 'author_id'),           # User-creator affinity
    ('user_age_group', 'post_category'),# Demographic preferences
    ('device', 'hour'),                 # Usage patterns
    ('user_id', 'post_category'),       # Interest matching
]
```

**Feature Normalization**:
```python
# Continuous features
age_normalized = (age - mean) / std

# Log transform for skewed distributions
num_friends_log = log(num_friends + 1)

# Clipping outliers
num_likes_clipped = min(num_likes, 99th_percentile)
```

---

### 6. Ranking Metrics

Unlike classification, ranking cares about **order**, not just correctness.

#### AUC-ROC (Area Under Curve)
Measures how well the model separates positives from negatives.

```
AUC = P(score(positive) > score(negative))
```

**Interpretation**:
- AUC = 0.5: Random ranking
- AUC = 0.7: Good
- AUC = 0.8: Very good
- AUC = 0.9+: Excellent

**Why it matters**: Robust to class imbalance, threshold-independent

**Limitation**: Doesn't care about top positions (treats all ranks equally)

#### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality of **top K results**.

```
DCG@K = Σᵢ₌₁ᴷ (2^{relevance_i} - 1) / log₂(i + 1)

NDCG@K = DCG@K / IDCG@K  (normalized by ideal DCG)
```

**Why the log discount?**
- Position 1 vs 2: Big difference
- Position 10 vs 11: Small difference
- Models human behavior: we care most about top results

**Example**:
```
Ranking: [relevant, relevant, not_relevant, relevant]
DCG = (2¹-1)/log₂(2) + (2¹-1)/log₂(3) + (2⁰-1)/log₂(4) + (2¹-1)/log₂(5)
    = 1/1 + 1/1.58 + 0/2 + 1/2.32
    = 1.00 + 0.63 + 0 + 0.43 = 2.06

Ideal: [relevant, relevant, relevant, not_relevant]
IDCG = 1.00 + 0.63 + 0.50 + 0 = 2.13

NDCG = 2.06 / 2.13 = 0.97 (very good!)
```

**Why it matters**:
- **Graded relevance**: Can handle multiple levels (click, like, share)
- **Position-aware**: Rewards good items at top
- **Industry standard**: Used by Google, Meta, etc.

#### MAP (Mean Average Precision)
Average precision across all users.

```
AP = (1/num_relevant) × Σ P(k) × rel(k)

where P(k) = precision at position k
```

**Why it matters**: Emphasizes precision at top positions

#### Calibration
Are predicted probabilities accurate?

```
If model predicts P(click) = 0.1 for 1000 posts,
~100 should actually be clicked
```

**Calibration plot**:
```
x-axis: Predicted probability bins [0-0.1, 0.1-0.2, ...]
y-axis: Actual click rate in each bin
Perfect: y = x (diagonal line)
```

**Why it matters**: Business decisions rely on probabilities (e.g., ad pricing)

---

### 7. Production Ranking Systems at Meta Scale

#### System Architecture
```
┌─────────────────────────────────────────────────────┐
│                    USER REQUEST                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            CANDIDATE GENERATION (Stage 1)           │
│  - Retrieve from millions → ~1000 candidates        │
│  - Use efficient retrieval (two-tower, hashing)     │
│  - Latency budget: ~20ms                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 RANKING (Stage 2)                   │
│  - Score ~1000 candidates with complex model        │
│  - Wide & Deep with all features                    │
│  - Multi-task predictions                           │
│  - Latency budget: ~30ms                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│             RE-RANKING & DIVERSIFICATION            │
│  - Apply business rules                             │
│  - Diversity (don't show 10 posts from one friend)  │
│  - Fairness constraints                             │
│  - Final selection: ~50 posts                       │
│  - Latency budget: ~10ms                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                RESPONSE TO USER                     │
└─────────────────────────────────────────────────────┘

Total latency budget: ~60ms
```

#### Scalability Challenges

**Challenge 1: Embedding Table Size**
```
10M users × 128 dims × 4 bytes = 5GB
100M posts × 128 dims × 4 bytes = 51GB
Total: 56GB just for embeddings!
```

**Solutions**:
- **Hashing trick**: Hash user_id to smaller space
- **Embedding sharding**: Split embeddings across machines
- **Quantization**: Use int8 instead of float32 (4x smaller)
- **Feature hashing**: Hash rare categories to shared embeddings

**Challenge 2: Training Data Volume**
```
2B users × 100 impressions/day × 365 days = 73 trillion examples/year
```

**Solutions**:
- **Sampling**: Don't use all negatives
- **Streaming**: Process data in streams, not batches
- **Distributed training**: Use FSDP across 100+ GPUs
- **Feature stores**: Precompute and cache features

**Challenge 3: Inference Latency**
```
Target: 30ms to score 1000 posts = 30μs per post
```

**Solutions**:
- **Model quantization**: INT8 inference (3-4x faster)
- **Batching**: Score all 1000 posts in parallel
- **Model distillation**: Train smaller student from large teacher
- **Feature caching**: Cache user features (don't recompute)
- **ONNX/TensorRT**: Optimize computation graph

**Challenge 4: Model Freshness**
```
User interests change daily, posts expire quickly
```

**Solutions**:
- **Incremental learning**: Update model with new data continuously
- **Online learning**: Update in real-time (tricky, can be unstable)
- **Frequent retraining**: Retrain daily or even hourly
- **A/B testing**: Shadow mode for new models

---

### 8. A/B Testing and Online Metrics

You can't just deploy a new ranking model and hope it works. You need **rigorous A/B testing**.

#### A/B Test Setup
```
Control (A): Current ranking model
Treatment (B): New ranking model

Users randomly assigned 50-50
Run for 1-2 weeks
Monitor metrics
```

#### Online Metrics (What Actually Matters)

**Engagement metrics**:
- Click-through rate (CTR)
- Like rate
- Comment rate
- Share rate
- Dwell time
- Posts per session

**Business metrics**:
- Daily active users (DAU)
- Session length
- Ad revenue per user
- User retention (7-day, 30-day)

**Quality metrics**:
- Hide/report rate
- Survey satisfaction scores
- Content diversity
- Misinformation exposure

#### Offline vs Online Metrics

| Offline (Training) | Online (Production) |
|-------------------|-------------------|
| AUC-ROC | CTR |
| NDCG@K | Engagement rate |
| Loss | Revenue |
| Fast (minutes) | Slow (weeks) |
| Cheap | Expensive |
| **Use for**: Iteration | **Use for**: Decisions |

**The disconnect**: High offline AUC doesn't guarantee high online CTR!

**Why?**
- **Distribution shift**: Training data is historical, production is live
- **Feedback loops**: Model affects what users see, which affects future training data
- **Position bias**: Offline assumes random ordering, online has algorithmic ordering
- **User diversity**: Model might work well on average but poorly for some segments

**Best practice**: Use offline metrics for fast iteration, online A/B tests for decisions.

---

## Implementation Guide

### Phase 1: Data Pipeline (Week 1, Days 1-2)

#### Understanding the Dataset

We'll use a synthetic dataset simulating Meta-scale interactions:

```
Training: 1M user-post pairs
Validation: 100K pairs
Test: 100K pairs

Users: 10K unique
Posts: 100K unique
Authors: 50K unique
```

**Data Schema**:
```python
{
    'user_id': int,              # 0-9999
    'age': int,                  # 18-80
    'gender': str,               # 'M', 'F', 'O'
    'num_friends': int,          # 10-5000
    'post_id': int,              # 0-99999
    'author_id': int,            # 0-49999
    'post_type': str,            # 'photo', 'video', 'text', 'link'
    'num_likes': int,            # 0-10000
    'num_comments': int,         # 0-1000
    'post_age_hours': float,     # 0-168 (1 week)
    'hour_of_day': int,          # 0-23
    'day_of_week': int,          # 0-6
    'device_type': str,          # 'mobile', 'desktop'
    'clicked': int,              # 0 or 1 (TARGET)
    'dwell_time_seconds': float, # 0-300 (TARGET, only if clicked=1)
}
```

#### Feature Processing Pipeline

**1. Categorical Encoding**:
```python
# Create vocabulary mappings
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
post_id_to_idx = {post_id: idx for idx, post_id in enumerate(unique_posts)}

# Apply encoding
encoded_user_id = user_id_to_idx[user_id]
```

**2. Numerical Normalization**:
```python
# Standard scaling
age_normalized = (age - age_mean) / age_std

# Log transformation for skewed features
num_friends_log = np.log1p(num_friends)  # log(1 + x)

# Min-max scaling
post_age_normalized = post_age_hours / 168.0  # Normalize to [0, 1]
```

**3. Cross Features** (for Wide component):
```python
# String concatenation for hashing
user_post_type = f"{user_id}_{post_type}"
user_author = f"{user_id}_{author_id}"
```

**Implementation**:
```python
class FeedDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)

        # Compute normalization stats from training data
        self.compute_stats()

    def compute_stats(self):
        """Compute mean/std for numerical features"""
        self.age_mean = self.data['age'].mean()
        self.age_std = self.data['age'].std()
        # ... compute for all numerical features

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Categorical features
        user_id = row['user_id']
        post_id = row['post_id']
        author_id = row['author_id']
        post_type = self.post_type_to_idx[row['post_type']]
        device_type = self.device_to_idx[row['device_type']]

        # Numerical features (normalized)
        age = (row['age'] - self.age_mean) / self.age_std
        num_friends = np.log1p(row['num_friends'])
        num_likes = np.log1p(row['num_likes'])
        # ... process all numerical features

        # Combine into feature dict
        features = {
            'categorical': {
                'user_id': user_id,
                'post_id': post_id,
                'author_id': author_id,
                'post_type': post_type,
                'device_type': device_type,
            },
            'numerical': torch.tensor([
                age, num_friends, num_likes, num_comments,
                post_age_hours, hour_of_day, day_of_week
            ], dtype=torch.float32),
            'cross': self.create_cross_features(row)
        }

        # Labels
        labels = {
            'clicked': row['clicked'],
            'dwell_time': row['dwell_time_seconds']
        }

        return features, labels
```

**Key implementation details**:
- ✅ Compute normalization stats on **training set only**
- ✅ Apply same stats to validation and test
- ✅ Handle missing values (fill with median or special token)
- ✅ Use efficient data loading (HDF5 or Parquet for large datasets)

---

### Phase 2: Model Architecture (Week 1, Days 3-5)

#### Complete Wide & Deep Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeepRanker(nn.Module):
    """
    Wide & Deep model for feed ranking.

    Architecture:
    - Deep: Embeddings + MLP for generalization
    - Wide: Linear model on cross features for memorization
    - Multi-task: Separate heads for CTR and dwell time
    """

    def __init__(self, config):
        super().__init__()

        # === EMBEDDINGS ===
        self.user_embedding = nn.Embedding(
            num_embeddings=config['num_users'],
            embedding_dim=config['user_emb_dim']
        )
        self.post_embedding = nn.Embedding(
            num_embeddings=config['num_posts'],
            embedding_dim=config['post_emb_dim']
        )
        self.author_embedding = nn.Embedding(
            num_embeddings=config['num_authors'],
            embedding_dim=config['author_emb_dim']
        )

        # Small categorical features
        self.post_type_embedding = nn.Embedding(4, 8)    # 4 types
        self.device_embedding = nn.Embedding(2, 8)       # 2 devices

        # === DEEP COMPONENT ===
        # Compute total input dimension
        deep_input_dim = (
            config['user_emb_dim'] +
            config['post_emb_dim'] +
            config['author_emb_dim'] +
            8 + 8 +  # post_type and device embeddings
            config['num_numerical_features']
        )

        self.deep_layers = nn.Sequential(
            nn.Linear(deep_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # === WIDE COMPONENT ===
        # Linear layer on cross features
        self.wide_layer = nn.Linear(config['num_cross_features'], 1)

        # === OUTPUT HEADS ===
        # CTR head (binary classification)
        self.ctr_head = nn.Linear(128 + 1, 1)  # 128 (deep) + 1 (wide)

        # Dwell time head (regression)
        self.dwell_head = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, features):
        """
        Args:
            features: Dict with keys:
                - categorical: Dict of categorical features
                - numerical: Tensor (batch_size, num_numerical)
                - cross: Tensor (batch_size, num_cross)

        Returns:
            ctr_logits: (batch_size, 1) - logits for click prediction
            dwell_pred: (batch_size, 1) - predicted dwell time
        """
        # === EMBEDDINGS ===
        user_emb = self.user_embedding(features['categorical']['user_id'])
        post_emb = self.post_embedding(features['categorical']['post_id'])
        author_emb = self.author_embedding(features['categorical']['author_id'])
        post_type_emb = self.post_type_embedding(features['categorical']['post_type'])
        device_emb = self.device_embedding(features['categorical']['device_type'])

        # === DEEP COMPONENT ===
        # Concatenate all deep features
        deep_input = torch.cat([
            user_emb,
            post_emb,
            author_emb,
            post_type_emb,
            device_emb,
            features['numerical']
        ], dim=1)

        deep_output = self.deep_layers(deep_input)  # (batch_size, 128)

        # === WIDE COMPONENT ===
        wide_output = self.wide_layer(features['cross'])  # (batch_size, 1)

        # === COMBINE ===
        combined = torch.cat([deep_output, wide_output], dim=1)

        # === MULTI-TASK OUTPUTS ===
        ctr_logits = self.ctr_head(combined)      # (batch_size, 1)
        dwell_pred = self.dwell_head(combined)    # (batch_size, 1)
        dwell_pred = F.relu(dwell_pred)           # Dwell time must be positive

        return ctr_logits, dwell_pred
```

**Design Choices Explained**:

1. **Embedding Dimensions**:
   - User/Post: 128-dim (large cardinality, complex patterns)
   - Author: 64-dim (medium cardinality)
   - Post type/Device: 8-dim (low cardinality)

2. **MLP Architecture**:
   - Start wide (512), gradually narrow (256 → 128)
   - Dropout for regularization (higher in early layers)
   - BatchNorm for stable training

3. **Why ReLU?**:
   - Non-linearity enables learning complex patterns
   - Simple, fast, works well

4. **Why separate heads?**:
   - Different tasks need different transformations
   - Allows task-specific fine-tuning

---

### Phase 3: Training Loop (Week 2, Days 1-3)

#### Multi-Task Loss Implementation

```python
def compute_multitask_loss(ctr_logits, ctr_labels, dwell_pred, dwell_labels, alpha=1.0, beta=0.1):
    """
    Combined loss for CTR and dwell time.

    Args:
        ctr_logits: (batch_size, 1) - raw logits
        ctr_labels: (batch_size,) - binary labels (0 or 1)
        dwell_pred: (batch_size, 1) - predicted dwell time
        dwell_labels: (batch_size,) - actual dwell time (0 if not clicked)
        alpha: Weight for CTR loss
        beta: Weight for dwell time loss

    Returns:
        total_loss: Weighted combination
        loss_dict: Individual losses for logging
    """
    # === CTR LOSS (Binary Cross-Entropy) ===
    ctr_loss = F.binary_cross_entropy_with_logits(
        ctr_logits.squeeze(),
        ctr_labels.float()
    )

    # === DWELL TIME LOSS (MSE, only on clicked examples) ===
    # Only compute dwell time loss for clicked examples
    clicked_mask = ctr_labels == 1

    if clicked_mask.sum() > 0:
        dwell_loss = F.mse_loss(
            dwell_pred.squeeze()[clicked_mask],
            dwell_labels[clicked_mask]
        )
    else:
        dwell_loss = torch.tensor(0.0, device=ctr_logits.device)

    # === COMBINED LOSS ===
    total_loss = alpha * ctr_loss + beta * dwell_loss

    return total_loss, {
        'total': total_loss.item(),
        'ctr': ctr_loss.item(),
        'dwell': dwell_loss.item()
    }
```

**Why only compute dwell loss on clicked examples?**
- Dwell time is 0 for non-clicked posts
- No signal to learn from
- Would just bias model to predict low dwell times

**Choosing alpha and beta**:
```python
# Option 1: Fixed weights (simple)
alpha = 1.0
beta = 0.1  # Scale down because dwell_loss is larger

# Option 2: Match loss magnitudes
alpha = 1.0
beta = initial_ctr_loss / initial_dwell_loss  # Make them similar scale

# Option 3: Task-specific importance
alpha = 2.0  # CTR is more important
beta = 1.0
```

#### Complete Training Loop

```python
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move to device
        features = move_to_device(features, device)
        ctr_labels = labels['clicked'].to(device)
        dwell_labels = labels['dwell_time'].to(device)

        # Forward pass
        ctr_logits, dwell_pred = model(features)

        # Compute loss
        loss, loss_dict = compute_multitask_loss(
            ctr_logits, ctr_labels,
            dwell_pred, dwell_labels,
            alpha=1.0, beta=0.1
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Logging
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"CTR: {loss_dict['ctr']:.4f}, "
                  f"Dwell: {loss_dict['dwell']:.4f}")

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_ctr_preds = []
    all_ctr_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = move_to_device(features, device)
            ctr_labels = labels['clicked'].to(device)
            dwell_labels = labels['dwell_time'].to(device)

            # Forward pass
            ctr_logits, dwell_pred = model(features)

            # Compute loss
            loss, _ = compute_multitask_loss(
                ctr_logits, ctr_labels,
                dwell_pred, dwell_labels
            )

            total_loss += loss.item()

            # Collect predictions for metrics
            ctr_preds = torch.sigmoid(ctr_logits).cpu().numpy()
            all_ctr_preds.extend(ctr_preds)
            all_ctr_labels.extend(ctr_labels.cpu().numpy())

    # Compute metrics
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_ctr_labels, all_ctr_preds)

    return total_loss / len(val_loader), auc


def train_model(model, train_loader, val_loader, config):
    """Complete training pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_auc = 0
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, device)

        # Logging
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping!")
                break

    return model
```

**Key Training Techniques**:

1. **Gradient Clipping**: Prevents exploding gradients (common in deep networks)
2. **Learning Rate Scheduling**: Reduce LR when validation loss plateaus
3. **Early Stopping**: Stop if validation AUC doesn't improve for N epochs
4. **Mixed Precision** (optional): Use `torch.cuda.amp` for faster training

---

### Phase 4: Evaluation & Metrics (Week 2, Days 4-5)

#### Implementing Ranking Metrics

```python
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def compute_ranking_metrics(predictions, labels, k_values=[10, 50, 100]):
    """
    Compute comprehensive ranking metrics.

    Args:
        predictions: List of (user_id, [(post_id, score), ...]) tuples
        labels: List of (user_id, [clicked_post_ids]) tuples
        k_values: List of K values for metrics like NDCG@K

    Returns:
        metrics: Dict of metric values
    """
    metrics = {}

    # === AUC-ROC ===
    all_scores = []
    all_labels = []
    for (user_id, post_scores), (_, clicked_posts) in zip(predictions, labels):
        for post_id, score in post_scores:
            all_scores.append(score)
            all_labels.append(1 if post_id in clicked_posts else 0)

    metrics['auc'] = roc_auc_score(all_labels, all_scores)
    metrics['logloss'] = log_loss(all_labels, all_scores)

    # === NDCG@K ===
    for k in k_values:
        ndcg_scores = []
        for (user_id, post_scores), (_, clicked_posts) in zip(predictions, labels):
            # Get top-k predictions
            top_k = sorted(post_scores, key=lambda x: x[1], reverse=True)[:k]

            # Compute DCG
            dcg = 0
            for rank, (post_id, score) in enumerate(top_k, start=1):
                relevance = 1 if post_id in clicked_posts else 0
                dcg += (2**relevance - 1) / np.log2(rank + 1)

            # Compute IDCG (ideal DCG)
            num_relevant = min(len(clicked_posts), k)
            idcg = sum((2**1 - 1) / np.log2(rank + 1)
                      for rank in range(1, num_relevant + 1))

            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)

    # === MAP (Mean Average Precision) ===
    ap_scores = []
    for (user_id, post_scores), (_, clicked_posts) in zip(predictions, labels):
        sorted_posts = sorted(post_scores, key=lambda x: x[1], reverse=True)

        num_relevant = 0
        precision_sum = 0
        for rank, (post_id, score) in enumerate(sorted_posts, start=1):
            if post_id in clicked_posts:
                num_relevant += 1
                precision_at_k = num_relevant / rank
                precision_sum += precision_at_k

        ap = precision_sum / len(clicked_posts) if len(clicked_posts) > 0 else 0
        ap_scores.append(ap)

    metrics['map'] = np.mean(ap_scores)

    return metrics
```

#### Calibration Analysis

```python
def plot_calibration(predictions, labels, n_bins=10):
    """
    Plot calibration curve.

    Perfectly calibrated: predicted probability matches actual rate
    """
    import matplotlib.pyplot as plt

    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    predicted_probs = []
    actual_rates = []

    for i in range(n_bins):
        # Find predictions in this bin
        in_bin = (predictions >= bins[i]) & (predictions < bins[i+1])

        if in_bin.sum() > 0:
            predicted_probs.append(predictions[in_bin].mean())
            actual_rates.append(labels[in_bin].mean())

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_probs, actual_rates, 'o-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Click Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

### Phase 5: Advanced Features (Week 3)

#### Hard Negative Mining

```python
class HardNegativeMiner:
    """
    Mine hard negatives: posts with high predicted scores but not clicked.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def mine_negatives(self, user_features, candidate_posts, k=10):
        """
        Find top-k hard negatives for a user.

        Args:
            user_features: Features for one user
            candidate_posts: List of post features (not clicked)
            k: Number of hard negatives to return

        Returns:
            hard_negatives: Top-k posts by model score
        """
        self.model.eval()

        with torch.no_grad():
            # Score all candidates
            scores = []
            for post in candidate_posts:
                features = combine_user_post_features(user_features, post)
                ctr_logit, _ = self.model(features)
                score = torch.sigmoid(ctr_logit).item()
                scores.append((post, score))

            # Sort by score (descending)
            sorted_posts = sorted(scores, key=lambda x: x[1], reverse=True)

            # Return top-k
            hard_negatives = [post for post, score in sorted_posts[:k]]

        return hard_negatives
```

**When to use hard negatives?**
- After initial training converges
- Periodically (e.g., every 5 epochs)
- Balance with random negatives (50-50 mix)

#### Feature Importance Analysis

```python
def analyze_feature_importance(model, val_loader, device):
    """
    Analyze which features matter most.

    Method: Remove each feature and measure AUC drop.
    """
    baseline_auc = compute_auc(model, val_loader, device)

    feature_importance = {}

    for feature_name in ['user_id', 'post_id', 'author_id', 'post_age', ...]:
        # Create modified loader with feature removed
        modified_loader = remove_feature(val_loader, feature_name)

        # Compute AUC without this feature
        modified_auc = compute_auc(model, modified_loader, device)

        # Importance = drop in AUC
        importance = baseline_auc - modified_auc
        feature_importance[feature_name] = importance

    # Sort by importance
    sorted_features = sorted(feature_importance.items(),
                           key=lambda x: x[1], reverse=True)

    print("Feature Importance:")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")

    return feature_importance
```

---

## Common Pitfalls & How to Avoid Them

### 1. Data Leakage

**Problem**: Using future information in training
```python
# BAD: Using post's current popularity
features['num_likes'] = post.current_likes  # Includes future likes!

# GOOD: Use popularity at impression time
features['num_likes'] = post.likes_at_impression_time
```

**How to avoid**:
- Use point-in-time features only
- Careful with aggregated features (use sliding windows)
- Temporal validation split (train on past, validate on future)

### 2. Popularity Bias

**Problem**: Model just learns to rank popular posts
```python
# Model learns: high num_likes → rank higher
# But popular posts get more exposure → more likes → feedback loop
```

**How to avoid**:
- Debias features (residualize num_likes)
- Use position-aware training
- Exploration: show some random posts

### 3. Cold Start

**Problem**: New users/posts have no history
```python
# New user: no user_id embedding trained
# New post: no engagement history
```

**How to avoid**:
- Content-based features (text, images)
- Use author embeddings for new posts
- Demographic features for new users
- Warm-start with similar users/posts

### 4. Position Bias

**Problem**: Top positions get more clicks regardless of relevance
```python
# Position 1: 10% CTR
# Position 10: 1% CTR (even if equally relevant)
```

**How to avoid**:
- Add position feature during training
- Debias labels using inverse propensity weighting
- Randomize positions during data collection

### 5. Feedback Loops

**Problem**: Model affects data distribution
```python
# Model ranks sports posts high → users click sports →
# model learns to rank sports even higher → repeat
```

**How to avoid**:
- Exploration (show random posts)
- Regular retraining with unbiased data
- Monitor distribution drift
- A/B test new models carefully

---

## Expected Results

After implementing and training your Wide & Deep model:

**Offline Metrics** (on test set):
- **AUC-ROC**: 0.75-0.80 (very good for this task)
- **NDCG@10**: 0.70-0.75
- **MAP**: 0.65-0.70
- **Calibration**: Within 5% of diagonal
- **Dwell Time MAE**: <20 seconds

**Inference Performance**:
- **Latency**: <10ms per example (CPU), <2ms (GPU)
- **Throughput**: >1000 QPS per CPU core
- **Model size**: 50-100MB (depends on embedding tables)

**Comparison with baselines**:
| Model | AUC | NDCG@10 | Latency |
|-------|-----|---------|---------|
| Random | 0.50 | 0.30 | 0.1ms |
| Logistic Regression | 0.68 | 0.55 | 1ms |
| Deep Only | 0.74 | 0.68 | 5ms |
| **Wide & Deep** | **0.78** | **0.73** | **8ms** |

---

## Success Criteria

You've mastered feed ranking when you can:

**Theory**:
- [ ] Explain Wide & Deep architecture and why both components matter
- [ ] Describe pointwise, pairwise, and listwise ranking approaches
- [ ] Explain multi-task learning benefits and challenges
- [ ] Discuss negative sampling strategies and trade-offs
- [ ] Calculate and interpret NDCG, MAP, and calibration

**Implementation**:
- [ ] Build complete data pipeline with proper feature engineering
- [ ] Implement Wide & Deep model from scratch
- [ ] Train with multi-task learning (CTR + dwell time)
- [ ] Implement hard negative mining
- [ ] Achieve target metrics (AUC >0.75, NDCG@10 >0.70)

**Production**:
- [ ] Optimize model for <10ms inference
- [ ] Explain Meta-scale challenges (billions of users, millisecond latency)
- [ ] Design A/B test for ranking model
- [ ] Identify and mitigate common pitfalls (data leakage, position bias, etc.)

**Advanced**:
- [ ] Implement two-stage retrieval + ranking pipeline
- [ ] Add feature interaction mechanisms (DCN, FM)
- [ ] Analyze feature importance
- [ ] Export model to ONNX for production

---

## Extensions & Next Steps

### 1. Two-Stage System (Retrieval + Ranking)

Implement **Stage 1: Candidate Generation**:
```python
# Fast retrieval model (two-tower)
user_tower = UserEncoder(user_features)
post_tower = PostEncoder(post_features)

# Similarity search
user_emb = user_tower(user)
post_embs = post_tower(all_posts)
similarities = user_emb @ post_embs.T
top_1000 = similarities.topk(1000)

# Then use Wide & Deep to rank top_1000
```

### 2. Advanced Feature Interactions

**Deep & Cross Network (DCN)**:
```python
# Explicit feature crossing
x0 = concat([user_emb, post_emb, numerical_features])
x1 = x0 * (W1 @ x0) + b1 + x0  # Cross layer
x2 = x0 * (W2 @ x1) + b2 + x1  # Cross layer
# Captures high-order feature interactions
```

**Factorization Machines (FM)**:
```python
# Learn pairwise feature interactions
interaction = sum(vi @ vj * xi * xj for i, j in all_pairs)
```

### 3. Online Learning

```python
# Update model with fresh data continuously
for batch in streaming_data:
    loss = model(batch)
    optimizer.step()

    # Periodic evaluation
    if step % 1000 == 0:
        evaluate_online_metrics()
```

### 4. Diversity & Exploration

```python
# Ensure diverse content in feed
ranked_posts = model.rank(candidates)

# Re-rank for diversity
final_ranking = diversify(
    ranked_posts,
    diversity_penalty=0.5,  # Penalize similar content
    exploration_rate=0.1     # 10% random posts
)
```

---

## Resources

### Foundational Papers
- **[Wide & Deep Learning (Google, 2016)](https://arxiv.org/abs/1606.07792)**: Original paper introducing the architecture
- **[Deep & Cross Network (Google, 2017)](https://arxiv.org/abs/1708.05123)**: Improved feature interactions
- **[DLRM: Deep Learning Recommendation Model (Meta, 2019)](https://arxiv.org/abs/1906.00091)**: Meta's approach to recommendations
- **[Learning to Rank for IR (2009)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-for-information-retrieval/)**: Comprehensive LTR overview

### Industry Blog Posts
- [Meta: Powered by AI: Instagram's Explore recommender system](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/)
- [Meta: How Machine Learning Powers Facebook's News Feed Ranking](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
- [LinkedIn: The AI Behind LinkedIn Recruiter search and recommendation systems](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems)
- [Twitter: The Recommendation Algorithm](https://blog.twitter.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm)

### Books
- **"Deep Learning for Recommender Systems" by Charu Aggarwal**: Comprehensive coverage of DL for recommendations
- **"Practical Recommender Systems" by Kim Falk**: Hands-on implementation guide

### Code Repositories
- [DLRM (Meta)](https://github.com/facebookresearch/dlrm): Official implementation
- [DeepCTR](https://github.com/shenweichen/DeepCTR): Collection of CTR models
- [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec): Transformer-based ranking

---

## Getting Started

**Week 1: Foundation**
- Day 1-2: Read theory, explore dataset, build data pipeline
- Day 3-5: Implement Wide & Deep model, unit test components

**Week 2: Training & Evaluation**
- Day 1-3: Train model, implement multi-task learning, tune hyperparameters
- Day 4-5: Implement ranking metrics, evaluate model, analyze results

**Week 3: Production & Advanced**
- Day 1-3: Optimize for production (quantization, ONNX export)
- Day 4-5: Implement hard negative mining, feature importance, A/B test design

**Ready to build a production-scale feed ranking system? Let's go! 🎯**
