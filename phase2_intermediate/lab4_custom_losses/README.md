# Lab 4: Custom Loss Functions - Beyond Cross-Entropy 📊

> **Time:** 3-4 hours
> **Difficulty:** Intermediate
> **Goal:** Master custom loss functions for specialized ML tasks

---

## 📖 Why This Lab Matters

Standard cross-entropy works well for balanced classification, but real-world ML faces complex challenges requiring specialized loss functions:

- **CLIP** (OpenAI) - InfoNCE contrastive loss for vision-language alignment
- **FaceNet** (Google) - Triplet loss for face recognition
- **RetinaNet** (Facebook) - Focal loss for object detection with class imbalance
- **SimCLR** - Contrastive learning for self-supervised representation
- **BERT** - Label smoothing for better generalization
- **Ranking systems** (Meta, Google) - Pairwise/listwise losses

The right loss function can:
- Handle class imbalance (99% negative, 1% positive)
- Learn better representations (embeddings for similarity)
- Improve generalization (avoid overconfidence)
- Optimize for specific metrics (ranking, retrieval)

**This lab teaches you to choose and design loss functions for any task.**

---

## 🧠 The Big Picture: Why One Loss Doesn't Fit All

### The Problem: Different Tasks, Different Objectives

**Cross-entropy assumes:**
- Balanced classes
- Independent samples
- Classification is the goal
- Calibrated probabilities

**Real-world challenges:**

```
Fraud Detection:
- 99.9% legitimate, 0.1% fraud
- Cross-entropy ignores rare class!

Face Recognition:
- Goal: Similar faces close, different faces far
- Cross-entropy optimizes classification, not similarity

Medical Diagnosis:
- False negative (missing cancer) >> false positive
- Need to weight errors differently

Image-Text Matching (CLIP):
- Learn joint embedding space
- Contrastive: positive pairs close, negative pairs far
```

### The Solution: Task-Specific Losses

| Task | Challenge | Loss Function | Why It Works |
|------|-----------|---------------|--------------|
| Imbalanced classification | Rare classes ignored | Focal Loss | Downweights easy examples |
| Metric learning | Need similarity metric | Triplet/Contrastive | Learns embedding space |
| Ranking | Order matters, not labels | Pairwise/Listwise | Optimizes ranking metrics |
| Calibration | Overconfident predictions | Label Smoothing | Prevents extreme probabilities |
| Multi-modal | Align different modalities | InfoNCE | Contrastive matching |

---

## 🔬 Deep Dive: Custom Loss Functions

### 1. Focal Loss: Handling Class Imbalance

**Paper:** "Focal Loss for Dense Object Detection" (Lin et al., 2017)

**The Problem:**

In imbalanced datasets, easy negatives dominate the loss:

```
Object Detection:
- Background: 99%  (easy negatives)
- Objects: 1%      (hard positives)

Cross-Entropy:
- Loss = 0.99 × easy_loss + 0.01 × hard_loss
- Gradients dominated by easy examples!
- Model learns to classify everything as background
```

**The Solution: Focal Loss**

```python
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

Where:
- p_t: predicted probability for true class
- α_t: class weight (handles class imbalance)
- γ: focusing parameter (default 2)
- (1 - p_t)^γ: modulating factor
```

**How it works:**

```
Easy example (p_t = 0.9):
  (1 - 0.9)^2 = 0.01  ← Very small weight!
  FL ≈ 0.01 × log(0.9) ≈ -0.001

Hard example (p_t = 0.2):
  (1 - 0.2)^2 = 0.64  ← Large weight
  FL ≈ 0.64 × log(0.2) ≈ -1.03

Result: Loss focuses on hard examples!
```

**Effect of γ:**
```
γ = 0: Standard cross-entropy (no modulation)
γ = 1: Linear down-weighting
γ = 2: Quadratic down-weighting (recommended)
γ = 5: Very aggressive focusing
```

**Implementation:**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Get probabilities
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal term: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        loss = alpha_weight * focal_weight * ce_loss
        return loss.mean()
```

**When to use:**
- Object detection (RetinaNet, FCOS)
- Imbalanced classification (fraud, rare disease)
- Hard negative mining
- Any task with easy/hard example imbalance

---

### 2. Contrastive Loss: Learning Similarity

**The Problem:**

Classification learns decision boundaries, not similarities:

```
Classification:
[Cat image] → [0.9, 0.05, 0.05] → "Cat"
[Dog image] → [0.05, 0.9, 0.05] → "Dog"

But we can't measure: "How similar are cat and dog images?"
```

**The Solution: Contrastive Loss**

Learn embeddings where similar samples are close, dissimilar are far.

```python
L = (1 - y) × ½ × D² + y × ½ × max(0, m - D)²

Where:
- D = ||f(x₁) - f(x₂)||₂  (Euclidean distance)
- y = 1 if similar, 0 if dissimilar
- m = margin (minimum distance for dissimilar pairs)
```

**Intuition:**

```
Similar pair (y=1):
  Loss = ½ × D²
  → Minimize distance (pull together)

Dissimilar pair (y=0):
  Loss = ½ × max(0, margin - D)²
  → If D < margin: push apart
  → If D ≥ margin: loss = 0 (far enough)
```

**Visualization:**

```
Before training (random embeddings):
Cat₁  Cat₂  Dog₁  Dog₂
  •     •     •     •
      (all over the place)

After training (good embeddings):
Cat₁  Cat₂         Dog₁  Dog₂
  •-----•             •----•
  ← similar →      ← similar →

  ←------- margin -------→
```

**Implementation:**

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
```

**When to use:**
- Siamese networks
- Signature verification
- Face verification (same/different person)
- Image retrieval
- Anomaly detection

---

### 3. Triplet Loss: Relative Similarity

**Paper:** "FaceNet: A Unified Embedding for Face Recognition" (Schroff et al., 2015)

**The Problem:**

Contrastive loss requires labeled pairs. Triplet loss uses relative comparisons:

```
"This person (anchor) is more similar to this (positive) than that (negative)"
```

**The Solution: Triplet Loss**

```python
L = max(0, D(a, p) - D(a, n) + margin)

Where:
- a: anchor sample
- p: positive sample (same class as anchor)
- n: negative sample (different class)
- D(·,·): distance function
- margin: minimum separation
```

**Intuition:**

```
Goal: D(anchor, positive) + margin < D(anchor, negative)

Triplet: (anchor, positive, negative)
         Cat₁    Cat₂      Dog₁

Distance:
  D(Cat₁, Cat₂) = 0.2  ← small (similar)
  D(Cat₁, Dog₁) = 1.5  ← large (dissimilar)

Loss = max(0, 0.2 - 1.5 + 0.3) = max(0, -1.0) = 0 ✓
       (no loss, triplet already satisfied)

Bad triplet:
  D(Cat₁, Cat₂) = 0.8
  D(Cat₁, Dog₁) = 0.9

Loss = max(0, 0.8 - 0.9 + 0.3) = max(0, 0.2) = 0.2
       (violation! positive too far or negative too close)
```

**Triplet Mining Strategies:**

**Easy triplets:**
```
D(a,p) + margin << D(a,n)
→ Loss = 0 (no learning)
→ Waste of computation
```

**Hard triplets:**
```
D(a,n) < D(a,p)  ← negative closer than positive!
→ High loss, strong gradient
→ Can destabilize training
```

**Semi-hard triplets (best):**
```
D(a,p) < D(a,n) < D(a,p) + margin
→ Positive closer, but negative within margin
→ Moderate loss, stable training
```

**Implementation:**

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute distances
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        # Triplet loss
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
```

**Online triplet mining:**

```python
def batch_all_triplet_loss(embeddings, labels, margin):
    """Compute triplet loss on all valid triplets in batch."""
    # Compute pairwise distances: (batch, batch)
    pairwise_dist = torch.cdist(embeddings, embeddings)

    # Create masks for valid triplets
    anchor_positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # Same class
    anchor_negative_mask = labels.unsqueeze(1) != labels.unsqueeze(0)  # Different class

    # Compute triplet loss for all valid combinations
    triplet_loss = []
    for i in range(len(labels)):
        # Positive distances (same class)
        positive_dists = pairwise_dist[i][anchor_positive_mask[i]]

        # Negative distances (different class)
        negative_dists = pairwise_dist[i][anchor_negative_mask[i]]

        # All triplets
        loss = positive_dists.unsqueeze(1) - negative_dists.unsqueeze(0) + margin
        loss = torch.relu(loss)
        triplet_loss.append(loss.mean())

    return torch.stack(triplet_loss).mean()
```

**When to use:**
- Face recognition (FaceNet)
- Person re-identification
- Image retrieval
- Metric learning
- Few-shot learning

---

### 4. Label Smoothing: Preventing Overconfidence

**Paper:** "Rethinking the Inception Architecture" (Szegedy et al., 2016)

**The Problem:**

Hard targets encourage overconfident predictions:

```
True label:  [0, 0, 1, 0]  ← one-hot
Predictions: [0.001, 0.001, 0.998, 0.000]  ← overconfident!

Issues:
- Poor calibration (probabilities not meaningful)
- Overfitting
- Inability to estimate uncertainty
```

**The Solution: Label Smoothing**

```python
Soft labels = (1 - ε) × hard_labels + ε / K

Where:
- ε: smoothing factor (typically 0.1)
- K: number of classes
```

**Example:**

```
Hard label (K=4):     [0, 0, 1, 0]
Smoothed (ε=0.1):     [0.025, 0.025, 0.925, 0.025]
                       ↑ tiny prob for wrong classes!

Effect:
- Target for correct class: 1.0 → 0.925
- Target for wrong classes: 0.0 → 0.025
```

**Why it helps:**

1. **Prevents overconfidence:** Model can't drive probabilities to exactly 0 or 1
2. **Regularization:** Penalizes extreme predictions
3. **Better calibration:** Probabilities match true frequencies
4. **Improved generalization:** Less overfitting

**Mathematical insight:**

```
Cross-entropy with label smoothing is equivalent to:
  H(q, p) + D_KL(u || p)

Where:
- H(q, p): cross-entropy with true distribution
- D_KL(u || p): KL divergence from uniform distribution
- Acts as entropy regularization!
```

**Implementation:**

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        """
        pred: (batch, classes) - logits
        target: (batch,) - class indices
        """
        pred = pred.log_softmax(dim=-1)

        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

**When to use:**
- Image classification (ResNet, EfficientNet)
- BERT pre-training
- Any classification where calibration matters
- Preventing overconfidence
- Improving generalization

---

### 5. InfoNCE: Contrastive Learning at Scale

**Paper:** "Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)

**Used in:** CLIP, SimCLR, MoCo

**The Problem:**

Standard contrastive loss compares pairs. For large-scale learning, we need to compare one positive against many negatives:

```
Image-Text Matching (CLIP):
- Image: "A dog playing fetch"
- Positive text: "A dog playing fetch"
- Negatives: All other texts in batch (thousands!)
```

**The Solution: InfoNCE**

```python
L = -log(exp(sim(a, p) / τ) / Σⱼ exp(sim(a, nⱼ) / τ))
  = -log(exp(s_pos / τ) / (exp(s_pos / τ) + Σⱼ exp(s_neg,j / τ)))

Where:
- sim(·,·): similarity function (cosine similarity)
- τ: temperature parameter
- p: positive sample
- {nⱼ}: negative samples
```

**Intuition:**

It's a classification problem where:
- Goal: Classify which of N+1 samples is the positive
- N negatives + 1 positive
- Larger batch = more negatives = better representations

**Temperature τ:**

```
τ = 0.1: Sharp distribution (confident)
  sim=0.9 → exp(0.9/0.1) = exp(9) = huge!
  sim=0.5 → exp(0.5/0.1) = exp(5) = large

τ = 1.0: Moderate distribution
  sim=0.9 → exp(0.9/1.0) = exp(0.9) = moderate

Effect: Lower τ → focus on hard negatives
```

**CLIP-style Implementation:**

```python
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        """
        image_features: (batch, dim)
        text_features: (batch, dim)
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(len(logits)).to(logits.device)

        # InfoNCE loss (symmetric)
        loss_i2t = self.criterion(logits, labels)  # Image → Text
        loss_t2i = self.criterion(logits.T, labels)  # Text → Image

        return (loss_i2t + loss_t2i) / 2
```

**Why it works:**

1. **Scalability:** Batch size = number of negatives (thousands!)
2. **End-to-end:** Jointly train both encoders
3. **Self-supervised:** No manual labels needed
4. **Strong representations:** Learned from rich comparisons

**When to use:**
- Contrastive pre-training (SimCLR, MoCo)
- Multi-modal learning (CLIP, ALIGN)
- Self-supervised learning
- Large-scale metric learning

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Why different tasks need different loss functions
- Focal loss mathematics and class imbalance handling
- Contrastive vs triplet loss for metric learning
- Label smoothing as regularization
- InfoNCE for large-scale contrastive learning
- When to use each loss function
- Multi-task loss combination strategies

**Practical Skills:**
- Implement focal loss from scratch
- Build contrastive and triplet loss functions
- Apply label smoothing to classification
- Create InfoNCE for multi-modal learning
- Combine multiple losses for multi-task learning
- Tune loss hyperparameters (margin, temperature, smoothing)
- Evaluate with task-specific metrics

---

## 🔑 Key Concepts

### 1. Loss Function Design Principles

**Good loss functions:**
- **Align with evaluation metric:** Optimize what you care about
- **Handle data distribution:** Imbalance, outliers, rare classes
- **Provide useful gradients:** Not too flat, not too steep
- **Computationally efficient:** Scale to large datasets
- **Hyperparameters:** Few and intuitive

### 2. Multi-Task Learning

**Combining losses:**

```python
# Weighted sum
total_loss = α × loss1 + β × loss2 + γ × loss3

# Adaptive weighting (uncertainty weighting)
total_loss = loss1 / (2 × σ1²) + loss2 / (2 × σ2²) + log(σ1) + log(σ2)
```

**Challenges:**
- Different scales (loss1=0.01, loss2=100)
- Different convergence rates
- Gradient imbalance

**Solutions:**
- Normalize losses
- Gradient balancing
- Adaptive weights

### 3. Loss Hyperparameters

| Loss | Key Hyperparameter | Typical Range | Effect |
|------|-------------------|---------------|--------|
| Focal | γ (focus) | 0-5 | Higher = more focus on hard examples |
| Focal | α (weight) | 0.25-0.75 | Class balancing |
| Triplet | margin | 0.1-2.0 | Min separation between classes |
| Contrastive | margin | 0.5-2.0 | Min distance for negative pairs |
| Label Smooth | ε | 0.0-0.2 | Higher = more smoothing |
| InfoNCE | τ (temperature) | 0.01-1.0 | Lower = sharper focus |

---

## 🧪 Exercises

### Exercise 1: Focal Loss for Imbalanced Data (45 mins)

**What You'll Learn:**
- Implementing focal loss from scratch
- Effect of focusing parameter γ
- Comparison with weighted cross-entropy
- Handling extreme class imbalance

**Why It Matters:**
Most real-world datasets are imbalanced (fraud detection: 99.9% vs 0.1%). Focal loss is production-ready solution used in RetinaNet and modern object detectors.

**Tasks:**
1. Implement `FocalLoss` class
2. Create imbalanced dataset (90% class 0, 10% class 1)
3. Train model with cross-entropy vs focal loss
4. Compare performance (accuracy, recall on rare class)
5. Visualize loss curves for easy vs hard examples
6. Tune γ parameter

**Expected results:**
```
Cross-Entropy:
  Class 0 (common): 99% accuracy
  Class 1 (rare):   20% accuracy  ← Fails on rare class!

Focal Loss (γ=2):
  Class 0 (common): 95% accuracy
  Class 1 (rare):   85% accuracy  ← Much better!
```

---

### Exercise 2: Contrastive Loss for Siamese Networks (60 mins)

**What You'll Learn:**
- Building Siamese network architecture
- Creating positive and negative pairs
- Contrastive loss implementation
- Embedding visualization

**Why It Matters:**
Siamese networks with contrastive loss power face verification, signature verification, and one-shot learning applications.

**Tasks:**
1. Create Siamese network (shared encoder)
2. Generate pairs from MNIST (same/different digit)
3. Implement contrastive loss
4. Train embedding network
5. Visualize embeddings (t-SNE)
6. Test on verification task

**Architecture:**
```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 128)  # Embedding dimension
        )

    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2
```

---

### Exercise 3: Triplet Loss for Metric Learning (90 mins)

**What You'll Learn:**
- Triplet mining strategies
- Hard negative mining
- Online vs offline triplet selection
- Evaluation with retrieval metrics

**Why It Matters:**
Triplet loss (FaceNet) revolutionized face recognition. Understanding triplet mining is crucial for modern metric learning.

**Tasks:**
1. Implement triplet loss
2. Create triplet sampling (online mining)
3. Implement semi-hard negative mining
4. Train on MNIST/Fashion-MNIST
5. Evaluate with retrieval metrics (Recall@K)
6. Visualize embedding space

**Triplet mining:**
```python
def get_triplets(embeddings, labels):
    """Mine semi-hard triplets from batch."""
    triplets = []

    for i in range(len(labels)):
        anchor_label = labels[i]
        anchor_emb = embeddings[i]

        # Positive: same class, different sample
        positive_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
        positives = embeddings[positive_mask]

        # Negative: different class
        negative_mask = labels != anchor_label
        negatives = embeddings[negative_mask]

        if len(positives) > 0 and len(negatives) > 0:
            # Sample one positive and one (semi-hard) negative
            pos_idx = random.randint(0, len(positives) - 1)
            # ... implement semi-hard negative selection ...
            triplets.append((anchor_emb, positives[pos_idx], negative))

    return triplets
```

---

### Exercise 4: Label Smoothing Regularization (30 mins)

**What You'll Learn:**
- Label smoothing implementation
- Effect on model calibration
- Comparison with standard cross-entropy
- Calibration metrics (ECE)

**Why It Matters:**
Label smoothing is standard in modern image classification (ResNet, EfficientNet) and NLP (BERT, GPT). Simple but effective regularization.

**Tasks:**
1. Implement label smoothing loss
2. Train CIFAR-10 classifier with/without smoothing
3. Compare test accuracy
4. Measure calibration (reliability diagram)
5. Analyze confidence distributions
6. Test different smoothing values (0.0, 0.1, 0.2)

---

### Exercise 5: InfoNCE for Multi-Modal Learning (90 mins)

**What You'll Learn:**
- CLIP-style architecture
- Batch contrastive learning
- Temperature scaling
- Large-scale metric learning

**Why It Matters:**
InfoNCE powers CLIP, SimCLR, and modern self-supervised learning. Understanding it opens door to foundation models.

**Tasks:**
1. Create simple image-text dataset
2. Build dual encoders (image + text)
3. Implement InfoNCE loss
4. Train with contrastive objective
5. Test zero-shot classification
6. Analyze effect of temperature and batch size

**Mini-CLIP implementation:**
```python
class MiniCLIP(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, embed_dim=256):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, images, texts):
        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(texts)
        return image_emb, text_emb
```

---

### Exercise 6: Multi-Task Loss Combination (60 mins)

**What You'll Learn:**
- Combining multiple losses
- Loss weighting strategies
- Gradient balancing
- Multi-task optimization

**Why It Matters:**
Production ML often optimizes multiple objectives simultaneously. Understanding multi-task losses is essential for complex systems.

**Tasks:**
1. Create multi-task problem (classify + reconstruct)
2. Implement multiple losses
3. Test different weighting strategies
4. Implement adaptive weighting
5. Analyze task trade-offs
6. Visualize gradient magnitudes

**Adaptive weighting:**
```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # Learnable uncertainty parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: list of task losses
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

        return sum(weighted_losses)
```

---

## 📝 Design Patterns

### Pattern 1: Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-BCE_loss)  # Probability of correct class
        F_loss = self.alpha * ((1 - p_t) ** self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss
```

### Pattern 2: Online Triplet Mining

```python
class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings)

        # Mine triplets
        triplet_loss = []
        for i in range(len(labels)):
            # Hardest positive (same class, max distance)
            pos_mask = labels == labels[i]
            pos_mask[i] = False  # Exclude anchor
            if pos_mask.sum() == 0:
                continue
            hardest_positive_dist = pairwise_dist[i][pos_mask].max()

            # Hardest negative (different class, min distance)
            neg_mask = labels != labels[i]
            if neg_mask.sum() == 0:
                continue
            hardest_negative_dist = pairwise_dist[i][neg_mask].min()

            # Triplet loss
            loss = torch.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss.append(loss)

        return torch.stack(triplet_loss).mean() if triplet_loss else torch.tensor(0.0)
```

### Pattern 3: Temperature-Scaled InfoNCE

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        # Normalize
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)

        # Similarity matrix
        logits = torch.matmul(features1, features2.T) / self.temperature

        # Contrastive loss (symmetric)
        labels = torch.arange(len(features1)).to(logits.device)
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)

        return (loss1 + loss2) / 2
```

---

## ✅ Solutions

Complete implementations in `solution/` directory.

**Files:**
- `01_focal_loss.py` - Focal loss for imbalanced data
- `02_contrastive_loss.py` - Siamese network training
- `03_triplet_loss.py` - Triplet mining and metric learning
- `04_label_smoothing.py` - Calibration with smoothing
- `05_infonce.py` - Contrastive multi-modal learning
- `06_multi_task.py` - Multi-task loss combination

Run examples:
```bash
cd solution
python 01_focal_loss.py
python 02_contrastive_loss.py
python 03_triplet_loss.py
python 04_label_smoothing.py
python 05_infonce.py
python 06_multi_task.py
```

---

## 🎓 Key Takeaways

1. **One loss doesn't fit all** - Different tasks need different objectives
2. **Focal loss handles imbalance** - Down-weights easy examples
3. **Contrastive/triplet for similarity** - Learn embedding spaces
4. **Label smoothing prevents overconfidence** - Better calibration
5. **InfoNCE scales contrastive learning** - Batch as negatives
6. **Temperature controls sharpness** - Lower = focus on hard negatives
7. **Multi-task needs careful weighting** - Balance competing objectives
8. **Match loss to evaluation metric** - Optimize what you care about

**The Decision Tree:**
```
Task?
├─ Imbalanced classification → Focal Loss
├─ Similarity/retrieval → Contrastive/Triplet/InfoNCE
├─ Better calibration → Label Smoothing
├─ Multi-modal → InfoNCE
└─ Multiple objectives → Multi-task combination
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**At Facebook/Meta (RetinaNet):**
- Object detection with focal loss
- Handles extreme class imbalance
- Real-time detection at scale
- Powers content understanding

**At Google (FaceNet):**
- Triplet loss for face recognition
- 200M face images training
- Powers Google Photos clustering
- Privacy-preserving face matching

**At OpenAI (CLIP):**
- InfoNCE on 400M image-text pairs
- Zero-shot transfer to downstream tasks
- Powers DALL-E, ChatGPT vision
- Foundation for multi-modal AI

**At Google (BERT):**
- Label smoothing for generalization
- Better calibrated predictions
- Standard in modern transformers
- Improved downstream task performance

---

## 🚀 Next Steps

1. **Complete all exercises** - Implement each loss from scratch
2. **Apply to real data** - Test on imbalanced/similarity tasks
3. **Read papers** - Focal Loss, FaceNet, CLIP
4. **Move to Lab 5** - Distributed Training

---

## 💪 Bonus Challenges

1. **Curriculum Focal Loss**
   - Start with γ=0, gradually increase
   - Smooth transition from easy to hard examples
   - Compare convergence speed

2. **Quadruplet Loss**
   - Extend triplet to quadruplets
   - Two positives, two negatives
   - Better embedding structure

3. **Supervised Contrastive Loss**
   - Multiple positives per anchor
   - All samples of same class
   - Stronger than triplet loss

4. **Self-Supervised Learning**
   - SimCLR-style data augmentation
   - Train with InfoNCE
   - Evaluate on downstream tasks

5. **Loss Landscape Visualization**
   - Plot loss surface
   - Compare different losses
   - Analyze optimization difficulty

---

## 📚 Essential Resources

**Papers:**
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) - Lin et al., 2017
- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832) - Schroff et al., 2015
- [Learning a Similarity Metric Discriminatively](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf) - Chopra et al., 2005 (Contrastive)
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) - van den Oord et al., 2018 (InfoNCE)
- [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567) - Szegedy et al., 2016 (Label Smoothing)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - Radford et al., 2021 (CLIP)

**Tutorials:**
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Focal Loss Explained](https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/)

---

## 🤔 Common Pitfalls

### Pitfall 1: Wrong Focal Loss Targets

```python
# ❌ Using class indices with focal loss
loss = focal_loss(logits, class_indices)  # Wrong!

# ✓ Use one-hot or probabilities
targets_onehot = F.one_hot(class_indices, num_classes)
loss = focal_loss(logits, targets_onehot.float())
```

### Pitfall 2: Triplet Batch Too Small

```python
# ❌ Batch size 8 (few triplets)
# Likely no valid triplets per batch!

# ✓ Batch size ≥ 32
# More samples = more diverse triplets
```

### Pitfall 3: Temperature Too High/Low

```python
# ❌ τ = 10.0 (too high, uniform distribution)
# ❌ τ = 0.001 (too low, numerical instability)

# ✓ τ = 0.07 (CLIP default)
# ✓ τ = 0.1 (SimCLR default)
```

### Pitfall 4: Forgetting to Normalize Embeddings

```python
# ❌ Raw embeddings (scale varies)
distance = torch.norm(emb1 - emb2)

# ✓ Normalized embeddings
emb1 = F.normalize(emb1, dim=-1)
emb2 = F.normalize(emb2, dim=-1)
distance = torch.norm(emb1 - emb2)
```

---

## 💡 Pro Tips

1. **Start with standard losses** - Baseline before custom
2. **Focal loss: γ=2, α=0.25** - Good defaults
3. **Triplet: mine semi-hard** - Avoid too easy/hard
4. **InfoNCE: larger batch better** - More negatives
5. **Label smoothing: ε=0.1** - Standard value
6. **Temperature: tune carefully** - Big impact
7. **Multi-task: normalize losses** - Similar scales

---

## ✨ You're Ready When...

- [ ] You understand why different tasks need different losses
- [ ] You can implement focal loss for class imbalance
- [ ] You've built siamese network with contrastive loss
- [ ] You understand triplet mining strategies
- [ ] You can apply label smoothing
- [ ] You've implemented InfoNCE
- [ ] You know when to use each loss function
- [ ] You can combine multiple losses for multi-task learning
- [ ] You understand hyperparameter effects (γ, margin, τ, ε)
- [ ] You've compared losses on real data

**Remember:** The loss function defines what your model optimizes. Choose wisely, and you've solved half the problem!
