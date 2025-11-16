# Lab 5: Knowledge Distillation - Compressing Models 10x 📚

> **Time:** 2-3 hours
> **Difficulty:** Expert
> **Goal:** Master knowledge distillation to compress large models into 10x smaller student models with 95%+ accuracy retention

---

## 📖 Why This Lab Matters

You've trained a massive transformer model. It achieves state-of-the-art accuracy. **But it's too large and slow for production.**

**The deployment reality:**
```python
# Your SOTA model
Teacher: BERT-large (340M parameters, 1.3 GB, 200ms latency)
Accuracy: 95% on benchmark

# Production constraints:
- Mobile deployment: <50 MB model size
- Real-time inference: <20ms latency
- IoT devices: <100 MB RAM available

# Traditional approach: Use smaller model
Student: BERT-tiny (14M parameters, 56 MB, 10ms latency)
Accuracy: 82% (13 point drop!) ← Unacceptable!
```

**The knowledge distillation solution:**
```python
# Train small model to mimic large model
Student (distilled): BERT-tiny (14M parameters, 56 MB, 10ms latency)
Accuracy: 92% (only 3 point drop!) ← Much better!

Compression: 24x smaller
Speedup: 20x faster
Accuracy retention: 97% of teacher performance
```

**Real-world impact:**
- **DistilBERT (Hugging Face):** 40% smaller, 60% faster, 97% accuracy of BERT
- **MobileBERT (Google):** 4x smaller than BERT, runs on phones
- **TinyBERT (Huawei):** 7.5x smaller, 9.4x faster, 96.8% accuracy retention
- **ALBERT (Google):** 18x fewer parameters, same accuracy as BERT-large

**This lab teaches you knowledge distillation:**
- Teacher-student framework
- Soft targets and temperature scaling
- Feature-based distillation
- Achieving 10x compression with <5% accuracy loss
- Production deployment of distilled models

**Master distillation, and you can deploy SOTA models anywhere—mobile, edge, browser.**

---

## 🧠 The Big Picture: Why Big Models Are Better Teachers

### The Knowledge Transfer Problem

**Observation:** Large models learn richer representations than their final predictions show.

**Example: Image classification**
```python
# Teacher model (ResNet-152) predicts "Golden Retriever"
Logits before softmax:
  Golden Retriever: 8.5
  Labrador:         7.8  ← Close second! Shares features
  German Shepherd:  6.2
  Cat:              0.1  ← Very different
  Car:             -2.3  ← Completely different

# After softmax (hard labels):
  Golden Retriever: 0.95  ← Winner takes all!
  Labrador:         0.04  ← Information lost
  German Shepherd:  0.01
  ...

# Small model trained on hard labels:
  Learns: "Golden Retriever = 1, everything else = 0"
  Missing: Similarity structure (Labrador is similar, Cat is not)
```

**Key insight:** Hard labels (0/1) discard rich similarity information. Soft labels (probability distributions) preserve it.

### Why Distillation Works

**Hypothesis (Hinton et al., 2015):**
> "The relative probabilities of incorrect classes tell us a lot about how the teacher model generalizes. A small model can learn this 'dark knowledge' more efficiently than learning from scratch."

**Intuition:**
```
Training from scratch:
  Student sees: [Image, Label=Dog]
  Learns: "This is a dog" (simple classification)

Training with distillation:
  Student sees: [Image, Teacher says: 95% dog, 4% wolf, 1% cat]
  Learns: "This is mostly dog, somewhat wolf-like, nothing like cat"
  Richer understanding of visual similarities!
```

**Mathematical view:**
- Hard labels: One-hot encoding, high entropy minimization
- Soft labels: Full probability distribution, preserves class relationships
- Temperature: Controls softness (higher T = softer distribution)

---

## 🔬 Deep Dive: Distillation Techniques

### 1. Response-Based Distillation (Classic)

**Method:** Train student to match teacher's output probabilities.

**Loss function:**
$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{soft}} + (1-\alpha) \cdot \mathcal{L}_{\text{hard}}$$

Where:
- $\mathcal{L}_{\text{soft}}$: KL divergence between teacher and student soft predictions
- $\mathcal{L}_{\text{hard}}$: Cross-entropy with ground truth labels
- $\alpha$: Balance between distillation and task loss (typically 0.5-0.9)

**Temperature scaling:**
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where:
- $z_i$: Logit for class $i$
- $T$: Temperature (higher = softer distribution)
- $T=1$: Standard softmax
- $T>1$: Softer probabilities (more information)

**Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """
    Compute distillation loss combining soft and hard targets.

    Args:
        student_logits: Raw outputs from student model
        teacher_logits: Raw outputs from teacher model
        labels: Ground truth labels
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss (vs task loss)
    """
    # Soft targets with temperature scaling
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)

    # Distillation loss (KL divergence)
    distill_loss = F.kl_div(
        soft_predictions,
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)  # Scale by T^2 (Hinton et al.)

    # Task loss (standard cross-entropy)
    task_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * distill_loss + (1 - alpha) * task_loss

    return total_loss

# Training loop
teacher.eval()  # Teacher in eval mode (frozen)
student.train()

for images, labels in dataloader:
    # Get teacher predictions (no gradients)
    with torch.no_grad():
        teacher_logits = teacher(images)

    # Get student predictions
    student_logits = student(images)

    # Compute distillation loss
    loss = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=3.0,
        alpha=0.7
    )

    # Optimize student
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Why temperature scaling works:**
```python
# Example: Without temperature (T=1)
Logits:  [8.5, 7.8, 6.2, 0.1]
Softmax: [0.73, 0.24, 0.03, 0.00]  ← Very peaked, little info in small values

# With temperature (T=4)
Logits/4:  [2.12, 1.95, 1.55, 0.02]
Softmax:   [0.41, 0.34, 0.22, 0.03]  ← More uniform, more information!

Student learns: "Class 0 and 1 are similar, class 2 somewhat related"
```

### 2. Feature-Based Distillation

**Method:** Match intermediate layer representations, not just final outputs.

**Motivation:** Early layers learn general features, late layers learn task-specific features. Transferring intermediate representations accelerates learning.

**Loss function:**
$$\mathcal{L}_{\text{feature}} = \sum_{l} \lambda_l \cdot \mathcal{L}_{\text{layer}}(F^S_l, F^T_l)$$

Where:
- $F^S_l$, $F^T_l$: Student and teacher features at layer $l$
- $\lambda_l$: Weight for layer $l$ (typically higher for middle layers)
- $\mathcal{L}_{\text{layer}}$: Distance metric (MSE, cosine distance)

**Implementation:**
```python
class FeatureDistillationLoss(nn.Module):
    def __init__(self, layer_indices=[3, 6, 9]):
        super().__init__()
        self.layer_indices = layer_indices

    def forward(self, student_features, teacher_features):
        """
        Match intermediate layer features.

        Args:
            student_features: List of feature maps from student
            teacher_features: List of feature maps from teacher
        """
        loss = 0

        for idx in self.layer_indices:
            s_feat = student_features[idx]
            t_feat = teacher_features[idx]

            # Align dimensions if different (student often smaller)
            if s_feat.shape != t_feat.shape:
                # Project student features to teacher dimension
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])

            # MSE loss on features
            loss += F.mse_loss(s_feat, t_feat)

        return loss / len(self.layer_indices)

# Modified training with feature matching
def train_with_feature_distillation(student, teacher, dataloader):
    feature_loss_fn = FeatureDistillationLoss(layer_indices=[3, 6, 9])

    for images, labels in dataloader:
        # Forward pass with feature extraction
        student_features, student_logits = student(images, return_features=True)

        with torch.no_grad():
            teacher_features, teacher_logits = teacher(images, return_features=True)

        # Response distillation
        response_loss = distillation_loss(student_logits, teacher_logits, labels)

        # Feature distillation
        feature_loss = feature_loss_fn(student_features, teacher_features)

        # Combined
        total_loss = response_loss + 0.5 * feature_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

**Attention transfer (Zagoruyko & Komodakis, 2017):**
```python
def attention_transfer_loss(student_features, teacher_features):
    """
    Match attention maps (spatial importance).
    """
    def compute_attention(features):
        # Attention = spatial L2 norm
        return torch.norm(features, p=2, dim=1, keepdim=True)

    s_attention = compute_attention(student_features)
    t_attention = compute_attention(teacher_features)

    # Normalize
    s_attention = F.normalize(s_attention.view(s_attention.size(0), -1))
    t_attention = F.normalize(t_attention.view(t_attention.size(0), -1))

    return F.mse_loss(s_attention, t_attention)
```

### 3. Relation-Based Distillation

**Method:** Transfer pairwise relationships between samples.

**Idea:** Instead of individual predictions, transfer how the teacher relates different inputs.

**Implementation:**
```python
def relation_distillation_loss(student_logits, teacher_logits):
    """
    Transfer similarity structure between samples in a batch.
    """
    # Compute pairwise similarities (teacher)
    teacher_sim = torch.mm(teacher_logits, teacher_logits.t())
    teacher_sim = F.softmax(teacher_sim / 0.5, dim=1)

    # Compute pairwise similarities (student)
    student_sim = torch.mm(student_logits, student_logits.t())
    student_sim = F.log_softmax(student_sim / 0.5, dim=1)

    # KL divergence between similarity distributions
    return F.kl_div(student_sim, teacher_sim, reduction='batchmean')
```

### 4. Self-Distillation

**Method:** Use the model as its own teacher (train multiple times, improving each iteration).

**Process:**
1. Train model normally → Model v1
2. Use Model v1 as teacher to train Model v2 (same architecture)
3. Use Model v2 as teacher to train Model v3
4. Repeat...

**Surprising result:** Model v2 often outperforms v1, even with identical architecture!

**Implementation:**
```python
# Round 1: Train from scratch
model_v1 = ResNet50()
train_standard(model_v1, dataloader, epochs=100)
# Accuracy: 92%

# Round 2: Self-distillation
model_v2 = ResNet50()  # Same architecture!
train_with_distillation(
    student=model_v2,
    teacher=model_v1,  # Use v1 as teacher
    dataloader=dataloader,
    epochs=100
)
# Accuracy: 93.5% (1.5 point improvement!)

# Why it works: Soft labels provide smoother optimization landscape
```

**Born-Again Networks (Furlanello et al., 2018):**
- Repeat self-distillation multiple times
- Each generation performs slightly better
- Typical improvement: 1-2% per generation (diminishing returns)

---

## 📊 Mathematical Foundations

### Knowledge Distillation Theory

**Objective:** Minimize divergence between teacher and student distributions.

**Hard label training:**
$$\mathcal{L}_{\text{hard}} = -\sum_{i=1}^{C} y_i \log p_i^S$$

Where $y_i$ is one-hot encoded (only one non-zero value).

**Soft label training (distillation):**
$$\mathcal{L}_{\text{soft}} = D_{KL}(P^T || P^S) = \sum_{i=1}^{C} p_i^T \log \frac{p_i^T}{p_i^S}$$

Where:
- $P^T = \text{softmax}(z^T / T)$: Teacher probabilities with temperature
- $P^S = \text{softmax}(z^S / T)$: Student probabilities with temperature

**Temperature scaling effect:**
$$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

As $T \to \infty$: Distribution becomes uniform (maximum entropy)
As $T \to 0$: Distribution becomes one-hot (minimum entropy)

**Information-theoretic view:**

Entropy of predictions:
$$H(P) = -\sum_i p_i \log p_i$$

Higher temperature → Higher entropy → More information per sample

**Example:**
```
T=1:  [0.95, 0.04, 0.01] → Entropy = 0.24 bits
T=3:  [0.65, 0.25, 0.10] → Entropy = 1.10 bits (4.6x more info!)
T=10: [0.45, 0.35, 0.20] → Entropy = 1.51 bits (6.3x more info!)
```

### Compression-Performance Tradeoff

**Model complexity vs accuracy:**

$$\text{Accuracy} \approx \alpha \log(\text{Parameters}) + \beta$$

**Distillation shifts this curve upward:**
```
Without distillation:
  10M params: 85% accuracy
  100M params: 90% accuracy
  1000M params: 93% accuracy

With distillation (teacher = 1000M):
  10M params: 89% accuracy (+4 points!)
  100M params: 92% accuracy (+2 points!)
  1000M params: 93% accuracy (same as teacher)
```

**Optimal compression ratio:**

Empirical observation (DistilBERT, TinyBERT):
$$r^* = \frac{\text{Teacher params}}{\text{Student params}} \approx 6\text{-}10$$

Beyond 10x compression, accuracy drops sharply (>10% loss).

### Gradient Flow Analysis

**Why distillation helps optimization:**

Standard training:
$$\nabla_\theta \mathcal{L}_{\text{hard}} = -\frac{\partial}{\partial \theta} \sum_i y_i \log p_i$$

For correct class: large gradient
For incorrect classes: zero gradient (no learning signal!)

**Distillation training:**
$$\nabla_\theta \mathcal{L}_{\text{soft}} = -\frac{\partial}{\partial \theta} \sum_i p_i^T \log p_i^S$$

All classes contribute gradients proportional to $p_i^T$!
More gradient signal → Faster, more stable learning

---

## 🏭 Production: Distillation at Scale

### DistilBERT (Hugging Face)

**Problem:** BERT-base (110M params) too large for production.

**Solution:** Distill to DistilBERT (66M params, 40% smaller).

**Architecture:**
- BERT-base: 12 layers, 768 hidden, 12 heads
- DistilBERT: 6 layers, 768 hidden, 12 heads (half depth)

**Training:**
```python
# Teacher: BERT-base fine-tuned on task
teacher = BertForSequenceClassification.from_pretrained('bert-base')

# Student: 6-layer distilled version
student = DistilBertForSequenceClassification(config)

# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temp=2.0):
    # Soft loss (KL divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temp, dim=-1),
        F.softmax(teacher_logits / temp, dim=-1),
        reduction='batchmean'
    ) * (temp ** 2)

    # Hard loss (task objective)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss

# Train on 1M examples
for batch in dataloader:
    teacher_logits = teacher(batch.input_ids).logits
    student_logits = student(batch.input_ids).logits

    loss = distillation_loss(student_logits, teacher_logits, batch.labels)
    loss.backward()
    optimizer.step()
```

**Results:**
- Size: 66M params (40% reduction)
- Speed: 60% faster inference
- Accuracy: 97% of BERT-base performance
- Use cases: Mobile apps, real-time search, chatbots

### MobileBERT (Google)

**Goal:** Run BERT on mobile phones.

**Constraints:**
- Model size: <100 MB
- Latency: <100ms per query (on Pixel 4)
- Accuracy: >95% of BERT-base

**Approach:**
1. **Bottleneck architecture:** Narrow hidden layers (512 instead of 768)
2. **Layer normalization:** Reduce computation
3. **Progressive knowledge transfer:** Layer-by-layer distillation

**Training:**
```python
# Train layer by layer
for layer_idx in range(24):
    # Freeze previous layers
    for param in student.layers[:layer_idx].parameters():
        param.requires_grad = False

    # Train current layer to match teacher
    for batch in dataloader:
        student_hidden = student.get_layer_output(batch, layer_idx)
        teacher_hidden = teacher.get_layer_output(batch, layer_idx)

        loss = F.mse_loss(student_hidden, teacher_hidden)
        loss.backward()
        optimizer.step()
```

**Results:**
- Size: 25M params (4.3x smaller than BERT-base)
- Speed: 75ms latency on Pixel 4
- Accuracy: 99.2% of BERT-base on GLUE benchmark

### TinyBERT (Huawei)

**Innovation:** Two-stage distillation (pre-training + fine-tuning).

**Stage 1: General distillation (pre-training)**
```python
# Distill on massive unlabeled data (Wikipedia, BookCorpus)
teacher = BertModel.from_pretrained('bert-base')
student = TinyBertModel(num_layers=4, hidden_size=312)

for batch in unlabeled_dataloader:
    # Attention distillation
    teacher_attentions = teacher(batch).attentions
    student_attentions = student(batch).attentions

    attention_loss = sum(
        F.mse_loss(s_attn, t_attn)
        for s_attn, t_attn in zip(student_attentions, teacher_attentions)
    )

    # Hidden state distillation
    teacher_hidden = teacher(batch).hidden_states
    student_hidden = student(batch).hidden_states

    hidden_loss = sum(
        F.mse_loss(s_hid, t_hid)
        for s_hid, t_hid in zip(student_hidden, teacher_hidden)
    )

    loss = attention_loss + hidden_loss
    loss.backward()
```

**Stage 2: Task-specific distillation (fine-tuning)**
```python
# Fine-tune on task with distillation
teacher = BertForSequenceClassification.from_pretrained('bert-base-finetuned')
student = TinyBertForSequenceClassification.from_pretrained('tinybert-general')

for batch in task_dataloader:
    # Standard distillation on task
    loss = distillation_loss(student(batch), teacher(batch), batch.labels)
    loss.backward()
```

**Results:**
- Size: 14.5M params (7.5x smaller)
- Speed: 9.4x faster inference
- Accuracy: 96.8% of BERT-base on GLUE
- Mobile deployment: Runs on phones, achieves 50-100ms latency

---

## 🎯 Learning Objectives

By the end of this lab, you will:

**Theory:**
- [ ] Understand why soft labels are better teachers than hard labels
- [ ] Explain temperature scaling and its effect on distributions
- [ ] Compare response-based vs feature-based distillation
- [ ] Analyze compression-performance tradeoffs
- [ ] Know when distillation works vs when it fails

**Implementation:**
- [ ] Train student models with knowledge distillation
- [ ] Implement temperature-scaled soft targets
- [ ] Apply feature-based distillation (intermediate layers)
- [ ] Use self-distillation to improve same-size models
- [ ] Compress models 10x with <5% accuracy loss

**Production Skills:**
- [ ] Distill large models for mobile deployment
- [ ] Achieve production latency requirements (<50ms)
- [ ] Choose optimal student architecture for compression ratio
- [ ] Debug distillation training (when student underperforms)
- [ ] Deploy distilled models to edge devices

---

## 💻 Exercises

### Exercise 1: Basic Knowledge Distillation (45 mins)

**What You'll Learn:**
- Implementing classic distillation loss
- Temperature scaling effects
- Balancing soft and hard losses
- Measuring accuracy retention

**Why It Matters:**
This is the foundational distillation technique (Hinton et al., 2015) used by DistilBERT, MobileBERT, and countless production systems. Understanding this enables all advanced techniques.

**Task:** Distill ResNet-50 (teacher) → ResNet-18 (student) on CIFAR-10.

**Starter code:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

# Load pre-trained teacher
teacher = models.resnet50(pretrained=True)
teacher.eval()  # Freeze teacher

# Initialize student
student = models.resnet18(pretrained=False)

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss

# Training loop
optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9)

for epoch in range(100):
    for images, labels in train_loader:
        # Teacher predictions (frozen)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # Student predictions
        student_logits = student(images)

        # Compute loss
        loss = distillation_loss(student_logits, teacher_logits, labels)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    student_acc = evaluate(student, test_loader)
    print(f"Epoch {epoch}: Student accuracy = {student_acc:.2%}")

# Compare with baseline (training from scratch)
baseline = models.resnet18(pretrained=False)
train_standard(baseline, train_loader, epochs=100)
baseline_acc = evaluate(baseline, test_loader)

print(f"Baseline (from scratch): {baseline_acc:.2%}")
print(f"Distilled: {student_acc:.2%}")
print(f"Teacher: {evaluate(teacher, test_loader):.2%}")
```

**Expected results:**
- Baseline ResNet-18: ~93% accuracy
- Distilled ResNet-18: ~95% accuracy (+2 points!)
- Teacher ResNet-50: ~96% accuracy

### Exercise 2: Feature Distillation (60 mins)

**What You'll Learn:**
- Extracting intermediate features
- Matching feature maps between layers
- Combining response and feature distillation
- Analyzing which layers matter most

**Why It Matters:**
Feature distillation (FitNets, 2015) often outperforms response-only distillation, especially for very small students. It's used in MobileBERT and TinyBERT for state-of-the-art compression.

**Task:** Add feature matching to improve student performance.

**Implementation:**
```python
class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_indices):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self.features = {}

        # Register hooks to extract features
        for name, module in model.named_modules():
            if name in layer_indices:
                module.register_forward_hook(self.save_feature(name))

    def save_feature(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features = {}
        logits = self.model(x)
        return logits, self.features

# Wrap models
teacher_extractor = FeatureExtractor(teacher, ['layer2', 'layer3', 'layer4'])
student_extractor = FeatureExtractor(student, ['layer2', 'layer3', 'layer4'])

# Feature distillation loss
def feature_loss(student_features, teacher_features):
    loss = 0
    for layer_name in student_features:
        s_feat = student_features[layer_name]
        t_feat = teacher_features[layer_name]

        # Match feature statistics
        s_mean = s_feat.mean(dim=(2, 3))
        t_mean = t_feat.mean(dim=(2, 3))
        loss += F.mse_loss(s_mean, t_mean)

    return loss / len(student_features)

# Training with feature distillation
for images, labels in train_loader:
    student_logits, student_features = student_extractor(images)

    with torch.no_grad():
        teacher_logits, teacher_features = teacher_extractor(images)

    # Response distillation
    response_loss = distillation_loss(student_logits, teacher_logits, labels)

    # Feature distillation
    feat_loss = feature_loss(student_features, teacher_features)

    # Combined
    total_loss = response_loss + 0.5 * feat_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

**Expected improvement:**
- Response-only: 95% accuracy
- Response + features: 95.5% accuracy (+0.5 points)

### Exercise 3: Compress BERT 10x (90 mins)

**What You'll Learn:**
- Compressing transformers (BERT, GPT)
- Layer-wise distillation
- Attention transfer
- Achieving extreme compression ratios

**Why It Matters:**
This simulates real production distillation (DistilBERT, TinyBERT). Transformers are the dominant architecture for NLP, and distilling them is essential for deployment.

**Task:** Compress BERT-base (110M params) to TinyBERT (14M params, ~8x compression).

**Architecture:**
```python
from transformers import BertModel, BertConfig

# Teacher: BERT-base
teacher = BertModel.from_pretrained('bert-base-uncased')
# 12 layers, 768 hidden, 12 heads, 110M params

# Student: TinyBERT
student_config = BertConfig(
    hidden_size=312,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=1200
)
student = BertModel(student_config)
# 4 layers, 312 hidden, 12 heads, 14M params
```

**Distillation:**
```python
def bert_distillation_loss(student_outputs, teacher_outputs, attention_weight=0.1):
    """
    Distill BERT by matching:
    1. Final layer outputs (embeddings)
    2. Intermediate attention maps
    3. Hidden states
    """
    # Embedding loss
    emb_loss = F.mse_loss(
        student_outputs.last_hidden_state,
        teacher_outputs.last_hidden_state
    )

    # Attention loss (match attention patterns)
    attention_loss = 0
    for s_attn, t_attn in zip(student_outputs.attentions, teacher_outputs.attentions):
        attention_loss += F.mse_loss(s_attn, t_attn)
    attention_loss /= len(student_outputs.attentions)

    # Hidden state loss
    hidden_loss = 0
    for s_hidden, t_hidden in zip(student_outputs.hidden_states, teacher_outputs.hidden_states):
        hidden_loss += F.mse_loss(s_hidden, t_hidden)
    hidden_loss /= len(student_outputs.hidden_states)

    return emb_loss + attention_weight * attention_loss + 0.5 * hidden_loss

# Train
for batch in dataloader:
    teacher_outputs = teacher(**batch, output_attentions=True, output_hidden_states=True)
    student_outputs = student(**batch, output_attentions=True, output_hidden_states=True)

    loss = bert_distillation_loss(student_outputs, teacher_outputs)
    loss.backward()
    optimizer.step()
```

**Results:**
- Compression: 110M → 14M params (7.9x)
- Speed: 9x faster inference
- Accuracy on GLUE: 96-98% of BERT-base

### Exercise 4: Self-Distillation (45 mins)

**What You'll Learn:**
- Using a model as its own teacher
- Born-again networks
- Why self-distillation improves performance
- Iterative refinement

**Why It Matters:**
Self-distillation can improve model accuracy without changing architecture or adding parameters. It's a "free lunch" optimization used in production to squeeze extra 1-2% accuracy.

**Task:** Apply self-distillation to improve ResNet-18 accuracy.

**Implementation:**
```python
# Generation 1: Train from scratch
model_gen1 = models.resnet18(pretrained=False)
train_standard(model_gen1, train_loader, epochs=100)
acc_gen1 = evaluate(model_gen1, test_loader)
print(f"Gen 1 (from scratch): {acc_gen1:.2%}")

# Generation 2: Self-distillation
model_gen2 = models.resnet18(pretrained=False)  # Same architecture!
model_gen1.eval()  # Freeze gen1

for epoch in range(100):
    for images, labels in train_loader:
        # Gen1 as teacher
        with torch.no_grad():
            teacher_logits = model_gen1(images)

        # Gen2 as student
        student_logits = model_gen2(images)

        # Distillation loss
        loss = distillation_loss(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

acc_gen2 = evaluate(model_gen2, test_loader)
print(f"Gen 2 (self-distilled): {acc_gen2:.2%}")
print(f"Improvement: +{acc_gen2 - acc_gen1:.2%}")

# Generation 3: Repeat
model_gen3 = models.resnet18(pretrained=False)
train_with_distillation(model_gen3, model_gen2, train_loader, epochs=100)
acc_gen3 = evaluate(model_gen3, test_loader)
print(f"Gen 3: {acc_gen3:.2%}")
```

**Expected results:**
- Gen 1: 93.0%
- Gen 2: 94.2% (+1.2 points)
- Gen 3: 94.8% (+0.6 points)
- Diminishing returns after 2-3 generations

### Exercise 5: Production Deployment (60 mins)

**What You'll Learn:**
- Exporting distilled models
- Quantizing student models (INT8)
- Benchmarking latency and memory
- Achieving <20ms latency on CPU

**Why It Matters:**
Distillation alone isn't enough—you must also optimize deployment. Combining distillation + quantization + TorchScript achieves 50-100x speedup for production.

**Task:** Deploy distilled ResNet-18 with <20ms latency.

**Optimization stack:**
```python
# Step 1: Distillation (already done)
student = models.resnet18()  # 11M params

# Step 2: Quantization (INT8)
student_quantized = torch.quantization.quantize_dynamic(
    student,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
# 11M params → 2.8 MB (4x compression)

# Step 3: TorchScript compilation
student_scripted = torch.jit.trace(student_quantized, torch.randn(1, 3, 224, 224))

# Step 4: Save for deployment
student_scripted.save("resnet18_distilled_quantized.pt")

# Benchmark
import time
latencies = []
for _ in range(1000):
    input_data = torch.randn(1, 3, 224, 224)
    start = time.time()
    output = student_scripted(input_data)
    latencies.append((time.time() - start) * 1000)

print(f"p50 latency: {np.percentile(latencies, 50):.2f}ms")
print(f"p99 latency: {np.percentile(latencies, 99):.2f}ms")
print(f"Model size: {os.path.getsize('resnet18_distilled_quantized.pt') / 1e6:.2f} MB")
```

**Expected results:**
- Baseline ResNet-50: 200ms latency, 100 MB size
- Distilled ResNet-18: 50ms latency, 45 MB size
- Distilled + Quantized: 18ms latency, 12 MB size ← Target achieved!

**Compression summary:**
| Optimization | Latency | Size | Accuracy |
|--------------|---------|------|----------|
| Baseline (ResNet-50) | 200ms | 100 MB | 96% |
| Distillation (ResNet-18) | 50ms | 45 MB | 95% |
| + Quantization (INT8) | 18ms | 12 MB | 94.5% |
| **Total improvement** | **11x faster** | **8x smaller** | **-1.5%** |

---

## ⚠️ Common Pitfalls

### 1. Temperature Too Low/High

**Symptom:** Student performs worse than training from scratch.

**Cause:** Temperature = 1 (hard labels) or Temperature > 10 (too soft).

**Solution:**
```python
# Optimal temperature: 3-5 for most tasks
for T in [1, 2, 3, 4, 5, 10]:
    student = train_with_temperature(T)
    acc = evaluate(student)
    print(f"T={T}: {acc:.2%}")

# Typical results:
# T=1: 93% (same as hard labels)
# T=3: 95% (optimal!)
# T=10: 92% (too soft, no information)
```

### 2. Alpha Too High (Ignoring Hard Labels)

**Symptom:** Student has lower accuracy on test set.

**Cause:** Alpha = 1.0 (only soft labels, no task supervision).

**Solution:**
```python
# Balance soft and hard losses
# Optimal: alpha = 0.5-0.9
loss = alpha * soft_loss + (1 - alpha) * hard_loss

# Recommended: 0.7 (70% distillation, 30% task)
```

### 3. Student Too Small

**Symptom:** 20x compression, but 15% accuracy drop.

**Cause:** Student capacity insufficient to learn teacher knowledge.

**Rule of thumb:**
- 2-5x compression: <2% accuracy loss
- 5-10x compression: 2-5% accuracy loss
- 10-20x compression: 5-10% accuracy loss
- >20x: >10% loss (usually unacceptable)

### 4. Teacher Overfitted

**Symptom:** Student performs worse than expected despite good teacher.

**Cause:** Teacher memorized training set, transfers noise.

**Solution:**
```python
# Use ensemble of teachers (reduces overfitting)
teachers = [model1, model2, model3]

def ensemble_teacher_logits(images):
    logits = []
    for teacher in teachers:
        with torch.no_grad():
            logits.append(teacher(images))
    return torch.stack(logits).mean(dim=0)  # Average predictions
```

### 5. Not Freezing Teacher

**Symptom:** Both teacher and student degrade during training.

**Cause:** Teacher weights updated during distillation.

**Solution:**
```python
# Always freeze teacher
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False
```

---

## 🏆 Expert Checklist for Mastery

**Foundations:**
- [ ] Understand why soft labels preserve more information than hard labels
- [ ] Explain temperature scaling and its effect on entropy
- [ ] Know response-based vs feature-based distillation
- [ ] Analyze compression-accuracy tradeoffs

**Implementation:**
- [ ] Implemented classic distillation (Hinton et al., 2015)
- [ ] Applied feature-based distillation
- [ ] Used self-distillation to improve models
- [ ] Compressed model 10x with <5% accuracy loss

**Production:**
- [ ] Distilled large model for mobile deployment
- [ ] Combined distillation + quantization + TorchScript
- [ ] Achieved <20ms latency on CPU
- [ ] Deployed distilled model to edge device

**Advanced:**
- [ ] Implemented attention transfer
- [ ] Used relation-based distillation
- [ ] Applied data-free distillation (no training data needed)
- [ ] Created ensemble of students (multi-teacher)

---

## 🚀 Next Steps

After mastering knowledge distillation:

1. **Neural Architecture Search (NAS)**
   - Automated student architecture design
   - Find optimal student for target compression ratio
   - Once-for-all networks

2. **Pruning + Distillation**
   - Combine structured pruning with distillation
   - Remove entire layers/channels
   - Achieve 50x compression

3. **Continual Distillation**
   - Update student as teacher improves
   - Lifelong learning
   - Incremental model updates

4. **Cross-Modal Distillation**
   - Image teacher → Text student
   - Multimodal to unimodal
   - Vision-language models

---

## 📚 References

**Papers:**
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [DistilBERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108)
- [TinyBERT (Jiao et al., 2020)](https://arxiv.org/abs/1909.10351)
- [MobileBERT (Sun et al., 2020)](https://arxiv.org/abs/2004.02984)
- [Born-Again Neural Networks (Furlanello et al., 2018)](https://arxiv.org/abs/1805.04770)

**Tools:**
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.DistillationTrainer)
- [PyTorch Model Distillation](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [Neural Network Distiller (Intel)](https://github.com/IntelLabs/distiller)

**Production Examples:**
- DistilBERT: 40% smaller, 60% faster (Hugging Face)
- MobileBERT: Runs on phones, 99% of BERT accuracy
- TinyBERT: 7.5x compression, 9.4x speedup

---

## 🎯 Solution

Complete implementation: `solution/knowledge_distillation.py`

**What you'll build:**
- Classic distillation trainer
- Feature-based distillation
- BERT compression (8x smaller)
- Self-distillation pipeline
- Production deployment (distill + quantize + TorchScript)
- Comprehensive benchmarking

**Next: Lab 6 - Online Evaluation & A/B Testing!**
