# Lab 2: Transformer Encoder - Understanding and Encoding Sequences 🏗️

> **Time:** 4-5 hours
> **Difficulty:** Intermediate
> **Goal:** Master the transformer encoder architecture that powers BERT, ViT, and modern understanding models

---

## 📖 Why This Lab Matters

The Transformer Encoder is the architecture behind some of the most impactful AI systems:

- **BERT** (Google) - Powers Google Search, understanding billions of queries daily
- **Vision Transformer (ViT)** - State-of-the-art image recognition
- **CLIP** (OpenAI) - Connects images and text
- **ESM** (Meta) - Protein structure prediction
- **CodeBERT** - Understanding code
- **RoBERTa** - Improved language understanding

The encoder transforms raw sequences (text, images as patches, proteins) into **rich contextual representations** that capture meaning, relationships, and structure.

**This lab teaches you the architecture that understands the world.**

---

## 🧠 The Big Picture: From Sequences to Understanding

### The Problem: Representing Sequential Data

**Challenge:** Convert variable-length sequences into fixed-size meaningful representations.

```
Input:  "The quick brown fox jumps over the lazy dog"
Output: Rich vector representation capturing:
        - Syntax (grammatical structure)
        - Semantics (meaning)
        - Context (relationships between words)
        - Task-specific features
```

**Why it's hard:**
- Variable length (5 words vs 500 words)
- Long-range dependencies ("The cat that sat on the mat" - "cat" and "sat" are related)
- Bidirectional context (word meaning depends on before AND after)
- Efficiency (process in parallel, not sequentially like RNNs)

### The Solution: Transformer Encoder

**Key innovation:** Stack multiple layers of self-attention + feed-forward networks.

```
Input Sequence
    ↓
Embedding + Positional Encoding
    ↓
[Encoder Layer 1] → Multi-head Self-Attention → Add & Norm → FFN → Add & Norm
    ↓
[Encoder Layer 2] → Multi-head Self-Attention → Add & Norm → FFN → Add & Norm
    ↓
    ...
    ↓
[Encoder Layer N]
    ↓
Contextual Representations
```

**Each layer:**
1. **Self-Attention:** Understand relationships between all tokens
2. **Feed-Forward:** Non-linear transformation per position
3. **Residuals:** Enable deep networks (skip connections)
4. **Layer Norm:** Stabilize training

**Stacking N layers:** Each layer refines understanding progressively.

---

## 🔬 Deep Dive: Encoder Architecture Components

### 1. Positional Encoding: Adding Position Information

**The Problem:**

Attention has no inherent notion of position!

```python
# These two sequences give SAME attention output:
seq1 = ["cat", "sat", "mat"]
seq2 = ["mat", "cat", "sat"]  # Different order!

# But they have different meanings:
"The cat sat on the mat" ≠ "The mat sat on the cat"
```

**The Solution: Positional Encoding**

Add position information to embeddings:

```python
output = word_embedding + positional_encoding
```

**Sinusoidal Positional Encoding (from "Attention is All You Need"):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: position in sequence (0, 1, 2, ...)
- i: dimension index (0 to d_model/2)
- d_model: embedding dimension
```

**Why sinusoidal?**

1. **Deterministic:** Same position always gets same encoding
2. **Bounded:** Values in [-1, 1]
3. **Relative positions:** PE(pos+k) can be expressed as linear function of PE(pos)
4. **Extrapolation:** Can handle sequences longer than training

**Visualization:**

```
Position 0: [sin(0/10000^0), cos(0/10000^0), sin(0/10000^0.02), cos(0/10000^0.02), ...]
Position 1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^0.02), cos(1/10000^0.02), ...]
...

Different dimensions encode position at different frequencies:
- Low dimensions: High frequency (changes every position)
- High dimensions: Low frequency (changes slowly)
```

**Learned vs Sinusoidal:**

| Type | Pros | Cons | Used In |
|------|------|------|---------|
| Sinusoidal | Extrapolates to longer sequences, no params | Fixed pattern | Original Transformer, many models |
| Learned | Flexible, data-adaptive | Limited to training length, extra params | BERT, GPT, ViT |

### 2. Layer Normalization: Training Stabilization

**The Problem:**

Deep networks have internal covariate shift:
```
Layer 1 output: mean=0.5, std=2.0
Layer 2 receives shifting distributions → hard to train!
```

**The Solution: Layer Normalization**

Normalize each sample independently across features:

```python
# For each sample:
μ = mean(x)
σ² = variance(x)
x_norm = (x - μ) / sqrt(σ² + ε)
output = γ * x_norm + β  # Learnable scale and shift
```

**Why it works:**
- Stabilizes gradient flow
- Allows higher learning rates
- Acts as regularization
- Enables deeper networks

**Pre-Norm vs Post-Norm:**

**Post-Norm (Original Transformer):**
```python
# Apply normalization AFTER residual
x = x + SubLayer(x)
x = LayerNorm(x)
```

**Pre-Norm (Modern Transformers):**
```python
# Apply normalization BEFORE sublayer
x = x + SubLayer(LayerNorm(x))
```

**Comparison:**

| Aspect | Post-Norm | Pre-Norm |
|--------|-----------|----------|
| Training stability | Less stable | More stable |
| Performance | Slightly better with good tuning | Good out-of-box |
| Depth | Harder to train very deep | Easier to train deep |
| Used in | Original Transformer | GPT-2, GPT-3, Modern models |

**Why Pre-Norm is preferred:**
- More stable gradient flow
- Easier to train deep networks (100+ layers)
- Less sensitive to learning rate

### 3. Feed-Forward Network: Position-Wise MLP

**After self-attention, apply the same MLP to each position independently:**

```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
       = ReLU(xW₁ + b₁)W₂ + b₂
```

**Architecture:**
```
d_model (512) → d_ff (2048) → d_model (512)
              ↑ ReLU/GELU
```

**Why it matters:**

1. **Non-linearity:** Self-attention is linear (weighted sum). FFN adds non-linear transformations.
2. **Capacity:** Most parameters are in FFN (2 × d_model × d_ff)
3. **Position-wise:** Each position processed independently (parallelizable)
4. **Bottleneck-expansion:** Expand then compress (information processing)

**Modern variants:**

**GELU Activation (used in BERT, GPT):**
```python
GELU(x) = x * Φ(x)  # Φ is CDF of standard normal
        ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Why GELU > ReLU:**
- Smooth (differentiable everywhere)
- Non-zero gradients for negative values
- Empirically better performance

**SwiGLU (used in LLaMA):**
```python
SwiGLU(x) = Swish(xW₁) ⊗ (xW₂)
Swish(x) = x * sigmoid(x)
```

**Parameter count:**
```
Standard FFN: 2 × d_model × d_ff
  = 2 × 512 × 2048 = 2,097,152 parameters per layer!
```

### 4. Residual Connections: Enabling Deep Networks

**The Problem:**

Deep networks suffer from vanishing gradients:
```
∂L/∂θ₁ = ∂L/∂θₙ × ∂θₙ/∂θₙ₋₁ × ... × ∂θ₂/∂θ₁
         ↑ Many multiplications → vanishes!
```

**The Solution: Skip Connections**

```python
output = x + SubLayer(x)
```

**Gradient flow:**
```
∂L/∂x = ∂L/∂output × (1 + ∂SubLayer/∂x)
        ↑ Always has "+1" term → gradients flow directly!
```

**Benefits:**
- Gradients flow directly to earlier layers
- Enables training very deep networks (100+ layers)
- Network can learn identity mapping (if SubLayer not helpful)
- Improves optimization landscape

**In Transformer Encoder:**

Two residual connections per layer:
```python
# Around self-attention
x = x + MultiHeadAttention(x)
x = LayerNorm(x)

# Around feed-forward
x = x + FFN(x)
x = LayerNorm(x)
```

---

## 🎯 Complete Encoder Layer

Putting it all together:

```python
class TransformerEncoderLayer(nn.Module):
    def forward(self, x, mask=None):
        # 1. Multi-head self-attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Residual + Norm

        # 2. Position-wise feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Residual + Norm

        return x
```

**Information flow:**

```
Input (batch, seq_len, d_model)
    ↓
Self-Attention (each token looks at all tokens)
    ↓ residual
LayerNorm
    ↓
Feed-Forward (process each position independently)
    ↓ residual
LayerNorm
    ↓
Output (batch, seq_len, d_model)
```

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Why positional encoding is necessary
- Sinusoidal vs learned positional encodings
- Layer normalization mathematics and placement
- Pre-norm vs post-norm architectures
- Feed-forward network role and capacity
- Residual connections and gradient flow
- How stacking layers builds hierarchical understanding
- BERT-style encoder applications

**Practical Skills:**
- Implement sinusoidal positional encoding
- Build complete transformer encoder layer
- Stack multiple encoder layers
- Apply masking for padding
- Add classification head for downstream tasks
- Train encoder for text classification
- Visualize encoder representations

---

## 🔑 Key Concepts

### 1. Hierarchical Representations

**Each encoder layer learns progressively abstract features:**

```
Layer 1: Local patterns (n-grams, adjacent words)
Layer 2: Phrases and simple syntax
Layer 3: Sentence structure and grammar
Layer 4: Semantic meaning and context
...
Layer 12: Abstract task-specific features
```

**BERT Analysis (12 layers):**
- **Lower layers (1-4):** Syntax, part-of-speech
- **Middle layers (5-8):** Semantic relationships
- **Upper layers (9-12):** Task-specific abstractions

### 2. Parameter Count

**For d_model=512, d_ff=2048, num_heads=8, num_layers=6:**

```
Per encoder layer:
- Multi-head attention: 4 × 512 × 512 = 1,048,576 (Q, K, V, O projections)
- Feed-forward: 2 × 512 × 2048 = 2,097,152
- Layer norms: ~2 × 512 = 1,024
Total per layer: ~3,146,752 parameters

6 layers: ~18.9M parameters
BERT-base (12 layers): ~110M parameters
BERT-large (24 layers): ~340M parameters
```

**Most parameters are in FFN!**

### 3. Computational Complexity

**Per encoder layer:**

| Component | Complexity | Dominant Factor |
|-----------|-----------|-----------------|
| Self-Attention | O(n² × d) | Sequence length n |
| Feed-Forward | O(n × d²) | Model dimension d |
| Total | O(n²×d + n×d²) | Depends on n vs d |

**Typical values:**
- Short sequences (n < d): FFN dominates
- Long sequences (n > d): Attention dominates

**Example (BERT-base):**
- n=512, d=768
- Attention: 512² × 768 ≈ 201M ops
- FFN: 512 × 768² ≈ 302M ops

---

## 🧪 Exercises

### Exercise 1: Positional Encoding (45 mins)

**What You'll Learn:**
- Sinusoidal encoding mathematics
- Why different frequencies for different dimensions
- Visualization of positional patterns
- Comparison with learned embeddings

**Why It Matters:**
Position encoding is what makes transformers understand order. Without it, transformers are permutation-invariant (bag of words). Understanding this deeply reveals how transformers capture sequential information.

**Tasks:**
1. Implement `get_positional_encoding(seq_len, d_model)`
2. Visualize encoding as heatmap
3. Compare different positions
4. Implement learned positional embeddings
5. Test extrapolation to longer sequences

**Expected behavior:**
```python
pe = get_positional_encoding(100, 512)
assert pe.shape == (100, 512)
assert torch.allclose(pe[0, 0], torch.sin(torch.tensor(0.0)))  # First element
# Different positions should have different encodings
assert not torch.allclose(pe[0], pe[1])
```

---

### Exercise 2: Layer Normalization (30 mins)

**What You'll Learn:**
- Layer norm implementation
- Difference from batch norm
- Effect on training stability
- Pre-norm vs post-norm comparison

**Why It Matters:**
Layer norm is crucial for training deep transformers. Understanding when and where to apply it affects model performance and training stability significantly.

**Tasks:**
1. Implement `LayerNorm` from scratch
2. Compare to `nn.LayerNorm`
3. Visualize effect on activation distribution
4. Test pre-norm vs post-norm configurations
5. Measure gradient flow differences

**Key insight:**
```python
# Batch Norm: Normalize across batch dimension
# Layer Norm: Normalize across feature dimension

# For input (batch=32, seq=512, d_model=768):
BatchNorm: Normalize over 32 samples (each feature independently)
LayerNorm: Normalize over 768 features (each sample independently)

# Why LayerNorm for transformers?
# - Works with batch size 1 (batch norm doesn't)
# - Variable sequence lengths
# - More stable for transformers
```

---

### Exercise 3: Feed-Forward Network (30 mins)

**What You'll Learn:**
- Position-wise MLP architecture
- Expansion ratio (d_ff / d_model)
- Activation function comparison (ReLU, GELU, SwiGLU)
- Parameter efficiency

**Why It Matters:**
FFN contains most parameters in transformers. Understanding its role and design choices helps optimize model capacity and efficiency.

**Tasks:**
1. Implement `PositionWiseFeedForward` class
2. Test different expansion ratios (2x, 4x, 8x)
3. Compare ReLU vs GELU activations
4. Measure computational cost
5. Analyze parameter distribution

**Comparison:**
```python
# Standard BERT FFN
d_model = 768
d_ff = 3072  # 4x expansion
params = 2 × 768 × 3072 = 4,718,592

# Efficient variant
d_ff = 2048  # 2.67x expansion
params = 2 × 768 × 2048 = 3,145,728  # 33% reduction

# Trade-off: capacity vs efficiency
```

---

### Exercise 4: Complete Encoder Layer (60 mins)

**What You'll Learn:**
- Combining all components
- Residual connection placement
- Dropout for regularization
- Pre-norm vs post-norm implementation

**Why It Matters:**
This is the core building block of BERT, ViT, and all encoder-based models. Implementing it from scratch builds deep understanding of how these models work.

**Tasks:**
1. Implement `TransformerEncoderLayer` class
2. Combine attention, FFN, residuals, layer norm
3. Add dropout for regularization
4. Implement both pre-norm and post-norm variants
5. Test forward pass with sample data
6. Verify gradient flow

**Architecture:**
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
```

---

### Exercise 5: Multi-Layer Encoder (60 mins)

**What You'll Learn:**
- Stacking encoder layers
- Parameter sharing vs independent layers
- Depth vs performance trade-off
- Memory and computation scaling

**Why It Matters:**
Understanding how depth affects model capacity and what each layer learns is crucial for designing efficient architectures.

**Tasks:**
1. Implement `TransformerEncoder` class (stack of N layers)
2. Add embedding and positional encoding
3. Implement parameter-efficient variants
4. Test with different depths (1, 6, 12 layers)
5. Analyze representation quality per layer
6. Measure computational cost

**Stacking pattern:**
```python
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)  # Final layer norm

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

---

### Exercise 6: Text Classification with Encoder (90 mins)

**What You'll Learn:**
- Adding classification head
- [CLS] token pooling (BERT-style)
- Mean pooling vs CLS pooling
- Fine-tuning vs training from scratch

**Why It Matters:**
This is how BERT is used in practice! Understanding how to adapt encoders for downstream tasks is essential for real-world applications.

**Tasks:**
1. Build complete encoder for sentiment classification
2. Add embedding layer and positional encoding
3. Implement [CLS] token pooling
4. Add classification head
5. Train on IMDB or AG News
6. Compare pooling strategies
7. Visualize learned representations

**Complete architecture:**
```python
class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=512,
                 num_layers=6, num_heads=8, d_ff=2048):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(num_layers, d_model,
                                         num_heads, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        # Embed and add positions
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Encode
        x = self.encoder(x, mask)

        # Pool (take [CLS] token or mean)
        pooled = x[:, 0, :]  # [CLS] token

        # Classify
        return self.classifier(pooled)
```

---

## 📝 Design Patterns

### Pattern 1: Sinusoidal Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

### Pattern 2: Pre-Norm Encoder Layer

```python
class PreNormEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm: normalize before sublayer
        x = x + self.dropout(self.self_attn(self.norm1(x),
                                            self.norm1(x),
                                            self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

### Pattern 3: BERT-Style Pooling

```python
class BERTPooling(nn.Module):
    """Different pooling strategies for encoder outputs."""

    def cls_pooling(self, x):
        """Use [CLS] token (first position)."""
        return x[:, 0, :]

    def mean_pooling(self, x, mask=None):
        """Average all tokens (excluding padding)."""
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(x.size()).float()
            return torch.sum(x * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return x.mean(dim=1)

    def max_pooling(self, x, mask=None):
        """Max over all tokens."""
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        return x.max(dim=1)[0]
```

---

## ✅ Solutions

Complete implementations in `solution/` directory.

**Files:**
- `01_positional_encoding.py` - Sinusoidal and learned encodings
- `02_encoder_layer.py` - Complete encoder layer
- `03_transformer_encoder.py` - Multi-layer encoder
- `04_text_classification.py` - End-to-end classification model

Run examples:
```bash
cd solution
python 01_positional_encoding.py
python 02_encoder_layer.py
python 03_transformer_encoder.py
python 04_text_classification.py
```

---

## 🎓 Key Takeaways

1. **Positional encoding adds order** - Without it, transformers are permutation-invariant
2. **Self-attention builds context** - Each layer refines understanding
3. **FFN adds capacity** - Most parameters, non-linear transformations
4. **Residuals enable depth** - Gradient flow for 100+ layer networks
5. **Layer norm stabilizes** - Enables higher learning rates
6. **Pre-norm is modern default** - More stable than post-norm
7. **Stacking creates hierarchy** - Lower layers: syntax, upper layers: semantics
8. **Encoders are versatile** - Text, vision, audio, proteins

**The Core Insight:**
```
Transformer Encoder =
    Attention (understand relationships) +
    FFN (non-linear processing) +
    Residuals (enable depth) +
    Normalization (stability)

Stacked N times for hierarchical understanding.
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**At Google (BERT in Search):**
- Processes billions of search queries daily
- BERT encoder understands query intent
- Improved search quality by 10%
- Handles 70+ languages

**At OpenAI (CLIP):**
- Encoder processes both images and text
- 400M image-text pairs training
- Powers DALL-E 2, ChatGPT vision
- Zero-shot image classification

**At Meta:**
- Content understanding (hate speech, misinformation)
- Multilingual translation (100+ languages)
- Feed ranking (post relevance)
- XLM-R encoder: trained on 2.5TB text

**At DeepMind (AlphaFold):**
- Protein sequence encoder
- Revolutionized biology
- Predicted 200M protein structures

**Cost and Scale:**
- BERT-base training: ~4 days on 16 TPUs (~$7K)
- BERT-large training: ~4 days on 64 TPUs (~$25K)
- Inference: Millions of requests/day

---

## 🚀 Next Steps

1. **Complete all exercises** - Build encoder from scratch
2. **Visualize layer representations** - What does each layer learn?
3. **Read BERT paper** - Pre-training strategies
4. **Move to Lab 3** - Full Transformer (encoder + decoder)

---

## 💪 Bonus Challenges

1. **Implement Vision Transformer (ViT)**
   - Split image into patches
   - Treat as sequence
   - Apply encoder
   - Classification on ImageNet

2. **Relative Positional Encoding**
   - Implement T5-style relative attention bias
   - Compare to absolute encoding
   - Test on long sequences

3. **Efficient Transformers**
   - Implement Linformer (linear attention)
   - Compare complexity O(n²) vs O(n)
   - Benchmark speed and memory

4. **Layer Analysis**
   - Train encoder on text classification
   - Extract representations from each layer
   - Visualize with PCA/t-SNE
   - Identify what each layer learns

5. **Multi-Task Encoder**
   - Train single encoder for multiple tasks
   - Shared encoder, task-specific heads
   - Compare to task-specific encoders

---

## 📚 Essential Resources

**Foundational Papers:**
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
  - Original transformer paper
  - Encoder and decoder architecture

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
  - Encoder-only architecture
  - Masked language modeling
  - Revolutionized NLP

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020 (ViT)
  - Transformers for vision
  - Encoder on image patches

**Analysis Papers:**
- [What Does BERT Look At?](https://arxiv.org/abs/1906.04341) - Clark et al., 2019
- [A Primer on Neural Network Architectures for Natural Language Processing](https://arxiv.org/abs/1807.10854)

**Tutorials:**
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

**Tools:**
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Pre-trained models
- [BertViz](https://github.com/jessevig/bertviz) - Attention visualization

---

## 🤔 Common Pitfalls

### Pitfall 1: Forgetting Positional Encoding

```python
# ❌ No position information!
x = self.embedding(tokens)
x = self.encoder(x)

# ✓ Add positions
x = self.embedding(tokens)
x = self.pos_encoding(x)  # Essential!
x = self.encoder(x)
```

**Why:** Attention is permutation-invariant. Without positions, "dog bites man" = "man bites dog"!

### Pitfall 2: Wrong Layer Norm Placement

```python
# ❌ Post-norm (harder to train deep networks)
x = LayerNorm(x + SubLayer(x))

# ✓ Pre-norm (modern, more stable)
x = x + SubLayer(LayerNorm(x))
```

**Why:** Pre-norm provides better gradient flow for deep networks (12+ layers).

### Pitfall 3: Ignoring Padding Masks

```python
# ❌ Attention to padding tokens!
output = encoder(x)  # Attends to <PAD> tokens

# ✓ Mask padding
mask = (x != PAD_TOKEN)
output = encoder(x, mask=mask)
```

**Why:** Padding tokens have no information and should be ignored.

### Pitfall 4: Inefficient Positional Encoding

```python
# ❌ Recompute every forward pass
pe = compute_positional_encoding(seq_len, d_model)
x = x + pe

# ✓ Precompute and buffer
self.register_buffer('pe', precomputed_pe)
x = x + self.pe[:, :seq_len, :]
```

**Why:** Positional encoding is deterministic - compute once, reuse always.

---

## 💡 Pro Tips

1. **Start with pre-norm** - More stable for deep networks
2. **Use GELU not ReLU** - Better performance in transformers
3. **Warmup learning rate** - Transformer training is sensitive to LR
4. **Gradient clipping** - Prevents explosion in early training
5. **Layer-wise LR decay** - Lower layers need smaller LR
6. **Pre-trained > random init** - Use BERT/RoBERTa when possible
7. **Monitor attention entropy** - Detect issues early

---

## ✨ You're Ready When...

- [ ] You understand why positional encoding is necessary
- [ ] You can explain sinusoidal encoding mathematics
- [ ] You've implemented complete encoder layer from scratch
- [ ] You understand pre-norm vs post-norm trade-offs
- [ ] You know why FFN expansion ratio matters
- [ ] You can explain residual connection benefits
- [ ] You've stacked multiple encoder layers
- [ ] You can add task-specific heads for classification
- [ ] You understand BERT-style [CLS] pooling
- [ ] You've trained an encoder on real data

**Remember:** The transformer encoder is the foundation of modern NLP and beyond. Master it, and you understand BERT, ViT, CLIP, and countless other models!
