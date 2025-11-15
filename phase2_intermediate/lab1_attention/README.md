# Lab 1: Attention Mechanisms - The Revolution in Deep Learning 🎯

> **Time:** 3-4 hours
> **Difficulty:** Intermediate
> **Goal:** Master attention mechanisms that power modern AI from GPT to BERT

---

## 📖 Why This Lab Matters

Attention is the single most important innovation in deep learning of the last decade. It's the foundation of:

- **GPT-4** (OpenAI) - Language models
- **BERT** (Google) - Understanding text
- **CLIP** (OpenAI) - Vision-language models
- **AlphaFold** (DeepMind) - Protein structure prediction
- **Stable Diffusion** - Image generation
- **Whisper** (OpenAI) - Speech recognition

Before attention (2017), neural networks struggled with sequences. After attention, we unlocked superhuman performance on language, vision, and beyond.

**This lab teaches you the mechanism that changed AI forever.**

---

## 🧠 The Big Picture: Why Attention Revolutionized Deep Learning

### The Problem: Fixed-Length Bottleneck

Before attention, sequence-to-sequence models worked like this:

```
Encoder:                    Decoder:
"The cat sat" → RNN → [h₁, h₂, h₃, h₄] → compress → [c]
                                                      ↓
                                            [c] → RNN → "Le chat"
                                            ↑
                                    Fixed-length vector!
```

**The bottleneck:**
- Entire input sentence compressed into ONE fixed-size vector `c`
- Long sentences lose information (compression artifacts)
- All input tokens treated equally (no focus on relevant parts)
- Performance degrades on long sequences

**Real example:**
```
Input (50 words): "The quick brown fox jumps over the lazy dog and then..."
Compressed to: [0.23, -0.45, 0.67, ...]  ← 512 numbers must represent everything!
Problem: Later words forgotten, details lost
```

### The Solution: Attention

**Key insight:** Don't compress everything into one vector. Instead, let the decoder LOOK AT all encoder outputs and FOCUS on relevant parts.

```
Encoder outputs: [h₁, h₂, h₃, ..., hₙ]  ← Keep ALL states!
                    ↓    ↓    ↓       ↓
Decoder: "Which words matter RIGHT NOW?"
         → Compute attention weights: [0.1, 0.6, 0.2, 0.1]
         → Weighted sum = context vector
         → Generate next word
```

**Breakthrough:**
- No fixed bottleneck
- Model learns what to focus on
- Works on sequences of any length
- Interpretable (visualize attention weights)

### Real-World Impact

**Before attention (2016):**
- Machine translation quality: 25 BLEU score
- Sequence limit: ~50 tokens

**After attention (2017+):**
- Machine translation quality: 40+ BLEU score
- Sequence limit: 100,000+ tokens (GPT-4)
- New capabilities: zero-shot learning, in-context learning

---

## 🔬 Deep Dive: Attention Mathematics

### The Core Formula: Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**This single formula powers GPT, BERT, and modern AI.**

Let's break it down:

### Step 1: Query, Key, Value

Think of attention like a **database lookup:**

```
Query (Q):   "What am I looking for?"
Key (K):     "What do I contain?"
Value (V):   "What do I actually return?"
```

**Analogy: YouTube Search**
```
Your query:     "pytorch tutorial"        ← Q
Video titles:   ["pytorch basics", ...]   ← K (for matching)
Video content:  [actual video data]       ← V (what you get)

Attention finds videos matching your query and returns their content.
```

**In neural networks:**
```python
Q = Linear_q(decoder_state)   # What decoder is looking for
K = Linear_k(encoder_outputs)  # What encoder states contain
V = Linear_v(encoder_outputs)  # What encoder states return
```

### Step 2: Compute Similarity (QK^T)

Compute how similar each query is to each key:

```
scores = Q @ K^T

Example:
Q: (1, 512)  # Current decoder state
K: (100, 512)  # 100 encoder tokens
scores: (1, 100)  # Similarity to each token
```

**Intuition:**
- High score → Query and key are similar → Pay attention!
- Low score → Query and key differ → Ignore

### Step 3: Scale by √d_k

**Problem:** For large dimensions, dot products get very large.

```
If d_k = 512:
  Q·K can be hundreds (large variance)
  softmax(hundreds) → [0.0000, 0.0000, 0.9999, 0.0000]  ← Almost one-hot!
  Gradients vanish (softmax saturated)
```

**Solution:** Divide by √d_k to stabilize:

```
scores = QK^T / √512 ≈ QK^T / 22.6
```

**Why specifically √d_k?**

Mathematical reason:
```
If Q, K ~ N(0, 1) (standard normal):
  Q·K ~ N(0, d_k)  ← Variance grows with dimension!
  Q·K / √d_k ~ N(0, 1)  ← Stable variance
```

This keeps gradients healthy!

### Step 4: Softmax (Normalize to Probabilities)

```
attention_weights = softmax(scores / √d_k)
                  = exp(scores) / Σ exp(scores)
```

**Effect:**
- Converts scores to probabilities (sum to 1)
- Amplifies differences (high scores higher, low scores lower)
- Differentiable (gradients flow)

**Example:**
```
scores:  [2.3, -1.2, 4.5, 0.1]
         ↓ softmax
weights: [0.12, 0.01, 0.85, 0.02]  ← Probabilities!
```

### Step 5: Weighted Sum (Compute Output)

```
output = attention_weights @ V
```

**Intuition:** Combine values based on attention weights.

```
weights: [0.1, 0.6, 0.2, 0.1]
values:  [v₁,  v₂,  v₃,  v₄]
output:  0.1*v₁ + 0.6*v₂ + 0.2*v₃ + 0.1*v₄
         ↑
         Mostly v₂ (highest weight)
```

### Complete Example

```python
# Toy example: 3 tokens, dimension 4
Q = [[1, 0, 1, 0]]        # (1, 4) - decoder query
K = [[1, 0, 0, 0],        # (3, 4) - encoder keys
     [0, 1, 1, 0],
     [0, 0, 1, 1]]
V = [[1, 2, 3, 4],        # (3, 4) - encoder values
     [5, 6, 7, 8],
     [9, 10, 11, 12]]

# Step 1: Compute scores
scores = Q @ K^T = [[2, 1, 1]]  # (1, 3)

# Step 2: Scale
scores = scores / √4 = [[1.0, 0.5, 0.5]]

# Step 3: Softmax
weights = softmax([[1.0, 0.5, 0.5]]) = [[0.5, 0.25, 0.25]]

# Step 4: Weighted sum
output = weights @ V = [[4.0, 5.0, 6.0, 7.0]]
                       ↑ Weighted average of values!
```

---

## 🎯 Multi-Head Attention: Parallel Perspectives

### Why Multiple Heads?

Single attention head learns ONE type of relationship:
```
Single head: "Focus on subject-verb agreement"
```

Multiple heads learn DIFFERENT relationships simultaneously:
```
Head 1: "Subject-verb agreement"
Head 2: "Coreference resolution" (pronouns)
Head 3: "Syntactic dependencies"
Head 4: "Semantic similarity"
...
```

**Analogy:** Multiple experts looking at the same data from different angles.

### Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O

where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Steps:**

1. **Project Q, K, V for each head:**
```python
# 8 heads, model_dim=512, head_dim=64
for i in range(num_heads):
    Q_i = Q @ W^Q_i  # (batch, seq, 512) @ (512, 64) = (batch, seq, 64)
    K_i = K @ W^K_i
    V_i = V @ W^V_i
```

2. **Apply attention per head:**
```python
    head_i = Attention(Q_i, K_i, V_i)  # (batch, seq, 64)
```

3. **Concatenate all heads:**
```python
output = Concat([head_1, ..., head_8])  # (batch, seq, 512)
```

4. **Final projection:**
```python
output = output @ W^O  # (batch, seq, 512)
```

**Why it works:**
- Each head has smaller dimension (64 vs 512)
- Total computation same as single head
- But learns 8 different patterns!

---

## 📊 Self-Attention vs Cross-Attention

### Self-Attention

**Query, key, value all come from the SAME sequence.**

```python
# Encoder self-attention
Q = K = V = encoder_input
attention = Attention(Q, K, V)
→ Each token attends to every other token in same sequence
```

**Use cases:**
- Understanding sentence structure (BERT)
- Image patch relationships (Vision Transformers)
- Token context (GPT)

**Example:** "The cat sat on the mat"
```
"cat" attends to: ["The", "cat", "sat", "on", "the", "mat"]
→ Learns "cat" is the subject doing the action
```

### Cross-Attention

**Query from one sequence, key/value from another.**

```python
# Decoder cross-attention
Q = decoder_state
K = V = encoder_outputs
attention = Attention(Q, K, V)
→ Decoder attends to encoder outputs
```

**Use cases:**
- Machine translation (attend to source while generating target)
- Image captioning (attend to image while generating text)
- Question answering (attend to context while generating answer)

**Example:** Translating "The cat" → "Le chat"
```
Decoder generating "chat":
Q = current decoder state
K, V = encoder outputs for ["The", "cat"]
→ Attends to "cat" to generate "chat"
```

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Why attention solves the fixed-length bottleneck
- How scaled dot-product attention works mathematically
- The role of queries, keys, and values
- Why scaling by √d_k prevents gradient issues
- Multi-head attention and parallel learning
- Self-attention vs cross-attention
- Computational complexity O(n²) and implications

**Practical Skills:**
- Implement scaled dot-product attention from scratch
- Build multi-head attention mechanism
- Apply causal masking for autoregressive models
- Create padding masks for variable-length sequences
- Implement cross-attention for seq2seq
- Visualize attention weights for interpretability

---

## 🔑 Key Concepts

### 1. Computational Complexity

**Attention is O(n²) in sequence length:**

```
For sequence length n, dimension d:
- Compute QK^T: O(n² · d)
- Softmax: O(n²)
- Weighted sum: O(n² · d)
Total: O(n² · d)
```

**Implications:**

| Sequence Length | Memory  | Time   |
|----------------|---------|--------|
| 512 tokens     | 1 GB    | 0.1s   |
| 2048 tokens    | 16 GB   | 1.6s   |
| 8192 tokens    | 256 GB  | 25s    |

**This is why:**
- GPT-3 limited to 2048 tokens
- Sparse attention variants exist (Longformer, BigBird)
- Flash Attention optimizes memory

### 2. Masking

**Causal Mask (for GPT-style models):**
```python
# Prevent attending to future tokens
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
scores = scores.masked_fill(mask, -inf)
```

**Why needed:**
```
Generating "The cat sat":
When predicting "sat", can only see ["The", "cat"]
Cannot see future tokens (would be cheating!)
```

**Padding Mask:**
```python
# Ignore padding tokens
pad_mask = (input == PAD_TOKEN)
scores = scores.masked_fill(pad_mask, -inf)
```

**Why needed:**
```
Batch: ["short", "this is longer sentence <PAD> <PAD>"]
Don't attend to <PAD> tokens (no information)
```

### 3. Attention Interpretation

**Attention weights reveal model behavior:**

```python
# Example: "The cat sat on the mat"
weights = model.attention_weights
→ [[0.1, 0.6, 0.1, 0.1, 0.05, 0.05],  # "sat" focuses on "cat"
   [0.05, 0.1, 0.05, 0.7, 0.05, 0.05], # "on" focuses on "mat"
   ...]
```

**Applications:**
- Debugging models (what is it looking at?)
- Trust and interpretability
- Discovering linguistic patterns

---

## 🧪 Exercises

### Exercise 1: Scaled Dot-Product Attention (45 mins)

**What You'll Learn:**
- The core attention mechanism
- Matrix operations for attention
- Why scaling matters
- Softmax normalization

**Why It Matters:**
This is the atomic unit of modern AI. GPT-4 has thousands of these attention blocks. Understanding one deeply means understanding them all.

**Tasks:**
1. Implement `scaled_dot_product_attention(Q, K, V, mask=None)`
2. Handle optional masking
3. Test with toy examples
4. Verify outputs make sense

**Expected behavior:**
```python
Q, K, V = torch.randn(1, 10, 64)  # 10 tokens, 64 dims
output, weights = scaled_dot_product_attention(Q, K, V)
assert output.shape == (1, 10, 64)
assert weights.shape == (1, 10, 10)
assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 10))  # Sum to 1
```

---

### Exercise 2: Multi-Head Attention (60 mins)

**What You'll Learn:**
- Splitting embeddings into heads
- Parallel attention computation
- Concatenation and projection
- Parameter management

**Why It Matters:**
Multi-head attention is what makes transformers powerful. It's used in EVERY transformer layer in BERT, GPT, and modern models.

**Tasks:**
1. Implement `MultiHeadAttention` class
2. Initialize Q, K, V projections for each head
3. Implement forward pass with head splitting
4. Add output projection
5. Verify parameter count

**Key insight:**
```python
# Standard: 512-dimensional single attention
params_single = 512 * 512 * 3  # Q, K, V projections

# Multi-head: 8 heads × 64 dimensions
params_multi = 512 * 512 * 3  # Same total parameters!

# But learns 8 different patterns instead of 1
```

---

### Exercise 3: Causal Masking (30 mins)

**What You'll Learn:**
- Creating causal masks
- Why autoregressive models need masking
- Applying masks to attention scores
- -inf trick for softmax

**Why It Matters:**
Every autoregressive model (GPT, language models) uses causal masking. Without it, the model would cheat by seeing future tokens!

**Tasks:**
1. Create `create_causal_mask(seq_len)` function
2. Test masking prevents future attention
3. Visualize masked attention weights
4. Implement padding mask combination

**Visualization:**
```
Causal mask (5 tokens):
[[0, -∞, -∞, -∞, -∞],    Token 0: can only see itself
 [0,  0, -∞, -∞, -∞],    Token 1: sees tokens 0-1
 [0,  0,  0, -∞, -∞],    Token 2: sees tokens 0-2
 [0,  0,  0,  0, -∞],    Token 3: sees tokens 0-3
 [0,  0,  0,  0,  0]]    Token 4: sees all tokens
```

---

### Exercise 4: Cross-Attention (45 mins)

**What You'll Learn:**
- Difference from self-attention
- Encoder-decoder attention
- Separate Q vs K,V sources
- Use in sequence-to-sequence models

**Why It Matters:**
Cross-attention enables:
- Machine translation (attend to source language)
- Image captioning (attend to image while writing)
- Multimodal models (CLIP, Flamingo)

**Tasks:**
1. Implement `CrossAttention` class
2. Take Q from decoder, K/V from encoder
3. Test with different sequence lengths
4. Compare to self-attention behavior

---

### Exercise 5: Attention Visualization (30 mins)

**What You'll Learn:**
- Extracting attention weights
- Heatmap visualization
- Interpreting attention patterns
- Head comparison

**Why It Matters:**
Visualization helps:
- Debug models (what is it focusing on?)
- Build trust (interpretability)
- Discover patterns (linguistic knowledge)
- Present results (papers, demos)

**Tasks:**
1. Extract attention weights from model
2. Plot as heatmap (matplotlib/seaborn)
3. Visualize all heads in multi-head attention
4. Analyze patterns (what does each head learn?)

**Example output:**
```
Head 1: Focuses on adjacent words (local context)
Head 2: Focuses on subject-verb pairs (syntax)
Head 3: Focuses on similar words (semantics)
```

---

## 📝 Design Patterns

### Pattern 1: Efficient Attention Implementation

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        d_k = Q.size(-1)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        output = attn @ V
        return output, attn
```

### Pattern 2: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # (batch, seq, d_model) → (batch, heads, seq, d_k)
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        # Project
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Attention
        output, attn = self.attention(Q, K, V, mask)

        # Concatenate heads
        batch_size, _, seq_len, _ = output.size()
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Output projection
        return self.W_o(output), attn
```

### Pattern 3: Causal Masking

```python
def create_causal_mask(seq_len, device='cpu'):
    """Create lower triangular mask for autoregressive attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return ~mask  # Invert (1 = attend, 0 = mask)

# Usage
mask = create_causal_mask(10)
output, attn = attention(Q, K, V, mask=mask)
```

---

## ✅ Solutions

Complete implementations in `solution/` directory.

**Files:**
- `01_self_attention.py` - Scaled dot-product attention
- `02_multi_head.py` - Multi-head attention
- `03_masking.py` - Causal and padding masks
- `04_visualization.py` - Attention heatmaps

Run examples:
```bash
cd solution
python 01_self_attention.py
python 02_multi_head.py
python 03_masking.py
python 04_visualization.py
```

---

## 🎓 Key Takeaways

1. **Attention is weighted sum** - Focus on relevant parts, ignore irrelevant
2. **QKV framework** - Query what to find, Key what to match, Value what to return
3. **Scaling prevents gradient issues** - √d_k keeps variance stable
4. **Multi-head learns diverse patterns** - Parallel experts looking at different aspects
5. **Self-attention = within sequence** - Understand context and relationships
6. **Cross-attention = between sequences** - Align and translate information
7. **O(n²) complexity** - Quadratic in sequence length (limiting factor)
8. **Masking enables autoregressive** - Prevent seeing future for language models

**The Revolution:**
```
Before: Fixed bottleneck, limited context, poor long sequences
After: Dynamic focus, unlimited context, state-of-the-art everywhere
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**At OpenAI (GPT-4):**
- 96 layers of multi-head attention
- 96 heads per layer (learning 9,216 different patterns!)
- Processes 32k tokens (1 billion attention computations per forward pass)
- Trained on trillions of tokens

**At Google (BERT):**
- Powers Google Search understanding
- Billions of queries per day
- Attention reveals what words are important
- Improved search quality by 10%

**At Meta:**
- Feed ranking (what posts to show)
- Content understanding (hate speech detection)
- Translation (50+ languages)
- All use attention mechanisms

**Cost implications:**
- Single GPT-4 query: ~$0.03
- Most cost is attention computation
- Optimizations (Flash Attention) reduce cost 3x

---

## 🚀 Next Steps

1. **Complete all exercises** - Build intuition through implementation
2. **Visualize attention** - See what models learn
3. **Read "Attention is All You Need"** - The paper that started it all
4. **Move to Lab 2** - Transformer Encoder (attention in action)

---

## 💪 Bonus Challenges

1. **Relative Positional Attention**
   - Implement relative position bias (T5-style)
   - Compare to absolute positions
   - Test on long sequences

2. **Flash Attention**
   - Implement memory-efficient attention
   - Compare memory usage to standard
   - Benchmark speed improvements

3. **Sparse Attention Patterns**
   - Implement sliding window attention (local)
   - Implement strided attention (global)
   - Compare coverage vs computation

4. **Attention Analysis**
   - Extract attention from pre-trained BERT
   - Analyze what different heads learn
   - Find syntactic vs semantic heads

5. **Cross-Modal Attention**
   - Implement image-text attention (CLIP-style)
   - Test on image captioning
   - Visualize what text attends to in image

---

## 📚 Essential Resources

**The Foundational Paper:**
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
  - The paper that started the transformer revolution
  - Introduced multi-head attention and transformers
  - Most cited ML paper of the decade

**Key Follow-up Papers:**
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al., 2019 (GPT-2)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022

**Tutorials:**
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng

**Visualizations:**
- [BertViz](https://github.com/jessevig/bertviz) - Attention visualization tool
- [Transformer Explainability](https://transformer-circuits.pub/) - Anthropic research

---

## 🤔 Common Pitfalls

### Pitfall 1: Forgetting to Scale

```python
# ❌ Unstable for large d_k
scores = Q @ K.T
attn = softmax(scores)

# ✓ Stable attention
scores = Q @ K.T / math.sqrt(d_k)
attn = softmax(scores)
```

**Why:** Large d_k → large dot products → saturated softmax → vanishing gradients

### Pitfall 2: Wrong Mask Application

```python
# ❌ Mask after softmax (doesn't work!)
attn = softmax(scores)
attn = attn * mask

# ✓ Mask before softmax
scores = scores.masked_fill(mask == 0, -1e9)
attn = softmax(scores)
```

**Why:** Softmax needs to ignore masked positions entirely (use -inf)

### Pitfall 3: Shape Confusion

```python
# ❌ Wrong transpose
scores = Q @ K  # Error! (batch, seq, d) @ (batch, seq, d)

# ✓ Correct transpose
scores = Q @ K.transpose(-2, -1)  # (batch, seq, d) @ (batch, d, seq)
```

**Why:** Attention is (seq_len, seq_len) matrix, not (seq_len, d)

### Pitfall 4: Inefficient Multi-Head

```python
# ❌ Separate linear layers per head (slow, many parameters)
self.heads = nn.ModuleList([
    nn.Linear(d_model, d_k) for _ in range(num_heads)
])

# ✓ Single projection, split afterward (fast, efficient)
self.W_q = nn.Linear(d_model, d_model)
# Then split into heads
```

---

## 💡 Pro Tips

1. **Always scale by √d_k** - Prevents gradient issues
2. **Use -1e9 for masking** - Effectively -∞ without numerical issues
3. **Check attention sum to 1** - Debug softmax
4. **Visualize early and often** - Understand what model learns
5. **Batch matrix multiply** - Use `@` for efficiency
6. **GPU benefits** - Attention is highly parallelizable
7. **Profile memory** - O(n²) memory can surprise you

---

## ✨ You're Ready When...

- [ ] You can explain why attention solves the bottleneck problem
- [ ] You understand the QKV framework intuitively
- [ ] You can derive the attention formula
- [ ] You've implemented scaled dot-product attention
- [ ] You understand why √d_k scaling matters
- [ ] You can explain multi-head attention benefits
- [ ] You know when to use self vs cross attention
- [ ] You can implement and apply causal masking
- [ ] You understand O(n²) complexity implications
- [ ] You can visualize and interpret attention weights

**Remember:** Attention is the most important mechanism in modern AI. Master it, and you understand the foundation of GPT, BERT, and beyond!
