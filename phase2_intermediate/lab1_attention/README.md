# Lab 1: Self-Attention Mechanism 🎯

> **Time:** 2-3 hours | **Difficulty:** Intermediate

---

## 📚 Theory Brief (15 mins)

### What is Attention?

Attention allows a model to focus on relevant parts of the input when processing each element. Instead of treating all inputs equally, attention computes a weighted sum where weights represent importance.

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What do I actually output?"
- **d_k**: Dimension for scaling (prevents vanishing gradients)

### Multi-Head Attention

Instead of one attention, use multiple parallel attention "heads":
- Each head learns different relationships
- Heads are concatenated and projected
- Allows model to attend to different representation subspaces

---

## 🎯 Learning Objectives

By the end of this lab, you'll be able to:
- Implement scaled dot-product attention
- Build multi-head attention from scratch
- Apply causal masking for autoregressive models
- Create padding masks
- Implement cross-attention
- Visualize attention weights

---

## 📝 Exercises

### Exercise 1: Basic Attention (30 mins)
Implement scaled dot-product attention.

**Tasks:**
- Create Q, K, V projections
- Compute attention scores
- Apply softmax
- Return weighted values

### Exercise 2: Multi-Head Attention (45 mins)
Extend to multiple parallel heads.

**Tasks:**
- Split embeddings into heads
- Apply attention per head
- Concatenate outputs
- Final projection

### Exercise 3: Masking (30 mins)
Implement causal and padding masks.

**Tasks:**
- Create lower triangular mask (causal)
- Create padding mask from sequence
- Apply masks before softmax

### Exercise 4: Attention Visualization (30 mins)
Visualize what the model attends to.

**Tasks:**
- Plot attention weights as heatmap
- Analyze attention patterns
- Compare different heads

---

## 💻 Code Structure

```python
# See solution/01_self_attention.py for complete implementation

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        # TODO: Initialize Q, K, V projections
        # TODO: Initialize output projection
        pass

    def forward(self, x, mask=None):
        # TODO: Project to Q, K, V
        # TODO: Split into heads
        # TODO: Compute attention
        # TODO: Concatenate heads
        # TODO: Output projection
        pass
```

---

## ✅ Solution

Complete working implementation in `solution/01_self_attention.py`

Run it:
```bash
cd solution
python 01_self_attention.py
```

---

## 🎓 Key Takeaways

1. **Attention = weighted sum** based on similarity
2. **Scaling by √d_k** prevents gradient issues
3. **Multi-head** captures different relationships
4. **Causal mask** prevents looking at future tokens
5. **Padding mask** ignores padding tokens

---

## 🚀 Next Steps

Once complete, move to **Lab 2: Transformer Encoder**!
