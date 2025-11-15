# Lab 2: Transformer Encoder 🏗️

> **Time:** 3-4 hours | **Difficulty:** Intermediate

---

## 📚 Theory Brief

The Transformer Encoder processes input sequences using self-attention and feed-forward layers:

```
Input → Embedding → Positional Encoding
  ↓
[Multi-Head Attention → Add & Norm → FFN → Add & Norm] × N layers
  ↓
Output Embeddings
```

**Key Components:**
1. **Positional Encoding:** Add position information to embeddings
2. **Self-Attention:** Model relationships between tokens
3. **Feed-Forward:** Non-linear transformation
4. **Residual Connections:** Improve gradient flow
5. **Layer Normalization:** Stabilize training

---

## 🎯 Learning Objectives

- Implement sinusoidal positional encoding
- Build Transformer encoder layers
- Stack multiple encoder layers
- Add classification head for downstream tasks

---

## 📝 Exercises

### Exercise 1: Positional Encoding (30 mins)
Implement sinusoidal position embeddings.

### Exercise 2: Encoder Layer (60 mins)
Build single encoder layer with attention + FFN + residuals.

### Exercise 3: Full Encoder (45 mins)
Stack N encoder layers into complete encoder.

### Exercise 4: Text Classification (60 mins)
Add pooling and classification head.

---

## ✅ Solution

See `solution/01_transformer_encoder.py`

```bash
python solution/01_transformer_encoder.py
```

---

## 🎓 Key Takeaways

- Positional encoding adds position information (sinusoidal or learned)
- Residual connections enable deep networks
- Layer norm stabilizes training
- Encoder outputs contextual representations

---

## 🚀 Next: Lab 3 - Full Transformer!
