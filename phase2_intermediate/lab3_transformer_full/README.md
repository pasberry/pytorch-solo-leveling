# Lab 3: Complete Transformer (Encoder-Decoder) 🎯

> **Time:** 4-5 hours | **Difficulty:** Advanced

---

## 📚 Theory Brief

The full Transformer combines an encoder and decoder for sequence-to-sequence tasks:

**Encoder:** Process source sequence
**Decoder:** Generate target sequence autoregressively

**Decoder adds:**
- Masked self-attention (causal)
- Cross-attention to encoder outputs

---

## 🎯 Learning Objectives

- Implement decoder with masked attention
- Add cross-attention mechanism
- Build complete encoder-decoder
- Implement greedy decoding
- Train for machine translation

---

## 📝 Exercises

1. **Decoder Layer** (90 mins)
2. **Cross-Attention** (60 mins)
3. **Full Model** (60 mins)
4. **Inference** (30 mins)

---

## ✅ Solution

See `solution/01_transformer_complete.py`

```bash
python solution/01_transformer_complete.py
```

---

## 🎓 Key Concepts

- Decoder uses causal masking (can't see future)
- Cross-attention attends to encoder outputs
- Teacher forcing during training
- Greedy/beam search for inference

---

## 🚀 Next: Lab 4 - Custom Loss Functions!
