# Lab 3: Complete Transformer - Encoder-Decoder Architecture 🎯

> **Time:** 5-6 hours
> **Difficulty:** Advanced
> **Goal:** Master the full transformer architecture for sequence-to-sequence tasks

---

## 📖 Why This Lab Matters

The complete Transformer (encoder-decoder) is the architecture behind revolutionary seq2seq models:

- **Original Transformer** - Machine translation breakthrough (2017)
- **T5** (Google) - Text-to-Text Transfer Transformer
- **BART** (Facebook) - Denoising autoencoder for NLP
- **mBART** - Multilingual translation (100+ languages)
- **Whisper** (OpenAI) - Speech-to-text transcription
- **Many modern LLMs** - Use decoder-only variants (GPT)

The encoder-decoder transformer takes one sequence and **generates** another:
- Translation: "Hello" → "Hola"
- Summarization: Long article → Short summary
- Question Answering: Context + Question → Answer

**This lab teaches you the architecture that translates, summarizes, and transforms sequences.**

---

## 🧠 The Big Picture: Sequence-to-Sequence Learning

### The Problem: Transforming Sequences

**Challenge:** Map input sequence to output sequence of different length.

```
Machine Translation:
Input:  "The cat sat on the mat" (6 words)
Output: "Le chat s'est assis sur le tapis" (7 words)

Summarization:
Input:  500-word article
Output: 50-word summary

Question Answering:
Input:  "Paris is the capital of France. Q: What is the capital of France?"
Output: "Paris"
```

**Why it's hard:**
- Variable input/output lengths
- Complex alignments (1-to-many, many-to-1)
- Long-range dependencies
- Must generate fluent, coherent output

### The Solution: Encoder-Decoder Transformer

**Two-stage architecture:**

```
Input Sequence
    ↓
┌─────────────┐
│   ENCODER   │  ← Understand input (bidirectional)
│  (6 layers) │
└─────────────┘
    ↓ Memory
    ↓
┌─────────────┐
│   DECODER   │  ← Generate output (autoregressive)
│  (6 layers) │
└─────────────┘
    ↓
Output Sequence (generated token-by-token)
```

**Key innovation: Decoder has THREE attention mechanisms:**

1. **Masked Self-Attention:** Look at previously generated tokens only
2. **Cross-Attention:** Attend to encoder outputs (source sequence)
3. **Feed-Forward:** Process information

---

## 🔬 Deep Dive: Decoder Architecture

### 1. Masked Self-Attention: Autoregressive Generation

**The Problem:**

During generation, decoder can only see past tokens:

```
Generating: "Le chat s'est assis"

When generating "assis":
✓ Can see: "Le", "chat", "s'est"  ← Past
✗ Cannot see: <future tokens>      ← Future (unknown!)
```

**The Solution: Causal Masking**

```python
# Causal mask prevents looking ahead
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

Example (5 tokens):
[[0, -∞, -∞, -∞, -∞],    # Token 0: sees only itself
 [0,  0, -∞, -∞, -∞],    # Token 1: sees 0-1
 [0,  0,  0, -∞, -∞],    # Token 2: sees 0-2
 [0,  0,  0,  0, -∞],    # Token 3: sees 0-3
 [0,  0,  0,  0,  0]]    # Token 4: sees 0-4
```

**Why needed:**
- **Training:** Prevent cheating (seeing future labels)
- **Inference:** Only past is available during generation
- **Consistency:** Train and inference must match

### 2. Cross-Attention: Encoder-Decoder Connection

**The Bridge:** Decoder attends to encoder outputs.

```python
# Self-attention: Q, K, V from decoder
self_attn = Attention(decoder, decoder, decoder)

# Cross-attention: Q from decoder, K,V from encoder
cross_attn = Attention(decoder, encoder, encoder)
               ↑         ↑        ↑
             Query    Key      Value
             "what?"  "what has" "what gives"
```

**Information flow:**

```
Encoder outputs: ["The", "cat", "sat", "on", "the", "mat"]
                    ↓      ↓     ↓     ↓     ↓      ↓
                    └──────┴─────┴─────┴─────┴──────┘
                              Cross-Attention
                                    ↑
Decoder: Generating "chat" ─────────┘
         Attends most to "cat" (alignment!)
```

**Why it works:**
- Decoder can focus on relevant source words
- Learns soft alignment (which source → which target)
- Handles different input/output lengths
- Interpretable (visualize which source words matter)

### 3. Complete Decoder Layer

```python
class TransformerDecoderLayer(nn.Module):
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention (look at past output)
        self_attn = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))

        # 2. Cross-attention (attend to encoder)
        cross_attn = self.cross_attention(x, encoder_output,
                                          encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x
```

**Three sub-layers per decoder layer!**

---

## 🎯 Training vs Inference: Critical Difference

### Teacher Forcing (Training)

**During training:** Give decoder the correct previous tokens.

```python
# Training: Parallel processing
Input:  "Le chat s'est assis"
Target: "Le chat s'est assis <EOS>"
                ↓
Decoder sees: "<BOS> Le chat s'est assis"  ← Correct previous tokens!
Predicts:     "Le chat s'est assis <EOS>"
```

**Benefits:**
- Fast (parallel across sequence)
- Stable gradients
- Efficient training

**Exposure bias problem:**
- Training: Always sees correct previous tokens
- Inference: May see own mistakes
- Distribution mismatch!

### Autoregressive Generation (Inference)

**During inference:** Feed decoder its own predictions.

```python
# Inference: Sequential generation
Step 1: Input: <BOS>             → Predict: "Le"
Step 2: Input: <BOS> Le          → Predict: "chat"
Step 3: Input: <BOS> Le chat     → Predict: "s'est"
...
Step N: Input: ... assis         → Predict: <EOS> (stop!)
```

**Sequential process:**
```
for step in range(max_length):
    # Generate one token
    output = decoder(input_so_far, encoder_memory)
    next_token = output.argmax(dim=-1)

    # Append to input
    input_so_far = torch.cat([input_so_far, next_token])

    # Stop if <EOS>
    if next_token == EOS_TOKEN:
        break
```

**Challenges:**
- Slow (sequential, can't parallelize)
- Error accumulation (one mistake → cascading errors)
- Search over exponential space

---

## 🔍 Decoding Strategies

### 1. Greedy Decoding

**Pick most likely token at each step.**

```python
def greedy_decode(decoder, encoder_output, max_len):
    output = [BOS_TOKEN]

    for _ in range(max_len):
        # Predict next token
        logits = decoder(output, encoder_output)
        next_token = logits[-1].argmax()  # Greedy!

        output.append(next_token)
        if next_token == EOS_TOKEN:
            break

    return output
```

**Pros:**
- Fast (O(n) where n = output length)
- Deterministic
- Simple

**Cons:**
- Myopic (locally optimal ≠ globally optimal)
- Can't recover from mistakes
- Often produces suboptimal sequences

**Example failure:**
```
Greedy:     "The cat sat" → "Le chat assis" (missing "s'est")
Better:     "The cat sat" → "Le chat s'est assis"

Why? First token choice constrains future.
```

### 2. Beam Search

**Maintain top-k hypotheses, expand best ones.**

```python
def beam_search(decoder, encoder_output, max_len, beam_width=5):
    # Initialize beams
    beams = [(score=0.0, tokens=[BOS_TOKEN])]

    for step in range(max_len):
        candidates = []

        for score, tokens in beams:
            if tokens[-1] == EOS_TOKEN:
                candidates.append((score, tokens))  # Keep finished
                continue

            # Expand with all vocab
            logits = decoder(tokens, encoder_output)
            probs = softmax(logits[-1])

            # Top-k extensions
            for token, prob in topk(probs, k=beam_width):
                new_score = score + log(prob)
                new_tokens = tokens + [token]
                candidates.append((new_score, new_tokens))

        # Keep top beam_width
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    return beams[0][1]  # Return best hypothesis
```

**Complexity:**
```
Greedy: O(n)               where n = sequence length
Beam:   O(n × k)           where k = beam width
        O(n × k × V)       with vocabulary size V
```

**Pros:**
- Better quality (explores multiple paths)
- Recovers from local mistakes
- Tunable (beam width)

**Cons:**
- Slower (k times greedy)
- More memory (store k hypotheses)
- Can still miss best sequence

**Beam width trade-off:**
```
k=1:  Greedy (fast, lower quality)
k=5:  Good balance (Google Translate uses this)
k=50: Expensive, diminishing returns
```

### 3. Sampling-Based Methods

**Temperature sampling:**
```python
# Control randomness
probs = softmax(logits / temperature)
next_token = sample(probs)

temperature = 0.1:  Near-deterministic (peaked distribution)
temperature = 1.0:  Standard (original distribution)
temperature = 2.0:  More random (flatter distribution)
```

**Top-k sampling:**
```python
# Sample from top-k tokens only
topk_probs, topk_indices = torch.topk(probs, k=40)
topk_probs = topk_probs / topk_probs.sum()
next_token = sample(topk_probs, topk_indices)
```

**Top-p (nucleus) sampling:**
```python
# Sample from smallest set with cumulative prob ≥ p
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
nucleus = cumsum <= p  # e.g., p=0.9
next_token = sample(sorted_probs[nucleus], sorted_indices[nucleus])
```

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Complete encoder-decoder architecture
- Why decoder needs masked self-attention
- How cross-attention connects encoder and decoder
- Teacher forcing vs autoregressive generation
- Greedy vs beam search trade-offs
- Exposure bias and mitigation strategies
- Original "Attention is All You Need" paper
- Applications to translation, summarization, seq2seq

**Practical Skills:**
- Implement decoder layer with three attention types
- Build complete transformer (encoder + decoder)
- Implement teacher forcing training
- Create greedy decoding algorithm
- Build beam search decoder
- Train seq2seq model (translation or summarization)
- Visualize cross-attention alignments

---

## 🔑 Key Concepts

### 1. Autoregressive Generation

**Definition:** Generate one token at a time, conditioning on all previous tokens.

```
P(y₁, y₂, ..., yₙ | x) = ∏ P(yᵢ | y₁, ..., yᵢ₋₁, x)
                         i=1

Chain rule factorization → Sequential generation
```

**Implication:**
- Can't parallelize generation (each token depends on previous)
- Inference is O(n) sequential steps
- Training can be parallel (teacher forcing)

### 2. Exposure Bias

**Problem:** Train on gold labels, test on model predictions.

```
Training:  decoder sees [gold, gold, gold, ...]
Inference: decoder sees [pred, pred, pred, ...]

If pred ≠ gold → distribution shift → cascading errors!
```

**Mitigation strategies:**
1. **Scheduled sampling:** Mix gold and predicted tokens during training
2. **Reinforcement learning:** Optimize for sequence-level reward
3. **Larger beam width:** Explore more hypotheses

### 3. Transformer Complexity

**Full transformer (encoder-decoder):**

```
Encoder:
- Self-attention: O(n² × d) per layer
- Feed-forward: O(n × d²) per layer
- N layers: O(N × (n² × d + n × d²))

Decoder:
- Masked self-attention: O(m² × d) per layer
- Cross-attention: O(m × n × d) per layer
- Feed-forward: O(m × d²) per layer
- N layers: O(N × (m² × d + m × n × d + m × d²))

Where:
- n = source sequence length
- m = target sequence length
- d = model dimension
- N = number of layers
```

**Memory:**
- Encoder: O(n² × N) for attention matrices
- Decoder: O(m² × N + m × n × N)

---

## 🧪 Exercises

### Exercise 1: Decoder Layer with Masked Attention (60 mins)

**What You'll Learn:**
- Implementing causal masking
- Three attention mechanisms in decoder
- Residual connections and layer norms
- Difference from encoder layer

**Why It Matters:**
The decoder is what makes transformers generative. Understanding masked self-attention is crucial for GPT, LLaMA, and all autoregressive models.

**Tasks:**
1. Implement `TransformerDecoderLayer` class
2. Add masked self-attention (causal mask)
3. Add cross-attention to encoder outputs
4. Add feed-forward network
5. Test with sample encoder outputs
6. Verify masking prevents future access

**Architecture:**
```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        x = x + self.dropout(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x)

        # Cross-attention
        x = x + self.dropout(self.cross_attn(x, encoder_output,
                                             encoder_output, src_mask))
        x = self.norm2(x)

        # Feed-forward
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)

        return x
```

---

### Exercise 2: Complete Transformer Model (90 mins)

**What You'll Learn:**
- Combining encoder and decoder
- Embedding layers and positional encoding
- Output projection layer
- Parameter initialization
- Forward pass coordination

**Why It Matters:**
This is the complete architecture from "Attention is All You Need" - understanding it deeply means you can implement any transformer variant.

**Tasks:**
1. Implement `Transformer` class
2. Add embeddings for source and target
3. Stack encoder layers (6 layers)
4. Stack decoder layers (6 layers)
5. Add final linear projection to vocabulary
6. Test end-to-end forward pass
7. Count parameters

**Complete architecture:**
```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1):
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        src = self.pos_encoding(self.src_embedding(src))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decode target
        tgt = self.pos_encoding(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        # Project to vocabulary
        return self.fc_out(tgt)
```

---

### Exercise 3: Training with Teacher Forcing (60 mins)

**What You'll Learn:**
- Teacher forcing implementation
- Creating training masks
- Loss computation for sequences
- Ignoring padding in loss
- Training loop for seq2seq

**Why It Matters:**
Teacher forcing is how all seq2seq models are trained. Understanding it reveals why training and inference differ.

**Tasks:**
1. Create training loop with teacher forcing
2. Generate causal mask for target
3. Create padding masks for source and target
4. Implement masked cross-entropy loss
5. Train on toy translation task
6. Monitor training/validation loss

**Training loop:**
```python
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        # Create masks
        src_mask = (src != PAD_TOKEN)
        tgt_mask = create_causal_mask(tgt.size(1))

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss (ignore padding)
        loss = criterion(logits.reshape(-1, vocab_size),
                        tgt_output.reshape(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

---

### Exercise 4: Greedy Decoding (45 mins)

**What You'll Learn:**
- Autoregressive generation
- Incremental decoding
- Stopping conditions
- Handling variable-length outputs

**Why It Matters:**
Greedy decoding is the simplest inference strategy. Understanding it is foundational for all generation methods.

**Tasks:**
1. Implement `greedy_decode()` function
2. Handle <BOS> and <EOS> tokens
3. Limit maximum length
4. Test on trained model
5. Compare generated vs ground truth
6. Visualize cross-attention

**Implementation:**
```python
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len=50):
    model.eval()

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Start with <BOS>
    tgt = torch.tensor([[BOS_TOKEN]]).to(src.device)

    for _ in range(max_len):
        # Create causal mask
        tgt_mask = create_causal_mask(tgt.size(1))

        # Decode
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Get next token (greedy)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append to sequence
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop if <EOS>
        if next_token.item() == EOS_TOKEN:
            break

    return tgt
```

---

### Exercise 5: Beam Search Decoding (90 mins)

**What You'll Learn:**
- Beam search algorithm
- Hypothesis scoring and ranking
- Handling finished vs active hypotheses
- Length normalization
- Complexity analysis

**Why It Matters:**
Beam search is used in production translation systems (Google Translate). Understanding it bridges research and real-world deployment.

**Tasks:**
1. Implement `beam_search()` function
2. Maintain top-k hypotheses
3. Handle finished sequences
4. Add length normalization
5. Compare to greedy (quality vs speed)
6. Tune beam width

**Advanced implementation:**
```python
@torch.no_grad()
def beam_search(model, src, src_mask, beam_width=5, max_len=50):
    model.eval()

    # Encode source once
    encoder_output = model.encode(src, src_mask)

    # Initialize beams: (score, tokens, finished)
    beams = [(0.0, [BOS_TOKEN], False)]

    for step in range(max_len):
        candidates = []

        for score, tokens, finished in beams:
            if finished:
                candidates.append((score, tokens, True))
                continue

            # Decode current sequence
            tgt = torch.tensor([tokens]).to(src.device)
            tgt_mask = create_causal_mask(len(tokens))
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

            # Get probabilities for next token
            logits = model.fc_out(output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand with top-k tokens
            topk_probs, topk_tokens = torch.topk(log_probs[0], beam_width)

            for prob, token in zip(topk_probs, topk_tokens):
                new_score = score + prob.item()
                new_tokens = tokens + [token.item()]
                new_finished = (token.item() == EOS_TOKEN)
                candidates.append((new_score, new_tokens, new_finished))

        # Keep top beam_width by score
        beams = sorted(candidates, key=lambda x: x[0] / len(x[1]),  # Length normalize
                      reverse=True)[:beam_width]

        # Stop if all beams finished
        if all(finished for _, _, finished in beams):
            break

    return beams[0][1]  # Best hypothesis
```

---

### Exercise 6: Translation Task (120 mins)

**What You'll Learn:**
- End-to-end seq2seq training
- Data preprocessing for translation
- Tokenization and vocabulary building
- BLEU score evaluation
- Hyperparameter tuning

**Why It Matters:**
This is a complete real-world application! You'll build a working translation system from scratch.

**Tasks:**
1. Load translation dataset (Multi30k or IWSLT)
2. Build vocabularies for source and target
3. Create data loaders with padding
4. Train transformer model
5. Implement BLEU score evaluation
6. Generate and evaluate translations
7. Visualize attention alignments

**Complete pipeline:**
```python
# 1. Data preprocessing
def build_vocab(sentences, max_vocab=10000):
    counter = Counter()
    for sent in sentences:
        counter.update(sent.split())
    vocab = {token: idx for idx, (token, _) in
             enumerate(counter.most_common(max_vocab), start=4)}
    vocab.update({PAD_TOKEN: 0, UNK_TOKEN: 1,
                 BOS_TOKEN: 2, EOS_TOKEN: 3})
    return vocab

# 2. Training
model = Transformer(len(src_vocab), len(tgt_vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

for epoch in range(20):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    print(f'Epoch {epoch}: Train={train_loss:.3f}, Val={val_loss:.3f}')

# 3. Evaluation
bleu_score = evaluate_bleu(model, test_loader, beam_width=5)
print(f'BLEU Score: {bleu_score:.2f}')
```

---

## 📝 Design Patterns

### Pattern 1: Efficient Mask Generation

```python
def create_masks(src, tgt, pad_token=0):
    """Create all masks for transformer."""
    # Source padding mask: (batch, 1, 1, src_len)
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)

    # Target padding mask: (batch, 1, tgt_len)
    tgt_padding_mask = (tgt != pad_token).unsqueeze(1)

    # Target causal mask: (tgt_len, tgt_len)
    tgt_len = tgt.size(1)
    tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()

    # Combine target masks: (batch, 1, tgt_len, tgt_len)
    tgt_mask = tgt_padding_mask.unsqueeze(2) & tgt_causal_mask

    return src_mask, tgt_mask
```

### Pattern 2: Teacher Forcing Training

```python
def train_step(model, src, tgt, criterion, optimizer):
    """Single training step with teacher forcing."""
    # Create masks
    src_mask, tgt_mask = create_masks(src, tgt)

    # Shift target: input is all but last, output is all but first
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    # Forward pass
    logits = model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])

    # Compute loss
    loss = criterion(logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
```

### Pattern 3: Beam Search with Length Normalization

```python
class BeamSearchDecoder:
    def __init__(self, model, beam_width=5, max_len=50, length_penalty=0.6):
        self.model = model
        self.beam_width = beam_width
        self.max_len = max_len
        self.alpha = length_penalty  # Length normalization factor

    def normalize_scores(self, score, length):
        """Length normalization from Google's NMT paper."""
        return score / (length ** self.alpha)

    def decode(self, src, src_mask):
        # ... beam search implementation with length normalization ...
        pass
```

---

## ✅ Solutions

Complete implementations in `solution/` directory.

**Files:**
- `01_decoder_layer.py` - Decoder with three attention types
- `02_transformer.py` - Complete encoder-decoder model
- `03_training.py` - Teacher forcing training loop
- `04_greedy_decode.py` - Greedy decoding
- `05_beam_search.py` - Beam search implementation
- `06_translation.py` - End-to-end translation system

Run examples:
```bash
cd solution
python 01_decoder_layer.py
python 02_transformer.py
python 03_training.py
python 04_greedy_decode.py
python 05_beam_search.py
python 06_translation.py
```

---

## 🎓 Key Takeaways

1. **Encoder understands, decoder generates** - Two-stage architecture
2. **Masked self-attention enables autoregressive** - No future peeking
3. **Cross-attention connects sequences** - Soft alignment between source and target
4. **Teacher forcing trains, autoregression infers** - Distribution mismatch is inherent
5. **Greedy is fast, beam search is better** - Quality vs speed trade-off
6. **Exposure bias is fundamental** - Training sees gold, inference sees predictions
7. **Original transformer is encoder-decoder** - GPT removed encoder, BERT removed decoder
8. **Seq2seq powers translation, summarization** - Any sequence transformation task

**The Architecture:**
```
Source → Encoder (bidirectional understanding)
           ↓ memory
       Decoder (autoregressive generation)
           ↓ cross-attention
       Target (token-by-token)
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**At Google (Google Translate):**
- Transformer-based since 2017
- 100+ language pairs
- Billions of translations daily
- Beam search with beam width 4-8
- Real-time translation at scale

**At Meta (mBART):**
- Multilingual denoising pretraining
- 25 languages simultaneously
- Translation, summarization, generation
- Shared encoder-decoder
- Fine-tuned for 50+ language pairs

**At OpenAI (Whisper):**
- Encoder-decoder for speech transcription
- Encoder processes audio spectrogram
- Decoder generates text
- 99 languages supported
- Handles code-switching

**At Google (T5):**
- Text-to-Text Transfer Transformer
- All NLP tasks as seq2seq
- Translation, QA, summarization, classification
- 11B parameter model
- Powers many Google products

**Cost and Scale:**
- Original transformer: 4 days on 8 GPUs
- T5-11B: Weeks on hundreds of TPUs
- Inference: Milliseconds per translation
- Beam search adds 3-5x latency

---

## 🚀 Next Steps

1. **Complete all exercises** - Build transformer from scratch
2. **Train on real data** - Translation or summarization
3. **Read "Attention is All You Need"** - The foundational paper
4. **Move to Lab 4** - Custom Loss Functions

---

## 💪 Bonus Challenges

1. **Scheduled Sampling**
   - Mix teacher forcing with model predictions during training
   - Reduce exposure bias
   - Measure impact on generation quality

2. **Length Reward in Beam Search**
   - Add length penalty/reward
   - Prevent overly short/long sequences
   - Tune with validation set

3. **Diverse Beam Search**
   - Group diverse hypotheses
   - Prevent duplicate translations
   - Improve output diversity

4. **Multilingual Translation**
   - Single model for N language pairs
   - Language tokens to specify target
   - Compare to bilingual models

5. **Attention Visualization**
   - Extract cross-attention weights
   - Create alignment heatmaps
   - Analyze translation decisions

---

## 📚 Essential Resources

**The Foundational Paper:**
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
  - Original transformer paper
  - Complete architecture and training details
  - Benchmark results on translation

**Follow-up Papers:**
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018 (encoder-only)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al., 2019 (GPT-2, decoder-only)
- [Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683) - Raffel et al., 2019 (T5)
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) - Lewis et al., 2019

**Decoding Methods:**
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Holtzman et al., 2019 (nucleus sampling)
- [Beam Search Strategies for Neural Machine Translation](https://arxiv.org/abs/1702.01806) - Freitag & Al-Onaizan, 2017

**Tutorials:**
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Seq2Seq Tutorial](https://huggingface.co/docs/transformers/tasks/translation)

---

## 🤔 Common Pitfalls

### Pitfall 1: Wrong Target Shifting

```python
# ❌ No shifting (decoder sees future!)
tgt_input = tgt
tgt_output = tgt

# ✓ Shift by one position
tgt_input = tgt[:, :-1]   # All but last
tgt_output = tgt[:, 1:]   # All but first
```

**Why:** Decoder must predict next token, not current token!

### Pitfall 2: Forgetting Causal Mask

```python
# ❌ Decoder sees future (cheating during training!)
output = decoder(tgt, encoder_output)

# ✓ Causal mask prevents future access
tgt_mask = create_causal_mask(tgt.size(1))
output = decoder(tgt, encoder_output, tgt_mask=tgt_mask)
```

**Why:** Training must match inference (autoregressive).

### Pitfall 3: Including Padding in Loss

```python
# ❌ Loss penalizes padding predictions
loss = criterion(logits, targets)

# ✓ Ignore padding tokens
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
loss = criterion(logits, targets)
```

**Why:** Padding tokens are meaningless, shouldn't affect training.

### Pitfall 4: Inefficient Beam Search

```python
# ❌ Re-encode source for each beam (wasteful!)
for beam in beams:
    encoder_output = model.encode(src)  # Redundant!

# ✓ Encode once, reuse for all beams
encoder_output = model.encode(src)  # Once!
for beam in beams:
    use encoder_output  # Reuse
```

**Why:** Source doesn't change across beams!

---

## 💡 Pro Tips

1. **Label smoothing helps** - Prevents overconfidence
2. **Warmup learning rate** - Start small, increase, then decay
3. **Gradient clipping essential** - Prevents explosion
4. **Share embeddings** - Source, target, output layer
5. **Use pre-norm** - More stable for deep models
6. **Beam width 4-8 is sweet spot** - Diminishing returns beyond
7. **Length normalization crucial** - Prevents short sequences
8. **Monitor attention entropy** - Detect issues early

---

## ✨ You're Ready When...

- [ ] You understand encoder-decoder architecture
- [ ] You can explain why decoder needs masked self-attention
- [ ] You've implemented cross-attention mechanism
- [ ] You understand teacher forcing vs autoregressive generation
- [ ] You can implement greedy decoding from scratch
- [ ] You've built beam search decoder
- [ ] You understand exposure bias problem
- [ ] You've trained a translation model end-to-end
- [ ] You can visualize cross-attention alignments
- [ ] You know trade-offs between decoding strategies

**Remember:** The transformer encoder-decoder is the architecture that enabled modern NLP. Master it, and you understand the foundation of translation, summarization, and sequence transformation!
