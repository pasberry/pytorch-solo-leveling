# Project 4: Speech Recognition (ASR) 🎤

> **Time:** 1 week
> **Difficulty:** Advanced
> **Goal:** Build an end-to-end automatic speech recognition system

---

## Why This Project Matters

Voice is the most natural form of human communication. Speech recognition (ASR) enables machines to understand spoken language, powering a revolution in human-computer interaction:

**Market & Impact**:
- **Voice assistants**: 4.2 billion devices (Siri, Alexa, Google Assistant)
- **Accessibility**: 466 million people worldwide with hearing loss benefit from live captions
- **Productivity**: Speech-to-text is 3x faster than typing (150 vs 40 WPM)
- **Market size**: $27B in 2023, projected $84B by 2032

**Real-world applications**:
- **Virtual assistants**: Siri, Alexa, Google Assistant (billions of daily queries)
- **Transcription**: Meetings, interviews, podcasts (Otter.ai, Rev.com)
- **Accessibility**: Live captions for deaf/hard-of-hearing (YouTube, Zoom)
- **Customer service**: Call center automation (saves billions annually)
- **Healthcare**: Medical transcription, voice commands for surgeons
- **Automotive**: Hands-free controls for safety

**Technical challenges**:
- **Variability**: Different accents, speaking speeds, background noise
- **Ambiguity**: "recognize speech" vs "wreck a nice beach"
- **Latency**: Real-time streaming requires <500ms delay
- **Scale**: Millions of hours of audio, dozens of languages
- **Accuracy**: 95%+ accuracy needed for production (humans: 96%)

---

## The Big Picture

### The Problem

**Input**: Audio waveform (sequence of amplitudes over time)
**Output**: Text transcription (sequence of characters/words)

**Why is ASR hard?**

1. **Temporal variability**: Same word spoken at different speeds
   ```
   "Hello" at normal speed: 0.5 seconds
   "Hellooo" drawn out: 1.5 seconds
   "Hello!" fast: 0.3 seconds
   ```

2. **Acoustic variability**: Different speakers, accents, emotions
   ```
   "Hello" from:
   - Male voice (deeper pitch)
   - Female voice (higher pitch)
   - British accent vs American accent
   - Whispered vs shouted
   ```

3. **Alignment problem**: No frame-by-frame labels
   ```
   Audio: [0.0s ← "hhhh" →] [0.2s ← "eeee" →] [0.3s ← "llll" →] [0.5s ← "oooo" →]
   Text:  "hello"

   Question: Which audio frames correspond to which letters?
   We don't know! (this is the alignment problem)
   ```

4. **Background noise**: Music, other speakers, traffic, etc.

### The Solution: End-to-End ASR

**Evolution of ASR**:
```
1970s-2000s: Traditional ASR (HMM-GMM)
  - Separate acoustic model, pronunciation model, language model
  - Complex pipeline, hard to optimize

2010s: Deep Learning (Hybrid DNN-HMM)
  - Neural networks for acoustic modeling
  - Still uses HMM for alignment

2015+: End-to-End ASR
  - Single neural network: Audio → Text
  - CTC, Attention, Transducer architectures
  - Simpler, better performance

2020+: Self-Supervised Pretraining
  - Wav2Vec 2.0, HuBERT, Whisper
  - Train on unlabeled audio, finetune on transcriptions
  - State-of-the-art results
```

**This project**: Build CTC-based end-to-end ASR system.

---

## Deep Dive: Speech Recognition Theory

### 1. Audio Fundamentals

#### Waveform Representation

**Sound = Vibration** → **Air pressure changes** → **Electrical signal** → **Digital samples**

```python
# Audio waveform
waveform = [0.1, 0.3, 0.5, 0.4, 0.2, -0.1, -0.3, ...]  # Amplitude values

# Sampling rate: Samples per second
sample_rate = 16000  # 16 kHz (16,000 samples/second)

# Duration
duration = len(waveform) / sample_rate  # seconds
```

**Nyquist theorem**: To capture frequency f, sample at ≥2f
- Human hearing: 20 Hz - 20 kHz
- Speech: 80 Hz - 8 kHz (most energy)
- **Common sample rates**:
  - 8 kHz: Telephony (low quality)
  - 16 kHz: Speech recognition (good for ASR)
  - 44.1 kHz: CD quality
  - 48 kHz: Professional audio

#### Spectrograms: Time-Frequency Representation

**Problem**: Waveform only shows amplitude over time, not frequency content.

**Solution**: Spectrogram - visualize frequency content over time.

**Short-Time Fourier Transform (STFT)**:
```
1. Split audio into short windows (e.g., 25ms)
2. Apply Fast Fourier Transform (FFT) to each window
3. Get frequency spectrum for each time step
4. Stack spectra → Spectrogram
```

**Mathematics**:
```
STFT(t, f) = Σ window(t) × waveform(t) × e^(-2πift)

Result: Complex matrix (time × frequency)
Magnitude: |STFT| → Spectrogram
```

**Implementation**:
```python
import torch
import torchaudio

# Load audio
waveform, sample_rate = torchaudio.load('audio.wav')

# Spectrogram transform
spectrogram = torchaudio.transforms.Spectrogram(
    n_fft=400,        # FFT size (25ms at 16kHz)
    hop_length=160,   # Hop size (10ms stride)
    power=2.0         # Power spectrogram
)(waveform)

# Shape: (freq_bins, time_steps)
# freq_bins = n_fft // 2 + 1 = 201
```

**Parameters**:
- **n_fft**: Window size (25ms typical for speech)
- **hop_length**: Stride between windows (10ms typical)
- **window**: Hann window (smooth edges)

#### Mel-Spectrogram: Perceptually Motivated

**Human hearing**: Logarithmic frequency perception
- Can distinguish 100 Hz vs 200 Hz easily
- Hard to distinguish 10,000 Hz vs 10,100 Hz

**Mel scale**: Approximates human perception
```
mel(f) = 2595 × log₁₀(1 + f/700)

Example:
100 Hz → 150 mel
200 Hz → 300 mel (2x frequency, 2x mel)
10,000 Hz → 3,900 mel
10,100 Hz → 3,910 mel (small change)
```

**Mel-spectrogram**: Apply mel filterbank to spectrogram
```python
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80          # 80 mel frequency bins
)(waveform)

# Shape: (80, time_steps)
```

**Why mel-spectrogram for ASR?**
- Captures perceptually relevant information
- Lower dimensionality (80 vs 201 frequency bins)
- Better generalization across speakers

**Log-mel spectrogram**: Take log for dynamic range
```python
log_mel = torch.log(mel_spectrogram + 1e-9)  # Add epsilon to avoid log(0)
```

**Normalization** (critical for training):
```python
# Mean-variance normalization
mean = log_mel.mean()
std = log_mel.std()
normalized = (log_mel - mean) / std
```

---

### 2. CTC Loss: Connectionist Temporal Classification

**The Alignment Problem**: We have audio frames and text, but don't know which frames correspond to which characters.

**Example**:
```
Audio frames (100 frames at 10ms each = 1 second):
[f1, f2, f3, ..., f100]

Text: "HELLO" (5 characters)

Question: Which frames map to which characters?
- Frames 1-20 → "H"?
- Frames 21-40 → "E"?
- Unknown!
```

**CTC Solution**: Don't need exact alignment!

#### CTC Alignment Rules

**Key ideas**:
1. Introduce **blank token** (ε)
2. Allow **repetitions** of characters
3. **Collapse** repetitions and remove blanks to get final text

**Example alignments** for "HELLO":
```
Valid alignments (among many):
εεHHεEELLLLOOOεεε
HHεεELLεεLεεOOOε
εεεHELLOεεεεεεεε

All collapse to: HELLO
```

**Collapsing rules**:
```
1. Consecutive repeated characters → Single character
   HHEELLLLOO → HELLO

2. Remove blanks (ε)
   εHεEεLεLεOε → HELLO

3. Blank separates repeated characters
   HεH → HH (two H's)
   HH → H (one H)
```

#### CTC Probability

**Model output**: Probability distribution over characters at each time step
```
Time step t=1: P(a|t=1) = 0.1, P(b|t=1) = 0.05, ..., P(ε|t=1) = 0.3
Time step t=2: P(a|t=2) = 0.2, P(b|t=2) = 0.01, ..., P(ε|t=2) = 0.1
...
```

**Alignment probability**: Product of character probabilities
```
P(εHεELLOε) = P(ε|t=1) × P(H|t=2) × P(ε|t=3) × P(E|t=4) × ...
```

**Label probability**: Sum over all valid alignments
```
P("HELLO") = Σ_{all alignments that collapse to "HELLO"} P(alignment)
```

**Challenge**: Exponentially many alignments! (billions for long sequences)

**Solution**: Dynamic programming (forward-backward algorithm)
```
α[t, s] = Probability of all alignments up to time t ending at label position s
β[t, s] = Probability of all alignments from time t to end starting at position s

P(label) = α[T, S] = sum over all final positions
```

#### CTC Loss

**Maximize** probability of correct label:
```
L_CTC = -log P(y | x)
      = -log Σ_{alignments → y} ∏_t P(alignment[t] | x, t)
```

**PyTorch implementation**:
```python
import torch
import torch.nn as nn

criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# Model outputs: (T, B, C) where T=time, B=batch, C=num_characters
log_probs = model(audio)  # (100, 32, 29) - 100 frames, 32 batch, 29 chars

# Targets: Concatenated labels (no padding)
targets = torch.tensor([2, 3, 5, 5, 10])  # "HELLO" encoded

# Input lengths: How many frames per example
input_lengths = torch.tensor([100, 95, 100, ...])  # Variable lengths

# Target lengths: How many characters per label
target_lengths = torch.tensor([5, 7, 4, ...])

loss = criterion(log_probs, targets, input_lengths, target_lengths)
```

**Key hyperparameters**:
- **blank=0**: Index of blank token (usually 0)
- **zero_infinity**: Set infinite losses to 0 (numerical stability)

---

### 3. Model Architectures for ASR

#### Architecture 1: CNN-BiLSTM-CTC

**Standard architecture** for CTC-based ASR:

```
Audio (waveform)
    ↓
Log-Mel Spectrogram (80 × T)
    ↓
CNN Layers (extract acoustic features)
    ├─ Conv2D(1 → 32, kernel=3×3)
    ├─ Conv2D(32 → 64, kernel=3×3)
    └─ Pool (reduce frequency dimension)
    ↓
Reshape for RNN (T × feature_dim)
    ↓
Bidirectional LSTM (model temporal dependencies)
    ├─ BiLSTM layer 1 (512 hidden units)
    ├─ BiLSTM layer 2 (512 hidden units)
    └─ BiLSTM layer 3 (512 hidden units)
    ↓
Linear Projection (hidden → num_chars)
    ↓
Log-Softmax (T × num_chars)
    ↓
CTC Loss / CTC Decoding
```

**Why CNN?**
- Local feature extraction
- Invariance to small shifts
- Reduce dimensionality

**Why BiLSTM?**
- **Temporal modeling**: Past context (left-to-right) + Future context (right-to-left)
- **Long-range dependencies**: Remember information across seconds of audio
- **Sequential processing**: Natural for sequential data like speech

**Implementation**:
```python
class CNNLSTM_CTC(nn.Module):
    def __init__(self, n_mels=80, n_class=29, hidden_dim=512, n_lstm_layers=3):
        super().__init__()

        # === CNN LAYERS ===
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce frequency dimension

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # After pooling: n_mels // 4
        cnn_output_dim = 128 * (n_mels // 4)

        # === BiLSTM LAYERS ===
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            bidirectional=True,
            batch_first=False,  # CTC expects (T, B, C)
            dropout=0.2
        )

        # === OUTPUT PROJECTION ===
        self.fc = nn.Linear(hidden_dim * 2, n_class)  # *2 for bidirectional

    def forward(self, x):
        # x: (B, 1, n_mels, T)

        # CNN
        x = self.cnn(x)  # (B, 128, n_mels//4, T)

        # Reshape for LSTM: (T, B, feature_dim)
        B, C, H, T = x.shape
        x = x.permute(3, 0, 1, 2)  # (T, B, C, H)
        x = x.reshape(T, B, C * H)  # (T, B, C*H)

        # BiLSTM
        x, _ = self.lstm(x)  # (T, B, hidden_dim*2)

        # Output projection
        x = self.fc(x)  # (T, B, n_class)

        # Log-softmax for CTC
        x = F.log_softmax(x, dim=2)

        return x  # (T, B, n_class)
```

#### Architecture 2: Transformer-CTC

**Modern approach**: Replace LSTM with Transformer

```
Audio
    ↓
Log-Mel Spectrogram
    ↓
CNN (feature extraction)
    ↓
Positional Encoding
    ↓
Transformer Encoder Layers
    ├─ Multi-head self-attention
    ├─ Feed-forward network
    └─ Layer norm + residual
    ↓
Linear Projection
    ↓
CTC Loss
```

**Advantages**:
- **Parallelizable**: Unlike LSTM, can train faster
- **Long-range dependencies**: Attention captures global context
- **Better performance**: State-of-the-art results

**Disadvantages**:
- **More data hungry**: Requires large datasets
- **Computational cost**: O(T²) attention complexity

---

### 4. CTC Decoding: From Probabilities to Text

**Problem**: Model outputs probabilities at each time step. How do we get the final transcription?

#### Greedy Decoding (Simple, Fast)

**Algorithm**:
```
1. For each time step, take character with highest probability
2. Collapse repetitions and remove blanks
```

**Example**:
```
Time steps: [H, H, ε, E, L, L, L, ε, O, O]
            ↓
Collapse:   [H, ε, E, L, ε, O]
            ↓
Remove ε:   [H, E, L, O]
            ↓
Result:     "HELO" ❌ (should be "HELLO")
```

**Implementation**:
```python
def greedy_decode(log_probs, blank=0):
    """
    Greedy CTC decoding.

    Args:
        log_probs: (T, num_chars) - log probabilities
        blank: Index of blank token

    Returns:
        decoded: List of character indices
    """
    # Take argmax at each time step
    indices = log_probs.argmax(dim=1)  # (T,)

    # Remove blanks and consecutive duplicates
    decoded = []
    prev_char = blank

    for char_idx in indices:
        char_idx = char_idx.item()

        if char_idx != blank and char_idx != prev_char:
            decoded.append(char_idx)

        prev_char = char_idx

    return decoded
```

**Pros**: Fast, simple
**Cons**: Suboptimal (can make mistakes)

#### Beam Search Decoding (Better, Slower)

**Idea**: Instead of taking top-1 at each step, maintain top-K hypotheses.

**Algorithm**:
```
1. Start with empty hypothesis
2. At each time step:
   - Expand each hypothesis with all possible characters
   - Keep top-K by total probability
3. Return best hypothesis at end
```

**Example** (beam size K=2):
```
Time 0:
Hypotheses: [""]

Time 1: (top chars: H=0.8, E=0.1, ...)
Expand:    ["H"] (0.8), ["E"] (0.1)
Keep top-2: ["H"], ["E"]

Time 2: (top chars: E=0.7, H=0.2, ...)
Expand from "H": ["HE"] (0.8×0.7), ["HH"] (0.8×0.2)
Expand from "E": ["EE"] (0.1×0.7), ["EH"] (0.1×0.2)
Keep top-2: ["HE"] (0.56), ["HH"] (0.16)

... continue for all time steps ...

Final: "HELLO" (highest probability path)
```

**Implementation**:
```python
def beam_search_decode(log_probs, beam_size=10, blank=0):
    """
    Beam search CTC decoding.

    Args:
        log_probs: (T, num_chars)
        beam_size: Number of hypotheses to keep

    Returns:
        best_path: Decoded sequence
    """
    T, num_chars = log_probs.shape

    # Initialize beam with empty hypothesis
    # beam = [(score, path)]
    beam = [(0.0, [])]

    for t in range(T):
        new_beam = []

        for score, path in beam:
            # Expand with all possible characters
            for char_idx in range(num_chars):
                new_score = score + log_probs[t, char_idx].item()
                new_path = path + [char_idx]
                new_beam.append((new_score, new_path))

        # Keep top-K
        new_beam = sorted(new_beam, key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_size]

    # Best hypothesis
    best_score, best_path = beam[0]

    # Collapse CTC path
    decoded = collapse_ctc_path(best_path, blank)

    return decoded
```

**Pros**: Better accuracy (explores multiple paths)
**Cons**: Slower (K times more work)

**Typical beam sizes**: 10-100 (trade-off accuracy vs speed)

#### Language Model Integration

**Problem**: CTC decoder has no language knowledge
- Might output "wreck a nice beach" instead of "recognize speech"

**Solution**: Combine acoustic model (CTC) with language model (LM)

**Scoring**:
```
score(text) = α × score_acoustic(text) + β × score_LM(text)

where:
- score_acoustic: CTC probability
- score_LM: Language model probability (e.g., from n-gram or neural LM)
- α, β: Tuning weights
```

**Example**:
```
Acoustic: "wreck a nice beach" = -5.2
Language: "wreck a nice beach" = -15.0 (unlikely phrase)
Combined: -5.2 - 15.0 = -20.2

Acoustic: "recognize speech" = -5.5
Language: "recognize speech" = -3.0 (common phrase)
Combined: -5.5 - 3.0 = -8.5 ✓ (better!)
```

**Implementation** (with KenLM):
```python
import kenlm

# Load language model
lm = kenlm.Model('language_model.arpa')

def beam_search_with_lm(log_probs, lm, alpha=0.7, beta=0.3, beam_size=100):
    # During beam search, score each hypothesis with:
    for score_acoustic, text in beam:
        score_lm = lm.score(text)
        combined_score = alpha * score_acoustic + beta * score_lm
```

---

### 5. Evaluation Metrics

#### Word Error Rate (WER)

**Primary metric** for ASR quality:
```
WER = (S + D + I) / N

where:
S = Substitutions (wrong word)
D = Deletions (missing word)
I = Insertions (extra word)
N = Total words in reference
```

**Example**:
```
Reference:   "the cat sat on the mat"
Hypothesis:  "the cat sit on a mat"

Alignment:
the cat sat on the mat
the cat sit on a   mat
        S      D I

S = 1 (sat → sit)
D = 1 (the → missing)
I = 1 (a → extra)
N = 6

WER = (1 + 1 + 1) / 6 = 50%
```

**Computing WER** (edit distance):
```python
import editdistance

def compute_wer(reference, hypothesis):
    """
    Compute Word Error Rate.

    Args:
        reference: "the cat sat on the mat"
        hypothesis: "the cat sit on a mat"
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Edit distance (minimum insertions/deletions/substitutions)
    distance = editdistance.eval(ref_words, hyp_words)

    wer = distance / len(ref_words)
    return wer
```

**Industry benchmarks**:
- **Clean speech (LibriSpeech test-clean)**: 2-3% WER (near-human)
- **Noisy speech (LibriSpeech test-other)**: 5-6% WER
- **Conversational speech**: 10-15% WER
- **Production goal**: <5% WER for good UX

#### Character Error Rate (CER)

**Same as WER** but at character level:
```
CER = (S + D + I) / N

where S, D, I computed on characters
```

**When to use**:
- **CER**: Languages without clear word boundaries (Chinese, Japanese)
- **WER**: Languages with spaces (English, French, etc.)

---

### 6. Production ASR Challenges

#### Challenge 1: Real-Time Streaming

**Batch ASR**: Process entire audio file
```
User speaks → Wait for completion → Transcribe entire audio
Latency: High (seconds)
Use case: Offline transcription, podcasts
```

**Streaming ASR**: Process audio chunks in real-time
```
User speaks → Chunks (100ms) → Transcribe immediately → Show partial results
Latency: Low (~200ms)
Use case: Virtual assistants, live captions
```

**Challenges**:
- **Causality**: Can only use past context, not future
- **Partial results**: Must provide meaningful intermediate transcriptions
- **Chunk boundaries**: Audio features may span chunk boundaries

**Solution**: Sliding window + online decoding
```python
# Streaming inference
chunk_size = 1600  # 100ms at 16kHz
overlap = 400      # 25ms overlap

for chunk in audio_stream:
    # Process chunk
    features = extract_features(chunk)
    logits = model(features)

    # Decode
    partial_text = decode(logits)

    # Show to user
    display(partial_text)
```

#### Challenge 2: Noise Robustness

**Real-world audio** is noisy:
- Background music
- Other speakers (cocktail party effect)
- Traffic, wind, fans
- Microphone quality variations

**Solutions**:

1. **Data augmentation**:
   ```python
   # Add noise
   clean_audio + 0.1 * noise_audio

   # Speed perturbation
   torchaudio.transforms.SpeedPerturb([0.9, 1.0, 1.1])

   # SpecAugment (mask time/frequency)
   ```

2. **Multi-condition training**:
   ```
   Train on diverse audio:
   - Clean speech
   - Noisy speech
   - Reverberant speech
   - Compressed audio (phone, VoIP)
   ```

3. **Denoising preprocessing**:
   ```
   Noisy audio → Denoiser → Clean audio → ASR model
   ```

#### Challenge 3: Accents and Dialects

**Problem**: Models trained on one accent (e.g., American English) struggle on others (British, Indian, Australian)

**Solutions**:

1. **Diverse training data**: Include all accents
2. **Accent adaptation**: Fine-tune on specific accent
3. **Accent-invariant features**: Use SSL pretraining (Wav2Vec 2.0)

#### Challenge 4: Domain Adaptation

**Problem**: Model trained on one domain (audiobooks) fails on another (medical dictation)

**Vocabulary mismatch**:
```
Audiobooks: "The character was..." → Common words
Medical: "The patient presents with..." → Technical terms
```

**Solutions**:

1. **Domain-specific LM**: Retrain language model on domain texts
2. **Fine-tuning**: Fine-tune acoustic model on domain audio
3. **Shallow fusion**: Combine generic AM with domain LM

---

### 7. Advanced: Encoder-Decoder vs CTC

**CTC**: What we've covered
- **Architecture**: Encoder → CTC → Text
- **Training**: CTC loss (alignment-free)
- **Decoding**: Greedy or beam search
- **Pros**: Simple, fast
- **Cons**: Conditional independence assumption (each character predicted independently)

**Encoder-Decoder (Attention-based)**:
```
Encoder (Audio → Hidden representations)
    ↓
Decoder (Hidden → Text, one character at a time)
    ↑
Attention (which encoder states to focus on)
```

**Example** (Listen, Attend, Spell):
```
Encoder: BiLSTM over audio features → h₁, h₂, ..., hₜ

Decoder (autoregressive):
Step 1: Attend to h₁...hₜ → Generate "H"
Step 2: Given "H", attend to h₁...hₜ → Generate "E"
Step 3: Given "HE", attend to h₁...hₜ → Generate "L"
...
```

**Attention mechanism**:
```
α_t = softmax(score(decoder_state, encoder_states))  # Attention weights
context = Σ α_t × h_t                                 # Weighted sum
output = f(context, decoder_state)                    # Next character
```

**Comparison**:
| | CTC | Attention |
|---|-----|-----------|
| **Alignment** | Implicit (marginalized) | Explicit (learned) |
| **Independence** | Characters independent | Autoregressive (uses previous) |
| **Training** | Faster (parallel) | Slower (sequential) |
| **Decoding** | Faster | Slower (beam search required) |
| **Performance** | Good | Better (with LM) |
| **Streaming** | Easy | Hard (needs future context) |

**Hybrid approach** (RNN-T, Conformer):
- Use both CTC and attention losses
- Best of both worlds

---

## Implementation Guide

### Phase 1: Data Pipeline (Days 1-2)

#### Dataset: LibriSpeech

**LibriSpeech**: 1000 hours of read English audiobooks
- **train-clean-100**: 100 hours, clean speech
- **dev-clean**: Validation set
- **test-clean**: Test set

```python
import torchaudio

# Download LibriSpeech
dataset = torchaudio.datasets.LIBRISPEECH(
    root="./data",
    url="train-clean-100",
    download=True
)

# Each sample: (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
waveform, sr, transcript, *_ = dataset[0]
```

#### Audio Preprocessing

```python
class AudioTransform:
    """Audio preprocessing pipeline"""

    def __init__(self, sample_rate=16000, n_mels=80):
        self.sample_rate = sample_rate

        # Mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels
        )

    def __call__(self, waveform):
        # Mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Log
        log_mel = torch.log(mel_spec + 1e-9)

        # Normalize (per utterance)
        mean = log_mel.mean()
        std = log_mel.std()
        normalized = (log_mel - mean) / (std + 1e-9)

        return normalized  # (n_mels, time)
```

#### Character Encoding

```python
class CharacterTokenizer:
    """Convert text to character indices"""

    def __init__(self):
        # Define vocabulary
        self.chars = [
            '<blank>',  # CTC blank (index 0)
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            ' ', "'",   # Space and apostrophe
        ]

        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text):
        """Text → indices"""
        text = text.lower()  # Lowercase
        indices = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """Indices → text"""
        chars = [self.idx_to_char[idx.item()] for idx in indices]
        return ''.join(chars)
```

#### Dataset Class

```python
class LibriSpeechDataset(Dataset):
    def __init__(self, root, url, transform=None):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=True)
        self.transform = transform or AudioTransform()
        self.tokenizer = CharacterTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sr, transcript, *_ = self.dataset[idx]

        # Audio features
        features = self.transform(waveform)  # (n_mels, T)

        # Text encoding
        labels = self.tokenizer.encode(transcript)

        return features, labels, len(features[0]), len(labels)
```

#### Batching (Variable Lengths)

```python
def collate_fn(batch):
    """
    Collate variable-length sequences.

    Pad to max length in batch.
    """
    features, labels, feature_lens, label_lens = zip(*batch)

    # Pad features to max length
    max_feature_len = max(feature_lens)
    padded_features = []
    for feat in features:
        pad_len = max_feature_len - feat.shape[1]
        padded = F.pad(feat, (0, pad_len))  # Pad time dimension
        padded_features.append(padded)

    features = torch.stack(padded_features)  # (B, n_mels, T_max)

    # Concatenate labels (CTC expects concatenated targets)
    labels = torch.cat(labels)

    # Lengths as tensors
    feature_lens = torch.tensor(feature_lens)
    label_lens = torch.tensor(label_lens)

    return features, labels, feature_lens, label_lens
```

---

### Phase 2: Model (Days 3-4)

Complete CNN-BiLSTM-CTC model (shown in earlier section).

---

### Phase 3: Training (Days 5-6)

```python
def train_asr(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # CTC Loss
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3
    )

    best_loss = float('inf')

    for epoch in range(config['num_epochs']):
        # === TRAINING ===
        model.train()
        train_loss = 0

        for features, labels, feature_lens, label_lens in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            feature_lens = feature_lens.to(device)
            label_lens = label_lens.to(device)

            # Forward
            log_probs = model(features)  # (T, B, num_chars)

            # CTC requires input_lengths (after CNN/pooling)
            # Compute based on CNN downsampling
            input_lengths = feature_lens // 4  # Adjust based on CNN stride

            # CTC loss
            loss = criterion(log_probs, labels, input_lengths, label_lens)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === VALIDATION ===
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_asr.pt')

        scheduler.step(val_loss)

    return model
```

---

### Phase 4: Evaluation (Day 7)

```python
def evaluate_wer(model, test_loader, tokenizer, device):
    """
    Evaluate Word Error Rate.
    """
    model.eval()

    total_wer = 0
    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for features, labels, feature_lens, label_lens in test_loader:
            features = features.to(device)

            # Forward
            log_probs = model(features)  # (T, B, num_chars)

            # Decode each example in batch
            for i in range(log_probs.shape[1]):
                # Greedy decode
                decoded_indices = greedy_decode(log_probs[:, i, :])
                decoded_text = tokenizer.decode(decoded_indices)

                # Ground truth
                start = sum(label_lens[:i])
                end = start + label_lens[i]
                true_indices = labels[start:end]
                true_text = tokenizer.decode(true_indices)

                # Compute WER
                wer = compute_wer(true_text, decoded_text)
                cer = compute_cer(true_text, decoded_text)

                total_wer += wer
                total_cer += cer
                num_samples += 1

    avg_wer = total_wer / num_samples
    avg_cer = total_cer / num_samples

    return avg_wer, avg_cer
```

---

## Expected Results

**LibriSpeech test-clean**:
| Model | WER | CER |
|-------|-----|-----|
| Baseline (no LM) | 15-20% | 5-8% |
| + Beam search | 12-15% | 4-6% |
| + Language model | 8-12% | 3-5% |
| + Large model + LM | 5-8% | 2-4% |
| State-of-the-art (Wav2Vec 2.0) | 2-3% | <1% |

**Real-time factor**: <0.5 (process 1s audio in <0.5s)

---

## Success Criteria

**Theory**:
- [ ] Explain audio representations (waveform, spectrogram, mel-spectrogram)
- [ ] Describe CTC alignment and loss
- [ ] Compare greedy vs beam search decoding
- [ ] Explain WER metric and its components
- [ ] Discuss production challenges (streaming, noise, accents)

**Implementation**:
- [ ] Build audio preprocessing pipeline
- [ ] Implement CNN-BiLSTM-CTC model
- [ ] Train with CTC loss and achieve <20% WER
- [ ] Implement greedy and beam search decoding
- [ ] Evaluate on test set

**Production**:
- [ ] Optimize for real-time streaming
- [ ] Implement data augmentation for robustness
- [ ] Integrate language model for better accuracy
- [ ] Export model (ONNX) for deployment

---

## Resources

**Papers**:
- [CTC: Connectionist Temporal Classification (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Listen, Attend and Spell (Google, 2015)](https://arxiv.org/abs/1508.01211)
- [SpecAugment (Google, 2019)](https://arxiv.org/abs/1904.08779)
- [Wav2Vec 2.0 (Meta, 2020)](https://arxiv.org/abs/2006.11477)
- [Whisper (OpenAI, 2022)](https://arxiv.org/abs/2212.04356)

**Datasets**:
- [LibriSpeech](http://www.openslr.org/12/)
- [Common Voice](https://commonvoice.mozilla.org/)
- [TED-LIUM](https://www.openslr.org/51/)

**Code**:
- [ESPnet](https://github.com/espnet/espnet)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [Wav2Letter](https://github.com/flashlight/wav2letter)

**Ready to build speech recognition? Let's go! 🎤**
