# Project 4: Speech Recognition Model 🎤

> **Time:** 1 week
> **Difficulty:** Intermediate-Advanced
> **Goal:** Build an ASR (Automatic Speech Recognition) pipeline

---

## 📚 Project Overview

Build a speech recognition system that converts audio to text. This project covers audio preprocessing, acoustic modeling, and sequence-to-sequence prediction.

**Real-world application:** Power voice assistants, video captioning, and accessibility features.

---

## 🎯 Architecture

```
Speech Recognition Pipeline

Audio (waveform)
    ↓
Preprocessing (MFCC / Log-Mel Spectrogram)
    ↓
CNN Feature Extractor
    ↓
Bidirectional LSTM
    ↓
CTC Decoder
    ↓
Text Transcription
```

---

## 🎓 Theory Brief

### Audio Preprocessing

**1. MFCC (Mel-Frequency Cepstral Coefficients):**
- Convert waveform to frequency domain
- Apply mel-scale (matches human perception)
- Take log and DCT
- Result: (time_steps, 40) features

**2. Log-Mel Spectrogram:**
- Short-Time Fourier Transform (STFT)
- Apply mel filterbank
- Take log
- Result: (time_steps, n_mels) features

### Model Architecture

**CNN:** Extract local patterns in spectrograms
**BiLSTM:** Model temporal dependencies (past + future context)
**CTC Loss:** Align audio to text without explicit alignment labels

### CTC (Connectionist Temporal Classification)

**Problem:** Audio and text have different lengths, no alignment

**Solution:** CTC introduces blank token and allows repetitions
- "h-e-ll-l-oo" → "hello"
- Handles variable-length sequences
- No need for frame-level labels

---

## 🚀 Milestones

### Milestone 1: Audio Data Pipeline
**Tasks:**
- [ ] Download dataset (LibriSpeech or Common Voice)
- [ ] Load audio files (torchaudio or librosa)
- [ ] Extract MFCC and log-mel spectrograms
- [ ] Implement audio augmentation (noise, pitch shift, time stretch)
- [ ] Build DataLoader with padding/batching

**Preprocessing Code:**
```python
import torchaudio

def extract_features(waveform, sample_rate):
    # Log-Mel Spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        n_mels=80
    )(waveform)
    log_mel = torch.log(mel_spec + 1e-9)
    return log_mel  # (n_mels, time)
```

---

### Milestone 2: CNN-BiLSTM Model
**Tasks:**
- [ ] Implement CNN feature extractor
- [ ] Implement BiLSTM encoder
- [ ] Add CTC output layer
- [ ] Character-level vocabulary

**Model Code:**
```python
class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_mels, n_class, hidden_dim=512):
        # CNN: 2-3 conv layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=64 * (n_mels // 4),
            hidden_size=hidden_dim,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )

        # Output projection
        self.fc = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, x):
        # x: (B, n_mels, time)
        x = x.unsqueeze(1)  # (B, 1, n_mels, time)
        x = self.cnn(x)  # (B, 64, n_mels', time')

        # Reshape for LSTM
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)  # (B, T, C*H)

        x, _ = self.lstm(x)  # (B, T, 2*hidden_dim)
        x = self.fc(x)  # (B, T, n_class)

        return x.log_softmax(dim=-1)
```

---

### Milestone 3: CTC Training
**Tasks:**
- [ ] Implement CTC loss
- [ ] Train with variable-length sequences
- [ ] Learning rate scheduling
- [ ] Gradient clipping (important for RNNs)

**Training Code:**
```python
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

def train_step(model, audio, targets, audio_lens, target_lens):
    log_probs = model(audio)  # (B, T, C)
    log_probs = log_probs.permute(1, 0, 2)  # (T, B, C) for CTC

    loss = criterion(log_probs, targets, audio_lens, target_lens)
    return loss
```

---

### Milestone 4: Decoding & Evaluation
**Tasks:**
- [ ] Implement greedy CTC decoding
- [ ] Implement beam search (optional)
- [ ] Calculate WER (Word Error Rate)
- [ ] Calculate CER (Character Error Rate)

**Decoding:**
```python
def greedy_decode(log_probs, blank=0):
    """Greedy CTC decoder"""
    indices = log_probs.argmax(dim=-1)  # (B, T)

    # Remove blanks and repetitions
    decoded = []
    for seq in indices:
        chars = []
        prev_char = blank
        for char in seq:
            if char != blank and char != prev_char:
                chars.append(char.item())
            prev_char = char
        decoded.append(chars)

    return decoded
```

**Metrics:**
- **WER (Word Error Rate):** % words incorrectly predicted
- **CER (Character Error Rate):** % characters incorrectly predicted

---

### Milestone 5: Data Augmentation & Optimization
**Tasks:**
- [ ] Implement SpecAugment (mask time/frequency)
- [ ] Add background noise
- [ ] Speed perturbation
- [ ] Mixed precision training
- [ ] Model quantization

**SpecAugment:**
```python
def spec_augment(spec, freq_mask=15, time_mask=35, n_masks=2):
    """Mask frequency and time for augmentation"""
    for _ in range(n_masks):
        # Frequency masking
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, spec.shape[0] - f)
        spec[f0:f0+f, :] = 0

        # Time masking
        t = random.randint(0, time_mask)
        t0 = random.randint(0, spec.shape[1] - t)
        spec[:, t0:t0+t] = 0

    return spec
```

---

## 🎯 Stretch Goal: Wav2Vec-Style Pretraining

Implement self-supervised pretraining:

**Approach:**
1. Mask portions of input audio
2. Train model to predict masked regions
3. Finetune on transcription task

**Benefits:**
- Learn from unlabeled audio
- Better feature representations
- Improved performance with less labeled data

---

## 🏭 Meta-Scale Considerations

### 1. Multi-Language Support
- Train language-specific models
- Multilingual model with language ID
- Code-switching detection

### 2. Real-Time Inference
- Streaming ASR (process audio chunks)
- Low-latency models (< 100ms)
- On-device inference

### 3. Noise Robustness
- Train on noisy audio
- Denoise before ASR
- Multi-condition training

### 4. Speaker Adaptation
- Speaker embeddings
- Personalized acoustic models
- Voice profile learning

---

## 📊 Expected Results

**Dataset: LibriSpeech (clean subset)**
- **WER:** < 10% on test-clean
- **CER:** < 5%
- **Real-time factor:** < 0.5 (process 1s audio in < 0.5s)

---

## 📚 Resources

**Papers:**
- [CTC: Connectionist Temporal Classification (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [SpecAugment (Google, 2019)](https://arxiv.org/abs/1904.08779)
- [Wav2Vec 2.0 (Meta, 2020)](https://arxiv.org/abs/2006.11477)

**Datasets:**
- [LibriSpeech](http://www.openslr.org/12/)
- [Common Voice](https://commonvoice.mozilla.org/)

---

## ✅ Success Criteria

- [ ] Build audio preprocessing pipeline
- [ ] Implement CNN-BiLSTM-CTC model
- [ ] Train with CTC loss
- [ ] Achieve < 15% WER
- [ ] Implement data augmentation
- [ ] Optimize for real-time inference

**Ready to build speech recognition? Let's go! 🎤**
