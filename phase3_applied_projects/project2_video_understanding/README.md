# Project 2: Video Understanding 🎬

> **Time:** 1.5-2 weeks
> **Difficulty:** Advanced
> **Goal:** Build a video classification and understanding system for short-form video content

---

## Why This Project Matters

Video is eating the internet. Every minute, users upload:
- **500 hours** of video to YouTube
- **600K** Reels/TikToks created
- **1M hours** of video watched on mobile

This explosion of video content creates massive challenges and opportunities:

**Business Impact**:
- **Content moderation**: Detect harmful content in billions of videos (Meta, YouTube)
- **Recommendations**: Power TikTok For You Page, Instagram Reels, YouTube Shorts
- **Advertising**: Understand video content for targeted ads ($70B+ market)
- **Search**: Enable video search across billions of hours of content
- **Accessibility**: Auto-generate captions for 5% of global population with hearing loss

**Technical Challenges**:
- **Temporal dimension**: Videos have time, not just space
- **Data volume**: 1 min video = 1,800 frames (at 30fps)
- **Compute**: Processing video is 100-1000x more expensive than images
- **Memory**: Loading video can easily exceed GPU memory

**Real-world applications**:
- **Meta**: Instagram Reels and Facebook Watch content understanding
- **TikTok**: Video classification for For You Page recommendations
- **YouTube**: Video categorization and recommendation
- **Snap**: Spotlight video ranking
- **Netflix**: Content analysis and thumbnail generation

---

## The Big Picture

### The Problem

**Input**: Video (sequence of frames + audio)
**Output**: Classification (action labels), embeddings (for search/recommendation), or temporal predictions (when actions occur)

**Why is video different from images?**
1. **Temporal dynamics**: Actions unfold over time (jumping, dancing, cooking)
2. **Motion information**: Movement patterns are critical (walking vs running)
3. **Context accumulation**: Understanding requires seeing multiple frames
4. **Data complexity**: 10-second video at 30fps = 300 frames = 300 images worth of data

**Key Challenges**:
- **Modeling time**: How to capture temporal patterns?
- **Efficiency**: Can't just process every frame independently (too slow)
- **Variable length**: Videos are different durations (5s vs 5 minutes)
- **Memory constraints**: Can't load entire video into GPU memory
- **Motion blur**: Fast motion creates blur (different from images)

### The Solution Landscape

Over the years, researchers have developed multiple approaches:

**Evolution of Video Understanding**:
```
2014: C3D (3D convolutions)
      ↓
2016: Two-Stream Networks (spatial + temporal)
      ↓
2018: I3D, R(2+1)D (improved 3D architectures)
      ↓
2019: SlowFast (dual-pathway processing)
      ↓
2021: TimeSformer, ViViT (Vision Transformers for video)
      ↓
2023: VideoMAE, InternVideo (self-supervised pretraining)
```

**This project**: You'll implement and compare multiple architectures to understand trade-offs.

---

## Deep Dive: Video Understanding Theory

### 1. Representing Video Data

#### Temporal Sampling Strategies

**The fundamental question**: Which frames do we use?

**Dense Sampling**:
```python
# Use all frames
frames = video[0:end]  # 300 frames for 10s video
```
**Pros**: Maximum information
**Cons**: Massive computation, redundant information

**Uniform Sampling**:
```python
# Sample N frames evenly
indices = np.linspace(0, video_length, num_frames)
frames = [video[i] for i in indices]
```
**Pros**: Covers entire video, computationally efficient
**Cons**: Might miss important events between samples

**Random Sampling**:
```python
# Random start + consecutive frames
start = random.randint(0, video_length - clip_length)
frames = video[start:start + clip_length]
```
**Pros**: Data augmentation, prevents overfitting to specific segments
**Cons**: Might miss key moments

**Sliding Window**:
```python
# Overlapping windows
clips = [video[i:i+16] for i in range(0, len(video)-16, stride=4)]
# Then aggregate predictions
```
**Pros**: Dense coverage, good for long videos
**Cons**: Computationally expensive

**Best practice**: Uniform sampling for short clips (< 30s), sliding window for long videos

#### Audio Processing

Video = Visual + Audio. While this project focuses on visual, production systems use both.

**Audio representations**:
- **Waveform**: Raw audio signal
- **Spectrogram**: Frequency over time (like image)
- **Log-mel spectrogram**: Mel-scale matches human hearing
- **MFCC**: Mel-frequency cepstral coefficients (compact representation)

**Why audio matters**:
- Speech indicates talking/conversation
- Music suggests dance/performance
- Environmental sounds (clapping, cheering) provide context

**Multi-modal fusion**:
```python
visual_features = visual_encoder(video_frames)
audio_features = audio_encoder(audio_spectrogram)
combined = torch.cat([visual_features, audio_features], dim=1)
prediction = classifier(combined)
```

---

### 2. 3D Convolutions: Spatial + Temporal

**Key Insight**: Extend 2D convolutions to 3D by adding temporal dimension.

**2D Convolution** (for images):
```
Conv2D: (in_channels, out_channels, H_kernel, W_kernel)
Input:  (B, C, H, W)
Output: (B, C', H', W')
```

**3D Convolution** (for videos):
```
Conv3D: (in_channels, out_channels, T_kernel, H_kernel, W_kernel)
Input:  (B, C, T, H, W)
Output: (B, C', T', H', W')
```

**Example**: 3x3x3 convolution
```python
conv3d = nn.Conv3d(
    in_channels=3,      # RGB
    out_channels=64,
    kernel_size=(3, 3, 3),  # Temporal=3, Spatial=3x3
    padding=(1, 1, 1)
)

# Process 16 frames of 224x224 RGB video
x = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)
out = conv3d(x)  # (1, 64, 16, 224, 224)
```

**What does 3D Conv learn?**
- **Spatial patterns**: Objects, textures (like 2D conv)
- **Temporal patterns**: Motion, changes over time
- **Spatiotemporal patterns**: Moving objects, actions

#### C3D Architecture (2014)

The pioneering 3D CNN for video.

```
Input: 16 frames × 112×112 RGB
   ↓
Conv3D(3→64, k=3×3×3) → ReLU → Pool(1×2×2)
   ↓
Conv3D(64→128, k=3×3×3) → ReLU → Pool(2×2×2)
   ↓
Conv3D(128→256, k=3×3×3) → ReLU → Pool(2×2×2)
   ↓
Conv3D(256→256, k=3×3×3) → ReLU → Pool(2×2×2)
   ↓
Conv3D(256→256, k=3×3×3) → ReLU → Pool(2×2×2)
   ↓
FC(4096) → ReLU → Dropout → FC(4096) → ReLU → Dropout
   ↓
FC(num_classes)
```

**Key insights from C3D**:
- 3×3×3 kernels work well
- Pooling (1, 2, 2) in early layers preserves temporal info
- Pretrained C3D features transfer well to other tasks

**Limitations**:
- Expensive (parameters grow cubically)
- Fixed input length (16 frames)
- Requires lots of data to train

#### R(2+1)D: Factorized 3D Convolutions (2018)

**Key insight**: Decompose 3D conv into 2D spatial + 1D temporal.

**Standard 3D conv**:
```
3×3×3 conv = 27 parameters per filter
Learns spatiotemporal features jointly
```

**R(2+1)D factorization**:
```
3×3×3 conv = (1×3×3 spatial) + (3×1×1 temporal)
           = 9 + 3 = 12 parameters per filter
Learns spatial and temporal features separately
```

**Why factorize?**

1. **More non-linearity**:
   - Standard: Conv3D → ReLU (1 ReLU)
   - R(2+1)D: Conv2D → ReLU → Conv1D → ReLU (2 ReLUs)
   - More non-linearity = more expressiveness

2. **Easier optimization**:
   - Spatial and temporal features learned separately
   - Easier to optimize

3. **Parameter efficiency**:
   - Fewer parameters for same receptive field

**Implementation**:
```python
class R2Plus1DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # Spatial convolution (2D)
        self.spatial = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size//2, kernel_size//2)
        )

        # Temporal convolution (1D)
        self.temporal = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size//2, 0, 0)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.spatial(x)
        x = self.relu(x)
        x = self.temporal(x)
        x = self.relu(x)
        return x
```

**Results**: R(2+1)D achieves similar accuracy to 3D CNN with fewer parameters and faster training.

---

### 3. Two-Stream Networks: Spatial + Temporal Streams

**Motivation**: Humans recognize actions using both:
- **Appearance**: What objects are present? (person, basketball, court)
- **Motion**: How are things moving? (jumping, throwing)

**Architecture**:
```
Video
  ├─→ RGB Frames ─→ Spatial Stream (CNN) ─→ Appearance features
  │
  └─→ Optical Flow ─→ Temporal Stream (CNN) ─→ Motion features
           │
           └─→ Fusion ─→ Classification
```

#### Optical Flow: Capturing Motion

**Optical flow**: Dense pixel-wise motion field between consecutive frames.

```
Frame t:   [person standing]
Frame t+1: [person mid-jump]
           ↓
Optical Flow: Vectors showing motion direction and magnitude
```

**Computing optical flow**:
```python
import cv2

# Read consecutive frames
frame1 = cv2.imread('frame_t.jpg')
frame2 = cv2.imread('frame_t+1.jpg')

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Compute optical flow (Farneback algorithm)
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2,
    None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# flow.shape = (H, W, 2)  # dx and dy for each pixel
```

**Representing optical flow**:
- **Magnitude + Angle**: How far and in what direction
- **Horizontal + Vertical**: dx and dy components
- **Stacked frames**: Stack flow from multiple frame pairs

**Two-Stream Network**:
```python
class TwoStreamNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Spatial stream: ResNet on RGB frames
        self.spatial_stream = resnet50(pretrained=True)
        self.spatial_stream.fc = nn.Linear(2048, num_classes)

        # Temporal stream: ResNet on optical flow
        self.temporal_stream = resnet50(pretrained=False)
        # Modify first conv to accept 2*L channels (L frame pairs)
        self.temporal_stream.conv1 = nn.Conv2d(
            20,  # 10 flow frames × 2 channels (dx, dy)
            64,
            kernel_size=7, stride=2, padding=3
        )
        self.temporal_stream.fc = nn.Linear(2048, num_classes)

    def forward(self, rgb_frames, flow_frames):
        # Spatial stream
        spatial_scores = self.spatial_stream(rgb_frames)

        # Temporal stream
        temporal_scores = self.temporal_stream(flow_frames)

        # Fusion (late fusion: average scores)
        return (spatial_scores + temporal_scores) / 2
```

**Fusion strategies**:

1. **Late fusion**: Average predictions from both streams
   ```python
   final = (spatial_pred + temporal_pred) / 2
   ```

2. **Early fusion**: Concatenate features before classification
   ```python
   combined = torch.cat([spatial_features, temporal_features], dim=1)
   final = classifier(combined)
   ```

3. **Weighted fusion**: Learn weights
   ```python
   final = alpha * spatial_pred + (1 - alpha) * temporal_pred
   ```

**Pros of two-stream**:
- Explicitly models motion
- State-of-the-art results (pre-Transformer era)
- Can use ImageNet pretrained models

**Cons**:
- Expensive: Need to compute optical flow
- Two separate networks to train
- Optical flow computation is slow

---

### 4. SlowFast Networks: Dual-Pathway Processing (2019)

**Biological inspiration**: Human visual system has two pathways:
- **Parvocellular (slow)**: High spatial resolution, low temporal resolution
- **Magnocellular (fast)**: Low spatial resolution, high temporal resolution

**SlowFast architecture** mimics this:

```
Video
  ├─→ Slow Pathway: Few frames, high spatial resolution
  │   (e.g., 4 frames at 224×224)
  │   Captures appearance, objects, scenes
  │
  └─→ Fast Pathway: Many frames, low spatial resolution
      (e.g., 32 frames at 112×112)
      Captures motion, temporal dynamics
      │
      └─→ Lateral connections ─→ Fusion ─→ Prediction
```

**Key design choices**:

1. **Temporal sampling**:
   - Slow: τ frames (e.g., 4)
   - Fast: α·τ frames (e.g., α=8, so 32 frames)

2. **Channel capacity**:
   - Slow: β channels (e.g., 256)
   - Fast: β/α channels (e.g., 32 channels if α=8)
   - Fast pathway is lightweight

3. **Lateral connections**:
   ```python
   # Send information from Fast to Slow
   slow_features = slow_pathway(slow_frames)
   fast_features = fast_pathway(fast_frames)

   # Lateral connection (e.g., time-strided conv)
   fast_to_slow = time_strided_conv(fast_features)
   slow_features = slow_features + fast_to_slow
   ```

**Why it works**:
- **Efficiency**: Fast pathway is lightweight (fewer channels)
- **Specialization**: Each pathway specializes (appearance vs motion)
- **Complementary**: Slow pathway provides context, fast pathway provides motion

**Implementation sketch**:
```python
class SlowFast(nn.Module):
    def __init__(self, num_classes, alpha=8, beta=1/8):
        super().__init__()
        self.alpha = alpha  # Fast/Slow frame ratio

        # Slow pathway (e.g., ResNet-50)
        self.slow_pathway = ResNet3D(layers=[3, 4, 6, 3], channels=64)

        # Fast pathway (lightweight ResNet)
        self.fast_pathway = ResNet3D(
            layers=[3, 4, 6, 3],
            channels=int(64 * beta)  # Fewer channels
        )

        # Lateral connections
        self.lateral_connections = nn.ModuleList([...])

        # Fusion and classifier
        self.classifier = nn.Linear(2048 + 256, num_classes)

    def forward(self, x):
        # Split into slow and fast frames
        slow_frames = x[:, :, ::self.alpha, :, :]  # Subsample temporally
        fast_frames = x  # All frames

        # Process pathways
        slow_features = self.slow_pathway(slow_frames)
        fast_features = self.fast_pathway(fast_frames)

        # Lateral connections (Fast → Slow)
        for i, lateral in enumerate(self.lateral_connections):
            slow_features = slow_features + lateral(fast_features)

        # Combine and classify
        combined = torch.cat([slow_features, fast_features], dim=1)
        return self.classifier(combined)
```

**Results**: SlowFast achieves state-of-the-art on action recognition with efficient computation.

---

### 5. Video Transformers: Attention for Temporal Modeling

**From images to video**:
- **Images**: ViT (Vision Transformer) patches → self-attention
- **Video**: Extend to spatiotemporal patches

#### TimeSformer (2021)

**Key idea**: Factorized space-time attention

**Standard video attention** (too expensive):
```python
# Flatten video into sequence
B, C, T, H, W = video.shape
num_patches = T * (H//16) * (W//16)  # e.g., 16 * 14 * 14 = 3136 patches

# Self-attention over all patches
attention_matrix = Q @ K^T  # (3136, 3136)
# Quadratic in number of patches! Huge memory cost
```

**TimeSformer factorization**:
```
Divided Space-Time Attention:
1. Temporal attention: Attend across time (same spatial location)
2. Spatial attention: Attend across space (same time)
3. Alternate between temporal and spatial
```

**Implementation**:
```python
class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T*HW, D) where HW = num_spatial_patches

        B, THW, D = x.shape
        T = self.num_frames
        HW = THW // T

        # Reshape to (B, T, HW, D)
        x = x.reshape(B, T, HW, D)

        # === TEMPORAL ATTENTION ===
        # For each spatial location, attend across time
        residual = x
        x_temporal = []
        for hw in range(HW):
            # Extract time sequence for this spatial location
            temporal_seq = x[:, :, hw, :]  # (B, T, D)

            # Self-attention across time
            attn_out, _ = self.temporal_attn(
                temporal_seq, temporal_seq, temporal_seq
            )
            x_temporal.append(attn_out)

        x = torch.stack(x_temporal, dim=2)  # (B, T, HW, D)
        x = self.norm1(x + residual)

        # === SPATIAL ATTENTION ===
        # For each time step, attend across space
        residual = x
        x_spatial = []
        for t in range(T):
            # Extract spatial sequence for this time
            spatial_seq = x[:, t, :, :]  # (B, HW, D)

            # Self-attention across space
            attn_out, _ = self.spatial_attn(
                spatial_seq, spatial_seq, spatial_seq
            )
            x_spatial.append(attn_out)

        x = torch.stack(x_spatial, dim=1)  # (B, T, HW, D)
        x = self.norm2(x + residual)

        # Reshape back to (B, T*HW, D)
        return x.reshape(B, THW, D)
```

**Complexity comparison**:
```
Joint space-time attention: O((T·H·W)²)
Divided space-time attention: O(T·H·W·(T + H·W))
                            = O(T²·H·W + T·H²·W²)

For T=8, H=W=14:
Joint: (8*14*14)² = 24M
Divided: 8²*14*14 + 8*14²*14² = 9K + 153K = 162K (150x smaller!)
```

**TimeSformer variants**:
1. **Space-only**: Apply ViT to each frame, pool over time
2. **Joint space-time**: Full attention (expensive)
3. **Divided space-time**: Factorized (efficient)
4. **Sparse local + global**: Attention to nearby + random global patches

**Best**: Divided space-time achieves best accuracy-efficiency trade-off.

---

### 6. Temporal Modeling Approaches Comparison

| Approach | Receptive Field | Parameters | Speed | Accuracy |
|----------|----------------|------------|-------|----------|
| **CNN (2D + pooling)** | Local | Low | Fast | Okay |
| **3D CNN (C3D)** | Local 3D | Medium | Medium | Good |
| **R(2+1)D** | Local 3D | Medium | Medium | Good |
| **Two-Stream** | Global (flow) | High | Slow | Very Good |
| **SlowFast** | Dual (local + global) | Medium | Fast | Very Good |
| **TimeSformer** | Global | High | Slow | Excellent |
| **CNN-LSTM** | Global (sequential) | Medium | Medium | Good |

**When to use what**:
- **Real-time inference**: 3D CNN (R(2+1)D), SlowFast
- **Accuracy priority**: TimeSformer, SlowFast
- **Limited data**: Pretrain on Kinetics, fine-tune
- **Variable length**: CNN-LSTM, Transformer
- **Resource constrained**: 2D CNN + temporal pooling

---

### 7. Video Classification vs Detection vs Segmentation

**Classification** (this project):
```
Input: Video
Output: Single label (e.g., "basketball")
```

**Temporal Action Detection**:
```
Input: Long video (minutes)
Output: Time intervals + labels
Example: [0:05-0:12: "jump", 0:15-0:30: "run"]
```

**Spatiotemporal Detection** (action localization):
```
Input: Video
Output: Bounding boxes over time + labels
Example: Track person bounding box + action label
```

**Video Segmentation**:
```
Input: Video
Output: Pixel-wise labels per frame
Example: Segment person, background, objects in every frame
```

**Progression**:
- Classification: Simplest, this project
- Temporal detection: Requires temporal proposals
- Spatiotemporal: Requires 3D bounding boxes
- Segmentation: Most complex, pixel-level

---

### 8. Production Considerations: Efficiency & Scale

#### Compute Cost

**Example**: Process 1M videos/day at 10s each

**Naive approach** (process every frame):
```
1M videos × 10s × 30fps = 300M frames/day
With ResNet-50: ~4 GFLOPs/frame
Total: 300M × 4 = 1.2B GFLOPs/day = 13M GFLOPs/second
Needs: ~1000 GPUs continuously!
```

**Optimizations**:

1. **Fewer frames**: Sample 8-16 frames instead of all 300
   ```
   1M videos × 16 frames = 16M frames/day (20x reduction!)
   ```

2. **Lower resolution**: 112×112 instead of 224×224
   ```
   4x fewer pixels = ~4x faster
   ```

3. **Model compression**:
   - Quantization (INT8): 4x smaller, 2-3x faster
   - Pruning: Remove 30-50% of weights
   - Distillation: Large model → small model

4. **Batching**: Process multiple videos in parallel
   ```python
   # Batch inference
   videos = [video1, video2, ..., video32]  # Batch size 32
   predictions = model(videos)  # GPU utilizes parallelism
   ```

5. **Two-stage pipeline**:
   ```
   Stage 1: Fast lightweight model on all videos
            → Filter out easy negatives
   Stage 2: Expensive model only on uncertain cases
   ```

#### Memory Management

**Problem**: Video doesn't fit in GPU memory
```
Batch size: 16 videos
Frames per video: 64
Resolution: 224×224
Channels: 3

Total: 16 × 64 × 224 × 224 × 3 × 4 bytes (float32)
     = 4.8 GB just for input!
```

**Solutions**:

1. **Gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint

   # Don't store intermediate activations, recompute during backward
   output = checkpoint(model.layer, input)
   # Trades compute for memory (2x slower, 10x less memory)
   ```

2. **Mixed precision (FP16)**:
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       output = model(input)  # Uses FP16 (half memory)
       loss = criterion(output, target)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Gradient accumulation**:
   ```python
   # Simulate large batch with small batches
   optimizer.zero_grad()
   for i, (video, label) in enumerate(small_batches):
       loss = model(video, label) / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **CPU offloading**:
   ```python
   # Keep model on GPU, load data on-the-fly
   for video, label in dataloader:  # DataLoader with prefetch
       video = video.cuda(non_blocking=True)  # Async transfer
       output = model(video)
   ```

#### Inference Optimization

**Goal**: Process video in < 100ms for real-time applications

**Techniques**:

1. **ONNX Export**:
   ```python
   import torch.onnx

   dummy_input = torch.randn(1, 3, 16, 224, 224)
   torch.onnx.export(
       model, dummy_input, "video_model.onnx",
       opset_version=11,
       input_names=['video'],
       output_names=['logits']
   )
   ```

2. **TensorRT** (NVIDIA GPUs):
   ```python
   import tensorrt as trt

   # Convert ONNX to TensorRT engine
   # 2-5x speedup with FP16
   ```

3. **Model quantization**:
   ```python
   import torch.quantization

   # Quantize to INT8
   model_int8 = torch.quantization.quantize_dynamic(
       model, {nn.Linear}, dtype=torch.qint8
   )
   # 4x smaller, 2-3x faster on CPU
   ```

4. **Early exit**:
   ```python
   class EarlyExitModel(nn.Module):
       def __init__(self):
           self.backbone = ResNet3D()
           self.early_exit = nn.Linear(512, num_classes)
           self.final_exit = nn.Linear(2048, num_classes)

       def forward(self, x):
           x = self.backbone.layer1(x)
           x = self.backbone.layer2(x)

           # Early exit for easy examples
           early_logits = self.early_exit(x)
           confidence = early_logits.softmax(dim=1).max()
           if confidence > 0.9:
               return early_logits  # Exit early!

           # Continue for hard examples
           x = self.backbone.layer3(x)
           x = self.backbone.layer4(x)
           return self.final_exit(x)
   ```

---

## Implementation Guide

### Phase 1: Video Data Pipeline (Week 1, Days 1-3)

#### Loading Videos Efficiently

**Libraries**:
- **decord**: Fast, Python-friendly
- **torchvision.io**: PyTorch native
- **OpenCV**: Widely used but slower
- **PyAV**: Python bindings for FFmpeg

**Recommended: decord**
```python
from decord import VideoReader, cpu, gpu

# Load video
vr = VideoReader('video.mp4', ctx=cpu(0))
print(f"Total frames: {len(vr)}")
print(f"FPS: {vr.get_avg_fps()}")

# Sample frames
indices = np.linspace(0, len(vr)-1, num_frames).astype(int)
frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

# Convert to tensor
frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
```

#### Video Dataset Implementation

```python
class VideoDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        video_dir,
        num_frames=16,
        frame_size=224,
        transform=None,
        sampling='uniform'
    ):
        """
        Args:
            annotations_file: CSV with columns [video_path, label]
            video_dir: Directory containing videos
            num_frames: Number of frames to sample
            frame_size: Spatial resolution
            transform: Augmentation pipeline
            sampling: 'uniform', 'random', or 'dense'
        """
        self.annotations = pd.read_csv(annotations_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.sampling = sampling

    def __len__(self):
        return len(self.annotations)

    def _sample_frames(self, video_reader, num_frames):
        """Sample frames from video"""
        total_frames = len(video_reader)

        if self.sampling == 'uniform':
            # Evenly spaced frames
            indices = np.linspace(0, total_frames-1, num_frames).astype(int)

        elif self.sampling == 'random':
            # Random start + consecutive frames
            if total_frames > num_frames:
                start = np.random.randint(0, total_frames - num_frames)
                indices = np.arange(start, start + num_frames)
            else:
                indices = np.linspace(0, total_frames-1, num_frames).astype(int)

        elif self.sampling == 'dense':
            # All frames (for short videos)
            indices = np.arange(total_frames)

        return indices

    def __getitem__(self, idx):
        # Get video path and label
        video_path = os.path.join(
            self.video_dir,
            self.annotations.iloc[idx]['video_path']
        )
        label = self.annotations.iloc[idx]['label']

        # Load video
        vr = VideoReader(video_path, ctx=cpu(0))

        # Sample frames
        frame_indices = self._sample_frames(vr, self.num_frames)
        frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

        # Convert to tensor: (T, H, W, C) → (T, C, H, W)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

        # Apply transforms
        if self.transform:
            frames = self.transform(frames)

        # Resize to target size
        frames = F.interpolate(
            frames,
            size=(self.frame_size, self.frame_size),
            mode='bilinear',
            align_corners=False
        )

        # Change to (C, T, H, W) for 3D conv
        frames = frames.permute(1, 0, 2, 3)

        return frames, label
```

#### Video Augmentation

```python
class VideoAugmentation:
    """
    Video-specific augmentation techniques
    """

    @staticmethod
    def temporal_crop(video, clip_length):
        """Randomly crop temporal segment"""
        T = video.shape[1]
        if T > clip_length:
            start = np.random.randint(0, T - clip_length)
            return video[:, start:start + clip_length, :, :]
        return video

    @staticmethod
    def spatial_crop(video, crop_size):
        """Random spatial crop (consistent across frames)"""
        C, T, H, W = video.shape
        top = np.random.randint(0, H - crop_size)
        left = np.random.randint(0, W - crop_size)
        return video[:, :, top:top+crop_size, left:left+crop_size]

    @staticmethod
    def random_horizontal_flip(video, p=0.5):
        """Flip video horizontally"""
        if np.random.random() < p:
            return torch.flip(video, dims=[3])  # Flip width
        return video

    @staticmethod
    def color_jitter(video, brightness=0.4, contrast=0.4, saturation=0.4):
        """Apply color jitter (same transform to all frames)"""
        transforms = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        )

        # Apply to each frame
        C, T, H, W = video.shape
        jittered = []
        for t in range(T):
            frame = video[:, t, :, :]  # (C, H, W)
            jittered.append(transforms(frame))

        return torch.stack(jittered, dim=1)  # (C, T, H, W)

    @staticmethod
    def normalize(video, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Normalize using ImageNet stats"""
        mean = torch.tensor(mean).view(3, 1, 1, 1)
        std = torch.tensor(std).view(3, 1, 1, 1)
        return (video - mean) / std


# Usage
transform_train = torchvision.transforms.Compose([
    lambda x: VideoAugmentation.temporal_crop(x, clip_length=16),
    lambda x: VideoAugmentation.spatial_crop(x, crop_size=224),
    lambda x: VideoAugmentation.random_horizontal_flip(x),
    lambda x: VideoAugmentation.color_jitter(x),
    lambda x: VideoAugmentation.normalize(x)
])
```

---

### Phase 2: Model Architecture (Week 1, Days 4-5)

#### Option A: R(2+1)D Implementation

```python
class R2Plus1DBlock(nn.Module):
    """
    R(2+1)D residual block:
    - 2D spatial convolution
    - 1D temporal convolution
    - Skip connection
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Compute intermediate channels (for factorization)
        mid_channels = int(
            (in_channels * out_channels * 3 * 3 * 3) /
            (in_channels * 3 * 3 + 3 * out_channels)
        )

        # 2D spatial convolution (1×3×3)
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 1D temporal convolution (3×1×1)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(1, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(out_channels)
        )

        # Skip connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(stride, stride, stride),
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # R(2+1)D convolution
        out = self.spatial(x)
        out = self.temporal(out)

        # Skip connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class R2Plus1DNet(nn.Module):
    """
    R(2+1)D network for video classification
    """

    def __init__(self, num_classes=400, num_frames=16):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # R(2+1)D blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []

        # First block with stride
        layers.append(R2Plus1DBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        logits = self.fc(x)
        return logits
```

#### Option B: CNN-LSTM Implementation

```python
class CNNLSTMVideoClassifier(nn.Module):
    """
    CNN-LSTM hybrid:
    - CNN extracts per-frame features
    - LSTM models temporal dependencies
    - Classification head
    """

    def __init__(self, num_classes=400, hidden_dim=512, num_lstm_layers=2):
        super().__init__()

        # Per-frame CNN (pretrained ResNet-18)
        resnet = torchvision.models.resnet18(pretrained=True)
        # Remove final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        cnn_output_dim = 512  # ResNet-18 output

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Attention pooling (optional)
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Classification head
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Reshape to process frames independently
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)

        # CNN feature extraction
        features = self.cnn(x)  # (B*T, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, 512)

        # Reshape back to sequence
        features = features.view(B, T, -1)  # (B, T, 512)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)  # (B, T, 1024)

        # Attention pooling over time
        attention_weights = self.attention(lstm_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled = (lstm_out * attention_weights).sum(dim=1)  # (B, 1024)

        # Classification
        logits = self.fc(pooled)
        return logits
```

---

### Phase 3: Training (Week 2, Days 1-2)

```python
def train_video_model(model, train_loader, val_loader, config):
    """
    Training loop for video classification
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=1e-4
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(videos)
                loss = criterion(logits, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                logits = model(videos)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total

        # Logging
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_video_model.pt')

        # Learning rate scheduling
        scheduler.step()

    return model
```

---

## Expected Results

**Dataset: Kinetics-50 (subset)**

| Model | Top-1 Acc | Top-5 Acc | FPS (GPU) | Params |
|-------|-----------|-----------|-----------|--------|
| 2D CNN + Pooling | 55% | 80% | 120 | 11M |
| R(2+1)D-18 | 65% | 88% | 40 | 33M |
| CNN-LSTM | 62% | 85% | 35 | 25M |
| SlowFast (if implemented) | 70% | 90% | 25 | 34M |
| TimeSformer (if implemented) | 72% | 92% | 15 | 122M |

**Full Kinetics-400** (if you have resources):
- R(2+1)D-50: 74% top-1
- SlowFast-50: 77% top-1
- TimeSformer: 79% top-1

---

## Success Criteria

**Theory**:
- [ ] Explain why video is different from images (temporal dimension)
- [ ] Describe 3D convolutions and how they capture spatiotemporal patterns
- [ ] Compare R(2+1)D, two-stream, SlowFast, and Transformers
- [ ] Discuss temporal sampling strategies and trade-offs
- [ ] Explain efficiency challenges and optimization techniques

**Implementation**:
- [ ] Build efficient video data pipeline with proper augmentation
- [ ] Implement at least one video model (R(2+1)D or CNN-LSTM)
- [ ] Train model with mixed precision and achieve >60% top-1 accuracy
- [ ] Generate video embeddings for retrieval/recommendation
- [ ] Optimize model for production (quantization, ONNX export)

**Production**:
- [ ] Explain compute and memory challenges at scale
- [ ] Design two-stage pipeline (fast filter + expensive model)
- [ ] Benchmark inference speed (target: >30 FPS on GPU)
- [ ] Discuss applications: Reels recommendation, content moderation, search

---

## Resources

**Papers**:
- [C3D: Learning Spatiotemporal Features (Facebook, 2014)](https://arxiv.org/abs/1412.0767)
- [Two-Stream Convolutional Networks (Oxford, 2014)](https://arxiv.org/abs/1406.2199)
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Facebook, 2018)](https://arxiv.org/abs/1711.11248)
- [SlowFast Networks (Facebook, 2019)](https://arxiv.org/abs/1812.03982)
- [TimeSformer: Is Space-Time Attention All You Need? (Facebook, 2021)](https://arxiv.org/abs/2102.05095)

**Code**:
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo)
- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [Decord](https://github.com/dmlc/decord)

**Datasets**:
- [Kinetics-400/600/700](https://deepmind.com/research/open-source/kinetics)
- [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)
- [ActivityNet](http://activity-net.org/)

**Ready to understand videos like Instagram Reels? Let's go! 🎬**
