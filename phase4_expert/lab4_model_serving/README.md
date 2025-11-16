# Lab 4: Model Serving & Production Inference ⚡

> **Time:** 2-3 hours
> **Difficulty:** Expert
> **Goal:** Master production model serving with TorchServe, quantization, and optimization for <50ms p99 latency

---

## 📖 Why This Lab Matters

You've trained a model. It achieves 95% accuracy on your test set. **Now comes the hard part: serving it to millions of users in production.**

**The reality check:**
```python
# Research code (works fine)
model.eval()
with torch.no_grad():
    prediction = model(input_tensor)  # Takes 200ms on CPU

# Production requirement:
# - 10,000 queries per second (QPS)
# - <50ms latency (p99)
# - Running on CPU (GPUs expensive for inference)

# Math:
# 200ms per query × 10,000 QPS = 2,000 seconds of compute needed per second
# You need 2,000 CPUs! ($$$$$$)
```

**Production ML reality:**
- **GPT-3 API:** <2s response time for 100+ billion parameter model serving millions of users
- **Google Search:** <100ms for personalized ranking across billions of pages
- **Meta News Feed:** <50ms to rank thousands of posts in real-time
- **Amazon product recommendations:** <10ms latency at scale

**The gap:** Research models are 10-100x too slow for production!

This lab teaches you production-grade model serving:
- **TorchServe:** Production serving framework (Meta's official solution)
- **Quantization:** 4x speedup + 4x memory reduction (INT8 vs FP32)
- **Dynamic batching:** Process multiple requests together (10x throughput)
- **Model optimization:** TorchScript, ONNX, operator fusion
- **Latency optimization:** p50 < 10ms, p99 < 50ms

**Master serving, and your models actually reach production—not just notebooks.**

---

## 🧠 The Big Picture: Research vs Production

### Research Model (Training)

```python
# Training code
model = ResNet50()
model.train()

for epoch in range(100):
    for batch in dataloader:  # Batch size: 64-256
        outputs = model(batch)
        loss.backward()
        optimizer.step()

# Characteristics:
# - Batch processing (high throughput)
# - GPU accelerated (A100s)
# - Latency doesn't matter (offline training)
# - Memory abundant (80GB GPU)
```

**Optimization target:** Maximize throughput (samples/second)

### Production Model (Inference)

```python
# Production serving
@app.post("/predict")
async def predict(request):
    input_data = preprocess(request.data)
    prediction = model(input_data)  # Single request!
    return {"prediction": prediction}

# Characteristics:
# - Single request (low latency)
# - CPU deployment (cost optimization)
# - Latency critical (<50ms p99)
# - Memory constrained (few GB RAM)
```

**Optimization target:** Minimize latency (ms per request)

### The Serving Problem

**Problem 1: CPU vs GPU economics**
```
GPU (A100):
  - Cost: $3/hour (cloud) or $15K (purchase)
  - Throughput: 1000 QPS (with batching)
  - Use case: High-volume batch serving

CPU (c5.4xlarge):
  - Cost: $0.68/hour (cloud)
  - Throughput: 50 QPS (single request)
  - Use case: Low-latency serving

For 10,000 QPS:
  GPUs: 10 × $3/hr = $30/hr ← Cheaper!
  CPUs: 200 × $0.68/hr = $136/hr

But: GPU inference has 50-100ms latency vs 10-20ms on CPU
```

**Problem 2: Batch size 1 is inefficient**
```
Model throughput by batch size (GPU):
  Batch 1:   100 samples/second
  Batch 32:  2000 samples/second (20x improvement!)
  Batch 128: 5000 samples/second (50x improvement!)

But production requests arrive one at a time!
```

**Problem 3: Model memory**
```
ResNet50 (FP32): 100MB parameters
BERT-large (FP32): 1.3GB parameters
GPT-3 (FP32): 700GB parameters (doesn't fit on single GPU!)

Quantization (INT8):
  ResNet50: 100MB → 25MB (4x reduction)
  BERT-large: 1.3GB → 325MB (4x reduction)

Can now serve 4x more models per machine!
```

---

## 🔬 Deep Dive: Production Serving Stack

### 1. TorchServe Architecture

**TorchServe** is Meta's production serving framework for PyTorch models.

**Components:**
```
┌─────────────────────────────────────┐
│         Load Balancer (Nginx)        │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │   TorchServe   │
       │   Frontend     │  ← HTTP/gRPC API
       └───────┬────────┘
               │
    ┌──────────┴──────────┐
    │   Model Workers     │  ← Parallel inference
    │  [Worker 1, 2, 3]   │
    └──────────┬──────────┘
               │
       ┌───────┴────────┐
       │  Model Store   │  ← .mar files (model archives)
       └────────────────┘
```

**Request flow:**
1. HTTP request → TorchServe frontend
2. Frontend → Queue request
3. Worker picks up request → Inference
4. Response → Client

**Dynamic batching:**
```python
# TorchServe automatically batches requests
# Config:
batch_size = 32
max_batch_delay = 50  # milliseconds

# Behavior:
# - Wait up to 50ms to collect 32 requests
# - If 32 requests arrive → batch immediately
# - If timeout (50ms) → process whatever we have
# - Automatic parallelization across workers
```

### 2. Model Archive Format (.mar)

**Create model archive:**
```bash
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet50.pth \
  --handler image_classifier \
  --extra-files index_to_name.json
```

**Archive contents:**
```
resnet50.mar/
├── model.py              # Model definition
├── resnet50.pth          # Trained weights
├── handler.py            # Custom inference logic
├── index_to_name.json    # Class labels
└── MANIFEST.json         # Metadata
```

**Custom handler:**
```python
from ts.torch_handler.base_handler import BaseHandler

class CustomHandler(BaseHandler):
    def preprocess(self, data):
        """Preprocess input data."""
        images = []
        for row in data:
            # Decode image
            image = Image.open(io.BytesIO(row.get("data")))
            # Resize and normalize
            image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    def inference(self, data):
        """Run model inference."""
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def postprocess(self, data):
        """Format output."""
        probs = torch.softmax(data, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

        results = []
        for prob, idx in zip(top5_prob, top5_idx):
            results.append({
                "class": self.mapping[str(idx.item())],
                "probability": prob.item()
            })
        return results
```

### 3. Quantization for Inference

**Problem:** FP32 models are slow and memory-heavy.

**Solution:** Quantize to INT8 (8-bit integers).

**Quantization math:**
```
FP32 representation: [-3.14159, 2.71828, ...]
  Each number: 32 bits (4 bytes)

INT8 quantization:
  1. Find range: min = -3.14159, max = 2.71828
  2. Compute scale: scale = (max - min) / 255 = 0.023
  3. Quantize: int8_value = round((fp32_value - min) / scale)

  Example:
    2.71828 → round((2.71828 - (-3.14159)) / 0.023) = 254
    -3.14159 → round((-3.14159 - (-3.14159)) / 0.023) = 0

  Each number: 8 bits (1 byte) → 4x compression!
```

**Quantization types:**

**1. Dynamic Quantization (simplest)**
```python
import torch.quantization

# Quantize model (post-training)
model_fp32 = MyModel()
model_fp32.load_state_dict(torch.load("model.pth"))
model_fp32.eval()

# Dynamic quantization: weights INT8, activations computed on-the-fly
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Quantize Linear layers only
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), "model_int8.pth")
```

**Speedup:** 2-4x on CPU inference
**Accuracy loss:** <1% for most models

**2. Static Quantization (best accuracy)**
```python
# Requires calibration data
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model_fp32)

# Calibrate with representative data
with torch.no_grad():
    for batch in calibration_data:
        model_prepared(batch)

# Convert to quantized model
model_int8 = torch.quantization.convert(model_prepared)
```

**Speedup:** 3-5x on CPU inference
**Accuracy loss:** <0.5% with proper calibration

**3. Quantization-Aware Training (QAT)**
```python
# Simulate quantization during training
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train as usual (with fake quantization)
for epoch in range(10):
    for batch in dataloader:
        loss = criterion(model_prepared(batch), labels)
        loss.backward()
        optimizer.step()

# Convert to INT8
model_int8 = torch.quantization.convert(model_prepared)
```

**Speedup:** 3-5x on CPU inference
**Accuracy loss:** <0.1% (best accuracy preservation)

**Quantization comparison:**

| Method | Speed | Accuracy | Effort | Use Case |
|--------|-------|----------|--------|----------|
| Dynamic | 2-4x | -1% | Low | Quick optimization |
| Static | 3-5x | -0.5% | Medium | Production default |
| QAT | 3-5x | -0.1% | High | Critical accuracy requirements |

### 4. TorchScript Optimization

**Problem:** Python interpreter overhead slows inference.

**Solution:** Compile to TorchScript (C++ runtime).

**TorchScript compilation:**
```python
# Method 1: Tracing (simpler, more compatible)
model = MyModel()
model.eval()

example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("model_traced.pt")

# Load in production (no Python dependencies!)
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(example_input)
```

**Method 2: Scripting (handles control flow)**
```python
# For models with if/loops
class MyDynamicModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Control flow!
            x = x * 2
        return self.layer(x)

# Tracing won't work (traces single execution path)
# Use scripting instead:
scripted_model = torch.jit.script(MyDynamicModel())
scripted_model.save("model_scripted.pt")
```

**Benefits:**
- 1.5-2x faster inference (no Python overhead)
- Can load in C++ (no Python runtime needed)
- Operator fusion (conv + bn + relu → single op)
- Constant folding (precompute static ops)

### 5. ONNX Export (cross-framework)

**ONNX (Open Neural Network Exchange):** Universal model format.

**Why ONNX:**
- Framework agnostic (PyTorch → TensorFlow, PyTorch → TensorRT)
- Optimized runtimes (ONNX Runtime 2-5x faster than PyTorch)
- Hardware acceleration (TensorRT for NVIDIA, OpenVINO for Intel)

**Export to ONNX:**
```python
import torch.onnx

model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},  # Variable batch size
        'output': {0: 'batch_size'}
    }
)
```

**Inference with ONNX Runtime:**
```python
import onnxruntime as ort

# Load model
session = ort.InferenceSession("model.onnx")

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

outputs = session.run(
    [output_name],
    {input_name: input_data.numpy()}
)
```

**ONNX + TensorRT (NVIDIA GPUs):**
```python
import tensorrt as trt

# Convert ONNX → TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()

parser = trt.OnnxParser(network, logger)
parser.parse_from_file("model.onnx")

# Build optimized engine
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

# Save
with open("model.trt", "wb") as f:
    f.write(engine.serialize())

# Inference: 5-10x faster than PyTorch on GPU!
```

---

## 📊 Mathematical Foundations

### Quantization Theory

**Linear quantization mapping:**
$$q = \text{round}\left(\frac{r}{S}\right) + Z$$

Where:
- $r$: Real (FP32) value
- $q$: Quantized (INT8) value
- $S$: Scale factor
- $Z$: Zero-point (offset)

**Scale and zero-point calculation:**
$$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$$
$$Z = q_{\min} - \text{round}\left(\frac{r_{\min}}{S}\right)$$

For INT8 (signed): $q_{\min} = -128$, $q_{\max} = 127$

**Dequantization (INT8 → FP32):**
$$r = S(q - Z)$$

**Quantized matrix multiplication:**
```
FP32: C = A × B
  Cost: N³ multiply-add operations (FP32)

INT8: C_q = (A_q - Z_A) × (B_q - Z_B)
  Cost: N³ multiply-add operations (INT8) ← 4x faster!

  Then dequantize:
    C = S_A × S_B × C_q + ...
```

**Error analysis:**
$$\epsilon = |r - \hat{r}| = |r - S(q - Z)|$$

For uniform quantization:
$$\epsilon_{\max} = \frac{S}{2}$$

**Signal-to-Quantization-Noise Ratio (SQNR):**
$$\text{SQNR} = 10 \log_{10} \frac{P_{\text{signal}}}{P_{\text{noise}}} \approx 6.02b + 1.76 \text{ dB}$$

Where $b$ = number of bits.
- INT8 (8 bits): SQNR ≈ 50 dB
- FP32 (23-bit mantissa): SQNR ≈ 140 dB

**Practical implication:** INT8 preserves ~99.99% of signal for typical neural networks.

### Latency Analysis

**End-to-end latency breakdown:**
```
Total latency = Network + Preprocessing + Inference + Postprocessing

Example (image classification):
  Network:        5ms  (send 500KB image over 1 Gbps)
  Preprocessing: 10ms  (decode JPEG, resize, normalize)
  Inference:     30ms  (model forward pass)
  Postprocessing: 2ms  (softmax, top-K)
  ─────────────────────
  Total:         47ms  (p50 latency)
```

**Inference latency components:**
```
Inference = Memory_load + Compute + Memory_store

ResNet50 (FP32, CPU):
  Memory load:  20ms (load 100MB weights from RAM)
  Compute:      25ms (50 GFLOP / 2 GFLOP/s)
  Memory store:  1ms (write 1000 outputs)
  ─────────────────────
  Total:        46ms

With quantization (INT8):
  Memory load:   5ms (load 25MB weights, 4x faster)
  Compute:       8ms (50 GFLOP / 6 GFLOP/s, INT8 faster)
  Memory store:  1ms
  ─────────────────────
  Total:        14ms  (3.3x speedup!)
```

**Batching throughput:**
$$\text{Throughput} = \frac{\text{Batch size}}{\text{Batch latency}}$$

Example:
```
Batch 1:  50ms latency → 20 QPS
Batch 8:  80ms latency → 100 QPS (5x throughput!)
Batch 32: 200ms latency → 160 QPS (8x throughput!)
```

**Optimal batch size:**
$$b^* = \arg\max_b \frac{b}{L(b)}$$
Where $L(b)$ = latency for batch size $b$.

Typically: $b^* \approx 8-32$ for CPU, $b^* \approx 64-128$ for GPU.

---

## 🏭 Production: Serving at Scale

### Meta's Feed Ranking (billions of requests/day)

**Architecture:**
```
User request → Load balancer → TorchServe cluster (100+ machines)
                                   ↓
                          [Model: Rank 1000 posts]
                                   ↓
                          Return top 10 posts
```

**Optimization stack:**
1. **Quantization:** FP32 → INT8 (4x memory, 3x speed)
2. **ONNX Runtime:** 2x faster than PyTorch
3. **Dynamic batching:** Batch size 32, 50ms max delay
4. **Model pruning:** Remove 30% of parameters (minimal accuracy loss)
5. **CPU optimization:** AVX-512 VNNI instructions (INT8 acceleration)

**Results:**
- **Latency:** p50 = 8ms, p99 = 25ms
- **Throughput:** 50,000 QPS per server (32-core CPU)
- **Cost:** $0.05 per 1M predictions (vs $0.50 without optimization)

### OpenAI's GPT-3 API

**Challenge:** Serve 175B parameter model with <2s latency.

**Architecture:**
```
Request → Router → GPU cluster (A100s)
           ↓
   [Model sharded across 8 GPUs]
           ↓
   Speculative decoding + KV cache
           ↓
   Response (streaming)
```

**Optimizations:**
1. **Model sharding:** Split across 8× A100 80GB GPUs
2. **KV cache:** Cache attention keys/values (50% speedup)
3. **Speculative decoding:** Generate multiple tokens per step
4. **Dynamic batching:** Batch 32-64 requests
5. **FP16 inference:** Mixed precision (2x memory, 1.5x speed)

**Results:**
- **Latency:** ~1-2s for 100 tokens (streaming)
- **Throughput:** ~100K tokens/second across cluster
- **Cost:** ~$0.002 per 1K tokens

### Google's BERT for Search

**Use case:** Semantic search for billions of queries/day.

**Optimization:**
1. **Distillation:** BERT-large (340M params) → DistilBERT (66M params, 6x faster)
2. **Quantization:** INT8 (4x smaller, 3x faster)
3. **TensorRT:** NVIDIA GPU acceleration (5x faster)
4. **Caching:** Cache embeddings for popular queries (80% hit rate)

**Deployment:**
- **Latency:** <20ms per query (p99)
- **Scale:** Handles 5 billion searches/day
- **Infrastructure:** 100+ GPU servers globally

---

## 🎯 Learning Objectives

By the end of this lab, you will:

**Theory:**
- [ ] Understand research vs production inference tradeoffs
- [ ] Explain quantization (dynamic, static, QAT)
- [ ] Know TorchScript compilation (tracing vs scripting)
- [ ] Analyze latency bottlenecks (memory, compute, I/O)
- [ ] Calculate throughput-latency tradeoffs with batching

**Implementation:**
- [ ] Deploy models with TorchServe
- [ ] Apply INT8 quantization (3-4x speedup)
- [ ] Export models to ONNX
- [ ] Compile to TorchScript
- [ ] Implement custom inference handlers
- [ ] Configure dynamic batching

**Production Skills:**
- [ ] Achieve <50ms p99 latency on CPU
- [ ] Handle 1000+ QPS on single machine
- [ ] Monitor serving metrics (latency, throughput, errors)
- [ ] Debug production inference issues
- [ ] Optimize cost (CPU vs GPU, batching, quantization)

---

## 💻 Exercises

### Exercise 1: Deploy Your First Model with TorchServe (45 mins)

**What You'll Learn:**
- Creating model archives (.mar)
- Starting TorchServe and registering models
- Making inference requests (HTTP API)
- Monitoring with metrics API

**Why It Matters:**
TorchServe is Meta's production-grade serving framework used by companies like Meta, Walmart, and Adobe. Mastering it is essential for deploying PyTorch models at scale.

**Task:** Deploy ResNet50 and serve image classification requests.

**Steps:**
```bash
# 1. Install TorchServe
pip install torchserve torch-model-archiver

# 2. Download pre-trained ResNet50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth

# 3. Create model archive
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet50-19c8e357.pth \
  --handler image_classifier \
  --extra-files index_to_name.json

# 4. Start TorchServe
mkdir model_store
mv resnet50.mar model_store/
torchserve --start \
  --model-store model_store \
  --models resnet50=resnet50.mar \
  --ncs

# 5. Test inference
curl http://localhost:8080/predictions/resnet50 -T kitten.jpg

# 6. Check metrics
curl http://localhost:8082/metrics
```

**Expected output:**
```json
{
  "tiger_cat": 0.9234,
  "tabby": 0.0612,
  "Egyptian_cat": 0.0123
}
```

### Exercise 2: Quantize a Model (INT8) (60 mins)

**What You'll Learn:**
- Applying dynamic quantization
- Static quantization with calibration
- Measuring speedup and accuracy loss
- Comparing FP32 vs INT8 performance

**Why It Matters:**
Quantization delivers 3-4x speedup and 4x memory reduction with minimal accuracy loss. Essential for CPU inference and deploying models on edge devices.

**Task:** Quantize ResNet50 and benchmark performance.

**Implementation:**
```python
import torch
import torchvision.models as models
import time

# Load pre-trained model
model_fp32 = models.resnet50(pretrained=True)
model_fp32.eval()

# Prepare input
dummy_input = torch.randn(1, 3, 224, 224)

# Benchmark FP32
start = time.time()
for _ in range(100):
    with torch.no_grad():
        output_fp32 = model_fp32(dummy_input)
time_fp32 = (time.time() - start) / 100

# Dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Benchmark INT8
start = time.time()
for _ in range(100):
    with torch.no_grad():
        output_int8 = model_int8(dummy_input)
time_int8 = (time.time() - start) / 100

# Compare
print(f"FP32 latency: {time_fp32*1000:.2f}ms")
print(f"INT8 latency: {time_int8*1000:.2f}ms")
print(f"Speedup: {time_fp32/time_int8:.2f}x")

# Check model size
fp32_size = sum(p.numel() * p.element_size() for p in model_fp32.parameters()) / 1e6
int8_size = sum(p.numel() * p.element_size() for p in model_int8.parameters()) / 1e6
print(f"FP32 size: {fp32_size:.2f} MB")
print(f"INT8 size: {int8_size:.2f} MB")
print(f"Compression: {fp32_size/int8_size:.2f}x")

# Accuracy comparison (on test data)
accuracy_fp32 = evaluate(model_fp32, test_loader)
accuracy_int8 = evaluate(model_int8, test_loader)
print(f"FP32 accuracy: {accuracy_fp32:.2%}")
print(f"INT8 accuracy: {accuracy_int8:.2%}")
print(f"Accuracy loss: {accuracy_fp32 - accuracy_int8:.2%}")
```

**Expected results:**
- Speedup: 2-4x on CPU
- Compression: 4x smaller
- Accuracy loss: <1%

### Exercise 3: Export to ONNX and Optimize (45 mins)

**What You'll Learn:**
- Exporting PyTorch models to ONNX
- Running inference with ONNX Runtime
- Comparing PyTorch vs ONNX performance
- Visualizing ONNX graph

**Why It Matters:**
ONNX Runtime often delivers 2-5x speedup over PyTorch inference. It's hardware-agnostic and integrates with TensorRT (NVIDIA), OpenVINO (Intel), and other accelerators.

**Task:** Export ResNet50 to ONNX and benchmark.

**Implementation:**
```python
import torch
import torch.onnx
import onnx
import onnxruntime as ort

# Export to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify ONNX model
onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model verified!")

# Inference with ONNX Runtime
session = ort.InferenceSession("resnet50.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

# Benchmark
import time

# PyTorch
start = time.time()
for _ in range(100):
    with torch.no_grad():
        pytorch_output = model(dummy_input)
pytorch_time = (time.time() - start) / 100

# ONNX Runtime
input_name = session.get_inputs()[0].name
start = time.time()
for _ in range(100):
    onnx_output = session.run(None, {input_name: to_numpy(dummy_input)})
onnx_time = (time.time() - start) / 100

print(f"PyTorch latency: {pytorch_time*1000:.2f}ms")
print(f"ONNX latency: {onnx_time*1000:.2f}ms")
print(f"Speedup: {pytorch_time/onnx_time:.2f}x")

# Verify outputs match
import numpy as np
np.testing.assert_allclose(
    to_numpy(pytorch_output),
    onnx_output[0],
    rtol=1e-3,
    atol=1e-5
)
print("Outputs match!")
```

**Expected results:**
- ONNX Runtime: 1.5-3x faster than PyTorch
- Outputs identical (within numerical precision)

### Exercise 4: Optimize for Low Latency (<50ms p99) (60 mins)

**What You'll Learn:**
- Profiling inference latency
- Identifying bottlenecks (I/O, preprocessing, model, postprocessing)
- Applying optimizations systematically
- Achieving production SLA (p99 <50ms)

**Why It Matters:**
Production systems have strict latency SLAs (e.g., <50ms p99). This exercise teaches you to meet those requirements through systematic optimization.

**Task:** Optimize an image classifier to meet <50ms p99 latency.

**Baseline:**
```python
import time
import numpy as np

class ImageClassifier:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        # Decode image
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocess
        probs = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

        return top5_prob, top5_idx

# Benchmark
classifier = ImageClassifier()
latencies = []

for _ in range(1000):
    image_bytes = load_random_image()
    start = time.time()
    classifier.predict(image_bytes)
    latencies.append((time.time() - start) * 1000)

print(f"p50: {np.percentile(latencies, 50):.2f}ms")
print(f"p99: {np.percentile(latencies, 99):.2f}ms")
# p50: 85ms, p99: 120ms ← Too slow!
```

**Optimizations:**
```python
# Optimization 1: Quantize model
self.model = torch.quantization.quantize_dynamic(
    self.model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
# p50: 40ms, p99: 65ms

# Optimization 2: Compile to TorchScript
self.model = torch.jit.trace(self.model, torch.randn(1, 3, 224, 224))
# p50: 30ms, p99: 50ms

# Optimization 3: Optimize preprocessing
# Use Pillow-SIMD (faster image decoding)
pip install pillow-simd
# p50: 25ms, p99: 45ms ← Success!

# Optimization 4: Export to ONNX Runtime
session = ort.InferenceSession("resnet50.onnx")
# p50: 18ms, p99: 35ms ← Even better!
```

**Final results:**
- p50: 18ms (4.7x improvement)
- p99: 35ms (3.4x improvement, <50ms target met!)

### Exercise 5: Production Serving with Dynamic Batching (90 mins)

**What You'll Learn:**
- Configuring TorchServe for high throughput
- Implementing dynamic batching
- Monitoring QPS and latency
- Load testing with Apache Bench

**Why It Matters:**
Dynamic batching is the key to high throughput. This exercise simulates production serving at 1000+ QPS with strict latency requirements.

**Task:** Deploy model with dynamic batching and achieve 1000 QPS.

**TorchServe config (config.properties):**
```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Dynamic batching
batch_size=32
max_batch_delay=50
default_workers_per_model=4

# Performance
max_request_size=1000000
number_of_netty_threads=8
job_queue_size=1000
```

**Custom handler with batching:**
```python
class BatchedHandler(BaseHandler):
    def handle(self, data, context):
        """Process batch of requests."""
        # Preprocess batch
        batch = self.preprocess(data)

        # Batch inference (32 requests at once!)
        predictions = self.inference(batch)

        # Postprocess batch
        results = self.postprocess(predictions)

        return results

    def preprocess(self, data):
        """Preprocess batch of images."""
        images = []
        for request in data:
            image = self.image_processing(request.get("data"))
            images.append(image)
        return torch.stack(images)  # Shape: [32, 3, 224, 224]

    def inference(self, batch):
        """Batch inference."""
        with torch.no_grad():
            return self.model(batch)  # Process all 32 at once!
```

**Load testing:**
```bash
# Start TorchServe with config
torchserve --start \
  --model-store model_store \
  --models resnet50=resnet50.mar \
  --ts-config config.properties

# Load test with Apache Bench
ab -n 10000 -c 100 \
  -p image.json \
  -T application/json \
  http://localhost:8080/predictions/resnet50

# Results:
# Requests per second: 1250 QPS
# Mean latency: 80ms
# p50: 65ms, p99: 120ms
```

**Monitor metrics:**
```bash
# TorchServe metrics
curl http://localhost:8082/metrics

# Key metrics:
# - QueriesHandled: 10000
# - QueueTime: 25ms (waiting in queue)
# - InferenceTime: 40ms (model inference)
# - Batch size: 28 (average)
```

**Success criteria:**
- ✓ QPS >= 1000
- ✓ p99 latency <150ms (with batching overhead acceptable)
- ✓ No errors or timeouts

---

## ⚠️ Common Pitfalls

### 1. Quantizing Without Calibration

**Symptom:** INT8 model has 5-10% accuracy loss (too high!).

**Cause:** Using dynamic quantization instead of static/QAT.

**Solution:**
```python
# Use static quantization with calibration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Calibrate with representative data (important!)
with torch.no_grad():
    for batch in calibration_loader:  # 1000+ samples
        model_prepared(batch)

model_int8 = torch.quantization.convert(model_prepared)
# Accuracy loss: <1%
```

### 2. Tracing Models with Control Flow

**Symptom:**
```
RuntimeError: Tracing failed: control flow detected
```

**Cause:** Model has `if`, `for`, `while` statements.

**Solution:**
```python
# Use scripting instead of tracing
scripted_model = torch.jit.script(model)
# OR: Annotate control flow
@torch.jit.script
def forward(x):
    if x.sum() > 0:
        ...
```

### 3. Ignoring Preprocessing Latency

**Symptom:** Model inference is 10ms, but end-to-end is 80ms.

**Cause:** Image decoding/preprocessing is slow.

**Solution:**
```python
# Profile each component
with profiler.profile():
    decode = decode_image(bytes)        # 40ms!
    preprocess = transform(decode)      # 30ms!
    inference = model(preprocess)       # 10ms

# Optimize preprocessing:
# - Use Pillow-SIMD for faster decoding
# - Resize on GPU if available
# - Cache preprocessed images
```

### 4. Suboptimal Batch Size

**Symptom:** Throughput plateaus at 500 QPS, could be higher.

**Cause:** Batch size too small (underutilizes hardware).

**Solution:**
```python
# Sweep batch sizes
for batch_size in [1, 4, 8, 16, 32, 64]:
    throughput = benchmark(batch_size)
    print(f"Batch {batch_size}: {throughput} QPS")

# Choose optimal (usually 16-32 for CPU, 64-128 for GPU)
```

### 5. Not Monitoring Production Metrics

**Symptom:** Latency degrades over time, but team doesn't notice until users complain.

**Cause:** No monitoring/alerting.

**Solution:**
```python
# Log metrics to Prometheus/Datadog
from prometheus_client import Histogram

latency_histogram = Histogram('inference_latency_seconds', 'Inference latency')

@latency_histogram.time()
def predict(input_data):
    return model(input_data)

# Alert if p99 > 100ms for 5 minutes
```

---

## 🏆 Expert Checklist for Mastery

**Foundations:**
- [ ] Understand research vs production inference differences
- [ ] Explain quantization (dynamic, static, QAT)
- [ ] Know TorchScript compilation methods (trace vs script)
- [ ] Analyze latency bottlenecks systematically

**Implementation:**
- [ ] Deployed model with TorchServe
- [ ] Applied INT8 quantization with <1% accuracy loss
- [ ] Exported model to ONNX
- [ ] Compiled to TorchScript
- [ ] Implemented custom handler with batching

**Production:**
- [ ] Achieved <50ms p99 latency
- [ ] Handled 1000+ QPS on single machine
- [ ] Monitored serving metrics (latency, QPS, errors)
- [ ] Debugged production inference issues
- [ ] Optimized cost (quantization, batching, CPU vs GPU)

**Advanced:**
- [ ] Integrated TensorRT for GPU acceleration
- [ ] Applied model pruning for further speedup
- [ ] Implemented A/B testing for model versions
- [ ] Built auto-scaling inference clusters
- [ ] Optimized for edge devices (mobile, IoT)

---

## 🚀 Next Steps

After mastering model serving:

1. **Advanced Optimization**
   - Pruning: Remove 50-90% of weights
   - Distillation: Compress large models → small models
   - Neural architecture search (NAS) for efficient models

2. **Hardware Acceleration**
   - TensorRT (NVIDIA GPUs): 5-10x speedup
   - OpenVINO (Intel CPUs): 3-5x speedup
   - CoreML (Apple devices): On-device inference

3. **Distributed Serving**
   - Model sharding (GPT-3 scale)
   - Load balancing across data centers
   - Kubernetes + KServe deployment

4. **Edge Deployment**
   - Mobile (TensorFlow Lite, PyTorch Mobile)
   - IoT (ONNX Runtime for ARM)
   - Browser (ONNX.js, TensorFlow.js)

---

## 📚 References

**Documentation:**
- [TorchServe Official Docs](https://pytorch.org/serve/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime](https://onnxruntime.ai/)

**Papers:**
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Google, 2018)](https://arxiv.org/abs/1712.05877)
- [Mixed Precision Training (NVIDIA, 2018)](https://arxiv.org/abs/1710.03740)

**Tools:**
- [TensorRT (NVIDIA)](https://developer.nvidia.com/tensorrt)
- [OpenVINO (Intel)](https://docs.openvino.ai/)
- [Apache TVM](https://tvm.apache.org/)

---

## 🎯 Solution

Complete implementation: `solution/model_serving.py`

**What you'll build:**
- TorchServe deployment pipeline
- Quantized model (INT8) with <1% accuracy loss
- ONNX export and optimization
- Custom handler with dynamic batching
- Latency optimization (<50ms p99)
- Production monitoring and metrics

**Next: Lab 5 - Knowledge Distillation!**
