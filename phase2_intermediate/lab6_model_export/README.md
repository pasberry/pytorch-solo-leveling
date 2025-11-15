# Lab 6: Model Export and Deployment - From Research to Production 🚢

> **Time:** 3-4 hours
> **Difficulty:** Intermediate
> **Goal:** Master model export, optimization, and deployment for production

---

## 📖 Why This Lab Matters

Training a model is just the beginning. Deploying to production requires optimization and conversion:

- **Tesla Autopilot** - TorchScript for real-time inference on car hardware
- **Meta** - ONNX for cross-platform ML (PyTorch → Caffe2/ONNX Runtime)
- **OpenAI API** - Quantized models for cost-effective serving
- **Mobile apps** - Pruned/quantized models for edge devices
- **Cloud serving** - Optimized inference for millions of requests/day

**Production requirements:**
- **Speed:** Millisecond latency (research: seconds)
- **Size:** MB not GB (fit on mobile/edge)
- **Platform:** CPU, mobile, browser (research: GPU)
- **Cost:** Optimize for inference $$$ (training is one-time)

**This lab teaches you to deploy research models to production.**

---

## 🧠 The Big Picture: Research vs Production

### The Problem: Training ≠ Deployment

**Research (training):**
```python
model = MyModel()  # Python
model = model.cuda()  # GPU
output = model(input)  # Eager execution
# Flexible, debuggable, slow
```

**Production (inference):**
```
Requirements:
- No Python dependency (embedded systems)
- Run on CPU/mobile (no GPU)
- Fast (<10ms latency)
- Small (<100MB size)
- Efficient (low cost)
```

**The gap:**
| Aspect | Research | Production |
|--------|----------|------------|
| Language | Python | C++/Java/JS |
| Hardware | GPU | CPU/Mobile/Edge |
| Speed | Seconds | Milliseconds |
| Size | GBs | MBs |
| Flexibility | High | Low |
| Cost | Training $$ | Inference $$$ |

---

## 🔬 Deep Dive: Export and Optimization Strategies

### 1. TorchScript: Python-Free PyTorch

**What:** Convert PyTorch models to intermediate representation (IR) that runs without Python.

**Why needed:**
```python
# Problem: Python dependency
model = torch.load('model.pth')  # Requires Python!
output = model(input)  # Requires Python interpreter

# Can't deploy to:
- C++ applications
- Mobile (iOS/Android)
- Embedded systems
- Production servers (want C++ for speed)
```

**Solution: TorchScript**
```
PyTorch Model (Python) → TorchScript (IR) → C++ Runtime
                         ↑
                    No Python needed!
```

**Two ways to create TorchScript:**

**1. Tracing:** Record operations on example input

```python
import torch

model = MyModel()
example_input = torch.randn(1, 3, 224, 224)

# Trace model
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("model_traced.pt")

# Load in Python (no training code needed!)
loaded = torch.jit.load("model_traced.pt")
output = loaded(example_input)

# Load in C++!
```

**How tracing works:**
```
Input → Model → Output
  ↓       ↓       ↓
Track all operations:
  conv2d(input, weight, bias)
  relu(x)
  maxpool(x)
  ...

Save as computation graph
```

**Limitations of tracing:**
```python
# ❌ Control flow not captured!
class ModelWithIf(nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Tracing only sees ONE path!
            return x * 2
        else:
            return x * 3

# Trace with positive input → only "x * 2" path saved
# Will fail on negative inputs!
```

**2. Scripting:** Compile Python code directly

```python
# Annotate with @torch.jit.script
@torch.jit.script
class ModelWithIf(nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Control flow preserved!
            return x * 2
        else:
            return x * 3

scripted = torch.jit.script(ModelWithIf())
scripted.save("model_scripted.pt")
```

**Scripting supports:**
- Control flow (if/else, loops)
- Type annotations
- Custom functions
- All of TorchScript's Python subset

**When to use:**
| Method | Use When | Limitations |
|--------|----------|-------------|
| Trace | Simple feed-forward models | No control flow |
| Script | Complex logic, control flow | Requires type annotations |

---

### 2. ONNX: Universal Model Format

**What:** Open Neural Network Exchange - platform-agnostic model format.

**Why needed:**
```
Problem: Platform lock-in
  Train in PyTorch → Deploy in PyTorch only

Want:
  Train in PyTorch → Deploy anywhere:
    - ONNX Runtime (C++, fastest)
    - TensorFlow Lite (mobile)
    - CoreML (iOS)
    - TensorRT (NVIDIA GPUs)
    - OpenVINO (Intel CPUs)
```

**ONNX as bridge:**
```
PyTorch  ──┐
TensorFlow ┼──→ ONNX ──→ ONNX Runtime
JAX       ──┘           TensorRT
                        CoreML
                        ...
```

**Export to ONNX:**
```python
import torch.onnx

model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,  # ONNX operator set version
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},  # Variable batch size
        'output': {0: 'batch_size'}
    }
)
```

**Load in ONNX Runtime (super fast!):**
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Run inference (C++ backend, very fast!)
output = session.run(None, {input_name: input_numpy})
```

**Speed comparison:**
```
Model: ResNet-50, batch=1, CPU

PyTorch eager:     100ms
PyTorch JIT:        80ms
ONNX Runtime:       40ms  ← 2.5x faster!
TensorRT (GPU):     5ms   ← 20x faster!
```

---

### 3. Quantization: Reduce Precision

**Idea:** Use INT8 instead of FP32 for weights and activations.

**Memory savings:**
```
FP32 model: 4 bytes/param
  ResNet-50: 25M params × 4 = 100MB

INT8 model: 1 byte/param
  ResNet-50: 25M params × 1 = 25MB  ← 4x smaller!
```

**Speed improvements:**
```
INT8 operations are faster than FP32:
  - CPU: 2-4x faster
  - Mobile: 3-5x faster
  - Specialized hardware (TPU): 10x+ faster
```

**Types of quantization:**

**1. Post-Training Quantization (PTQ):**
```python
# Train model normally in FP32
model = train_model()

# Quantize after training (no retraining!)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Which layers to quantize
    dtype=torch.qint8
)

# 4x smaller, 2-4x faster!
```

**2. Quantization-Aware Training (QAT):**
```python
# Simulate quantization during training
model = MyModel()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train (learns to be robust to quantization!)
train(model_prepared)

# Convert to quantized
quantized_model = torch.quantization.convert(model_prepared)
```

**Accuracy comparison:**
```
Original FP32:               76.5% accuracy
Post-Training Quantization:  76.0% accuracy  ← 0.5% drop
Quantization-Aware Training: 76.4% accuracy  ← 0.1% drop!
```

**When to use:**
| Method | Speed | Accuracy | Use When |
|--------|-------|----------|----------|
| PTQ | Fast (no training) | Good (-0.5%) | Most cases |
| QAT | Slow (retrain) | Best (-0.1%) | Accuracy critical |

---

### 4. Pruning: Remove Unnecessary Weights

**Idea:** Set small weights to zero, remove them.

**Why it works:**
```
Neural networks are over-parameterized:
  Many weights ≈ 0
  Can remove 50-90% with minimal accuracy loss!
```

**Pruning methods:**

**1. Magnitude Pruning:** Remove smallest weights
```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in conv layer
prune.l1_unstructured(
    module.conv1,
    name='weight',
    amount=0.3  # Remove 30%
)

# Weights with smallest magnitude → 0
```

**2. Structured Pruning:** Remove entire channels/filters
```python
# Remove entire filters (better hardware utilization)
prune.ln_structured(
    module.conv1,
    name='weight',
    amount=0.3,
    n=2,
    dim=0  # Prune output channels
)

# Reduces actual model size (not just zeros)
```

**Pruning + Fine-tuning:**
```python
# 1. Prune
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5  # Remove 50% of weights!
)

# 2. Fine-tune (recover accuracy)
train(model, epochs=10)

# 3. Make permanent
for module, name in parameters_to_prune:
    prune.remove(module, name)
```

**Accuracy vs sparsity:**
```
Sparsity  |  Accuracy
0%        |  76.5%  (original)
30%       |  76.3%  (-0.2%)
50%       |  75.8%  (-0.7%)
70%       |  74.0%  (-2.5%)
90%       |  65.0%  (-11.5%)  ← Too much!

Sweet spot: 50-70% sparsity
```

---

### 5. Knowledge Distillation: Train Smaller Model

**Idea:** Large "teacher" model teaches small "student" model.

```
Teacher (large, accurate):  100M params, 80% accuracy
                ↓ distill
Student (small, fast):       10M params, 78% accuracy

10x smaller, 2% accuracy drop!
```

**How it works:**
```python
# Teacher: pre-trained large model
teacher = LargeModel()  # 100M params
teacher.eval()

# Student: small model
student = SmallModel()  # 10M params

# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, T=3.0):
    # Soft targets from teacher (smoothed probabilities)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)

    # KL divergence (match teacher distribution)
    distill_loss = F.kl_div(student_log_probs, teacher_probs,
                           reduction='batchmean') * (T * T)

    # Hard targets (true labels)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combine
    return 0.7 * distill_loss + 0.3 * hard_loss

# Train student
for batch in dataloader:
    student_out = student(batch)
    teacher_out = teacher(batch)
    loss = distillation_loss(student_out, teacher_out, labels)
    loss.backward()
    optimizer.step()
```

**Why temperature T:**
```
T = 1:  Hard probabilities  [0.9, 0.05, 0.05]
T = 3:  Soft probabilities  [0.6, 0.25, 0.15]
        ↑ More information! Student learns from mistakes
```

**Use cases:**
- Mobile deployment (need small model)
- Edge devices (limited memory)
- Ensemble → single model
- Dark knowledge transfer

---

## 🎯 Learning Objectives

**Theoretical Understanding:**
- Research vs production requirements
- TorchScript tracing vs scripting
- ONNX cross-platform deployment
- Quantization mathematics (INT8 vs FP32)
- Pruning structured vs unstructured
- Knowledge distillation theory
- Latency vs throughput tradeoffs
- Model serving at scale

**Practical Skills:**
- Export model to TorchScript
- Convert to ONNX format
- Apply post-training quantization
- Implement quantization-aware training
- Prune neural networks
- Distill large model to small model
- Measure inference speed and size
- Deploy model to production

---

## 🔑 Key Concepts

### 1. Latency vs Throughput

**Latency:** Time for single request
```
User request → Model → Response
     ↑____________↓
     100ms latency

Important for: Real-time systems (chatbots, autocomplete)
```

**Throughput:** Requests per second
```
1000 users → Model (batch) → 1000 responses
            100ms total
            = 10,000 requests/second

Important for: Batch processing (recommendation, ranking)
```

**Optimization strategies:**
```
Reduce latency:
  - Quantization (faster compute)
  - Pruning (less compute)
  - Smaller models

Increase throughput:
  - Batching (process multiple requests together)
  - GPU serving (parallel processing)
  - Model parallelism
```

### 2. Model Serving Patterns

**Pattern 1: REST API**
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load("model.pt")

@app.post("/predict")
def predict(input: Input):
    with torch.no_grad():
        output = model(input.data)
    return {"prediction": output.tolist()}
```

**Pattern 2: Batch Inference**
```python
# Collect requests for 100ms
requests_batch = collect_requests(timeout=0.1)

# Process batch
with torch.no_grad():
    predictions = model(torch.stack(requests_batch))

# Return to users
for request, pred in zip(requests_batch, predictions):
    request.respond(pred)
```

**Pattern 3: Model Caching**
```python
# Cache common inputs
cache = {}

def predict(input):
    key = hash(input)
    if key in cache:
        return cache[key]  # Instant!

    output = model(input)
    cache[key] = output
    return output
```

### 3. Deployment Platforms

| Platform | Use Case | Format | Hardware |
|----------|----------|--------|----------|
| ONNX Runtime | Production CPU inference | ONNX | CPU |
| TensorRT | GPU serving (NVIDIA) | ONNX/TorchScript | GPU |
| TorchServe | PyTorch serving | TorchScript | CPU/GPU |
| CoreML | iOS apps | CoreML | Apple |
| TFLite | Android/mobile | TFLite | ARM |
| OpenVINO | Intel CPUs | ONNX | Intel |

---

## 🧪 Exercises

### Exercise 1: TorchScript Export (45 mins)

**What You'll Learn:**
- Tracing vs scripting differences
- Handling control flow
- Loading in C++ (optional)
- Performance comparison

**Why It Matters:**
TorchScript is production standard for PyTorch. Tesla, Meta, and many others use it for deployment.

**Tasks:**
1. Export model with tracing
2. Export model with scripting
3. Handle control flow correctly
4. Compare trace vs script performance
5. Test with different inputs
6. Measure inference speed

---

### Exercise 2: ONNX Conversion (60 mins)

**What You'll Learn:**
- ONNX export process
- Dynamic axes for variable batch size
- ONNX Runtime inference
- Cross-platform deployment

**Why It Matters:**
ONNX Runtime is often 2-3x faster than PyTorch eager. Used widely in production.

**Tasks:**
1. Export model to ONNX
2. Verify ONNX model correctness
3. Run inference with ONNX Runtime
4. Compare speed (PyTorch vs ONNX)
5. Test dynamic batch sizes
6. Visualize ONNX graph

---

### Exercise 3: Post-Training Quantization (45 mins)

**What You'll Learn:**
- Dynamic quantization
- Static quantization with calibration
- Accuracy vs size trade-off
- Quantization backends (fbgemm, qnnpack)

**Why It Matters:**
4x smaller models, 2-4x faster inference, <1% accuracy drop. Critical for mobile and edge deployment.

**Tasks:**
1. Apply dynamic quantization
2. Apply static quantization
3. Measure model size reduction
4. Measure inference speedup
5. Measure accuracy change
6. Compare quantization backends

**Implementation:**
```python
# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# Static quantization (better accuracy)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Calibrate on representative data
with torch.no_grad():
    for data in calibration_data:
        model_prepared(data)

quantized_model = torch.quantization.convert(model_prepared)
```

---

### Exercise 4: Model Pruning (60 mins)

**What You'll Learn:**
- Unstructured vs structured pruning
- Global vs local pruning
- Iterative pruning with fine-tuning
- Sparsity patterns

**Why It Matters:**
50-70% sparsity with minimal accuracy loss. Critical for edge devices and cost optimization.

**Tasks:**
1. Implement magnitude pruning
2. Apply structured pruning
3. Iteratively prune and fine-tune
4. Measure accuracy vs sparsity curve
5. Remove pruned weights permanently
6. Combine pruning + quantization

---

### Exercise 5: Knowledge Distillation (90 mins)

**What You'll Learn:**
- Teacher-student training
- Temperature scaling
- Soft vs hard targets
- Compression ratios

**Why It Matters:**
10x smaller models with only 2% accuracy drop. How production models are compressed (BERT → DistilBERT).

**Tasks:**
1. Train large teacher model
2. Create small student model
3. Implement distillation loss
4. Train student with distillation
5. Compare student vs teacher accuracy
6. Measure size and speed improvements

---

### Exercise 6: End-to-End Deployment (120 mins)

**What You'll Learn:**
- Complete deployment pipeline
- Model serving with FastAPI
- Batch inference
- Monitoring and logging

**Why It Matters:**
This is the full production workflow. Understanding this means you can deploy models to real systems.

**Tasks:**
1. Optimize model (quantize + prune)
2. Export to TorchScript/ONNX
3. Build REST API with FastAPI
4. Implement batching
5. Add monitoring
6. Load test and measure latency/throughput

**Starter code:**
```python
from fastapi import FastAPI
import torch
import asyncio

app = FastAPI()
model = torch.jit.load("optimized_model.pt")
request_queue = []

@app.on_event("startup")
async def startup():
    # Start batch inference worker
    asyncio.create_task(batch_worker())

async def batch_worker():
    while True:
        await asyncio.sleep(0.1)  # Wait 100ms
        if request_queue:
            process_batch(request_queue)

@app.post("/predict")
async def predict(data: Input):
    # Add to batch queue
    future = asyncio.Future()
    request_queue.append((data, future))
    return await future
```

---

## 📝 Design Patterns

### Pattern 1: TorchScript Export

```python
# Tracing for simple models
model = SimpleModel()
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)
traced.save("model_traced.pt")

# Scripting for complex models
@torch.jit.script
class ComplexModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # Control flow preserved
            return x * 2
        return x * 3

scripted = torch.jit.script(ComplexModel())
scripted.save("model_scripted.pt")
```

### Pattern 2: Quantization Pipeline

```python
# 1. Post-training quantization (quick)
quantized = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# 2. Quantization-aware training (best)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model = torch.quantization.prepare_qat(model)
train(model)  # Train with fake quantization
quantized = torch.quantization.convert(model)

# 3. Save and deploy
torch.jit.save(torch.jit.script(quantized), "quantized.pt")
```

### Pattern 3: Iterative Pruning

```python
import torch.nn.utils.prune as prune

# Define pruning strategy
parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.fc1, 'weight'),
]

for iteration in range(5):
    # Prune 20% each iteration
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2
    )

    # Fine-tune
    train(model, epochs=2)

    # Evaluate
    accuracy = evaluate(model)
    print(f"Iteration {iteration}: {accuracy:.2f}%")

# Make permanent
for module, name in parameters_to_prune:
    prune.remove(module, name)
```

---

## ✅ Solutions

Complete implementations in `solution/` directory.

**Files:**
- `01_torchscript.py` - TorchScript export
- `02_onnx.py` - ONNX conversion
- `03_quantization.py` - Quantization techniques
- `04_pruning.py` - Model pruning
- `05_distillation.py` - Knowledge distillation
- `06_deployment.py` - Production deployment

Run examples:
```bash
cd solution
python 01_torchscript.py
python 02_onnx.py
python 03_quantization.py
python 04_pruning.py
python 05_distillation.py
python 06_deployment.py
```

---

## 🎓 Key Takeaways

1. **Training ≠ Deployment** - Different requirements, different optimizations
2. **TorchScript removes Python** - Deploy anywhere (C++, mobile, embedded)
3. **ONNX is universal** - Train once, deploy everywhere
4. **Quantization: 4x smaller, 2-4x faster** - Minimal accuracy loss
5. **Pruning: 50-70% sparsity** - Remove unnecessary weights
6. **Distillation: 10x compression** - Large teacher → small student
7. **Latency vs throughput** - Optimize for your use case
8. **Optimization is iterative** - Quantize → prune → distill

**The Optimization Pipeline:**
```
FP32 Model (100MB, 100ms)
    ↓ quantize
INT8 Model (25MB, 40ms)
    ↓ prune
Sparse Model (12MB, 30ms)
    ↓ distill
Small Model (5MB, 15ms)

20x smaller, 6.7x faster!
```

---

## 🔗 Connections to Production ML

### Why This Matters at Scale

**At Meta:**
- PyTorch → ONNX → Caffe2 deployment
- Billions of inferences/day
- Quantization for cost savings
- Custom ONNX operators for Meta hardware

**At OpenAI (API):**
- Quantized models for cost optimization
- TensorRT for GPU serving
- Batching for throughput
- Cost: Inference >> training

**At Tesla:**
- TorchScript for car deployment
- C++ runtime (no Python overhead)
- Real-time inference (<10ms)
- Pruned models for embedded hardware

**Cost implications:**
```
Serving 1M requests/day:

FP32 model:
  Latency: 100ms
  Hardware: 10 GPUs
  Cost: $20/day × 10 = $200/day

INT8 quantized:
  Latency: 40ms
  Hardware: 4 GPUs
  Cost: $20/day × 4 = $80/day

Savings: $120/day = $43K/year!
```

---

## 🚀 Next Steps

1. **Complete all exercises** - Full deployment pipeline
2. **Deploy real model** - Production serving
3. **Read optimization papers** - Quantization, pruning, distillation
4. **Move to Phase 3** - Advanced architectures and techniques

---

## 💪 Bonus Challenges

1. **Mobile Deployment**
   - Convert to TensorFlow Lite
   - Deploy to Android app
   - Measure on-device performance

2. **TensorRT Optimization**
   - Export to TensorRT
   - Benchmark on NVIDIA GPU
   - Compare to ONNX Runtime

3. **Mixed Precision Inference**
   - FP16 inference on GPU
   - Measure speedup
   - Accuracy comparison

4. **Model Compression Pipeline**
   - Combine quantization + pruning + distillation
   - Measure cumulative savings
   - Track accuracy degradation

5. **A/B Testing Framework**
   - Deploy optimized vs original model
   - Compare latency/accuracy in production
   - Cost-benefit analysis

---

## 📚 Essential Resources

**Papers:**
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al., 2015
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) - Jacob et al., 2017
- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) - Han et al., 2015 (Pruning)
- [ONNX: Open Neural Network Exchange](https://arxiv.org/abs/1907.04464)

**Documentation:**
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

**Tools:**
- [TorchServe](https://pytorch.org/serve/) - Model serving
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Fast inference
- [Netron](https://netron.app/) - Model visualization

---

## 🤔 Common Pitfalls

### Pitfall 1: Forgetting eval() Mode

```python
# ❌ Export in training mode (includes dropout, batchnorm updates)
torch.onnx.export(model, ...)

# ✓ Always export in eval mode
model.eval()
torch.onnx.export(model, ...)
```

### Pitfall 2: Wrong Quantization Backend

```python
# ❌ Using wrong backend for hardware
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # ARM/mobile

# ✓ Use correct backend
# CPU (x86): 'fbgemm'
# Mobile (ARM): 'qnnpack'
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
```

### Pitfall 3: Not Calibrating Static Quantization

```python
# ❌ Quantize without calibration (poor accuracy!)
quantized = torch.quantization.convert(model_prepared)

# ✓ Calibrate on representative data first
for data in calibration_data:
    model_prepared(data)  # Collect statistics
quantized = torch.quantization.convert(model_prepared)
```

### Pitfall 4: Pruning Without Fine-tuning

```python
# ❌ Prune and evaluate immediately (accuracy drop!)
prune.global_unstructured(params, amount=0.5)
accuracy = evaluate(model)  # Low!

# ✓ Fine-tune after pruning
prune.global_unstructured(params, amount=0.5)
train(model, epochs=10)  # Recover accuracy
accuracy = evaluate(model)  # Better!
```

---

## 💡 Pro Tips

1. **Start with ONNX Runtime** - Often easiest 2x speedup
2. **Quantize first** - Biggest bang for buck
3. **Benchmark on target hardware** - CPU/GPU/mobile differ
4. **Use representative data for calibration** - Critical for quantization
5. **Iterative pruning works best** - Prune 20% → fine-tune → repeat
6. **Distillation needs temperature** - T=3 is good start
7. **Monitor accuracy throughout** - Track degradation
8. **Profile before optimizing** - Find real bottlenecks

---

## ✨ You're Ready When...

- [ ] You understand research vs production gap
- [ ] You can export models to TorchScript
- [ ] You've converted models to ONNX
- [ ] You can apply quantization (PTQ and QAT)
- [ ] You've pruned neural networks
- [ ] You understand knowledge distillation
- [ ] You know latency vs throughput tradeoffs
- [ ] You can deploy model with REST API
- [ ] You've measured real speedups and size reductions
- [ ] You understand production deployment patterns

**Remember:** Training is 10% of the work. Deployment and optimization are 90%. Master this lab, and you can ship AI to production!
