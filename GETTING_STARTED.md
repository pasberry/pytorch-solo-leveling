# Getting Started with PyTorch Solo Leveling 🚀

Welcome! This guide will help you set up your environment and start your journey from PyTorch beginner to production ML expert.

---

## 🎯 Prerequisites

Before starting, make sure you have:
- **Python 3.8 or higher**
- **Basic Python programming knowledge**
- **Basic understanding of:**
  - Linear algebra (vectors, matrices)
  - Calculus (derivatives, chain rule)
  - Machine learning concepts (helpful but not required)
- **Hardware:**
  - Minimum: CPU with 8GB RAM
  - Recommended: GPU with 8GB+ VRAM (CUDA-compatible)
- **Enthusiasm to learn!** 🔥

---

## ⚡ Quick Setup (5 minutes)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd pytorch-solo-leveling
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (if you have CUDA)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

You should see:
```
PyTorch 2.1.0+cpu (or cu118)
CUDA available: True (or False)
```

---

## 📚 Understanding the Curriculum

### Structure

The curriculum is divided into **4 phases**:

1. **Phase 1: Foundations (Weeks 1-3)**
   - Learn PyTorch basics
   - Build your first neural networks
   - Complete 6 labs + checkpoint project

2. **Phase 2: Intermediate (Weeks 4-7)**
   - Master Transformers and modern architectures
   - Distributed training and optimization
   - Complete 6 labs + checkpoint project

3. **Phase 3: Applied Projects (Weeks 8-16)**
   - Build 5 production-scale ML systems
   - Feed ranking, video understanding, embeddings, speech, and bonus projects

4. **Phase 4: Expert Topics (Weeks 17-20)**
   - Advanced distributed training
   - Production ML systems
   - Fairness, monitoring, deployment

### Learning Flow

```
Read Theory → Complete Lab → Get Feedback → Move to Next Lab
                                   ↓
                          After all labs: Checkpoint Project
                                   ↓
                              Pass Quiz → Next Phase
```

---

## 🎮 Your First Steps

### Step 1: Read the Full Curriculum (10 mins)
```bash
cat CURRICULUM.md
```
Or open in your favorite editor.

### Step 2: Understand Your Progress Tracker (5 mins)
```bash
cat progress.json
```
This JSON file tracks your progress through the curriculum. Update it as you complete each lab/project.

### Step 3: Start Phase 1 (Now!)
```bash
cd phase1_foundations
cat README.md
```

Read the Phase 1 overview, then dive into Lab 1:
```bash
cd lab1_tensors
cat README.md
```

### Step 4: Complete Lab 1 (1-2 hours)
Work through the exercises:
- `01_tensor_basics.py`
- `02_tensor_operations.py`
- `03_autograd_basics.py`
- `04_linear_regression.py`

Solutions are available in `solution/` directory, but **try yourself first!**

### Step 5: Run the Code
```bash
cd solution
python 01_tensor_basics.py
python 02_tensor_operations.py
python 03_autograd_basics.py
python 04_linear_regression.py
```

---

## 💡 Learning Tips

### Do's ✅
- **Code along:** Don't just read, implement everything
- **Experiment:** Try variations, break things, learn from errors
- **Ask questions:** Use comments to ask "why" and "how"
- **Review regularly:** Revisit earlier labs to solidify understanding
- **Update progress.json:** Track your journey
- **Take notes:** Document learnings and challenges

### Don'ts ❌
- **Don't skip labs:** Each builds on the previous one
- **Don't copy-paste:** Type out code to build muscle memory
- **Don't rush:** Understanding > Speed
- **Don't skip theory:** Know the "why" behind the "how"
- **Don't work in isolation:** Get feedback after each milestone

---

## 🤝 Getting Help

### If You're Stuck:
1. **Read error messages carefully** - They often tell you exactly what's wrong
2. **Check the solution code** - But understand it, don't just copy
3. **Review the theory brief** - Make sure you understand concepts
4. **Ask specific questions** - "Why does X happen?" rather than "It doesn't work"
5. **Search PyTorch docs** - https://pytorch.org/docs/

### Resources:
- **PyTorch Documentation:** https://pytorch.org/docs/
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **PyTorch Forums:** https://discuss.pytorch.org/
- **Papers with Code:** https://paperswithcode.com/

---

## 📊 Tracking Your Progress

After each lab/project:

1. **Update progress.json:**
```json
{
  "lab1_tensors": {
    "status": "completed",
    "exercises_completed": ["01", "02", "03", "04"],
    "notes": "Learned about autograd and computational graphs. Need to review broadcasting."
  }
}
```

2. **Record learnings:**
- What did you learn?
- What was challenging?
- What do you want to explore more?

3. **Note achievements:**
- First working model
- First GPU training
- First distributed training run

---

## 🎯 Success Metrics

You're making progress when you can:

**After Phase 1:**
- [ ] Create and manipulate tensors
- [ ] Explain autograd and computational graphs
- [ ] Build custom nn.Module layers
- [ ] Train models on GPU
- [ ] Implement a CNN from scratch

**After Phase 2:**
- [ ] Implement Transformer from scratch
- [ ] Use distributed training (DDP/FSDP)
- [ ] Export models (ONNX, TorchScript)
- [ ] Optimize for production inference

**After Phase 3:**
- [ ] Build production-scale ranking systems
- [ ] Process and model video data
- [ ] Implement efficient retrieval with embeddings
- [ ] Build speech recognition pipelines

**After Phase 4:**
- [ ] Deploy production ML systems
- [ ] Handle fairness and bias
- [ ] Monitor and maintain models
- [ ] Contribute to production codebases

---

## 🚀 Ready to Start?

You're all set! Here's your first task:

```bash
cd phase1_foundations/lab1_tensors
python solution/01_tensor_basics.py
```

After running it, **implement it yourself** in the `starter/` directory!

---

## 📅 Suggested Schedule

**Full-Time (40 hrs/week):** Complete in 10-12 weeks
**Part-Time (10 hrs/week):** Complete in 20-24 weeks
**Casual (5 hrs/week):** Complete in 40-48 weeks

**Sample Week 1 Schedule:**
- Monday (2h): Lab 1 - Tensors basics
- Tuesday (2h): Lab 1 - Operations and autograd
- Wednesday (2h): Lab 1 - Linear regression
- Thursday (2h): Lab 2 - Custom modules
- Friday (2h): Lab 3 - Training loop

---

## 🏆 Final Words

**Remember:**
- Everyone starts as a beginner
- Progress > Perfection
- Consistency > Intensity
- Understanding > Speed

**"The journey of a thousand miles begins with a single step."** - Lao Tzu

You're about to take that step. Ready to level up? 🔥

**Let's go! Type "start" when you're ready to begin Phase 1!**
