# Project 5: Bonus Advanced Projects 🚀

> **Time:** 1-2 weeks per project
> **Difficulty:** Advanced-Expert
> **Goal:** Choose 2-3 advanced projects to expand your ML expertise

---

## 📚 Overview

After completing the core projects (Feed Ranking, Video Understanding, Embeddings, Speech), choose from these advanced topics to deepen your expertise in specific areas.

**Pick 2-3 projects that interest you most!**

---

## 🎯 Available Projects

### 1. CLIP-Style Multimodal Model 🖼️📝
**Goal:** Build contrastive image-text embeddings

**Architecture:**
```
Image Encoder (Vision Transformer / ResNet)
Text Encoder (Transformer)
    ↓
Contrastive Learning (InfoNCE)
    ↓
Shared Embedding Space
```

**What You'll Learn:**
- Vision transformers
- Contrastive learning
- Cross-modal retrieval
- Zero-shot classification

**Dataset:** COCO Captions or Conceptual Captions

**Expected Results:**
- Zero-shot ImageNet: > 30% top-1
- Image-text retrieval: > 60% Recall@10

**Resources:**
- [CLIP Paper (OpenAI, 2021)](https://arxiv.org/abs/2103.00020)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)

---

### 2. Diffusion Model for Image Generation 🎨
**Goal:** Generate high-quality images from noise

**Architecture:**
```
DDPM (Denoising Diffusion Probabilistic Model)

Forward Process:  x0 → x1 → ... → xT (add noise)
Reverse Process:  xT → ... → x1 → x0 (denoise with neural network)
```

**What You'll Learn:**
- Diffusion process (forward & reverse)
- Score-based generative models
- Noise scheduling
- Classifier-free guidance

**Dataset:** CIFAR-10 or CelebA

**Expected Results:**
- FID (Fréchet Inception Distance): < 10 on CIFAR-10
- High-quality samples

**Resources:**
- [DDPM Paper (2020)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

---

### 3. Graph Neural Network (GNN) 🕸️
**Goal:** Learn on graph-structured data (social networks)

**Architecture:**
```
Graph Convolutional Network (GCN)

Input: Graph (nodes + edges)
    ↓
Message Passing (aggregate neighbor features)
    ↓
Node/Graph Embeddings
    ↓
Downstream Task (classification, link prediction)
```

**What You'll Learn:**
- Graph representation learning
- Message passing
- Node/edge/graph-level tasks
- Attention on graphs (GAT)

**Dataset:** Cora, PubMed, or synthetic social network

**Tasks:**
- Node classification
- Link prediction
- Community detection

**Expected Results:**
- Node classification accuracy: > 80% on Cora

**Resources:**
- [GCN Paper (2017)](https://arxiv.org/abs/1609.02907)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

### 4. PPO Reinforcement Learning 🎮
**Goal:** Train agent with policy optimization

**Architecture:**
```
Actor-Critic

Actor (Policy Network):  State → Action Probabilities
Critic (Value Network):  State → Value Estimate
    ↓
PPO Loss (clipped objective)
    ↓
Update Policy
```

**What You'll Learn:**
- Policy gradient methods
- Actor-critic architecture
- PPO (Proximal Policy Optimization)
- Advantage estimation

**Environment:** CartPole, LunarLander, or Atari

**Expected Results:**
- CartPole: solve in < 1000 episodes
- LunarLander: average score > 200

**Resources:**
- [PPO Paper (OpenAI, 2017)](https://arxiv.org/abs/1707.06347)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

---

### 5. Variational Autoencoder (VAE) 🧬
**Goal:** Learn latent representations for generation

**Architecture:**
```
Encoder:  x → μ, σ (latent distribution)
Sampling: z ~ N(μ, σ)
Decoder:  z → x_reconstructed

Loss = Reconstruction Loss + KL Divergence
```

**What You'll Learn:**
- Variational inference
- Latent variable models
- Reparameterization trick
- Disentangled representations (β-VAE)

**Dataset:** MNIST, CIFAR-10, CelebA

**Expected Results:**
- Clear reconstructions
- Smooth latent space interpolations
- Disentangled factors

**Resources:**
- [VAE Paper (2013)](https://arxiv.org/abs/1312.6114)
- [β-VAE (2017)](https://openreview.net/forum?id=Sy2fzU9gl)

---

### 6. Retrieval-Augmented Generation (RAG) 📚
**Goal:** Build LLM with external knowledge retrieval

**Architecture:**
```
Question
    ↓
Retriever (Dense / Sparse)
    ↓
Top-K Relevant Documents
    ↓
Generator (LLM)
    ↓
Answer (conditioned on retrieved docs)
```

**What You'll Learn:**
- Dense retrieval (embeddings)
- Sparse retrieval (BM25)
- Knowledge grounding
- LLM prompting

**Dataset:** Natural Questions, MS MARCO

**Expected Results:**
- Improve LLM accuracy with retrieval
- Better factuality

**Resources:**
- [RAG Paper (Meta, 2020)](https://arxiv.org/abs/2005.11401)
- [LangChain](https://github.com/langchain-ai/langchain)

---

### 7. LLM Finetuning with LoRA 🤖
**Goal:** Efficiently finetune large language models

**Architecture:**
```
Pretrained LLM (frozen)
    +
Low-Rank Adaptation (LoRA)
    ↓
Finetuned Model (only adapt low-rank matrices)
```

**What You'll Learn:**
- Parameter-efficient finetuning
- LoRA (Low-Rank Adaptation)
- Instruction tuning
- PEFT (Parameter-Efficient Fine-Tuning)

**Model:** LLaMA, Mistral, or GPT-2

**Dataset:** Alpaca, Dolly, or custom instruction dataset

**Expected Results:**
- Finetune 7B model on single GPU
- Improved task performance

**Resources:**
- [LoRA Paper (Microsoft, 2021)](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [QLoRA (2023)](https://arxiv.org/abs/2305.14314)

---

## 🎯 How to Choose

**Choose based on your interests:**

- **Computer Vision?** → CLIP, Diffusion, VAE
- **NLP/LLMs?** → RAG, LoRA
- **Graph Data?** → GNN
- **RL/Games?** → PPO
- **Generative Models?** → VAE, Diffusion

**Career-focused:**
- **Recommendation Systems:** All projects (especially GNN for social networks)
- **Generative AI:** Diffusion, VAE, LoRA
- **Search/Retrieval:** CLIP, RAG
- **Autonomous Systems:** PPO

---

## 🏆 Completion Criteria

For each project you choose:

- [ ] Understand the theory and math
- [ ] Implement from scratch (no copy-paste)
- [ ] Train to reasonable performance
- [ ] Experiment with variations
- [ ] Document learnings and challenges
- [ ] Present results (visualizations, metrics)

---

## 🚀 Getting Started

1. **Pick 2-3 projects** that excite you
2. **Read the papers** to understand theory
3. **Find starter code** for reference (but implement yourself)
4. **Start with simple dataset** (MNIST, CIFAR-10)
5. **Scale up** once it works
6. **Experiment** with improvements

---

## 💡 Meta-Scale Considerations

Each project has production considerations:

**CLIP:** Billion-scale image-text pairs, multimodal search
**Diffusion:** Efficient sampling, latent diffusion for speed
**GNN:** Scaling to billions of nodes/edges, graph sampling
**PPO:** Distributed RL, off-policy learning
**VAE:** Large-scale generation, disentanglement
**RAG:** Billion-doc retrieval, efficient indexing
**LoRA:** Multi-task adapters, dynamic adapter selection

---

## 🤝 Support

For each project, you'll receive:
- Detailed implementation guidance
- Code review and feedback
- Debugging help
- Optimization tips
- Ideas for extensions

**Pick your projects and let's build! 🚀**
