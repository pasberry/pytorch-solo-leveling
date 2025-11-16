# Lab 2: Vector Database & Billion-Scale Similarity Search 🔍

> **Time:** 2-3 hours
> **Difficulty:** Expert
> **Goal:** Master FAISS to build production-scale semantic search systems for billion-vector datasets

---

## 📖 Why This Lab Matters

You've built an embedding model. It's trained, it's accurate, and it converts text to 768-dimensional vectors. **Now what?**

You need to **search** those embeddings. Fast. At scale.

**The scale challenge:**
- **Meta's production search:** 10 billion+ user profiles, sub-50ms latency
- **Google's semantic search:** Billions of web pages, real-time queries
- **OpenAI's embedding API:** Millions of document chunks per customer
- **Spotify's music recommendations:** 100M+ songs, instant results

**The naive approach fails:**
```python
# Linear search - O(N) time complexity
def find_similar(query_vector, database_vectors):
    similarities = cosine_similarity(query_vector, database_vectors)  # Compare ALL vectors
    return similarities.argsort()[-10:]  # Top 10

# For 1 billion vectors:
# Time: ~10 seconds per query (unacceptable!)
# Memory: 1B vectors × 768 dims × 4 bytes = 3 TB RAM
```

**You need a vector database.**

This lab teaches you **FAISS (Facebook AI Similarity Search)** - the technology powering:
- **Meta's search infrastructure** (billions of user embeddings)
- **OpenAI's retrieval systems** (ChatGPT plugin ecosystem)
- **Pinecone, Weaviate, Milvus** (built on FAISS algorithms)
- **Google's ScaNN** (similar approximate NN approach)

**Master FAISS, and you can build search systems at internet scale.**

---

## 🧠 The Big Picture: Why Linear Search Fails

### The Search Problem

**Given:**
- Database: N vectors (e.g., 1 billion documents)
- Query: 1 vector (user search query)
- Task: Find K nearest neighbors (top 10 most similar)

**Naive solution (Exact search):**
```python
distances = []
for i in range(N):
    distance = compute_distance(query, database[i])  # O(d) where d = dimensions
    distances.append(distance)

top_k = sorted(distances)[:K]  # O(N log N)
```

**Time complexity: O(N × d)**

**For N=1 billion, d=768:**
```
Operations: 1,000,000,000 × 768 = 768 billion comparisons
Time (on CPU): ~10-30 seconds per query
Memory: 1B × 768 × 4 bytes = 3 TB
```

**Production requirement: <100ms per query**

❌ Linear search is **100-300x too slow**!

### The Evolution of Vector Search

**Stage 1: K-D Trees (1970s-1990s)**
```
Problem: Works for low dimensions (d < 10)
Curse of dimensionality: Performance degrades exponentially with dimensions
For d=768: Worse than linear search!
```

**Stage 2: Locality-Sensitive Hashing (2000s)**
```
Idea: Hash similar vectors to same buckets
Problem: Requires many hash tables for good recall
Memory: 10-100x original dataset size
```

**Stage 3: Inverted File Index (IVF) (2010s)**
```
Idea: Cluster vectors, search only nearest clusters
Speedup: 10-1000x faster than linear
Memory: 1-2x original dataset
Used by: Facebook, Google, Spotify
```

**Stage 4: Graph-Based Search (HNSW) (2016-Present)**
```
Idea: Build navigable small-world graph
Speedup: 100-10,000x faster than linear
Recall: >95% with proper tuning
Used by: Modern vector databases
```

**Stage 5: Product Quantization (2011-Present)**
```
Idea: Compress vectors from 768D × 4 bytes to 768D × 1 byte (or less!)
Memory: 4-64x compression
Speedup: 2-10x from reduced memory bandwidth
Used by: All production systems at scale
```

---

## 🔬 Deep Dive: FAISS Architecture

### FAISS Index Types

FAISS provides multiple index types, each with different **accuracy-speed-memory tradeoffs**.

### 1. IndexFlatL2 (Exact Search)

**Algorithm:** Brute-force linear search with L2 distance.

```python
import faiss
import numpy as np

# Create index
d = 768  # Dimension
index = faiss.IndexFlatL2(d)

# Add vectors
vectors = np.random.random((1000000, d)).astype('float32')
index.add(vectors)

# Search
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k=10)
```

**Characteristics:**
- **Accuracy:** 100% (exact search)
- **Speed:** O(N × d) - slow for large N
- **Memory:** N × d × 4 bytes (full vectors in RAM)
- **Use case:** Small datasets (<1M vectors), baseline comparison

**When to use:**
- Dataset <1 million vectors
- Need 100% recall
- Latency <100ms acceptable

### 2. IndexIVFFlat (Inverted File Index)

**Algorithm:** Cluster vectors using k-means, search only nearest clusters.

**How it works:**
```
Training phase:
1. Cluster N vectors into C centroids using k-means
2. Assign each vector to nearest centroid
3. Build inverted index: centroid_id → list of vectors

Query phase:
1. Find nprobe nearest centroids to query
2. Search only vectors in those nprobe clusters
3. Return top K overall

Speedup: Search only (nprobe/C) × N vectors instead of N
```

**Implementation:**
```python
# Create IVF index
nlist = 1000  # Number of clusters
quantizer = faiss.IndexFlatL2(d)  # Quantizer to find nearest clusters
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train (cluster the data)
index.train(vectors)

# Add vectors
index.add(vectors)

# Search with nprobe parameter
index.nprobe = 10  # Search 10 nearest clusters
distances, indices = index.search(query, k=10)
```

**Parameters:**
- **nlist:** Number of clusters (typical: sqrt(N) to 4*sqrt(N))
- **nprobe:** Clusters to search (typical: 1-100)
- Higher nprobe → better recall, slower search

**Tradeoffs:**
| nprobe | Recall | Search Time | Use Case |
|--------|--------|-------------|----------|
| 1 | 60-70% | 1x (fastest) | Coarse search |
| 10 | 85-90% | 5x | Production default |
| 100 | 95-98% | 30x | High-accuracy requirements |
| nlist | 100% | Same as Flat | Exact (defeats purpose) |

**Example (1M vectors):**
```
nlist = 1000 clusters
Average cluster size: 1M / 1000 = 1000 vectors

With nprobe=10:
  Vectors searched: 10 × 1000 = 10,000 (1% of total!)
  Speedup: ~100x vs linear search
  Recall: ~90%
```

### 3. IndexHNSW (Hierarchical Navigable Small World)

**Algorithm:** Build multi-layer proximity graph for efficient traversal.

**How it works:**
```
Structure: Multi-layer graph
  Layer 2: [sparse, long-range connections]
  Layer 1: [medium density]
  Layer 0: [dense, all vectors]

Search algorithm:
1. Start at top layer (sparse)
2. Greedily navigate to nearest neighbor
3. Move down to next layer
4. Repeat until bottom layer
5. Final refinement at layer 0

Time complexity: O(log N) on average!
```

**Implementation:**
```python
# Create HNSW index
M = 32  # Number of connections per layer
index = faiss.IndexHNSWFlat(d, M)

# Set search parameters
index.hnsw.efConstruction = 40  # Quality of graph construction
index.hnsw.efSearch = 16        # Beam width during search

# Add vectors (no training needed!)
index.add(vectors)

# Search
distances, indices = index.search(query, k=10)
```

**Parameters:**
- **M:** Connections per node (16-64). Higher M → better recall, more memory
- **efConstruction:** Build quality (40-500). Higher → better graph, slower build
- **efSearch:** Search beam width (16-512). Higher → better recall, slower search

**Performance characteristics:**
```
Memory: ~4x more than IVF (graph structure)
Build time: ~10x slower than IVF (graph construction expensive)
Search speed: ~2-5x faster than IVF at same recall
Recall: Excellent (95-99% with proper tuning)

Use when: Memory not constrained, want best search speed
```

### 4. Product Quantization (PQ) for Compression

**Problem:** 1 billion vectors × 768D × 4 bytes = **3 TB memory**

**Solution:** Compress vectors from FP32 to compact codes.

**Algorithm:**
```
1. Split each 768D vector into M sub-vectors (e.g., M=96, each 8D)
2. For each sub-space, cluster into K centroids (e.g., K=256 = 2^8)
3. Replace each 8D sub-vector with 1-byte centroid ID
4. Compressed size: 96 bytes instead of 768×4 = 3072 bytes (32x compression!)

Original vector (768D):
  [0.23, 0.45, ..., 0.12]  → 3072 bytes (768 × 4)

PQ compressed (96 bytes):
  [centroid_id_1, centroid_id_2, ..., centroid_id_96]  → 96 bytes
```

**Implementation:**
```python
# Create IVF + PQ index
nlist = 1000
M = 96  # Number of sub-vectors
nbits = 8  # Bits per sub-vector (256 centroids)

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

# Train (learn clusters AND PQ codebooks)
index.train(vectors)

# Add vectors (stored compressed!)
index.add(vectors)

# Search
index.nprobe = 10
distances, indices = index.search(query, k=10)
```

**Compression tradeoff:**
| Compression | Bytes/Vector | Memory (1B vectors) | Recall | Use Case |
|-------------|--------------|---------------------|--------|----------|
| None (Flat) | 3072 | 3 TB | 100% | Small scale |
| PQ96 | 96 | 96 GB | 90-95% | Typical production |
| PQ64 | 64 | 64 GB | 85-90% | High compression |
| PQ32 | 32 | 32 GB | 75-85% | Extreme compression |

**Example: Meta's production (10B embeddings):**
```
Without PQ: 10B × 3KB = 30 TB (impossible!)
With PQ96:  10B × 96 bytes = 960 GB (fits on 8x A100 80GB!)
Recall:     ~92% (acceptable for production)
```

### 5. GPU Acceleration

**Problem:** Even with IVFPQ, searching 1B vectors takes ~50-100ms on CPU.

**Solution:** Run search on GPU (100-1000x parallelism).

```python
# Use GPU resources
res = faiss.StandardGpuResources()

# Convert CPU index to GPU
cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
cpu_index.train(vectors)

# Transfer to GPU
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Search on GPU (10-100x faster!)
gpu_index.add(vectors)
distances, indices = gpu_index.search(query, k=10)
```

**GPU speedup:**
| Index Type | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| Flat | 10 s | 50 ms | 200x |
| IVF | 100 ms | 5 ms | 20x |
| IVFPQ | 50 ms | 2 ms | 25x |

**Multi-GPU scaling:**
```python
# Shard across 8 GPUs
gpu_indices = []
for i in range(8):
    gpu_indices.append(faiss.index_cpu_to_gpu(res, i, cpu_index))

# Each GPU handles 1/8 of database
# Query all in parallel → 8x speedup!
```

---

## 📊 Mathematical Foundations

### Distance Metrics

**L2 Distance (Euclidean):**
```
d(x, y) = sqrt(Σ(x_i - y_i)²)

Properties:
- Range: [0, ∞)
- Sensitive to magnitude
- Used for: Normalized embeddings, spatial data
```

**Inner Product (Dot Product):**
```
d(x, y) = -Σ(x_i × y_i)  (negative for max-heap)

Properties:
- Range: (-∞, ∞)
- Considers magnitude and angle
- Used for: Non-normalized embeddings
```

**Cosine Similarity:**
```
sim(x, y) = (Σ(x_i × y_i)) / (||x|| × ||y||)
distance = 1 - sim(x, y)

Properties:
- Range: [0, 2] (for distance)
- Magnitude-invariant (only angle matters)
- Used for: Text embeddings, normalized vectors

Note: For normalized vectors, inner product ≡ cosine similarity
```

**FAISS implementation:**
```python
# L2 distance
index_l2 = faiss.IndexFlatL2(d)

# Inner product (for cosine similarity, normalize first)
faiss.normalize_L2(vectors)  # Normalize to unit length
index_ip = faiss.IndexFlatIP(d)  # Inner product
```

### Product Quantization Mathematics

**Goal:** Compress d-dimensional vector to M bytes.

**Algorithm:**
```
1. Split vector into M sub-vectors:
   x = [x₁, x₂, ..., x_M] where each x_i ∈ ℝ^(d/M)

2. For each sub-space i, learn codebook C_i with K centroids:
   C_i = {c_i^1, c_i^2, ..., c_i^K} via k-means

3. Quantize each sub-vector to nearest centroid:
   q(x_i) = argmin_{c ∈ C_i} ||x_i - c||²

4. Store only centroid IDs:
   PQ(x) = [id₁, id₂, ..., id_M] where id_i ∈ [0, K-1]

5. Compressed size:
   M × log₂(K) bits

   For M=96, K=256 (8 bits):
   96 × 8 bits = 96 bytes (vs 3072 bytes original)
```

**Distance computation with PQ:**
```
Original: d(x, y) = Σᵢ(xᵢ - yᵢ)²

With PQ:
d(PQ(x), y) ≈ Σⱼ d(qⱼ(xⱼ), yⱼ)²

where qⱼ(xⱼ) is the centroid representing sub-vector j

Approximation error:
ε = |d(x,y) - d(PQ(x), y)| / d(x,y)
Typical: ε < 10% with proper tuning
```

**Asymmetric Distance Computation (ADC):**
```
Key insight: Query stays in full precision!

1. Precompute distance tables:
   For each sub-space j, for each centroid k:
     table[j][k] = ||y_j - c_j^k||²

2. For compressed database vector PQ(x) = [id₁, ..., id_M]:
   d(PQ(x), y) = Σⱼ table[j][idⱼ]

Time: O(M) lookups instead of O(d) computations!
Speedup: 10-100x for distance computation
```

### IVF Search Analysis

**Clustering math:**
```
Given N vectors, nlist clusters:
  Expected cluster size: N / nlist

K-means objective:
  minimize Σᵢ Σ_{x∈cluster_i} ||x - centroid_i||²

Training complexity: O(iterations × N × nlist × d)
Typical: 10-20 iterations
```

**Search complexity:**
```
Without IVF: O(N × d)

With IVF (nprobe clusters):
  Step 1: Find nprobe nearest centroids: O(nlist × d)
  Step 2: Search vectors in nprobe clusters: O((nprobe × N/nlist) × d)

  Total: O(nlist × d + (nprobe × N/nlist) × d)
       ≈ O(nprobe × N/nlist × d)  (when nlist << N)

Speedup: N / (nprobe × N/nlist) = nlist / nprobe

Example:
  nlist=1000, nprobe=10
  Speedup: 1000/10 = 100x
```

**Recall estimation:**
```
Probability that true NN is in nprobe clusters:

Assuming uniform distribution:
  P(NN in nprobe) ≈ nprobe / nlist

In practice (with clustering):
  Recall higher because similar vectors cluster together

  Typical recall:
    nprobe=1:   60-70%
    nprobe=10:  85-92%
    nprobe=50:  95-98%
    nprobe=nlist: 100%
```

### HNSW Graph Theory

**Small-world graph properties:**
```
1. High clustering: Most neighbors are also neighbors of each other
2. Short paths: Average path length ~log(N)
3. Navigability: Can efficiently route to any node

HNSW construction:
  - Layer l has n_l = N × (1/2)^l nodes
  - Layer 0 has all N nodes
  - Top layer has ~1 node
  - Each node has M connections per layer

Total edges: ~N × M × log₂(N)
Memory: ~N × M × log₂(N) × 4 bytes

For N=1M, M=32:
  Edges: 1M × 32 × 20 = 640M edges
  Memory: 640M × 4 bytes = 2.5 GB
```

**Search complexity:**
```
Expected hops per layer: O(efSearch)
Number of layers: O(log N)

Total complexity: O(efSearch × log N)

For N=1 billion:
  log₂(1B) ≈ 30 layers
  efSearch=16
  Operations: ~500 distance computations (vs 1 billion for linear!)
```

---

## 🏭 Production: How Meta Uses Vector Search

### Meta's Social Search (10B+ Embeddings)

**Problem:** Find similar users, posts, and groups among billions of entities.

**Architecture:**
```
Database: 10 billion user/content embeddings (768D)
QPS: 100,000+ queries per second (peak)
Latency SLA: p99 < 50ms
Hardware: 100+ servers, 800+ GPUs (A100)
```

**FAISS Configuration:**
```python
# Index design
nlist = 100000  # 100K clusters (sqrt(10B) = 100K)
M = 64          # PQ compression (64 bytes per vector)

index = faiss.IndexIVFPQ(
    quantizer=faiss.IndexFlatL2(768),
    d=768,
    nlist=nlist,
    M=M,
    nbits=8
)

# Sharding strategy
num_shards = 100  # Each shard handles 100M vectors
per_shard_memory = 100M × 64 bytes = 6.4 GB

# GPU configuration
gpu_per_shard = 1  # Each shard fits on 1 GPU
total_gpus = 100   # Distributed across 100 GPUs
```

**Query flow:**
```
1. User query → Embedding model → 768D vector
2. Scatter query to all 100 shards (parallel)
3. Each shard searches 100M vectors with IVFPQ
   - nprobe=20 (search 20 clusters)
   - Time: ~2-5ms per shard on GPU
4. Gather top-K from each shard
5. Merge and rerank top-1000 candidates
6. Return top-10 to user

Total latency: ~10-20ms (p50), ~40-50ms (p99)
```

**Key optimizations:**
1. **GPU batching:** Process 32-128 queries simultaneously
2. **Async serving:** Non-blocking I/O for scatter-gather
3. **Caching:** Cache popular queries (30% hit rate)
4. **Reranking:** Recompute exact distance for top-1000 candidates
5. **Index updates:** Rolling updates every 6-12 hours (new embeddings)

### OpenAI's Embedding Search

**Use case:** Retrieval for GPT plugins, ChatGPT document search

**Scale:**
```
Embeddings: Millions of customer documents
Queries: Thousands per second across all customers
Model: text-embedding-ada-002 (1536D)
```

**Architecture:**
```python
# Per-customer index (multi-tenancy)
customer_indices = {}

for customer_id, documents in customer_data.items():
    # Each customer gets isolated index
    index = faiss.IndexHNSWFlat(1536, 32)
    embeddings = embed_model(documents)
    index.add(embeddings)
    customer_indices[customer_id] = index

# Query with isolation
def search(customer_id, query):
    index = customer_indices[customer_id]
    query_emb = embed_model(query)
    distances, indices = index.search(query_emb, k=10)
    return [customer_docs[i] for i in indices]
```

**Challenges:**
- **Multi-tenancy:** Each customer needs isolated index
- **Dynamic updates:** New documents added continuously
- **Latency:** <100ms for interactive chat experience
- **Cost:** Balance accuracy vs infrastructure cost

---

## 🎯 Learning Objectives

By the end of this lab, you will:

**Theory:**
- [ ] Understand why linear search fails at scale (curse of dimensionality)
- [ ] Explain IVF, HNSW, and PQ algorithms
- [ ] Analyze recall-speed-memory tradeoffs for each index type
- [ ] Calculate memory requirements for billion-scale databases
- [ ] Design index configurations for production requirements

**Implementation:**
- [ ] Build FAISS indexes (Flat, IVF, HNSW, IVFPQ)
- [ ] Train and tune index parameters (nlist, nprobe, M, efSearch)
- [ ] Apply product quantization for 10-30x compression
- [ ] Benchmark recall, latency, and memory usage
- [ ] Deploy GPU-accelerated search
- [ ] Implement sharding for multi-GPU scaling

**Production Skills:**
- [ ] Index 1M+ vectors with <100ms query latency
- [ ] Achieve >90% recall with <10% memory overhead
- [ ] Monitor and optimize index performance
- [ ] Handle dynamic updates (add/delete vectors)
- [ ] Design for high QPS (thousands of queries/second)

---

## 🔑 Key Concepts

### 1. Approximate Nearest Neighbor (ANN)

**Definition:** Find neighbors that are "close enough" instead of exact nearest.

**Tradeoff:**
```
Exact search: 100% recall, O(N) time
ANN search:   90-99% recall, O(log N) time

For N=1 billion:
  Exact: 1 billion comparisons (~10 seconds)
  ANN:   ~1,000 comparisons (~1 millisecond)

  1000x speedup at cost of 1-10% recall loss!
```

**When ANN is acceptable:**
- Search/recommendation systems (user won't notice 95% vs 100% recall)
- Information retrieval (top-10 results still relevant)
- Real-time systems where latency matters more than perfection

### 2. Index Training

**Why training is needed:**
```
IVF: Must learn cluster centroids (k-means)
PQ:  Must learn sub-space codebooks (quantization)

Training data: Sample of database (typically 100K-1M vectors)
Training time: Minutes to hours depending on size

Once trained, index can add new vectors without retraining
(until distribution shift requires retraining)
```

**Example:**
```python
# Train on sample
train_vectors = vectors[:100000]
index.train(train_vectors)

# Add full database
index.add(vectors)

# Add new vectors later (no retraining)
index.add(new_vectors)
```

### 3. Recall vs Latency Tradeoff

**Recall:** Fraction of true nearest neighbors found.
```
recall = (# true NNs found) / K

Example: Query for top-10, ANN returns 9 correct → recall = 0.9
```

**Tuning for recall:**
```python
# IVF: Increase nprobe
index.nprobe = 1   # Fast, ~70% recall
index.nprobe = 10  # Balanced, ~90% recall  ← Production default
index.nprobe = 50  # Slow, ~97% recall

# HNSW: Increase efSearch
index.hnsw.efSearch = 16   # Fast, ~85% recall
index.hnsw.efSearch = 64   # Balanced, ~95% recall  ← Production default
index.hnsw.efSearch = 256  # Slow, ~99% recall
```

**Production guidelines:**
- **User-facing search:** 90-95% recall (users don't notice)
- **Recommendation systems:** 85-92% recall (diversity valued)
- **Deduplication:** 98-100% recall (must catch duplicates)

### 4. Sharding for Scale

**Problem:** 10 billion vectors don't fit on one machine.

**Solution:** Partition database across machines, search in parallel.

```python
# Partition vectors
num_shards = 10
shard_size = len(vectors) // num_shards

shards = []
for i in range(num_shards):
    shard_vectors = vectors[i*shard_size:(i+1)*shard_size]

    # Each shard gets own index
    shard_index = faiss.IndexIVFPQ(...)
    shard_index.train(shard_vectors)
    shard_index.add(shard_vectors)
    shards.append(shard_index)

# Query all shards in parallel
def search(query, k=10):
    results = []
    for shard in shards:
        distances, indices = shard.search(query, k=k)
        results.extend(zip(distances[0], indices[0]))

    # Merge and return top-K overall
    results.sort(key=lambda x: x[0])
    return results[:k]
```

**Scaling:**
- 1 shard: 100M vectors, 1 GPU, 10ms latency
- 10 shards: 1B vectors, 10 GPUs, 15ms latency (parallel!)
- 100 shards: 10B vectors, 100 GPUs, 20ms latency

---

## 💻 Exercises

### Exercise 1: Build Your First Vector Database (45 mins)

**What You'll Learn:**
- Creating FAISS indexes
- Adding and searching vectors
- Comparing exact vs approximate search
- Measuring recall

**Why It Matters:**
This is the foundation of every search system. From Google's semantic search to ChatGPT's document retrieval, it all starts here. Understanding exact vs approximate search clarifies why billion-scale systems need ANN.

**Task:** Index 1 million embeddings and measure search performance.

**Starter code:**
```python
import faiss
import numpy as np
import time

# Generate 1M random embeddings (768D)
d = 768
n = 1000000
vectors = np.random.random((n, d)).astype('float32')

# 1. Build exact search index
index_exact = faiss.IndexFlatL2(d)
index_exact.add(vectors)

# 2. Build approximate search index (IVF)
nlist = 1000
quantizer = faiss.IndexFlatL2(d)
index_approx = faiss.IndexIVFFlat(quantizer, d, nlist)
index_approx.train(vectors)
index_approx.add(vectors)
index_approx.nprobe = 10

# 3. Compare performance
query = np.random.random((1, d)).astype('float32')

# Exact search
start = time.time()
D_exact, I_exact = index_exact.search(query, k=10)
time_exact = time.time() - start

# Approximate search
start = time.time()
D_approx, I_approx = index_approx.search(query, k=10)
time_approx = time.time() - start

# Calculate recall
recall = len(set(I_exact[0]) & set(I_approx[0])) / 10

print(f"Exact search: {time_exact*1000:.2f}ms")
print(f"Approx search: {time_approx*1000:.2f}ms")
print(f"Speedup: {time_exact/time_approx:.1f}x")
print(f"Recall: {recall*100:.1f}%")
```

**Expected results:**
- Exact search: ~50-100ms
- Approx search: ~2-5ms (10-20x faster)
- Recall: ~90%

### Exercise 2: Compare Index Types (60 mins)

**What You'll Learn:**
- Performance characteristics of IVF vs HNSW
- Memory consumption of different indexes
- Tuning parameters for optimal performance
- Recall-latency-memory tradeoff curves

**Why It Matters:**
Production systems must balance multiple constraints:
- Search latency (user experience)
- Memory cost (infrastructure cost)
- Recall (quality)
Choosing the wrong index costs millions in wasted hardware or poor user experience.

**Task:** Benchmark Flat, IVF, HNSW, and IVFPQ on 1M vectors.

**Implementation:**
```python
indexes = {
    'Flat': faiss.IndexFlatL2(d),
    'IVF': faiss.IndexIVFFlat(quantizer, d, nlist=1000),
    'HNSW': faiss.IndexHNSWFlat(d, M=32),
    'IVFPQ': faiss.IndexIVFPQ(quantizer, d, nlist=1000, M=96, nbits=8)
}

results = []
for name, index in indexes.items():
    # Train if needed
    if hasattr(index, 'train'):
        index.train(vectors[:100000])

    # Add vectors
    start = time.time()
    index.add(vectors)
    add_time = time.time() - start

    # Measure memory
    if name == 'Flat':
        memory = vectors.nbytes
    elif name == 'IVFPQ':
        memory = n * 96  # PQ codes
    else:
        memory = vectors.nbytes  # Full vectors

    # Search
    start = time.time()
    for _ in range(100):
        query = np.random.random((1, d)).astype('float32')
        D, I = index.search(query, k=10)
    search_time = (time.time() - start) / 100

    # Calculate recall vs exact
    recall = calculate_recall(index, index_exact, n_queries=100)

    results.append({
        'Index': name,
        'Memory (GB)': memory / 1e9,
        'Add Time (s)': add_time,
        'Search Time (ms)': search_time * 1000,
        'Recall (%)': recall * 100
    })

print(pd.DataFrame(results))
```

**Expected results:**
| Index | Memory (GB) | Add Time (s) | Search (ms) | Recall (%) |
|-------|-------------|--------------|-------------|------------|
| Flat | 3.0 | 1 | 50 | 100 |
| IVF | 3.0 | 30 | 3 | 90 |
| HNSW | 12.0 | 180 | 1 | 95 |
| IVFPQ | 0.1 | 45 | 2 | 88 |

**Analysis:**
- Flat: Baseline (exact but slow)
- IVF: Good balance for production
- HNSW: Best search speed, high memory
- IVFPQ: Best memory efficiency, good speed

### Exercise 3: Product Quantization for Compression (60 mins)

**What You'll Learn:**
- How PQ compresses vectors 10-30x
- Tradeoff between compression and accuracy
- Optimal M and nbits parameters
- Memory calculation for billion-scale

**Why It Matters:**
Billion-scale systems are memory-constrained:
- 10B vectors × 3KB = 30 TB (impossible on GPU)
- 10B vectors × 96 bytes = 960 GB (fits on 12x A100 80GB!)
PQ makes billion-scale search economically feasible.

**Task:** Compare different PQ configurations and measure compression vs recall.

**Implementation:**
```python
pq_configs = [
    {'M': 96, 'nbits': 8},   # 96 bytes
    {'M': 64, 'nbits': 8},   # 64 bytes
    {'M': 32, 'nbits': 8},   # 32 bytes
    {'M': 96, 'nbits': 4},   # 48 bytes (16 centroids per sub-space)
]

for config in pq_configs:
    M, nbits = config['M'], config['nbits']

    # Create IVFPQ index
    index = faiss.IndexIVFPQ(quantizer, d, nlist=1000, M=M, nbits=nbits)
    index.train(vectors[:100000])
    index.add(vectors)
    index.nprobe = 10

    # Measure compressed size
    if nbits == 8:
        bytes_per_vector = M
    else:
        bytes_per_vector = M * nbits / 8

    total_memory = n * bytes_per_vector
    compression_ratio = (n * d * 4) / total_memory

    # Measure recall
    recall = calculate_recall(index, index_exact, n_queries=1000)

    print(f"M={M}, nbits={nbits}:")
    print(f"  Bytes/vector: {bytes_per_vector}")
    print(f"  Total memory: {total_memory/1e9:.2f} GB")
    print(f"  Compression: {compression_ratio:.1f}x")
    print(f"  Recall: {recall*100:.1f}%")
```

**Expected results:**
| M | nbits | Bytes/vec | Memory (GB) | Compression | Recall (%) |
|---|-------|-----------|-------------|-------------|------------|
| 96 | 8 | 96 | 0.10 | 32x | 88 |
| 64 | 8 | 64 | 0.06 | 48x | 84 |
| 32 | 8 | 32 | 0.03 | 96x | 76 |
| 96 | 4 | 48 | 0.05 | 64x | 82 |

**Conclusion:** M=96, nbits=8 is the sweet spot (32x compression, ~88% recall).

### Exercise 4: GPU-Accelerated Search (45 mins)

**What You'll Learn:**
- Moving indexes to GPU
- GPU vs CPU performance comparison
- Batch query processing
- Multi-GPU sharding

**Why It Matters:**
CPUs are 10-100x slower than GPUs for vector search:
- CPU: 50ms per query (20 QPS)
- GPU: 2ms per query (500 QPS)
For production serving at 10,000 QPS, GPUs are mandatory.

**Task:** Benchmark CPU vs GPU search, then scale to multiple GPUs.

**Implementation:**
```python
# Create index on CPU
cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist=1000)
cpu_index.train(vectors[:100000])
cpu_index.add(vectors)
cpu_index.nprobe = 10

# Move to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Benchmark single query
queries = np.random.random((1000, d)).astype('float32')

# CPU
start = time.time()
for q in queries:
    cpu_index.search(q.reshape(1, -1), k=10)
cpu_time = time.time() - start

# GPU (sequential)
start = time.time()
for q in queries:
    gpu_index.search(q.reshape(1, -1), k=10)
gpu_seq_time = time.time() - start

# GPU (batched)
start = time.time()
D, I = gpu_index.search(queries, k=10)  # Batch all queries
gpu_batch_time = time.time() - start

print(f"CPU: {cpu_time:.3f}s ({1000/cpu_time:.1f} QPS)")
print(f"GPU (sequential): {gpu_seq_time:.3f}s ({1000/gpu_seq_time:.1f} QPS)")
print(f"GPU (batched): {gpu_batch_time:.3f}s ({1000/gpu_batch_time:.1f} QPS)")
print(f"Speedup: {cpu_time/gpu_batch_time:.1f}x")
```

**Expected results:**
- CPU: 50s (20 QPS)
- GPU sequential: 5s (200 QPS) - 10x speedup
- GPU batched: 1s (1000 QPS) - 50x speedup!

### Exercise 5: Production-Scale System (90 mins)

**What You'll Learn:**
- End-to-end search pipeline
- Sharding and load balancing
- Monitoring and optimization
- Handling dynamic updates

**Why It Matters:**
This simulates a real production system handling millions of users:
- Continuous index updates (new documents)
- High QPS (thousands of queries/second)
- SLA requirements (p99 < 100ms)
- Cost optimization (minimize GPU count)

**Task:** Build a system that can:
1. Index 10M vectors
2. Handle 1000 QPS
3. Maintain <100ms p99 latency
4. Support dynamic updates

**Architecture:**
```python
class ProductionVectorDB:
    def __init__(self, d=768, num_shards=4):
        self.d = d
        self.num_shards = num_shards
        self.shards = []

        # Create sharded indexes
        for i in range(num_shards):
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist=1000, M=96, nbits=8)

            # Move to GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, i % 4, index)  # Distribute across GPUs

            self.shards.append(gpu_index)

    def add(self, vectors):
        # Partition vectors across shards
        n = len(vectors)
        shard_size = n // self.num_shards

        for i, shard in enumerate(self.shards):
            shard_vectors = vectors[i*shard_size:(i+1)*shard_size]
            shard.add(shard_vectors)

    def search(self, queries, k=10):
        # Search all shards in parallel (async)
        shard_results = []
        for shard in self.shards:
            D, I = shard.search(queries, k=k)
            shard_results.append((D, I))

        # Merge results
        merged = self._merge_results(shard_results, k)
        return merged

    def _merge_results(self, shard_results, k):
        # Merge top-K from each shard
        batch_size = shard_results[0][0].shape[0]
        final_D = []
        final_I = []

        for i in range(batch_size):
            # Collect all distances and indices for query i
            all_D = []
            all_I = []
            for D, I in shard_results:
                all_D.extend(D[i])
                all_I.extend(I[i])

            # Sort and take top-K
            sorted_idx = np.argsort(all_D)[:k]
            final_D.append([all_D[j] for j in sorted_idx])
            final_I.append([all_I[j] for j in sorted_idx])

        return np.array(final_D), np.array(final_I)

# Benchmark
db = ProductionVectorDB(d=768, num_shards=4)
vectors = np.random.random((10000000, 768)).astype('float32')

# Train and add
print("Training...")
for shard in db.shards:
    shard.train(vectors[:100000])

print("Adding vectors...")
db.add(vectors)

# Benchmark QPS
queries = np.random.random((1000, 768)).astype('float32')
start = time.time()
results = db.search(queries, k=10)
elapsed = time.time() - start

qps = len(queries) / elapsed
latency_ms = elapsed / len(queries) * 1000

print(f"QPS: {qps:.0f}")
print(f"Avg latency: {latency_ms:.2f}ms")
print(f"Target QPS (1000): {'✓' if qps >= 1000 else '✗'}")
print(f"Target latency (<100ms): {'✓' if latency_ms < 100 else '✗'}")
```

**Success criteria:**
- ✓ QPS >= 1000
- ✓ Latency <100ms
- ✓ Memory <40 GB (10M × 96 bytes × 4 shards / 4 GPUs = 10 GB per GPU)

---

## ⚠️ Common Pitfalls

### 1. Forgetting to Normalize Vectors

**Symptom:** Poor recall with IndexFlatIP (inner product).

**Cause:** Inner product is sensitive to magnitude.

**Solution:**
```python
# Always normalize for cosine similarity
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(d)  # Now equivalent to cosine
```

### 2. Not Training Index Before Adding

**Symptom:**
```
RuntimeError: Index not trained
```

**Solution:**
```python
# Must train IVF/PQ indexes
if hasattr(index, 'train'):
    index.train(train_vectors)  # Use sample (100K-1M vectors)

index.add(vectors)  # Now works
```

### 3. Using Too Many/Few Clusters

**Symptom:** Poor recall or slow search.

**Rule of thumb:**
```python
# IVF nlist
nlist = int(np.sqrt(n))  # For n vectors

# Examples:
n = 1,000,000  → nlist = 1,000
n = 100,000,000 → nlist = 10,000
n = 1,000,000,000 → nlist = 31,623
```

### 4. Insufficient nprobe for Recall

**Symptom:** Recall <80% even with good index.

**Solution:**
```python
# Start with nprobe = nlist / 100
index.nprobe = max(1, nlist // 100)

# Tune up if recall insufficient
for nprobe in [1, 10, 20, 50, 100]:
    index.nprobe = nprobe
    recall = measure_recall(index)
    print(f"nprobe={nprobe}: recall={recall:.2%}")
```

### 5. Running Out of GPU Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use PQ compression: `IndexIVFPQ` instead of `IndexIVFFlat`
2. Shard across multiple GPUs
3. Use CPU index for larger-than-GPU datasets

---

## 🏆 Expert Checklist for Mastery

**Foundations:**
- [ ] Understand why linear search fails for high-dimensional data
- [ ] Explain IVF clustering algorithm and search process
- [ ] Know how HNSW graphs enable logarithmic search
- [ ] Understand Product Quantization compression
- [ ] Can calculate memory for billion-scale systems

**Implementation:**
- [ ] Built and compared Flat, IVF, HNSW, IVFPQ indexes
- [ ] Trained indexes on large datasets (1M+ vectors)
- [ ] Tuned parameters for 90%+ recall
- [ ] Achieved <10ms search latency on 1M vectors
- [ ] Applied 32x PQ compression with <10% recall loss

**Production:**
- [ ] Deployed GPU-accelerated search
- [ ] Implemented sharding for multi-GPU scaling
- [ ] Built system handling 1000+ QPS
- [ ] Monitored recall, latency, and memory
- [ ] Handled dynamic index updates

**Advanced:**
- [ ] Understand recall-latency-memory tradeoff curves
- [ ] Can design index for specific production requirements
- [ ] Know when to use IVF vs HNSW vs hybrid
- [ ] Familiar with advanced techniques (IVFPQ-R, ScaNN, DiskANN)

---

## 🚀 Next Steps

After mastering vector search:

1. **Integrate with LLMs**
   - Build RAG (Retrieval-Augmented Generation) system
   - Semantic search over documentation
   - Context retrieval for ChatGPT-like interfaces

2. **Explore Vector Databases**
   - Pinecone, Weaviate, Milvus (built on FAISS)
   - Managed vector search services
   - Production deployment patterns

3. **Advanced Techniques**
   - Learned sparse embeddings (SPLADE)
   - Hybrid search (dense + sparse)
   - Multi-modal search (text + images)

4. **Scale to Billions**
   - DiskANN (disk-based billion-scale)
   - Distributed FAISS across clusters
   - Production monitoring and optimization

---

## 📚 References

**Papers:**
- [Product Quantization for Nearest Neighbor Search (Jégou et al., 2011)](https://hal.inria.fr/inria-00514462v2/document)
- [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW (Malkov & Yashunin, 2016)](https://arxiv.org/abs/1603.09320)
- [Billion-scale similarity search with GPUs (Johnson et al., 2017)](https://arxiv.org/abs/1702.08734)

**Documentation:**
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Best Practices](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

**Production Examples:**
- Meta's social search (10B+ vectors)
- OpenAI's embeddings API
- Spotify's music recommendations

---

## 🎯 Solution

Complete implementation: `solution/vector_database.py`

**What you'll build:**
- FAISS index builder for multiple index types
- Recall-latency benchmarking framework
- GPU-accelerated search pipeline
- Production sharding and load balancing
- Dynamic index updates
- Complete monitoring and metrics

**Next: Lab 3 - Model Fairness & Bias Detection!**
