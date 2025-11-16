"""
Exercise 2: Compare Index Types
Benchmark Flat, IVF, HNSW, and IVFPQ on 1M vectors

Learning objectives:
- Performance characteristics of different indexes
- Memory consumption comparison
- Tuning parameters for optimal performance
- Understanding recall-latency-memory tradeoffs
"""

import faiss
import numpy as np
import time
import psutil
import os


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def calculate_recall(exact_indices, approx_indices, k=10):
    """Calculate recall@k"""
    recalls = []
    for i in range(len(exact_indices)):
        exact_set = set(exact_indices[i])
        approx_set = set(approx_indices[i])
        recall = len(exact_set & approx_set) / k
        recalls.append(recall)
    return np.mean(recalls)


def main():
    print("=" * 80)
    print("Exercise 2: Compare FAISS Index Types")
    print("=" * 80)

    # Generate 1M embeddings
    d = 768
    n = 1000000
    print(f"\n1. Generating {n:,} vectors of dimension {d}...")
    np.random.seed(42)
    vectors = np.random.random((n, d)).astype('float32')
    faiss.normalize_L2(vectors)

    # Generate queries
    n_queries = 1000
    queries = np.random.random((n_queries, d)).astype('float32')
    faiss.normalize_L2(queries)
    k = 10

    print(f"Generated {n:,} vectors and {n_queries} queries")

    # Results storage
    results = []

    # ========================================
    # Index 1: Flat (Exact Search)
    # ========================================
    print("\n2. Testing IndexFlatL2 (Exact Search)")
    print("-" * 80)

    mem_before = get_memory_usage_mb()

    print("Building Flat index...")
    index_flat = faiss.IndexFlatL2(d)

    build_start = time.time()
    index_flat.add(vectors)
    build_time = time.time() - build_start

    mem_after = get_memory_usage_mb()
    memory_mb = mem_after - mem_before

    # Search
    print("Searching...")
    search_start = time.time()
    D_exact, I_exact = index_flat.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    results.append({
        'Index': 'Flat',
        'Build Time (s)': build_time,
        'Memory (MB)': memory_mb,
        'Latency (ms)': search_time,
        'Recall (%)': 100.0,
        'Notes': 'Exact search (baseline)'
    })

    print(f"  Build time: {build_time:.2f}s")
    print(f"  Memory: {memory_mb:.0f} MB")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Recall: 100%")

    # ========================================
    # Index 2: IVF (Inverted File Index)
    # ========================================
    print("\n3. Testing IndexIVFFlat")
    print("-" * 80)

    mem_before = get_memory_usage_mb()

    nlist = 1000
    print(f"Building IVF index with {nlist} clusters...")
    quantizer = faiss.IndexFlatL2(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

    build_start = time.time()
    print("  Training...")
    index_ivf.train(vectors)
    print("  Adding vectors...")
    index_ivf.add(vectors)
    build_time = time.time() - build_start

    mem_after = get_memory_usage_mb()
    memory_mb = mem_after - mem_before

    # Search with nprobe=10
    index_ivf.nprobe = 10
    print(f"Searching with nprobe={index_ivf.nprobe}...")
    search_start = time.time()
    D_ivf, I_ivf = index_ivf.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    recall = calculate_recall(I_exact, I_ivf, k) * 100

    results.append({
        'Index': 'IVF',
        'Build Time (s)': build_time,
        'Memory (MB)': memory_mb,
        'Latency (ms)': search_time,
        'Recall (%)': recall,
        'Notes': f'nlist={nlist}, nprobe=10'
    })

    print(f"  Build time: {build_time:.2f}s")
    print(f"  Memory: {memory_mb:.0f} MB")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Recall: {recall:.1f}%")

    # ========================================
    # Index 3: HNSW (Hierarchical NSW)
    # ========================================
    print("\n4. Testing IndexHNSWFlat")
    print("-" * 80)

    mem_before = get_memory_usage_mb()

    M = 32  # Connections per layer
    print(f"Building HNSW index with M={M}...")
    index_hnsw = faiss.IndexHNSWFlat(d, M)
    index_hnsw.hnsw.efConstruction = 40
    index_hnsw.hnsw.efSearch = 16

    build_start = time.time()
    index_hnsw.add(vectors)
    build_time = time.time() - build_start

    mem_after = get_memory_usage_mb()
    memory_mb = mem_after - mem_before

    # Search
    print("Searching...")
    search_start = time.time()
    D_hnsw, I_hnsw = index_hnsw.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    recall = calculate_recall(I_exact, I_hnsw, k) * 100

    results.append({
        'Index': 'HNSW',
        'Build Time (s)': build_time,
        'Memory (MB)': memory_mb,
        'Latency (ms)': search_time,
        'Recall (%)': recall,
        'Notes': f'M={M}, efSearch=16'
    })

    print(f"  Build time: {build_time:.2f}s")
    print(f"  Memory: {memory_mb:.0f} MB")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Recall: {recall:.1f}%")

    # ========================================
    # Index 4: IVFPQ (IVF + Product Quantization)
    # ========================================
    print("\n5. Testing IndexIVFPQ (with compression)")
    print("-" * 80)

    mem_before = get_memory_usage_mb()

    nlist = 1000
    M = 96  # Sub-vectors
    nbits = 8  # Bits per sub-vector
    print(f"Building IVFPQ index: nlist={nlist}, M={M}, nbits={nbits}...")

    quantizer = faiss.IndexFlatL2(d)
    index_pq = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

    build_start = time.time()
    print("  Training...")
    index_pq.train(vectors)
    print("  Adding vectors...")
    index_pq.add(vectors)
    build_time = time.time() - build_start

    mem_after = get_memory_usage_mb()
    memory_mb = mem_after - mem_before

    # Search
    index_pq.nprobe = 10
    print(f"Searching with nprobe={index_pq.nprobe}...")
    search_start = time.time()
    D_pq, I_pq = index_pq.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    recall = calculate_recall(I_exact, I_pq, k) * 100

    # Calculate compression ratio
    original_size_mb = (n * d * 4) / (1024 * 1024)  # FP32
    compressed_size_mb = (n * M) / (1024 * 1024)  # M bytes per vector
    compression_ratio = original_size_mb / compressed_size_mb

    results.append({
        'Index': 'IVFPQ',
        'Build Time (s)': build_time,
        'Memory (MB)': memory_mb,
        'Latency (ms)': search_time,
        'Recall (%)': recall,
        'Notes': f'{compression_ratio:.1f}x compression'
    })

    print(f"  Build time: {build_time:.2f}s")
    print(f"  Memory: {memory_mb:.0f} MB")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Recall: {recall:.1f}%")
    print(f"  Compression: {compression_ratio:.1f}x")

    # ========================================
    # Summary Table
    # ========================================
    print("\n6. Summary Comparison")
    print("=" * 80)
    print(f"{'Index':<12} {'Build (s)':<12} {'Memory (MB)':<15} {'Latency (ms)':<15} {'Recall (%)':<12} {'Notes':<25}")
    print("-" * 80)

    for r in results:
        print(f"{r['Index']:<12} {r['Build Time (s)']:<12.1f} {r['Memory (MB)']:<15.0f} "
              f"{r['Latency (ms)']:<15.2f} {r['Recall (%)']:<12.1f} {r['Notes']:<25}")

    # ========================================
    # Analysis
    # ========================================
    print("\n7. Analysis & Recommendations")
    print("=" * 80)
    print("""
**Flat (Exact Search)**
  - Use when: Dataset <1M vectors, need 100% recall, latency <100ms OK
  - Pros: Perfect accuracy, no training needed
  - Cons: Slow (O(N) search), high memory

**IVF (Inverted File Index)**
  - Use when: Dataset 1M-100M vectors, need 85-95% recall, <10ms latency
  - Pros: Good balance of speed/accuracy, reasonable memory
  - Cons: Requires training, tuning nprobe vs recall

**HNSW (Hierarchical NSW)**
  - Use when: Need best search speed, memory not constrained, high recall
  - Pros: Fastest search (O(log N)), excellent recall
  - Cons: 4-5x more memory than IVF, slower build time

**IVFPQ (IVF + Product Quantization)**
  - Use when: Dataset 100M-1B+ vectors, memory constrained, 85-92% recall OK
  - Pros: 10-30x compression, enables billion-scale search
  - Cons: Training expensive, some accuracy loss

**Production Recommendations:**
  - <10M vectors: Use Flat or IVF
  - 10M-100M vectors: Use IVF or HNSW
  - 100M-1B vectors: Use IVFPQ
  - >1B vectors: Use IVFPQ with sharding across multiple machines/GPUs
    """)

    print("=" * 80)
    print("Exercise 2 Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
