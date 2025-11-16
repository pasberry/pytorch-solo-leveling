"""
Exercise 3: Product Quantization
Compress 100M vectors using Product Quantization (PQ)

Learning objectives:
- Understanding product quantization compression
- Measuring compression ratio vs accuracy tradeoff
- Optimizing PQ parameters (M, nbits)
- Billion-scale memory optimization
"""

import faiss
import numpy as np
import time


def main():
    print("=" * 80)
    print("Exercise 3: Product Quantization for Compression")
    print("=" * 80)

    # Generate 10M embeddings (simulating large dataset)
    d = 768
    n = 10000000  # 10M vectors
    print(f"\n1. Generating {n:,} vectors of dimension {d}...")
    np.random.seed(42)

    # Generate in batches to avoid memory issues
    batch_size = 1000000
    vectors = []
    for i in range(0, n, batch_size):
        batch = np.random.random((min(batch_size, n - i), d)).astype('float32')
        faiss.normalize_L2(batch)
        vectors.append(batch)
    vectors = np.vstack(vectors)

    print(f"Generated {n:,} vectors")

    # Generate queries
    n_queries = 100
    queries = np.random.random((n_queries, d)).astype('float32')
    faiss.normalize_L2(queries)
    k = 10

    # Calculate original size
    original_size_bytes = n * d * 4  # FP32 = 4 bytes
    original_size_gb = original_size_bytes / (1024**3)
    print(f"\nOriginal vector size: {original_size_gb:.2f} GB")

    # ========================================
    # Baseline: Flat index (no compression)
    # ========================================
    print("\n2. Baseline: Flat Index (No Compression)")
    print("-" * 80)

    print("Building Flat index...")
    index_flat = faiss.IndexFlatL2(d)

    build_start = time.time()
    index_flat.add(vectors)
    build_time = time.time() - build_start

    # Search to get ground truth
    print("Searching for ground truth...")
    search_start = time.time()
    D_exact, I_exact = index_flat.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    print(f"  Build time: {build_time:.2f}s")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Memory: {original_size_gb:.2f} GB")
    print(f"  Recall: 100%")

    # ========================================
    # Product Quantization - Various Settings
    # ========================================
    print("\n3. Product Quantization Experiments")
    print("-" * 80)

    results = []

    # Experiment 1: Standard PQ (96 sub-vectors, 8 bits)
    print("\nExperiment 1: PQ with M=96, nbits=8")
    M = 96
    nbits = 8

    index_pq1 = faiss.IndexPQ(d, M, nbits)

    train_start = time.time()
    print("  Training quantizer...")
    index_pq1.train(vectors[:1000000])  # Train on 1M samples
    train_time = time.time() - train_start

    add_start = time.time()
    print("  Adding vectors...")
    index_pq1.add(vectors)
    add_time = time.time() - add_start

    # Search
    search_start = time.time()
    D_pq1, I_pq1 = index_pq1.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    # Calculate recall
    recall = calculate_recall(I_exact, I_pq1, k)

    # Memory calculation
    compressed_size_bytes = n * M  # M bytes per vector
    compressed_size_gb = compressed_size_bytes / (1024**3)
    compression_ratio = original_size_gb / compressed_size_gb

    results.append({
        'Config': f'PQ M={M}, nbits={nbits}',
        'Train Time (s)': train_time,
        'Add Time (s)': add_time,
        'Search (ms)': search_time,
        'Memory (GB)': compressed_size_gb,
        'Compression': f'{compression_ratio:.1f}x',
        'Recall (%)': recall * 100
    })

    print(f"  Train time: {train_time:.2f}s")
    print(f"  Add time: {add_time:.2f}s")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Compressed size: {compressed_size_gb:.2f} GB ({compression_ratio:.1f}x compression)")
    print(f"  Recall@10: {recall*100:.1f}%")

    # Experiment 2: More compression (64 sub-vectors, 8 bits)
    print("\nExperiment 2: PQ with M=64, nbits=8 (more compression)")
    M = 64
    nbits = 8

    index_pq2 = faiss.IndexPQ(d, M, nbits)

    train_start = time.time()
    print("  Training quantizer...")
    index_pq2.train(vectors[:1000000])
    train_time = time.time() - train_start

    add_start = time.time()
    print("  Adding vectors...")
    index_pq2.add(vectors)
    add_time = time.time() - add_start

    # Search
    search_start = time.time()
    D_pq2, I_pq2 = index_pq2.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    recall = calculate_recall(I_exact, I_pq2, k)

    compressed_size_bytes = n * M
    compressed_size_gb = compressed_size_bytes / (1024**3)
    compression_ratio = original_size_gb / compressed_size_gb

    results.append({
        'Config': f'PQ M={M}, nbits={nbits}',
        'Train Time (s)': train_time,
        'Add Time (s)': add_time,
        'Search (ms)': search_time,
        'Memory (GB)': compressed_size_gb,
        'Compression': f'{compression_ratio:.1f}x',
        'Recall (%)': recall * 100
    })

    print(f"  Train time: {train_time:.2f}s")
    print(f"  Add time: {add_time:.2f}s")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Compressed size: {compressed_size_gb:.2f} GB ({compression_ratio:.1f}x compression)")
    print(f"  Recall@10: {recall*100:.1f}%")

    # Experiment 3: Less compression but better accuracy (128 sub-vectors)
    print("\nExperiment 3: PQ with M=128, nbits=8 (less compression, better recall)")
    M = 128
    nbits = 8

    # Note: d must be divisible by M, so for d=768, valid M values are divisors
    # 768 = 2^8 * 3, so M could be 96, 128, 192, 256, 384, 768
    # Let's use 96 again but with nbits=16 for comparison
    M = 96
    nbits = 16

    print(f"  (Using M=96, nbits=16 for comparison - not all M values work with d=768)")

    index_pq3 = faiss.IndexPQ(d, M, nbits)

    train_start = time.time()
    print("  Training quantizer...")
    index_pq3.train(vectors[:1000000])
    train_time = time.time() - train_start

    add_start = time.time()
    print("  Adding vectors...")
    index_pq3.add(vectors)
    add_time = time.time() - add_start

    # Search
    search_start = time.time()
    D_pq3, I_pq3 = index_pq3.search(queries, k)
    search_time = (time.time() - search_start) / n_queries * 1000

    recall = calculate_recall(I_exact, I_pq3, k)

    # With nbits=16, each sub-vector uses 2 bytes
    compressed_size_bytes = n * M * 2
    compressed_size_gb = compressed_size_bytes / (1024**3)
    compression_ratio = original_size_gb / compressed_size_gb

    results.append({
        'Config': f'PQ M={M}, nbits={nbits}',
        'Train Time (s)': train_time,
        'Add Time (s)': add_time,
        'Search (ms)': search_time,
        'Memory (GB)': compressed_size_gb,
        'Compression': f'{compression_ratio:.1f}x',
        'Recall (%)': recall * 100
    })

    print(f"  Train time: {train_time:.2f}s")
    print(f"  Add time: {add_time:.2f}s")
    print(f"  Search latency: {search_time:.2f} ms")
    print(f"  Compressed size: {compressed_size_gb:.2f} GB ({compression_ratio:.1f}x compression)")
    print(f"  Recall@10: {recall*100:.1f}%")

    # ========================================
    # Summary Table
    # ========================================
    print("\n4. Compression Comparison")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Train (s)':<12} {'Search (ms)':<12} {'Memory (GB)':<12} {'Compression':<12} {'Recall (%)':<12}")
    print("-" * 80)
    print(f"{'Flat (no compression)':<25} {build_time:<12.1f} {search_time:<12.2f} {original_size_gb:<12.2f} {'1.0x':<12} {'100.0':<12}")

    for r in results:
        print(f"{r['Config']:<25} {r['Train Time (s)']:<12.1f} {r['Search (ms)']:<12.2f} "
              f"{r['Memory (GB)']:<12.2f} {r['Compression']:<12} {r['Recall (%)']:<12.1f}")

    # ========================================
    # Analysis
    # ========================================
    print("\n5. Analysis & Key Insights")
    print("=" * 80)
    print("""
**Product Quantization (PQ) Theory:**

PQ works by dividing each d-dimensional vector into M sub-vectors:
  - Original vector: [x1, x2, ..., x768]
  - Split into M=96 sub-vectors of d/M=8 dimensions each
  - Each sub-vector is quantized to one of 2^nbits centroids (256 for nbits=8)
  - Store only the centroid IDs (1-2 bytes) instead of floats (32 bytes)

**Compression Math:**
  - Original: d * 4 bytes = 768 * 4 = 3,072 bytes per vector
  - PQ (M=96, nbits=8): M * 1 byte = 96 bytes per vector
  - Compression ratio: 3,072 / 96 = 32x

**Memory Savings for 100M vectors:**
  - Original: 100M * 3KB = 300 GB
  - PQ compressed: 100M * 96 bytes = 9.6 GB
  - Savings: 290 GB (97% reduction!)

**Recall vs Compression Tradeoff:**
  - More sub-vectors (M) → Better recall, less compression
  - Fewer sub-vectors (M) → More compression, lower recall
  - More bits (nbits) → Better recall, larger memory
  - Typical production: M=64-96, nbits=8, recall 85-95%

**When to Use PQ:**
  - Dataset >100M vectors
  - Memory constrained (can't fit full vectors in RAM)
  - Can tolerate 5-15% recall loss
  - Need fast search (faster than Flat index)

**Production Examples:**
  - **Meta's visual search:** 10B images, PQ compression, <100ms latency
  - **Pinterest's related pins:** Billions of pins, PQ + IVF hybrid
  - **Google's semantic search:** Billions of documents, multi-stage retrieval

**Best Practices:**
  1. Train on representative sample (1M vectors is usually enough)
  2. Use d divisible by M (e.g., d=768 works with M=96, 64, 48, 32)
  3. Start with M=96, nbits=8 as baseline
  4. Measure recall on validation set before deploying
  5. Consider IVF+PQ hybrid for best speed/recall balance
    """)

    print("\n" + "=" * 80)
    print("Exercise 3 Complete!")
    print("=" * 80)
    print(f"\nKey Takeaway: Product Quantization enables 10-30x compression,")
    print(f"making billion-scale vector search feasible on single machines!")


def calculate_recall(exact_indices, approx_indices, k=10):
    """Calculate recall@k"""
    recalls = []
    for i in range(len(exact_indices)):
        exact_set = set(exact_indices[i])
        approx_set = set(approx_indices[i])
        recall = len(exact_set & approx_set) / k
        recalls.append(recall)
    return np.mean(recalls)


if __name__ == "__main__":
    main()
