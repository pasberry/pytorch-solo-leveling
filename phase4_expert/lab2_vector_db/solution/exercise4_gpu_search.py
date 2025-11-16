"""
Exercise 4: GPU-Accelerated Search
Move FAISS indexes to GPU for 10-100x faster search

Learning objectives:
- Understanding CPU vs GPU search performance
- Moving indexes to GPU with faiss-gpu
- Optimizing batch size for throughput
- Multi-GPU sharding for massive scale
"""

import faiss
import numpy as np
import time


def main():
    print("=" * 80)
    print("Exercise 4: GPU-Accelerated FAISS Search")
    print("=" * 80)

    # Check GPU availability
    print("\n1. Checking GPU availability...")
    try:
        ngpus = faiss.get_num_gpus()
        print(f"Number of GPUs available: {ngpus}")

        if ngpus == 0:
            print("\n⚠️  No GPUs detected. This exercise will demonstrate the API,")
            print("   but you need a GPU to see actual speedups.")
            print("   On CPU, we'll simulate with smaller dataset.\n")
            use_gpu = False
            # Use smaller dataset for CPU demo
            n = 100000
            n_queries = 100
        else:
            print(f"✓ GPU(s) detected! Will demonstrate GPU acceleration.")
            use_gpu = True
            # Use larger dataset for GPU
            n = 10000000
            n_queries = 10000

    except Exception as e:
        print(f"⚠️  FAISS GPU not available: {e}")
        print("   Falling back to CPU demo with smaller dataset.\n")
        use_gpu = False
        n = 100000
        n_queries = 100

    # Generate embeddings
    d = 768
    print(f"\n2. Generating {n:,} vectors of dimension {d}...")
    np.random.seed(42)

    # Generate in batches
    batch_size = 1000000
    vectors = []
    for i in range(0, n, batch_size):
        batch = np.random.random((min(batch_size, n - i), d)).astype('float32')
        faiss.normalize_L2(batch)
        vectors.append(batch)
    vectors = np.vstack(vectors)

    # Generate queries
    queries = np.random.random((n_queries, d)).astype('float32')
    faiss.normalize_L2(queries)
    k = 100  # Return top 100 neighbors

    print(f"Generated {n:,} vectors and {n_queries:,} queries")

    # ========================================
    # CPU Baseline
    # ========================================
    print("\n3. CPU Baseline: IndexFlatL2")
    print("-" * 80)

    print("Building CPU index...")
    index_cpu = faiss.IndexFlatL2(d)

    build_start = time.time()
    index_cpu.add(vectors)
    build_time = time.time() - build_start

    print(f"  Build time: {build_time:.2f}s")
    print(f"  Index size: {index_cpu.ntotal:,} vectors")

    # Warmup
    _ = index_cpu.search(queries[:10], k)

    # Single query latency
    print("\nCPU search (single query)...")
    search_start = time.time()
    D_cpu, I_cpu = index_cpu.search(queries[:1], k)
    single_latency_cpu = (time.time() - search_start) * 1000

    # Batch search throughput
    print(f"CPU search (batch of {n_queries:,} queries)...")
    search_start = time.time()
    D_cpu_batch, I_cpu_batch = index_cpu.search(queries, k)
    batch_time_cpu = time.time() - search_start
    throughput_cpu = n_queries / batch_time_cpu

    print(f"  Single query latency: {single_latency_cpu:.2f} ms")
    print(f"  Batch time: {batch_time_cpu:.2f}s")
    print(f"  Throughput: {throughput_cpu:.0f} queries/sec")

    # ========================================
    # GPU Search (if available)
    # ========================================
    if use_gpu:
        print("\n4. GPU Acceleration: Moving Index to GPU")
        print("-" * 80)

        # Create GPU resources
        print("Setting up GPU resources...")
        res = faiss.StandardGpuResources()

        # Convert CPU index to GPU
        print("Moving index to GPU...")
        gpu_start = time.time()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0
        gpu_transfer_time = time.time() - gpu_start

        print(f"  Transfer time: {gpu_transfer_time:.2f}s")
        print(f"  Index on GPU: {index_gpu.ntotal:,} vectors")

        # Warmup
        _ = index_gpu.search(queries[:10], k)

        # Single query latency
        print("\nGPU search (single query)...")
        search_start = time.time()
        D_gpu, I_gpu = index_gpu.search(queries[:1], k)
        single_latency_gpu = (time.time() - search_start) * 1000

        # Batch search throughput
        print(f"GPU search (batch of {n_queries:,} queries)...")
        search_start = time.time()
        D_gpu_batch, I_gpu_batch = index_gpu.search(queries, k)
        batch_time_gpu = time.time() - search_start
        throughput_gpu = n_queries / batch_time_gpu

        print(f"  Single query latency: {single_latency_gpu:.2f} ms")
        print(f"  Batch time: {batch_time_gpu:.2f}s")
        print(f"  Throughput: {throughput_gpu:.0f} queries/sec")

        # Speedup
        speedup = throughput_gpu / throughput_cpu
        print(f"\n  🚀 GPU Speedup: {speedup:.1f}x faster than CPU!")

        # Verify correctness
        matches = np.sum(I_cpu_batch == I_gpu_batch) / (n_queries * k) * 100
        print(f"  ✓ Results match CPU: {matches:.1f}%")

        # ========================================
        # Batch Size Optimization
        # ========================================
        print("\n5. Batch Size Optimization")
        print("-" * 80)

        batch_sizes = [1, 10, 100, 1000, 10000]
        print(f"\nTesting different batch sizes (k={k})...")
        print(f"{'Batch Size':<15} {'CPU (QPS)':<15} {'GPU (QPS)':<15} {'Speedup':<10}")
        print("-" * 60)

        for bs in batch_sizes:
            if bs > n_queries:
                continue

            test_queries = queries[:bs]

            # CPU
            start = time.time()
            _ = index_cpu.search(test_queries, k)
            cpu_time = time.time() - start
            cpu_qps = bs / cpu_time

            # GPU
            start = time.time()
            _ = index_gpu.search(test_queries, k)
            gpu_time = time.time() - start
            gpu_qps = bs / gpu_time

            speedup = gpu_qps / cpu_qps

            print(f"{bs:<15} {cpu_qps:<15.0f} {gpu_qps:<15.0f} {speedup:<10.1f}x")

        # ========================================
        # Multi-GPU (if available)
        # ========================================
        if ngpus > 1:
            print(f"\n6. Multi-GPU Sharding ({ngpus} GPUs available)")
            print("-" * 80)

            print(f"Sharding index across {ngpus} GPUs...")

            # Create multi-GPU index
            index_multi_gpu = faiss.index_cpu_to_all_gpus(index_cpu)

            print(f"  Index sharded across {ngpus} GPUs")

            # Warmup
            _ = index_multi_gpu.search(queries[:10], k)

            # Batch search
            print(f"\nMulti-GPU search (batch of {n_queries:,} queries)...")
            search_start = time.time()
            D_mgpu, I_mgpu = index_multi_gpu.search(queries, k)
            batch_time_mgpu = time.time() - search_start
            throughput_mgpu = n_queries / batch_time_mgpu

            print(f"  Batch time: {batch_time_mgpu:.2f}s")
            print(f"  Throughput: {throughput_mgpu:.0f} queries/sec")
            print(f"  🚀 Speedup vs CPU: {throughput_mgpu/throughput_cpu:.1f}x")
            print(f"  🚀 Speedup vs Single GPU: {throughput_mgpu/throughput_gpu:.1f}x")

    else:
        # CPU-only demonstration
        print("\n4. GPU API Demonstration (CPU mode)")
        print("-" * 80)
        print("""
Since no GPU is available, here's how you would use GPU acceleration:

```python
import faiss

# Create GPU resources
res = faiss.StandardGpuResources()

# Move index to GPU 0
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Search on GPU
D, I = index_gpu.search(queries, k)

# For multi-GPU (if you have multiple GPUs):
index_multi_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
D, I = index_multi_gpu.search(queries, k)
```

**Expected Speedups (based on production benchmarks):**
  - Single GPU: 10-50x faster than CPU (depends on batch size)
  - 4 GPUs: 30-150x faster than CPU
  - 8 GPUs: 50-200x faster than CPU

**When GPU Acceleration Helps Most:**
  - Large batch queries (1000+ queries at once)
  - High k values (k=100-1000)
  - Production serving with high QPS (1000+ queries/sec)
        """)

    # ========================================
    # Analysis
    # ========================================
    print("\n" + ("7" if use_gpu else "5") + ". Analysis & Best Practices")
    print("=" * 80)
    print("""
**CPU vs GPU Tradeoffs:**

**CPU Search:**
  - Pros: No data transfer overhead, works everywhere
  - Cons: Slower for large batches, limited parallelism
  - Best for: <100 QPS, small batches, limited GPU budget

**GPU Search:**
  - Pros: 10-100x faster for batches, massive parallelism
  - Cons: Data transfer cost, GPU memory limits
  - Best for: >1000 QPS, large batches, production serving

**Batch Size Optimization:**
  - Small batches (1-10): CPU competitive, GPU overhead dominates
  - Medium batches (100-1000): GPU starts to win (10-20x)
  - Large batches (1000-10000): GPU dominates (30-100x)
  - Production tip: Batch incoming queries for 10-50ms to maximize GPU util

**GPU Memory Management:**
  - GPU RAM is limited (16-80 GB typical)
  - Can't fit entire index? Use:
    1. PQ compression to reduce index size 10-30x
    2. Multi-GPU sharding to distribute load
    3. CPU index + GPU search (copy on demand)

**Multi-GPU Strategies:**
  - **Replication:** Same index on all GPUs (max throughput)
  - **Sharding:** Split index across GPUs (handle larger datasets)
  - FAISS supports both with `index_cpu_to_all_gpus()`

**Production Examples:**

**Meta's Feed Ranking:**
  - 100B+ user embeddings
  - 8 GPU cluster with sharding
  - Serve 50K QPS with <10ms p99 latency
  - Use IVFPQ + GPU for 100x speedup vs CPU

**Pinterest's Visual Search:**
  - 5B+ pin embeddings
  - 4 GPUs per machine
  - Batch queries in 20ms windows
  - Achieve 10K QPS per machine

**Best Practices:**
  1. Build index on CPU, move to GPU for search
  2. Batch queries to 100-1000 for optimal GPU utilization
  3. Use FP16 for 2x memory savings (GPU supports native FP16)
  4. Monitor GPU memory usage (FAISS provides `getMemoryInfo()`)
  5. Shard across multiple GPUs when index > GPU memory
  6. Consider cost: 1 GPU can replace 10-50 CPU cores for search
    """)

    print("\n" + "=" * 80)
    print("Exercise 4 Complete!")
    print("=" * 80)

    if use_gpu:
        print(f"\nKey Takeaway: GPU acceleration provides {speedup:.0f}x speedup,")
        print(f"enabling real-time search at production scale (1000+ QPS)!")
    else:
        print("\nKey Takeaway: GPU acceleration enables 10-100x speedups,")
        print("making it essential for high-throughput production serving!")


if __name__ == "__main__":
    main()
