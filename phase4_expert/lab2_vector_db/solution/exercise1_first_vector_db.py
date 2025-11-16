"""
Exercise 1: Build Your First Vector Database
Create FAISS indexes and compare exact vs approximate search

Learning objectives:
- Creating FAISS indexes
- Adding and searching vectors
- Comparing exact vs approximate search
- Measuring recall
"""

import faiss
import numpy as np
import time


def main():
    print("=" * 70)
    print("Exercise 1: Build Your First Vector Database")
    print("=" * 70)

    # Generate 1M random embeddings (768D - BERT-like)
    print("\n1. Generating 1M embeddings (768D)...")
    d = 768  # Dimension
    n = 1000000  # 1 million vectors
    np.random.seed(42)
    vectors = np.random.random((n, d)).astype('float32')

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)

    print(f"Generated {n:,} vectors of dimension {d}")
    print(f"Memory: {vectors.nbytes / 1e9:.2f} GB")

    # 1. Build exact search index
    print("\n2. Building exact search index (IndexFlatL2)...")
    index_exact = faiss.IndexFlatL2(d)
    index_exact.add(vectors)
    print(f"Exact index built: {index_exact.ntotal:,} vectors")

    # 2. Build approximate search index (IVF)
    print("\n3. Building approximate search index (IndexIVFFlat)...")
    nlist = 1000  # Number of clusters
    quantizer = faiss.IndexFlatL2(d)
    index_approx = faiss.IndexIVFFlat(quantizer, d, nlist)

    # Train the index (k-means clustering)
    print(f"Training IVF with {nlist} clusters...")
    start = time.time()
    index_approx.train(vectors)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")

    # Add vectors
    print("Adding vectors to IVF index...")
    index_approx.add(vectors)
    print(f"IVF index built: {index_approx.ntotal:,} vectors")

    # Set search parameters
    index_approx.nprobe = 10  # Search 10 nearest clusters

    # 3. Compare performance
    print("\n4. Performance Comparison")
    print("-" * 70)

    # Generate 100 random query vectors
    n_queries = 100
    queries = np.random.random((n_queries, d)).astype('float32')
    faiss.normalize_L2(queries)
    k = 10  # Top-10 nearest neighbors

    # Exact search
    print("\nExact search (brute force):")
    start = time.time()
    D_exact, I_exact = index_exact.search(queries, k)
    time_exact = (time.time() - start) / n_queries
    print(f"  Average latency: {time_exact * 1000:.2f} ms per query")
    print(f"  Throughput: {1 / time_exact:.0f} QPS")

    # Approximate search
    print("\nApproximate search (IVF with nprobe=10):")
    start = time.time()
    D_approx, I_approx = index_approx.search(queries, k)
    time_approx = (time.time() - start) / n_queries
    print(f"  Average latency: {time_approx * 1000:.2f} ms per query")
    print(f"  Throughput: {1 / time_approx:.0f} QPS")
    print(f"  Speedup: {time_exact / time_approx:.1f}x")

    # 4. Calculate recall
    print("\n5. Recall Analysis")
    print("-" * 70)

    def calculate_recall(exact_indices, approx_indices):
        """Calculate recall@k"""
        recalls = []
        for i in range(len(exact_indices)):
            exact_set = set(exact_indices[i])
            approx_set = set(approx_indices[i])
            recall = len(exact_set & approx_set) / k
            recalls.append(recall)
        return np.mean(recalls)

    recall = calculate_recall(I_exact, I_approx)
    print(f"Recall@{k}: {recall * 100:.1f}%")
    print(f"Average neighbors found: {recall * k:.1f} out of {k}")

    # 5. Try different nprobe values
    print("\n6. Recall vs Speed Tradeoff (varying nprobe)")
    print("-" * 70)
    print(f"{'nprobe':<10} {'Latency (ms)':<15} {'Recall (%)':<15} {'Speedup':<10}")
    print("-" * 70)

    for nprobe in [1, 5, 10, 20, 50, 100]:
        index_approx.nprobe = nprobe

        # Measure latency
        start = time.time()
        D_approx, I_approx = index_approx.search(queries, k)
        latency = (time.time() - start) / n_queries * 1000

        # Calculate recall
        recall = calculate_recall(I_exact, I_approx)
        speedup = time_exact / (latency / 1000)

        print(f"{nprobe:<10} {latency:<15.2f} {recall * 100:<15.1f} {speedup:<10.1f}x")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Exact search (Flat): 100% accurate but slow (O(N))")
    print("2. Approximate search (IVF): 10-20x faster with 85-95% recall")
    print("3. nprobe controls recall-speed tradeoff:")
    print("   - Low nprobe (1-5): Fast but lower recall (~60-80%)")
    print("   - Medium nprobe (10-20): Balanced (~85-92%)")
    print("   - High nprobe (50-100): Slower but high recall (~95-98%)")
    print("4. For production: Choose nprobe where recall meets requirements")
    print("=" * 70)


if __name__ == "__main__":
    main()
