"""
Exercise 5: Production-Scale Vector Database
Build a complete production system with sharding, monitoring, and index management

Learning objectives:
- Building production-ready vector search service
- Index sharding for billion-scale datasets
- Monitoring and observability
- Index updates and versioning
- Handling real-world constraints (latency, memory, cost)
"""

import faiss
import numpy as np
import time
from typing import List, Tuple
import pickle
import os
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SearchMetrics:
    """Metrics for monitoring search performance"""
    latency_ms: float
    recall: float
    throughput_qps: float
    memory_gb: float


class ShardedVectorDB:
    """
    Production vector database with sharding, monitoring, and best practices

    This demonstrates a simplified production system similar to:
    - Meta's embedding search (10B+ vectors)
    - Pinterest's visual search (5B+ pins)
    - Semantic search for large document corpora
    """

    def __init__(self, dimension: int, num_shards: int = 4, use_gpu: bool = False):
        """
        Initialize sharded vector database

        Args:
            dimension: Vector dimensionality
            num_shards: Number of shards to split data across
            use_gpu: Whether to use GPU acceleration
        """
        self.dimension = dimension
        self.num_shards = num_shards
        self.use_gpu = use_gpu
        self.shards = []
        self.shard_sizes = []
        self.metrics = defaultdict(list)

        print(f"Initializing ShardedVectorDB:")
        print(f"  Dimension: {dimension}")
        print(f"  Num shards: {num_shards}")
        print(f"  GPU enabled: {use_gpu}")

    def build_index(self, vectors: np.ndarray, index_type: str = "IVFPQ"):
        """
        Build sharded index from vectors

        Args:
            vectors: (N, D) array of vectors
            index_type: Type of index ("Flat", "IVF", "IVFPQ", "HNSW")
        """
        n = len(vectors)
        shard_size = (n + self.num_shards - 1) // self.num_shards

        print(f"\nBuilding {index_type} index with {self.num_shards} shards...")
        print(f"Total vectors: {n:,}")
        print(f"Vectors per shard: ~{shard_size:,}")

        build_start = time.time()

        for shard_id in range(self.num_shards):
            start_idx = shard_id * shard_size
            end_idx = min((shard_id + 1) * shard_size, n)
            shard_vectors = vectors[start_idx:end_idx]

            print(f"\n  Shard {shard_id+1}/{self.num_shards}: {len(shard_vectors):,} vectors")

            # Create index based on type
            if index_type == "Flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif index_type == "IVF":
                nlist = min(1000, len(shard_vectors) // 100)
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                print(f"    Training IVF with {nlist} clusters...")
                index.train(shard_vectors)
                index.nprobe = 10
            elif index_type == "IVFPQ":
                nlist = min(1000, len(shard_vectors) // 100)
                M = 96
                nbits = 8
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, M, nbits)
                print(f"    Training IVFPQ (nlist={nlist}, M={M})...")
                # Train on subsample if too large
                train_size = min(len(shard_vectors), 1000000)
                index.train(shard_vectors[:train_size])
                index.nprobe = 10
            elif index_type == "HNSW":
                M = 32
                index = faiss.IndexHNSWFlat(self.dimension, M)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 16
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            # Add vectors
            print(f"    Adding vectors...")
            index.add(shard_vectors)

            # Move to GPU if requested
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, shard_id % faiss.get_num_gpus(), index)
                    print(f"    ✓ Moved to GPU {shard_id % faiss.get_num_gpus()}")
                except:
                    print(f"    ⚠️  GPU not available, staying on CPU")

            self.shards.append(index)
            self.shard_sizes.append(len(shard_vectors))

        build_time = time.time() - build_start
        print(f"\n✓ Index built in {build_time:.2f}s")

        # Calculate memory usage
        memory_gb = self._estimate_memory()
        print(f"✓ Estimated memory: {memory_gb:.2f} GB")

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search across all shards and merge results

        Args:
            queries: (Q, D) array of query vectors
            k: Number of nearest neighbors to return

        Returns:
            distances: (Q, k) distances
            indices: (Q, k) global indices
        """
        n_queries = len(queries)

        # Search each shard
        all_distances = []
        all_indices = []

        search_start = time.time()

        for shard_id, index in enumerate(self.shards):
            D, I = index.search(queries, k)

            # Adjust indices to global space
            offset = sum(self.shard_sizes[:shard_id])
            I = I + offset

            all_distances.append(D)
            all_indices.append(I)

        # Merge results from all shards
        all_distances = np.concatenate(all_distances, axis=1)  # (Q, k*num_shards)
        all_indices = np.concatenate(all_indices, axis=1)

        # Select top-k from merged results
        top_k_positions = np.argsort(all_distances, axis=1)[:, :k]

        final_distances = np.zeros((n_queries, k), dtype=np.float32)
        final_indices = np.zeros((n_queries, k), dtype=np.int64)

        for i in range(n_queries):
            final_distances[i] = all_distances[i, top_k_positions[i]]
            final_indices[i] = all_indices[i, top_k_positions[i]]

        search_time = time.time() - search_start
        latency_ms = search_time / n_queries * 1000
        throughput = n_queries / search_time

        # Record metrics
        self.metrics['latency_ms'].append(latency_ms)
        self.metrics['throughput_qps'].append(throughput)

        return final_distances, final_indices

    def add_vectors(self, vectors: np.ndarray, shard_id: int = None):
        """
        Add new vectors to index (online updates)

        Args:
            vectors: Vectors to add
            shard_id: Which shard to add to (None = auto-assign)
        """
        if shard_id is None:
            # Auto-assign to smallest shard
            shard_id = np.argmin(self.shard_sizes)

        self.shards[shard_id].add(vectors)
        self.shard_sizes[shard_id] += len(vectors)

        print(f"Added {len(vectors):,} vectors to shard {shard_id}")

    def save(self, path: str):
        """Save index to disk"""
        os.makedirs(path, exist_ok=True)

        for shard_id, index in enumerate(self.shards):
            shard_path = os.path.join(path, f"shard_{shard_id}.index")
            faiss.write_index(faiss.index_gpu_to_cpu(index) if self.use_gpu else index,
                            shard_path)

        # Save metadata
        metadata = {
            'dimension': self.dimension,
            'num_shards': self.num_shards,
            'shard_sizes': self.shard_sizes,
            'use_gpu': self.use_gpu
        }
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✓ Index saved to {path}")

    def load(self, path: str):
        """Load index from disk"""
        # Load metadata
        with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        self.dimension = metadata['dimension']
        self.num_shards = metadata['num_shards']
        self.shard_sizes = metadata['shard_sizes']

        # Load shards
        self.shards = []
        for shard_id in range(self.num_shards):
            shard_path = os.path.join(path, f"shard_{shard_id}.index")
            index = faiss.read_index(shard_path)

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, shard_id % faiss.get_num_gpus(), index)
                except:
                    pass

            self.shards.append(index)

        print(f"✓ Index loaded from {path}")

    def get_metrics(self) -> dict:
        """Get performance metrics"""
        if not self.metrics['latency_ms']:
            return {}

        return {
            'avg_latency_ms': np.mean(self.metrics['latency_ms']),
            'p50_latency_ms': np.percentile(self.metrics['latency_ms'], 50),
            'p95_latency_ms': np.percentile(self.metrics['latency_ms'], 95),
            'p99_latency_ms': np.percentile(self.metrics['latency_ms'], 99),
            'avg_throughput_qps': np.mean(self.metrics['throughput_qps']),
            'memory_gb': self._estimate_memory()
        }

    def _estimate_memory(self) -> float:
        """Estimate total memory usage in GB"""
        total_vectors = sum(self.shard_sizes)
        # Rough estimate: depends on index type
        bytes_per_vector = 100  # Conservative estimate for IVFPQ
        return total_vectors * bytes_per_vector / (1024**3)


def main():
    print("=" * 80)
    print("Exercise 5: Production-Scale Vector Database")
    print("=" * 80)

    # ========================================
    # 1. Generate Production-Scale Dataset
    # ========================================
    print("\n1. Generating Production-Scale Dataset")
    print("-" * 80)

    d = 768
    n = 5000000  # 5M vectors (simulating 50M-500M in real production)
    n_queries = 1000
    k = 10

    print(f"Dataset size: {n:,} vectors")
    print(f"Dimension: {d}")
    print(f"Simulating production workload with {n_queries:,} queries")

    np.random.seed(42)

    # Generate in batches
    print("\nGenerating vectors...")
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

    print(f"✓ Generated {n:,} vectors and {n_queries:,} queries")

    # ========================================
    # 2. Build Production Index
    # ========================================
    print("\n2. Building Production Index")
    print("-" * 80)

    # Check GPU availability
    try:
        ngpus = faiss.get_num_gpus()
        use_gpu = ngpus > 0
    except:
        use_gpu = False

    # Create sharded database
    db = ShardedVectorDB(
        dimension=d,
        num_shards=4,
        use_gpu=use_gpu
    )

    # Build with IVFPQ (best for production: fast + compressed)
    db.build_index(vectors, index_type="IVFPQ")

    # ========================================
    # 3. Benchmark Performance
    # ========================================
    print("\n3. Performance Benchmarking")
    print("-" * 80)

    # Warmup
    _ = db.search(queries[:10], k)

    # Run benchmark
    print(f"\nSearching {n_queries:,} queries...")
    D, I = db.search(queries, k)

    # Get metrics
    metrics = db.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Average latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"  P50 latency: {metrics['p50_latency_ms']:.2f} ms")
    print(f"  P95 latency: {metrics['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency: {metrics['p99_latency_ms']:.2f} ms")
    print(f"  Throughput: {metrics['avg_throughput_qps']:.0f} queries/sec")
    print(f"  Memory usage: {metrics['memory_gb']:.2f} GB")

    # ========================================
    # 4. SLA Validation
    # ========================================
    print("\n4. Production SLA Validation")
    print("-" * 80)

    # Typical production SLAs
    sla_p99_latency = 50  # ms
    sla_min_throughput = 100  # QPS
    sla_max_memory = 50  # GB

    print(f"\nProduction SLAs:")
    print(f"  P99 latency: < {sla_p99_latency} ms")
    print(f"  Throughput: > {sla_min_throughput} QPS")
    print(f"  Memory: < {sla_max_memory} GB")

    print(f"\nValidation Results:")
    p99_pass = metrics['p99_latency_ms'] < sla_p99_latency
    throughput_pass = metrics['avg_throughput_qps'] > sla_min_throughput
    memory_pass = metrics['memory_gb'] < sla_max_memory

    print(f"  {'✓' if p99_pass else '✗'} P99 latency: {metrics['p99_latency_ms']:.2f} ms")
    print(f"  {'✓' if throughput_pass else '✗'} Throughput: {metrics['avg_throughput_qps']:.0f} QPS")
    print(f"  {'✓' if memory_pass else '✗'} Memory: {metrics['memory_gb']:.2f} GB")

    if p99_pass and throughput_pass and memory_pass:
        print("\n  🎉 All SLAs met! Ready for production.")
    else:
        print("\n  ⚠️  Some SLAs not met. Tuning required:")
        if not p99_pass:
            print("     - Increase nprobe for IVF")
            print("     - Use GPU acceleration")
            print("     - Increase batch size")
        if not throughput_pass:
            print("     - Enable GPU acceleration")
            print("     - Increase num_shards")
            print("     - Use more efficient index (HNSW)")
        if not memory_pass:
            print("     - Increase PQ compression (lower M)")
            print("     - Increase num_shards to distribute")

    # ========================================
    # 5. Online Updates
    # ========================================
    print("\n5. Online Index Updates")
    print("-" * 80)

    # Simulate adding new vectors
    new_vectors = np.random.random((10000, d)).astype('float32')
    faiss.normalize_L2(new_vectors)

    print(f"Adding {len(new_vectors):,} new vectors...")
    db.add_vectors(new_vectors)

    print(f"✓ Index now contains {sum(db.shard_sizes):,} vectors")

    # ========================================
    # 6. Persistence
    # ========================================
    print("\n6. Index Persistence")
    print("-" * 80)

    index_path = "/tmp/production_vector_db"
    print(f"Saving index to {index_path}...")
    db.save(index_path)

    print(f"\nLoading index from {index_path}...")
    db_loaded = ShardedVectorDB(dimension=d, num_shards=4, use_gpu=use_gpu)
    db_loaded.load(index_path)

    # Verify loaded index works
    D_loaded, I_loaded = db_loaded.search(queries[:10], k)
    print(f"✓ Loaded index verified (searched {len(queries[:10])} queries)")

    # ========================================
    # 7. Production Best Practices Summary
    # ========================================
    print("\n7. Production Best Practices")
    print("=" * 80)
    print("""
**Architecture Design:**

1. **Sharding Strategy**
   - Shard by vector ID ranges (demonstrated above)
   - Alternative: Shard by content type (users, items, documents)
   - Rule of thumb: Keep shards at 10M-100M vectors each

2. **Index Selection**
   - <10M vectors: Flat or IVF
   - 10M-100M: IVF or HNSW
   - 100M-1B: IVFPQ
   - >1B: IVFPQ + multi-machine sharding

3. **Hardware Sizing**
   - CPU-only: 1 core per 1M vectors for 100 QPS
   - GPU: 1 GPU can handle 10M-100M vectors at 1000+ QPS
   - RAM: 2-4x index size for headroom

**Operational Excellence:**

4. **Monitoring**
   - Track p50, p95, p99 latencies (not just average!)
   - Monitor memory usage and index size
   - Alert on SLA violations (p99 > 50ms)
   - Track recall on validation set

5. **Index Updates**
   - Batch updates every 5-15 minutes (not real-time)
   - Use dual indexes for zero-downtime updates:
     * Build new index in background
     * Swap indexes atomically
     * Delete old index
   - For truly real-time: Use hybrid system (fast index + batch rebuild)

6. **Scaling Patterns**
   - Vertical: Bigger machine, more RAM, GPUs
   - Horizontal: Shard across machines
   - Hybrid: GPU cluster with IVFPQ sharding
   - At Meta scale (10B+ vectors): 50-100 GPU machines

**Cost Optimization:**

7. **Infrastructure Costs**
   - CPU-only: $0.10/1M vectors/month (AWS c5.4xlarge)
   - GPU-accelerated: $1.00/1M vectors/month (AWS p3.2xlarge)
   - Compression saves 10-30x on storage and transfer costs
   - Consider spot instances for batch index rebuilds (70% savings)

8. **Performance vs Cost Tradeoffs**
   - IVFPQ: Best cost/performance ratio for 100M+ vectors
   - GPU: Use only if need >1000 QPS (10x more expensive than CPU)
   - Multi-region: Replicate index for lower latency (2-3x cost)

**Real-World Examples:**

**Meta's Embedding Search (10B+ user profiles):**
  - 100 GPU machines with IVFPQ
  - 50K QPS peak, <10ms p99 latency
  - Sharded by user ID ranges
  - Index rebuilt every 6 hours

**Pinterest's Visual Search (5B+ pins):**
  - 50 machines with IVFPQ + HNSW hybrid
  - 10K QPS, <20ms p99
  - Content-based sharding (fashion, food, decor)
  - Online updates every 10 minutes

**OpenAI's Embedding API:**
  - Multi-tenant (millions of customers)
  - Per-customer indexes (1K-10M vectors each)
  - Auto-scaling based on QPS
  - SLA: 95% of requests <200ms
    """)

    print("\n" + "=" * 80)
    print("Exercise 5 Complete!")
    print("=" * 80)
    print(f"""
🎓 **Key Takeaways:**

1. **Sharding** enables scaling beyond single-machine limits
2. **Monitoring** (especially p99 latency) is critical for production
3. **Index persistence** enables fast restarts and disaster recovery
4. **Online updates** require careful design (batch vs real-time tradeoffs)
5. **Cost optimization** matters at scale (PQ compression, GPU usage)

You now have the knowledge to build production vector search systems
serving billions of vectors at thousands of queries per second!

**Next Steps:**
  - Deploy on real hardware (CPU vs GPU benchmarks)
  - Integrate with your ML model (generate embeddings)
  - Set up monitoring and alerting
  - Scale to billions of vectors
  - Optimize for your specific SLAs and cost constraints
    """)


if __name__ == "__main__":
    main()
