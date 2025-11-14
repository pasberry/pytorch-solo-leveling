"""
Vector Database with FAISS
Build scalable similarity search for billion-scale embeddings
"""

import torch
import numpy as np
import faiss
import time
from typing import List, Tuple


class VectorDatabase:
    """
    Production vector database with multiple index types

    Index Types:
    - Flat: Exact search (slow but accurate)
    - IVF: Inverted file index (fast approximate search)
    - HNSW: Hierarchical navigable small world (fastest)
    - PQ: Product quantization (compressed)
    """

    def __init__(self, dimension: int, index_type: str = 'ivf', use_gpu: bool = False):
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None
        self.id_map = {}  # Map internal IDs to external IDs

    def build_index(self, embeddings: np.ndarray, nlist: int = 100):
        """
        Build FAISS index

        Args:
            embeddings: (N, D) numpy array
            nlist: Number of clusters for IVF (more = faster, less accurate)
        """
        N, D = embeddings.shape
        assert D == self.dimension

        if self.index_type == 'flat':
            # Exact search (brute force)
            self.index = faiss.IndexFlatIP(D)  # Inner product (cosine if normalized)

        elif self.index_type == 'ivf':
            # IVF index: cluster embeddings, search within clusters
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFFlat(quantizer, D, nlist)

            # Train on sample
            print(f"Training IVF index with {nlist} clusters...")
            self.index.train(embeddings)

        elif self.index_type == 'hnsw':
            # HNSW: Hierarchical graph for fast ANN
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(D, M)
            self.index.hnsw.efConstruction = 40  # Build-time quality
            self.index.hnsw.efSearch = 16  # Search-time quality

        elif self.index_type == 'pq':
            # Product Quantization: compress embeddings
            m = 8  # Number of sub-vectors
            nbits = 8  # Bits per sub-vector
            self.index = faiss.IndexPQ(D, m, nbits)

            print(f"Training PQ index...")
            self.index.train(embeddings)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add embeddings
        print(f"Adding {N:,} embeddings...")
        self.index.add(embeddings)

        # Move to GPU if requested
        if self.use_gpu and torch.cuda.is_available():
            print("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        print(f"Index built! Total vectors: {self.index.ntotal:,}")

    def search(self, queries: np.ndarray, k: int = 10, nprobe: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors

        Args:
            queries: (Q, D) query embeddings
            k: Number of neighbors to return
            nprobe: Number of clusters to search (for IVF)

        Returns:
            distances: (Q, k) similarity scores
            indices: (Q, k) indices of nearest neighbors
        """
        assert self.index is not None, "Index not built yet"

        # Set nprobe for IVF
        if self.index_type == 'ivf':
            self.index.nprobe = nprobe

        # Search
        distances, indices = self.index.search(queries, k)

        return distances, indices

    def add_with_ids(self, embeddings: np.ndarray, ids: List[str]):
        """
        Add embeddings with custom IDs

        Args:
            embeddings: (N, D) numpy array
            ids: List of N custom IDs
        """
        # Use IDMap wrapper
        if not isinstance(self.index, faiss.IndexIDMap):
            self.index = faiss.IndexIDMap(self.index)

        # Convert string IDs to integers
        int_ids = np.arange(len(self.id_map), len(self.id_map) + len(ids), dtype=np.int64)
        for idx, str_id in zip(int_ids, ids):
            self.id_map[int(idx)] = str_id

        self.index.add_with_ids(embeddings, int_ids)

    def remove(self, ids: List[int]):
        """Remove embeddings by IDs"""
        if isinstance(self.index, faiss.IndexIDMap):
            id_selector = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(np.array(ids, dtype=np.int64)))
            self.index.remove_ids(id_selector)

    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index, path)
        print(f"Index saved to {path}")

    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(path)
        if self.use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        print(f"Index loaded from {path}")


def benchmark_index_types(embeddings, queries, k=10):
    """Benchmark different index types"""
    print("\n" + "=" * 60)
    print("Benchmarking Index Types")
    print("=" * 60)

    results = []
    dimension = embeddings.shape[1]

    for index_type in ['flat', 'ivf', 'hnsw']:
        print(f"\n{index_type.upper()} Index:")

        # Build index
        db = VectorDatabase(dimension, index_type=index_type)
        start = time.time()
        db.build_index(embeddings)
        build_time = time.time() - start

        # Search
        start = time.time()
        distances, indices = db.search(queries, k=k)
        search_time = time.time() - start

        print(f"  Build time: {build_time:.2f}s")
        print(f"  Search time: {search_time*1000:.2f}ms for {len(queries)} queries")
        print(f"  Per-query: {search_time/len(queries)*1000:.2f}ms")

        results.append({
            'type': index_type,
            'build_time': build_time,
            'search_time': search_time,
            'per_query_ms': search_time / len(queries) * 1000
        })

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Vector Database with FAISS")
    print("=" * 60)

    # Configuration
    N = 1_000_000  # 1M embeddings
    D = 768  # Dimension
    Q = 100  # Number of queries

    print(f"\nDataset:")
    print(f"Embeddings: {N:,}")
    print(f"Dimension: {D}")
    print(f"Queries: {Q}")

    # Generate random embeddings (in practice, these are from your model)
    print("\nGenerating embeddings...")
    embeddings = np.random.randn(N, D).astype('float32')

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    queries = np.random.randn(Q, D).astype('float32')
    faiss.normalize_L2(queries)

    # Example 1: Flat index (exact search)
    print("\n" + "=" * 60)
    print("Example 1: Flat Index (Exact Search)")
    print("=" * 60)

    db_flat = VectorDatabase(D, index_type='flat')
    db_flat.build_index(embeddings[:100000])  # Use subset for demo

    distances, indices = db_flat.search(queries[:5], k=5)

    print(f"\nTop-5 results for first query:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  Rank {i+1}: Index={idx}, Score={dist:.4f}")

    # Example 2: IVF index (fast approximate search)
    print("\n" + "=" * 60)
    print("Example 2: IVF Index (Approximate Search)")
    print("=" * 60)

    db_ivf = VectorDatabase(D, index_type='ivf')
    db_ivf.build_index(embeddings, nlist=100)

    # Trade-off: nprobe (more = slower but more accurate)
    for nprobe in [1, 5, 10, 20]:
        start = time.time()
        distances, indices = db_ivf.search(queries, k=10, nprobe=nprobe)
        elapsed = time.time() - start

        print(f"nprobe={nprobe}: {elapsed*1000:.2f}ms ({elapsed/len(queries)*1000:.2f}ms per query)")

    # Example 3: Product Quantization (compressed)
    print("\n" + "=" * 60)
    print("Example 3: Product Quantization (Compression)")
    print("=" * 60)

    # Original size
    original_size_mb = embeddings.nbytes / 1024**2
    print(f"Original size: {original_size_mb:.2f}MB")

    # With PQ compression (8 bytes per vector)
    compressed_size_mb = (N * 8) / 1024**2
    compression_ratio = original_size_mb / compressed_size_mb

    print(f"Compressed size: {compressed_size_mb:.2f}MB")
    print(f"Compression ratio: {compression_ratio:.1f}x")

    # Benchmark
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    # Use smaller dataset for demo
    sample_embeddings = embeddings[:100000]
    sample_queries = queries[:100]

    results = benchmark_index_types(sample_embeddings, sample_queries, k=10)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Index Type':<15} {'Build Time':<15} {'Per Query':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r['type']:<15} {r['build_time']:<15.2f}s {r['per_query_ms']:<15.2f}ms")

    print("\n" + "=" * 60)
    print("Production Recommendations:")
    print("=" * 60)
    print("✓ < 1M vectors: Use HNSW (fast, accurate)")
    print("✓ 1M - 100M vectors: Use IVF with nlist=sqrt(N)")
    print("✓ > 100M vectors: Use IVF + PQ for compression")
    print("✓ Real-time updates: Use Flat or HNSW")
    print("✓ Batch updates: Use IVF (retrain periodically)")
    print("✓ GPU: 10-100x speedup for large batches")

    print("\n" + "=" * 60)
    print("Trade-offs:")
    print("=" * 60)
    print("Accuracy vs Speed:")
    print("  Flat > HNSW > IVF > PQ")
    print("Memory Usage:")
    print("  Flat > HNSW > IVF > PQ")
    print("Build Time:")
    print("  PQ > IVF > HNSW > Flat")
