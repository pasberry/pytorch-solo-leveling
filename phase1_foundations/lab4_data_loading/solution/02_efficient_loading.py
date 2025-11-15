"""
Phase 1 Lab 4: Efficient Data Loading
Learn optimization techniques for data loading: prefetching, multiprocessing, pin memory
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from typing import Optional


class SlowDataset(Dataset):
    """Dataset with simulated slow loading (e.g., disk I/O)."""
    def __init__(self, num_samples: int, input_dim: int, delay: float = 0.001):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.delay = delay

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Simulate slow I/O
        time.sleep(self.delay)

        # Generate data
        x = torch.randn(self.input_dim)
        y = torch.randint(0, 10, (1,)).item()

        return x, y


def benchmark_dataloader(
    dataset: Dataset,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
    persistent_workers: bool = False
) -> float:
    """
    Benchmark DataLoader with different settings.

    Args:
        dataset: Dataset to load
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        Time taken to iterate through full dataset
    """
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

    start_time = time.time()

    for batch in loader:
        # Simulate some processing
        pass

    return time.time() - start_time


class TransformDataset(Dataset):
    """Dataset with on-the-fly transforms."""
    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        transform: Optional[callable] = None
    ):
        # Store only raw data
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.labels[idx]

        # Apply transform on-the-fly
        if self.transform:
            x = self.transform(x)

        return x, y


class PrecomputedTransformDataset(Dataset):
    """Dataset with precomputed transforms (faster but uses more memory)."""
    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        transform: Optional[callable] = None
    ):
        # Generate and transform all data upfront
        raw_data = torch.randn(num_samples, input_dim)

        if transform:
            self.data = torch.stack([transform(x) for x in raw_data])
        else:
            self.data = raw_data

        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def normalize_transform(x: torch.Tensor) -> torch.Tensor:
    """Simple normalization transform."""
    return (x - x.mean()) / (x.std() + 1e-8)


class IterableDatasetExample(torch.utils.data.IterableDataset):
    """
    Example of IterableDataset for streaming data.
    Useful for very large datasets that don't fit in memory.
    """
    def __init__(self, start: int, end: int, input_dim: int):
        self.start = start
        self.end = end
        self.input_dim = input_dim

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process
            iter_start = self.start
            iter_end = self.end
        else:
            # Multi-process: split workload
            per_worker = int(np.ceil((self.end - self.start) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        # Generate samples on-the-fly
        for idx in range(iter_start, iter_end):
            x = torch.randn(self.input_dim)
            y = idx % 10  # Simple label
            yield x, y


def demo_num_workers():
    """Demonstrate effect of num_workers on loading speed."""
    print("=" * 70)
    print("DEMO: Effect of num_workers")
    print("=" * 70)

    dataset = SlowDataset(num_samples=200, input_dim=100, delay=0.005)

    print(f"Dataset size: {len(dataset)}")
    print(f"Testing different num_workers settings...\n")

    results = {}
    for num_workers in [0, 2, 4, 8]:
        elapsed = benchmark_dataloader(
            dataset,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False
        )
        results[num_workers] = elapsed
        print(f"num_workers={num_workers}: {elapsed:.2f}s")

    # Show speedup
    baseline = results[0]
    print(f"\nSpeedup vs num_workers=0:")
    for num_workers, elapsed in results.items():
        if num_workers > 0:
            speedup = baseline / elapsed
            print(f"  num_workers={num_workers}: {speedup:.2f}x faster")

    print("=" * 70)


def demo_pin_memory():
    """Demonstrate effect of pin_memory (relevant for CUDA)."""
    print("\n" + "=" * 70)
    print("DEMO: Effect of pin_memory")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping pin_memory demo.")
        print("=" * 70)
        return

    dataset = SlowDataset(num_samples=200, input_dim=100, delay=0.001)
    device = torch.device('cuda')

    print(f"Testing with and without pin_memory...\n")

    # Without pin_memory
    loader_no_pin = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=False
    )

    start = time.time()
    for x, y in loader_no_pin:
        x = x.to(device)
    time_no_pin = time.time() - start

    # With pin_memory
    loader_pin = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )

    start = time.time()
    for x, y in loader_pin:
        x = x.to(device)
    time_pin = time.time() - start

    print(f"Without pin_memory: {time_no_pin:.2f}s")
    print(f"With pin_memory: {time_pin:.2f}s")
    print(f"Speedup: {time_no_pin / time_pin:.2f}x")

    print("=" * 70)


def demo_transform_strategies():
    """Compare on-the-fly vs precomputed transforms."""
    print("\n" + "=" * 70)
    print("DEMO: Transform Strategies")
    print("=" * 70)

    # On-the-fly transforms
    dataset_onthefly = TransformDataset(
        num_samples=1000,
        input_dim=100,
        transform=normalize_transform
    )

    loader_onthefly = DataLoader(dataset_onthefly, batch_size=32, num_workers=4)

    start = time.time()
    for _ in loader_onthefly:
        pass
    time_onthefly = time.time() - start

    # Precomputed transforms
    dataset_precomputed = PrecomputedTransformDataset(
        num_samples=1000,
        input_dim=100,
        transform=normalize_transform
    )

    loader_precomputed = DataLoader(dataset_precomputed, batch_size=32, num_workers=4)

    start = time.time()
    for _ in loader_precomputed:
        pass
    time_precomputed = time.time() - start

    print(f"On-the-fly transforms: {time_onthefly:.2f}s")
    print(f"Precomputed transforms: {time_precomputed:.2f}s")
    print(f"Speedup: {time_onthefly / time_precomputed:.2f}x")

    print("\nTrade-off:")
    print("  On-the-fly: Less memory, more CPU during training")
    print("  Precomputed: More memory, less CPU during training")

    print("=" * 70)


def demo_iterable_dataset():
    """Demonstrate IterableDataset for streaming."""
    print("\n" + "=" * 70)
    print("DEMO: IterableDataset")
    print("=" * 70)

    dataset = IterableDatasetExample(start=0, end=100, input_dim=20)

    # Single worker
    loader_single = DataLoader(dataset, batch_size=10, num_workers=0)
    print("Single worker:")
    batch_count = 0
    for x, y in loader_single:
        batch_count += 1
    print(f"  Batches: {batch_count}")

    # Multiple workers
    loader_multi = DataLoader(dataset, batch_size=10, num_workers=4)
    print("\nMultiple workers (4):")
    batch_count = 0
    for x, y in loader_multi:
        batch_count += 1
    print(f"  Batches: {batch_count}")

    print("\nNote: With multiple workers, each worker processes a subset of data")

    print("=" * 70)


def demo_persistent_workers():
    """Demonstrate persistent_workers for faster epoch transitions."""
    print("\n" + "=" * 70)
    print("DEMO: Persistent Workers")
    print("=" * 70)

    dataset = SlowDataset(num_samples=100, input_dim=50, delay=0.001)

    # Without persistent workers
    loader_no_persist = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=False
    )

    start = time.time()
    for epoch in range(3):
        for _ in loader_no_persist:
            pass
    time_no_persist = time.time() - start

    # With persistent workers
    loader_persist = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=True
    )

    start = time.time()
    for epoch in range(3):
        for _ in loader_persist:
            pass
    time_persist = time.time() - start

    print(f"Without persistent workers: {time_no_persist:.2f}s")
    print(f"With persistent workers: {time_persist:.2f}s")
    print(f"Speedup: {time_no_persist / time_persist:.2f}x")

    print("\nNote: persistent_workers keeps workers alive between epochs")
    print("  Pros: Faster epoch transitions")
    print("  Cons: More memory usage")

    print("=" * 70)


def print_best_practices():
    """Print data loading best practices."""
    print("\n" + "=" * 70)
    print("DATA LOADING BEST PRACTICES")
    print("=" * 70)

    practices = [
        ("1. Use num_workers > 0", "Parallel data loading speeds up training significantly"),
        ("2. Enable pin_memory for GPU", "Faster host-to-device transfers"),
        ("3. Use persistent_workers", "Faster multi-epoch training"),
        ("4. Appropriate batch size", "Balance memory and throughput"),
        ("5. Prefetch batches", "Overlap data loading with computation"),
        ("6. Cache small datasets", "Avoid repeated I/O for data that fits in memory"),
        ("7. Use IterableDataset for large data", "Stream data that doesn't fit in memory"),
        ("8. Profile your pipeline", "Identify bottlenecks with torch.utils.bottleneck"),
    ]

    for title, description in practices:
        print(f"\n{title}")
        print(f"  → {description}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_num_workers()
    demo_pin_memory()
    demo_transform_strategies()
    demo_iterable_dataset()
    demo_persistent_workers()
    print_best_practices()

    print("\n✓ All efficient loading demonstrations complete!")
