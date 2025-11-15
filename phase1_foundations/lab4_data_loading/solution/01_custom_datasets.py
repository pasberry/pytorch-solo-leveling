"""
Phase 1 Lab 4: Custom Datasets
Learn to create custom Dataset classes for various data types
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Callable, List
import json
from pathlib import Path


class SyntheticDataset(Dataset):
    """
    Simple synthetic dataset for demonstration.

    Args:
        num_samples: Number of samples to generate
        input_dim: Dimension of input features
        num_classes: Number of output classes
        transform: Optional transform to apply to features
    """
    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        num_classes: int,
        transform: Optional[Callable] = None
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.transform = transform

        # Generate synthetic data
        self.features = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            (features, label) tuple
        """
        features = self.features[idx]
        label = self.labels[idx]

        # Apply transform if specified
        if self.transform:
            features = self.transform(features)

        return features, label


class CSVDataset(Dataset):
    """
    Dataset that loads data from CSV file.

    Args:
        csv_path: Path to CSV file
        feature_columns: List of column names for features
        label_column: Column name for label
        transform: Optional transform
    """
    def __init__(
        self,
        csv_path: Path,
        feature_columns: List[str],
        label_column: str,
        transform: Optional[Callable] = None
    ):
        import pandas as pd

        self.transform = transform

        # Load CSV
        df = pd.read_csv(csv_path)

        # Extract features and labels
        self.features = torch.tensor(
            df[feature_columns].values,
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            df[label_column].values,
            dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label


class SequenceDataset(Dataset):
    """
    Dataset for sequence data (e.g., time series, text).

    Args:
        sequences: List of sequences
        labels: List of labels
        max_length: Maximum sequence length (will pad/truncate)
        pad_value: Value to use for padding
    """
    def __init__(
        self,
        sequences: List[List[float]],
        labels: List[int],
        max_length: int,
        pad_value: float = 0.0
    ):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence: Padded sequence tensor
            label: Label tensor
            length: Actual sequence length (before padding)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Get actual length
        actual_length = min(len(sequence), self.max_length)

        # Truncate or pad sequence
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [self.pad_value] * (self.max_length - len(sequence))

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(actual_length, dtype=torch.long)
        )


class ImageFolderDataset(Dataset):
    """
    Custom dataset for loading images from folder structure:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg

    Args:
        root: Root directory
        transform: Optional transform for images
    """
    def __init__(self, root: Path, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform

        # Find all image files and create class mapping
        self.samples = []
        self.class_to_idx = {}

        for idx, class_dir in enumerate(sorted(self.root.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = idx

            # Find all image files in this class
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and return image and label."""
        from PIL import Image

        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


class CachedDataset(Dataset):
    """
    Wrapper that caches dataset items in memory for faster access.
    Useful for small datasets that fit in memory.

    Args:
        dataset: Underlying dataset
        cache_size: Maximum number of items to cache (None = all)
    """
    def __init__(self, dataset: Dataset, cache_size: Optional[int] = None):
        self.dataset = dataset
        self.cache_size = cache_size or len(dataset)
        self.cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Load from underlying dataset
        item = self.dataset[idx]

        # Add to cache if there's space
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item

        return item

    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning with multiple labels.

    Args:
        features: Input features
        labels_dict: Dictionary mapping task names to labels
    """
    def __init__(
        self,
        features: torch.Tensor,
        labels_dict: dict
    ):
        self.features = features
        self.labels_dict = labels_dict
        self.task_names = list(labels_dict.keys())

        # Validate all tasks have same number of samples
        assert all(
            len(labels) == len(features)
            for labels in labels_dict.values()
        ), "All tasks must have same number of samples"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            features: Input features
            labels: Dictionary mapping task names to labels
        """
        features = self.features[idx]
        labels = {
            task: self.labels_dict[task][idx]
            for task in self.task_names
        }
        return features, labels


def collate_fn_multitask(batch):
    """Custom collate function for multi-task dataset."""
    features = torch.stack([item[0] for item in batch])
    labels = {}

    # Collect labels for each task
    task_names = batch[0][1].keys()
    for task in task_names:
        task_labels = torch.stack([item[1][task] for item in batch])
        labels[task] = task_labels

    return features, labels


def collate_fn_sequences(batch):
    """Custom collate function for sequences with different lengths."""
    sequences, labels, lengths = zip(*batch)

    # Stack tensors
    sequences = torch.stack(sequences)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)

    # Sort by length (descending) for packing
    lengths, sorted_idx = lengths.sort(descending=True)
    sequences = sequences[sorted_idx]
    labels = labels[sorted_idx]

    return sequences, labels, lengths


def demo_synthetic_dataset():
    """Demonstrate synthetic dataset."""
    print("=" * 70)
    print("DEMO: Synthetic Dataset")
    print("=" * 70)

    # Create dataset
    dataset = SyntheticDataset(
        num_samples=1000,
        input_dim=20,
        num_classes=5
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dim: {dataset.input_dim}")
    print(f"Num classes: {dataset.num_classes}")

    # Get a sample
    features, label = dataset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Sample label: {label}")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through batches
    for batch_idx, (features, labels) in enumerate(loader):
        if batch_idx == 0:
            print(f"\nFirst batch:")
            print(f"  Features shape: {features.shape}")
            print(f"  Labels shape: {labels.shape}")
        break

    print("=" * 70)


def demo_sequence_dataset():
    """Demonstrate sequence dataset."""
    print("\n" + "=" * 70)
    print("DEMO: Sequence Dataset")
    print("=" * 70)

    # Create synthetic sequences with different lengths
    sequences = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0],
        [10.0]
    ]
    labels = [0, 1, 0, 1]

    dataset = SequenceDataset(
        sequences=sequences,
        labels=labels,
        max_length=5,
        pad_value=0.0
    )

    print(f"Dataset size: {len(dataset)}")

    # Show padded sequences
    for idx in range(len(dataset)):
        seq, label, length = dataset[idx]
        print(f"\nSample {idx}:")
        print(f"  Sequence: {seq}")
        print(f"  Label: {label}")
        print(f"  Actual length: {length}")

    # Use custom collate function
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn_sequences
    )

    for sequences, labels, lengths in loader:
        print(f"\nBatch:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Lengths: {lengths}")
        break

    print("=" * 70)


def demo_multitask_dataset():
    """Demonstrate multi-task dataset."""
    print("\n" + "=" * 70)
    print("DEMO: Multi-Task Dataset")
    print("=" * 70)

    # Create synthetic multi-task data
    features = torch.randn(100, 20)
    labels_dict = {
        'classification': torch.randint(0, 3, (100,)),
        'regression': torch.randn(100),
        'binary': torch.randint(0, 2, (100,))
    }

    dataset = MultiTaskDataset(features, labels_dict)

    print(f"Dataset size: {len(dataset)}")
    print(f"Tasks: {dataset.task_names}")

    # Get a sample
    features, labels = dataset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Sample labels:")
    for task, label in labels.items():
        print(f"  {task}: {label}")

    # Create DataLoader with custom collate
    loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn_multitask
    )

    for features, labels in loader:
        print(f"\nBatch:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels:")
        for task, task_labels in labels.items():
            print(f"    {task}: {task_labels.shape}")
        break

    print("=" * 70)


def demo_cached_dataset():
    """Demonstrate cached dataset."""
    print("\n" + "=" * 70)
    print("DEMO: Cached Dataset")
    print("=" * 70)

    import time

    # Create base dataset
    base_dataset = SyntheticDataset(
        num_samples=1000,
        input_dim=100,
        num_classes=10
    )

    # Wrap with cache
    cached_dataset = CachedDataset(base_dataset, cache_size=500)

    # Time first access
    start = time.time()
    for i in range(100):
        _ = cached_dataset[i]
    first_time = time.time() - start

    # Time second access (should be faster)
    start = time.time()
    for i in range(100):
        _ = cached_dataset[i]
    second_time = time.time() - start

    print(f"First access time: {first_time:.4f}s")
    print(f"Second access time (cached): {second_time:.4f}s")
    print(f"Speedup: {first_time / second_time:.2f}x")
    print(f"Cache size: {len(cached_dataset.cache)}")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_synthetic_dataset()
    demo_sequence_dataset()
    demo_multitask_dataset()
    demo_cached_dataset()

    print("\n✓ All dataset demonstrations complete!")
