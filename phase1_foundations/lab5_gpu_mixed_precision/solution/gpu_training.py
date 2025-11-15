"""
Phase 1 Lab 5: GPU Training and Mixed Precision
Learn to train models on GPU with automatic mixed precision (AMP)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import time
from typing import Dict, Tuple


class SimpleModel(nn.Module):
    """Simple model for benchmarking."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def create_dataloader(num_samples: int, input_dim: int, num_classes: int, batch_size: int = 32):
    """Create synthetic dataloader."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def train_epoch_cpu(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> Tuple[float, float]:
    """Train one epoch on CPU."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def train_epoch_gpu(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train one epoch on GPU."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        # Move to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def train_epoch_amp(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler
) -> Tuple[float, float]:
    """Train one epoch with Automatic Mixed Precision."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def demo_cpu_vs_gpu():
    """Compare CPU vs GPU training speed."""
    print("=" * 70)
    print("DEMO: CPU vs GPU Training")
    print("=" * 70)

    # Setup
    input_dim, hidden_dim, output_dim = 512, 1024, 10
    num_samples, batch_size = 10000, 128
    num_epochs = 5

    train_loader = create_dataloader(num_samples, input_dim, output_dim, batch_size)

    # CPU Training
    print("\nTraining on CPU...")
    model_cpu = SimpleModel(input_dim, hidden_dim, output_dim)
    optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(num_epochs):
        loss, acc = train_epoch_cpu(model_cpu, train_loader, criterion, optimizer_cpu)
    cpu_time = time.time() - start

    print(f"CPU Time: {cpu_time:.2f}s ({cpu_time/num_epochs:.2f}s/epoch)")

    # GPU Training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nTraining on GPU ({torch.cuda.get_device_name(0)})...")

        model_gpu = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
        optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=0.001)

        # Warmup
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_gpu(inputs)
            break

        start = time.time()
        for epoch in range(num_epochs):
            loss, acc = train_epoch_gpu(model_gpu, train_loader, criterion, optimizer_gpu, device)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start

        print(f"GPU Time: {gpu_time:.2f}s ({gpu_time/num_epochs:.2f}s/epoch)")
        print(f"\nSpeedup: {cpu_time / gpu_time:.2f}x faster on GPU")
    else:
        print("\nCUDA not available. Skipping GPU comparison.")

    print("=" * 70)


def demo_mixed_precision():
    """Demonstrate Automatic Mixed Precision training."""
    print("\n" + "=" * 70)
    print("DEMO: Automatic Mixed Precision (AMP)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping AMP demo.")
        print("=" * 70)
        return

    device = torch.device('cuda')
    input_dim, hidden_dim, output_dim = 512, 2048, 10
    num_samples, batch_size = 10000, 128
    num_epochs = 5

    train_loader = create_dataloader(num_samples, input_dim, output_dim, batch_size)
    criterion = nn.CrossEntropyLoss()

    # FP32 Training
    print("\nTraining with FP32 (full precision)...")
    model_fp32 = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(num_epochs):
        loss, acc = train_epoch_gpu(model_fp32, train_loader, criterion, optimizer_fp32, device)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

    print(f"FP32 Time: {fp32_time:.2f}s")
    print(f"FP32 Peak Memory: {fp32_memory:.2f} GB")

    # AMP Training
    print("\nTraining with AMP (mixed precision)...")
    model_amp = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
    scaler = GradScaler()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(num_epochs):
        loss, acc = train_epoch_amp(model_amp, train_loader, criterion, optimizer_amp, device, scaler)
    torch.cuda.synchronize()
    amp_time = time.time() - start
    amp_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

    print(f"AMP Time: {amp_time:.2f}s")
    print(f"AMP Peak Memory: {amp_memory:.2f} GB")

    print(f"\nSpeedup: {fp32_time / amp_time:.2f}x faster with AMP")
    print(f"Memory savings: {(1 - amp_memory / fp32_memory) * 100:.1f}%")

    print("=" * 70)


def demo_gpu_best_practices():
    """Demonstrate GPU best practices."""
    print("\n" + "=" * 70)
    print("GPU TRAINING BEST PRACTICES")
    print("=" * 70)

    practices = [
        ("1. Use pin_memory=True in DataLoader",
         "Speeds up host-to-device transfers"),

        ("2. Move model to GPU before creating optimizer",
         "Ensures optimizer states are on GPU"),

        ("3. Use torch.cuda.amp for mixed precision",
         "2-3x speedup with minimal accuracy loss"),

        ("4. Use larger batch sizes on GPU",
         "GPUs have more memory and parallelism"),

        ("5. Call torch.cuda.synchronize() for accurate timing",
         "GPU ops are asynchronous by default"),

        ("6. Use non_blocking=True for transfers when possible",
         "Overlap data transfer with computation"),

        ("7. Profile with torch.profiler",
         "Identify bottlenecks in your training"),

        ("8. Empty cache periodically with torch.cuda.empty_cache()",
         "Free up unused cached memory"),
    ]

    for title, description in practices:
        print(f"\n{title}")
        print(f"  → {description}")

    print("\n" + "=" * 70)


def demo_device_management():
    """Demonstrate device management utilities."""
    print("\n" + "=" * 70)
    print("DEMO: Device Management")
    print("=" * 70)

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {total_memory:.2f} GB")

        # Allocate tensor
        x = torch.randn(1000, 1000, device='cuda')
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2

        print(f"\nAfter allocating 1000x1000 tensor:")
        print(f"  Allocated memory: {allocated:.2f} MB")
        print(f"  Cached memory: {cached:.2f} MB")

        # Clear
        del x
        torch.cuda.empty_cache()
        print(f"\nAfter deleting and clearing cache:")
        print(f"  Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("\nCUDA not available on this system.")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_device_management()
    demo_cpu_vs_gpu()
    demo_mixed_precision()
    demo_gpu_best_practices()

    print("\n✓ All GPU training demonstrations complete!")
