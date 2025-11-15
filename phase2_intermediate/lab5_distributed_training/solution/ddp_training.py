"""
Phase 2 Lab 5: Distributed Data Parallel (DDP) Training
Learn to train models across multiple GPUs using DDP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class SimpleModel(nn.Module):
    """Simple model for DDP demonstration."""
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


def setup(rank: int, world_size: int):
    """
    Initialize distributed training process group.

    Args:
        rank: Unique identifier for this process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()


def train_ddp(rank: int, world_size: int, epochs: int = 5):
    """
    Train model using DDP on a single GPU.

    Args:
        rank: GPU ID
        world_size: Total number of GPUs
        epochs: Number of training epochs
    """
    print(f"Running DDP on rank {rank}/{world_size}")

    # Setup distributed training
    setup(rank, world_size)

    # Create model and move to GPU
    model = SimpleModel(input_dim=100, hidden_dim=128, output_dim=10).to(rank)

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Create dataset with DistributedSampler
    dataset_size = 1000
    X = torch.randn(dataset_size, 100)
    y = torch.randint(0, 10, (dataset_size,))
    dataset = TensorDataset(X, y)

    # DistributedSampler ensures each GPU gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        # Set epoch for sampler (ensures different shuffling each epoch)
        sampler.set_epoch(epoch)

        ddp_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(rank), targets.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Only print from rank 0 to avoid duplicate output
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            accuracy = 100.0 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")

    # Cleanup
    cleanup()


def save_ddp_checkpoint(rank: int, model: DDP, optimizer: optim.Optimizer, epoch: int, path: str):
    """
    Save DDP checkpoint from rank 0.

    Args:
        rank: Current process rank
        model: DDP model
        optimizer: Optimizer
        epoch: Current epoch
        path: Save path
    """
    if rank == 0:
        # Only rank 0 saves checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Note: model.module for DDP
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


def demo_ddp():
    """Demonstrate DDP training."""
    print("=" * 70)
    print("DEMO: Distributed Data Parallel (DDP) Training")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. DDP requires GPUs.")
        print("=" * 70)
        return

    world_size = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {world_size}")

    if world_size < 2:
        print("DDP demo requires at least 2 GPUs.")
        print("Showing single-GPU equivalent instead...\n")

        # Single GPU training for demonstration
        device = torch.device('cuda:0')
        model = SimpleModel(100, 128, 10).to(device)

        X = torch.randn(1000, 100)
        y = torch.randint(0, 10, (1000,))
        dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Training on single GPU...")
        for epoch in range(3):
            model.train()
            total_loss = 0.0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")
    else:
        print(f"Launching DDP training on {world_size} GPUs...")

        # Spawn processes for each GPU
        mp.spawn(
            train_ddp,
            args=(world_size, 3),
            nprocs=world_size,
            join=True
        )

    print("\n" + "=" * 70)


def demo_gradient_accumulation():
    """Demonstrate gradient accumulation for effective larger batch size."""
    print("\n" + "=" * 70)
    print("DEMO: Gradient Accumulation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleModel(100, 128, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create data
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

    accumulation_steps = 4  # Effective batch size = 16 * 4 = 64

    print(f"Batch size: 16")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {16 * accumulation_steps}")
    print()

    model.train()
    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx < 20:  # Print first few updates
                print(f"Step {(batch_idx + 1) // accumulation_steps}: Updated weights")

    print("\nGradient accumulation allows training with larger effective batch sizes")
    print("=" * 70)


def print_ddp_best_practices():
    """Print DDP best practices."""
    print("\n" + "=" * 70)
    print("DDP BEST PRACTICES")
    print("=" * 70)

    practices = [
        ("1. Use DistributedSampler", "Ensures each GPU gets different data"),
        ("2. Set sampler.set_epoch(epoch)", "Ensures different shuffling each epoch"),
        ("3. Wrap model with DDP", "Synchronizes gradients across GPUs"),
        ("4. Use model.module for checkpoints", "Access underlying model in DDP wrapper"),
        ("5. Only save from rank 0", "Avoid duplicate checkpoint saves"),
        ("6. Use nccl backend for GPU", "Fastest backend for NVIDIA GPUs"),
        ("7. Pin memory in DataLoader", "Faster host-to-device transfers"),
        ("8. Use gradient accumulation", "Train with larger effective batch sizes"),
    ]

    for title, description in practices:
        print(f"\n{title}")
        print(f"  → {description}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Note: DDP requires running with torch.distributed.launch or mp.spawn
    demo_ddp()
    demo_gradient_accumulation()
    print_ddp_best_practices()

    print("\n✓ Distributed training demonstrations complete!")
