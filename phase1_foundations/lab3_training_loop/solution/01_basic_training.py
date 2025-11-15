"""
Phase 1 Lab 3: Basic Training Loop
Learn to build a complete training loop with validation and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import time
from pathlib import Path


class SimpleClassifier(nn.Module):
    """Simple MLP for demonstration."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def create_synthetic_dataset(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic dataset for training and validation.

    Args:
        num_samples: Total number of samples
        input_dim: Input feature dimension
        num_classes: Number of output classes
        train_split: Fraction of data for training

    Returns:
        train_loader, val_loader
    """
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    # Split into train/val
    num_train = int(num_samples * train_split)
    X_train, X_val = X[:num_train], X[num_train:]
    y_train, y_val = y[:num_train], y[num_train:]

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The neural network
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: The neural network
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate average metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer = None
) -> int:
    """
    Load model checkpoint.

    Returns:
        Starting epoch
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}")
    print(f"Metrics: {metrics}")

    return epoch + 1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: Path = Path("checkpoints"),
    save_every: int = 5
):
    """
    Complete training loop with validation and checkpointing.

    Args:
        model: The neural network
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Track best validation accuracy
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("Starting training...")
    print("=" * 70)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Calculate epoch time
        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                checkpoint_dir / "best_model.pt"
            )
            print(f"  ✓ New best model! (Val Acc: {best_val_acc:.2f}%)")

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            )

        print()

    print("=" * 70)
    print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")

    return history


def demo_basic_training():
    """Demonstrate basic training loop."""
    print("=" * 70)
    print("DEMO: Basic Training Loop")
    print("=" * 70)
    print()

    # Set random seed
    torch.manual_seed(42)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Create dataset
    print("Creating synthetic dataset...")
    train_loader, val_loader = create_synthetic_dataset(
        num_samples=1000,
        input_dim=20,
        num_classes=5
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print()

    # Create model
    model = SimpleClassifier(input_dim=20, hidden_dim=64, output_dim=5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        save_every=5
    )

    return history


def demo_resume_training():
    """Demonstrate resuming from checkpoint."""
    print("\n" + "=" * 70)
    print("DEMO: Resume Training from Checkpoint")
    print("=" * 70)
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path("checkpoints/best_model.pt")

    if not checkpoint_path.exists():
        print("No checkpoint found. Run demo_basic_training() first.")
        return

    # Create dataset
    train_loader, val_loader = create_synthetic_dataset(
        num_samples=1000,
        input_dim=20,
        num_classes=5
    )

    # Create model and optimizer
    model = SimpleClassifier(input_dim=20, hidden_dim=64, output_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load checkpoint
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # Continue training
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    print("\nContinuing training for 5 more epochs...")
    for epoch in range(start_epoch, start_epoch + 5):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}]")
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}% | Val Acc: {val_metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    # Run demonstrations
    history = demo_basic_training()
    demo_resume_training()

    print("\n✓ Training demonstrations complete!")
